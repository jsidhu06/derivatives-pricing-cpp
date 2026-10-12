#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/exceptions.hpp"
#include "dp/implied_volatility.hpp"
#include "dp/market_data.hpp"
#include "dp/params.hpp"
#include "dp/simulation.hpp"
#include "dp/underlying.hpp"
#include "dp/valuation.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <optional>
#include <span>
#include <utility>

using namespace dp;
using dp::test::close_to;
using dp::test::kMaturity;
using dp::test::kPricingDate;
using dp::test::message_contains;

namespace {

MarketData md(double rate = 0.05) {
    return {kPricingDate, DiscountCurve::flat(rate), "USD"};
}

VanillaSpec vanilla(OptionType type,
                    ExerciseType exercise = ExerciseType::European,
                    double strike = 100.0,
                    DateTime maturity = kMaturity) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = exercise,
                        .strike = strike,
                        .maturity = maturity,
                        .currency = "USD"});
}

OptionValuation bsm(OptionType type, double vol) {
    return {UnderlyingData(100.0, vol, md()), vanilla(type), PricingMethod::Analytical};
}

}  // namespace

// ---------------------------------------------------------------------------
// Golden values from the Python reference implementation
// ---------------------------------------------------------------------------

namespace {

// Field order mirrors the generator's column order; padding is irrelevant
// for a test fixture.
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct ImpliedVolGoldenRow {
    const char* label;
    const char* engine;
    const char* solver;
    double spot;
    double strike;
    double rate;
    double div_yield;
    double div_amount;
    int div_day;
    double true_vol;
    int maturity_days;
    bool is_call;
    bool is_american;
    double target_price;
    double implied_vol;
};

#include "golden/implied_vol_reference.inc"

/// The generator's engine settings (see the .inc header).
constexpr int kTreeSteps = 200;
constexpr int kPdeSpotSteps = 80;
constexpr int kPdeTimeSteps = 100;

/// The row's valuation at `vol`.
OptionValuation replay(const ImpliedVolGoldenRow& row, double vol) {
    const MarketData market(kPricingDate, DiscountCurve::flat(row.rate), "USD");
    std::optional<DiscountCurve> q_curve;
    if (row.div_yield != 0.0) {
        q_curve = DiscountCurve::flat(row.div_yield);
    }
    DividendSchedule dividends;
    if (row.div_amount != 0.0) {
        dividends.push_back({kPricingDate + std::chrono::days{row.div_day}, row.div_amount});
    }
    const UnderlyingData u(row.spot, vol, market, std::move(dividends), std::move(q_curve));
    const VanillaSpec s(
        {.option_type = row.is_call ? OptionType::Call : OptionType::Put,
         .exercise_type = row.is_american ? ExerciseType::American : ExerciseType::European,
         .strike = row.strike,
         .maturity = kPricingDate + std::chrono::days{row.maturity_days}});
    if (std::strcmp(row.engine, "TREE") == 0) {
        return {u, s, PricingMethod::Binomial, BinomialParams{.num_steps = kTreeSteps}};
    }
    if (std::strcmp(row.engine, "PDE") == 0) {
        return {u,
                s,
                PricingMethod::PdeFd,
                PdeParams{.spot_steps = kPdeSpotSteps, .time_steps = kPdeTimeSteps}};
    }
    return {u, s, PricingMethod::Analytical};
}

ImpliedVolMethod solver_of(const char* code) {
    if (std::strcmp(code, "BI") == 0) {
        return ImpliedVolMethod::Bisection;
    }
    // The Python reference's bracketing solver is scipy's brentq; ours is
    // TOMS 748. Both stop within tol of the root, which is all the golden
    // rows compare.
    if (std::strcmp(code, "BR") == 0) {
        return ImpliedVolMethod::Toms748;
    }
    return ImpliedVolMethod::NewtonRaphson;
}

}  // namespace

TEST_CASE("implied vol matches the Python reference implementation", "[implied_vol][golden]") {
    for (const auto& row : std::span(kImpliedVolGolden)) {
        INFO("row " << row.label);
        const auto result = implied_volatility(
            row.target_price, replay(row, 0.2), {.method = solver_of(row.solver), .max_iter = 200});
        REQUIRE(result.converged);
        // `tol` (1e-8) is a vol tolerance in both libraries: each stops within
        // tol of the same root, so the vols differ by at most 2 * tol (measured
        // 3.8e-9; Newton, which takes the same steps in both, 1e-14).
        CHECK_THAT(result.implied_vol, close_to(row.implied_vol, 0.0, 2e-8));
        // Repricing error is about vega * (distance to the root): tiny for
        // Newton, which lands on the root, and up to vega * tol for the
        // bracketing solvers, which return the midpoint of a bracket narrower
        // than tol (measured 3.2e-11 and 1.4e-7 on these rows).
        const bool newton = solver_of(row.solver) == ImpliedVolMethod::NewtonRaphson;
        CHECK_THAT(replay(row, result.implied_vol).present_value(),
                   close_to(row.target_price, 0.0, newton ? 2e-8 : 5e-7));
    }
}

// ---------------------------------------------------------------------------
// Round trips (ported from the Python suite)
// ---------------------------------------------------------------------------

TEST_CASE("BSM implied vol recovers a 300% vol", "[implied_vol]") {
    const auto type = GENERATE(OptionType::Call, OptionType::Put);
    const auto method = GENERATE(ImpliedVolMethod::NewtonRaphson, ImpliedVolMethod::Toms748);
    const double target = bsm(type, 3.0).present_value();

    const auto result =
        implied_volatility(target, bsm(type, 0.2), {.method = method, .max_iter = 200});
    REQUIRE(result.converged);
    CHECK_THAT(result.implied_vol, close_to(3.0, 0.0, 1e-7));
}

TEST_CASE("implied vol flags vols the price cannot determine", "[implied_vol]") {
    // At a 0.1-0.2% vol an at-the-spot option sits at its forward intrinsic
    // value: vega is negligible, so a wide range of vols reprice the target to
    // 1e-7 (the old price-tolerance solvers returned e.g. 0.0067 for 0.001 and
    // called it converged). The root finder converges and the result
    // reprices, but it is flagged as not well determined.
    const auto type = GENERATE(OptionType::Call, OptionType::Put);
    const double true_vol = GENERATE(1.0e-3, 2.0e-3);
    const auto method = GENERATE(
        ImpliedVolMethod::NewtonRaphson, ImpliedVolMethod::Bisection, ImpliedVolMethod::Toms748);
    const double target = bsm(type, true_vol).present_value();

    const dp::test::WarningCapture capture;
    const auto result =
        implied_volatility(target, bsm(type, 0.2), {.method = method, .max_iter = 200});
    CHECK(result.converged);
    CHECK_FALSE(result.well_determined);
    CHECK(capture.any_contains("not determined by the price"));
    CHECK_THAT(bsm(type, result.implied_vol).present_value(), close_to(target, 0.0, 1e-7));
}

TEST_CASE("a tiny vol at the forward is well determined", "[implied_vol]") {
    // With the strike at the forward, vega stays ~0.4 S sqrt(T) even at 0.2%
    // vol, so the price pins the vol down and it must be recovered to tol.
    const auto method = GENERATE(
        ImpliedVolMethod::NewtonRaphson, ImpliedVolMethod::Bisection, ImpliedVolMethod::Toms748);
    const OptionValuation probe = bsm(OptionType::Call, 0.2);
    const double forward = 100.0 / probe.discount_curve().df(probe.maturity_year_fraction());
    const auto at_vol = [&](double vol) {
        return OptionValuation(UnderlyingData(100.0, vol, md()),
                               vanilla(OptionType::Call, ExerciseType::European, forward),
                               PricingMethod::Analytical);
    };
    const auto result = implied_volatility(
        at_vol(2.0e-3).present_value(), at_vol(0.2), {.method = method, .tol = 1e-10});
    REQUIRE(result.converged);
    CHECK_THAT(result.implied_vol, close_to(2.0e-3, 0.0, 1e-9));
}

TEST_CASE("far out of the money, every method agrees to tol", "[implied_vol]") {
    // K = 130, 1 month: vega is small but resolvable. The old price
    // tolerance let Newton and bisection stop ~5e-6 vol away.
    const DateTime maturity = kPricingDate + std::chrono::days{30};
    const auto at_vol = [&](double vol) {
        return OptionValuation(UnderlyingData(100.0, vol, md()),
                               vanilla(OptionType::Call, ExerciseType::European, 130.0, maturity),
                               PricingMethod::Analytical);
    };
    const double target = at_vol(0.2).present_value();
    const auto method = GENERATE(
        ImpliedVolMethod::NewtonRaphson, ImpliedVolMethod::Bisection, ImpliedVolMethod::Toms748);
    const auto result = implied_volatility(target, at_vol(0.5), {.method = method});
    REQUIRE(result.converged);
    CHECK_THAT(result.implied_vol, close_to(0.2, 0.0, 1e-8));
}

TEST_CASE("bracketing solvers converge near the upper vol bound", "[implied_vol]") {
    const auto method = GENERATE(ImpliedVolMethod::Bisection, ImpliedVolMethod::Toms748);
    const double target = bsm(OptionType::Put, 4.9).present_value();
    const auto result =
        implied_volatility(target, bsm(OptionType::Put, 0.2), {.method = method, .max_iter = 300});
    REQUIRE(result.converged);
    CHECK_THAT(result.implied_vol, close_to(4.9, 0.0, 1e-7));
}

TEST_CASE("implied vol rejects a negative min_vega", "[implied_vol]") {
    CHECK_THROWS_MATCHES((ImpliedVolParams{.min_vega = -1.0}.validate()),
                         ValidationError,
                         message_contains("min_vega must be >= 0"));
}

TEST_CASE("American binomial implied vol round-trips", "[implied_vol]") {
    const auto type = GENERATE(OptionType::Call, OptionType::Put);
    const auto method = GENERATE(
        ImpliedVolMethod::NewtonRaphson, ImpliedVolMethod::Bisection, ImpliedVolMethod::Toms748);
    std::optional<DiscountCurve> q_curve;
    if (type == OptionType::Call) {
        q_curve = DiscountCurve::flat(0.02);  // so early exercise matters for calls
    }
    const auto at_vol = [&](double vol) {
        return OptionValuation(UnderlyingData(100.0, vol, md(), {}, q_curve),
                               vanilla(type, ExerciseType::American),
                               PricingMethod::Binomial,
                               BinomialParams{.num_steps = 200});
    };
    const double target = at_vol(0.3).present_value();
    const auto result = implied_volatility(target, at_vol(0.12), {.method = method, .tol = 1e-6});
    REQUIRE(result.converged);
    CHECK_THAT(result.implied_vol, close_to(0.3, 0.0, 5e-3));
}

TEST_CASE("Newton-Raphson uses fewer pricer calls than bisection", "[implied_vol]") {
    const double target = bsm(OptionType::Call, 0.35).present_value();
    const auto newton = implied_volatility(target, bsm(OptionType::Call, 0.2));
    const auto bisection = implied_volatility(
        target, bsm(OptionType::Call, 0.2), {.method = ImpliedVolMethod::Bisection});
    REQUIRE(newton.converged);
    REQUIRE(bisection.converged);
    CHECK(newton.iterations < bisection.iterations);
}

TEST_CASE("an unconverged solve is never well determined", "[implied_vol]") {
    // Three TOMS 748 iterations cannot reach a 1e-14 vol tolerance. The vol
    // at the steep ATM point would pass the vega check, but an unconverged
    // estimate is not a usable answer.
    const double target = bsm(OptionType::Call, 0.35).present_value();
    const auto result =
        implied_volatility(target,
                           bsm(OptionType::Call, 0.2),
                           {.method = ImpliedVolMethod::Toms748, .tol = 1e-14, .max_iter = 3});
    CHECK_FALSE(result.converged);
    CHECK_FALSE(result.well_determined);
}

TEST_CASE("Newton falls back to bisection when it cannot be certified", "[implied_vol]") {
    // On a 16-step tree a deep in-the-money American call is close to
    // piecewise linear in vol, so the numerical vega is a poor slope and
    // Newton's result fails the vol-space check. The solver must then
    // return exactly what plain bisection returns, having spent extra
    // pricings on the failed Newton attempt.
    const MarketData market(kPricingDate, DiscountCurve::flat(0.05), "USD");
    const VanillaSpec call = vanilla(OptionType::Call, ExerciseType::American, 40.0);
    const BinomialParams tree{.num_steps = 16};
    const double target =
        OptionValuation(UnderlyingData(100.0, 0.4, market), call, PricingMethod::Binomial, tree)
            .present_value();
    const OptionValuation start(
        UnderlyingData(100.0, 0.02, market), call, PricingMethod::Binomial, tree);

    const auto newton =
        implied_volatility(target, start, {.method = ImpliedVolMethod::NewtonRaphson});
    const auto bisection =
        implied_volatility(target, start, {.method = ImpliedVolMethod::Bisection});
    REQUIRE(newton.converged);
    CHECK_THAT(newton.implied_vol, close_to(0.4, 0.0, 1e-6));
    CHECK(newton.implied_vol == bisection.implied_vol);
    CHECK(newton.iterations == bisection.iterations);
    CHECK(newton.function_evaluations > bisection.function_evaluations);
}

// ---------------------------------------------------------------------------
// Validation and unsupported inputs
// ---------------------------------------------------------------------------

TEST_CASE("implied vol rejects prices outside the no-arbitrage bounds", "[implied_vol]") {
    SECTION("European call above spot") {
        CHECK_THROWS_MATCHES(implied_volatility(101.0, bsm(OptionType::Call, 0.2)),
                             ValidationError,
                             message_contains("outside no-arbitrage bounds"));
    }
    SECTION("European put above the discounted strike") {
        const double upper = 100.0 * DiscountCurve::flat(0.05).df(year_fraction(
                                         kPricingDate, kMaturity, DayCountConvention::Act365F));
        CHECK_THROWS_MATCHES(implied_volatility(upper + 1.0, bsm(OptionType::Put, 0.2)),
                             ValidationError,
                             message_contains("outside no-arbitrage bounds"));
    }
    SECTION("American put below intrinsic") {
        const OptionValuation ov(UnderlyingData(80.0, 0.2, md()),
                                 vanilla(OptionType::Put, ExerciseType::American),
                                 PricingMethod::Binomial);
        CHECK_THROWS_MATCHES(implied_volatility(19.0, ov),
                             ValidationError,
                             message_contains("outside no-arbitrage bounds"));
    }
    SECTION("negative and non-finite targets") {
        // Zero sits below an ATM call's no-arbitrage floor (S - K df > 0).
        CHECK_THROWS_MATCHES(implied_volatility(0.0, bsm(OptionType::Call, 0.2)),
                             ValidationError,
                             message_contains("outside no-arbitrage bounds"));
        CHECK_THROWS_MATCHES(implied_volatility(-1.0, bsm(OptionType::Call, 0.2)),
                             ValidationError,
                             message_contains("non-negative"));
        CHECK_THROWS_MATCHES(implied_volatility(dp::test::kNaN, bsm(OptionType::Call, 0.2)),
                             ValidationError,
                             message_contains("finite"));
    }
}

TEST_CASE("implied vol rejects unsupported contracts and engines", "[implied_vol]") {
    SECTION("Monte Carlo") {
        const GbmProcess gbm(md(),
                             {.initial_value = 100.0, .volatility = 0.2},
                             {.paths = 1000, .end_date = kMaturity, .num_steps = 12});
        const OptionValuation ov(gbm, vanilla(OptionType::Call), PricingMethod::MonteCarlo);
        CHECK_THROWS_MATCHES(implied_volatility(10.0, ov),
                             UnsupportedFeatureError,
                             message_contains("ANALYTICAL, BINOMIAL, or PDE_FD"));
    }
    SECTION("barrier spec") {
        const OptionValuation ov(UnderlyingData(100.0, 0.2, md()),
                                 BarrierSpec({.option_type = OptionType::Call,
                                              .exercise_type = ExerciseType::European,
                                              .strike = 100.0,
                                              .maturity = kMaturity,
                                              .barrier = 130.0,
                                              .direction = BarrierDirection::Up,
                                              .action = BarrierAction::Out,
                                              .currency = "USD"}),
                                 PricingMethod::Analytical);
        CHECK_THROWS_MATCHES(implied_volatility(5.0, ov),
                             UnsupportedFeatureError,
                             message_contains("barrier specs"));
    }
    SECTION("custom payoff") {
        const OptionValuation ov(
            UnderlyingData(100.0, 0.2, md()),
            PayoffSpec({.exercise_type = ExerciseType::European,
                        .maturity = kMaturity,
                        .payoff_fn = [](double s) { return s > 100.0 ? 1.0 : 0.0; },
                        .currency = "USD"}),
            PricingMethod::Binomial);
        CHECK_THROWS_MATCHES(
            implied_volatility(0.5, ov), UnsupportedFeatureError, message_contains("PayoffSpec"));
    }
}

TEST_CASE("implied vol validates its params", "[implied_vol]") {
    const auto ov = bsm(OptionType::Call, 0.2);
    const double target = ov.present_value();
    CHECK_THROWS_MATCHES(implied_volatility(target, ov, {.vol_lower = 0.5, .vol_upper = 0.4}),
                         ValidationError,
                         message_contains("vol_lower < vol_upper"));
    CHECK_THROWS_MATCHES(implied_volatility(target, ov, {.tol = 0.0}),
                         ValidationError,
                         message_contains("tol must be positive"));
    CHECK_THROWS_MATCHES(implied_volatility(target, ov, {.max_iter = 0}),
                         ValidationError,
                         message_contains("max_iter"));
    CHECK_THROWS_MATCHES(implied_volatility(target, ov, {.initial_vol = -0.1}),
                         ValidationError,
                         message_contains("initial_vol"));
}

TEST_CASE("implied vol reports an unbracketable target", "[implied_vol]") {
    // A 60-vol price sits just under the call's price ceiling; six doublings
    // of a 0.02 upper bound only reach 1.28, so the target stays unbracketed.
    const auto ov = bsm(OptionType::Call, 0.2);
    const double near_ceiling = bsm(OptionType::Call, 60.0).present_value();
    CHECK_THROWS_MATCHES(
        implied_volatility(near_ceiling, ov, {.vol_lower = 0.01, .vol_upper = 0.02}),
        ConvergenceError,
        message_contains("Could not bracket the implied volatility"));
}

// ---------------------------------------------------------------------------
// Solver diagnostics, defaults and robustness
// ---------------------------------------------------------------------------

namespace {

OptionValuation american_put(PricingMethod method,
                             double vol,
                             const std::optional<DiscountCurve>& curve = std::nullopt) {
    const MarketData market(kPricingDate, curve.value_or(DiscountCurve::flat(0.05)), "USD");
    std::optional<ValuationParams> params;
    if (method == PricingMethod::Binomial) {
        params = BinomialParams{.num_steps = 200};
    } else {
        params = PdeParams{.spot_steps = 80, .time_steps = 100};
    }
    return {UnderlyingData(100.0, vol, market),
            vanilla(OptionType::Put, ExerciseType::American, 110.0),
            method,
            params};
}

}  // namespace

TEST_CASE("implied vol reports pricing calls", "[implied_vol]") {
    const double target = bsm(OptionType::Call, 0.35).present_value();
    const auto result = implied_volatility(target, bsm(OptionType::Call, 0.2), {});
    REQUIRE(result.converged);
    CHECK(result.well_determined);
    // Bracketing prices both ends; each Newton step prices once more.
    CHECK(result.function_evaluations >= result.iterations + 2);
}

TEST_CASE("numerical engines default to TOMS 748", "[implied_vol]") {
    // TOMS 748 needs no vega; Newton's numerical vega costs 2 extra pricings a
    // step.
    const auto engine = GENERATE(PricingMethod::Binomial, PricingMethod::PdeFd);
    const double target = american_put(engine, 0.3).present_value();
    const auto by_default = implied_volatility(target, american_put(engine, 0.2), {});
    const auto newton = implied_volatility(
        target, american_put(engine, 0.2), {.method = ImpliedVolMethod::NewtonRaphson});
    REQUIRE(by_default.converged);
    REQUIRE(newton.converged);
    CHECK_THAT(by_default.implied_vol, close_to(0.3, 0.0, 1e-6));
    CHECK(by_default.function_evaluations < newton.function_evaluations);
}

TEST_CASE("a quote at intrinsic value is reported as not well determined", "[implied_vol]") {
    // Regression: a deep in-the-money American put is worth its intrinsic
    // value at every small volatility, so the solve ends near zero vol. The
    // conditioning check then bumped the vol down by 1% for a numerical vega
    // and threw on the negative volatility.
    const MarketData market(kPricingDate, DiscountCurve::flat(0.05), "USD");
    const VanillaSpec put = vanilla(OptionType::Put, ExerciseType::American, 120.0);
    const OptionValuation start(UnderlyingData(100.0, 0.3, market), put, PricingMethod::Binomial);

    const dp::test::WarningCapture warnings;
    const auto result = implied_volatility(20.0, start, {});
    CHECK(result.implied_vol < 0.01);
    CHECK_FALSE(result.well_determined);
    CHECK(warnings.any_contains("is not determined by the price"));
}

TEST_CASE("a root on a bracket end returns without solving", "[implied_vol]") {
    const double target = bsm(OptionType::Call, 0.4).present_value();
    const auto result = implied_volatility(
        target, bsm(OptionType::Call, 0.2), {.vol_lower = 0.1, .vol_upper = 0.4});
    REQUIRE(result.converged);
    CHECK(result.implied_vol == 0.4);
    CHECK(result.iterations == 0);
}

TEST_CASE("a bracketing failure reports the interval and residuals", "[implied_vol]") {
    const double target = bsm(OptionType::Call, 3.0).present_value();
    CHECK_THROWS_MATCHES(
        implied_volatility(
            target, bsm(OptionType::Call, 0.2), {.vol_lower = 0.01, .vol_upper = 0.02}),
        ConvergenceError,
        message_contains("gives price residuals"));
}

TEST_CASE("the tree vol floor uses the steepest step of a term curve", "[implied_vol]") {
    // The CRR floor is max_i |r_i - q_i| sqrt(dt), not the average zero rate:
    // on this steep curve the late steps' forward rate is far above the
    // average, so bracketing at the old floor priced an inadmissible tree.
    const Eigen::ArrayXd times{{0.25, 0.5, 1.0}};
    const Eigen::ArrayXd rates{{0.01, 0.10, 0.30}};
    const auto steep = DiscountCurve::from_zero_rates(times, rates);
    const double target = american_put(PricingMethod::Binomial, 0.2, steep).present_value();
    const auto result =
        implied_volatility(target, american_put(PricingMethod::Binomial, 0.3, steep), {});
    CHECK(result.converged);
}

TEST_CASE("numerical engine prices are non-decreasing in vol", "[implied_vol]") {
    // The root finders assume it. Flat (not decreasing) at low vol, where this
    // deep ITM American put is worth exactly its intrinsic value 10.
    const auto engine = GENERATE(PricingMethod::Binomial, PricingMethod::PdeFd);
    const dp::test::WarningCapture quiet;  // low-vol PDE grids warn
    double previous = 0.0;
    for (int i = 0; i < 40; ++i) {
        const double vol = 0.05 + 0.95 * i / 39.0;
        const double price = american_put(engine, vol).present_value();
        CAPTURE(vol, price, previous);
        CHECK(price >= previous);
        previous = price;
    }
}
