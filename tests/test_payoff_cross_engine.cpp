// Custom PayoffSpec contracts across the tree, PDE and Monte Carlo engines:
// PV and Greek agreement, American >= European, replication by vanilla legs,
// and discrete dividends (ported from test_payoff_spec.py). Per-engine
// smoke tests live in the engine test files.

#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/market_data.hpp"
#include "dp/params.hpp"
#include "dp/simulation.hpp"
#include "dp/underlying.hpp"
#include "dp/valuation.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <string_view>
#include <variant>

using namespace dp;
using Catch::Matchers::WithinRel;
using dp::test::close_to;
using dp::test::flat_curve;
using dp::test::kMaturity;
using dp::test::kPricingDate;

namespace {

constexpr double kRate = 0.05;

double capped_strangle(double s) {
    return std::min(40.0, std::max(90.0 - s, 0.0) + std::max(s - 110.0, 0.0));
}
double bull_call_spread(double s) {
    return std::max(s - 95.0, 0.0) - std::max(s - 115.0, 0.0);
}
double bear_put_spread(double s) {
    return std::max(105.0 - s, 0.0) - std::max(85.0 - s, 0.0);
}
double straddle(double s) {
    return std::max(s - 100.0, 0.0) + std::max(100.0 - s, 0.0);
}
double digital_call(double s) {
    return s > 100.0 ? 1.0 : 0.0;
}

struct Payoff {
    std::string_view label;
    double (*fn)(double) = nullptr;
};

constexpr Payoff kBullSpread{"bull spread", bull_call_spread};
constexpr Payoff kBearSpread{"bear spread", bear_put_spread};
constexpr Payoff kStraddle{"straddle", straddle};
constexpr Payoff kCappedStrangle{"capped strangle", capped_strangle};

MarketData md() {
    return {kPricingDate, flat_curve(kPricingDate, kMaturity, kRate), "USD"};
}

UnderlyingData ud(DividendSchedule dividends = {}) {
    return {100.0, 0.20, md(), std::move(dividends)};
}

GbmProcess gbm() {
    return {md(),
            {.initial_value = 100.0, .volatility = 0.20},
            {.paths = 100'000, .end_date = kMaturity, .num_steps = 52}};
}

PayoffSpec payoff_spec(double (*fn)(double), ExerciseType exercise = ExerciseType::European) {
    return PayoffSpec(
        {.exercise_type = exercise, .maturity = kMaturity, .payoff_fn = fn, .currency = "USD"});
}

VanillaSpec vanilla(OptionType type, double strike) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = ExerciseType::European,
                        .strike = strike,
                        .maturity = kMaturity,
                        .currency = "USD"});
}

const BinomialParams kTree{.num_steps = 500};
const PdeParams kPde{.spot_steps = 400, .time_steps = 400};
const PdeParams kPdeAmerican{
    .spot_steps = 400, .time_steps = 400, .omega = 1.2, .tol = 1e-8, .max_iter = 2000};

PdeParams pde_params(ExerciseType exercise) {
    return exercise == ExerciseType::American ? kPdeAmerican : kPde;
}

OptionValuation tree(const OptionSpec& s, const UnderlyingData& u = ud()) {
    return {u, s, PricingMethod::Binomial, kTree};
}
OptionValuation pde(const OptionSpec& s, const UnderlyingData& u = ud()) {
    const auto exercise = std::visit([](const auto& x) { return x.exercise_type(); }, s);
    return {u, s, PricingMethod::PdeFd, pde_params(exercise)};
}
OptionValuation mc(const OptionSpec& s) {
    return {gbm(), s, PricingMethod::MonteCarlo, MonteCarloParams{}};
}

constexpr GreekOptions kNumerical{.method = GreekCalculationMethod::Numerical};
constexpr GreekOptions kTreeGreek{.method = GreekCalculationMethod::Tree};

}  // namespace

// ---------------------------------------------------------------------------
// Present value across engines
// ---------------------------------------------------------------------------

TEST_CASE("tree and PDE agree on custom payoffs", "[payoff][cross_engine]") {
    const auto payoff = GENERATE(kBullSpread, kBearSpread, kStraddle, kCappedStrangle);
    const auto exercise = GENERATE(ExerciseType::European, ExerciseType::American);
    CAPTURE(payoff.label, to_string(exercise));
    const auto s = payoff_spec(payoff.fn, exercise);
    CHECK_THAT(pde(s).present_value(), WithinRel(tree(s).present_value(), 2e-2));
}

TEST_CASE("digital call: PDE matches a fine tree and stays below its payout", "[payoff][pde]") {
    const auto s = payoff_spec(digital_call);
    const double pde_pv = pde(s).present_value();
    const double tree_pv =
        OptionValuation(ud(), s, PricingMethod::Binomial, BinomialParams{.num_steps = 2000})
            .present_value();
    CHECK(pde_pv > 0.0);
    CHECK(pde_pv < std::exp(-kRate));  // pays at most 1, discounted one year
    CHECK_THAT(pde_pv, WithinRel(tree_pv, 2e-2));
}

TEST_CASE("American custom payoffs are worth at least the European on the tree and PDE",
          "[payoff][american]") {
    const auto payoff = GENERATE(kCappedStrangle, kBullSpread);
    CAPTURE(payoff.label);
    const auto eu = payoff_spec(payoff.fn);
    const auto am = payoff_spec(payoff.fn, ExerciseType::American);
    CHECK(tree(am).present_value() >= tree(eu).present_value() - 1e-8);
    CHECK(pde(am).present_value() >= pde(eu).present_value() - 1e-8);
}

TEST_CASE("custom payoffs respect their no-arbitrage caps", "[payoff]") {
    const double df = std::exp(-kRate);
    CHECK(tree(payoff_spec(bull_call_spread)).present_value() <= 20.0 * df + 0.01);
    CHECK(tree(payoff_spec(capped_strangle)).present_value() <= 40.0 * df + 0.01);
}

// ---------------------------------------------------------------------------
// Replication by vanilla legs (European)
// ---------------------------------------------------------------------------

TEST_CASE("custom payoffs replicate their vanilla legs on the tree and PDE",
          "[payoff][replication]") {
    const auto method = GENERATE(PricingMethod::Binomial, PricingMethod::PdeFd);
    CAPTURE(to_string(method));
    const auto value = [method](const OptionSpec& s) {
        return method == PricingMethod::Binomial ? tree(s).present_value() : pde(s).present_value();
    };
    const auto call = [&](double k) { return value(vanilla(OptionType::Call, k)); };
    const auto put = [&](double k) { return value(vanilla(OptionType::Put, k)); };
    // Python checks 1%. The tree prices on one lattice, so replication is
    // exact up to rounding. On the PDE the custom payoff keeps the legacy
    // 4 * S span while each vanilla leg gets its own auto (sigma*sqrt(T))
    // domain, so the residual is the difference of discretisations
    // (measured <= 6e-4).
    const double rtol = method == PricingMethod::Binomial ? 1e-12 : 1.5e-3;

    CHECK_THAT(value(payoff_spec(bull_call_spread)), WithinRel(call(95.0) - call(115.0), rtol));
    CHECK_THAT(value(payoff_spec(straddle)), WithinRel(call(100.0) + put(100.0), rtol));
    // Capped at 40 on both wings: short the 50 put and the 150 call.
    CHECK_THAT(value(payoff_spec(capped_strangle)),
               WithinRel(put(90.0) - put(50.0) + call(110.0) - call(150.0), rtol));
}

// ---------------------------------------------------------------------------
// Greeks
// ---------------------------------------------------------------------------

TEST_CASE("custom payoff Greeks have the right signs on the tree", "[payoff][greeks]") {
    const auto at_300 = [](double (*fn)(double)) {
        return OptionValuation(
            ud(), payoff_spec(fn), PricingMethod::Binomial, BinomialParams{.num_steps = 300});
    };
    const auto spread = at_300(bull_call_spread);
    const double delta = spread.delta(kNumerical);
    CHECK(delta > 0.0);
    CHECK(delta <= 1.0 + 1e-6);
    CHECK(spread.theta(kNumerical) < 0.0);
    CHECK(std::isfinite(spread.rho(kNumerical)));

    const auto strangle = at_300(capped_strangle);
    CHECK(strangle.gamma(kNumerical) > 0.0);
    CHECK(strangle.vega(kNumerical) > 0.0);
}

TEST_CASE("tree-extracted Greeks track bump-and-revalue on a capped strangle", "[payoff][greeks]") {
    const auto ov = tree(payoff_spec(capped_strangle));
    CHECK_THAT(ov.gamma(kTreeGreek), WithinRel(ov.gamma(kNumerical), 0.10));
    const double theta_tree = ov.theta(kTreeGreek);
    const double theta_numerical = ov.theta(kNumerical);
    CHECK(theta_tree < 0.0);
    CHECK(theta_numerical < 0.0);
    CHECK_THAT(theta_tree, WithinRel(theta_numerical, 0.10));
}

TEST_CASE("custom payoff Greeks agree across engines", "[payoff][greeks][cross_engine]") {
    SECTION("European delta: PDE vs tree") {
        const auto payoff = GENERATE(kBullSpread, kStraddle, kCappedStrangle);
        CAPTURE(payoff.label);
        const auto s = payoff_spec(payoff.fn);
        CHECK_THAT(pde(s).delta(), WithinRel(tree(s).delta(), 5e-2));
    }
    SECTION("American delta: PDE vs tree") {
        const auto payoff = GENERATE(kBullSpread, kStraddle);
        CAPTURE(payoff.label);
        const auto s = payoff_spec(payoff.fn, ExerciseType::American);
        CHECK_THAT(pde(s).delta(), WithinRel(tree(s).delta(), 5e-2));
    }
    SECTION("European vega: PDE vs tree") {
        const auto payoff = GENERATE(kBullSpread, kStraddle, kCappedStrangle);
        CAPTURE(payoff.label);
        const auto s = payoff_spec(payoff.fn);
        CHECK_THAT(pde(s).vega(), WithinRel(tree(s).vega(), 0.10));
    }
    SECTION("European theta: PDE vs tree") {
        const auto payoff = GENERATE(kBullSpread, kStraddle);
        CAPTURE(payoff.label);
        const auto s = payoff_spec(payoff.fn);
        CHECK_THAT(pde(s).theta(), close_to(tree(s).theta(), 0.10, 0.01));
    }
}

// ---------------------------------------------------------------------------
// Monte Carlo (slow under the sanitizer build: 100k paths x 52 dates per run)
// ---------------------------------------------------------------------------

TEST_CASE("Monte Carlo agrees with the tree on custom payoffs",
          "[payoff][mc][cross_engine][slow]") {
    SECTION("present value") {
        const auto payoff = GENERATE(kBullSpread, kBearSpread, kStraddle, kCappedStrangle);
        const auto exercise = GENERATE(ExerciseType::European, ExerciseType::American);
        CAPTURE(payoff.label, to_string(exercise));
        const auto s = payoff_spec(payoff.fn, exercise);
        const double tree_pv = tree(s).present_value();
        const double mc_pv = mc(s).present_value();
        // LSM is a lower-biased Bermudan on 52 dates; its Laguerre basis in
        // S / mean(S) fits the non-monotone payoffs worst (see McEngine)...
        const bool wide =
            exercise == ExerciseType::American &&
            (payoff.label == kStraddle.label || payoff.label == kCappedStrangle.label);
        CHECK_THAT(mc_pv, WithinRel(tree_pv, wide ? 8e-2 : 3e-2));
        if (exercise == ExerciseType::American) {
            // ...so it sits below the tree (measured 1.5-4.6% below), not just near it.
            CHECK(mc_pv < tree_pv * 1.005);
        }
    }
    SECTION("American is worth at least the European") {
        const auto payoff = GENERATE(kCappedStrangle, kBullSpread);
        CAPTURE(payoff.label);
        CHECK(mc(payoff_spec(payoff.fn, ExerciseType::American)).present_value() >=
              mc(payoff_spec(payoff.fn)).present_value() - 1e-8);
    }
    SECTION("European numerical delta") {
        const auto payoff = GENERATE(kBullSpread, kStraddle, kCappedStrangle);
        CAPTURE(payoff.label);
        const auto s = payoff_spec(payoff.fn);
        CHECK_THAT(mc(s).delta(kNumerical), WithinRel(tree(s).delta(), 5e-2));
    }
    SECTION("digital call against its closed form") {
        // Cash-or-nothing: e^{-rT} N(d2). The tree oscillates on this
        // discontinuous payoff (500 steps: 3% low), so it is no reference.
        const double d2 = (kRate - 0.5 * 0.20 * 0.20) / 0.20;
        const double closed_form = std::exp(-kRate) * 0.5 * std::erfc(-d2 / std::numbers::sqrt2);
        CHECK_THAT(mc(payoff_spec(digital_call)).present_value(),
                   WithinRel(closed_form, 1e-2));  // measured 0.2%
    }
}

// ---------------------------------------------------------------------------
// Discrete dividends
// ---------------------------------------------------------------------------

TEST_CASE("PDE and tree agree on custom payoffs with a discrete dividend",
          "[payoff][dividends][cross_engine]") {
    const auto with_div = ud({{make_datetime(2025, 7, 1), 2.0}});
    const auto payoff = GENERATE(kBullSpread, kStraddle);
    const auto exercise = GENERATE(ExerciseType::European, ExerciseType::American);
    CAPTURE(payoff.label, to_string(exercise));
    const auto s = payoff_spec(payoff.fn, exercise);
    CHECK_THAT(pde(s, with_div).present_value(),
               WithinRel(tree(s, with_div).present_value(), 2e-2));
}
