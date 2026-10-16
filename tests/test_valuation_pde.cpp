#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/exceptions.hpp"
#include "dp/market_data.hpp"
#include "dp/params.hpp"
#include "dp/underlying.hpp"
#include "dp/valuation.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <span>

using namespace dp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using dp::test::close_to;
using dp::test::kMaturity;
using dp::test::kPricingDate;
using dp::test::message_contains;

namespace {

MarketData md(double rate = 0.05) {
    return {kPricingDate, DiscountCurve::flat(rate), "USD"};
}

UnderlyingData ud(double spot = 100.0,
                  double vol = 0.20,
                  std::optional<DiscountCurve> q_curve = std::nullopt,
                  DividendSchedule dividends = {},
                  MarketData market = md()) {
    return {spot, vol, std::move(market), std::move(dividends), std::move(q_curve)};
}

VanillaSpec spec(OptionType type,
                 ExerciseType exercise = ExerciseType::European,
                 double strike = 100.0,
                 DateTime maturity = kMaturity) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = exercise,
                        .strike = strike,
                        .maturity = maturity,
                        .currency = "USD"});
}

double pde_pv(const UnderlyingData& u, const OptionSpec& s, PdeParams params = {}) {
    return OptionValuation(u, s, PricingMethod::PdeFd, params).present_value();
}

constexpr GreekOptions kGrid{.method = GreekCalculationMethod::Grid};
constexpr GreekOptions kNumerical{.method = GreekCalculationMethod::Numerical};

}  // namespace

// ---------------------------------------------------------------------------
// Scheme / grid equivalence (ported from test_pde.py)
// ---------------------------------------------------------------------------

TEST_CASE("PDE scheme and grid variants agree for European options", "[pde]") {
    const auto q_curve = DiscountCurve::flat(0.01);
    const auto u = ud(100.0, 0.20, q_curve);
    const auto call = spec(OptionType::Call);

    const double baseline = pde_pv(u, call, {.spot_steps = 160, .time_steps = 240});

    for (const auto method : {PdeMethod::Implicit,
                              PdeMethod::Explicit,
                              PdeMethod::ExplicitHull,
                              PdeMethod::CrankNicolson}) {
        for (const auto grid : {PdeSpaceGrid::Spot, PdeSpaceGrid::LogSpot}) {
            const PdeParams params{.spot_steps = 160,
                                   .time_steps = 240,
                                   .method = method,
                                   .space_grid = grid,
                                   .american_solver = PdeEarlyExercise::Intrinsic};
            const bool explicit_method =
                method == PdeMethod::Explicit || method == PdeMethod::ExplicitHull;
            if (explicit_method && grid == PdeSpaceGrid::Spot) {
                CHECK_THROWS_MATCHES(pde_pv(u, call, params),
                                     StabilityError,
                                     message_contains("Explicit spot-grid scheme likely unstable"));
                continue;
            }
            INFO("method " << to_string(method) << " grid " << to_string(grid));
            CHECK_THAT(pde_pv(u, call, params), WithinRel(baseline, 5e-3));
        }
    }
}

TEST_CASE("PDE scheme, grid and solver variants agree for American options", "[pde]") {
    const auto u = ud(95.0);
    const auto put = spec(OptionType::Put, ExerciseType::American);
    const double baseline = pde_pv(u, put, {.spot_steps = 160, .time_steps = 240});

    for (const auto method : {PdeMethod::Implicit,
                              PdeMethod::Explicit,
                              PdeMethod::ExplicitHull,
                              PdeMethod::CrankNicolson}) {
        for (const auto grid : {PdeSpaceGrid::Spot, PdeSpaceGrid::LogSpot}) {
            for (const auto solver : {PdeEarlyExercise::Intrinsic, PdeEarlyExercise::GaussSeidel}) {
                const PdeParams params{.spot_steps = 160,
                                       .time_steps = 240,
                                       .method = method,
                                       .space_grid = grid,
                                       .american_solver = solver};
                const bool explicit_method =
                    method == PdeMethod::Explicit || method == PdeMethod::ExplicitHull;
                if (explicit_method && solver == PdeEarlyExercise::GaussSeidel) {
                    CHECK_THROWS_MATCHES(pde_pv(u, put, params),
                                         UnsupportedFeatureError,
                                         message_contains("GAUSS_SEIDEL is not supported"));
                    continue;
                }
                if (explicit_method && grid == PdeSpaceGrid::Spot) {
                    CHECK_THROWS_AS(pde_pv(u, put, params), StabilityError);
                    continue;
                }
                INFO("method " << to_string(method) << " grid " << to_string(grid) << " solver "
                               << to_string(solver));
                CHECK_THAT(pde_pv(u, put, params), WithinRel(baseline, 5e-3));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Agreement with BSM / binomial
// ---------------------------------------------------------------------------

TEST_CASE("PDE European agrees with BSM", "[pde]") {
    const auto type = GENERATE(OptionType::Call, OptionType::Put);
    const auto u = ud();
    const OptionValuation bsm(u, spec(type), PricingMethod::Analytical);
    const OptionValuation pde(
        u, spec(type), PricingMethod::PdeFd, PdeParams{.spot_steps = 400, .time_steps = 400});
    CHECK_THAT(pde.present_value(), WithinRel(bsm.present_value(), 1e-3));
    CHECK_THAT(pde.delta(kGrid), WithinRel(bsm.delta(), 1e-2));
    CHECK_THAT(pde.gamma(kGrid), WithinRel(bsm.gamma(), 1e-2));
    CHECK_THAT(pde.theta(kGrid), WithinRel(bsm.theta(), 2e-2));
}

TEST_CASE("PDE American put agrees with the binomial tree", "[pde]") {
    const auto u = ud();
    const auto put = spec(OptionType::Put, ExerciseType::American);
    const double pde = pde_pv(u, put, {.spot_steps = 400, .time_steps = 400});
    const double tree =
        OptionValuation(u, put, PricingMethod::Binomial, BinomialParams{.num_steps = 2000})
            .present_value();
    CHECK_THAT(pde, WithinRel(tree, 2e-3));
    CHECK(pde > pde_pv(u, spec(OptionType::Put), {.spot_steps = 400, .time_steps = 400}));
}

TEST_CASE("log-grid calls keep put-call parity at large sigma * sqrt(T)", "[pde][edge]") {
    // The log-grid diffusion weight is fitted so the stencil carries V = S
    // exactly. Unfitted, the call's forward part drifted by ~1.5e-4
    // (sigma sqrt(T))^4 in the exponent: a parity gap of -0.66 at 80% vol
    // over 10 years, and a 5-year call at 500% vol priced at 9.05.
    const auto [vol, years] = GENERATE(std::pair{0.8, 10}, std::pair{2.0, 5}, std::pair{5.0, 5});
    CAPTURE(vol, years);
    const auto u = ud(100.0, vol, DiscountCurve::flat(0.01), {}, md(0.03));
    const DateTime maturity = kPricingDate + std::chrono::days{365 * years};
    const auto call = spec(OptionType::Call, ExerciseType::European, 70.0, maturity);
    const auto put = spec(OptionType::Put, ExerciseType::European, 70.0, maturity);
    const double c = pde_pv(u, call);
    const double p = pde_pv(u, put);
    const double t = static_cast<double>(years);
    const double forward_leg = 100.0 * std::exp(-0.01 * t) - 70.0 * std::exp(-0.03 * t);
    CHECK_THAT(c - p, WithinAbs(forward_leg, 2e-4));  // measured <= 8e-5 (time stepping)
    CHECK_THAT(
        c, WithinRel(OptionValuation(u, call, PricingMethod::Analytical).present_value(), 1e-3));
}

TEST_CASE("grid theta is negative and close to BSM", "[pde][greeks]") {
    const auto type = GENERATE(OptionType::Call, OptionType::Put);
    const auto u = ud();
    const OptionValuation pde(
        u, spec(type), PricingMethod::PdeFd, PdeParams{.spot_steps = 400, .time_steps = 400});
    const OptionValuation bsm(u, spec(type), PricingMethod::Analytical);
    const double theta = pde.theta(kGrid);
    CHECK(theta < 0.0);
    CHECK_THAT(theta, WithinRel(bsm.theta(), 2e-2));
}

TEST_CASE("grid Greeks are the auto-selected method for PDE", "[pde][greeks]") {
    const OptionValuation ov(ud(), spec(OptionType::Call), PricingMethod::PdeFd, PdeParams{});
    CHECK(ov.delta() == ov.delta(kGrid));
    CHECK(ov.gamma() == ov.gamma(kGrid));
    CHECK(ov.theta() == ov.theta(kGrid));
    // vega/rho fall back to bump-and-revalue; GRID is rejected for them.
    CHECK_THROWS_MATCHES(
        ov.vega(kGrid), ValidationError, message_contains("Grid extraction is not available"));
    CHECK_THROWS_MATCHES(
        ov.rho(kGrid), ValidationError, message_contains("Grid extraction is not available"));
    CHECK(ov.vega() > 0.0);
}

TEST_CASE("numerical PDE Greeks reuse the base valuation's grid domain", "[pde][greeks]") {
    // Regression: the auto domain is sized from the volatility and the
    // forward, so the two legs of a vega or rho bump were solved on different
    // grids and the difference in grid error biased the Greek (vega by 1%,
    // rho by 1.5% here on the Spot grid). The bumped legs now inherit the
    // base valuation's domain.
    const auto grid = GENERATE(PdeSpaceGrid::Spot, PdeSpaceGrid::LogSpot);
    CAPTURE(grid);
    const VanillaSpec call = spec(OptionType::Call, ExerciseType::European, 105.0);
    const OptionValuation pde(ud(), call, PricingMethod::PdeFd, PdeParams{.space_grid = grid});
    const OptionValuation bsm(ud(), call, PricingMethod::Analytical);
    CHECK_THAT(pde.vega(), WithinRel(bsm.vega(), 1e-3));
    CHECK_THAT(pde.rho(), WithinRel(bsm.rho(), 1e-3));
}

TEST_CASE("deep ITM American grid Greeks short-circuit to intrinsic", "[pde][greeks]") {
    // Spot far in the exercise region of an American put.
    const OptionValuation ov(
        ud(40.0), spec(OptionType::Put, ExerciseType::American), PricingMethod::PdeFd, PdeParams{});
    CHECK(ov.delta(kGrid) == -1.0);
    CHECK(ov.gamma(kGrid) == 0.0);
    CHECK(ov.theta(kGrid) == 0.0);
}

// ---------------------------------------------------------------------------
// Custom payoffs
// ---------------------------------------------------------------------------

namespace {

double capped_strangle(double s) {
    return std::min(40.0, std::max(90.0 - s, 0.0) + std::max(s - 110.0, 0.0));
}

PayoffSpec payoff_spec(ExerciseType exercise = ExerciseType::European) {
    return PayoffSpec({.exercise_type = exercise,
                       .maturity = kMaturity,
                       .payoff_fn = capped_strangle,
                       .currency = "USD"});
}

}  // namespace

TEST_CASE("custom payoff prices across schemes and grids", "[pde][payoff]") {
    const auto q_curve = DiscountCurve::flat(0.01);
    const auto u = ud(100.0, 0.20, q_curve);
    const auto exercise = GENERATE(ExerciseType::European, ExerciseType::American);
    const auto s = payoff_spec(exercise);
    const double baseline = pde_pv(u, s, {.spot_steps = 400, .time_steps = 400});
    CHECK(baseline > 0.0);
    CHECK(baseline < 40.0);

    const double rtol = exercise == ExerciseType::American ? 2e-2 : 5e-3;
    for (const auto method : {PdeMethod::Implicit, PdeMethod::ExplicitHull}) {
        const bool explicit_method = method == PdeMethod::ExplicitHull;
        const PdeParams params{.spot_steps = 400,
                               .time_steps = 400,
                               .method = method,
                               .space_grid = PdeSpaceGrid::LogSpot,
                               .american_solver = explicit_method ? PdeEarlyExercise::Intrinsic
                                                                  : PdeEarlyExercise::GaussSeidel};
        CHECK_THAT(pde_pv(u, s, params), WithinRel(baseline, rtol));
    }
}

TEST_CASE("custom payoff replicates vanilla legs on the PDE grid", "[pde][payoff]") {
    const PayoffSpec spread(
        {.exercise_type = ExerciseType::European,
         .maturity = kMaturity,
         .payoff_fn = [](double s) { return std::max(s - 95.0, 0.0) - std::max(s - 115.0, 0.0); },
         .currency = "USD"});
    const PdeParams params{.spot_steps = 400, .time_steps = 400};
    const double spread_pv = pde_pv(ud(), spread, params);
    const double long_leg =
        pde_pv(ud(), spec(OptionType::Call, ExerciseType::European, 95.0), params);
    const double short_leg =
        pde_pv(ud(), spec(OptionType::Call, ExerciseType::European, 115.0), params);
    CHECK_THAT(spread_pv, WithinRel(long_leg - short_leg, 5e-3));
}

TEST_CASE("explicit boundary model matches the auto fit", "[pde][payoff]") {
    // The capped strangle is affine on both wings, so an exact model and the
    // fitted one should agree closely.
    const PayoffSpec with_model({.exercise_type = ExerciseType::European,
                                 .maturity = kMaturity,
                                 .payoff_fn = capped_strangle,
                                 .currency = "USD",
                                 .boundary_model = PayoffBoundaryModel{{0.0, 40.0}, {0.0, 40.0}}});
    const PdeParams params{.spot_steps = 300, .time_steps = 300};
    CHECK_THAT(pde_pv(ud(), with_model, params),
               WithinRel(pde_pv(ud(), payoff_spec(), params), 1e-3));
}

// ---------------------------------------------------------------------------
// Dividends / validation / control variate
// ---------------------------------------------------------------------------

TEST_CASE("discrete dividends reduce the PDE call price", "[pde]") {
    const auto base = pde_pv(ud(), spec(OptionType::Call));
    const auto with_div =
        pde_pv(ud(100.0, 0.20, std::nullopt, {{kPricingDate + std::chrono::days{180}, 1.5}}),
               spec(OptionType::Call));
    CHECK(with_div < base);
}

TEST_CASE("PDE rejects zero volatility", "[pde][edge]") {
    CHECK_THROWS_MATCHES(pde_pv(ud(100.0, 0.0), spec(OptionType::Call)),
                         ValidationError,
                         message_contains("volatility must be positive"));
}

TEST_CASE("PDE control variate equals the explicit adjustment identity", "[pde][cv]") {
    const UnderlyingData u(100.0, 0.20, md(0.04));
    const auto amer = spec(OptionType::Put, ExerciseType::American, 95.0);
    const auto euro = spec(OptionType::Put, ExerciseType::European, 95.0);
    const PdeParams base{.spot_steps = 60, .time_steps = 60};
    PdeParams cv = base;
    cv.control_variate_european = true;

    const double american_raw = pde_pv(u, amer, base);
    const double european_num = pde_pv(u, euro, base);
    const double european_bsm = OptionValuation(u, euro, PricingMethod::Analytical).present_value();
    CHECK_THAT(pde_pv(u, amer, cv), WithinAbs(american_raw + (european_bsm - european_num), 1e-6));
}

// ---------------------------------------------------------------------------
// Golden values from the Python reference implementation
// ---------------------------------------------------------------------------

namespace {

// Field order mirrors the generator's column order; padding is irrelevant
// for a test fixture.
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct PdeGoldenRow {
    const char* label;
    double spot;
    double strike;
    double rate;
    double div_yield;
    double vol;
    int maturity_days;
    int spot_steps;
    int time_steps;
    const char* method;
    const char* space;
    const char* solver;
    int rannacher;
    double div_amount;
    int div_day;
    bool is_call;
    bool is_american;
    bool control_variate;
    double pv;
    double delta_grid;
    double gamma_grid;
    double theta_grid;
    double vega_num;
    double rho_num;
};

#include "golden/pde_reference.inc"

PdeMethod method_of(const char* code) {
    if (std::strcmp(code, "IM") == 0) {
        return PdeMethod::Implicit;
    }
    if (std::strcmp(code, "EX") == 0) {
        return PdeMethod::Explicit;
    }
    if (std::strcmp(code, "EH") == 0) {
        return PdeMethod::ExplicitHull;
    }
    return PdeMethod::CrankNicolson;
}

OptionValuation replay(const PdeGoldenRow& row) {
    const DateTime maturity = kPricingDate + std::chrono::days{row.maturity_days};
    const MarketData market(kPricingDate, DiscountCurve::flat(row.rate), "USD");
    std::optional<DiscountCurve> q_curve;
    if (row.div_yield != 0.0) {
        q_curve = DiscountCurve::flat(row.div_yield);
    }
    DividendSchedule dividends;
    if (row.div_amount != 0.0) {
        dividends.push_back({kPricingDate + std::chrono::days{row.div_day}, row.div_amount});
    }
    const UnderlyingData u(row.spot, row.vol, market, std::move(dividends), std::move(q_curve));
    const VanillaSpec s(
        {.option_type = row.is_call ? OptionType::Call : OptionType::Put,
         .exercise_type = row.is_american ? ExerciseType::American : ExerciseType::European,
         .strike = row.strike,
         .maturity = maturity});
    const PdeParams params{
        .spot_steps = row.spot_steps,
        .time_steps = row.time_steps,
        .method = method_of(row.method),
        .rannacher_steps = row.rannacher,
        .space_grid = std::strcmp(row.space, "L") == 0 ? PdeSpaceGrid::LogSpot : PdeSpaceGrid::Spot,
        .american_solver = std::strcmp(row.solver, "IN") == 0 ? PdeEarlyExercise::Intrinsic
                                                              : PdeEarlyExercise::GaussSeidel,
        .control_variate_european = row.control_variate,
    };
    return {u, s, PricingMethod::PdeFd, params};
}

}  // namespace

TEST_CASE("PDE matches the Python reference implementation", "[pde][golden]") {
    for (const auto& row : std::span(kPdeGolden)) {
        INFO("row " << row.label);
        const auto ov = replay(row);
        CHECK_THAT(ov.present_value(), close_to(row.pv, 1e-9, 1e-11));
        if (row.control_variate) {
            continue;  // CV rows pin PV only.
        }
        CHECK_THAT(ov.delta(kGrid), close_to(row.delta_grid, 1e-8, 1e-10));
        CHECK_THAT(ov.gamma(kGrid), close_to(row.gamma_grid, 1e-7, 1e-9));
        CHECK_THAT(ov.theta(kGrid), close_to(row.theta_grid, 1e-7, 1e-9));
        CHECK_THAT(ov.vega(kNumerical), close_to(row.vega_num, 1e-7, 1e-9));
        CHECK_THAT(ov.rho(kNumerical), close_to(row.rho_num, 1e-7, 1e-9));
    }
}

// ---------------------------------------------------------------------------
// PSOR convergence
// ---------------------------------------------------------------------------

namespace {

/// One PSOR sweep per time step: never converges.
PdeParams starved_psor() {
    return {.spot_steps = 100,
            .time_steps = 50,
            .max_iter = 1,
            .american_solver = PdeEarlyExercise::GaussSeidel};
}

BarrierSpec american_down_put(BarrierAction action) {
    return BarrierSpec({.option_type = OptionType::Put,
                        .exercise_type = ExerciseType::American,
                        .strike = 100.0,
                        .maturity = kMaturity,
                        .barrier = 85.0,
                        .direction = BarrierDirection::Down,
                        .action = action,
                        .currency = "USD"});
}

}  // namespace

TEST_CASE("PSOR non-convergence warns instead of passing silently", "[pde][psor]") {
    SECTION("vanilla") {
        const dp::test::WarningCapture capture;
        (void)pde_pv(ud(), spec(OptionType::Put, ExerciseType::American), starved_psor());
        CHECK(capture.any_contains("PDE: PSOR did not converge within max_iter=1"));
    }
    SECTION("barrier knock-out") {
        const dp::test::WarningCapture capture;
        (void)pde_pv(ud(), american_down_put(BarrierAction::Out), starved_psor());
        CHECK(capture.any_contains("PDE barrier: PSOR did not converge"));
    }
    SECTION("barrier knock-in (coupled surfaces)") {
        const dp::test::WarningCapture capture;
        (void)pde_pv(ud(), american_down_put(BarrierAction::In), starved_psor());
        CHECK(capture.any_contains("PDE barrier KI: PSOR did not converge"));
    }
}

TEST_CASE("PSOR converges silently with the default parameters", "[pde][psor]") {
    const dp::test::WarningCapture capture;
    (void)pde_pv(ud(),
                 spec(OptionType::Put, ExerciseType::American),
                 PdeParams{.american_solver = PdeEarlyExercise::GaussSeidel});
    CHECK_FALSE(capture.any_contains("PSOR"));
}

TEST_CASE("control variate requires the engine to share ANALYTICAL's dividend model",
          "[pde][binomial][cv][dividends]") {
    // ANALYTICAL and BINOMIAL use the escrowed model; PDE_FD drops the spot on
    // the ex-date. Mixing the two in the correction is rejected.
    const DateTime ex_date = kPricingDate + std::chrono::days{120};
    const UnderlyingData with_dividend(100.0, 0.25, md(0.03), {{ex_date, 2.0}});
    const auto amer = spec(OptionType::Put, ExerciseType::American, 100.0);
    const auto euro = spec(OptionType::Put, ExerciseType::European, 100.0);

    SECTION("PDE_FD is rejected") {
        PdeParams cv{.spot_steps = 60, .time_steps = 60};
        cv.control_variate_european = true;
        CHECK_THROWS_MATCHES(pde_pv(with_dividend, amer, cv),
                             UnsupportedFeatureError,
                             message_contains("PDE_FD with discrete dividends"));
    }
    SECTION("BINOMIAL shares the escrowed model") {
        const BinomialParams base{.num_steps = 300};
        const BinomialParams cv{.num_steps = 300, .control_variate_european = true};
        const auto tree = [&](const OptionSpec& s, const BinomialParams& p) {
            return OptionValuation(with_dividend, s, PricingMethod::Binomial, p).present_value();
        };
        const double euro_bsm =
            OptionValuation(with_dividend, euro, PricingMethod::Analytical).present_value();
        CHECK_THAT(tree(amer, cv),
                   WithinAbs(tree(amer, base) + (euro_bsm - tree(euro, base)), 1e-8));
        // Same model: the correction is only the tree's small European error.
        CHECK(std::abs(euro_bsm - tree(euro, base)) < 1e-2);
    }
    SECTION("dividends after maturity are ignored") {
        const UnderlyingData after(
            100.0, 0.25, md(0.03), {{kMaturity + std::chrono::days{30}, 2.0}});
        PdeParams cv{.spot_steps = 60, .time_steps = 60};
        cv.control_variate_european = true;
        CHECK(pde_pv(after, amer, cv) > 0.0);
    }
}
