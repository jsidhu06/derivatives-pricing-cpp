// Engine behaviour at the edges of the input space: deep moneyness, near
// expiry, extreme rates and volatilities, American early-exercise limits,
// and the sigma -> 0 limits of the barrier formula and Monte Carlo
// (ported from test_edge_cases.py). BSM-only edges live in
// test_valuation_bsm.cpp; tree zero-vol rejections in
// test_valuation_binomial.cpp.

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
#include <optional>

using namespace dp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using dp::test::close_to;
using dp::test::flat_curve;
using dp::test::kPricingDate;

namespace {

// ~0.5y, as in the Python edge-case suite.
const DateTime kHalfYear = make_datetime(2025, 7, 3);
constexpr double kRate = 0.05;

double half_year_fraction() {
    return year_fraction(kPricingDate, kHalfYear, DayCountConvention::Act365F);
}

MarketData md(double rate = kRate, DateTime maturity = kHalfYear) {
    return {kPricingDate, flat_curve(kPricingDate, maturity, rate), "USD"};
}

UnderlyingData ud(double spot = 100.0,
                  double vol = 0.20,
                  MarketData market = md(),
                  std::optional<DiscountCurve> q_curve = std::nullopt) {
    return {spot, vol, std::move(market), {}, std::move(q_curve)};
}

VanillaSpec spec(OptionType type,
                 double strike = 100.0,
                 ExerciseType exercise = ExerciseType::European,
                 DateTime maturity = kHalfYear) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = exercise,
                        .strike = strike,
                        .maturity = maturity,
                        .currency = "USD"});
}

double pv(const UnderlyingData& u,
          const OptionSpec& s,
          PricingMethod method,
          ValuationParams params) {
    return OptionValuation(u, s, method, params).present_value();
}

double bsm_pv(const UnderlyingData& u, const OptionSpec& s) {
    return OptionValuation(u, s, PricingMethod::Bsm).present_value();
}

double tree_pv(const UnderlyingData& u, const OptionSpec& s, int num_steps) {
    return pv(u, s, PricingMethod::Binomial, BinomialParams{.num_steps = num_steps});
}

double pde_pv(const UnderlyingData& u, const OptionSpec& s, PdeParams params) {
    return pv(u, s, PricingMethod::PdeFd, params);
}

const PdeParams kLogSpot{.spot_steps = 300, .time_steps = 200, .space_grid = PdeSpaceGrid::LogSpot};

}  // namespace

// ---------------------------------------------------------------------------
// Deep moneyness: discounted forward intrinsic, or ~0
// ---------------------------------------------------------------------------

TEST_CASE("deep moneyness prices the discounted intrinsic on the tree and PDE", "[edge]") {
    const double df = std::exp(-kRate * half_year_fraction());
    const double itm_call = 200.0 - 50.0 * df;  // q = 0, so the spot is undiscounted
    const double itm_put = 200.0 * df - 50.0;

    // Python checks 1% (tree) and 2% (PDE); there is no time value left at
    // this moneyness, so both engines land on it to far tighter tolerances
    // (measured: tree ~1e-14, PDE call ~4e-7).
    SECTION("binomial tree") {
        CHECK_THAT(tree_pv(ud(200.0), spec(OptionType::Call, 50.0), 200),
                   WithinRel(itm_call, 1e-10));
        CHECK_THAT(tree_pv(ud(50.0), spec(OptionType::Put, 200.0), 200), WithinRel(itm_put, 1e-10));
        CHECK(tree_pv(ud(50.0), spec(OptionType::Call, 200.0), 200) < 1e-6);
        CHECK(tree_pv(ud(200.0), spec(OptionType::Put, 50.0), 200) < 1e-6);
    }
    SECTION("PDE on a log-spot grid") {
        CHECK_THAT(pde_pv(ud(200.0), spec(OptionType::Call, 50.0), kLogSpot),
                   WithinRel(itm_call, 1e-6));
        CHECK_THAT(pde_pv(ud(50.0), spec(OptionType::Put, 200.0), kLogSpot),
                   WithinRel(itm_put, 1e-6));
        CHECK(pde_pv(ud(50.0), spec(OptionType::Call, 200.0), kLogSpot) < 1e-6);
        CHECK(pde_pv(ud(200.0), spec(OptionType::Put, 50.0), kLogSpot) < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Near expiry: value -> intrinsic
// ---------------------------------------------------------------------------

TEST_CASE("one-day expiry is near intrinsic on the tree and PDE", "[edge]") {
    const auto maturity = kPricingDate + std::chrono::days{1};
    const auto u = ud(100.0, 0.20, md(kRate, maturity));
    struct Case {
        OptionType type = OptionType::Call;
        double strike = 0.0;
        double intrinsic = 0.0;
    };
    const auto c = GENERATE(Case{OptionType::Call, 90.0, 10.0},
                            Case{OptionType::Put, 110.0, 10.0},
                            Case{OptionType::Call, 110.0, 0.0},
                            Case{OptionType::Put, 90.0, 0.0});
    const auto s = spec(c.type, c.strike, ExerciseType::European, maturity);
    const double tree = tree_pv(u, s, 50);
    const double pde = pde_pv(u, s, {.spot_steps = 200, .time_steps = 50});
    CHECK_THAT(tree, WithinAbs(c.intrinsic, 0.5));
    CHECK_THAT(pde, WithinAbs(c.intrinsic, 0.5));
    // ...and both still agree with the closed form (measured < 1e-6).
    const double bsm = bsm_pv(u, s);
    CHECK_THAT(tree, close_to(bsm, 1e-4, 1e-4));
    CHECK_THAT(pde, close_to(bsm, 1e-4, 1e-4));
}

// ---------------------------------------------------------------------------
// Extreme rates and volatilities
// ---------------------------------------------------------------------------

TEST_CASE("negative rates still price on the PDE", "[pde][edge]") {
    const dp::test::WarningCapture capture;  // negative rates warn; that's fine here
    const auto u = ud(100.0, 0.20, md(-0.02));
    for (const auto type : {OptionType::Call, OptionType::Put}) {
        const double value = pde_pv(u, spec(type), {.spot_steps = 200, .time_steps = 200});
        CHECK(std::isfinite(value));
        CHECK(value > 0.0);
        CHECK_THAT(value, WithinRel(bsm_pv(u, spec(type)), 5e-3));
    }
}

TEST_CASE("high rates move the tree price the right way and match BSM", "[binomial][edge]") {
    const auto low = ud(100.0, 0.20, md(0.01));
    const auto high = ud(100.0, 0.20, md(0.30));
    CHECK(tree_pv(high, spec(OptionType::Call), 200) > tree_pv(low, spec(OptionType::Call), 200));
    CHECK(tree_pv(high, spec(OptionType::Put), 200) < tree_pv(low, spec(OptionType::Put), 200));

    const auto extreme = ud(100.0, 0.20, md(0.25));
    CHECK_THAT(tree_pv(extreme, spec(OptionType::Call), 500),
               WithinRel(bsm_pv(extreme, spec(OptionType::Call)), 1e-2));
}

TEST_CASE("very high volatility prices on the tree and PDE", "[edge]") {
    SECTION("sigma = 200% on the tree is finite, positive and BSM-like") {
        const auto u = ud(100.0, 2.0);
        for (const auto type : {OptionType::Call, OptionType::Put}) {
            const double value = tree_pv(u, spec(type), 200);
            CHECK(std::isfinite(value));
            CHECK(value > 0.0);
            CHECK_THAT(value, WithinRel(bsm_pv(u, spec(type)), 1e-2));
        }
    }
    SECTION("sigma = 150% on a log-spot PDE grid matches BSM") {
        const auto u = ud(100.0, 1.5);
        const PdeParams params{
            .spot_steps = 300, .time_steps = 300, .space_grid = PdeSpaceGrid::LogSpot};
        CHECK_THAT(pde_pv(u, spec(OptionType::Call), params),
                   WithinRel(bsm_pv(u, spec(OptionType::Call)), 2e-2));
    }
}

// ---------------------------------------------------------------------------
// American early-exercise limits
// ---------------------------------------------------------------------------

TEST_CASE("deep ITM American put is worth at least intrinsic", "[edge][american]") {
    struct Case {
        double spot = 0.0;
        double strike = 0.0;
    };
    const auto c =
        GENERATE(Case{50.0, 200.0}, Case{40.0, 200.0}, Case{25.0, 180.0}, Case{10.0, 150.0});
    const auto put = spec(OptionType::Put, c.strike, ExerciseType::American);
    const double intrinsic = c.strike - c.spot;
    CHECK(tree_pv(ud(c.spot), put, 500) >= intrinsic - 1e-8);
    CHECK(pde_pv(ud(c.spot), put, {.spot_steps = 300, .time_steps = 300}) >= intrinsic - 0.01);
}

TEST_CASE("American put one to three days out is near intrinsic", "[binomial][edge][american]") {
    const int days = GENERATE(1, 2, 3);
    const auto maturity = kPricingDate + std::chrono::days{days};
    const auto u = ud(80.0, 0.20, md(kRate, maturity));
    const auto put = spec(OptionType::Put, 100.0, ExerciseType::American, maturity);
    CHECK_THAT(tree_pv(u, put, 100), WithinAbs(20.0, 0.5));
}

TEST_CASE("deep ITM American call with a dividend yield dominates the European",
          "[binomial][edge][american]") {
    struct Case {
        double spot = 0.0;
        double strike = 0.0;
    };
    const auto c = GENERATE(Case{120.0, 80.0}, Case{150.0, 90.0}, Case{200.0, 100.0});
    const auto u = ud(c.spot, 0.20, md(), flat_curve(kPricingDate, kHalfYear, 0.05));
    const double american =
        tree_pv(u, spec(OptionType::Call, c.strike, ExerciseType::American), 500);
    const double european = tree_pv(u, spec(OptionType::Call, c.strike), 500);
    CHECK(american >= european - 1e-8);
    CHECK(american >= c.spot - c.strike - 1e-8);
}

TEST_CASE("PDE American call without dividends equals the European", "[pde][edge][american]") {
    // Early exercise of a call is never optimal when q = 0, so the
    // projected solve must leave the European value untouched.
    const PdeParams params{.spot_steps = 400, .time_steps = 400};
    const auto u = ud();
    const double american =
        pde_pv(u, spec(OptionType::Call, 100.0, ExerciseType::American), params);
    const double european = pde_pv(u, spec(OptionType::Call), params);
    CHECK_THAT(american, WithinRel(european, 1e-10));  // measured ~3e-15
}

// ---------------------------------------------------------------------------
// sigma -> 0 limits
// ---------------------------------------------------------------------------

namespace {

BarrierSpec continuous_barrier(double barrier,
                               BarrierDirection direction,
                               BarrierAction action,
                               OptionType type,
                               double strike) {
    return BarrierSpec({.option_type = type,
                        .exercise_type = ExerciseType::European,
                        .strike = strike,
                        .maturity = kHalfYear,
                        .barrier = barrier,
                        .direction = direction,
                        .action = action,
                        .currency = "USD"});
}

}  // namespace

TEST_CASE("analytical barriers fall back to the deterministic forward as sigma -> 0",
          "[barrier][edge]") {
    // (H/S)^(2 lambda) with lambda ~ 1/sigma^2 overflows as sigma -> 0; the
    // engine must return the deterministic-forward limit, never NaN.
    const double maturity = half_year_fraction();
    const double df = std::exp(-kRate * maturity);
    const double forward = 100.0 * std::exp(kRate * maturity);  // ~102.54

    SECTION("up-and-out call whose forward stays below the barrier survives") {
        const double vol = GENERATE(0.0, 1e-12, 1e-10, 1e-8, 1e-4);
        const double value =
            bsm_pv(ud(100.0, vol),
                   continuous_barrier(
                       130.0, BarrierDirection::Up, BarrierAction::Out, OptionType::Call, 100.0));
        CHECK(std::isfinite(value));
        CHECK_THAT(value, close_to(df * (forward - 100.0), 1e-10, 0.0));
    }
    SECTION("up-and-out call whose forward crosses the barrier is knocked out") {
        const double vol = GENERATE(0.0, 1e-12, 1e-8);
        CHECK(
            bsm_pv(ud(100.0, vol),
                   continuous_barrier(
                       102.0, BarrierDirection::Up, BarrierAction::Out, OptionType::Call, 100.0)) ==
            0.0);
    }
    SECTION("down-and-out put whose forward stays above the barrier survives") {
        const double vol = GENERATE(0.0, 1e-12, 1e-8);
        const double value =
            bsm_pv(ud(100.0, vol),
                   continuous_barrier(
                       80.0, BarrierDirection::Down, BarrierAction::Out, OptionType::Put, 110.0));
        CHECK(std::isfinite(value));
        CHECK_THAT(value, close_to(df * (110.0 - forward), 1e-10, 0.0));
    }
    SECTION("up-and-in call whose forward never reaches the barrier is worthless") {
        const double vol = GENERATE(0.0, 1e-12, 1e-8);
        CHECK(
            bsm_pv(ud(100.0, vol),
                   continuous_barrier(
                       130.0, BarrierDirection::Up, BarrierAction::In, OptionType::Call, 100.0)) ==
            0.0);
    }
}

TEST_CASE("BSM and Monte Carlo agree exactly at zero volatility", "[mc][edge]") {
    // Every path is the deterministic forward, so MC has zero variance and
    // both engines return the discounted forward intrinsic.
    struct Case {
        OptionType type = OptionType::Call;
        double strike = 0.0;
    };
    const auto c = GENERATE(Case{OptionType::Call, 90.0},
                            Case{OptionType::Put, 110.0},
                            Case{OptionType::Call, 110.0},
                            Case{OptionType::Put, 90.0});
    const auto s = spec(c.type, c.strike);
    const GbmProcess process(md(),
                             {.initial_value = 100.0, .volatility = 0.0},
                             {.paths = 1000, .end_date = kHalfYear, .num_steps = 12});
    const double mc = OptionValuation(process, s, PricingMethod::MonteCarlo).present_value();
    const double bsm = bsm_pv(ud(100.0, 0.0), s);
    CHECK(std::isfinite(mc));
    CHECK_THAT(mc, close_to(bsm, 1e-10, 1e-10));
}
