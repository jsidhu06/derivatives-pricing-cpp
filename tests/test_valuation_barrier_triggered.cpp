// Barriers already triggered at the pricing date: every greek on every
// engine / greek-method path collapses to the triggered instrument.
//
// A knock-out is dead (its value is the rebate, independent of spot and
// vol): delta / gamma / vega are exactly 0, theta and rho are 0 unless an
// AT_EXPIRY rebate carries discounting. A knock-in has become the vanilla:
// every greek equals an independently built vanilla's, bit for bit. Ports
// TestInceptionTriggeredGreekShortCircuits from the Python suite — the
// NUMERICAL paths exercise the facade's short-circuits, TREE / GRID the
// engines' native triggered handling.

#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/market_data.hpp"
#include "dp/params.hpp"
#include "dp/underlying.hpp"
#include "dp/valuation.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <optional>

using namespace dp;
using dp::test::close_to;
using dp::test::flat_curve;
using dp::test::kMaturity;
using dp::test::kPricingDate;

namespace {

using G = GreekCalculationMethod;

struct Dispatch {
    PricingMethod method;
    GreekCalculationMethod greek;
    const char* id;
};

/// Every path that can produce delta / gamma / theta.
constexpr std::array<Dispatch, 5> kAllPaths{{{PricingMethod::Bsm, G::Numerical, "bsm_num"},
                                             {PricingMethod::Binomial, G::Numerical, "bin_num"},
                                             {PricingMethod::Binomial, G::Tree, "bin_tree"},
                                             {PricingMethod::PdeFd, G::Numerical, "pde_num"},
                                             {PricingMethod::PdeFd, G::Grid, "pde_grid"}}};
/// Vega and rho are bump-and-revalue only on barriers.
constexpr std::array<Dispatch, 3> kNumericalPaths{
    {{PricingMethod::Bsm, G::Numerical, "bsm_num"},
     {PricingMethod::Binomial, G::Numerical, "bin_num"},
     {PricingMethod::PdeFd, G::Numerical, "pde_num"}}};

constexpr double kRebate = 5.0;

const MarketData& market() {
    static const MarketData md(kPricingDate, flat_curve(kPricingDate, kMaturity, 0.05), "USD");
    return md;
}

/// Spot sits on the barrier, so a continuous barrier is triggered.
UnderlyingData underlying() {
    return {100.0, 0.20, market()};
}

BarrierSpec triggered_spec(BarrierAction action,
                           double rebate = 0.0,
                           RebateTiming timing = RebateTiming::AtHit) {
    return BarrierSpec({.option_type = OptionType::Call,
                        .exercise_type = ExerciseType::European,
                        .strike = 100.0,
                        .maturity = kMaturity,
                        .barrier = 100.0,
                        .direction = BarrierDirection::Down,
                        .action = action,
                        .rebate = rebate,
                        .rebate_timing = timing,
                        .currency = "USD"});
}

/// The vanilla a triggered knock-in collapses to, priced with the params the
/// facade resolves for the barrier (so the comparison is exact).
OptionValuation vanilla_equivalent(PricingMethod method) {
    const VanillaSpec spec({.option_type = OptionType::Call,
                            .exercise_type = ExerciseType::European,
                            .strike = 100.0,
                            .maturity = kMaturity,
                            .currency = "USD"});
    std::optional<ValuationParams> params;
    if (method == PricingMethod::Binomial) {
        params = BinomialParams{.num_steps = 1000};
    } else if (method == PricingMethod::PdeFd) {
        params = PdeParams::for_barriers(BarrierMonitoring::Continuous);
    }
    return {underlying(), spec, method, params};
}

double maturity_df() {
    return market().discount_curve().df(year_fraction(kPricingDate, kMaturity));
}

}  // namespace

TEST_CASE("triggered knock-outs have zero spot and vol greeks", "[barrier][greeks][triggered]") {
    struct Rebate {
        double amount;
        RebateTiming timing;
        double expected_pv;
    };
    const std::array<Rebate, 3> rebates{
        {{0.0, RebateTiming::AtHit, 0.0},
         {kRebate, RebateTiming::AtHit, kRebate},
         {kRebate, RebateTiming::AtExpiry, kRebate * maturity_df()}}};
    for (const auto& r : rebates) {
        const BarrierSpec spec = triggered_spec(BarrierAction::Out, r.amount, r.timing);
        for (const auto& path : kAllPaths) {
            INFO(path.id << " rebate " << r.amount << " " << to_string(r.timing));
            const OptionValuation ov(underlying(), spec, path.method);
            CHECK_THAT(ov.present_value(), close_to(r.expected_pv, 0.0, 1e-10));
            CHECK(ov.delta({.method = path.greek}) == 0.0);
            CHECK(ov.gamma({.method = path.greek}) == 0.0);
        }
        for (const auto& path : kNumericalPaths) {
            INFO(path.id << " rebate " << r.amount << " " << to_string(r.timing));
            const OptionValuation ov(underlying(), spec, path.method);
            CHECK(ov.vega({.method = path.greek}) == 0.0);
        }
    }
}

TEST_CASE("triggered knock-out theta and rho carry only an at-expiry rebate",
          "[barrier][greeks][triggered]") {
    const double ttm = year_fraction(kPricingDate, kMaturity);
    const double r = -std::log(maturity_df()) / ttm;

    SECTION("no rebate and at-hit rebates: nothing left to carry") {
        for (const double rebate : {0.0, kRebate}) {
            const BarrierSpec spec =
                triggered_spec(BarrierAction::Out, rebate, RebateTiming::AtHit);
            for (const auto& path : kAllPaths) {
                INFO(path.id << " rebate " << rebate);
                const OptionValuation ov(underlying(), spec, path.method);
                CHECK(ov.theta({.method = path.greek}) == 0.0);
            }
            for (const auto& path : kNumericalPaths) {
                INFO(path.id << " rebate " << rebate);
                const OptionValuation ov(underlying(), spec, path.method);
                CHECK(ov.rho({.method = path.greek}) == 0.0);
            }
        }
    }

    SECTION("at-expiry rebate accretes like a zero-coupon bond") {
        const BarrierSpec spec =
            triggered_spec(BarrierAction::Out, kRebate, RebateTiming::AtExpiry);
        // theta = r * PV per calendar day.
        const double expected_theta = r * kRebate * maturity_df() / 365.0;
        // rho: central difference of R * df over a 1% parallel shift, per 1%.
        const auto& curve = market().discount_curve();
        const double df_up = curve.bump_parallel_zero_rate(0.005).df(ttm);
        const double df_dn = curve.bump_parallel_zero_rate(-0.005).df(ttm);
        const double expected_rho = kRebate * (df_up - df_dn) / 0.01 * 0.01;
        for (const auto& path : kAllPaths) {
            INFO(path.id);
            const OptionValuation ov(underlying(), spec, path.method);
            CHECK_THAT(ov.theta({.method = path.greek}), close_to(expected_theta, 0.0, 1e-6));
        }
        for (const auto& path : kNumericalPaths) {
            INFO(path.id);
            const OptionValuation ov(underlying(), spec, path.method);
            CHECK_THAT(ov.rho({.method = path.greek}), close_to(expected_rho, 0.0, 1e-6));
        }
    }
}

TEST_CASE("triggered knock-ins reproduce the vanilla's greeks exactly",
          "[barrier][greeks][triggered]") {
    const BarrierSpec spec = triggered_spec(BarrierAction::In);
    for (const auto& path : kAllPaths) {
        INFO(path.id);
        const OptionValuation ki(underlying(), spec, path.method);
        const OptionValuation vanilla = vanilla_equivalent(path.method);
        CHECK(ki.delta({.method = path.greek}) == vanilla.delta({.method = path.greek}));
        CHECK(ki.gamma({.method = path.greek}) == vanilla.gamma({.method = path.greek}));
        CHECK(ki.theta({.method = path.greek}) == vanilla.theta({.method = path.greek}));
    }
    for (const auto& path : kNumericalPaths) {
        INFO(path.id);
        const OptionValuation ki(underlying(), spec, path.method);
        const OptionValuation vanilla = vanilla_equivalent(path.method);
        CHECK(ki.vega({.method = path.greek}) == vanilla.vega({.method = path.greek}));
        CHECK(ki.rho({.method = path.greek}) == vanilla.rho({.method = path.greek}));
    }
}
