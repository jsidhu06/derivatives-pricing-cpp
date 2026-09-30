#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/exceptions.hpp"
#include "dp/market_data.hpp"
#include "dp/params.hpp"
#include "dp/simulation.hpp"
#include "dp/underlying.hpp"
#include "dp/valuation.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <optional>
#include <string>
#include <utility>

using namespace dp;
using dp::test::close_to;
using dp::test::kMaturity;
using dp::test::kPricingDate;
using dp::test::message_contains;
using dp::test::WarningCapture;

namespace {

MarketData md(double rate = 0.05) {
    return {kPricingDate, DiscountCurve::flat(rate), "USD"};
}

GbmProcess gbm(int paths = 200'000, int num_steps = 26) {
    return {md(),
            {.initial_value = 100.0, .volatility = 0.25},
            {.paths = paths, .end_date = kMaturity, .num_steps = num_steps}};
}

BarrierSpec::Params base_params(double barrier,
                                BarrierDirection direction,
                                BarrierAction action,
                                OptionType type = OptionType::Call,
                                ExerciseType exercise = ExerciseType::European) {
    return {.option_type = type,
            .exercise_type = exercise,
            .strike = 100.0,
            .maturity = kMaturity,
            .barrier = barrier,
            .direction = direction,
            .action = action,
            .currency = "USD"};
}

double analytic_pv(BarrierSpec::Params params) {
    const UnderlyingData ud(100.0, 0.25, md());
    return OptionValuation(ud, BarrierSpec(std::move(params)), PricingMethod::Bsm).present_value();
}

}  // namespace

// ---------------------------------------------------------------------------
// Statistical agreement with the analytical engine
// ---------------------------------------------------------------------------

TEST_CASE("discrete MC barriers agree with the binomial engine", "[barrier][mc]") {
    // The binomial half-step tree is the exact-in-model discrete reference;
    // the BGK closed form carries a multi-percent approximation error at
    // monthly monitoring with the barrier this close to spot.
    const auto action = GENERATE(BarrierAction::In, BarrierAction::Out);
    auto params = base_params(115.0, BarrierDirection::Up, action);
    params.monitoring = BarrierMonitoring::Discrete;
    params.num_observations = 12;

    const double mc_pv =
        OptionValuation(gbm(), BarrierSpec(params), PricingMethod::MonteCarlo).present_value();
    const UnderlyingData ud(100.0, 0.25, md());
    const double tree_pv =
        OptionValuation(ud, BarrierSpec(params), PricingMethod::Binomial).present_value();
    CHECK_THAT(mc_pv, close_to(tree_pv, 2.5e-2, 5e-3));
}

TEST_CASE("continuous MC barriers agree with the closed form via the Brownian bridge",
          "[barrier][mc]") {
    const auto action = GENERATE(BarrierAction::In, BarrierAction::Out);
    const auto params = base_params(115.0, BarrierDirection::Up, action);

    // Bridge survival-weighting removes the discretisation bias a naive
    // grid check would carry, so 26 steps suffice for ~1% agreement.
    const double mc_pv =
        OptionValuation(gbm(), BarrierSpec(params), PricingMethod::MonteCarlo).present_value();
    CHECK_THAT(mc_pv, close_to(analytic_pv(params), 1.5e-2, 5e-3));
}

TEST_CASE("MC knock-in plus knock-out equals the vanilla on the same paths", "[barrier][mc]") {
    const auto monitoring = GENERATE(BarrierMonitoring::Continuous, BarrierMonitoring::Discrete);
    auto ki = base_params(115.0, BarrierDirection::Up, BarrierAction::In);
    auto ko = base_params(115.0, BarrierDirection::Up, BarrierAction::Out);
    GbmProcess process = gbm(50'000);
    if (monitoring == BarrierMonitoring::Discrete) {
        for (auto* p : {&ki, &ko}) {
            p->monitoring = BarrierMonitoring::Discrete;
            p->num_observations = 12;
        }
        // The facade injects monitoring dates into the barrier processes'
        // grids; give the vanilla the same dates so all three valuations
        // share one simulation bit-for-bit.
        auto monitoring_dates = date_range(kPricingDate, kMaturity, 13);
        monitoring_dates.erase(monitoring_dates.begin());
        process = process.with_observation_dates(std::move(monitoring_dates));
    }
    const double ki_pv =
        OptionValuation(process, BarrierSpec(ki), PricingMethod::MonteCarlo).present_value();
    const double ko_pv =
        OptionValuation(process, BarrierSpec(ko), PricingMethod::MonteCarlo).present_value();

    const VanillaSpec vanilla({.option_type = OptionType::Call,
                               .exercise_type = ExerciseType::European,
                               .strike = 100.0,
                               .maturity = kMaturity,
                               .currency = "USD"});
    const double vanilla_pv =
        OptionValuation(process, vanilla, PricingMethod::MonteCarlo).present_value();

    // Same seed => same paths; the KI and KO weights are exact complements,
    // so parity holds to float noise, not just statistically.
    CHECK_THAT(ki_pv + ko_pv, close_to(vanilla_pv, 1e-10, 1e-12));
}

// ---------------------------------------------------------------------------
// American LSM with barrier state
// ---------------------------------------------------------------------------

TEST_CASE("American MC knock-ins bracket sensibly", "[barrier][mc]") {
    auto params = base_params(85.0, BarrierDirection::Down, BarrierAction::In, OptionType::Put);
    const GbmProcess process = gbm(50'000);
    const double european =
        OptionValuation(process, BarrierSpec(params), PricingMethod::MonteCarlo).present_value();
    params.exercise_type = ExerciseType::American;
    const double american =
        OptionValuation(process, BarrierSpec(params), PricingMethod::MonteCarlo).present_value();
    CHECK(american >= european - 0.02);
    CHECK(american > 0.0);
}

TEST_CASE("triggered American MC knock-in collapses to the vanilla LSM", "[barrier][mc]") {
    // Spot past a down barrier at 101: every path is active from inception,
    // so the barrier LSM must reproduce the vanilla LSM on the same paths.
    const auto params = base_params(
        101.0, BarrierDirection::Down, BarrierAction::In, OptionType::Put, ExerciseType::American);
    const GbmProcess process = gbm(20'000);
    const double barrier_pv =
        OptionValuation(process, BarrierSpec(params), PricingMethod::MonteCarlo).present_value();

    const VanillaSpec vanilla({.option_type = OptionType::Put,
                               .exercise_type = ExerciseType::American,
                               .strike = 100.0,
                               .maturity = kMaturity,
                               .currency = "USD"});
    const double vanilla_pv =
        OptionValuation(process, vanilla, PricingMethod::MonteCarlo).present_value();
    CHECK_THAT(barrier_pv, close_to(vanilla_pv, 1e-12, 1e-14));
}

TEST_CASE("triggered MC knock-out pays its rebate", "[barrier][mc]") {
    auto params = base_params(
        99.0, BarrierDirection::Up, BarrierAction::Out, OptionType::Call, ExerciseType::American);
    params.rebate = 3.0;
    params.rebate_timing = RebateTiming::AtHit;
    const double pv = OptionValuation(gbm(1000, 12), BarrierSpec(params), PricingMethod::MonteCarlo)
                          .present_value();
    CHECK(pv == 3.0);
}

TEST_CASE("LSM bias warning fires for up-and-out calls", "[barrier][mc]") {
    const WarningCapture warnings;
    const auto params = base_params(
        115.0, BarrierDirection::Up, BarrierAction::Out, OptionType::Call, ExerciseType::American);
    const OptionValuation ov(gbm(1000, 12), BarrierSpec(params), PricingMethod::MonteCarlo);
    REQUIRE_FALSE(warnings.empty());
    CHECK(warnings.messages().front().find("downward bias") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Process and greek policy
// ---------------------------------------------------------------------------

TEST_CASE("MC barrier process and greek policy", "[barrier][mc]") {
    SECTION("continuous monitoring requires GBM") {
        const JdProcess jd(
            md(),
            {.initial_value = 100.0, .volatility = 0.25, .lambd = 0.3, .mu = -0.1, .delta = 0.1},
            {.paths = 1000, .end_date = kMaturity, .num_steps = 12});
        CHECK_THROWS_MATCHES(OptionValuation(jd,
                                             BarrierSpec(base_params(
                                                 115.0, BarrierDirection::Up, BarrierAction::Out)),
                                             PricingMethod::MonteCarlo),
                             UnsupportedFeatureError,
                             message_contains("Continuous barrier monitoring requires GBMProcess"));
    }

    SECTION("discrete monitoring works for jump diffusion") {
        const JdProcess jd(
            md(),
            {.initial_value = 100.0, .volatility = 0.25, .lambd = 0.3, .mu = -0.1, .delta = 0.1},
            {.paths = 20'000, .end_date = kMaturity, .num_steps = 12});
        auto params = base_params(115.0, BarrierDirection::Up, BarrierAction::Out);
        params.monitoring = BarrierMonitoring::Discrete;
        params.num_observations = 12;
        const double pv =
            OptionValuation(jd, BarrierSpec(params), PricingMethod::MonteCarlo).present_value();
        CHECK(pv >= 0.0);
    }

    SECTION("numerical gamma is blocked on MC barriers, delta clamps and works") {
        const OptionValuation ov(
            gbm(20'000, 12),
            BarrierSpec(base_params(115.0, BarrierDirection::Up, BarrierAction::Out)),
            PricingMethod::MonteCarlo);
        CHECK_THROWS_MATCHES(ov.gamma(),
                             UnsupportedFeatureError,
                             message_contains("Numerical gamma is not supported"));
        CHECK(std::isfinite(ov.delta()));
    }

    SECTION("control variate runs on American barrier MC") {
        const MonteCarloParams cv{.control_variate_european = true};
        auto params = base_params(85.0,
                                  BarrierDirection::Down,
                                  BarrierAction::Out,
                                  OptionType::Put,
                                  ExerciseType::American);
        const double pv =
            OptionValuation(gbm(20'000, 12), BarrierSpec(params), PricingMethod::MonteCarlo, cv)
                .present_value();
        CHECK(std::isfinite(pv));
    }
}
