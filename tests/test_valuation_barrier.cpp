#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/exceptions.hpp"
#include "dp/market_data.hpp"
#include "dp/simulation.hpp"
#include "dp/underlying.hpp"
#include "dp/valuation.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <cmath>
#include <optional>
#include <span>
#include <vector>

using namespace dp;
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
                  double vol = 0.25,
                  double rate = 0.05,
                  double div_yield = 0.02) {
    std::optional<DiscountCurve> q_curve;
    if (div_yield != 0.0) {
        q_curve = DiscountCurve::flat(div_yield);
    }
    return {spot, vol, md(rate), {}, std::move(q_curve)};
}

BarrierSpec barrier_spec(BarrierSpec::Params params) {
    if (!params.currency.has_value()) {
        params.currency = "USD";
    }
    return BarrierSpec(params);
}

BarrierSpec::Params base_params(double barrier,
                                BarrierDirection direction,
                                BarrierAction action,
                                OptionType type = OptionType::Call,
                                double strike = 100.0) {
    return {.option_type = type,
            .exercise_type = ExerciseType::European,
            .strike = strike,
            .maturity = kMaturity,
            .barrier = barrier,
            .direction = direction,
            .action = action};
}

VanillaSpec vanilla_spec(OptionType type, double strike = 100.0) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = ExerciseType::European,
                        .strike = strike,
                        .maturity = kMaturity,
                        .currency = "USD"});
}

constexpr GreekOptions kNumerical{.method = GreekCalculationMethod::Numerical};

}  // namespace

// ---------------------------------------------------------------------------
// Structural properties
// ---------------------------------------------------------------------------

TEST_CASE("knock-in plus knock-out equals the vanilla", "[barrier]") {
    const auto type = GENERATE(OptionType::Call, OptionType::Put);
    const auto direction = GENERATE(BarrierDirection::Up, BarrierDirection::Down);
    const double barrier = direction == BarrierDirection::Up ? 120.0 : 80.0;

    const auto underlying = ud();
    const double knock_in =
        OptionValuation(underlying,
                        barrier_spec(base_params(barrier, direction, BarrierAction::In, type)),
                        PricingMethod::Analytical)
            .present_value();
    const double knock_out =
        OptionValuation(underlying,
                        barrier_spec(base_params(barrier, direction, BarrierAction::Out, type)),
                        PricingMethod::Analytical)
            .present_value();
    const double vanilla =
        OptionValuation(underlying, vanilla_spec(type), PricingMethod::Analytical).present_value();

    CHECK_THAT(knock_in + knock_out, close_to(vanilla, 1e-10, 1e-12));
}

TEST_CASE("up-and-out call with barrier at or below strike is worthless", "[barrier]") {
    const OptionValuation ov(
        ud(),
        barrier_spec(base_params(
            110.0, BarrierDirection::Up, BarrierAction::Out, OptionType::Call, /*strike=*/115.0)),
        PricingMethod::Analytical);
    CHECK(ov.present_value() == 0.0);
}

TEST_CASE("discrete monitoring raises a knock-out's value", "[barrier]") {
    // Fewer observations -> fewer chances to knock out -> the BGK-adjusted
    // barrier moves away from the spot and the knock-out is worth more.
    auto discrete = base_params(120.0, BarrierDirection::Up, BarrierAction::Out);
    discrete.monitoring = BarrierMonitoring::Discrete;
    discrete.num_observations = 12;

    const double continuous_pv =
        OptionValuation(ud(),
                        barrier_spec(base_params(120.0, BarrierDirection::Up, BarrierAction::Out)),
                        PricingMethod::Analytical)
            .present_value();
    const double discrete_pv =
        OptionValuation(ud(), barrier_spec(discrete), PricingMethod::Analytical).present_value();
    CHECK(discrete_pv > continuous_pv);
}

TEST_CASE("sigma -> 0 collapses to the deterministic forward", "[barrier]") {
    // r = 5%, q = 0: the forward 100*e^0.05 = 105.13 crosses an up barrier at
    // 104 (knock-out dies; knock-in is the discounted forward intrinsic) and
    // never reaches one at 110 (knock-out lives, knock-in expires worthless).
    const auto underlying = ud(100.0, /*vol=*/0.0, 0.05, 0.0);
    const double maturity = year_fraction(kPricingDate, kMaturity, DayCountConvention::Act365F);
    const double df = std::exp(-0.05 * maturity);
    const double forward = 100.0 * std::exp(0.05 * maturity);

    const double dead_ko =
        OptionValuation(underlying,
                        barrier_spec(base_params(104.0, BarrierDirection::Up, BarrierAction::Out)),
                        PricingMethod::Analytical)
            .present_value();
    CHECK(dead_ko == 0.0);

    const double live_ko =
        OptionValuation(underlying,
                        barrier_spec(base_params(110.0, BarrierDirection::Up, BarrierAction::Out)),
                        PricingMethod::Analytical)
            .present_value();
    CHECK_THAT(live_ko, close_to(df * (forward - 100.0), 1e-12, 1e-14));

    const double hit_ki =
        OptionValuation(underlying,
                        barrier_spec(base_params(104.0, BarrierDirection::Up, BarrierAction::In)),
                        PricingMethod::Analytical)
            .present_value();
    CHECK_THAT(hit_ki, close_to(df * (forward - 100.0), 1e-12, 1e-14));

    const double missed_ki =
        OptionValuation(underlying,
                        barrier_spec(base_params(110.0, BarrierDirection::Up, BarrierAction::In)),
                        PricingMethod::Analytical)
            .present_value();
    CHECK(missed_ki == 0.0);
}

// ---------------------------------------------------------------------------
// Inception-triggered barriers
// ---------------------------------------------------------------------------

TEST_CASE("barrier triggered at inception", "[barrier]") {
    // Spot 100 is past an up barrier at 99 under continuous monitoring.
    auto ko = base_params(99.0, BarrierDirection::Up, BarrierAction::Out);

    SECTION("knock-out dies to its rebate") {
        CHECK(OptionValuation(ud(), barrier_spec(ko), PricingMethod::Analytical).present_value() ==
              0.0);

        ko.rebate = 3.0;
        ko.rebate_timing = RebateTiming::AtHit;
        CHECK(OptionValuation(ud(), barrier_spec(ko), PricingMethod::Analytical).present_value() ==
              3.0);

        ko.rebate_timing = RebateTiming::AtExpiry;
        const double maturity = year_fraction(kPricingDate, kMaturity, DayCountConvention::Act365F);
        const double df = DiscountCurve::flat(0.05).df(maturity);
        CHECK_THAT(
            OptionValuation(ud(), barrier_spec(ko), PricingMethod::Analytical).present_value(),
            close_to(3.0 * df, 1e-12, 1e-14));
    }

    SECTION("knock-in collapses to the vanilla, greeks included") {
        const auto ki = barrier_spec(base_params(101.0, BarrierDirection::Down, BarrierAction::In));
        const OptionValuation triggered(ud(), ki, PricingMethod::Analytical);
        const OptionValuation vanilla(
            ud(), vanilla_spec(OptionType::Call), PricingMethod::Analytical);

        CHECK_THAT(triggered.present_value(), close_to(vanilla.present_value(), 1e-12, 1e-14));
        CHECK_THAT(triggered.delta(), close_to(vanilla.delta(kNumerical), 1e-12, 1e-14));
        CHECK_THAT(triggered.gamma(), close_to(vanilla.gamma(kNumerical), 1e-12, 1e-14));
        CHECK_THAT(triggered.vega(), close_to(vanilla.vega(kNumerical), 1e-12, 1e-14));
        CHECK_THAT(triggered.theta(), close_to(vanilla.theta(kNumerical), 1e-12, 1e-14));
        CHECK_THAT(triggered.rho(), close_to(vanilla.rho(kNumerical), 1e-12, 1e-14));
    }

    SECTION("knocked-out rebate sensitivities") {
        ko.rebate = 3.0;
        ko.rebate_timing = RebateTiming::AtExpiry;
        const OptionValuation triggered(ud(), barrier_spec(ko), PricingMethod::Analytical);
        CHECK(triggered.delta() == 0.0);
        CHECK(triggered.gamma() == 0.0);
        CHECK(triggered.vega() == 0.0);
        // theta = r * R * df / 365: the discounted rebate unwinds at r.
        const double maturity = year_fraction(kPricingDate, kMaturity, DayCountConvention::Act365F);
        const double df = DiscountCurve::flat(0.05).df(maturity);
        CHECK_THAT(triggered.theta(), close_to(0.05 * 3.0 * df / 365.0, 1e-12, 1e-14));
        CHECK(triggered.rho() < 0.0);  // higher rates discount the rebate harder
    }
}

// ---------------------------------------------------------------------------
// Theta identity
// ---------------------------------------------------------------------------

TEST_CASE("auto theta uses the PDE identity and tracks numerical theta", "[barrier][greeks]") {
    const OptionValuation ov(
        ud(),
        barrier_spec(base_params(120.0, BarrierDirection::Up, BarrierAction::Out)),
        PricingMethod::Analytical);
    // Note: up-and-out call theta is legitimately positive near the barrier
    // (the knock-out risk decays as time passes), so only agreement between
    // the identity and the bump is asserted, not a sign.
    const double identity = ov.theta();
    const double numerical = ov.theta(kNumerical);
    CHECK_THAT(numerical, WithinRel(identity, 0.05));
}

// ---------------------------------------------------------------------------
// Unsupported combinations
// ---------------------------------------------------------------------------

TEST_CASE("analytical barrier rejections", "[barrier]") {
    SECTION("discrete dividends") {
        const UnderlyingData with_divs(
            100.0, 0.25, md(), {{kPricingDate + std::chrono::days{90}, 1.0}});
        CHECK_THROWS_MATCHES(OptionValuation(with_divs,
                                             barrier_spec(base_params(
                                                 120.0, BarrierDirection::Up, BarrierAction::Out)),
                                             PricingMethod::Analytical),
                             UnsupportedFeatureError,
                             message_contains("does not support discrete dividends"));
    }

    SECTION("explicit monitoring dates need a numerical engine") {
        auto params = base_params(120.0, BarrierDirection::Up, BarrierAction::Out);
        params.monitoring = BarrierMonitoring::Discrete;
        params.monitoring_dates = std::vector<DateTime>{kMaturity};
        const OptionValuation ov(ud(), barrier_spec(params), PricingMethod::Analytical);
        CHECK_THROWS_MATCHES(ov.present_value(),
                             UnsupportedFeatureError,
                             message_contains("requires num_observations"));
    }

    SECTION("explicit ANALYTICAL greek method") {
        const OptionValuation ov(
            ud(),
            barrier_spec(base_params(120.0, BarrierDirection::Up, BarrierAction::Out)),
            PricingMethod::Analytical);
        CHECK_THROWS_MATCHES(ov.delta({.method = GreekCalculationMethod::Analytical}),
                             UnsupportedFeatureError,
                             message_contains("do not support GreekCalculationMethod.ANALYTICAL"));
    }

    SECTION("numerical theta on a discretely monitored barrier") {
        auto params = base_params(120.0, BarrierDirection::Up, BarrierAction::Out);
        params.monitoring = BarrierMonitoring::Discrete;
        params.num_observations = 12;
        const OptionValuation ov(ud(), barrier_spec(params), PricingMethod::Analytical);
        CHECK_THROWS_MATCHES(ov.theta(kNumerical),
                             UnsupportedFeatureError,
                             message_contains("Numerical theta is not supported"));
    }

    SECTION("discrete barrier with the spot already past it") {
        // Regression: the pricing date is not an observation, so the option
        // is live, but the closed form is only valid with the spot on the
        // live side. It returned 1.29 against ~5.3 from the PDE and tree.
        auto params = base_params(105.0, BarrierDirection::Down, BarrierAction::Out);
        params.monitoring = BarrierMonitoring::Discrete;
        params.num_observations = 4;
        const OptionValuation ov(ud(), barrier_spec(params), PricingMethod::Analytical);
        CHECK_THROWS_MATCHES(ov.present_value(),
                             UnsupportedFeatureError,
                             message_contains("requires the spot on the live side"));
        // The numerical engines price it.
        CHECK(OptionValuation(ud(), barrier_spec(params), PricingMethod::PdeFd).present_value() >
              0.0);
    }

    SECTION("American exercise") {
        auto params = base_params(120.0, BarrierDirection::Up, BarrierAction::Out);
        params.exercise_type = ExerciseType::American;
        CHECK_THROWS_MATCHES(OptionValuation(ud(), barrier_spec(params), PricingMethod::Analytical),
                             UnsupportedFeatureError,
                             message_contains("ANALYTICAL is only applicable to European"));
    }
}

// ---------------------------------------------------------------------------
// Golden values from the Python reference implementation
// ---------------------------------------------------------------------------

namespace {

struct BarrierGoldenRow {
    const char* label;
    double spot;
    double strike;
    double barrier;
    double rate;
    double div_yield;
    double vol;
    int maturity_days;
    int num_observations;
    double rebate;
    bool rebate_at_hit;
    bool is_up;
    bool is_in;
    bool is_call;
    double pv;
    double theta_auto;
    double delta_num;
    double gamma_num;
    double vega_num;
    double rho_num;
};

#include "golden/barrier_reference.inc"

OptionValuation replay(const BarrierGoldenRow& row) {
    const DateTime maturity = kPricingDate + std::chrono::days{row.maturity_days};
    std::optional<DiscountCurve> q_curve;
    if (row.div_yield != 0.0) {
        q_curve = DiscountCurve::flat(row.div_yield);
    }
    const UnderlyingData underlying(row.spot, row.vol, md(row.rate), {}, std::move(q_curve));
    const BarrierSpec spec(
        {.option_type = row.is_call ? OptionType::Call : OptionType::Put,
         .exercise_type = ExerciseType::European,
         .strike = row.strike,
         .maturity = maturity,
         .barrier = row.barrier,
         .direction = row.is_up ? BarrierDirection::Up : BarrierDirection::Down,
         .action = row.is_in ? BarrierAction::In : BarrierAction::Out,
         .monitoring =
             row.num_observations > 0 ? BarrierMonitoring::Discrete : BarrierMonitoring::Continuous,
         .rebate = row.rebate,
         .rebate_timing = row.rebate_at_hit ? RebateTiming::AtHit : RebateTiming::AtExpiry,
         .currency = "USD",
         .num_observations =
             row.num_observations > 0 ? std::optional<int>(row.num_observations) : std::nullopt});
    return {underlying, spec, PricingMethod::Analytical};
}

}  // namespace

TEST_CASE("analytical barriers match the Python reference implementation", "[barrier][golden]") {
    for (const auto& row : std::span(kBarrierGolden)) {
        INFO("row " << row.label);
        const auto ov = replay(row);
        CHECK_THAT(ov.present_value(), close_to(row.pv, 1e-11, 1e-13));
        CHECK_THAT(ov.theta(), close_to(row.theta_auto, 1e-9, 1e-11));
        CHECK_THAT(ov.delta(kNumerical), close_to(row.delta_num, 1e-8, 1e-10));
        CHECK_THAT(ov.gamma(kNumerical), close_to(row.gamma_num, 1e-8, 1e-10));
        CHECK_THAT(ov.vega(kNumerical), close_to(row.vega_num, 1e-8, 1e-10));
        CHECK_THAT(ov.rho(kNumerical), close_to(row.rho_num, 1e-8, 1e-10));
    }
}
