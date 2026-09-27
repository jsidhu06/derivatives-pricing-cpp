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

#include <array>
#include <chrono>
#include <cmath>
#include <optional>
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

OptionValuation pde(const UnderlyingData& underlying, BarrierSpec::Params params) {
    return {underlying, BarrierSpec(std::move(params)), PricingMethod::PdeFd};
}

}  // namespace

// ---------------------------------------------------------------------------
// Paper anchors and cross-engine agreement
// ---------------------------------------------------------------------------

TEST_CASE("PDE barriers reproduce the Boyle-Tian (1998) Table 8 values", "[barrier][pde]") {
    // Down-and-out call: S=100, K=100, H=95, sigma=0.20, r=10%, T=0.5y.
    // Paper PVs and per-frequency tolerances mirror the Python suite
    // (PDE rtol 0.002, except the paper's likely-biased "hourly" row).
    const DateTime maturity =
        kPricingDate + std::chrono::hours{24 * 182} + std::chrono::hours{12};  // 182.5 days
    const MarketData market(kPricingDate, DiscountCurve::flat(0.10, 2.0), "USD");
    const UnderlyingData underlying(100.0, 0.20, market);
    const auto make_spec = [&](std::optional<int> observations) {
        return BarrierSpec({.option_type = OptionType::Call,
                            .exercise_type = ExerciseType::European,
                            .strike = 100.0,
                            .maturity = maturity,
                            .barrier = 95.0,
                            .direction = BarrierDirection::Down,
                            .action = BarrierAction::Out,
                            .monitoring = observations.has_value() ? BarrierMonitoring::Discrete
                                                                   : BarrierMonitoring::Continuous,
                            .currency = "USD",
                            .num_observations = observations});
    };

    struct PaperCase {
        std::optional<int> observations;
        double paper_pv = 0.0;
        double rtol = 0.0;
    };
    const std::array<PaperCase, 5> cases{{{std::nullopt, 5.7163, 0.002},
                                          {500, 5.8881, 0.020},  // paper's biased "hourly" row
                                          {125, 6.1669, 0.002},
                                          {26, 6.6176, 0.002},
                                          {6, 7.3100, 0.002}}};
    for (const auto& c : cases) {
        INFO("observations " << (c.observations.has_value() ? *c.observations : -1));
        const OptionValuation ov(underlying, make_spec(c.observations), PricingMethod::PdeFd);
        CHECK_THAT(ov.present_value(), close_to(c.paper_pv, c.rtol, 1e-4));
    }
}

TEST_CASE("PDE barriers agree with the analytical engine", "[barrier][pde]") {
    const auto type = GENERATE(OptionType::Call, OptionType::Put);
    const auto direction = GENERATE(BarrierDirection::Up, BarrierDirection::Down);
    const auto action = GENERATE(BarrierAction::In, BarrierAction::Out);
    const double barrier = direction == BarrierDirection::Up ? 115.0 : 85.0;

    auto params = base_params(barrier, direction, action, type);
    const double pde_pv = pde(ud(), params).present_value();
    const double analytic_pv =
        OptionValuation(ud(), BarrierSpec(params), PricingMethod::Bsm).present_value();

    // Continuous barriers on a truncated grid: Python's paper-table PDE
    // tolerance (rtol 2e-4 .. 4e-3 depending on the case) motivates 5e-3.
    CHECK_THAT(pde_pv, close_to(analytic_pv, 5e-3, 1e-3));
}

TEST_CASE("European KI parity components are consistent", "[barrier][pde]") {
    // V_KI + V_KO = V_vanilla on the PDE engines (no rebate), by parity.
    const auto direction = GENERATE(BarrierDirection::Up, BarrierDirection::Down);
    const double barrier = direction == BarrierDirection::Up ? 115.0 : 85.0;

    const double knock_in =
        pde(ud(), base_params(barrier, direction, BarrierAction::In)).present_value();
    const double knock_out =
        pde(ud(), base_params(barrier, direction, BarrierAction::Out)).present_value();
    const VanillaSpec vanilla({.option_type = OptionType::Call,
                               .exercise_type = ExerciseType::European,
                               .strike = 100.0,
                               .maturity = kMaturity,
                               .currency = "USD"});
    const double vanilla_pv =
        OptionValuation(ud(),
                        vanilla,
                        PricingMethod::PdeFd,
                        PdeParams::for_barriers(BarrierMonitoring::Continuous))
            .present_value();
    CHECK_THAT(knock_in + knock_out, close_to(vanilla_pv, 5e-3, 1e-3));
}

TEST_CASE("American PDE knock-ins dominate the European and stay below the vanilla",
          "[barrier][pde]") {
    auto params = base_params(85.0, BarrierDirection::Down, BarrierAction::In, OptionType::Put);
    const double european = pde(ud(), params).present_value();
    params.exercise_type = ExerciseType::American;
    const double american = pde(ud(), params).present_value();

    const VanillaSpec vanilla({.option_type = OptionType::Put,
                               .exercise_type = ExerciseType::American,
                               .strike = 100.0,
                               .maturity = kMaturity,
                               .currency = "USD"});
    const double vanilla_american =
        OptionValuation(ud(), vanilla, PricingMethod::PdeFd).present_value();

    CHECK(american >= european - 1e-10);
    CHECK(american <= vanilla_american + 1e-10);
}

TEST_CASE("discrete monitoring orders PDE knock-out values correctly", "[barrier][pde]") {
    auto continuous = base_params(115.0, BarrierDirection::Up, BarrierAction::Out);
    auto discrete = continuous;
    discrete.monitoring = BarrierMonitoring::Discrete;
    discrete.num_observations = 12;

    const double continuous_pv = pde(ud(), continuous).present_value();
    const double discrete_pv = pde(ud(), discrete).present_value();
    CHECK(continuous_pv < discrete_pv);
}

// ---------------------------------------------------------------------------
// Inception-triggered barriers and greek policy
// ---------------------------------------------------------------------------

TEST_CASE("PDE barrier triggered at inception", "[barrier][pde]") {
    auto ko = base_params(99.0, BarrierDirection::Up, BarrierAction::Out);
    ko.rebate = 3.0;
    ko.rebate_timing = RebateTiming::AtHit;
    const auto ko_ov = pde(ud(), ko);
    CHECK(ko_ov.present_value() == 3.0);
    CHECK(ko_ov.delta() == 0.0);
    CHECK(ko_ov.gamma() == 0.0);

    const auto ki_ov = pde(ud(), base_params(101.0, BarrierDirection::Down, BarrierAction::In));
    const VanillaSpec vanilla({.option_type = OptionType::Call,
                               .exercise_type = ExerciseType::European,
                               .strike = 100.0,
                               .maturity = kMaturity,
                               .currency = "USD"});
    const OptionValuation vanilla_ov(ud(),
                                     vanilla,
                                     PricingMethod::PdeFd,
                                     PdeParams::for_barriers(BarrierMonitoring::Continuous));
    CHECK_THAT(ki_ov.present_value(), close_to(vanilla_ov.present_value(), 1e-12, 1e-14));
    CHECK_THAT(ki_ov.delta(), close_to(vanilla_ov.delta(), 1e-12, 1e-14));
    CHECK_THAT(ki_ov.theta(), close_to(vanilla_ov.theta(), 1e-12, 1e-14));
}

TEST_CASE("PDE barrier greek policy", "[barrier][pde][greeks]") {
    const auto ov = pde(ud(), base_params(115.0, BarrierDirection::Up, BarrierAction::Out));

    SECTION("numerical bump-and-revalue works on the PDE engine") {
        // Unlike the binomial engine, the PDE grid topology is bump-invariant
        // enough that the facade allows NUMERICAL greeks (vega has no GRID
        // method, so auto-select lands there directly).
        CHECK(std::isfinite(ov.vega()));
        CHECK(std::isfinite(ov.rho()));
        CHECK(std::isfinite(ov.delta({.method = GreekCalculationMethod::Numerical})));
    }

    SECTION("numerical theta on a discretely monitored barrier is blocked") {
        auto params = base_params(115.0, BarrierDirection::Up, BarrierAction::Out);
        params.monitoring = BarrierMonitoring::Discrete;
        params.num_observations = 12;
        const auto discrete_ov = pde(ud(), params);
        CHECK(std::isfinite(discrete_ov.theta()));  // GRID theta is fine
        CHECK_THROWS_MATCHES(discrete_ov.theta({.method = GreekCalculationMethod::Numerical}),
                             UnsupportedFeatureError,
                             message_contains("Numerical theta is not supported"));
    }
}

// ---------------------------------------------------------------------------
// Golden values from the Python reference implementation
// ---------------------------------------------------------------------------

namespace {

struct PdeBarrierGoldenRow {
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
    bool is_american;
    double div_amount;
    int div_day;
    double pv;
    double delta_grid;
    double gamma_grid;
    double theta_grid;
    double vega_num;
    double rho_num;
};

#include "golden/pde_barrier_reference.inc"

OptionValuation replay(const PdeBarrierGoldenRow& row) {
    const DateTime maturity = kPricingDate + std::chrono::days{row.maturity_days};
    std::optional<DiscountCurve> q_curve;
    if (row.div_yield != 0.0) {
        q_curve = DiscountCurve::flat(row.div_yield);
    }
    DividendSchedule dividends;
    if (row.div_amount != 0.0) {
        dividends.push_back({kPricingDate + std::chrono::days{row.div_day}, row.div_amount});
    }
    const UnderlyingData underlying(
        row.spot, row.vol, md(row.rate), std::move(dividends), std::move(q_curve));
    const BarrierSpec spec(
        {.option_type = row.is_call ? OptionType::Call : OptionType::Put,
         .exercise_type = row.is_american ? ExerciseType::American : ExerciseType::European,
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
    return {underlying, spec, PricingMethod::PdeFd};
}

constexpr GreekOptions kNumerical{.method = GreekCalculationMethod::Numerical};

}  // namespace

TEST_CASE("PDE barriers match the Python reference implementation", "[barrier][pde][golden]") {
    for (const auto& row : kPdeBarrierGolden) {
        INFO("row " << row.label);
        const auto ov = replay(row);
        // Same parity tolerances as the vanilla PDE golden table: the
        // backward solves accumulate ~1e-9 relative float noise between
        // Eigen and numpy operation orderings.
        CHECK_THAT(ov.present_value(), close_to(row.pv, 1e-9, 1e-11));
        CHECK_THAT(ov.delta(), close_to(row.delta_grid, 1e-8, 1e-10));
        CHECK_THAT(ov.gamma(), close_to(row.gamma_grid, 1e-7, 1e-9));
        CHECK_THAT(ov.theta(), close_to(row.theta_grid, 1e-7, 1e-9));
        CHECK_THAT(ov.vega(kNumerical), close_to(row.vega_num, 1e-7, 1e-9));
        CHECK_THAT(ov.rho(kNumerical), close_to(row.rho_num, 1e-7, 1e-9));
    }
}
