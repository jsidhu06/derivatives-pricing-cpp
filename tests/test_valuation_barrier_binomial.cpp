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
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace dp;
using dp::test::close_to;
using dp::test::kMaturity;
using dp::test::kPricingDate;
using dp::test::message_contains;
using dp::test::WarningCapture;

namespace {

constexpr int kBaseSteps = 500;

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

OptionValuation binomial(const UnderlyingData& underlying,
                         BarrierSpec::Params params,
                         int steps = kBaseSteps) {
    return {underlying,
            BarrierSpec(std::move(params)),
            PricingMethod::Binomial,
            BinomialParams{.num_steps = steps}};
}

constexpr GreekOptions kNumerical{.method = GreekCalculationMethod::Numerical};

}  // namespace

// ---------------------------------------------------------------------------
// Cross-engine agreement with the analytical closed forms
// ---------------------------------------------------------------------------

TEST_CASE("binomial barriers agree with the analytical engine", "[barrier][binomial]") {
    const auto type = GENERATE(OptionType::Call, OptionType::Put);
    const auto direction = GENERATE(BarrierDirection::Up, BarrierDirection::Down);
    const auto action = GENERATE(BarrierAction::In, BarrierAction::Out);
    const double barrier = direction == BarrierDirection::Up ? 115.0 : 85.0;

    auto params = base_params(barrier, direction, action, type);
    const double tree_pv = binomial(ud(), params).present_value();
    const double analytic_pv =
        OptionValuation(ud(), BarrierSpec(params), PricingMethod::Bsm).present_value();

    // Loose sanity sweep across all 8 types at 500 base steps — bit-exact
    // parity with the Python trees is asserted by the golden table below,
    // and paper-value accuracy by the Boyle-Tian test above.
    CHECK_THAT(tree_pv, close_to(analytic_pv, 1e-2, 2e-3));
}

TEST_CASE("discrete monitoring orders knock-out values correctly", "[barrier][binomial]") {
    auto continuous = base_params(115.0, BarrierDirection::Up, BarrierAction::Out);
    auto discrete = continuous;
    discrete.monitoring = BarrierMonitoring::Discrete;
    discrete.num_observations = 12;

    const double continuous_pv = binomial(ud(), continuous).present_value();
    const double discrete_pv = binomial(ud(), discrete).present_value();
    const VanillaSpec vanilla({.option_type = OptionType::Call,
                               .exercise_type = ExerciseType::European,
                               .strike = 100.0,
                               .maturity = kMaturity,
                               .currency = "USD"});
    const double vanilla_pv =
        OptionValuation(
            ud(), vanilla, PricingMethod::Binomial, BinomialParams{.num_steps = kBaseSteps})
            .present_value();

    // Fewer knock-out observations can only help the holder, and no barrier
    // at all helps most.
    CHECK(continuous_pv < discrete_pv);
    CHECK(discrete_pv < vanilla_pv);
}

TEST_CASE("American knock-out is worth at least the European", "[barrier][binomial]") {
    auto params = base_params(85.0, BarrierDirection::Down, BarrierAction::Out, OptionType::Put);
    const double european = binomial(ud(), params).present_value();
    params.exercise_type = ExerciseType::American;
    const double american = binomial(ud(), params).present_value();
    CHECK(american >= european - 1e-12);
}

TEST_CASE("explicit monitoring dates snap to tree steps", "[barrier][binomial]") {
    auto continuous = base_params(115.0, BarrierDirection::Up, BarrierAction::Out);
    auto explicit_dates = continuous;
    explicit_dates.monitoring = BarrierMonitoring::Discrete;
    std::vector<DateTime> dates;
    for (int month = 1; month <= 12; ++month) {
        dates.push_back(kPricingDate + std::chrono::days{30 * month});
    }
    explicit_dates.monitoring_dates = std::move(dates);

    const double sparse = binomial(ud(), explicit_dates).present_value();
    const double dense = binomial(ud(), continuous).present_value();
    // Fewer observations -> fewer knock-out chances -> higher KO value.
    CHECK(sparse > dense);
}

// ---------------------------------------------------------------------------
// Greek guards
// ---------------------------------------------------------------------------

TEST_CASE("binomial barrier greek guards", "[barrier][binomial][greeks]") {
    SECTION("numerical bump-and-revalue is blocked (rho exempt)") {
        const auto ov =
            binomial(ud(), base_params(115.0, BarrierDirection::Up, BarrierAction::Out));
        CHECK_THROWS_MATCHES(ov.delta(kNumerical),
                             UnsupportedFeatureError,
                             message_contains("Binomial NUMERICAL bump-and-revalue greeks"));
        CHECK_THROWS_MATCHES(ov.gamma(kNumerical),
                             UnsupportedFeatureError,
                             message_contains("Binomial NUMERICAL bump-and-revalue greeks"));
        CHECK_THROWS_MATCHES(ov.theta(kNumerical),
                             UnsupportedFeatureError,
                             message_contains("Binomial NUMERICAL bump-and-revalue greeks"));
        CHECK_THROWS_MATCHES(ov.vega(kNumerical),
                             UnsupportedFeatureError,
                             message_contains("Binomial NUMERICAL bump-and-revalue greeks"));
        // Vega has no TREE method, so auto-select lands on NUMERICAL and hits
        // the same guard.
        CHECK_THROWS_MATCHES(ov.vega(),
                             UnsupportedFeatureError,
                             message_contains("Binomial NUMERICAL bump-and-revalue greeks"));
        CHECK(std::isfinite(ov.rho()));
    }

    SECTION("tree delta and gamma refuse a straddling stencil") {
        // The barrier sits within one CRR step of the spot, so the step-1
        // up-node already lies past the absorbing boundary.
        const auto ov =
            binomial(ud(), base_params(100.5, BarrierDirection::Up, BarrierAction::Out));
        CHECK_THROWS_MATCHES(
            ov.delta(), UnsupportedFeatureError, message_contains("step-1 extraction stencil"));
        CHECK_THROWS_MATCHES(
            ov.gamma(), UnsupportedFeatureError, message_contains("step-2 extraction stencil"));
    }

    SECTION("a step-2 straddle refuses gamma but leaves PV, delta and theta") {
        // Spot 72.4 under an up barrier at 73: the step-2 up-up node lands on
        // the barrier under Boyle-Lau alignment. The American put is deep in
        // the money, so early exercise locks PV = K - S = 7.6, delta = -1 and
        // theta = 0 — only the step-2 gamma stencil is unusable.
        const MarketData market(kPricingDate, DiscountCurve::flat(0.05, 2.0), "USD");
        auto params = base_params(73.0,
                                  BarrierDirection::Up,
                                  BarrierAction::Out,
                                  OptionType::Put,
                                  ExerciseType::American);
        params.strike = 80.0;
        const OptionValuation ov(UnderlyingData(72.4, 0.205, market),
                                 BarrierSpec(std::move(params)),
                                 PricingMethod::Binomial);
        CHECK_THAT(ov.present_value(), close_to(7.6, 0.0, 1e-6));
        CHECK_THAT(ov.delta(), close_to(-1.0, 0.0, 1e-6));
        CHECK_THAT(ov.theta(), close_to(0.0, 0.0, 1e-6));
        CHECK_THROWS_MATCHES(
            ov.gamma(), UnsupportedFeatureError, message_contains("step-2 extraction stencil"));
    }

    SECTION("unattainable Boyle-Lau alignment warns and caps the steps") {
        const WarningCapture warnings;
        const auto ov =
            binomial(ud(), base_params(100.05, BarrierDirection::Up, BarrierAction::Out));
        CHECK(ov.present_value() >= 0.0);
        REQUIRE_FALSE(warnings.empty());
        CHECK(warnings.messages().front().find("Boyle-Lau step alignment") != std::string::npos);
    }
}

TEST_CASE("binomial barriers reject discrete dividends", "[barrier][binomial]") {
    // The escrowed-dividend tree does not preserve barrier-hit dynamics at
    // ex-dates; vanillas with the same dividends stay allowed.
    const UnderlyingData with_dividend(
        100.0, 0.25, md(), {{kPricingDate + std::chrono::days{90}, 2.0}});
    CHECK_THROWS_MATCHES(
        binomial(with_dividend, base_params(115.0, BarrierDirection::Up, BarrierAction::Out)),
        UnsupportedFeatureError,
        message_contains("Binomial pricing of barrier options with discrete dividends"));
}

// ---------------------------------------------------------------------------
// Inception-triggered barriers
// ---------------------------------------------------------------------------

TEST_CASE("binomial barrier triggered at inception", "[barrier][binomial]") {
    auto ko = base_params(99.0, BarrierDirection::Up, BarrierAction::Out);
    ko.rebate = 3.0;
    const auto ko_ov = binomial(ud(), ko);
    CHECK(ko_ov.present_value() == 3.0);
    CHECK(ko_ov.delta() == 0.0);
    CHECK(ko_ov.gamma() == 0.0);

    const auto ki = base_params(101.0, BarrierDirection::Down, BarrierAction::In);
    const auto ki_ov = binomial(ud(), ki);
    const VanillaSpec vanilla({.option_type = OptionType::Call,
                               .exercise_type = ExerciseType::European,
                               .strike = 100.0,
                               .maturity = kMaturity,
                               .currency = "USD"});
    const OptionValuation vanilla_ov(
        ud(), vanilla, PricingMethod::Binomial, BinomialParams{.num_steps = kBaseSteps});
    CHECK_THAT(ki_ov.present_value(), close_to(vanilla_ov.present_value(), 1e-12, 1e-14));
    CHECK_THAT(ki_ov.delta(), close_to(vanilla_ov.delta(), 1e-12, 1e-14));
}

// ---------------------------------------------------------------------------
// Golden values from the Python reference implementation
// ---------------------------------------------------------------------------

namespace {

struct BinomialBarrierGoldenRow {
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
    double pv;
    double delta_tree;
    double gamma_tree;
    double theta_tree;
    double rho_num;
};

#include "golden/binomial_barrier_reference.inc"

OptionValuation replay(const BinomialBarrierGoldenRow& row) {
    const DateTime maturity = kPricingDate + std::chrono::days{row.maturity_days};
    std::optional<DiscountCurve> q_curve;
    if (row.div_yield != 0.0) {
        q_curve = DiscountCurve::flat(row.div_yield);
    }
    const UnderlyingData underlying(row.spot, row.vol, md(row.rate), {}, std::move(q_curve));
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
    return {underlying, spec, PricingMethod::Binomial, BinomialParams{.num_steps = kBaseSteps}};
}

}  // namespace

TEST_CASE("binomial barriers match the Python reference implementation",
          "[barrier][binomial][golden]") {
    for (const auto& row : std::span(kBinomialBarrierGolden)) {
        INFO("row " << row.label);
        const auto ov = replay(row);
        CHECK_THAT(ov.present_value(), close_to(row.pv, 1e-10, 1e-12));
        CHECK_THAT(ov.delta(), close_to(row.delta_tree, 1e-10, 1e-12));
        CHECK_THAT(ov.gamma(), close_to(row.gamma_tree, 1e-10, 1e-12));
        CHECK_THAT(ov.theta(), close_to(row.theta_tree, 1e-10, 1e-12));
        CHECK_THAT(ov.rho(kNumerical), close_to(row.rho_num, 1e-8, 1e-10));
    }
}
