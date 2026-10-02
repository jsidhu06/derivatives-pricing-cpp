#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/exceptions.hpp"
#include "dp/forward.hpp"
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

using namespace dp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using dp::test::kMaturity;
using dp::test::kPricingDate;
using dp::test::message_contains;

namespace {

MarketData md(double rate = 0.05) {
    return {kPricingDate, DiscountCurve::flat(rate), "USD"};
}

GbmProcess gbm(int paths = 100'000,
               double spot = 100.0,
               double vol = 0.20,
               GbmParams extra = {.initial_value = 0, .volatility = 0}) {
    GbmParams params = std::move(extra);
    params.initial_value = spot;
    params.volatility = vol;
    return {md(), std::move(params), {.paths = paths, .end_date = kMaturity, .num_steps = 26}};
}

VanillaSpec spec(OptionType type,
                 ExerciseType exercise = ExerciseType::European,
                 double strike = 100.0) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = exercise,
                        .strike = strike,
                        .maturity = kMaturity,
                        .currency = "USD"});
}

constexpr GreekOptions kPathwise{.method = GreekCalculationMethod::Pathwise};
constexpr GreekOptions kLikelihoodRatio{.method = GreekCalculationMethod::LikelihoodRatio};

}  // namespace

// ---------------------------------------------------------------------------
// Simulation layer
// ---------------------------------------------------------------------------

TEST_CASE("SimulationConfig validation", "[mc][simulation]") {
    CHECK_THROWS_MATCHES(SimulationConfig{.paths = 0}.validate(),
                         ValidationError,
                         message_contains("paths must be a positive integer"));
    CHECK_THROWS_MATCHES(SimulationConfig{.paths = 10}.validate(),
                         ValidationError,
                         message_contains("end_date must be provided"));
    CHECK_THROWS_MATCHES((SimulationConfig{.paths = 10, .end_date = kMaturity}.validate()),
                         ValidationError,
                         message_contains("num_steps must be provided"));
    CHECK_THROWS_MATCHES((SimulationConfig{.paths = 10,
                                           .end_date = kMaturity,
                                           .num_steps = 12,
                                           .time_grid = std::vector{kPricingDate, kMaturity}}
                              .validate()),
                         ValidationError,
                         message_contains("end_date and num_steps must be omitted"));
    CHECK_NOTHROW((SimulationConfig{.paths = 10, .time_grid = std::vector{kPricingDate, kMaturity}}
                       .validate()));
}

TEST_CASE("simulation grid starts at pricing date and snaps ex-div dates", "[mc][simulation]") {
    const auto ex_date = kPricingDate + std::chrono::days{100};
    const GbmProcess process(
        md(),
        {.initial_value = 100.0, .volatility = 0.2, .discrete_dividends = {{ex_date, 1.0}}},
        {.paths = 10, .end_date = kMaturity, .num_steps = 12});
    const auto dates = process.build_dates();
    CHECK(dates.front() == kPricingDate);
    CHECK(dates.back() == kMaturity);
    CHECK(std::find(dates.begin(), dates.end(), ex_date) != dates.end());
    CHECK(dates.size() >= 14);
}

TEST_CASE("simulation is reproducible per seed", "[mc][simulation]") {
    const auto process = gbm(1000);
    const auto a = process.simulate(7);
    const auto b = process.simulate(7);
    const auto c = process.simulate(8);
    CHECK(a.paths == b.paths);
    CHECK(a.paths != c.paths);
    // Antithetic layout: the second half mirrors the first half's shocks.
    CHECK(a.normals(0, 0) == -a.normals(500, 0));
}

TEST_CASE("terminal spots are risk-neutral on average", "[mc][simulation]") {
    const auto process = gbm(200'000);
    const auto result = process.simulate(42);
    const double mean_terminal = result.paths.col(result.paths.cols() - 1).mean();
    const auto& curve = process.discount_curve();
    const double fwd = forward_price({.spot = 100.0,
                                      .pricing_date = kPricingDate,
                                      .maturity = kMaturity,
                                      .discount_curve = &curve});
    CHECK_THAT(mean_terminal, WithinRel(fwd, 5e-3));
}

TEST_CASE("odd path count downgrades antithetic with a warning", "[mc][simulation]") {
    const dp::test::WarningCapture capture;
    const GbmProcess process(md(),
                             {.initial_value = 100.0, .volatility = 0.2},
                             {.paths = 101, .end_date = kMaturity, .num_steps = 4});
    (void)process.simulate(1);
    CHECK(capture.any_contains("antithetic"));
}

// ---------------------------------------------------------------------------
// European pricing and Greeks vs BSM (statistical agreement)
// ---------------------------------------------------------------------------

TEST_CASE("MC European agrees with BSM", "[mc]") {
    const auto type = GENERATE(OptionType::Call, OptionType::Put);
    const UnderlyingData ud(100.0, 0.20, md());
    const OptionValuation bsm(ud, spec(type), PricingMethod::Bsm);
    const OptionValuation mc(gbm(200'000), spec(type), PricingMethod::MonteCarlo);
    CHECK_THAT(mc.present_value(), WithinRel(bsm.present_value(), 1e-2));
}

TEST_CASE("pathwise and LR Greeks agree with BSM analytical", "[mc][greeks]") {
    const auto type = GENERATE(OptionType::Call, OptionType::Put);
    const UnderlyingData ud(100.0, 0.20, md());
    const OptionValuation bsm(ud, spec(type), PricingMethod::Bsm);
    const OptionValuation mc(gbm(200'000), spec(type), PricingMethod::MonteCarlo);

    CHECK_THAT(mc.delta(kPathwise), WithinRel(bsm.delta(), 2e-2));
    CHECK_THAT(mc.vega(kPathwise), WithinRel(bsm.vega(), 3e-2));
    CHECK_THAT(mc.theta(kPathwise), WithinRel(bsm.theta(), 3e-2));
    CHECK_THAT(mc.rho(kPathwise), WithinRel(bsm.rho(), 2e-2));
    CHECK_THAT(mc.gamma(kPathwise), WithinRel(bsm.gamma(), 5e-2));

    CHECK_THAT(mc.delta(kLikelihoodRatio), WithinRel(bsm.delta(), 3e-2));
    CHECK_THAT(mc.vega(kLikelihoodRatio), WithinRel(bsm.vega(), 5e-2));
    CHECK_THAT(mc.theta(kLikelihoodRatio), WithinRel(bsm.theta(), 5e-2));
    CHECK_THAT(mc.rho(kLikelihoodRatio), WithinRel(bsm.rho(), 3e-2));
}

TEST_CASE("pathwise is the auto-selected Greek method for eligible MC", "[mc][greeks]") {
    const OptionValuation mc(gbm(20'000), spec(OptionType::Call), PricingMethod::MonteCarlo);
    CHECK(mc.delta() == mc.delta(kPathwise));
    CHECK(mc.vega() == mc.vega(kPathwise));
}

TEST_CASE("numerical MC Greeks share common random numbers", "[mc][greeks]") {
    // The fixed default seed makes bump-and-revalue smooth; a numerical delta
    // should sit close to the pathwise one.
    const OptionValuation mc(gbm(100'000), spec(OptionType::Call), PricingMethod::MonteCarlo);
    const double numerical = mc.delta({.method = GreekCalculationMethod::Numerical});
    CHECK_THAT(numerical, WithinRel(mc.delta(kPathwise), 2e-2));
}

// ---------------------------------------------------------------------------
// American (Longstaff-Schwartz)
// ---------------------------------------------------------------------------

TEST_CASE("LSM American put sits between European and the tree price", "[mc][american]") {
    const auto process = gbm(200'000);
    const double european =
        OptionValuation(process, spec(OptionType::Put), PricingMethod::MonteCarlo).present_value();
    const double american =
        OptionValuation(
            process, spec(OptionType::Put, ExerciseType::American), PricingMethod::MonteCarlo)
            .present_value();
    const UnderlyingData ud(100.0, 0.20, md());
    const double tree = OptionValuation(ud,
                                        spec(OptionType::Put, ExerciseType::American),
                                        PricingMethod::Binomial,
                                        BinomialParams{.num_steps = 2000})
                            .present_value();
    CHECK(american > european);
    CHECK_THAT(american, WithinRel(tree, 1e-2));
}

TEST_CASE("MC control variate equals the explicit adjustment identity", "[mc][cv]") {
    const auto process = gbm(50'000, 100.0, 0.25);
    const auto amer = spec(OptionType::Put, ExerciseType::American, 110.0);
    const auto euro = spec(OptionType::Put, ExerciseType::European, 110.0);
    const MonteCarloParams base{};
    const MonteCarloParams cv{.control_variate_european = true};

    const double american_raw =
        OptionValuation(process, amer, PricingMethod::MonteCarlo, base).present_value();
    const double european_mc =
        OptionValuation(process, euro, PricingMethod::MonteCarlo, base).present_value();
    const UnderlyingData ud(100.0, 0.25, md());
    const double european_bsm = OptionValuation(ud, euro, PricingMethod::Bsm).present_value();
    const double american_cv =
        OptionValuation(process, amer, PricingMethod::MonteCarlo, cv).present_value();
    CHECK_THAT(american_cv, WithinAbs(american_raw + (european_bsm - european_mc), 1e-8));
}

// ---------------------------------------------------------------------------
// Facade pairing and Greek-method validation
// ---------------------------------------------------------------------------

TEST_CASE("pricing method and underlying type must pair up", "[mc][valuation]") {
    const UnderlyingData ud(100.0, 0.20, md());
    CHECK_THROWS_MATCHES(OptionValuation(ud, spec(OptionType::Call), PricingMethod::MonteCarlo),
                         ConfigurationError,
                         message_contains("requires underlying to be a path process"));
    CHECK_THROWS_MATCHES(OptionValuation(gbm(100), spec(OptionType::Call), PricingMethod::Bsm),
                         ConfigurationError,
                         message_contains("BSM pricing requires an UnderlyingData instance"));
    CHECK_THROWS_MATCHES(OptionValuation(gbm(100), spec(OptionType::Call), PricingMethod::Binomial),
                         ConfigurationError,
                         message_contains("BINOMIAL pricing requires an UnderlyingData instance"));
    CHECK_THROWS_MATCHES(OptionValuation(gbm(100), spec(OptionType::Call), PricingMethod::PdeFd),
                         ConfigurationError,
                         message_contains("PDE_FD pricing requires an UnderlyingData instance"));
}

TEST_CASE("pathwise/LR eligibility is enforced", "[mc][greeks]") {
    const OptionValuation american(
        gbm(1000), spec(OptionType::Put, ExerciseType::American), PricingMethod::MonteCarlo);
    CHECK_THROWS_MATCHES(
        american.delta(kPathwise), ValidationError, message_contains("vanilla European options"));

    const PayoffSpec payoff({.exercise_type = ExerciseType::European,
                             .maturity = kMaturity,
                             .payoff_fn = [](double s) { return std::max(s - 95.0, 0.0); },
                             .currency = "USD"});
    const OptionValuation custom(gbm(1000), payoff, PricingMethod::MonteCarlo);
    CHECK_THROWS_MATCHES(custom.delta(kLikelihoodRatio),
                         ValidationError,
                         message_contains("vanilla European options"));
    CHECK(std::isfinite(custom.delta({.method = GreekCalculationMethod::Numerical})));

    const OptionValuation vanilla(gbm(1000), spec(OptionType::Call), PricingMethod::MonteCarlo);
    CHECK_THROWS_MATCHES(vanilla.gamma(kLikelihoodRatio),
                         ValidationError,
                         message_contains("likelihood_ratio is not available for gamma"));

    const GbmProcess with_div(
        md(),
        {.initial_value = 100.0,
         .volatility = 0.2,
         .discrete_dividends = {{kPricingDate + std::chrono::days{100}, 1.0}}},
        {.paths = 1000, .end_date = kMaturity, .num_steps = 12});
    const OptionValuation div_ov(with_div, spec(OptionType::Call), PricingMethod::MonteCarlo);
    CHECK_THROWS_MATCHES(
        div_ov.delta(kPathwise), UnsupportedFeatureError, message_contains("discrete dividends"));
    CHECK_THROWS_MATCHES(div_ov.delta(kLikelihoodRatio),
                         UnsupportedFeatureError,
                         message_contains("discrete dividends"));
    // Auto-selection therefore falls back to NUMERICAL and still works.
    CHECK(std::isfinite(div_ov.delta()));
}

TEST_CASE("LR estimators reject zero volatility", "[mc][greeks][edge]") {
    const OptionValuation ov(
        gbm(1000, 100.0, 0.0), spec(OptionType::Call), PricingMethod::MonteCarlo);
    CHECK_THROWS_MATCHES(ov.delta(kLikelihoodRatio),
                         UnsupportedFeatureError,
                         message_contains("Likelihood-ratio delta is undefined at sigma=0"));
    CHECK_THROWS_MATCHES(ov.vega(kLikelihoodRatio),
                         UnsupportedFeatureError,
                         message_contains("Likelihood-ratio vega is undefined at sigma=0"));
    CHECK_THROWS_MATCHES(ov.theta(kLikelihoodRatio),
                         UnsupportedFeatureError,
                         message_contains("Likelihood-ratio theta is undefined at sigma=0"));
    CHECK_THROWS_MATCHES(ov.rho(kLikelihoodRatio),
                         UnsupportedFeatureError,
                         message_contains("Likelihood-ratio rho is undefined at sigma=0"));
}

TEST_CASE("discrete dividends reduce the MC call price", "[mc]") {
    const double base =
        OptionValuation(gbm(100'000), spec(OptionType::Call), PricingMethod::MonteCarlo)
            .present_value();
    const GbmProcess with_div(
        md(),
        {.initial_value = 100.0,
         .volatility = 0.2,
         .discrete_dividends = {{kPricingDate + std::chrono::days{180}, 2.0}}},
        {.paths = 100'000, .end_date = kMaturity, .num_steps = 26});
    const double with_div_pv =
        OptionValuation(with_div, spec(OptionType::Call), PricingMethod::MonteCarlo)
            .present_value();
    CHECK(with_div_pv < base);
}
