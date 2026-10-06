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
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>
#include <thread>
#include <vector>

using namespace dp;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinRel;
using dp::test::flat_curve;
using dp::test::kMaturity;
using dp::test::kPricingDate;
using dp::test::message_contains;

namespace {

MarketData md() {
    return {kPricingDate, flat_curve(kPricingDate, kMaturity, 0.05), "USD"};
}
UnderlyingData ud() {
    return {100.0, 0.2, md()};
}

VanillaSpec vanilla(OptionType type = OptionType::Call,
                    ExerciseType exercise = ExerciseType::European,
                    std::optional<std::string> currency = "USD",
                    DateTime maturity = kMaturity) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = exercise,
                        .strike = 100.0,
                        .maturity = maturity,
                        .currency = std::move(currency)});
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST_CASE("OptionValuation exposes its inputs", "[valuation]") {
    const OptionValuation ov(ud(), vanilla(), PricingMethod::Analytical);
    CHECK(ov.strike() == 100.0);
    CHECK(ov.option_type() == OptionType::Call);
    CHECK(ov.exercise_type() == ExerciseType::European);
    CHECK(ov.pricing_method() == PricingMethod::Analytical);
    CHECK(ov.maturity() == kMaturity);
    CHECK(ov.pricing_date() == kPricingDate);
    CHECK(ov.currency() == "USD");
    CHECK(ov.contract_size() == 100.0);
    CHECK_FALSE(ov.params().has_value());
    CHECK(std::get<UnderlyingData>(ov.underlying()).initial_value() == 100.0);
    CHECK(std::holds_alternative<VanillaSpec>(ov.spec()));
    CHECK(ov.maturity_year_fraction() == 1.0);
}

TEST_CASE("OptionValuation inherits the underlying currency when the spec has none",
          "[valuation]") {
    const OptionValuation ov(ud(),
                             vanilla(OptionType::Call, ExerciseType::European, std::nullopt),
                             PricingMethod::Analytical);
    CHECK(ov.currency() == "USD");
}

TEST_CASE("OptionValuation rejects maturity on or before the pricing date", "[valuation]") {
    CHECK_THROWS_MATCHES(
        OptionValuation(ud(),
                        vanilla(OptionType::Call, ExerciseType::European, "USD", kPricingDate),
                        PricingMethod::Analytical),
        ValidationError,
        message_contains("Option maturity must be after pricing_date"));
}

TEST_CASE("OptionValuation rejects cross-currency specs", "[valuation]") {
    CHECK_THROWS_MATCHES(OptionValuation(ud(),
                                         vanilla(OptionType::Call, ExerciseType::European, "EUR"),
                                         PricingMethod::Analytical),
                         UnsupportedFeatureError,
                         message_contains("Cross-currency valuation is not supported"));
}

TEST_CASE("BSM rejects American exercise at construction", "[valuation]") {
    CHECK_THROWS_MATCHES(
        OptionValuation(
            ud(), vanilla(OptionType::Call, ExerciseType::American), PricingMethod::Analytical),
        UnsupportedFeatureError,
        message_contains("ANALYTICAL is only applicable to European"));
}

TEST_CASE("ANALYTICAL rejects PayoffSpec", "[valuation]") {
    const PayoffSpec spec({.exercise_type = ExerciseType::European,
                           .maturity = kMaturity,
                           .payoff_fn = [](double s) { return s > 100.0 ? s - 100.0 : 0.0; },
                           .currency = "USD"});
    CHECK_THROWS_MATCHES(OptionValuation(ud(), spec, PricingMethod::Analytical),
                         UnsupportedFeatureError,
                         message_contains("ANALYTICAL pricing requires a CALL or PUT"));
}

TEST_CASE("Monte Carlo requires a path simulation", "[valuation]") {
    CHECK_THROWS_MATCHES(OptionValuation(ud(), vanilla(), PricingMethod::MonteCarlo),
                         ConfigurationError,
                         message_contains("GbmProcess"));
}

TEST_CASE("params must match the pricing method", "[valuation]") {
    CHECK_THROWS_MATCHES(
        OptionValuation(ud(), vanilla(), PricingMethod::Analytical, BinomialParams{}),
        ConfigurationError,
        message_contains("pricing_method=ANALYTICAL does not accept valuation params"));
    CHECK_THROWS_MATCHES(
        OptionValuation(ud(), vanilla(), PricingMethod::Binomial, MonteCarloParams{}),
        ConfigurationError,
        message_contains("pricing_method=BINOMIAL requires params=BinomialParams"));
    CHECK_THROWS_MATCHES(OptionValuation(ud(), vanilla(), PricingMethod::PdeFd, BinomialParams{}),
                         ConfigurationError,
                         message_contains("pricing_method=PDE_FD requires params=PdeParams"));
    // Invalid params are rejected before the engine is built.
    CHECK_THROWS_MATCHES(
        OptionValuation(ud(), vanilla(), PricingMethod::Binomial, BinomialParams{.num_steps = 0}),
        ValidationError,
        message_contains("num_steps must be >= 1"));
}

TEST_CASE("describe() summarises the valuation", "[valuation]") {
    const OptionValuation ov(ud(), vanilla(), PricingMethod::Analytical);
    const auto text = ov.describe();
    CHECK_THAT(text, ContainsSubstring("OptionValuation(ANALYTICAL, EUROPEAN, CALL"));
    CHECK_THAT(text, ContainsSubstring("S=100"));
    CHECK_THAT(text, ContainsSubstring("K=100"));
    CHECK_THAT(text, ContainsSubstring("T=2026-01-01"));
    CHECK_THAT(text, ContainsSubstring("pricing_date=2025-01-01"));
    CHECK_THAT(text, ContainsSubstring("VanillaSpec)"));
}

TEST_CASE("OptionValuation is movable", "[valuation]") {
    OptionValuation a(ud(), vanilla(), PricingMethod::Analytical);
    const double pv = a.present_value();
    OptionValuation b(std::move(a));
    CHECK(b.present_value() == pv);
    CHECK(b.delta() > 0.0);
    std::vector<OptionValuation> pool;
    pool.push_back(std::move(b));
    CHECK(pool.front().present_value() == pv);
}

// ---------------------------------------------------------------------------
// Greek method resolution / bump validation
// ---------------------------------------------------------------------------

TEST_CASE("explicit Greek methods are validated against the engine", "[valuation][greeks]") {
    const OptionValuation ov(ud(), vanilla(), PricingMethod::Analytical);
    CHECK_THROWS_MATCHES(ov.delta({.method = GreekCalculationMethod::Tree}),
                         ValidationError,
                         message_contains("Tree greeks are only available for BINOMIAL"));
    CHECK_THROWS_MATCHES(ov.delta({.method = GreekCalculationMethod::Grid}),
                         ValidationError,
                         message_contains("Grid greeks are only available for PDE_FD"));
    CHECK_THROWS_MATCHES(ov.delta({.method = GreekCalculationMethod::Pathwise}),
                         ValidationError,
                         message_contains("pathwise greeks are only available for MONTE_CARLO"));
    CHECK_THROWS_MATCHES(ov.gamma({.method = GreekCalculationMethod::LikelihoodRatio}),
                         ValidationError,
                         message_contains("likelihood_ratio greeks are only available"));
    CHECK_NOTHROW(ov.delta({.method = GreekCalculationMethod::Analytical}));
    CHECK_NOTHROW(ov.delta({.method = GreekCalculationMethod::Numerical}));

    // ANALYTICAL is BSM-only.
    for (const auto method : {PricingMethod::Binomial, PricingMethod::PdeFd}) {
        INFO(to_string(method));
        const OptionValuation numerical_engine(ud(), vanilla(), method);
        CHECK_THROWS_MATCHES(
            numerical_engine.delta({.method = GreekCalculationMethod::Analytical}),
            ValidationError,
            message_contains("Analytical greeks are only available for ANALYTICAL"));
    }
}

TEST_CASE("a bump is rejected for non-bumping methods and must be positive",
          "[valuation][greeks]") {
    const OptionValuation ov(ud(), vanilla(), PricingMethod::Analytical);
    CHECK_THROWS_MATCHES(ov.delta({.bump = 0.5}),
                         ValidationError,
                         message_contains("does not use a finite-difference bump"));
    CHECK_THROWS_MATCHES(ov.vega({.method = GreekCalculationMethod::Analytical, .bump = 0.01}),
                         ValidationError,
                         message_contains("epsilon=0.01 was passed"));
    CHECK_THROWS_MATCHES(ov.theta({.method = GreekCalculationMethod::Numerical, .bump = -1.0}),
                         ValidationError,
                         message_contains("time_bump_days must be strictly positive"));
    CHECK_THROWS_MATCHES(ov.rho({.method = GreekCalculationMethod::Numerical, .bump = 0.0}),
                         ValidationError,
                         message_contains("rate_bump must be strictly positive"));
    CHECK_THROWS_AS(ov.delta({.method = GreekCalculationMethod::Numerical, .bump = dp::test::kNaN}),
                    ValidationError);
    CHECK_NOTHROW(ov.delta({.method = GreekCalculationMethod::Numerical, .bump = 0.5}));
}

// ---------------------------------------------------------------------------
// Caching
// ---------------------------------------------------------------------------

TEST_CASE("repeated calls are cached and distinct bumps cache separately", "[valuation][cache]") {
    const OptionValuation ov(ud(), vanilla(), PricingMethod::Analytical);
    const double pv = ov.present_value();
    CHECK(ov.present_value() == pv);

    const GreekOptions a{.method = GreekCalculationMethod::Numerical, .bump = 0.5};
    const GreekOptions b{.method = GreekCalculationMethod::Numerical, .bump = 1.0};
    const double d1 = ov.delta(a);
    const double d2 = ov.delta(b);
    CHECK(d1 != d2);
    CHECK(ov.delta(a) == d1);
    CHECK(ov.delta(b) == d2);
    // Analytical (default) is a different cache entry from NUMERICAL.
    CHECK(ov.delta() != d1);
}

TEST_CASE("concurrent queries on one instance agree", "[valuation][cache][threads]") {
    const OptionValuation ov(ud(), vanilla(), PricingMethod::Analytical);
    const GreekOptions num{.method = GreekCalculationMethod::Numerical};

    std::vector<double> results(16);
    std::vector<std::thread> workers;
    workers.reserve(results.size());
    for (double& slot : results) {
        workers.emplace_back([&ov, &slot, num] {
            slot = ov.present_value() + ov.delta(num) + ov.gamma() + ov.vega(num) + ov.theta() +
                   ov.rho(num);
        });
    }
    for (auto& w : workers) {
        w.join();
    }
    for (const double r : results) {
        CHECK(r == results.front());
    }
}

// ---------------------------------------------------------------------------
// Purity
// ---------------------------------------------------------------------------

TEST_CASE("pricing and Greeks do not mutate the underlying", "[valuation][purity]") {
    const auto q_curve = flat_curve(kPricingDate, kMaturity, 0.02);
    const UnderlyingData underlying(100.0, 0.2, md(), {}, q_curve);
    const OptionValuation ov(underlying, vanilla(), PricingMethod::Analytical);

    const GreekOptions num{.method = GreekCalculationMethod::Numerical};
    (void)ov.present_value();
    (void)ov.delta(num);
    (void)ov.gamma(num);
    (void)ov.vega(num);
    (void)ov.theta(num);
    (void)ov.rho(num);

    CHECK(underlying.initial_value() == 100.0);
    CHECK(underlying.volatility() == 0.2);
    CHECK(underlying.pricing_date() == kPricingDate);
    CHECK_THAT(underlying.discount_curve().flat_rate().value_or(0.0), WithinRel(0.05, 1e-12));
    const auto q_rate = underlying.dividend_curve().value_or(DiscountCurve::flat(0.0)).flat_rate();
    CHECK_THAT(q_rate.value_or(0.0), WithinRel(0.02, 1e-12));
    // The valuation's own copy is likewise untouched.
    CHECK(std::get<UnderlyingData>(ov.underlying()).initial_value() == 100.0);
    CHECK(std::get<UnderlyingData>(ov.underlying()).volatility() == 0.2);
}
