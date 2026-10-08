#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/diagnostics.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/implied_volatility.hpp"
#include "dp/market_data.hpp"
#include "dp/params.hpp"
#include "dp/simulation.hpp"
#include "dp/underlying.hpp"
#include "dp/valuation.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <latch>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// Thread-safety: an OptionValuation is documented as safe to query from many
// threads. Its engines fill lazy caches (tree, PDE solution, simulated paths)
// on first use, so the risky moment is many threads hitting a cold instance
// at once. Each test releases its threads together with a latch and requires
// bit-identical results to a single-threaded run on a fresh instance. Run
// under ThreadSanitizer by the ci-tsan preset (label "threads").

using namespace dp;
using dp::test::kMaturity;
using dp::test::kPricingDate;

namespace {

constexpr int kThreads = 8;

using Queries = std::function<std::vector<double>(const OptionValuation&)>;

MarketData md() {
    return {kPricingDate, DiscountCurve::flat(0.05), "USD"};
}

UnderlyingData ud() {
    return {100.0, 0.20, md()};
}

GbmProcess gbm() {
    return {md(),
            {.initial_value = 100.0, .volatility = 0.20},
            {.paths = 20'000, .end_date = kMaturity, .num_steps = 12}};
}

VanillaSpec vanilla(OptionType type, ExerciseType exercise) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = exercise,
                        .strike = 100.0,
                        .maturity = kMaturity,
                        .currency = "USD"});
}

BarrierSpec barrier(BarrierAction action) {
    return BarrierSpec({.option_type = OptionType::Call,
                        .exercise_type = ExerciseType::European,
                        .strike = 100.0,
                        .maturity = kMaturity,
                        .barrier = 80.0,
                        .direction = BarrierDirection::Down,
                        .action = action,
                        .currency = "USD"});
}

constexpr GreekOptions kNumerical{.method = GreekCalculationMethod::Numerical};
constexpr GreekOptions kPathwise{.method = GreekCalculationMethod::Pathwise};
constexpr GreekOptions kLikelihoodRatio{.method = GreekCalculationMethod::LikelihoodRatio};

/// PV and every Greek through the auto-selected method, plus numerical delta.
std::vector<double> default_queries(const OptionValuation& ov) {
    return {ov.present_value(),
            ov.delta(),
            ov.gamma(),
            ov.vega(),
            ov.theta(),
            ov.rho(),
            ov.delta(kNumerical)};
}

/// Many threads query one cold instance; each must see exactly what a
/// single-threaded run on a fresh instance sees.
void check_shared_instance(const std::function<OptionValuation()>& make, const Queries& queries) {
    const std::vector<double> expected = queries(make());

    const OptionValuation shared = make();
    std::vector<std::vector<double>> results(kThreads);
    std::latch start(kThreads);
    {
        std::vector<std::jthread> workers;
        workers.reserve(kThreads);
        for (auto& slot : results) {
            workers.emplace_back([&] {
                start.arrive_and_wait();
                slot = queries(shared);
            });
        }
    }
    for (const auto& result : results) {
        CHECK(result == expected);
    }
}

}  // namespace

TEST_CASE("concurrent queries on a cold binomial instance agree", "[threads][binomial]") {
    check_shared_instance(
        [] {
            return OptionValuation(ud(),
                                   vanilla(OptionType::Put, ExerciseType::American),
                                   PricingMethod::Binomial,
                                   BinomialParams{.num_steps = 400});
        },
        default_queries);
}

TEST_CASE("concurrent queries on a cold PDE instance agree", "[threads][pde]") {
    check_shared_instance(
        [] {
            return OptionValuation(ud(),
                                   vanilla(OptionType::Put, ExerciseType::American),
                                   PricingMethod::PdeFd,
                                   PdeParams{.spot_steps = 100, .time_steps = 100});
        },
        default_queries);
}

TEST_CASE("concurrent queries on a cold Monte Carlo European instance agree", "[threads][mc]") {
    check_shared_instance(
        [] {
            return OptionValuation(gbm(),
                                   vanilla(OptionType::Call, ExerciseType::European),
                                   PricingMethod::MonteCarlo);
        },
        [](const OptionValuation& ov) {
            return std::vector<double>{ov.present_value(),
                                       ov.std_error().value_or(-1.0),
                                       ov.delta(kPathwise),
                                       ov.vega(kPathwise),
                                       ov.theta(kPathwise),
                                       ov.rho(kPathwise),
                                       ov.gamma(kPathwise),
                                       ov.delta(kLikelihoodRatio),
                                       ov.vega(kLikelihoodRatio),
                                       ov.delta(kNumerical)};
        });
}

TEST_CASE("concurrent queries on a cold Monte Carlo American instance agree", "[threads][mc]") {
    check_shared_instance(
        [] {
            return OptionValuation(
                gbm(), vanilla(OptionType::Put, ExerciseType::American), PricingMethod::MonteCarlo);
        },
        [](const OptionValuation& ov) {
            return std::vector<double>{
                ov.present_value(), ov.std_error().value_or(-1.0), ov.delta(kNumerical)};
        });
}

TEST_CASE("concurrent queries on cold barrier instances agree", "[threads][barrier]") {
    SECTION("analytical knock-out (PDE-identity theta, numerical Greeks)") {
        check_shared_instance(
            [] {
                return OptionValuation(
                    ud(), barrier(BarrierAction::Out), PricingMethod::Analytical);
            },
            default_queries);
    }
    SECTION("PDE knock-in (lazily built vanilla leg)") {
        check_shared_instance(
            [] {
                auto params = PdeParams::for_barriers(BarrierMonitoring::Continuous);
                params.spot_steps = 100;
                params.time_steps = 100;
                return OptionValuation(
                    ud(), barrier(BarrierAction::In), PricingMethod::PdeFd, params);
            },
            default_queries);
    }
    SECTION("Monte Carlo knock-out") {
        check_shared_instance(
            [] {
                return OptionValuation(
                    gbm(), barrier(BarrierAction::Out), PricingMethod::MonteCarlo);
            },
            [](const OptionValuation& ov) {
                return std::vector<double>{ov.present_value(), ov.delta(kNumerical)};
            });
    }
}

TEST_CASE("independent valuations and implied vols run in parallel", "[threads][implied_vol]") {
    // Each thread builds its own valuations; the only shared state is the
    // process-wide logger and warning handler. Quotes are priced at 25% vol
    // and solved from valuations built at 15%, so the solvers search: the
    // analytical Newton solve would otherwise start on the root.
    const auto work = [](int i) {
        const double spot = 90.0 + i;
        const UnderlyingData quoted(spot, 0.25, md());
        const UnderlyingData start_vol(spot, 0.15, md());
        const auto european = vanilla(OptionType::Call, ExerciseType::European);
        const auto american = vanilla(OptionType::Put, ExerciseType::American);
        const BinomialParams tree{.num_steps = 200};

        const double european_quote =
            OptionValuation(quoted, european, PricingMethod::Analytical).present_value();
        const double american_quote =
            OptionValuation(quoted, american, PricingMethod::Binomial, tree).present_value();
        const auto european_iv = implied_volatility(
            european_quote, OptionValuation(start_vol, european, PricingMethod::Analytical));
        const auto american_iv = implied_volatility(
            american_quote, OptionValuation(start_vol, american, PricingMethod::Binomial, tree));
        return std::vector<double>{european_quote,
                                   american_quote,
                                   european_iv.implied_vol,
                                   american_iv.implied_vol,
                                   static_cast<double>(european_iv.iterations),
                                   static_cast<double>(american_iv.iterations)};
    };

    std::vector<std::vector<double>> expected;
    expected.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        expected.push_back(work(i));
    }

    std::vector<std::vector<double>> results(kThreads);
    std::latch start(kThreads);
    {
        std::vector<std::jthread> workers;
        workers.reserve(kThreads);
        for (int i = 0; i < kThreads; ++i) {
            workers.emplace_back([&, i] {
                start.arrive_and_wait();
                results[static_cast<std::size_t>(i)] = work(i);
            });
        }
    }
    CHECK(results == expected);
}

TEST_CASE("warnings from many threads all reach the handler", "[threads][diagnostics]") {
    // The handler must be thread-safe (documented); the library must neither
    // drop nor tear concurrent warnings while it dispatches them.
    std::atomic<int> count{0};
    set_warning_handler([&count](std::string_view message) {
        if (message.find("from thread") != std::string_view::npos) {
            count.fetch_add(1, std::memory_order_relaxed);
        }
    });
    constexpr int kWarningsPerThread = 200;
    std::latch start(kThreads);
    {
        std::vector<std::jthread> workers;
        workers.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t) {
            workers.emplace_back([&start, t] {
                start.arrive_and_wait();
                const std::string message = "warning from thread " + std::to_string(t);
                for (int i = 0; i < kWarningsPerThread; ++i) {
                    warn(message);
                }
            });
        }
    }
    set_warning_handler({});
    CHECK(count.load() == kThreads * kWarningsPerThread);
}
