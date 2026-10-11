// Pricing-engine benchmarks. Build and run in Release only (the `bench`
// preset); debug and sanitizer timings are not representative.
//
// Every iteration constructs a fresh `OptionValuation`: results are memoised
// per instance, so reusing one would time a cache lookup. The timings
// therefore include contract validation and engine setup, as a caller pays.
//
// Library warnings are silenced: the implied-vol bracket probes extreme
// volatilities that warn about the PDE domain, and logging them would be
// timed too.

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

#include <string_view>
#include <utility>

#include <benchmark/benchmark.h>

using namespace dp;

namespace {

const DateTime kPricingDate = make_datetime(2025, 1, 1);
const DateTime kMaturity = make_datetime(2026, 1, 1);

constexpr double kSpot = 100.0;
constexpr double kStrike = 100.0;
constexpr double kVol = 0.20;
constexpr double kRate = 0.05;
constexpr double kDivYield = 0.02;

MarketData market() {
    return {kPricingDate, DiscountCurve::flat(kRate), "USD"};
}

UnderlyingData underlying(double vol = kVol) {
    return {kSpot, vol, market(), {}, DiscountCurve::flat(kDivYield)};
}

VanillaSpec vanilla(OptionType type, ExerciseType exercise) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = exercise,
                        .strike = kStrike,
                        .maturity = kMaturity,
                        .currency = "USD"});
}

BarrierSpec down_and_out_call(BarrierMonitoring monitoring) {
    BarrierSpec::Params params{.option_type = OptionType::Call,
                               .exercise_type = ExerciseType::European,
                               .strike = kStrike,
                               .maturity = kMaturity,
                               .barrier = 90.0,
                               .direction = BarrierDirection::Down,
                               .action = BarrierAction::Out,
                               .monitoring = monitoring,
                               .currency = "USD"};
    if (monitoring == BarrierMonitoring::Discrete) {
        params.num_observations = 52;  // weekly
    }
    return BarrierSpec(std::move(params));
}

GbmProcess gbm(int paths, int steps) {
    return {market(),
            {.initial_value = kSpot,
             .volatility = kVol,
             .dividend_curve = DiscountCurve::flat(kDivYield)},
            {.paths = paths, .end_date = kMaturity, .num_steps = steps}};
}

PdeParams pde_grid(int nodes) {
    return {.spot_steps = nodes, .time_steps = nodes};
}

// ── Analytical ──────────────────────────────────────────────────────

void analytical_european(benchmark::State& state) {
    const UnderlyingData ud = underlying();
    const VanillaSpec spec = vanilla(OptionType::Call, ExerciseType::European);
    for (auto _ : state) {
        const OptionValuation valuation(ud, spec, PricingMethod::Analytical);
        benchmark::DoNotOptimize(valuation.present_value());
    }
}
BENCHMARK(analytical_european);

void analytical_european_greeks(benchmark::State& state) {
    const UnderlyingData ud = underlying();
    const VanillaSpec spec = vanilla(OptionType::Call, ExerciseType::European);
    for (auto _ : state) {
        const OptionValuation valuation(ud, spec, PricingMethod::Analytical);
        benchmark::DoNotOptimize(valuation.present_value());
        benchmark::DoNotOptimize(valuation.delta());
        benchmark::DoNotOptimize(valuation.gamma());
        benchmark::DoNotOptimize(valuation.vega());
        benchmark::DoNotOptimize(valuation.theta());
        benchmark::DoNotOptimize(valuation.rho());
    }
}
BENCHMARK(analytical_european_greeks);

void analytical_barrier(benchmark::State& state) {
    const UnderlyingData ud = underlying();
    const BarrierSpec spec = down_and_out_call(BarrierMonitoring::Continuous);
    for (auto _ : state) {
        const OptionValuation valuation(ud, spec, PricingMethod::Analytical);
        benchmark::DoNotOptimize(valuation.present_value());
    }
}
BENCHMARK(analytical_barrier);

// ── Binomial (CRR) ──────────────────────────────────────────────────

void binomial_american_put(benchmark::State& state) {
    const UnderlyingData ud = underlying();
    const VanillaSpec spec = vanilla(OptionType::Put, ExerciseType::American);
    const BinomialParams params{.num_steps = static_cast<int>(state.range(0))};
    for (auto _ : state) {
        const OptionValuation valuation(ud, spec, PricingMethod::Binomial, params);
        benchmark::DoNotOptimize(valuation.present_value());
    }
}
BENCHMARK(binomial_american_put)->Arg(500)->Arg(5000)->Unit(benchmark::kMillisecond);

// ── PDE finite differences ──────────────────────────────────────────

void pde_european_call(benchmark::State& state) {
    const UnderlyingData ud = underlying();
    const VanillaSpec spec = vanilla(OptionType::Call, ExerciseType::European);
    const PdeParams params = pde_grid(static_cast<int>(state.range(0)));
    for (auto _ : state) {
        const OptionValuation valuation(ud, spec, PricingMethod::PdeFd, params);
        benchmark::DoNotOptimize(valuation.present_value());
    }
}
BENCHMARK(pde_european_call)->Arg(200)->Arg(800)->Unit(benchmark::kMillisecond);

/// Crank-Nicolson with projected SOR for the early-exercise constraint.
void pde_american_put(benchmark::State& state) {
    const UnderlyingData ud = underlying();
    const VanillaSpec spec = vanilla(OptionType::Put, ExerciseType::American);
    const PdeParams params = pde_grid(static_cast<int>(state.range(0)));
    for (auto _ : state) {
        const OptionValuation valuation(ud, spec, PricingMethod::PdeFd, params);
        benchmark::DoNotOptimize(valuation.present_value());
    }
}
BENCHMARK(pde_american_put)->Arg(200)->Arg(800)->Unit(benchmark::kMillisecond);

/// `PdeParams::for_barriers` presets: Crank-Nicolson for continuous
/// monitoring, explicit-Hull for discrete.
void pde_barrier(benchmark::State& state, BarrierMonitoring monitoring) {
    const UnderlyingData ud = underlying();
    const BarrierSpec spec = down_and_out_call(monitoring);
    const PdeParams params = PdeParams::for_barriers(monitoring);
    for (auto _ : state) {
        const OptionValuation valuation(ud, spec, PricingMethod::PdeFd, params);
        benchmark::DoNotOptimize(valuation.present_value());
    }
}
BENCHMARK_CAPTURE(pde_barrier, continuous, BarrierMonitoring::Continuous)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(pde_barrier, discrete_weekly, BarrierMonitoring::Discrete)
    ->Unit(benchmark::kMillisecond);

// ── Monte Carlo ─────────────────────────────────────────────────────

/// Weekly steps, antithetic paths with moment matching (the defaults).
void monte_carlo_european(benchmark::State& state) {
    const auto paths = static_cast<int>(state.range(0));
    const GbmProcess process = gbm(paths, 52);
    const VanillaSpec spec = vanilla(OptionType::Call, ExerciseType::European);
    for (auto _ : state) {
        const OptionValuation valuation(process, spec, PricingMethod::MonteCarlo);
        benchmark::DoNotOptimize(valuation.present_value());
    }
    state.counters["paths/s"] =
        benchmark::Counter(paths, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(monte_carlo_european)->Arg(10'000)->Arg(100'000)->Unit(benchmark::kMillisecond);

/// Longstaff-Schwartz with 50 exercise dates and a cubic polynomial basis.
void monte_carlo_american_lsm(benchmark::State& state) {
    const auto paths = static_cast<int>(state.range(0));
    const GbmProcess process = gbm(paths, 50);
    const VanillaSpec spec = vanilla(OptionType::Put, ExerciseType::American);
    for (auto _ : state) {
        const OptionValuation valuation(process, spec, PricingMethod::MonteCarlo);
        benchmark::DoNotOptimize(valuation.present_value());
    }
    state.counters["paths/s"] =
        benchmark::Counter(paths, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(monte_carlo_american_lsm)->Arg(10'000)->Arg(100'000)->Unit(benchmark::kMillisecond);

// ── Implied volatility ──────────────────────────────────────────────

/// Recover sigma = 0.20 from the analytical price. The valuation passed to the
/// solver is at sigma = 0.35, Newton's starting point.
void implied_vol_analytical(benchmark::State& state, ImpliedVolMethod method) {
    const VanillaSpec spec = vanilla(OptionType::Call, ExerciseType::European);
    const double target =
        OptionValuation(underlying(), spec, PricingMethod::Analytical).present_value();
    const OptionValuation valuation(
        UnderlyingData(kSpot, 0.35, market(), {}, DiscountCurve::flat(kDivYield)),
        spec,
        PricingMethod::Analytical);
    const ImpliedVolParams params{.method = method};
    int evaluations = 0;
    for (auto _ : state) {
        ImpliedVolResult result = implied_volatility(target, valuation, params);
        evaluations = result.function_evaluations;
        benchmark::DoNotOptimize(result.implied_vol);
    }
    state.counters["pricings"] = evaluations;
}
BENCHMARK_CAPTURE(implied_vol_analytical, newton, ImpliedVolMethod::NewtonRaphson)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(implied_vol_analytical, toms748, ImpliedVolMethod::Toms748)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(implied_vol_analytical, bisection, ImpliedVolMethod::Bisection)
    ->Unit(benchmark::kMicrosecond);

/// American put on a 200 x 200 PDE grid: each solver evaluation is a full
/// PDE solve, so `function_evaluations` sets the cost. Newton's numerical
/// vega costs two extra solves per step, which is why TOMS 748 is the
/// default for this engine.
void implied_vol_pde_american(benchmark::State& state, ImpliedVolMethod method) {
    const VanillaSpec spec = vanilla(OptionType::Put, ExerciseType::American);
    const PdeParams grid = pde_grid(200);
    const double target =
        OptionValuation(underlying(), spec, PricingMethod::PdeFd, grid).present_value();
    const OptionValuation valuation(
        UnderlyingData(kSpot, 0.35, market(), {}, DiscountCurve::flat(kDivYield)),
        spec,
        PricingMethod::PdeFd,
        grid);
    const ImpliedVolParams params{.method = method};
    int evaluations = 0;
    for (auto _ : state) {
        ImpliedVolResult result = implied_volatility(target, valuation, params);
        evaluations = result.function_evaluations;
        benchmark::DoNotOptimize(result.implied_vol);
    }
    state.counters["pricings"] = evaluations;
}
BENCHMARK_CAPTURE(implied_vol_pde_american, newton, ImpliedVolMethod::NewtonRaphson)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(implied_vol_pde_american, toms748, ImpliedVolMethod::Toms748)
    ->Unit(benchmark::kMillisecond);

}  // namespace

int main(int argc, char** argv) {
    dp::set_warning_handler([](std::string_view /*message*/) {});
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
