# derivatives-pricing-cpp

C++20 port of the [`derivatives-pricing`](https://github.com/jsidhu06/derivatives-pricing)
Python library: options pricing and Greeks with a unified API across analytical,
binomial-tree, finite-difference and Monte Carlo engines.

## Status

Early development. Foundations, the valuation facade, all four pricing
engines (with barriers) and implied volatility are in place:

| Module | Header | Notes |
|---|---|---|
| Enums | `dp/enums.hpp` | option/exercise/method/day-count/barrier enums with `to_string` |
| Exceptions | `dp/exceptions.hpp` | hierarchy rooted at `dp::DerivativesPricingError` |
| Dates | `dp/dates.hpp` | `DateTime` (µs, timezone-free), day-count conventions, `date_range` |
| Discount curve | `dp/discount_curve.hpp` | log-linear DF interpolation, forwards, zero-rate bumps |
| Market data | `dp/market_data.hpp` | pricing date + curve + currency + day count |
| Dividends / forwards | `dp/dividends.hpp`, `dp/forward.hpp` | PV of cash dividends, forward price, put-call parity |
| Normal distribution | `dp/math/normal.hpp` | CDF / PDF / quantile via Boost.Math |
| Diagnostics | `dp/diagnostics.hpp` | pluggable warning handler, log level |
| Contracts | `dp/contracts.hpp` | `VanillaSpec`, `PayoffSpec`, `BarrierSpec`; `OptionSpec` variant; `contract_event_dates` (maturity + monitoring dates a simulation grid must hold) |
| Params | `dp/params.hpp` | `MonteCarloParams`, `BinomialParams`, `PdeParams` |
| Underlying | `dp/underlying.hpp` | `UnderlyingData` with `with_*` bump helpers |
| Valuation | `dp/valuation.hpp` | `OptionValuation` facade: cached PV, Monte Carlo standard error and Greeks, bump-and-revalue |
| Analytical engines | (internal) | `PricingMethod::Analytical`: Black-Scholes-Merton closed-form European vanilla PV + analytical Greeks, and the barrier engine below |
| Analytical barrier engine | (internal) | Reiner-Rubinstein closed forms for all 8 knock-in/out barrier types; Broadie-Glasserman-Kou discrete-monitoring correction; at-hit / at-expiry rebates; PDE-identity theta; barrier-clamped numerical greeks; inception-trigger handling |
| Binomial engine | (internal) | CRR trees: vanilla + custom payoffs, European/American, escrowed discrete dividends, tree Greeks, arbitrage guard, European control variate; barriers via Boyle-Lau aligned trees (continuous) and half-step placement (discrete), knock-in two-state recursion, rebates, near-barrier greek guards |
| PDE engine | (internal) | implicit / explicit / explicit-Hull / Rannacher-CN on spot or log-spot grids; PSOR or intrinsic American; dividend jump conditions; affine boundary models; CFL guard; grid Greeks; barriers via grid truncation (continuous KO), monitoring-date resets (discrete), two-surface coupled American knock-ins, and European KI parity |
| Simulation | `dp/simulation.hpp` | `GbmProcess` / `JdProcess` (Merton) / `SrdProcess` (CIR, full-truncation Euler) + `SimulationConfig`: PCG64 + inverse-CDF normals, antithetic + moment matching, observation-date grids, discrete dividends; `CorrelationContext` for Cholesky-correlated multi-asset draws on one shared time grid (built with `shared_time_grid`; every process is validated against it) |
| MC engine | (internal) | European (discounted terminal payoff) and American (ridge-regularised Laguerre Longstaff-Schwartz); pathwise + likelihood-ratio Greeks; CRN by default; `std_error()` with antithetic pairs averaged, plus a high-error warning; barriers via pathwise monitoring checks (discrete, any process) and Brownian-bridge crossing probabilities (continuous, GBM), barrier-state LSM with optional barrier-aware basis |
| Implied volatility | `dp/implied_volatility.hpp` | `implied_volatility(target, valuation, params)` for vanillas priced by `Analytical` (European), binomial or PDE (European/American): no-arbitrage bound checks, bracket expansion, Boost.Math safeguarded Newton-Raphson (analytical / numerical vega, certified in vol space, bisection fallback), bisection and TOMS 748; one absolute vol tolerance for every method, and quotes whose vega is too small to determine the vol (far out of the money, near intrinsic) are flagged `well_determined = false`; Newton by default for `Analytical`, TOMS 748 for the tree and PDE (no vega needed) |

**Discrete dividends** use one of two standard models, depending on the
engine. `Analytical` and `Binomial` use the escrowed model: the spot less the
PV of dividends to maturity follows GBM. `PdeFd` and `MonteCarlo` drop the
spot by each cash amount on its ex-date. The two give different prices for
the same inputs, so compare engines within a model. For the same reason the
European control variate is rejected for `PdeFd` and `MonteCarlo` when there
are dividends during the option's life: its correction would mix the two
models. Both models share QuantLib's forward: with a dividend yield as well,
each dividend's PV is discounted at `r - q`.

**Theta on a sloped curve** depends on the method. Analytical and Monte Carlo
pathwise / likelihood-ratio theta use the zero rates to maturity (as
QuantLib's `BlackCalculator`); tree and PDE theta use the short-end forward
rates, holding today's calendar curve fixed (as QuantLib's FD engines);
numerical theta slides the curve with the pricing date, which brings in the
forward rate at maturity. On a flat curve they agree. One convention across
all methods is planned.

Engines are verified against the Python reference implementation on golden
tables generated by `tools/gen_*_golden.py` (prices; Greeks at looser
tolerances): BSM 152 rows to 1e-11; binomial 116 rows to 1e-10; PDE 57 rows
to 1e-9; analytical barriers 46 rows to 1e-11; binomial barriers 36 rows to
1e-10; PDE barriers 39 rows to 1e-9; implied vol 90 rows to 2e-8 in vol;
market conventions (term-structure curves, ACT/360 / ACT/365.25 / 30/360,
intraday timestamps) 71 rows across BSM / binomial / PDE. Regenerate with the
gitignored `.venv-goldenref` virtualenv, pinned to the reference library's
lockfile so the dependencies are exact:

```bash
.venv-goldenref/bin/pip-sync derivatives-pricing/requirements/dev.txt
.venv-goldenref/bin/pip install --no-deps -e ./derivatives-pricing
```

Barrier engines are also held to published tables
(`tests/test_barrier_papers.cpp`): Boyle-Tian (1998) Tables 3, 6, 8 and 9,
Broadie-Glasserman (1997) Tables 2.1 and 2.4, and Dai-Kwok (2004), with the
Python suite's tolerances.

Monte Carlo agreement with the Python reference is *statistical* (the RNG
streams are deliberately not numpy-compatible): estimators are tested
against BSM analytical values and the binomial American price, plus
seed-pinned reproducibility. The jump-diffusion engine is additionally
tested against the Merton (1976) closed-form series and the square-root
diffusion against the exact CIR conditional mean.

Three names differ from the Python library (an abbreviation, a keyword
workaround, and a different algorithm); the rest match:

| Python | C++ |
|---|---|
| `JDParams.lambd` | `JdParams::jump_intensity` |
| `MonteCarloParams.deg` | `MonteCarloParams::basis_degree` |
| `ImpliedVolMethod.BRENTQ` (scipy `brentq`) | `ImpliedVolMethod::Toms748` (Boost TOMS 748, a different bracketing method) |

Planned: pybind11 bindings. Asian options are out of scope for v1.

## Quick start

```cpp
#include "dp/dp.hpp"

#include <format>
#include <iostream>

int main() {
    using namespace dp;

    const auto pricing_date = make_datetime(2025, 1, 1);
    const auto maturity = make_datetime(2025, 7, 1);

    const MarketData market(pricing_date, DiscountCurve::flat(0.05), "USD");
    const UnderlyingData underlying(100.0, 0.20, market);
    const VanillaSpec spec({.option_type = OptionType::Call,
                            .exercise_type = ExerciseType::European,
                            .strike = 105.0,
                            .maturity = maturity});

    const OptionValuation val(underlying, spec, PricingMethod::Analytical);
    std::cout << std::format("PV {:.4f}  delta {:.4f}\n", val.present_value(), val.delta());
    std::cout << std::format("numerical vega {:.4f}\n",
                             val.vega({.method = GreekCalculationMethod::Numerical}));
}
```

## Build

Requirements: CMake ≥ 3.25, Ninja, a C++20 compiler (GCC 13+, Clang 17+),
network access on first configure (dependencies are fetched and pinned via
`FetchContent`: Eigen 3.4, Boost.Math 1.86, spdlog 1.15, Catch2 3.8, and Google
Benchmark 1.9 when benchmarks are enabled).

```bash
cmake --preset debug          # ASan + UBSan
cmake --build --preset debug
ctest --preset debug

cmake --workflow --preset release   # configure + build + test in one go
```

Presets: `debug`, `release`, `relwithdebinfo`, `bench`, `tidy` (clang-tidy on every TU),
`ci-*` variants with `-Werror`, `ci-tsan` (ThreadSanitizer, clang-18; runs the
`[threads]` tests), and `clang-debug` / `clang-release` (clang-18, `-Werror`) to
exercise the second compiler in the CI matrix locally:

```bash
cmake --workflow --preset clang-debug
```

Tests tagged `[slow]` (the heavier paper tables and Monte Carlo cross-checks)
take minutes under the sanitizer build, so the debug test presets skip them;
release presets run everything. To run them in a debug build anyway:
`build/debug/tests/dp_tests "[slow]"`.

### Tooling

```bash
pre-commit install            # clang-format, cmake-format, codespell, hygiene hooks
pre-commit run --all-files
```

Formatting is `.clang-format` (Google base, 4-space indent, 100 columns); static
analysis is `.clang-tidy`. Both run in CI.

## Benchmarks

`bench/` holds a [Google Benchmark](https://github.com/google/benchmark) suite,
off by default (`DP_BUILD_BENCHMARKS`). Build it with the `bench` preset
(Release, no tests); debug and sanitizer timings are not representative.

```bash
cmake --preset bench && cmake --build --preset bench
build/bench/bench/dp_bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

CI builds the suite in the release jobs and runs one iteration of each
benchmark, to check that they run; shared runners are too noisy for timings.

## Design notes

- **Dates.** `dp::DateTime` is `std::chrono::sys_time<std::chrono::microseconds>`.
  It is timezone-free by construction, so the Python library's runtime
  "naive datetime" checks disappear. ACT/* conventions honour time of day;
  30/360 US uses calendar fields.
- **Arrays.** Eigen `ArrayXd` for vector-valued inputs and outputs; linear
  algebra (Cholesky, least squares) will use Eigen's decompositions.
- **Warnings.** Recoverable conditions (curve extrapolation, negative rates,
  conflicting dividend inputs) go through `dp::warn`, which applications can
  redirect with `dp::set_warning_handler`. Hard errors throw.
- **Monte Carlo.** Will target *statistical* agreement with the Python
  reference (within standard error), never bitwise reproduction of numpy's
  random streams.

## License

MIT — see `LICENSE`.
