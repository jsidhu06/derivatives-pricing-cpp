# derivatives-pricing-cpp

[![CI](https://github.com/jsidhu06/derivatives-pricing-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/jsidhu06/derivatives-pricing-cpp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jsidhu06/derivatives-pricing-cpp/branch/main/graph/badge.svg)](https://codecov.io/gh/jsidhu06/derivatives-pricing-cpp)

A C++20 library for options pricing and Greeks computation, with a unified API
across analytical, binomial tree, finite-difference and Monte Carlo engines. A C++ implementation of the
[`derivatives-pricing`](https://github.com/jsidhu06/derivatives-pricing)
Python library, built for teaching, research, and production-adjacent workflows.

## Features

- **Contracts.** European and American vanillas, custom payoffs, and all
  eight knock-in / knock-out barrier types with continuous or discrete
  monitoring and rebates.
- **Engines.**
  - Analytical: Black-Scholes-Merton (vanilla) and Reiner-Rubinstein (barriers) with BGK
  adjustment for discrete monitoring.
  - Binomial: Cox-Ross-Rubinstein trees.
  - PDE: implicit, explicit and Crank-Nicolson schemes with Rannacher
    smoothing, on spot or log-spot grids.
  - Monte Carlo: GBM, Merton jump-diffusion and CIR processes;
    Longstaff-Schwartz for American exercise.
- **Greeks.** Delta, gamma, vega, theta and rho: analytical, tree, grid,
  pathwise, likelihood-ratio or bump-and-revalue, depending on the engine.
- **Implied volatility.** Newton-Raphson, TOMS 748 and bisection, for
  European and American vanillas.
- **Market data.** Term-structure discount and dividend curves, discrete cash
  dividends, and ACT/360, ACT/365F, ACT/365.25 and 30/360 day counts.

## Quick start

```cpp
#include "dp/dp.hpp"

#include <format>
#include <iostream>

int main() {
    using namespace dp;

    const MarketData market(make_datetime(2025, 1, 1), DiscountCurve::flat(0.05), "USD");
    const UnderlyingData underlying(100.0, 0.20, market);  // spot, volatility

    const VanillaSpec put({.option_type = OptionType::Put,
                           .exercise_type = ExerciseType::American,
                           .strike = 105.0,
                           .maturity = make_datetime(2026, 1, 1)});

    const OptionValuation tree(underlying, put, PricingMethod::Binomial);
    const OptionValuation pde(underlying, put, PricingMethod::PdeFd);

    std::cout << std::format("binomial  PV {:.4f}  delta {:.4f}\n",
                             tree.present_value(), tree.delta());
    std::cout << std::format("PDE       PV {:.4f}  delta {:.4f}\n",
                             pde.present_value(), pde.delta());

    // Volatility implied by a market quote of 9.50.
    const ImpliedVolResult iv = implied_volatility(9.50, tree);
    std::cout << std::format("implied vol {:.4f}\n", iv.implied_vol);
}
```

```text
binomial  PV 8.7416  delta -0.5304
PDE       PV 8.7382  delta -0.5306
implied vol 0.2200
```

More in [`examples/`](examples): the same contract across all four engines
with Greeks, barrier options, Monte Carlo with standard errors, and an
implied-volatility smile. They build with the library as `dp_example_*`.

## Validation

The test suite has 355 test cases and runs in CI on GCC 13 and Clang 18,
under AddressSanitizer, UndefinedBehaviorSanitizer and ThreadSanitizer, with
clang-tidy and `-Werror`.

- **Reference tables.** About 600 golden rows generated from the Python
  library (`tools/gen_*_golden.py`): prices to 1e-8 or tighter for the
  analytical, binomial and PDE engines, barriers included, and implied vols
  to 2e-8.
- **Published results.** Barrier prices from Boyle-Tian (1998),
  Broadie-Glasserman (1997) and Dai-Kwok (2004); American puts from
  Longstaff-Schwartz (2001).
- **Convergence.** Measured orders of convergence: first order for the
  binomial tree, second order for Crank-Nicolson, and 1/sqrt(N) for the
  Monte Carlo standard error.
- **Monte Carlo.** Tested statistically against closed forms:
  Black-Scholes-Merton within standard error, and the Merton (1976) series
  and the exact CIR mean within fixed tolerances. Random streams are
  reproducible from a seed.
- **Cross-engine.** The same contract priced by different engines must agree
  within each engine's discretisation error.

## Build and test

Requirements: CMake 3.25 or later, Ninja and a C++20 compiler (GCC 13+ or
Clang 17+). Dependencies are fetched and pinned by `FetchContent` on first
configure: Eigen 3.4, Boost.Math 1.86, spdlog 1.15, Catch2 3.8, and Google
Benchmark 1.9 when benchmarks are enabled.

```bash
cmake --workflow --preset release   # configure, build and test

cmake --preset debug                # ASan + UBSan
cmake --build --preset debug
ctest --preset debug
```

Other presets: `relwithdebinfo`, `bench`, `coverage`, `tidy` (clang-tidy),
`ci-*` (`-Werror`), `ci-tsan` (ThreadSanitizer) and `clang-debug` /
`clang-release`. The debug presets skip tests tagged `[slow]`; the release
presets run everything.

For a coverage report (needs [`gcovr`](https://gcovr.com)):

```bash
cmake --preset coverage && cmake --build --preset coverage && ctest --preset coverage
gcovr build/coverage
```

Formatting (`.clang-format`) and static analysis (`.clang-tidy`) run in CI
and through `pre-commit`.

To regenerate the golden tables, install the reference Python library into a
virtualenv pinned to its lockfile:

```bash
.venv-goldenref/bin/pip-sync derivatives-pricing/requirements/dev.txt
.venv-goldenref/bin/pip install --no-deps -e ./derivatives-pricing
```

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

- **Dates.** `dp::DateTime` is a `std::chrono` time point with microsecond
  resolution and no timezone. ACT/* day counts honour the time of day.
- **Discrete dividends.** `Analytical` and `Binomial` use the escrowed
  dividend model; `PdeFd` and `MonteCarlo` drop the spot by the dividend
  amount on each ex-date.
- **Warnings and errors.** Recoverable conditions, such as curve
  extrapolation or a truncated PDE domain, go through `dp::warn`, which
  applications can redirect with `dp::set_warning_handler`. Invalid inputs
  throw.
- **Thread safety.** An `OptionValuation` is immutable once constructed and
  caches its results; one instance can be queried from several threads.
- **Dependencies over hand-rolled code.** Eigen for arrays and linear
  algebra, Boost.Math for distributions and root finding, PCG for random
  numbers.

Asian options are out of scope for the first version.

## License

MIT. See `LICENSE`.
