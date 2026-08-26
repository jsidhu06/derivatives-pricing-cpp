# derivatives-pricing-cpp

C++20 port of the [`derivatives-pricing`](https://github.com/jsidhu06/derivatives-pricing)
Python library: options pricing and Greeks with a unified API across analytical,
binomial-tree, finite-difference and Monte Carlo engines.

## Status

Early development. Phase 1 (foundations) is in place:

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

Planned (in order): BSM + analytical Greeks → contract specs and valuation
facade → binomial trees → PDE finite differences → Monte Carlo (GBM/JD/SRD,
Longstaff-Schwartz) → barrier analytics and implied vol → pybind11 bindings.
Asian options are out of scope for v1.

## Build

Requirements: CMake ≥ 3.25, Ninja, a C++20 compiler (GCC 13+, Clang 17+),
network access on first configure (dependencies are fetched and pinned via
`FetchContent`: Eigen 3.4, Boost.Math 1.86, spdlog 1.15, Catch2 3.8).

```bash
cmake --preset debug          # ASan + UBSan
cmake --build --preset debug
ctest --preset debug

cmake --workflow --preset release   # configure + build + test in one go
```

Presets: `debug`, `release`, `relwithdebinfo`, `tidy` (clang-tidy on every TU),
and `ci-*` variants with `-Werror`.

### Tooling

```bash
pre-commit install            # clang-format, cmake-format, codespell, hygiene hooks
pre-commit run --all-files
```

Formatting is `.clang-format` (Google base, 4-space indent, 100 columns); static
analysis is `.clang-tidy`. Both run in CI.

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
