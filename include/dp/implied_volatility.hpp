#pragma once

#include "dp/enums.hpp"
#include "dp/valuation.hpp"

#include <optional>

/// Implied volatility: invert a deterministic pricer for the volatility that
/// reproduces an observed option price.
namespace dp {

/// Solver controls for `implied_volatility`.
struct ImpliedVolParams {
    /// Root finder. `Toms748` is Boost.Math's TOMS 748 (Alefeld, Potra &
    /// Shi): a bracketing method that, like Brent's, mixes interpolation
    /// steps with bisection safeguards, with better worst-case convergence.
    /// Unset: picked by engine. Safeguarded Newton-Raphson for `Analytical`
    /// (closed-form vega is cheap); TOMS 748 for `Binomial` and `PdeFd` (no
    /// vega needed, whereas a numerical vega costs two extra pricings per
    /// Newton step).
    std::optional<ImpliedVolMethod> method = std::nullopt;
    /// Starting point for Newton-Raphson. When unset: the valuation's own
    /// volatility for `Analytical`, an extrinsic-value heuristic otherwise.
    std::optional<double> initial_vol = std::nullopt;
    /// Search interval for the volatility (annualised). Expanded outwards
    /// (halving `vol_lower`, doubling `vol_upper`, up to 6 times each) when
    /// it does not bracket the target.
    double vol_lower = 1.0e-6;
    double vol_upper = 5.0;
    /// Absolute tolerance on the implied volatility (annualised vol units,
    /// e.g. 1e-8 = 1e-6 vol points), the same for every method: the result
    /// is within `tol` of the root. Bisection and TOMS 748 stop once the bracket
    /// is within `tol`; a Newton result is accepted only if the price
    /// residual changes sign across `[vol - tol, vol + tol]`, otherwise it
    /// falls back to bisection. A price tolerance would not do: the vol error
    /// is the price error divided by vega, so one price tolerance is loose
    /// far from the money and tight near it.
    double tol = 1.0e-8;
    int max_iter = 100;
    /// Smallest vega (price change per unit of vol, i.e. per 100 vol points)
    /// at the solution for which the price is taken to determine the vol.
    /// Below it the result has `well_determined = false` and a warning is
    /// issued: a far out-of-the-money or near-intrinsic quote barely moves
    /// with vol, so any rounding in the price is a large vol error. Unset:
    /// `1e-6 * max(spot, strike)`; 0 disables the check.
    std::optional<double> min_vega = std::nullopt;

    void validate() const;
};

/// Solver output. Use `implied_vol` only when `well_determined` is true
/// (which implies `converged`).
struct ImpliedVolResult {
    double implied_vol = 0.0;
    /// Iterations of the final root finder; 0 when the root sat on a bracket
    /// end.
    int iterations = 0;
    /// True when the root finder located the volatility to within `tol`.
    bool converged = false;
    /// Model pricings used in total: bracketing, solving and the final
    /// conditioning check, each numerical-vega bump counted. For tree and
    /// PDE engines this, not `iterations`, measures the cost.
    int function_evaluations = 0;
    /// True when the solve converged and vega at the solution is at least
    /// `min_vega`, i.e. the price actually pins the volatility down; false
    /// for a result that did not converge. A converged but not
    /// well-determined result reprices the target, yet small rounding in the
    /// price would move it a lot (far out of the money, near intrinsic).
    bool well_determined = false;
};

/// Volatility at which `valuation` reprices to `target_price` (per unit of
/// underlying, contract size not applied).
///
/// Supported: `VanillaSpec` on `UnderlyingData`, priced by `Analytical` (European) or
/// by the binomial / PDE engines (European or American). Every evaluation
/// revalues a copy of `valuation` with the volatility replaced, so the
/// engine parameters carry over. Newton-Raphson uses the analytical vega for
/// `Analytical` and the numerical vega otherwise, and falls back to bisection if it
/// fails to converge.
///
/// Throws `UnsupportedFeatureError` for barrier / custom-payoff specs and
/// Monte Carlo pricing (no guaranteed monotone, deterministic P(sigma)),
/// `ValidationError` for a target outside the no-arbitrage price bounds or
/// invalid params, and `ConvergenceError` when the target cannot be
/// bracketed within the (expanded) volatility interval.
[[nodiscard]] ImpliedVolResult implied_volatility(double target_price,
                                                  const OptionValuation& valuation,
                                                  const ImpliedVolParams& params = {});

}  // namespace dp
