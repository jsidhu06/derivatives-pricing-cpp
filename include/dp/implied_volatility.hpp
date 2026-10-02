#pragma once

#include "dp/enums.hpp"
#include "dp/valuation.hpp"

#include <optional>

/// Implied volatility: invert a deterministic pricer for the volatility that
/// reproduces an observed option price.
namespace dp {

/// Solver controls for `implied_volatility`.
struct ImpliedVolParams {
    /// Root finder. `Brent` runs Boost.Math's TOMS 748 (Alefeld-Potra-Shi),
    /// a bracketing method in the Brent family with better worst-case
    /// convergence.
    ImpliedVolMethod method = ImpliedVolMethod::NewtonRaphson;
    /// Starting point for Newton-Raphson. When unset: the valuation's own
    /// volatility for BSM, an extrinsic-value heuristic otherwise.
    std::optional<double> initial_vol = std::nullopt;
    /// Search interval for the volatility (annualised). Expanded outwards
    /// (halving `vol_lower`, doubling `vol_upper`, up to 6 times each) when
    /// it does not bracket the target.
    double vol_lower = 1.0e-6;
    double vol_upper = 5.0;
    /// Absolute tolerance. A volatility whose price lies within `tol` of the
    /// target is accepted outright; the bracketing solvers (and Newton's
    /// bisection fallback) also stop once the volatility bracket is narrower
    /// than `tol`, so their repricing error can reach ~vega * tol / 2.
    double tol = 1.0e-8;
    int max_iter = 100;

    void validate() const;
};

/// Solver output.
struct ImpliedVolResult {
    double implied_vol;
    /// Pricer evaluations used by the final solver (bracketing excluded).
    int iterations;
    bool converged;
};

/// Volatility at which `valuation` reprices to `target_price` (per unit of
/// underlying, contract size not applied).
///
/// Supported: `VanillaSpec` on `UnderlyingData`, priced by BSM (European) or
/// by the binomial / PDE engines (European or American). Every evaluation
/// revalues a copy of `valuation` with the volatility replaced, so the
/// engine parameters carry over. Newton-Raphson uses the analytical vega for
/// BSM and the numerical vega otherwise, and falls back to bisection if it
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
