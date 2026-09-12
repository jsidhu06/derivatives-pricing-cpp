#pragma once

#include "engines/engine.hpp"
#include "valuation_state.hpp"

#include <Eigen/Core>

#include <optional>

/// Internal: finite-difference (PDE) engine for vanilla and custom payoffs,
/// European and American exercise.
namespace dp::engines {

/// Black-Scholes PDE solved on a spot or log-spot grid with implicit,
/// explicit, explicit-Hull or (Rannacher-smoothed) Crank-Nicolson time
/// stepping. American exercise via intrinsic projection or projected SOR.
/// Discrete dividends enter as jump conditions on tau-snapped grid dates;
/// custom payoffs use affine wing boundary models (explicit or fitted).
///
/// Native "grid" Greeks read the solved surface: parabolic-Lagrange delta at
/// exactly spot, cubic-Lagrange gamma with a parabolic fallback, and theta
/// via the Black-Scholes PDE identity — with an early-exercise short-circuit
/// where the spot node sits in the exercise region.
///
/// Caching: the solve is stored in a `mutable` member; every public entry
/// point is reached through the facade's per-instance memoisation lock, so
/// the engine performs no locking of its own.
class PdeEngine final : public PricingEngine {
public:
    explicit PdeEngine(const detail::ValuationState& state);

    [[nodiscard]] std::string_view name() const noexcept override { return "PDE_FD"; }
    [[nodiscard]] double present_value() const override;
    [[nodiscard]] double delta() const override;
    [[nodiscard]] double gamma() const override;
    /// Per calendar day.
    [[nodiscard]] double theta() const override;

private:
    /// PDE solve outputs needed for the price and the grid Greeks.
    struct Solution {
        double price;
        Eigen::ArrayXd spots;        ///< Spot grid S (length M+1).
        Eigen::ArrayXd values;       ///< V at pricing time.
        Eigen::ArrayXd values_prev;  ///< V one time step earlier.
        double last_dtau;            ///< Size of that last step.
    };

    [[nodiscard]] const Solution& solution() const;
    /// (delta, gamma) when the spot node sits in the American early-exercise
    /// region — the stencils are unreliable there; nullopt otherwise.
    [[nodiscard]] std::optional<std::pair<double, double>> intrinsic_short_circuit(
        const Eigen::ArrayXd& S, const Eigen::ArrayXd& V, Eigen::Index j) const;

    const detail::ValuationState& state_;
    mutable std::optional<Solution> solution_;
};

}  // namespace dp::engines
