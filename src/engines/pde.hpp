#pragma once

#include "engines/engine.hpp"
#include "valuation_state.hpp"

#include <Eigen/Core>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

/// Internal: finite-difference (PDE) engine for vanilla, custom and barrier
/// payoffs, European and American exercise.
namespace dp::engines {

/// Black-Scholes PDE solved on a spot or log-spot grid with implicit,
/// explicit, explicit-Hull or (Rannacher-smoothed) Crank-Nicolson time
/// stepping. American exercise via intrinsic projection or projected SOR.
/// Discrete dividends enter as jump conditions on tau-snapped grid dates;
/// custom payoffs use affine wing boundary models (explicit or fitted).
///
/// Barriers: continuous knock-outs truncate the grid at the barrier (the
/// barrier becomes a Dirichlet boundary carrying the rebate value); discrete
/// knock-outs keep the full grid, anchor the barrier half a cell between
/// nodes, and apply barrier resets at monitoring taus. American knock-ins
/// solve a two-surface coupled PDE (active/inactive); European knock-ins use
/// native-surface parity `V_KI = V_vanilla + R df_T - V_KO` on the cached
/// KO + vanilla components.
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

    [[nodiscard]] Solution compute_vanilla() const;
    /// Black-Scholes-identity theta read from one solved surface.
    [[nodiscard]] double grid_theta_identity(const Solution& result) const;

    // ── Barrier machinery ───────────────────────────────────────────
    [[nodiscard]] bool is_european_knock_in() const;
    /// Monitoring dates converted to tau (time remaining), sorted ascending.
    [[nodiscard]] std::vector<double> barrier_monitoring_taus() const;
    [[nodiscard]] Solution compute_barrier_knock_out() const;
    /// Two-surface coupled active/inactive solve for American knock-ins.
    [[nodiscard]] Solution compute_barrier_knock_in() const;
    /// Cached (knock-out, vanilla) solves backing European KI parity.
    [[nodiscard]] const std::pair<Solution, Solution>& european_ki_components() const;
    /// Final tau step of the barrier time grid (for the rebate theta).
    [[nodiscard]] double barrier_last_dtau() const;
    /// Per-day theta of the discounted AT_EXPIRY rebate leg (0 otherwise).
    [[nodiscard]] double discounted_rebate_theta(double last_dtau) const;
    /// Lazily built vanilla-equivalent engine for inception-triggered
    /// knock-ins (the contract has collapsed to the vanilla).
    [[nodiscard]] const PdeEngine& vanilla_equivalent_engine() const;

    const detail::ValuationState& state_;
    mutable std::optional<Solution> solution_;
    mutable std::optional<std::pair<Solution, Solution>> ki_components_;
    mutable std::optional<detail::ValuationState> vanilla_state_;
    mutable std::unique_ptr<PdeEngine> vanilla_engine_;
};

}  // namespace dp::engines
