#pragma once

#include "dp/contracts.hpp"

#include "engines/pde.hpp"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

/// Internal: PDE engine for single-barrier options.
namespace dp::engines {

/// Barrier extension of the PDE engine.
///
/// Continuous knock-outs truncate the grid at the barrier (the barrier
/// becomes a Dirichlet boundary carrying the rebate value); discrete
/// knock-outs keep the full grid, anchor the barrier half a cell between
/// nodes, and apply barrier resets at monitoring taus. American knock-ins
/// solve a two-surface coupled PDE (active/inactive); European knock-ins use
/// native-surface parity `V_KI = V_vanilla + R df_T - V_KO` on the cached
/// KO + vanilla components, with Greeks extracted per component grid.
///
/// Inception-triggered barriers short-circuit: a knock-out is its rebate, a
/// knock-in delegates to a vanilla-equivalent engine.
class PdeBarrierEngine final : public PdeEngine {
public:
    /// Requires a `BarrierSpec` in `state`.
    explicit PdeBarrierEngine(const detail::ValuationState& state);

    [[nodiscard]] double present_value() const override;
    [[nodiscard]] double delta() const override;
    [[nodiscard]] double gamma() const override;
    /// Per calendar day.
    [[nodiscard]] double theta() const override;

private:
    [[nodiscard]] Solution compute_solution() const override;

    [[nodiscard]] const BarrierSpec& spec() const;
    [[nodiscard]] bool is_european_knock_in() const;
    /// Monitoring dates converted to tau (time remaining), sorted ascending.
    [[nodiscard]] std::vector<double> barrier_monitoring_taus() const;
    /// Far-field `(smin, smax)` targets for the barrier grids: the inherited
    /// domain of a bumped valuation, else `pde::barrier_domain`. A continuous
    /// barrier replaces its side.
    [[nodiscard]] std::pair<double, double> far_field_domain() const override;
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

    mutable std::optional<std::pair<Solution, Solution>> ki_components_;
    mutable std::optional<detail::ValuationState> vanilla_state_;
    mutable std::unique_ptr<PdeEngine> vanilla_engine_;
};

}  // namespace dp::engines
