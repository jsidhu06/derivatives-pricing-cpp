#pragma once

#include "engines/engine.hpp"
#include "valuation_state.hpp"

#include <Eigen/Core>

#include <optional>
#include <vector>

/// Internal: Monte Carlo engine (path processes) for vanilla, custom and
/// barrier payoffs.
namespace dp::engines {

using BoolArray = Eigen::Array<bool, Eigen::Dynamic, 1>;
using IndexArray = Eigen::Array<Eigen::Index, Eigen::Dynamic, 1>;

/// European pricing by discounted terminal payoff; American pricing by
/// Longstaff-Schwartz regression on a ridge-regularised Laguerre basis.
/// Pathwise and likelihood-ratio Greeks for European vanillas.
///
/// Barriers: discrete monitoring by pathwise checks at monitoring dates
/// (any process); continuous monitoring GBM-only via Brownian-bridge
/// crossing probabilities (survival weighting for Europeans, Bernoulli
/// bridge sampling inside the American LSM). American knock-outs can use a
/// barrier-aware LSM basis (`MonteCarloParams.barrier_aware_basis`).
///
/// Caching: one simulation per engine instance (seeded from
/// `MonteCarloParams.random_seed`), reused by PV and every estimator — this
/// is what makes bump-and-revalue Greeks common-random-number by default.
/// All entry points run under the facade's per-instance lock.
class McEngine final : public PricingEngine {
public:
    explicit McEngine(const detail::ValuationState& state);

    [[nodiscard]] std::string_view name() const noexcept override { return "MONTE_CARLO"; }
    [[nodiscard]] double present_value() const override;

    [[nodiscard]] double delta_pathwise() const override;
    [[nodiscard]] double vega_pathwise() const override;
    [[nodiscard]] double theta_pathwise() const override;
    [[nodiscard]] double rho_pathwise() const override;
    [[nodiscard]] double delta_lr() const override;
    [[nodiscard]] double vega_lr() const override;
    [[nodiscard]] double theta_lr() const override;
    [[nodiscard]] double rho_lr() const override;
    [[nodiscard]] double gamma_pathwise_fd(std::optional<double> epsilon) const override;

private:
    struct Terminal {
        Eigen::ArrayXd spot;  ///< S_T per path.
        Eigen::Index index;   ///< Grid index of maturity.
        double ttm;
        double df;
    };

    [[nodiscard]] const SimulationResult& simulation() const;
    [[nodiscard]] Terminal terminal() const;
    /// Effective terminal N(0,1): sum(sqrt(dt_i) Z_i) / sqrt(T).
    [[nodiscard]] Eigen::ArrayXd effective_terminal_z(Eigen::Index maturity_index,
                                                      double ttm) const;
    [[nodiscard]] Eigen::ArrayXd payoff_at(const Eigen::ArrayXd& spots) const;
    [[nodiscard]] Eigen::ArrayXd present_value_pathwise() const;

    // ── Barrier machinery ───────────────────────────────────────────
    /// European barrier legs per path: the weight applied to the discounted
    /// vanilla payoff (0/1 for discrete, survival probability for
    /// continuous) and the already-discounted rebate PV.
    struct BarrierWeights {
        Eigen::ArrayXd weight;
        Eigen::ArrayXd rebate_pv;
    };

    /// American barrier state from the forward pass: cumulative hit flags
    /// per exercise date (index 0 = pricing date) and each path's first hit
    /// step within that window (-1 if never hit).
    struct BarrierPathState {
        std::vector<BoolArray> ever_hit;
        IndexArray first_hit_step;
    };

    /// Sorted unique grid indices at which the barrier is observed,
    /// clipped to `end_index` (every index for continuous monitoring).
    [[nodiscard]] std::vector<Eigen::Index> barrier_monitoring_indices(
        const SimulationResult& sim, Eigen::Index end_index) const;

    /// Per-path discounted European barrier values (payoff weighting plus
    /// rebate legs).
    [[nodiscard]] Eigen::ArrayXd barrier_pathwise_european() const;
    /// Continuous monitoring: Brownian-bridge survival weighting.
    [[nodiscard]] BarrierWeights continuous_weights(const SimulationResult& sim,
                                                    Eigen::Index end,
                                                    const Eigen::ArrayXd& grid_dfs) const;
    /// Discrete monitoring: pathwise barrier checks at monitoring dates.
    [[nodiscard]] BarrierWeights discrete_weights(const SimulationResult& sim,
                                                  Eigen::Index end,
                                                  const Eigen::ArrayXd& grid_dfs) const;

    /// Per-path discounted American barrier values via barrier-state LSM.
    [[nodiscard]] Eigen::ArrayXd barrier_pathwise_american() const;
    /// Forward pass over the exercise window [start, start + n_times):
    /// pathwise checks (discrete) or Bernoulli bridge sampling (continuous).
    [[nodiscard]] BarrierPathState american_barrier_state(const SimulationResult& sim,
                                                          Eigen::Index start,
                                                          Eigen::Index n_times) const;
    /// Terminal path values before backward induction.
    [[nodiscard]] Eigen::ArrayXd initial_barrier_values(const Eigen::ArrayXd& intrinsic_end,
                                                        const BoolArray& hit_end) const;
    /// One backward-induction step for an American knock-out: regress on
    /// alive + ITM paths; knocked-out paths carry their rebate leg.
    [[nodiscard]] Eigen::ArrayXd knock_out_step_values(Eigen::Index t,
                                                       const Eigen::ArrayXd& spots,
                                                       const Eigen::ArrayXd& intrinsic,
                                                       const Eigen::ArrayXd& discounted_next,
                                                       const BarrierPathState& state,
                                                       const Eigen::ArrayXd& dfs) const;
    /// One backward-induction step for an American knock-in: regress on
    /// active + ITM paths; inactive paths carry the discounted continuation
    /// with no exercise right.
    [[nodiscard]] Eigen::ArrayXd knock_in_step_values(const Eigen::ArrayXd& spots,
                                                      const Eigen::ArrayXd& intrinsic,
                                                      const Eigen::ArrayXd& discounted_next,
                                                      const BoolArray& hit_t) const;
    [[nodiscard]] std::pair<double, double> risk_free_and_div_rates(double ttm) const;

    const detail::ValuationState& state_;
    mutable std::optional<SimulationResult> simulation_;
};

}  // namespace dp::engines
