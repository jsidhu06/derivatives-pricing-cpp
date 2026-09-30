#pragma once

#include "engines/engine.hpp"
#include "valuation_state.hpp"

#include <Eigen/Core>

#include <optional>
#include <vector>

/// Internal: Monte Carlo engine (path processes) for vanilla, custom and
/// barrier payoffs.
namespace dp::engines {

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
    /// Sorted unique grid indices at which the barrier is observed,
    /// clipped to `end_index` (every index for continuous monitoring).
    [[nodiscard]] std::vector<Eigen::Index> barrier_monitoring_indices(
        const SimulationResult& sim, Eigen::Index end_index) const;
    /// Per-path discounted European barrier values (payoff weighting plus
    /// rebate legs).
    [[nodiscard]] Eigen::ArrayXd barrier_pathwise_european() const;
    /// Per-path discounted American barrier values via barrier-state LSM.
    [[nodiscard]] Eigen::ArrayXd barrier_pathwise_american() const;
    [[nodiscard]] std::pair<double, double> risk_free_and_div_rates(double ttm) const;

    const detail::ValuationState& state_;
    mutable std::optional<SimulationResult> simulation_;
};

}  // namespace dp::engines
