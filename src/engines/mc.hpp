#pragma once

#include "engines/engine.hpp"
#include "valuation_state.hpp"

#include <Eigen/Core>

#include <optional>

/// Internal: Monte Carlo engine (GBM paths) for vanilla and custom payoffs.
namespace dp::engines {

/// European pricing by discounted terminal payoff; American pricing by
/// Longstaff-Schwartz regression on a ridge-regularised Laguerre basis.
/// Pathwise and likelihood-ratio Greeks for European vanillas.
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
    [[nodiscard]] std::pair<double, double> risk_free_and_div_rates(double ttm) const;

    const detail::ValuationState& state_;
    mutable std::optional<SimulationResult> simulation_;
};

}  // namespace dp::engines
