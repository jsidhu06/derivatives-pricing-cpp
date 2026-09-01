#pragma once

#include "engines/engine.hpp"
#include "valuation_state.hpp"

#include <Eigen/Core>

#include <optional>

/// Internal: Cox-Ross-Rubinstein binomial-tree engine for vanilla and custom
/// payoffs, European and American exercise.
namespace dp::engines {

/// CRR tree with per-step forward rates from the discount/dividend curves and
/// escrowed-dividend handling of discrete cash dividends. Native tree Greeks
/// (delta, gamma, theta) follow Hull Ch. 21.
///
/// Caching: the tree setup and the backward-induction lattice are computed
/// once and stored in `mutable` members. Every public entry point is reached
/// through the facade's memoisation lock (one recursive mutex per
/// `OptionValuation`), so this engine performs no locking of its own.
class BinomialEngine final : public PricingEngine {
public:
    explicit BinomialEngine(const detail::ValuationState& state);

    [[nodiscard]] std::string_view name() const noexcept override { return "BINOMIAL"; }
    [[nodiscard]] double present_value() const override;
    /// Hull: (f_u - f_d) / (S_u - S_d) at step 1.
    [[nodiscard]] double delta() const override;
    /// Hull: difference of step-2 deltas over half the step-2 spot spread.
    /// Requires `num_steps >= 2`.
    [[nodiscard]] double gamma() const override;
    /// Hull: (f_ud - f_0) / (2 dt), per calendar day. Requires `num_steps >= 2`.
    [[nodiscard]] double theta() const override;

private:
    /// Deterministic tree inputs, computed once.
    struct Setup {
        double delta_t;                   ///< Uniform step in year fractions.
        Eigen::ArrayXd discount_factors;  ///< Per-step DFs, length n.
        Eigen::ArrayXd p;                 ///< Per-step risk-neutral up probabilities.
        Eigen::MatrixXd spot_lattice;     ///< (n+1, n+1); row = down moves, col = time.
    };

    [[nodiscard]] int num_steps() const;
    [[nodiscard]] bool early_exercise() const;
    [[nodiscard]] const Setup& setup() const;
    /// Backward-induction value lattice, (n+1, n+1).
    [[nodiscard]] const Eigen::MatrixXd& option_lattice() const;
    /// Payoff at every lattice node (vanilla intrinsic or PayoffSpec).
    [[nodiscard]] Eigen::MatrixXd intrinsic_values(const Eigen::MatrixXd& spots) const;

    const detail::ValuationState& state_;
    mutable std::optional<Setup> setup_;
    mutable std::optional<Eigen::MatrixXd> option_lattice_;
};

}  // namespace dp::engines
