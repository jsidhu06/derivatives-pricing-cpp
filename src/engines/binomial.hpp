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
    /// Deterministic tree inputs, computed once. O(n) storage.
    struct Setup {
        double delta_t;                   ///< Uniform step in year fractions.
        Eigen::ArrayXd discount_factors;  ///< Per-step DFs, length n.
        Eigen::ArrayXd p;                 ///< Per-step risk-neutral up probabilities.
        Eigen::ArrayXd up_powers;         ///< u^(k-n) for k = 0..2n.
        Eigen::ArrayXd pv_remaining;      ///< Escrow add-back per date (empty when no dividends).
        double escrowed_spot;             ///< Spot minus PV of all dividends.
    };

    /// Backward-induction results: the PV plus the handful of early-time
    /// nodes required by the Hull tree Greeks (f = option value, s = spot;
    /// indices are (down-moves, time)). Step-2 fields are NaN when n < 2.
    struct Solution {
        double pv;
        double f01, f11, s01, s11;       ///< Step-1 nodes (delta).
        double f02, f12, f22;            ///< Step-2 values (gamma, theta).
        double s02, s12, s22;            ///< Step-2 spots (gamma).
    };

    [[nodiscard]] int num_steps() const;
    [[nodiscard]] bool early_exercise() const;
    [[nodiscard]] const Setup& setup() const;
    [[nodiscard]] const Solution& solution() const;
    /// Spot at node (down-moves `i`, step `t`), matching the reference
    /// lattice bit-for-bit: `escrowed_spot * u^(t-2i) + pv_remaining[t]`.
    [[nodiscard]] double spot_at(const Setup& s, int i, int t) const;

    const detail::ValuationState& state_;
    mutable std::optional<Setup> setup_;
    mutable std::optional<Solution> solution_;
};

}  // namespace dp::engines
