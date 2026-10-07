#pragma once

#include "engines/engine.hpp"
#include "valuation_state.hpp"

#include <Eigen/Core>

#include <optional>
#include <vector>

/// Internal: Cox-Ross-Rubinstein binomial-tree engine for vanilla, custom
/// and barrier payoffs, European and American exercise.
namespace dp::engines {

/// CRR tree with per-step forward rates from the discount/dividend curves and
/// escrowed-dividend handling of discrete cash dividends. Native tree Greeks
/// (delta, gamma, theta) follow Hull Ch. 21.
///
/// Extension: subclasses replace the rollback through `compute_solution()`
/// and the step count through `num_steps()`, and may override the public
/// entry points for payoff-specific special cases, falling back to the base
/// implementations (`BinomialBarrierEngine`).
///
/// Caching: the tree setup and the backward-induction lattice are computed
/// once and stored in `mutable` members. Every public entry point is reached
/// through the facade's memoisation lock (one recursive mutex per
/// `OptionValuation`), so this engine performs no locking of its own.
class BinomialEngine : public PricingEngine {
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
    /// Steps use the forward rates, so this is the decay with the calendar
    /// curve held fixed (see `OptionValuation::theta`).
    [[nodiscard]] double theta() const override;

protected:
    /// Deterministic tree inputs, computed once. O(n) storage.
    struct Setup {
        int num_steps;                    ///< Tree steps n (cached from `num_steps()`).
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
        double f01, f11, s01, s11;  ///< Step-1 nodes (delta).
        double f02, f12, f22;       ///< Step-2 values (gamma, theta).
        double s02, s12, s22;       ///< Step-2 spots (gamma).
    };

    /// Tree step count hook: the configured base here; subclasses may align
    /// it to payoff features (barriers).
    [[nodiscard]] virtual int num_steps() const;
    /// Rollback hook: the vanilla/custom-payoff solve here; subclasses
    /// replace it. Reached only through the caching `solution()`.
    [[nodiscard]] virtual Solution compute_solution() const;

    [[nodiscard]] int base_num_steps() const;
    [[nodiscard]] bool early_exercise() const;
    [[nodiscard]] const Setup& setup() const;
    /// Cached `compute_solution()`.
    [[nodiscard]] const Solution& solution() const;
    [[nodiscard]] static Solution nan_solution();
    /// Record the step-1 / step-2 option values and spots the tree Greeks
    /// need, as the rollback passes through step `t`.
    static void capture_step_nodes(Solution& out,
                                   const Setup& s,
                                   const std::vector<double>& values,
                                   int t);
    /// Spot at node (down-moves `i`, step `t`), matching the reference
    /// lattice bit-for-bit: `escrowed_spot * u^(t-2i) + pv_remaining[t]`.
    [[nodiscard]] static double spot_at(const Setup& s, int i, int t);

    const detail::ValuationState& state_;

private:
    mutable std::optional<Setup> setup_;
    mutable std::optional<Solution> solution_;
};

/// Barrier extension of the binomial engine.
///
/// Continuous monitoring uses Boyle-Lau (1994) step alignment to place a
/// tree layer on the barrier; discrete monitoring uses half-step placement
/// (Cheuk-Vorst / Boyle-Tian analog) and applies the barrier only at
/// monitoring steps. Knock-outs solve one barrier-aware rollback; knock-ins
/// roll an explicit active/inactive pair so American knock-ins and KI
/// rebates need no parity shortcut.
///
/// Inception-triggered barriers short-circuit: a knock-out is its rebate
/// lattice, a knock-in the vanilla rollback. Delta and gamma refuse when
/// their extraction stencil straddles a continuously monitored barrier.
class BinomialBarrierEngine final : public BinomialEngine {
public:
    /// Requires a `BarrierSpec` in `state`.
    explicit BinomialBarrierEngine(const detail::ValuationState& state);

    [[nodiscard]] double present_value() const override;
    [[nodiscard]] double delta() const override;
    [[nodiscard]] double gamma() const override;

private:
    /// Boyle-Lau-aligned step count (cached).
    [[nodiscard]] int num_steps() const override;
    /// KO lattice, or the KI inactive-state lattice (the contract's state
    /// at the root); triggered specs collapse as described above.
    [[nodiscard]] Solution compute_solution() const override;

    [[nodiscard]] const BarrierSpec& spec() const;
    [[nodiscard]] int resolve_barrier_num_steps() const;
    /// True at every step where the barrier is observed (index 0 never is).
    [[nodiscard]] std::vector<char> monitoring_step_mask(int num_steps) const;
    /// Value a knocked-out node takes at each step: 0 without a rebate, the
    /// rebate for AT_HIT, its discounted-to-step value for AT_EXPIRY.
    [[nodiscard]] Eigen::ArrayXd ko_rebate_values(const Setup& s) const;
    [[nodiscard]] Solution solve_knock_out() const;
    [[nodiscard]] Solution solve_knock_in() const;
    /// Deterministic rebate lattice for a KO already dead at inception.
    [[nodiscard]] Solution resolved_knock_out_solution() const;
    /// Guard for Hull tree greeks: true when a node in the step-`step`
    /// extraction stencil lies at or past a continuously monitored barrier.
    [[nodiscard]] bool stencil_straddles_barrier(int step) const;

    mutable std::optional<int> barrier_steps_;
};

}  // namespace dp::engines
