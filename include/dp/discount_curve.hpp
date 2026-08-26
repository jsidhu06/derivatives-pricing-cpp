#pragma once

#include <Eigen/Core>

#include <optional>

/// Deterministic discount curve with log-linear interpolation.
namespace dp {

/// Discount curve defined by discount factors at increasing year fractions.
///
/// Interpolation is log-linear in the discount factor (piecewise-constant
/// forward rates). Outside the knot range the log-DF is held flat and a
/// warning is emitted via `dp::warn`. `df(0)` is always exactly 1.
///
/// Instances are immutable after construction.
class DiscountCurve {
public:
    /// Construct from knot times (strictly increasing year fractions) and the
    /// corresponding discount factors (all > 0). Requires at least two knots.
    /// Discount factors > 1 (negative rates) are accepted with a warning.
    DiscountCurve(Eigen::ArrayXd times, Eigen::ArrayXd dfs);

    /// Flat continuously-compounded curve on `[0, end_time]`.
    [[nodiscard]] static DiscountCurve flat(double rate, double end_time = 100.0);

    /// Curve from piecewise-constant forward rates. `times` has N+1 entries
    /// starting at 0; `forwards` has N entries, one per interval.
    [[nodiscard]] static DiscountCurve from_forwards(const Eigen::ArrayXd& times,
                                                     const Eigen::ArrayXd& forwards);

    /// Curve from continuously-compounded zero rates at each time.
    [[nodiscard]] static DiscountCurve from_zero_rates(const Eigen::ArrayXd& times,
                                                       const Eigen::ArrayXd& zero_rates);

    /// Knot times (year fractions).
    [[nodiscard]] const Eigen::ArrayXd& times() const noexcept { return times_; }
    /// Knot discount factors.
    [[nodiscard]] const Eigen::ArrayXd& dfs() const noexcept { return dfs_; }

    /// The flat continuously-compounded rate if the curve is flat, else nullopt.
    [[nodiscard]] std::optional<double> flat_rate() const;

    /// Interpolated discount factor at year fraction `t`.
    [[nodiscard]] double df(double t) const;

    /// Element-wise discount factors.
    [[nodiscard]] Eigen::ArrayXd df(const Eigen::ArrayXd& t) const;

    /// Continuously-compounded forward rate on `[t0, t1]`, i.e. the `f` with
    /// `D(t1) = D(t0) * exp(-f (t1 - t0))`. Requires `t1 > t0`.
    [[nodiscard]] double forward_rate(double t0, double t1) const;

    /// Forward rate on each consecutive interval of a strictly increasing grid.
    /// Returns `grid.size() - 1` values.
    [[nodiscard]] Eigen::ArrayXd step_forward_rates(const Eigen::ArrayXd& grid) const;

    /// New curve with every zero rate shifted by `bump`.
    [[nodiscard]] DiscountCurve bump_parallel_zero_rate(double bump) const;

private:
    /// Log-linear interpolation of log(DF), flat beyond the knots, no origin
    /// special-casing and no warnings.
    [[nodiscard]] double interp_log_df(double t) const noexcept;

    Eigen::ArrayXd times_;
    Eigen::ArrayXd dfs_;
    Eigen::ArrayXd log_dfs_;
};

}  // namespace dp
