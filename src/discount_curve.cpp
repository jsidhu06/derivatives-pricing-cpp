#include "dp/discount_curve.hpp"

#include "dp/diagnostics.hpp"
#include "dp/exceptions.hpp"

#include "detail/validate.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <format>
#include <optional>

namespace dp {

namespace {

constexpr double kOriginTolerance = 1e-12;
constexpr double kNegativeRateTolerance = 1e-12;

bool strictly_increasing(const Eigen::ArrayXd& values) {
    for (Eigen::Index i = 1; i < values.size(); ++i) {
        if (values[i] <= values[i - 1]) {
            return false;
        }
    }
    return true;
}

bool is_origin(double t) noexcept {
    return std::abs(t) <= kOriginTolerance;
}

}  // namespace

DiscountCurve::DiscountCurve(Eigen::ArrayXd times, Eigen::ArrayXd dfs)
    : times_(std::move(times)), dfs_(std::move(dfs)) {
    if (times_.size() != dfs_.size()) {
        throw ValidationError("times and dfs must be 1D arrays of the same length");
    }
    if (times_.size() < 2) {
        throw ValidationError("DiscountCurve requires at least 2 data points");
    }
    if (!times_.isFinite().all() || !dfs_.isFinite().all()) {
        throw ValidationError("times and dfs must be finite");
    }
    if (!strictly_increasing(times_)) {
        throw ValidationError("times must be strictly increasing");
    }
    if ((dfs_ <= 0.0).any()) {
        throw ValidationError("discount factors must be positive");
    }
    if ((dfs_ > 1.0 + kNegativeRateTolerance).any()) {
        warn("Discount factors > 1 detected (negative rates)");
    }
    // Interpolation knots anchored at (0, DF = 1): when the first knot is
    // after 0, the first zero rate applies flat back to the origin.
    if (times_[0] > 0.0) {
        knot_times_.resize(times_.size() + 1);
        knot_times_ << 0.0, times_;
        knot_log_dfs_.resize(times_.size() + 1);
        knot_log_dfs_ << 0.0, dfs_.log();
    } else {
        knot_times_ = times_;
        knot_log_dfs_ = dfs_.log();
    }
}

DiscountCurve DiscountCurve::flat(double rate, double end_time) {
    detail::require_finite(rate, "rate");
    if (!std::isfinite(end_time) || end_time <= 0.0) {
        throw ValidationError("end_time must be positive");
    }
    Eigen::ArrayXd times(2);
    times << 0.0, end_time;
    return {times, (-rate * times).exp()};
}

DiscountCurve DiscountCurve::from_forwards(const Eigen::ArrayXd& times,
                                           const Eigen::ArrayXd& forwards) {
    if (times.size() < 2) {
        throw ValidationError("times must include at least [0, T]");
    }
    if (forwards.size() != times.size() - 1) {
        throw ValidationError("forwards must have length len(times) - 1");
    }
    if (!is_origin(times[0])) {
        throw ValidationError("times must start at 0.0");
    }
    Eigen::ArrayXd dfs(times.size());
    double cum_rate = 0.0;
    dfs[0] = 1.0;
    for (Eigen::Index i = 1; i < times.size(); ++i) {
        cum_rate += forwards[i - 1] * (times[i] - times[i - 1]);
        dfs[i] = std::exp(-cum_rate);
    }
    return {times, dfs};
}

DiscountCurve DiscountCurve::from_zero_rates(const Eigen::ArrayXd& times,
                                             const Eigen::ArrayXd& zero_rates) {
    if (times.size() != zero_rates.size()) {
        throw ValidationError("times and zero_rates must have the same length");
    }
    if (times.size() < 1) {
        throw ValidationError("times must contain at least one tenor");
    }
    return {times, (-zero_rates * times).exp()};
}

std::optional<double> DiscountCurve::flat_rate() const {
    const auto last = times_.size() - 1;
    const double t_last = times_[last];
    if (t_last <= 0.0) {
        return std::nullopt;
    }
    // Infer rate from the last tenor and check every knot is consistent with it.
    const double candidate = -std::log(dfs_[last]) / t_last;
    const Eigen::ArrayXd implied = (-candidate * times_).exp();
    constexpr double kRtol = 1e-10;
    constexpr double kAtol = 1e-12;
    const bool all_close = ((dfs_ - implied).abs() <= kAtol + kRtol * implied.abs()).all();
    if (all_close) {
        return candidate;
    }
    return std::nullopt;
}

double DiscountCurve::interp_log_df(double t) const noexcept {
    const auto& x = knot_times_;
    const auto& y = knot_log_dfs_;
    const auto n = x.size();
    // Beyond either end the adjacent segment's forward rate continues.
    if (t <= x[0]) {
        const double forward = (y[0] - y[1]) / (x[1] - x[0]);
        return y[0] - forward * (t - x[0]);
    }
    if (t >= x[n - 1]) {
        const double forward = (y[n - 2] - y[n - 1]) / (x[n - 1] - x[n - 2]);
        return y[n - 1] - forward * (t - x[n - 1]);
    }
    // First knot strictly greater than t; its predecessor is the left knot.
    const auto upper = std::upper_bound(x.begin(), x.end(), t);
    const Eigen::Index i = std::distance(x.begin(), upper) - 1;
    const double w = (t - x[i]) / (x[i + 1] - x[i]);
    return y[i] + w * (y[i + 1] - y[i]);
}

double DiscountCurve::df(double t) const {
    if (is_origin(t)) {
        return 1.0;
    }
    const double t_min = knot_times_[0];
    const double t_max = knot_times_[knot_times_.size() - 1];
    if (t < t_min || t > t_max) {
        warn(std::format(
            "Extrapolating discount curve outside [{:.4f}, {:.4f}] — flat forward rate assumed",
            t_min,
            t_max));
    }
    return std::exp(interp_log_df(t));
}

Eigen::ArrayXd DiscountCurve::df(const Eigen::ArrayXd& t) const {
    const double t_min = knot_times_[0];
    const double t_max = knot_times_[knot_times_.size() - 1];

    bool any_outside = false;
    Eigen::ArrayXd out(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        const double ti = t[i];
        if (is_origin(ti)) {
            out[i] = 1.0;
            continue;
        }
        any_outside = any_outside || ti < t_min || ti > t_max;
        out[i] = std::exp(interp_log_df(ti));
    }
    if (any_outside) {
        warn(std::format(
            "Extrapolating discount curve outside [{:.4f}, {:.4f}] — flat forward rate assumed",
            t_min,
            t_max));
    }
    return out;
}

double DiscountCurve::forward_rate(double t0, double t1) const {
    if (t1 <= t0) {
        throw ValidationError("Need t1 > t0");
    }
    return (std::log(df(t0)) - std::log(df(t1))) / (t1 - t0);
}

Eigen::ArrayXd DiscountCurve::step_forward_rates(const Eigen::ArrayXd& grid) const {
    if (!strictly_increasing(grid)) {
        throw ValidationError("grid must be strictly increasing");
    }
    const Eigen::ArrayXd log_df_grid = df(grid).log();
    const auto n = grid.size();
    Eigen::ArrayXd out(std::max<Eigen::Index>(n - 1, 0));
    for (Eigen::Index i = 0; i + 1 < n; ++i) {
        out[i] = (log_df_grid[i] - log_df_grid[i + 1]) / (grid[i + 1] - grid[i]);
    }
    return out;
}

DiscountCurve DiscountCurve::bump_parallel_zero_rate(double bump) const {
    detail::require_finite(bump, "bump");
    return {times_, dfs_ * (-bump * times_).exp()};
}

}  // namespace dp
