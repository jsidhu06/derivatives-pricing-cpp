#include "engines/pde.hpp"

#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/diagnostics.hpp"
#include "dp/exceptions.hpp"
#include "dp/params.hpp"

#include "detail/barrier.hpp"
#include "detail/log.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <variant>
#include <vector>

namespace dp::engines {

namespace {

using Eigen::ArrayXd;
using Eigen::Index;

constexpr double kDaysPerYear = 365.0;
constexpr double kTauResolution = 1e12;  // taus are matched at 12 decimal places

/// Round to 12 decimal places, ties to even (matches `np.round(x, 12)`),
/// so taus from different arithmetic paths compare equal.
double round12(double x) {
    return std::nearbyint(x * kTauResolution) / kTauResolution;
}

/// `np.linspace(start, stop, n + 1)`: `start + i * step` with the endpoint
/// pinned exactly.
ArrayXd linspace(double start, double stop, int intervals) {
    const double step = (stop - start) / intervals;
    ArrayXd out(intervals + 1);
    for (int i = 0; i <= intervals; ++i) {
        out[i] = start + step * i;
    }
    out[intervals] = stop;
    return out;
}

/// `np.interp(x, xs, ys)` for one query point: linear interpolation with the
/// end values held flat outside the range.
double interp_linear(double x, const ArrayXd& xs, const ArrayXd& ys) {
    const Index n = xs.size();
    if (x <= xs[0]) {
        return ys[0];
    }
    if (x >= xs[n - 1]) {
        return ys[n - 1];
    }
    const auto hi = std::distance(xs.begin(), std::upper_bound(xs.begin(), xs.end(), x));
    const Index lo = hi - 1;
    const double slope = (ys[hi] - ys[lo]) / (xs[hi] - xs[lo]);
    return slope * (x - xs[lo]) + ys[lo];
}

/// Thomas algorithm for a tridiagonal system; diagonals are named as in the
/// reference implementation (lower/diag/upper of lengths n-1, n, n-1).
ArrayXd solve_tridiagonal_thomas(const ArrayXd& lower,
                                 ArrayXd diag,
                                 const ArrayXd& upper,
                                 ArrayXd rhs) {
    const Index n = diag.size();
    if (rhs.size() != n) {
        throw ValidationError("rhs length must match diag length");
    }
    if (lower.size() != n - 1 || upper.size() != n - 1) {
        throw ValidationError("lower/upper must have length n-1");
    }
    for (Index i = 1; i < n; ++i) {
        const double w = lower[i - 1] / diag[i - 1];
        diag[i] -= w * upper[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    ArrayXd x(n);
    x[n - 1] = rhs[n - 1] / diag[n - 1];
    for (Index i = n - 2; i >= 0; --i) {
        x[i] = (rhs[i] - upper[i] * x[i + 1]) / diag[i];
    }
    return x;
}

/// Tau grid snapping extra times (dividends) onto the uniform base grid.
ArrayXd build_tau_grid(double time_to_maturity, int time_steps, const std::vector<double>& extras) {
    ArrayXd base = linspace(0.0, time_to_maturity, time_steps);
    if (extras.empty()) {
        return base;
    }
    std::vector<double> merged;
    merged.reserve(static_cast<std::size_t>(base.size()) + extras.size());
    for (const double tau : base) {
        merged.push_back(round12(tau));
    }
    for (const double tau : extras) {
        merged.push_back(round12(tau));
    }
    std::sort(merged.begin(), merged.end());
    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
    return Eigen::Map<const ArrayXd>(merged.data(), static_cast<Index>(merged.size()));
}

/// (tau, amount) for dividends with pricing_date <= ex_date <= maturity,
/// amounts merged per rounded tau, sorted ascending.
std::vector<std::pair<double, double>> dividend_tau_schedule(const detail::ValuationState& state) {
    const auto& dividends = state.underlying.discrete_dividends();
    if (dividends.empty()) {
        return {};
    }
    const double ttm = state.maturity_year_fraction();
    std::map<double, double> schedule;
    for (const auto& [ex_date, amount] : dividends) {
        if (state.pricing_date() <= ex_date && ex_date <= state.maturity()) {
            const double t =
                year_fraction(state.pricing_date(), ex_date, state.day_count_convention());
            schedule[round12(ttm - t)] += amount;
        }
    }
    return {schedule.begin(), schedule.end()};
}

/// Cash dividend jump condition V(S, t^-) = V(S - D, t^+).
void apply_dividend_jump(ArrayXd& values,
                         const ArrayXd& grid,
                         double amount,
                         PdeSpaceGrid space_grid) {
    if (amount == 0.0) {
        return;
    }
    const ArrayXd spot_grid = space_grid == PdeSpaceGrid::LogSpot ? grid.exp().eval() : grid;
    ArrayXd shifted(values.size());
    for (Index i = 0; i < values.size(); ++i) {
        shifted[i] = interp_linear(spot_grid[i] - amount, spot_grid, values);
    }
    values = std::move(shifted);
}

// ── Affine wing boundary models (custom payoffs) ────────────────────

WingBoundary fit_affine_boundary_model(const PayoffSpec& payoff,
                                       std::string_view wing,
                                       const ArrayXd& spot_samples) {
    const Index n = spot_samples.size();
    if (n < 3) {
        throw ConfigurationError("spot_samples must be a 1D array with at least three points");
    }
    for (Index i = 1; i < n; ++i) {
        if (spot_samples[i] <= spot_samples[i - 1]) {
            throw ConfigurationError("spot_samples must be strictly increasing");
        }
    }
    Eigen::VectorXd y(n);
    for (Index i = 0; i < n; ++i) {
        y[i] = payoff.payoff(spot_samples[i]);
    }
    Eigen::MatrixXd A(n, 2);
    A.col(0) = spot_samples.matrix();
    A.col(1).setOnes();
    const Eigen::Vector2d fit = A.colPivHouseholderQr().solve(y);
    const double slope = fit[0];
    const double intercept = fit[1];

    const Eigen::VectorXd residuals = y - (A * fit);
    const double ss_res = residuals.squaredNorm();
    const Eigen::VectorXd centred = y.array() - y.mean();
    const double ss_tot = centred.squaredNorm();
    if (ss_tot > 1e-30) {
        const double r_squared = 1.0 - ss_res / ss_tot;
        if (r_squared < 0.99) {
            warn(
                std::format("Affine boundary fit on {} wing has R²={:.4f}; boundary values may be "
                            "inaccurate. Consider providing explicit PayoffBoundaryModel.",
                            wing,
                            r_squared));
        }
    }
    return {.slope = slope, .intercept = intercept};
}

/// V(S, t) ~ slope * S * dq_fwd + intercept * df_fwd under risk-neutral pricing.
double continuation_from_affine(double spot,
                                const WingBoundary& wing,
                                double df_fwd,
                                double dq_fwd) {
    return wing.slope * spot * dq_fwd + wing.intercept * df_fwd;
}

/// Dirichlet boundary values at S = smin (left) and S = smax (right).
std::pair<double, double> boundary_values(std::optional<OptionType> option_type,
                                          std::optional<double> strike,
                                          double smin,
                                          double smax,
                                          double df_fwd,
                                          double dq_fwd,
                                          bool early_exercise,
                                          const PayoffSpec* payoff,
                                          const std::optional<PayoffBoundaryModel>& model) {
    if (payoff != nullptr) {
        if (!model.has_value()) {
            throw ConfigurationError(
                "boundary values require a resolved payoff boundary model for custom payoffs");
        }
        double left = continuation_from_affine(smin, model->left, df_fwd, dq_fwd);
        double right = continuation_from_affine(smax, model->right, df_fwd, dq_fwd);
        if (early_exercise) {
            left = std::max(left, payoff->payoff(smin));
            right = std::max(right, payoff->payoff(smax));
        }
        return {left, right};
    }

    if (!strike.has_value()) {
        throw ConfigurationError("vanilla PDE boundary values require a strike");
    }
    const double k = *strike;
    if (option_type == OptionType::Put) {
        const double intrinsic = std::max(k - smin, 0.0);
        const double continuation = k * df_fwd - smin * dq_fwd;
        const double left = early_exercise ? std::max(continuation, intrinsic) : continuation;
        return {left, 0.0};
    }
    const double continuation = smax * dq_fwd - k * df_fwd;
    const double intrinsic = std::max(smax - k, 0.0);
    const double right =
        early_exercise ? std::max(continuation, intrinsic) : std::max(continuation, 0.0);
    return {0.0, right};
}

// ── Grids ───────────────────────────────────────────────────────────

struct SpaceGridBuild {
    ArrayXd grid;     ///< PDE coordinate (spot, or log-spot z).
    ArrayXd spots;    ///< exp(grid) on log grids; == grid on spot grids.
    double dx = 0.0;  ///< dS or dz.
};

/// Log-spot grid. Explicit schemes target Hull's stability spacing
/// dz = vol * sqrt(3 dt) when it covers the domain. When `anchor_spot` is
/// given the grid is sized so the anchor sits exactly on an interior node
/// (or halfway between two nodes with `anchor_half_step`): CN/IMPLICIT grow
/// dz on the binding half, explicit schemes keep dz fixed by stability and
/// shift the grid in place instead.
SpaceGridBuild build_log_grid(double spot,
                              double strike,
                              double time_to_maturity,
                              double volatility,
                              double smax_mult,
                              int spot_steps,
                              int time_steps,
                              PdeMethod method,
                              std::optional<double> anchor_spot = std::nullopt,
                              bool anchor_half_step = false) {
    if (anchor_half_step && !anchor_spot.has_value()) {
        throw ValidationError("anchor_half_step requires anchor_spot to be provided");
    }
    const double smax = smax_mult * std::max(spot, strike);
    const double smin = std::max(std::max(spot, strike) / smax_mult, 1.0e-8);
    const double zmin_target = std::log(smin);
    const double zmax_target = std::log(smax);

    const bool explicit_scheme = method == PdeMethod::Explicit || method == PdeMethod::ExplicitHull;

    double dz = 0.0;
    double zmin = zmin_target;
    if (explicit_scheme) {
        const double d_tau = time_to_maturity / time_steps;
        const double dz_hull = volatility * std::sqrt(3.0 * d_tau);
        const double grid_width = spot_steps * dz_hull;
        if ((zmax_target - zmin_target) > grid_width) {
            dz = (zmax_target - zmin_target) / spot_steps;
        } else {
            dz = dz_hull;
            const double center = std::log(spot);
            zmin = center - 0.5 * grid_width;
            double zmax = center + 0.5 * grid_width;
            if (zmin > zmin_target) {
                const double shift = zmin_target - zmin;
                zmin += shift;
                zmax += shift;
            }
            if (zmax < zmax_target) {
                zmin += zmax_target - zmax;
            }
        }
    } else {
        dz = (zmax_target - zmin_target) / spot_steps;
    }

    const auto make_grid = [&](double origin, double step) {
        ArrayXd z(spot_steps + 1);
        for (int i = 0; i <= spot_steps; ++i) {
            z[i] = origin + step * i;
        }
        return z;
    };

    if (!anchor_spot.has_value()) {
        const ArrayXd z = make_grid(zmin, dz);
        return {.grid = z, .spots = z.exp(), .dx = dz};
    }

    if (*anchor_spot <= 0.0) {
        throw ValidationError("anchor_spot must be positive for log-spot grids");
    }
    const double z_anchor = std::log(*anchor_spot);
    if (z_anchor < zmin_target || z_anchor > zmax_target) {
        throw ValidationError("anchor_spot must lie within the log-grid target domain");
    }
    const double anchor_offset = anchor_half_step ? 0.5 : 0.0;

    int j_anchor = 0;
    if (explicit_scheme) {
        // dz fixed by Hull's heuristic: shift the grid in place while
        // preserving cover of the target span.
        const int j_min = std::max(
            0, static_cast<int>(std::ceil((z_anchor - zmin_target) / dz - anchor_offset - 1e-12)));
        const int j_max =
            std::min(anchor_half_step ? spot_steps - 1 : spot_steps,
                     static_cast<int>(std::floor(spot_steps - (zmax_target - z_anchor) / dz -
                                                 anchor_offset + 1e-12)));
        if (j_min > j_max) {
            throw StabilityError("Unable to align anchor_spot on the log grid with current setup");
        }
        const auto preferred = static_cast<int>(std::round((z_anchor - zmin) / dz - anchor_offset));
        j_anchor = std::min(std::max(preferred, j_min), j_max);
    } else {
        // CN/IMPLICIT: dz is free — grow it on the binding half of the anchor.
        const double span = zmax_target - zmin_target;
        const auto j_opt = static_cast<int>(
            std::round(spot_steps * (z_anchor - zmin_target) / span - anchor_offset));
        j_anchor = std::max(anchor_half_step ? 0 : 1, std::min(spot_steps - 1, j_opt));
        const double left_dz = (z_anchor - zmin_target) / (j_anchor + anchor_offset);
        const double right_dz = (zmax_target - z_anchor) / (spot_steps - j_anchor - anchor_offset);
        dz = std::max(left_dz, right_dz);
    }

    zmin = z_anchor - (j_anchor + anchor_offset) * dz;
    ArrayXd z = make_grid(zmin, dz);
    if (!anchor_half_step) {
        z[j_anchor] = z_anchor;
    }
    return {.grid = z, .spots = z.exp(), .dx = dz};
}

/// Uniform spot grid, optionally aligning an anchor to a node or midpoint.
SpaceGridBuild build_spot_grid(double smin,
                               double smax,
                               int spot_steps,
                               std::optional<double> anchor_spot = std::nullopt,
                               bool anchor_half_step = false) {
    if (!anchor_spot.has_value()) {
        const ArrayXd grid = linspace(smin, smax, spot_steps);
        return {.grid = grid, .spots = grid, .dx = (smax - smin) / spot_steps};
    }
    if (smin >= *anchor_spot || *anchor_spot >= smax) {
        throw ValidationError("anchor_spot must lie strictly inside the spot-grid domain");
    }
    const double ratio = (*anchor_spot - smin) / (smax - smin);

    const auto make_grid = [&](double step) {
        ArrayXd grid(spot_steps + 1);
        for (int i = 0; i <= spot_steps; ++i) {
            grid[i] = smin + step * i;
        }
        return grid;
    };

    if (anchor_half_step) {
        const int j_max = std::min(spot_steps - 1,
                                   static_cast<int>(std::floor(spot_steps * ratio - 0.5 + 1e-12)));
        if (j_max < 0) {
            throw StabilityError("Unable to align anchor_spot on the spot grid with current setup");
        }
        const auto preferred = static_cast<int>(std::round(spot_steps * ratio - 0.5));
        const int j_anchor = std::min(std::max(preferred, 0), j_max);
        const double d_spot = (*anchor_spot - smin) / (j_anchor + 0.5);
        const ArrayXd grid = make_grid(d_spot);
        return {.grid = grid, .spots = grid, .dx = d_spot};
    }

    const int j_max =
        std::min(spot_steps - 1, static_cast<int>(std::floor(spot_steps * ratio + 1e-12)));
    if (j_max < 1) {
        throw StabilityError("Unable to align anchor_spot on the spot grid with current setup");
    }
    const auto preferred = static_cast<int>(std::round(spot_steps * ratio));
    const int j_anchor = std::min(std::max(preferred, 1), j_max);
    const double d_spot = (*anchor_spot - smin) / j_anchor;
    ArrayXd grid = make_grid(d_spot);
    grid[j_anchor] = *anchor_spot;
    return {.grid = grid, .spots = grid, .dx = d_spot};
}

/// Log-spot grid for a continuous knock-out: the barrier is a boundary node.
/// Explicit-family schemes preserve Hull's dz scale; CN/IMPLICIT honor
/// spot_steps directly.
SpaceGridBuild build_ko_continuous_log_grid(double smin_target,
                                            double smax_target,
                                            double ref_price,
                                            double smax_mult,
                                            BarrierDirection direction,
                                            double volatility,
                                            double time_to_maturity,
                                            int spot_steps,
                                            int time_steps,
                                            PdeMethod method) {
    const bool explicit_scheme = method == PdeMethod::Explicit || method == PdeMethod::ExplicitHull;
    double zlo = 0.0;
    double zhi = 0.0;
    if (direction == BarrierDirection::Down) {
        zlo = std::log(smin_target);
        zhi = std::log(smax_target);
        if (explicit_scheme) {
            const double dz_hull = volatility * std::sqrt(3.0 * (time_to_maturity / time_steps));
            const double grid_width = spot_steps * dz_hull;
            if ((zhi - zlo) <= grid_width) {
                zhi = zlo + grid_width;
            }
        }
    } else {
        zhi = std::log(smax_target);
        zlo = std::log(std::max(ref_price / smax_mult, 1.0e-8));
        if (explicit_scheme) {
            const double dz_hull = volatility * std::sqrt(3.0 * (time_to_maturity / time_steps));
            const double grid_width = spot_steps * dz_hull;
            if ((zhi - zlo) <= grid_width) {
                zlo = zhi - grid_width;
            }
        }
    }
    const ArrayXd z = linspace(zlo, zhi, spot_steps);
    return {.grid = z, .spots = z.exp(), .dx = (zhi - zlo) / spot_steps};
}

// ── Spatial operator coefficients ───────────────────────────────────

struct OperatorCoeffs {
    ArrayXd gamma;  ///< Sub-diagonal weight.
    ArrayXd beta;   ///< Diagonal weight.
    ArrayXd alpha;  ///< Super-diagonal weight.
};

/// `implicit_discounting` (Hull's explicit scheme) excludes the rV term from
/// beta; it is applied as a 1/(1 + r dt) divisor in the step instead.
OperatorCoeffs spot_operator_coeffs(const ArrayXd& interior_spots,
                                    double dS,
                                    double r,
                                    double q,
                                    double volatility,
                                    bool implicit_discounting) {
    const ArrayXd diffusion = (volatility * volatility) * interior_spots.square() / (dS * dS);
    const ArrayXd drift = (r - q) * interior_spots / dS;
    return {
        .gamma = 0.5 * (diffusion - drift),
        .beta = implicit_discounting ? (-diffusion).eval() : (-(diffusion + r)).eval(),
        .alpha = 0.5 * (diffusion + drift),
    };
}

OperatorCoeffs log_operator_coeffs(
    double dz, double r, double q, double volatility, bool implicit_discounting, Index size) {
    const double mu = r - q - 0.5 * volatility * volatility;
    const double diffusion = (volatility * volatility) / (dz * dz);
    const double drift = mu / dz;
    return {
        .gamma = ArrayXd::Constant(size, 0.5 * (diffusion - drift)),
        .beta = ArrayXd::Constant(size, implicit_discounting ? -diffusion : -(diffusion + r)),
        .alpha = ArrayXd::Constant(size, 0.5 * (diffusion + drift)),
    };
}

// ── Time steps ──────────────────────────────────────────────────────

struct TimeStep {
    double tau_start;
    double tau_end;
    PdeMethod method;
};

/// Rannacher smoothing: the first `rannacher_steps` CN intervals become two
/// implicit half-steps each (Pooley-Vetzal-Forsyth 2003).
std::vector<TimeStep> build_time_step_schedule(const ArrayXd& tau_grid,
                                               PdeMethod method,
                                               int rannacher_steps) {
    std::vector<TimeStep> steps;
    steps.reserve(static_cast<std::size_t>(tau_grid.size()));
    for (Index n = 1; n < tau_grid.size(); ++n) {
        const double tau_start = tau_grid[n - 1];
        const double tau_end = tau_grid[n];
        if (method == PdeMethod::CrankNicolson && rannacher_steps > 0 && n <= rannacher_steps) {
            const double tau_mid = 0.5 * (tau_start + tau_end);
            steps.push_back({tau_start, tau_mid, PdeMethod::Implicit});
            steps.push_back({tau_mid, tau_end, PdeMethod::Implicit});
        } else {
            steps.push_back({tau_start, tau_end, method});
        }
    }
    return steps;
}

void check_explicit_spot_stability(const ArrayXd& tau_grid,
                                   double volatility,
                                   double smax,
                                   double dS,
                                   double time_to_maturity,
                                   const DiscountCurve& discount_curve,
                                   const std::optional<DiscountCurve>& dividend_curve,
                                   bool implicit_discounting) {
    if (tau_grid.size() < 2) {
        return;
    }
    const Index n = tau_grid.size() - 1;
    double max_dt = 0.0;
    double r_max = 0.0;
    double rq_abs_max = 0.0;
    for (Index i = 0; i < n; ++i) {
        max_dt = std::max(max_dt, tau_grid[i + 1] - tau_grid[i]);
        const double t_next = time_to_maturity - tau_grid[i + 1];
        const double t_prev = time_to_maturity - tau_grid[i];
        const double r = discount_curve.forward_rate(t_next, t_prev);
        const double q =
            dividend_curve.has_value() ? dividend_curve->forward_rate(t_next, t_prev) : 0.0;
        r_max = std::max(r_max, r);
        rq_abs_max = std::max(rq_abs_max, std::abs(r - q));
    }
    if (max_dt <= 0.0) {
        return;
    }

    const double diffusion_max = (volatility * volatility) * (smax * smax) / (dS * dS);
    double dt_max = 0.0;
    std::string_view mode;
    if (implicit_discounting) {
        dt_max = 1.0 / diffusion_max;
        mode = "Hull implicit discounting";
    } else {
        const double r_pos = std::max(r_max, 0.0);
        dt_max =
            (diffusion_max + r_pos) > 0.0 ? 1.0 / (diffusion_max + r_pos) : 1.0 / diffusion_max;
        mode = "pure explicit discounting";
    }
    if (max_dt > dt_max) {
        const int min_steps = static_cast<int>(std::ceil(time_to_maturity / dt_max));
        throw StabilityError(std::format(
            "Explicit spot-grid scheme likely unstable (CFL violation, {}). max_dt={:.4g} "
            "exceeds dt_max={:.4g}. Increase time_steps to >= {}, or use "
            "log-spot/implicit/CN.",
            mode,
            max_dt,
            dt_max,
            min_steps));
    }

    const double drift_max = rq_abs_max * smax / dS;
    if (drift_max > diffusion_max) {
        throw StabilityError(std::format(
            "Explicit spot-grid scheme likely unstable/oscillatory: drift dominates diffusion "
            "in the central-difference stencil. diffusion_max={:.4g}, drift_max={:.4g}. Refine "
            "the spot grid (smaller dS), reduce smax, or use log-spot/implicit/CN.",
            diffusion_max,
            drift_max));
    }
}

// ── Time-step kernels ───────────────────────────────────────────────

/// Forward-Euler step. `r_dt > 0` applies Hull's implicit discounting of rV.
ArrayXd explicit_step(const ArrayXd& v_prev,
                      const OperatorCoeffs& scaled,
                      double left,
                      double right,
                      const ArrayXd* intrinsic,
                      double r_dt) {
    const Index m = v_prev.size() - 2;
    ArrayXd v_new = v_prev;
    v_new.segment(1, m) =
        (-scaled.gamma * v_prev.head(m) + (1.0 - scaled.beta) * v_prev.segment(1, m) -
         scaled.alpha * v_prev.tail(m)) /
        (1.0 + r_dt);
    v_new[0] = left;
    v_new[v_new.size() - 1] = right;
    if (intrinsic != nullptr) {
        v_new = v_new.max(*intrinsic);
    }
    return v_new;
}

/// Projected SOR (Gauss-Seidel with overrelaxation) for American exercise.
int psor_solve(ArrayXd& x,
               const ArrayXd& exercise,
               const ArrayXd& rhs,
               const ArrayXd& lower,
               const ArrayXd& diag,
               const ArrayXd& upper,
               double v_left,
               double v_right,
               double omega,
               double tol,
               int max_iter) {
    const Index n = x.size();
    for (Index k = 0; k < n; ++k) {
        if (x[k] < exercise[k]) {
            x[k] = exercise[k];
        }
    }
    int iter_used = max_iter;
    for (int iter = 0; iter < max_iter; ++iter) {
        double max_diff = 0.0;
        for (Index k = 0; k < n; ++k) {
            const double left_val = k > 0 ? x[k - 1] : v_left;
            const double right_val = k < n - 1 ? x[k + 1] : v_right;
            const double gs = (rhs[k] - lower[k] * left_val - upper[k] * right_val) / diag[k];
            const double old = x[k];
            double next = old + omega * (gs - old);
            if (next < exercise[k]) {
                next = exercise[k];
            }
            x[k] = next;
            max_diff = std::max(max_diff, std::abs(next - old));
        }
        if (max_diff < tol) {
            iter_used = iter + 1;
            break;
        }
    }
    return iter_used;
}

/// One implicit or Crank-Nicolson step, optionally with early exercise.
/// Returns the PSOR iteration count, or nullopt when PSOR was not used.
std::optional<int> implicit_cn_step(const ArrayXd& v_prev,
                                    ArrayXd& v,
                                    OperatorCoeffs scaled,
                                    double left,
                                    double right,
                                    PdeMethod method,
                                    const ArrayXd* intrinsic,
                                    PdeEarlyExercise american_solver,
                                    double omega,
                                    double tol,
                                    int max_iter) {
    const Index m = v_prev.size() - 2;
    if (m < 2) {
        throw ValidationError("spot_steps must be >= 3");
    }
    if (method == PdeMethod::CrankNicolson) {
        scaled.gamma *= 0.5;
        scaled.beta *= 0.5;
        scaled.alpha *= 0.5;
    }
    const ArrayXd& a = scaled.gamma;
    const ArrayXd& c = scaled.alpha;
    const ArrayXd diag = 1.0 + scaled.beta;

    ArrayXd rhs(m);
    if (method == PdeMethod::Implicit) {
        rhs = v_prev.segment(1, m);
    } else {
        rhs = -a * v_prev.head(m) + (1.0 - scaled.beta) * v_prev.segment(1, m) - c * v_prev.tail(m);
    }

    v[0] = left;
    v[v.size() - 1] = right;

    ArrayXd rhs_adj = rhs;
    rhs_adj[0] -= a[0] * v[0];
    rhs_adj[m - 1] -= c[m - 1] * v[v.size() - 1];

    ArrayXd x = solve_tridiagonal_thomas(a.tail(m - 1), diag, c.head(m - 1), rhs_adj);

    if (intrinsic == nullptr) {
        v.segment(1, m) = x;
        return std::nullopt;
    }
    const ArrayXd exercise = intrinsic->segment(1, m);
    if (american_solver == PdeEarlyExercise::Intrinsic) {
        v.segment(1, m) = x.max(exercise);
        return std::nullopt;
    }
    const int iters =
        psor_solve(x, exercise, rhs, a, diag, c, v[0], v[v.size() - 1], omega, tol, max_iter);
    v.segment(1, m) = x;
    return iters;
}

/// One PDE time-step restricted to the contiguous sub-interior
/// `[j_lo, j_hi]` — used by the coupled KI solve's inactive surface.
/// `scaled` holds full-interior coefficients (node i -> index i-1).
/// Only the explicit branch writes the sub-grid boundary nodes into the
/// result (mirroring the reference); implicit/CN callers re-impose them.
ArrayXd subgrid_pde_step(const ArrayXd& v,
                         const ArrayXd& v_prev,
                         Index j_lo,
                         Index j_hi,
                         const OperatorCoeffs& scaled,
                         double left,
                         double right,
                         PdeMethod method,
                         double r_dt) {
    ArrayXd v_new = v;
    const Index m = j_hi - j_lo + 1;
    const ArrayXd a_s = scaled.gamma.segment(j_lo - 1, m);
    const ArrayXd b_s = scaled.beta.segment(j_lo - 1, m);
    const ArrayXd c_s = scaled.alpha.segment(j_lo - 1, m);

    if (method == PdeMethod::Explicit || method == PdeMethod::ExplicitHull) {
        v_new.segment(j_lo, m) =
            (-a_s * v_prev.segment(j_lo - 1, m) + (1.0 - b_s) * v_prev.segment(j_lo, m) -
             c_s * v_prev.segment(j_lo + 1, m)) /
            (1.0 + r_dt);
        v_new[j_lo - 1] = left;
        v_new[j_hi + 1] = right;
    } else if (method == PdeMethod::Implicit) {
        const ArrayXd diag = 1.0 + b_s;
        ArrayXd rhs = v_prev.segment(j_lo, m);
        rhs[0] -= a_s[0] * left;
        rhs[m - 1] -= c_s[m - 1] * right;
        v_new.segment(j_lo, m) =
            solve_tridiagonal_thomas(a_s.tail(m - 1), diag, c_s.head(m - 1), rhs);
    } else {
        const ArrayXd a_h = a_s * 0.5;
        const ArrayXd b_h = b_s * 0.5;
        const ArrayXd c_h = c_s * 0.5;
        const ArrayXd diag = 1.0 + b_h;
        ArrayXd rhs = -a_h * v_prev.segment(j_lo - 1, m) + (1.0 - b_h) * v_prev.segment(j_lo, m) -
                      c_h * v_prev.segment(j_lo + 1, m);
        rhs[0] -= a_h[0] * left;
        rhs[m - 1] -= c_h[m - 1] * right;
        v_new.segment(j_lo, m) =
            solve_tridiagonal_thomas(a_h.tail(m - 1), diag, c_h.head(m - 1), rhs);
    }
    return v_new;
}

// ── Core solver ─────────────────────────────────────────────────────

const PdeParams& pde_params(const detail::ValuationState& state) {
    const auto* params =
        state.params.has_value() ? std::get_if<PdeParams>(&*state.params) : nullptr;
    if (params == nullptr) {
        throw ConfigurationError("PdeEngine requires PdeParams");
    }
    return *params;
}

void validate_fd_inputs(const detail::ValuationState& state,
                        const PdeParams& params,
                        double time_to_maturity,
                        bool early_exercise,
                        bool has_custom_payoff) {
    if (!has_custom_payoff && !state.option_type.has_value()) {
        throw UnsupportedFeatureError("FD PDE valuation supports only vanilla CALL/PUT.");
    }
    if (time_to_maturity <= 0.0) {
        throw ValidationError("time_to_maturity must be positive");
    }
    if (state.underlying.volatility() <= 0.0) {
        throw ValidationError("volatility must be positive");
    }
    if ((params.method == PdeMethod::Explicit || params.method == PdeMethod::ExplicitHull) &&
        early_exercise && params.american_solver == PdeEarlyExercise::GaussSeidel) {
        throw UnsupportedFeatureError("GAUSS_SEIDEL is not supported with explicit time stepping");
    }
}

}  // namespace

// ── Engine ──────────────────────────────────────────────────────────

PdeEngine::PdeEngine(const detail::ValuationState& state) : state_(state) {}

const PdeEngine::Solution& PdeEngine::solution() const {
    if (solution_.has_value()) {
        return *solution_;
    }
    // Public entry points short-circuit inception-triggered barriers before
    // ever reaching the solvers, so no triggered handling is needed here.
    if (const auto* barrier = std::get_if<BarrierSpec>(&state_.spec)) {
        if (barrier->action() == BarrierAction::Out) {
            solution_.emplace(compute_barrier_knock_out());
        } else if (state_.exercise_type() == ExerciseType::American) {
            solution_.emplace(compute_barrier_knock_in());
        } else {
            // European knock-in via native-surface parity:
            // V_KI = V_vanilla + R df_T - V_KO (R = 0 reduces to parity).
            const auto& [ko, van] = european_ki_components();
            const double ttm = state_.maturity_year_fraction();
            const auto& curve = state_.discount_curve();
            const double df_maturity = curve.df(ttm);
            const double rebate = barrier->rebate();
            const double price = van.price + rebate * df_maturity - ko.price;

            const double previous_ttm =
                ko.last_dtau > 0.0 ? std::max(ttm - ko.last_dtau, 0.0) : ttm;
            const double rebate_prev = rebate * curve.df(previous_ttm);

            // Grid greeks read the KO grid (the best single surface we
            // have); reconstruct the KI surface on it via parity.
            ArrayXd values(ko.spots.size());
            ArrayXd values_prev(ko.spots.size());
            for (Index i = 0; i < ko.spots.size(); ++i) {
                const double s = ko.spots[i];
                values[i] =
                    interp_linear(s, van.spots, van.values) + rebate * df_maturity - ko.values[i];
                values_prev[i] =
                    interp_linear(s, van.spots, van.values_prev) + rebate_prev - ko.values_prev[i];
            }
            solution_.emplace(Solution{
                .price = price,
                .spots = ko.spots,
                .values = std::move(values),
                .values_prev = std::move(values_prev),
                .last_dtau = ko.last_dtau,
            });
        }
        return *solution_;
    }
    solution_.emplace(compute_vanilla());
    return *solution_;
}

PdeEngine::Solution PdeEngine::compute_vanilla() const {
    const PdeParams& params = pde_params(state_);
    const bool early = state_.exercise_type() == ExerciseType::American;
    const auto* custom = std::get_if<PayoffSpec>(&state_.spec);
    const double spot = state_.underlying.initial_value();
    const std::optional<double> strike = state_.strike();
    const double volatility = state_.underlying.volatility();
    const double ttm = state_.maturity_year_fraction();
    const auto& curve = state_.discount_curve();
    const auto& q_curve = state_.underlying.dividend_curve();

    validate_fd_inputs(state_, params, ttm, early, custom != nullptr);

    const int m_steps = params.spot_steps;
    const double ref_price = strike.has_value() ? std::max(spot, *strike) : spot;

    ArrayXd grid;
    ArrayXd S;
    double dx = 0.0;
    if (params.space_grid == PdeSpaceGrid::Spot) {
        const double smax = params.smax_mult * ref_price;
        grid = linspace(0.0, smax, m_steps);
        S = grid;
        dx = smax / m_steps;
    } else {
        auto built = build_log_grid(spot,
                                    strike.value_or(spot),
                                    ttm,
                                    volatility,
                                    params.smax_mult,
                                    m_steps,
                                    params.time_steps,
                                    params.method);
        grid = std::move(built.grid);
        S = std::move(built.spots);
        dx = built.dx;
    }
    const double smin = S[0];
    const double smax = S[S.size() - 1];

    // Terminal payoff.
    ArrayXd payoff(S.size());
    if (custom != nullptr) {
        for (Index i = 0; i < S.size(); ++i) {
            payoff[i] = custom->payoff(S[i]);
        }
    } else {
        if (!strike.has_value()) {
            throw ConfigurationError("vanilla PDE pricing requires a strike");
        }
        if (state_.option_type == OptionType::Put) {
            payoff = (*strike - S).cwiseMax(0.0);
        } else {
            payoff = (S - *strike).cwiseMax(0.0);
        }
    }

    // Affine wing boundary models for custom payoffs: explicit metadata or a
    // local fit on the actual boundary nodes.
    std::optional<PayoffBoundaryModel> boundary_model;
    if (custom != nullptr) {
        boundary_model = custom->boundary_model();
        if (!boundary_model.has_value()) {
            boundary_model = PayoffBoundaryModel{
                .left = fit_affine_boundary_model(*custom, "left", S.head(4)),
                .right = fit_affine_boundary_model(*custom, "right", S.tail(4)),
            };
        } else if (params.space_grid == PdeSpaceGrid::LogSpot) {
            warn(
                "Explicit PayoffBoundaryModel with LOG_SPOT grid is interpreted as an affine "
                "boundary model on the finite truncated PDE domain, not as a true payoff tail "
                "asymptote. Ensure the supplied boundary model is appropriate at the actual "
                "grid boundaries.");
        }
    }

    ArrayXd v = payoff;
    const ArrayXd* intrinsic = early ? &payoff : nullptr;

    std::map<double, double> dividend_map;
    for (const auto& [tau, amount] : dividend_tau_schedule(state_)) {
        dividend_map.emplace(tau, amount);
    }

    // Maturity-date dividend (tau = 0): immediate jump on the terminal slice.
    if (const auto it = dividend_map.find(0.0); it != dividend_map.end()) {
        apply_dividend_jump(v, grid, it->second, params.space_grid);
        if (early) {
            v = v.max(payoff);
        }
        dividend_map.erase(it);
    }
    // Pricing-date dividend (tau = ttm): applied as a spot shift at the end.
    std::optional<double> pricing_div;
    if (const auto it = dividend_map.find(round12(ttm)); it != dividend_map.end()) {
        pricing_div = it->second;
        dividend_map.erase(it);
    }

    std::vector<double> dividend_taus;
    dividend_taus.reserve(dividend_map.size());
    for (const auto& [tau, amount] : dividend_map) {
        dividend_taus.push_back(tau);
    }
    const ArrayXd tau_grid = build_tau_grid(ttm, params.time_steps, dividend_taus);

    if ((params.method == PdeMethod::Explicit || params.method == PdeMethod::ExplicitHull) &&
        params.space_grid == PdeSpaceGrid::Spot) {
        check_explicit_spot_stability(tau_grid,
                                      volatility,
                                      smax,
                                      dx,
                                      ttm,
                                      curve,
                                      q_curve,
                                      params.method == PdeMethod::ExplicitHull);
    }

    const double df_full = curve.df(ttm);
    const double dq_full = q_curve.has_value() ? q_curve->df(ttm) : 1.0;

    const auto steps = build_time_step_schedule(tau_grid, params.method, params.rannacher_steps);

    ArrayXd v_prev = v;
    double last_dtau = 0.0;

    int psor_steps = 0;
    long psor_total_iters = 0;
    int psor_max_iters = 0;
    int psor_not_converged = 0;

    for (const auto& step : steps) {
        const double d_tau = step.tau_end - step.tau_start;
        const double t_prev = ttm - step.tau_start;
        const double t_curr = ttm - step.tau_end;

        const double r = curve.forward_rate(t_curr, t_prev);
        const double q = q_curve.has_value() ? q_curve->forward_rate(t_curr, t_prev) : 0.0;
        const bool implicit_discounting = step.method == PdeMethod::ExplicitHull;

        const OperatorCoeffs coeffs =
            params.space_grid == PdeSpaceGrid::Spot
                ? spot_operator_coeffs(
                      S.segment(1, m_steps - 1), dx, r, q, volatility, implicit_discounting)
                : log_operator_coeffs(dx, r, q, volatility, implicit_discounting, m_steps - 1);

        const double df_fwd = df_full / curve.df(t_curr);
        const double dq_fwd = q_curve.has_value() ? dq_full / q_curve->df(t_curr) : 1.0;

        const auto [left, right] = boundary_values(
            state_.option_type, strike, smin, smax, df_fwd, dq_fwd, early, custom, boundary_model);

        v_prev = v;
        last_dtau = d_tau;

        const OperatorCoeffs scaled{
            .gamma = -d_tau * coeffs.gamma,
            .beta = -d_tau * coeffs.beta,
            .alpha = -d_tau * coeffs.alpha,
        };

        if (step.method == PdeMethod::Explicit || step.method == PdeMethod::ExplicitHull) {
            v = explicit_step(
                v_prev, scaled, left, right, intrinsic, implicit_discounting ? r * d_tau : 0.0);
        } else {
            const auto psor_iters = implicit_cn_step(v_prev,
                                                     v,
                                                     scaled,
                                                     left,
                                                     right,
                                                     step.method,
                                                     intrinsic,
                                                     params.american_solver,
                                                     params.omega,
                                                     params.tol,
                                                     params.max_iter);
            if (psor_iters.has_value()) {
                psor_steps += 1;
                psor_total_iters += *psor_iters;
                psor_max_iters = std::max(psor_max_iters, *psor_iters);
                if (*psor_iters == params.max_iter) {
                    psor_not_converged += 1;
                }
            }
        }

        if (!dividend_map.empty()) {
            if (const auto it = dividend_map.find(round12(step.tau_end));
                it != dividend_map.end()) {
                apply_dividend_jump(v, grid, it->second, params.space_grid);
                if (early) {
                    v = v.max(payoff);
                }
            }
        }
    }

    if (psor_steps > 0) {
        detail::logger()->debug("PDE PSOR steps={} avg_iters={:.2f} max_iters={} not_converged={}",
                                psor_steps,
                                static_cast<double>(psor_total_iters) / psor_steps,
                                psor_max_iters,
                                psor_not_converged);
    }

    // The input spot is cum-dividend: a pricing-date dividend shifts the
    // interpolation point.
    const double interp_spot = spot - pricing_div.value_or(0.0);
    const double price = interp_linear(interp_spot, S, v);

    return Solution{
        .price = price,
        .spots = std::move(S),
        .values = std::move(v),
        .values_prev = std::move(v_prev),
        .last_dtau = last_dtau,
    };
}

double PdeEngine::present_value() const {
    if (const auto* barrier = std::get_if<BarrierSpec>(&state_.spec)) {
        if (detail::barrier_triggered_at_inception(state_)) {
            if (barrier->action() == BarrierAction::Out) {
                if (barrier->rebate() <= 0.0) {
                    return 0.0;
                }
                if (barrier->rebate_timing() == RebateTiming::AtHit) {
                    return barrier->rebate();
                }
                return barrier->rebate() *
                       state_.discount_curve().df(state_.maturity_year_fraction());
            }
            return vanilla_equivalent_engine().present_value();
        }
    }
    return solution().price;
}

// ── Barrier machinery ───────────────────────────────────────────────

bool PdeEngine::is_european_knock_in() const {
    const auto* barrier = std::get_if<BarrierSpec>(&state_.spec);
    return barrier != nullptr && barrier->action() == BarrierAction::In &&
           barrier->exercise_type() == ExerciseType::European;
}

std::vector<double> PdeEngine::barrier_monitoring_taus() const {
    std::vector<double> taus;
    const auto dates = detail::barrier_monitoring_dates(state_);
    if (!dates.has_value()) {
        return taus;  // continuous monitoring
    }
    const double ttm = state_.maturity_year_fraction();
    for (const DateTime date : *dates) {
        if (date >= state_.pricing_date() && date <= state_.maturity()) {
            const double t =
                year_fraction(state_.pricing_date(), date, state_.day_count_convention());
            taus.push_back(round12(ttm - t));
        }
    }
    std::sort(taus.begin(), taus.end());
    return taus;
}

const std::pair<PdeEngine::Solution, PdeEngine::Solution>& PdeEngine::european_ki_components()
    const {
    if (!ki_components_.has_value()) {
        // The vanilla leg reads the barrier spec's strike/type through the
        // shared state accessors, so `compute_vanilla` doubles as the
        // reference `_fd_core` call (European by construction here).
        ki_components_.emplace(compute_barrier_knock_out(), compute_vanilla());
    }
    return *ki_components_;
}

double PdeEngine::barrier_last_dtau() const {
    const double ttm = state_.maturity_year_fraction();
    std::vector<double> extra_taus;
    for (const auto& [tau, amount] : dividend_tau_schedule(state_)) {
        if (tau > 1.0e-12 && tau < ttm - 1.0e-12) {
            extra_taus.push_back(tau);
        }
    }
    for (const double tau : barrier_monitoring_taus()) {
        extra_taus.push_back(tau);
    }
    const ArrayXd tau_grid = build_tau_grid(ttm, pde_params(state_).time_steps, extra_taus);
    if (tau_grid.size() < 2) {
        return 0.0;
    }
    return tau_grid[tau_grid.size() - 1] - tau_grid[tau_grid.size() - 2];
}

double PdeEngine::discounted_rebate_theta(double last_dtau) const {
    const auto& spec = std::get<BarrierSpec>(state_.spec);
    if (spec.rebate() <= 0.0 || spec.rebate_timing() != RebateTiming::AtExpiry ||
        last_dtau <= 0.0) {
        return 0.0;
    }
    const double ttm = state_.maturity_year_fraction();
    const auto& curve = state_.discount_curve();
    const double current_value = spec.rebate() * curve.df(ttm);
    const double previous_value = spec.rebate() * curve.df(std::max(ttm - last_dtau, 0.0));
    return (previous_value - current_value) / last_dtau / kDaysPerYear;
}

const PdeEngine& PdeEngine::vanilla_equivalent_engine() const {
    if (vanilla_engine_ == nullptr) {
        const auto& spec = std::get<BarrierSpec>(state_.spec);
        vanilla_state_.emplace(detail::ValuationState{
            .underlying = state_.underlying,
            .spec = OptionSpec{detail::vanilla_equivalent_spec(spec)},
            .pricing_method = PricingMethod::PdeFd,
            .params = state_.params,
            .currency = state_.currency,
            .option_type = spec.option_type(),
        });
        vanilla_engine_ = std::make_unique<PdeEngine>(*vanilla_state_);
    }
    return *vanilla_engine_;
}

PdeEngine::Solution PdeEngine::compute_barrier_knock_out() const {
    const PdeParams& params = pde_params(state_);
    const auto& spec = std::get<BarrierSpec>(state_.spec);
    const bool early = state_.exercise_type() == ExerciseType::American;
    const double spot = state_.underlying.initial_value();
    const double strike = spec.strike();
    const double barrier = spec.barrier();
    const bool is_up = spec.direction() == BarrierDirection::Up;
    const bool is_call = spec.option_type() == OptionType::Call;
    const bool continuous = spec.monitoring() == BarrierMonitoring::Continuous;
    const double rebate = spec.rebate();
    const double volatility = state_.underlying.volatility();
    const double ttm = state_.maturity_year_fraction();
    const auto& curve = state_.discount_curve();
    const auto& q_curve = state_.underlying.dividend_curve();

    validate_fd_inputs(state_, params, ttm, early, /*has_custom_payoff=*/false);

    const std::vector<double> monitoring_taus = barrier_monitoring_taus();
    const int m_steps = params.spot_steps;
    const double ref_price = std::max(spot, strike);

    // For continuous KO the grid is truncated at the barrier (the barrier
    // becomes a boundary node); discrete KO uses the full grid with the
    // barrier anchored half a cell between nodes.
    SpaceGridBuild built;
    if (params.space_grid == PdeSpaceGrid::LogSpot) {
        if (continuous) {
            const double smin_target =
                is_up ? std::max(ref_price / params.smax_mult, 1.0e-8) : barrier;
            const double smax_target = is_up ? barrier : params.smax_mult * ref_price;
            built = build_ko_continuous_log_grid(smin_target,
                                                 smax_target,
                                                 ref_price,
                                                 params.smax_mult,
                                                 spec.direction(),
                                                 volatility,
                                                 ttm,
                                                 m_steps,
                                                 params.time_steps,
                                                 params.method);
        } else {
            built = build_log_grid(spot,
                                   strike,
                                   ttm,
                                   volatility,
                                   params.smax_mult,
                                   m_steps,
                                   params.time_steps,
                                   params.method,
                                   barrier,
                                   /*anchor_half_step=*/true);
        }
    } else {
        if (continuous) {
            const double smin_target =
                is_up ? std::max(ref_price / params.smax_mult, 1.0e-8) : barrier;
            const double smax_target = is_up ? barrier : params.smax_mult * ref_price;
            built = build_spot_grid(smin_target, smax_target, m_steps);
        } else {
            built = build_spot_grid(
                0.0, params.smax_mult * ref_price, m_steps, barrier, /*anchor_half_step=*/true);
        }
    }
    const ArrayXd grid = std::move(built.grid);
    const ArrayXd S = std::move(built.spots);
    const double dx = built.dx;
    const double smin = S[0];
    const double smax = S[S.size() - 1];

    const auto hit = [&](Index i) { return is_up ? S[i] >= barrier : S[i] <= barrier; };

    ArrayXd payoff(S.size());
    if (is_call) {
        payoff = (S - strike).cwiseMax(0.0);
    } else {
        payoff = (strike - S).cwiseMax(0.0);
    }
    // American intrinsic stays the vanilla payoff at any in-life moment,
    // regardless of the KO-zone zeroing applied to the terminal slice below.
    const ArrayXd intrinsic_values = payoff;
    const ArrayXd* intrinsic = early ? &intrinsic_values : nullptr;

    std::set<double> monitoring_tau_set;
    for (const double tau : monitoring_taus) {
        monitoring_tau_set.insert(round12(tau));
    }
    if (!continuous && monitoring_tau_set.contains(0.0)) {
        for (Index i = 0; i < S.size(); ++i) {
            if (hit(i)) {
                payoff[i] = 0.0;
            }
        }
    }
    ArrayXd v = payoff;

    std::map<double, double> dividend_map;
    for (const auto& [tau, amount] : dividend_tau_schedule(state_)) {
        dividend_map.emplace(tau, amount);
    }
    if (const auto it = dividend_map.find(0.0); it != dividend_map.end()) {
        apply_dividend_jump(v, grid, it->second, params.space_grid);
        if (early) {
            v = v.max(payoff);
        }
        dividend_map.erase(it);
    }
    std::optional<double> pricing_div;
    if (const auto it = dividend_map.find(round12(ttm)); it != dividend_map.end()) {
        pricing_div = it->second;
        dividend_map.erase(it);
    }

    std::vector<double> extra_taus;
    extra_taus.reserve(dividend_map.size());
    for (const auto& [tau, amount] : dividend_map) {
        extra_taus.push_back(tau);
    }
    if (!continuous) {
        extra_taus.insert(extra_taus.end(), monitoring_taus.begin(), monitoring_taus.end());
    }
    const ArrayXd tau_grid = build_tau_grid(ttm, params.time_steps, extra_taus);

    if ((params.method == PdeMethod::Explicit || params.method == PdeMethod::ExplicitHull) &&
        params.space_grid == PdeSpaceGrid::Spot) {
        check_explicit_spot_stability(tau_grid,
                                      volatility,
                                      smax,
                                      dx,
                                      ttm,
                                      curve,
                                      q_curve,
                                      params.method == PdeMethod::ExplicitHull);
    }

    const double df_full = curve.df(ttm);
    const double dq_full = q_curve.has_value() ? q_curve->df(ttm) : 1.0;
    const auto steps = build_time_step_schedule(tau_grid, params.method, params.rannacher_steps);

    ArrayXd v_prev = v;
    double last_dtau = 0.0;
    int psor_steps = 0;
    long psor_total_iters = 0;
    int psor_max_iters = 0;
    int psor_not_converged = 0;

    for (const auto& step : steps) {
        const double d_tau = step.tau_end - step.tau_start;
        const double t_prev = ttm - step.tau_start;
        const double t_curr = ttm - step.tau_end;

        const double r = curve.forward_rate(t_curr, t_prev);
        const double q = q_curve.has_value() ? q_curve->forward_rate(t_curr, t_prev) : 0.0;
        const bool implicit_discounting = step.method == PdeMethod::ExplicitHull;

        const OperatorCoeffs coeffs =
            params.space_grid == PdeSpaceGrid::Spot
                ? spot_operator_coeffs(
                      S.segment(1, m_steps - 1), dx, r, q, volatility, implicit_discounting)
                : log_operator_coeffs(dx, r, q, volatility, implicit_discounting, m_steps - 1);

        const double df_fwd = df_full / curve.df(t_curr);
        const double dq_fwd = q_curve.has_value() ? dq_full / q_curve->df(t_curr) : 1.0;

        double left = 0.0;
        double right = 0.0;
        if (continuous) {
            // The barrier sits on a grid boundary carrying the rebate value.
            double barrier_bv = 0.0;
            if (rebate != 0.0) {
                barrier_bv = spec.rebate_timing() == RebateTiming::AtHit ? rebate : rebate * df_fwd;
            }
            if (!is_up) {
                left = barrier_bv;
                if (!is_call) {
                    right = 0.0;
                } else {
                    const double continuation = smax * dq_fwd - strike * df_fwd;
                    const double intrinsic_bv = std::max(smax - strike, 0.0);
                    right =
                        early ? std::max(continuation, intrinsic_bv) : std::max(continuation, 0.0);
                }
            } else {
                right = barrier_bv;
                if (!is_call) {
                    const double intrinsic_bv = std::max(strike - smin, 0.0);
                    const double continuation = strike * df_fwd - smin * dq_fwd;
                    left = early ? std::max(continuation, intrinsic_bv) : continuation;
                } else {
                    left = 0.0;
                }
            }
        } else {
            std::tie(left, right) = boundary_values(state_.option_type,
                                                    strike,
                                                    smin,
                                                    smax,
                                                    df_fwd,
                                                    dq_fwd,
                                                    early,
                                                    nullptr,
                                                    std::nullopt);
        }

        v_prev = v;
        last_dtau = d_tau;

        const OperatorCoeffs scaled{
            .gamma = -d_tau * coeffs.gamma,
            .beta = -d_tau * coeffs.beta,
            .alpha = -d_tau * coeffs.alpha,
        };

        if (step.method == PdeMethod::Explicit || step.method == PdeMethod::ExplicitHull) {
            v = explicit_step(
                v_prev, scaled, left, right, intrinsic, implicit_discounting ? r * d_tau : 0.0);
        } else {
            const auto psor_iters = implicit_cn_step(v_prev,
                                                     v,
                                                     scaled,
                                                     left,
                                                     right,
                                                     step.method,
                                                     intrinsic,
                                                     params.american_solver,
                                                     params.omega,
                                                     params.tol,
                                                     params.max_iter);
            if (psor_iters.has_value()) {
                psor_steps += 1;
                psor_total_iters += *psor_iters;
                psor_max_iters = std::max(psor_max_iters, *psor_iters);
                if (*psor_iters == params.max_iter) {
                    psor_not_converged += 1;
                }
            }
        }

        if (!dividend_map.empty()) {
            if (const auto it = dividend_map.find(round12(step.tau_end));
                it != dividend_map.end()) {
                apply_dividend_jump(v, grid, it->second, params.space_grid);
                if (early) {
                    v = v.max(intrinsic_values);
                }
            }
        }

        // Discrete barrier reset at monitoring dates: knocked-out nodes take
        // the rebate value; early exercise is re-imposed on surviving nodes.
        if (!continuous && monitoring_tau_set.contains(round12(step.tau_end))) {
            double reset_value = 0.0;
            if (rebate != 0.0) {
                reset_value =
                    spec.rebate_timing() == RebateTiming::AtHit ? rebate : rebate * df_fwd;
            }
            for (Index i = 0; i < S.size(); ++i) {
                if (hit(i)) {
                    v[i] = reset_value;
                } else if (early && intrinsic_values[i] > v[i]) {
                    v[i] = intrinsic_values[i];
                }
            }
        }
    }

    if (psor_steps > 0) {
        detail::logger()->debug(
            "PDE barrier PSOR steps={} avg_iters={:.2f} max_iters={} not_converged={}",
            psor_steps,
            static_cast<double>(psor_total_iters) / psor_steps,
            psor_max_iters,
            psor_not_converged);
    }

    const double interp_spot = spot - pricing_div.value_or(0.0);
    const double price = interp_linear(interp_spot, S, v);
    return Solution{
        .price = price,
        .spots = S,
        .values = std::move(v),
        .values_prev = std::move(v_prev),
        .last_dtau = last_dtau,
    };
}

PdeEngine::Solution PdeEngine::compute_barrier_knock_in() const {
    const PdeParams& params = pde_params(state_);
    const auto& spec = std::get<BarrierSpec>(state_.spec);
    const bool early = state_.exercise_type() == ExerciseType::American;
    const double spot = state_.underlying.initial_value();
    const double strike = spec.strike();
    const double barrier = spec.barrier();
    const bool is_up = spec.direction() == BarrierDirection::Up;
    const bool is_call = spec.option_type() == OptionType::Call;
    const bool continuous = spec.monitoring() == BarrierMonitoring::Continuous;
    // KI rebates are AT_EXPIRY by contract: paid at maturity if never hit.
    const double rebate = spec.rebate();
    const double volatility = state_.underlying.volatility();
    const double ttm = state_.maturity_year_fraction();
    const auto& curve = state_.discount_curve();
    const auto& q_curve = state_.underlying.dividend_curve();

    validate_fd_inputs(state_, params, ttm, early, /*has_custom_payoff=*/false);

    const std::vector<double> monitoring_taus = barrier_monitoring_taus();
    const int m_steps = params.spot_steps;
    const double ref_price = std::max(spot, strike);

    // Full (untruncated) grid, anchored on the barrier: on a node for
    // continuous coupling, half a cell off for discrete monitoring.
    SpaceGridBuild built;
    if (params.space_grid == PdeSpaceGrid::LogSpot) {
        built = build_log_grid(spot,
                               strike,
                               ttm,
                               volatility,
                               params.smax_mult,
                               m_steps,
                               params.time_steps,
                               params.method,
                               barrier,
                               /*anchor_half_step=*/!continuous);
    } else {
        built = build_spot_grid(0.0,
                                params.smax_mult * ref_price,
                                m_steps,
                                barrier,
                                /*anchor_half_step=*/!continuous);
    }
    const ArrayXd grid = std::move(built.grid);
    const ArrayXd S = std::move(built.spots);
    const double dx = built.dx;
    const double smin = S[0];
    const double smax = S[S.size() - 1];

    const auto hit = [&](Index i) { return is_up ? S[i] >= barrier : S[i] <= barrier; };

    ArrayXd payoff(S.size());
    if (is_call) {
        payoff = (S - strike).cwiseMax(0.0);
    } else {
        payoff = (strike - S).cwiseMax(0.0);
    }
    const ArrayXd* intrinsic = early ? &payoff : nullptr;

    // Active surface: the barrier has been hit (vanilla behaviour).
    // Inactive surface: not yet hit; terminal value is the maturity rebate.
    ArrayXd v_act = payoff;
    ArrayXd v_inact = ArrayXd::Constant(S.size(), rebate);

    std::map<double, double> dividend_map;
    for (const auto& [tau, amount] : dividend_tau_schedule(state_)) {
        dividend_map.emplace(tau, amount);
    }
    if (const auto it = dividend_map.find(0.0); it != dividend_map.end()) {
        apply_dividend_jump(v_act, grid, it->second, params.space_grid);
        if (early) {
            v_act = v_act.max(payoff);
        }
        // The inactive terminal surface is spot-independent, so the
        // maturity-date jump is a no-op for it.
        dividend_map.erase(it);
    }
    std::optional<double> pricing_div;
    if (const auto it = dividend_map.find(round12(ttm)); it != dividend_map.end()) {
        pricing_div = it->second;
        dividend_map.erase(it);
    }

    std::set<double> monitoring_tau_set;
    for (const double tau : monitoring_taus) {
        monitoring_tau_set.insert(round12(tau));
    }

    std::vector<double> extra_taus;
    extra_taus.reserve(dividend_map.size());
    for (const auto& [tau, amount] : dividend_map) {
        extra_taus.push_back(tau);
    }
    if (!continuous) {
        extra_taus.insert(extra_taus.end(), monitoring_taus.begin(), monitoring_taus.end());
    }
    const ArrayXd tau_grid = build_tau_grid(ttm, params.time_steps, extra_taus);

    // Nearest node to the barrier level, for the sub-grid coupling.
    Index j_barrier = 0;
    (S - barrier).abs().minCoeff(&j_barrier);

    // Terminal coupling: a barrier already triggered at maturity moves the
    // inactive state onto the active payoff.
    if (continuous || monitoring_tau_set.contains(0.0)) {
        for (Index i = 0; i < S.size(); ++i) {
            if (hit(i)) {
                v_inact[i] = v_act[i];
            }
        }
    }

    // Continuous monitoring solves the inactive surface only on the far side
    // of the barrier, with the active value at the barrier node as the inner
    // Dirichlet boundary (implicit coupling — no operator-splitting error).
    Index j_lo = 1;
    Index j_hi = m_steps - 1;
    if (continuous) {
        if (!is_up) {
            j_lo = j_barrier + 1;
        } else {
            j_hi = j_barrier - 1;
        }
    }

    const double df_full = curve.df(ttm);
    const double dq_full = q_curve.has_value() ? q_curve->df(ttm) : 1.0;
    const auto steps = build_time_step_schedule(tau_grid, params.method, params.rannacher_steps);

    ArrayXd v_act_prev = v_act;
    ArrayXd v_inact_prev = v_inact;
    double last_dtau = 0.0;
    int psor_steps = 0;
    long psor_total_iters = 0;
    int psor_max_iters = 0;
    int psor_not_converged = 0;

    for (const auto& step : steps) {
        const double d_tau = step.tau_end - step.tau_start;
        const double t_prev = ttm - step.tau_start;
        const double t_curr = ttm - step.tau_end;

        const double r = curve.forward_rate(t_curr, t_prev);
        const double q = q_curve.has_value() ? q_curve->forward_rate(t_curr, t_prev) : 0.0;
        const bool implicit_discounting = step.method == PdeMethod::ExplicitHull;

        const OperatorCoeffs coeffs =
            params.space_grid == PdeSpaceGrid::Spot
                ? spot_operator_coeffs(
                      S.segment(1, m_steps - 1), dx, r, q, volatility, implicit_discounting)
                : log_operator_coeffs(dx, r, q, volatility, implicit_discounting, m_steps - 1);

        const double df_fwd = df_full / curve.df(t_curr);
        const double dq_fwd = q_curve.has_value() ? dq_full / q_curve->df(t_curr) : 1.0;

        const auto [left, right] = boundary_values(
            state_.option_type, strike, smin, smax, df_fwd, dq_fwd, early, nullptr, std::nullopt);

        v_act_prev = v_act;
        v_inact_prev = v_inact;
        last_dtau = d_tau;

        const OperatorCoeffs scaled{
            .gamma = -d_tau * coeffs.gamma,
            .beta = -d_tau * coeffs.beta,
            .alpha = -d_tau * coeffs.alpha,
        };
        const double r_dt = implicit_discounting ? r * d_tau : 0.0;

        // Step A: active surface (vanilla, with early exercise).
        if (step.method == PdeMethod::Explicit || step.method == PdeMethod::ExplicitHull) {
            v_act = explicit_step(v_act_prev, scaled, left, right, intrinsic, r_dt);
        } else {
            const auto psor_iters = implicit_cn_step(v_act_prev,
                                                     v_act,
                                                     scaled,
                                                     left,
                                                     right,
                                                     step.method,
                                                     intrinsic,
                                                     params.american_solver,
                                                     params.omega,
                                                     params.tol,
                                                     params.max_iter);
            if (psor_iters.has_value()) {
                psor_steps += 1;
                psor_total_iters += *psor_iters;
                psor_max_iters = std::max(psor_max_iters, *psor_iters);
                if (*psor_iters == params.max_iter) {
                    psor_not_converged += 1;
                }
            }
        }

        // Step B: inactive surface (pure continuation, no exercise).
        const double rebate_bv = rebate * df_fwd;
        if (continuous) {
            const double left_inact = !is_up ? v_act[j_barrier] : rebate_bv;
            const double right_inact = !is_up ? rebate_bv : v_act[j_barrier];
            if (j_lo <= j_hi) {
                v_inact = subgrid_pde_step(v_inact,
                                           v_inact_prev,
                                           j_lo,
                                           j_hi,
                                           scaled,
                                           left_inact,
                                           right_inact,
                                           step.method,
                                           r_dt);
            }
            // Complete the slice: impose the hit-side coupling region and
            // the far-field boundary.
            if (!is_up) {
                v_inact.head(j_barrier + 1) = v_act.head(j_barrier + 1);
                v_inact[v_inact.size() - 1] = rebate_bv;
            } else {
                v_inact.tail(S.size() - j_barrier) = v_act.tail(S.size() - j_barrier);
                v_inact[0] = rebate_bv;
            }
        } else {
            // Discrete monitoring: full-grid solve with a one-step
            // look-ahead boundary proxy on the risky side (activation can
            // only occur at the next observation), coupling applied at
            // monitoring dates.
            const bool is_monitoring_step = monitoring_tau_set.contains(round12(step.tau_end));
            double left_inact = rebate_bv;
            double right_inact = rebate_bv;
            if (!is_up) {
                left_inact = is_monitoring_step ? v_act[0] : v_act_prev[0];
            } else {
                right_inact = is_monitoring_step ? v_act[v_act.size() - 1]
                                                 : v_act_prev[v_act_prev.size() - 1];
            }
            if (step.method == PdeMethod::Explicit || step.method == PdeMethod::ExplicitHull) {
                v_inact =
                    explicit_step(v_inact_prev, scaled, left_inact, right_inact, nullptr, r_dt);
            } else {
                implicit_cn_step(v_inact_prev,
                                 v_inact,
                                 scaled,
                                 left_inact,
                                 right_inact,
                                 step.method,
                                 nullptr,
                                 params.american_solver,
                                 params.omega,
                                 params.tol,
                                 params.max_iter);
            }
            if (is_monitoring_step) {
                for (Index i = 0; i < S.size(); ++i) {
                    if (hit(i)) {
                        v_inact[i] = v_act[i];
                    }
                }
            }
        }

        // Discrete dividend jumps on both surfaces.
        if (!dividend_map.empty()) {
            if (const auto it = dividend_map.find(round12(step.tau_end));
                it != dividend_map.end()) {
                apply_dividend_jump(v_act, grid, it->second, params.space_grid);
                if (early) {
                    v_act = v_act.max(payoff);
                }
                apply_dividend_jump(v_inact, grid, it->second, params.space_grid);
                if (continuous) {
                    if (!is_up) {
                        v_inact.head(j_barrier + 1) = v_act.head(j_barrier + 1);
                    } else {
                        v_inact.tail(S.size() - j_barrier) = v_act.tail(S.size() - j_barrier);
                    }
                }
            }
        }
    }

    if (psor_steps > 0) {
        detail::logger()->debug(
            "PDE barrier KI PSOR steps={} avg_iters={:.2f} max_iters={} not_converged={}",
            psor_steps,
            static_cast<double>(psor_total_iters) / psor_steps,
            psor_max_iters,
            psor_not_converged);
    }

    // Price from the inactive surface: the option starts inactive.
    const double interp_spot = spot - pricing_div.value_or(0.0);
    const double price = interp_linear(interp_spot, S, v_inact);
    return Solution{
        .price = price,
        .spots = S,
        .values = std::move(v_inact),
        .values_prev = std::move(v_inact_prev),
        .last_dtau = last_dtau,
    };
}

// ── Grid Greeks ─────────────────────────────────────────────────────

namespace {

/// Nearest interior grid index to `spot` (searchsorted, clamped).
Index spot_grid_index(const ArrayXd& S, double spot) {
    const Index j = std::distance(S.begin(), std::lower_bound(S.begin(), S.end(), spot));
    return std::max<Index>(1, std::min<Index>(j, S.size() - 2));
}

/// Parabolic-Lagrange first derivative evaluated exactly at `spot`.
double grid_delta_at_spot(const ArrayXd& S, const ArrayXd& V, Index j, double spot) {
    const double x0 = S[j - 1];
    const double x1 = S[j];
    const double x2 = S[j + 1];
    return V[j - 1] * (2.0 * spot - x1 - x2) / ((x0 - x1) * (x0 - x2)) +
           V[j] * (2.0 * spot - x0 - x2) / ((x1 - x0) * (x1 - x2)) +
           V[j + 1] * (2.0 * spot - x0 - x1) / ((x2 - x0) * (x2 - x1));
}

/// Non-uniform three-point gamma stencil at index `j`.
double grid_gamma_at_index(const ArrayXd& S, const ArrayXd& V, Index j) {
    const double h_up = S[j + 1] - S[j];
    const double h_dn = S[j] - S[j - 1];
    return 2.0 * (V[j + 1] * h_dn + V[j - 1] * h_up - V[j] * (h_up + h_dn)) /
           (h_up * h_dn * (h_up + h_dn));
}

/// Cubic-Lagrange second derivative evaluated exactly at `spot`.
double grid_gamma_at_spot(const ArrayXd& S, const ArrayXd& V, Index j, double spot) {
    const Index jc = std::max<Index>(1, std::min<Index>(j, S.size() - 3));
    const double x0 = S[jc - 1];
    const double x1 = S[jc];
    const double x2 = S[jc + 1];
    const double x3 = S[jc + 2];
    const double d0 = (x0 - x1) * (x0 - x2) * (x0 - x3);
    const double d1 = (x1 - x0) * (x1 - x2) * (x1 - x3);
    const double d2 = (x2 - x0) * (x2 - x1) * (x2 - x3);
    const double d3 = (x3 - x0) * (x3 - x1) * (x3 - x2);
    return V[jc - 1] * (6.0 * spot - 2.0 * (x1 + x2 + x3)) / d0 +
           V[jc] * (6.0 * spot - 2.0 * (x0 + x2 + x3)) / d1 +
           V[jc + 1] * (6.0 * spot - 2.0 * (x0 + x1 + x3)) / d2 +
           V[jc + 2] * (6.0 * spot - 2.0 * (x0 + x1 + x2)) / d3;
}

/// Cubic gamma disagrees with parabolic beyond this fraction => treat the
/// cubic as polluted by PDE noise and fall back to the parabolic value.
constexpr double kGammaCubicParabolicRelTol = 0.5;

double grid_gamma_safe(const ArrayXd& S, const ArrayXd& V, Index j, double spot) {
    const double parabolic = grid_gamma_at_index(S, V, j);
    const double cubic = grid_gamma_at_spot(S, V, j, spot);
    const double scale = std::max({std::abs(parabolic), std::abs(cubic), 1e-6});
    if (std::abs(cubic - parabolic) > kGammaCubicParabolicRelTol * scale) {
        return parabolic;
    }
    return cubic;
}

/// Parabolic-Lagrange interpolation of V at exactly `spot`.
double grid_value_at_spot(const ArrayXd& S, const ArrayXd& V, Index j, double spot) {
    const double x0 = S[j - 1];
    const double x1 = S[j];
    const double x2 = S[j + 1];
    return V[j - 1] * (spot - x1) * (spot - x2) / ((x0 - x1) * (x0 - x2)) +
           V[j] * (spot - x0) * (spot - x2) / ((x1 - x0) * (x1 - x2)) +
           V[j + 1] * (spot - x0) * (spot - x1) / ((x2 - x0) * (x2 - x1));
}

}  // namespace

std::optional<std::pair<double, double>> PdeEngine::intrinsic_short_circuit(const ArrayXd& S,
                                                                            const ArrayXd& V,
                                                                            Index j) const {
    // American vanilla/barrier only: in the exercise region the local value
    // function is the intrinsic, so delta = ±1 and gamma = 0 exactly, while
    // the PDE stencils are polluted by PSOR oscillations.
    if (!std::holds_alternative<VanillaSpec>(state_.spec) &&
        !std::holds_alternative<BarrierSpec>(state_.spec)) {
        return std::nullopt;
    }
    if (state_.exercise_type() != ExerciseType::American) {
        return std::nullopt;
    }
    const auto strike = state_.strike();
    if (!strike.has_value()) {
        return std::nullopt;
    }
    const double s_node = S[j];
    double intrinsic = 0.0;
    double sign = 0.0;
    if (state_.option_type == OptionType::Call) {
        intrinsic = s_node - *strike;
        sign = 1.0;
    } else {
        intrinsic = *strike - s_node;
        sign = -1.0;
    }
    if (intrinsic <= 0.0) {
        return std::nullopt;
    }
    if (std::abs(V[j] - intrinsic) > std::max(1e-8, 1e-6 * intrinsic)) {
        return std::nullopt;
    }
    return std::pair{sign, 0.0};
}

double PdeEngine::delta() const {
    if (const auto* barrier = std::get_if<BarrierSpec>(&state_.spec)) {
        if (detail::barrier_triggered_at_inception(state_)) {
            return barrier->action() == BarrierAction::Out ? 0.0
                                                           : vanilla_equivalent_engine().delta();
        }
        if (is_european_knock_in()) {
            // Native-surface parity: extract on each component grid rather
            // than differentiating the reconstructed KI surface.
            const auto& [ko, van] = european_ki_components();
            const double spot = state_.underlying.initial_value();
            const Index j_van = spot_grid_index(van.spots, spot);
            const Index j_ko = spot_grid_index(ko.spots, spot);
            return grid_delta_at_spot(van.spots, van.values, j_van, spot) -
                   grid_delta_at_spot(ko.spots, ko.values, j_ko, spot);
        }
    }
    const Solution& r = solution();
    const double spot = state_.underlying.initial_value();
    const Index j = spot_grid_index(r.spots, spot);
    if (const auto sc = intrinsic_short_circuit(r.spots, r.values, j); sc.has_value()) {
        return sc->first;
    }
    return grid_delta_at_spot(r.spots, r.values, j, spot);
}

double PdeEngine::gamma() const {
    if (const auto* barrier = std::get_if<BarrierSpec>(&state_.spec)) {
        if (detail::barrier_triggered_at_inception(state_)) {
            return barrier->action() == BarrierAction::Out ? 0.0
                                                           : vanilla_equivalent_engine().gamma();
        }
        if (is_european_knock_in()) {
            const auto& [ko, van] = european_ki_components();
            const double spot = state_.underlying.initial_value();
            const Index j_van = spot_grid_index(van.spots, spot);
            const Index j_ko = spot_grid_index(ko.spots, spot);
            return grid_gamma_safe(van.spots, van.values, j_van, spot) -
                   grid_gamma_safe(ko.spots, ko.values, j_ko, spot);
        }
    }
    const Solution& r = solution();
    const double spot = state_.underlying.initial_value();
    const Index j = spot_grid_index(r.spots, spot);
    if (const auto sc = intrinsic_short_circuit(r.spots, r.values, j); sc.has_value()) {
        return sc->second;
    }
    return grid_gamma_safe(r.spots, r.values, j, spot);
}

double PdeEngine::grid_theta_identity(const Solution& result) const {
    // Black-Scholes PDE identity: theta = rV - (r-q) S delta - 0.5 sigma^2 S^2 gamma,
    // per calendar day. Matches the accuracy of the grid delta/gamma.
    if (result.last_dtau <= 0.0) {
        return 0.0;
    }
    const double spot = state_.underlying.initial_value();
    const Index j = spot_grid_index(result.spots, spot);
    const double pv_at_spot = grid_value_at_spot(result.spots, result.values, j, spot);
    const double delta_val = grid_delta_at_spot(result.spots, result.values, j, spot);
    const double gamma_val = grid_gamma_safe(result.spots, result.values, j, spot);

    const double sigma = state_.underlying.volatility();
    const double dt_probe = std::max(result.last_dtau, 1.0e-8);
    const double r = state_.discount_curve().forward_rate(0.0, dt_probe);
    const auto& q_curve = state_.underlying.dividend_curve();
    const double q = q_curve.has_value() ? q_curve->forward_rate(0.0, dt_probe) : 0.0;

    const double theta_annual =
        r * pv_at_spot - (r - q) * spot * delta_val - 0.5 * sigma * sigma * spot * spot * gamma_val;
    return theta_annual / kDaysPerYear;
}

double PdeEngine::theta() const {
    if (const auto* barrier = std::get_if<BarrierSpec>(&state_.spec)) {
        if (detail::barrier_triggered_at_inception(state_)) {
            if (barrier->action() == BarrierAction::Out) {
                return discounted_rebate_theta(barrier_last_dtau());
            }
            return vanilla_equivalent_engine().theta();
        }
        if (is_european_knock_in()) {
            // Parity with the discounted-rebate leg's own theta.
            const auto& [ko, van] = european_ki_components();
            return grid_theta_identity(van) + discounted_rebate_theta(ko.last_dtau) -
                   grid_theta_identity(ko);
        }
    }
    const Solution& res = solution();
    const double spot = state_.underlying.initial_value();
    const Index j = spot_grid_index(res.spots, spot);
    if (intrinsic_short_circuit(res.spots, res.values, j).has_value()) {
        return 0.0;
    }
    return grid_theta_identity(res);
}

}  // namespace dp::engines
