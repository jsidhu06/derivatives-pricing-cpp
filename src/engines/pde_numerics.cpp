#include "engines/pde_numerics.hpp"

#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/diagnostics.hpp"
#include "dp/exceptions.hpp"
#include "dp/params.hpp"

#include "detail/log.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <variant>
#include <vector>

namespace dp::engines::pde {

namespace {

constexpr double kTauResolution = 1e12;  // taus are matched at 12 decimal places

/// V(S, t) ~ slope * S * dq_fwd + intercept * df_fwd under risk-neutral pricing.
double continuation_from_affine(double spot,
                                const WingBoundary& wing,
                                double df_fwd,
                                double dq_fwd) {
    return wing.slope * spot * dq_fwd + wing.intercept * df_fwd;
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

}  // namespace

double round12(double x) {
    return std::nearbyint(x * kTauResolution) / kTauResolution;
}

ArrayXd linspace(double start, double stop, int intervals) {
    const double step = (stop - start) / intervals;
    ArrayXd out(intervals + 1);
    for (int i = 0; i <= intervals; ++i) {
        out[i] = start + step * i;
    }
    out[intervals] = stop;
    return out;
}

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
    // Forward-intrinsic asymptote max(+/-(S dq - K df), 0) at each boundary:
    // exact when the boundary is far (in sigma*sqrt(T) units) from the strike
    // on either side, so it also holds when the strike lies outside the
    // domain (the auto vanilla domain need not contain it). On the legacy
    // domains it reduces to the classic put 0 above / call 0 below.
    const double k = *strike;
    const double sign = option_type == OptionType::Call ? 1.0 : -1.0;
    const auto boundary = [&](double s_b) {
        const double continuation = std::max(sign * (s_b * dq_fwd - k * df_fwd), 0.0);
        return early_exercise ? std::max(continuation, std::max(sign * (s_b - k), 0.0))
                              : continuation;
    };
    return {boundary(smin), boundary(smax)};
}

namespace {

/// Auto domain margin (and widened barrier far sides) in standard
/// deviations of ln(S_T) around the spot and forward.
constexpr double kDomainNumStd = 6.0;
/// Boundaries closer than this (standard deviations) warn; barrier far
/// sides closer than this are widened (LogSpot, auto).
constexpr double kMinDomainNumStd = 3.0;
/// Warn below this many grid nodes per standard deviation at the spot.
constexpr double kMinNodesPerStd = 3.0;

}  // namespace

double log_forward(const detail::ValuationState& state) {
    const double spot = state.underlying.initial_value();
    const double ttm = state.maturity_year_fraction();
    const auto& curve = state.discount_curve();
    const auto& q_curve = state.underlying.dividend_curve();
    const double df_r = curve.df(ttm);
    const double df_q = q_curve.has_value() ? q_curve->df(ttm) : 1.0;
    double pv_divs = 0.0;
    for (const auto& [tau, amount] : dividend_tau_schedule(state)) {
        pv_divs += amount * curve.df(std::max(ttm - tau, 0.0));
    }
    return std::log(std::max(spot - pv_divs, 1.0e-8 * spot) * df_q / df_r);
}

Domain fitted_vanilla_domain(const detail::ValuationState& state,
                             double strike,
                             double log_fwd,
                             PdeSpaceGrid space_grid) {
    const double spot = state.underlying.initial_value();
    const double sd = state.underlying.volatility() * std::sqrt(state.maturity_year_fraction());
    const double log_spot = std::log(spot);
    const double smin = std::exp(std::min(log_spot, log_fwd) - (kDomainNumStd * sd));
    double smax = std::exp(std::max(log_spot, log_fwd) + (kDomainNumStd * sd));
    if (space_grid == PdeSpaceGrid::Spot) {
        smax = std::min(smax, kDefaultSmaxMult * std::max(spot, strike));
    }
    return {smin, smax};
}

Domain barrier_domain(const detail::ValuationState& state,
                      double strike,
                      double log_fwd,
                      const PdeParams& params) {
    const double spot = state.underlying.initial_value();
    const double mult = resolved_smax_mult(params);
    const double ref = std::max(spot, strike);
    double smax = mult * ref;
    double smin = params.space_grid == PdeSpaceGrid::Spot ? 0.0 : std::max(ref / mult, 1.0e-8);
    if (!params.smax_mult.has_value() && params.space_grid == PdeSpaceGrid::LogSpot) {
        const double sd = state.underlying.volatility() * std::sqrt(state.maturity_year_fraction());
        const double centre_lo = std::min(std::log(spot), log_fwd);
        const double centre_hi = std::max(std::log(spot), log_fwd);
        if (centre_lo - std::log(smin) < kMinDomainNumStd * sd) {
            smin = std::exp(centre_lo - (kDomainNumStd * sd));
        }
        if (std::log(smax) - centre_hi < kMinDomainNumStd * sd) {
            smax = std::exp(centre_hi + (kDomainNumStd * sd));
        }
    }
    return {smin, smax};
}

void warn_if_domain_truncated(std::optional<double> smin,
                              std::optional<double> smax,
                              double spot,
                              double log_fwd,
                              double time_to_maturity,
                              double volatility) {
    const double sd = volatility * std::sqrt(time_to_maturity);
    double reach = std::numeric_limits<double>::infinity();
    if (smin.has_value() && *smin > 0.0) {
        reach = std::min(reach, (std::min(std::log(spot), log_fwd) - std::log(*smin)) / sd);
    }
    if (smax.has_value()) {
        reach = std::min(reach, (std::log(*smax) - std::max(std::log(spot), log_fwd)) / sd);
    }
    if (reach < kMinDomainNumStd) {
        warn(std::format(
            "PDE domain boundary is {:.2g} standard deviations of ln(S_T) (sigma*sqrt(T) = "
            "{:.3g}) from the spot/forward; the price is biased by the far-field boundary "
            "condition. Leave smax_mult unset with the LOG_SPOT grid (auto domain) or "
            "increase smax_mult.",
            reach,
            sd));
    }
}

void warn_if_under_resolved(double spot,
                            double spacing,
                            double time_to_maturity,
                            double volatility,
                            PdeSpaceGrid space_grid) {
    const double sd = volatility * std::sqrt(time_to_maturity);
    const double nodes_per_std =
        space_grid == PdeSpaceGrid::LogSpot ? sd / spacing : spot * sd / spacing;
    if (nodes_per_std < kMinNodesPerStd) {
        warn(std::format(
            "PDE grid has {:.2g} nodes per standard deviation of ln(S_T) (sigma*sqrt(T) = "
            "{:.3g}); the price is unreliable. Increase spot_steps or narrow the domain "
            "(smax_mult).",
            nodes_per_std,
            sd));
    }
}

SpaceGridBuild build_log_grid(double spot,
                              double strike,
                              double time_to_maturity,
                              double volatility,
                              double smax_mult,
                              int spot_steps,
                              int time_steps,
                              PdeMethod method,
                              std::optional<double> anchor_spot,
                              bool anchor_half_step,
                              std::optional<Domain> domain) {
    if (anchor_half_step && !anchor_spot.has_value()) {
        throw ValidationError("anchor_half_step requires anchor_spot to be provided");
    }
    const auto [smin, smax] = domain.value_or(Domain{
        std::max(std::max(spot, strike) / smax_mult, 1.0e-8), smax_mult * std::max(spot, strike)});
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

SpaceGridBuild build_spot_grid(double smin,
                               double smax,
                               int spot_steps,
                               std::optional<double> anchor_spot,
                               bool anchor_half_step) {
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

// ── Parameters and validation ───────────────────────────────────────

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

// ── Grid Greek stencils ─────────────────────────────────────────────

Index spot_grid_index(const ArrayXd& S, double spot) {
    const Index j = std::distance(S.begin(), std::lower_bound(S.begin(), S.end(), spot));
    return std::max<Index>(1, std::min<Index>(j, S.size() - 2));
}

double grid_delta_at_spot(const ArrayXd& S, const ArrayXd& V, Index j, double spot) {
    const double x0 = S[j - 1];
    const double x1 = S[j];
    const double x2 = S[j + 1];
    return V[j - 1] * (2.0 * spot - x1 - x2) / ((x0 - x1) * (x0 - x2)) +
           V[j] * (2.0 * spot - x0 - x2) / ((x1 - x0) * (x1 - x2)) +
           V[j + 1] * (2.0 * spot - x0 - x1) / ((x2 - x0) * (x2 - x1));
}

double grid_gamma_safe(const ArrayXd& S, const ArrayXd& V, Index j, double spot) {
    const double parabolic = grid_gamma_at_index(S, V, j);
    const double cubic = grid_gamma_at_spot(S, V, j, spot);
    const double scale = std::max({std::abs(parabolic), std::abs(cubic), 1e-6});
    if (std::abs(cubic - parabolic) > kGammaCubicParabolicRelTol * scale) {
        return parabolic;
    }
    return cubic;
}

double grid_value_at_spot(const ArrayXd& S, const ArrayXd& V, Index j, double spot) {
    const double x0 = S[j - 1];
    const double x1 = S[j];
    const double x2 = S[j + 1];
    return V[j - 1] * (spot - x1) * (spot - x2) / ((x0 - x1) * (x0 - x2)) +
           V[j] * (spot - x0) * (spot - x2) / ((x1 - x0) * (x1 - x2)) +
           V[j + 1] * (spot - x0) * (spot - x1) / ((x2 - x0) * (x2 - x1));
}

}  // namespace dp::engines::pde
