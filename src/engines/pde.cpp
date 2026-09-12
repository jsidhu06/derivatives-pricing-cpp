#include "engines/pde.hpp"

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
#include <map>
#include <optional>
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
    ArrayXd grid;   ///< PDE coordinate (spot, or log-spot z).
    ArrayXd spots;  ///< exp(grid) on log grids; == grid on spot grids.
    double dx;      ///< dS or dz.
};

/// Log-spot grid (no barrier anchoring — that arrives with the barrier
/// engine). Explicit schemes target Hull's stability spacing
/// dz = vol * sqrt(3 dt) when it covers the domain.
SpaceGridBuild build_log_grid(double spot,
                              double strike,
                              double time_to_maturity,
                              double volatility,
                              double smax_mult,
                              int spot_steps,
                              int time_steps,
                              PdeMethod method) {
    const double smax = smax_mult * std::max(spot, strike);
    const double smin = std::max(std::max(spot, strike) / smax_mult, 1.0e-8);
    const double zmin_target = std::log(smin);
    const double zmax_target = std::log(smax);

    double dz = 0.0;
    double zmin = zmin_target;
    if (method == PdeMethod::Explicit || method == PdeMethod::ExplicitHull) {
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

    ArrayXd z(spot_steps + 1);
    for (int i = 0; i <= spot_steps; ++i) {
        z[i] = zmin + dz * i;
    }
    return {.grid = z, .spots = z.exp(), .dx = dz};
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
    const double interp_spot = pricing_div.has_value() ? spot - *pricing_div : spot;
    const double price = interp_linear(interp_spot, S, v);

    solution_.emplace(Solution{
        .price = price,
        .spots = std::move(S),
        .values = std::move(v),
        .values_prev = std::move(v_prev),
        .last_dtau = last_dtau,
    });
    return *solution_;
}

double PdeEngine::present_value() const {
    return solution().price;
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
    // American vanilla only: in the exercise region the local value function
    // is the intrinsic, so delta = ±1 and gamma = 0 exactly, while the PDE
    // stencils are polluted by PSOR oscillations.
    if (!std::holds_alternative<VanillaSpec>(state_.spec) ||
        state_.exercise_type() != ExerciseType::American) {
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
    const Solution& r = solution();
    const double spot = state_.underlying.initial_value();
    const Index j = spot_grid_index(r.spots, spot);
    if (const auto sc = intrinsic_short_circuit(r.spots, r.values, j); sc.has_value()) {
        return sc->first;
    }
    return grid_delta_at_spot(r.spots, r.values, j, spot);
}

double PdeEngine::gamma() const {
    const Solution& r = solution();
    const double spot = state_.underlying.initial_value();
    const Index j = spot_grid_index(r.spots, spot);
    if (const auto sc = intrinsic_short_circuit(r.spots, r.values, j); sc.has_value()) {
        return sc->second;
    }
    return grid_gamma_safe(r.spots, r.values, j, spot);
}

double PdeEngine::theta() const {
    // Black-Scholes PDE identity: theta = rV - (r-q) S delta - 0.5 sigma^2 S^2 gamma,
    // per calendar day. Matches the accuracy of the grid delta/gamma.
    const Solution& res = solution();
    const double spot = state_.underlying.initial_value();
    const Index j = spot_grid_index(res.spots, spot);
    if (intrinsic_short_circuit(res.spots, res.values, j).has_value()) {
        return 0.0;
    }
    if (res.last_dtau <= 0.0) {
        return 0.0;
    }
    const double pv_at_spot = grid_value_at_spot(res.spots, res.values, j, spot);
    const double delta_val = grid_delta_at_spot(res.spots, res.values, j, spot);
    const double gamma_val = grid_gamma_safe(res.spots, res.values, j, spot);

    const double sigma = state_.underlying.volatility();
    const double dt_probe = std::max(res.last_dtau, 1.0e-8);
    const double r = state_.discount_curve().forward_rate(0.0, dt_probe);
    const auto& q_curve = state_.underlying.dividend_curve();
    const double q = q_curve.has_value() ? q_curve->forward_rate(0.0, dt_probe) : 0.0;

    const double theta_annual =
        r * pv_at_spot - (r - q) * spot * delta_val - 0.5 * sigma * sigma * spot * spot * gamma_val;
    return theta_annual / kDaysPerYear;
}

}  // namespace dp::engines
