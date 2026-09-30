#include "engines/mc.hpp"

#include "dp/dates.hpp"
#include "dp/diagnostics.hpp"
#include "dp/exceptions.hpp"
#include "dp/params.hpp"

#include "detail/barrier.hpp"
#include "detail/log.hpp"

#include <Eigen/Dense>
#include <pcg_random.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <variant>
#include <vector>

namespace dp::engines {

namespace {

using Eigen::ArrayXd;
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

constexpr double kDaysPerYear = 365.0;

double year_days(DayCountConvention dcc) {
    switch (dcc) {
        case DayCountConvention::Act360:
        case DayCountConvention::Thirty360Us:
            return 360.0;
        case DayCountConvention::Act365_25:
            return 365.25;
        case DayCountConvention::Act365F:
            return 365.0;
    }
    throw ConfigurationError("Unhandled DayCountConvention in year_days");
}

const MonteCarloParams& mc_params(const detail::ValuationState& state) {
    const auto* params =
        state.params.has_value() ? std::get_if<MonteCarloParams>(&*state.params) : nullptr;
    if (params == nullptr) {
        throw ConfigurationError("McEngine requires MonteCarloParams");
    }
    return *params;
}

/// Index of the grid entry nearest to `target` (year-fraction match with a
/// ~1 second tolerance under the day-count convention).
Index resolve_time_index(const std::vector<DateTime>& dates,
                         const ArrayXd& time_grid,
                         DateTime target,
                         std::string_view label,
                         const detail::ValuationState& state) {
    if (dates.empty()) {
        throw ValidationError(std::string(label) + ": time_grid is empty.");
    }
    if (const auto it = std::find(dates.begin(), dates.end(), target); it != dates.end()) {
        return std::distance(dates.begin(), it);
    }
    const double target_yf = year_fraction(dates.front(), target, state.day_count_convention());
    Index best = 0;
    double best_diff = std::abs(time_grid[0] - target_yf);
    for (Index i = 1; i < time_grid.size(); ++i) {
        const double diff = std::abs(time_grid[i] - target_yf);
        if (diff < best_diff) {
            best = i;
            best_diff = diff;
        }
    }
    if (best_diff > 1.0 / (year_days(state.day_count_convention()) * 86400.0)) {
        throw ValidationError(std::string(label) + " not found in underlying time_grid.");
    }
    return best;
}

void warn_if_high_std_error(const ArrayXd& pv_pathwise,
                            double pv_mean,
                            const MonteCarloParams& params,
                            std::string_view label) {
    if (!params.std_error_warn_ratio.has_value()) {
        return;
    }
    const Index n = pv_pathwise.size();
    if (n < 2) {
        return;
    }
    const double mean = pv_pathwise.mean();
    const double variance =
        (pv_pathwise - mean).square().sum() / static_cast<double>(n - 1);  // ddof = 1
    const double std_error = std::sqrt(variance / static_cast<double>(n));
    const double ratio = std_error / std::max(std::abs(pv_mean), 1.0e-12);
    detail::logger()->debug(
        "MC {} std_error={:.6g} ratio={:.6g} paths={}", label, std_error, ratio, n);
    if (ratio > *params.std_error_warn_ratio) {
        detail::logger()->warn(
            "MC {} standard error high: std_error={:.6g} ratio={:.6g} (>{:.3g}) paths={}",
            label,
            std_error,
            ratio,
            *params.std_error_warn_ratio,
            n);
    }
}

/// Laguerre design matrix, shape (n, deg + 1); standard recurrence.
MatrixXd laguerre_basis(const ArrayXd& x, int deg) {
    MatrixXd basis(x.size(), deg + 1);
    basis.col(0).setOnes();
    if (deg >= 1) {
        basis.col(1) = (1.0 - x).matrix();
    }
    for (int k = 1; k < deg; ++k) {
        basis.col(k + 1) =
            (((2.0 * k + 1.0 - x) * basis.col(k).array() - k * basis.col(k - 1).array()) /
             (k + 1.0))
                .matrix();
    }
    return basis;
}

/// Ridge-regularised LSM continuation estimate; zero for OTM paths, mean
/// fallback when the ITM cross-section is too thin for a stable fit.
ArrayXd ridge_lsm_continuation(const ArrayXd& spots,
                               const ArrayXd& discounted_next,
                               const Eigen::Array<bool, Eigen::Dynamic, 1>& itm,
                               std::optional<double> strike,
                               const MonteCarloParams& params) {
    ArrayXd continuation = ArrayXd::Zero(spots.size());
    const Index n_itm = itm.count();
    if (n_itm == 0) {
        return continuation;
    }

    ArrayXd spot_itm(n_itm);
    VectorXd y_itm(n_itm);
    Index k = 0;
    for (Index i = 0; i < spots.size(); ++i) {
        if (itm[i]) {
            spot_itm[k] = spots[i];
            y_itm[k] = discounted_next[i];
            ++k;
        }
    }

    if (n_itm < std::max<Index>(params.min_itm, params.deg + 1)) {
        const double mean = y_itm.mean();
        for (Index i = 0; i < spots.size(); ++i) {
            if (itm[i]) {
                continuation[i] = mean;
            }
        }
        return continuation;
    }

    const double normaliser = strike.has_value() ? *strike : std::max(spot_itm.mean(), 1e-12);
    const MatrixXd X = laguerre_basis(spot_itm / normaliser, params.deg);
    const MatrixXd xtx =
        X.transpose() * X + params.ridge_lambda * MatrixXd::Identity(X.cols(), X.cols());
    const VectorXd beta = xtx.ldlt().solve(X.transpose() * y_itm);
    const VectorXd fitted = X * beta;

    k = 0;
    for (Index i = 0; i < spots.size(); ++i) {
        if (itm[i]) {
            continuation[i] = fitted[k++];
        }
    }
    return continuation;
}

/// LSM value at the pricing date. Spot is constant across paths there, so
/// the exercise decision compares two scalars; `immediate_exercise` is
/// empty when no exercise right exists yet (untriggered knock-in).
ArrayXd time_zero_lsm_value(ArrayXd discounted, std::optional<double> immediate_exercise) {
    if (immediate_exercise.has_value() && *immediate_exercise > discounted.mean()) {
        return ArrayXd::Constant(discounted.size(), *immediate_exercise);
    }
    return discounted;
}

// ── Barrier helpers ─────────────────────────────────────────────────

/// Paths observed at or through the barrier.
BoolArray barrier_hit(const ArrayXd& spots, double barrier, bool is_up) {
    if (is_up) {
        return spots >= barrier;
    }
    return spots <= barrier;
}

/// Fold one monitoring step's hits into the cumulative state, stamping
/// `step` as the first hit for paths not already knocked.
void record_hits(BoolArray& ever_hit,
                 IndexArray& first_hit_step,
                 const BoolArray& step_hit,
                 Index step) {
    for (Index p = 0; p < ever_hit.size(); ++p) {
        if (step_hit[p] && !ever_hit[p]) {
            first_hit_step[p] = step;
        }
        ever_hit[p] = ever_hit[p] || step_hit[p];
    }
}

/// PCG64 stream selector for the American-barrier Bernoulli bridge draws,
/// independent of the diffusion stream sharing the same seed (mirrors
/// numpy's `SeedSequence.spawn` in the Python reference).
constexpr std::uint64_t kBridgeStream = 0x42524447;  // "BRDG"

/// Uniform in the open interval (0, 1) — see simulation.cpp.
double open_uniform(pcg64& rng) {
    constexpr double kScale = 0x1.0p-53;
    return (static_cast<double>(rng() >> 11) + 0.5) * kScale;
}

/// Conditional probability that a log-normal path crosses the barrier
/// between two observed endpoints (Brownian-bridge reflection result).
/// At sigma == 0 the deterministic path is monotone, so only endpoint
/// crossings (caught by the mask) can occur.
ArrayXd brownian_bridge_hit_prob(const ArrayXd& spot_start,
                                 const ArrayXd& spot_end,
                                 double barrier,
                                 double sigma,
                                 double dt_step,
                                 bool is_up) {
    ArrayXd p = ArrayXd::Zero(spot_start.size());
    const bool sigma_positive = sigma > 0.0;
    const double denom = sigma * sigma * dt_step;
    for (Index i = 0; i < spot_start.size(); ++i) {
        const bool crossed = is_up ? (spot_start[i] >= barrier || spot_end[i] >= barrier)
                                   : (spot_start[i] <= barrier || spot_end[i] <= barrier);
        if (crossed) {
            p[i] = 1.0;
        } else if (sigma_positive) {
            const double log_a =
                is_up ? std::log(barrier / spot_start[i]) : std::log(spot_start[i] / barrier);
            const double log_b =
                is_up ? std::log(barrier / spot_end[i]) : std::log(spot_end[i] / barrier);
            p[i] = std::clamp(std::exp(-2.0 * log_a * log_b / denom), 0.0, 1.0);
        }
    }
    return p;
}

/// Positive log-distance from the absorbing barrier (0 on the dead side).
ArrayXd barrier_log_distance(const ArrayXd& spots, double barrier, bool is_up) {
    const ArrayXd safe = spots.cwiseMax(1.0e-12);
    if (is_up) {
        return (barrier / safe).log().cwiseMax(0.0);
    }
    return (safe / barrier).log().cwiseMax(0.0);
}

/// Barrier-aware design matrix: the Laguerre basis in moneyness augmented
/// with barrier-distance features, so the regression can represent the
/// sharp continuation cliff near the absorbing boundary.
MatrixXd barrier_lsm_design_matrix(
    const ArrayXd& spots, double strike, double barrier, bool is_up, int deg) {
    const ArrayXd x = spots / strike;
    const MatrixXd base = laguerre_basis(x, deg);
    const ArrayXd dist = barrier_log_distance(spots, barrier, is_up);
    const ArrayXd inv_dist = dist.cwiseMax(5.0e-2).inverse();
    const ArrayXd near_barrier = (-4.0 * dist).exp();

    const Index extra = deg >= 2 ? 6 : 5;
    MatrixXd design(spots.size(), base.cols() + extra);
    design.leftCols(base.cols()) = base;
    Index c = base.cols();
    design.col(c++) = dist.matrix();
    design.col(c++) = dist.square().matrix();
    design.col(c++) = inv_dist.matrix();
    design.col(c++) = near_barrier.matrix();
    design.col(c++) = (x * near_barrier).matrix();
    if (deg >= 2) {
        design.col(c) = (base.col(1).array() * near_barrier).matrix();
    }
    return design;
}

/// Fit ridge regression on (X, y) and predict on X.
VectorXd ridge_fit_predict(const MatrixXd& design, const VectorXd& y, double ridge_lambda) {
    const MatrixXd xtx = design.transpose() * design +
                         ridge_lambda * MatrixXd::Identity(design.cols(), design.cols());
    const VectorXd beta = xtx.ldlt().solve(design.transpose() * y);
    return design * beta;
}

/// numpy's default (linear-interpolation) quantile of `sorted_values`.
double quantile_linear(ArrayXd values, double q) {
    std::sort(values.begin(), values.end());
    const double h = q * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<Index>(std::floor(h));
    const Index hi = std::min<Index>(lo + 1, values.size() - 1);
    return values[lo] + (h - static_cast<double>(lo)) * (values[hi] - values[lo]);
}

/// Barrier-aware continuation estimate for American knock-out LSM: the
/// barrier-distance basis plus a near/far split so the fit can track the
/// continuation cliff without distorting the smooth far region.
ArrayXd ridge_barrier_lsm_continuation(const ArrayXd& spots,
                                       const ArrayXd& discounted_next,
                                       const BoolArray& itm,
                                       double strike,
                                       double barrier,
                                       bool is_up,
                                       const MonteCarloParams& params) {
    ArrayXd continuation = ArrayXd::Zero(spots.size());
    const Index n_itm = itm.count();
    if (n_itm == 0) {
        return continuation;
    }

    ArrayXd spot_itm(n_itm);
    VectorXd y_itm(n_itm);
    Index k = 0;
    for (Index i = 0; i < spots.size(); ++i) {
        if (itm[i]) {
            spot_itm[k] = spots[i];
            y_itm[k] = discounted_next[i];
            ++k;
        }
    }

    const MatrixXd design = barrier_lsm_design_matrix(spot_itm, strike, barrier, is_up, params.deg);
    const Index min_regression_paths = std::max<Index>(params.min_itm, design.cols() + 2);
    VectorXd fitted(n_itm);
    if (n_itm < min_regression_paths) {
        fitted.setConstant(y_itm.mean());
    } else {
        const ArrayXd dist_itm = barrier_log_distance(spot_itm, barrier, is_up);
        const double split_threshold = std::min(quantile_linear(dist_itm, 0.35), 0.25);
        const BoolArray near_mask = dist_itm <= split_threshold;
        const Index n_near = near_mask.count();
        const Index n_far = n_itm - n_near;
        const Index min_bucket = std::max<Index>(12, design.cols() + 1);

        if (n_near < min_bucket || n_far < min_bucket) {
            fitted = ridge_fit_predict(design, y_itm, params.ridge_lambda);
        } else {
            // Fit near- and far-of-barrier buckets independently.
            MatrixXd design_near(n_near, design.cols());
            MatrixXd design_far(n_far, design.cols());
            VectorXd y_near(n_near);
            VectorXd y_far(n_far);
            Index a = 0;
            Index b = 0;
            for (Index i = 0; i < n_itm; ++i) {
                if (near_mask[i]) {
                    design_near.row(a) = design.row(i);
                    y_near[a++] = y_itm[i];
                } else {
                    design_far.row(b) = design.row(i);
                    y_far[b++] = y_itm[i];
                }
            }
            const VectorXd fit_near = ridge_fit_predict(design_near, y_near, params.ridge_lambda);
            const VectorXd fit_far = ridge_fit_predict(design_far, y_far, params.ridge_lambda);
            a = 0;
            b = 0;
            for (Index i = 0; i < n_itm; ++i) {
                fitted[i] = near_mask[i] ? fit_near[a++] : fit_far[b++];
            }
        }
    }

    k = 0;
    for (Index i = 0; i < spots.size(); ++i) {
        if (itm[i]) {
            continuation[i] = fitted[k++];
        }
    }
    return continuation;
}

}  // namespace

McEngine::McEngine(const detail::ValuationState& state) : state_(state) {
    const auto* barrier = std::get_if<BarrierSpec>(&state_.spec);
    if (barrier == nullptr) {
        return;
    }
    if (barrier->monitoring() == BarrierMonitoring::Continuous && !state_.underlying.is_gbm()) {
        throw UnsupportedFeatureError(
            "Continuous barrier monitoring requires GBMProcess (log-normal paths). Use "
            "DISCRETE monitoring for non-GBM processes.");
    }
    if (barrier->action() == BarrierAction::Out &&
        barrier->exercise_type() == ExerciseType::American &&
        !detail::barrier_triggered_at_inception(state_)) {
        const bool down_out_put = barrier->direction() == BarrierDirection::Down &&
                                  barrier->option_type() == OptionType::Put;
        const bool up_out_call = barrier->direction() == BarrierDirection::Up &&
                                 barrier->option_type() == OptionType::Call;
        if (down_out_put || up_out_call) {
            warn(
                "American barrier Monte Carlo LSM can show significant downward bias for "
                "down-and-out puts and up-and-out calls. Prefer PDE_FD for production "
                "pricing.");
        }
    }
}

const SimulationResult& McEngine::simulation() const {
    if (!simulation_.has_value()) {
        simulation_.emplace(state_.underlying.simulate(mc_params(state_).random_seed));
    }
    return *simulation_;
}

ArrayXd McEngine::payoff_at(const ArrayXd& spots) const {
    if (const auto* custom = std::get_if<PayoffSpec>(&state_.spec)) {
        ArrayXd out(spots.size());
        for (Index i = 0; i < spots.size(); ++i) {
            out[i] = custom->payoff(spots[i]);
        }
        return out;
    }
    const double strike = state_.strike().value_or(0.0);
    if (state_.option_type == OptionType::Call) {
        return (spots - strike).cwiseMax(0.0);
    }
    return (strike - spots).cwiseMax(0.0);
}

McEngine::Terminal McEngine::terminal() const {
    const SimulationResult& sim = simulation();
    const Index idx =
        resolve_time_index(sim.dates, sim.time_grid, state_.maturity(), "maturity", state_);
    const double ttm = state_.maturity_year_fraction();
    return {.spot = sim.paths.col(idx).array(),
            .index = idx,
            .ttm = ttm,
            .df = state_.discount_curve().df(ttm)};
}

ArrayXd McEngine::effective_terminal_z(Index maturity_index, double ttm) const {
    const SimulationResult& sim = simulation();
    const ArrayXd deltas =
        sim.time_grid.segment(1, maturity_index) - sim.time_grid.head(maturity_index);
    const double covered = deltas.sum();
    if (std::abs(covered - ttm) > 1e-8 + 1e-5 * std::abs(ttm)) {
        throw NumericalError(
            std::format("Time deltas sum {} does not match time to maturity {}", covered, ttm));
    }
    ArrayXd z = ArrayXd::Zero(sim.paths.rows());
    for (Index t = 0; t < maturity_index; ++t) {
        z += std::sqrt(deltas[t]) * sim.normals.col(t).array();
    }
    return z / std::sqrt(ttm);
}

std::vector<Index> McEngine::barrier_monitoring_indices(const SimulationResult& sim,
                                                        Index end_index) const {
    const auto& spec = std::get<BarrierSpec>(state_.spec);
    std::vector<Index> indices;
    if (spec.monitoring() == BarrierMonitoring::Continuous) {
        indices.reserve(static_cast<std::size_t>(end_index) + 1);
        for (Index i = 0; i <= end_index; ++i) {
            indices.push_back(i);
        }
        return indices;
    }

    std::set<Index> unique_indices;
    const auto dates = detail::barrier_monitoring_dates(state_);
    if (!dates.has_value()) {
        throw ValidationError(
            "Discrete barrier monitoring requires num_observations or monitoring_dates.");
    }
    for (const DateTime date : *dates) {
        unique_indices.insert(
            resolve_time_index(sim.dates, sim.time_grid, date, "monitoring date", state_));
    }
    for (const Index index : unique_indices) {
        if (index <= end_index) {
            indices.push_back(index);
        }
    }
    return indices;
}

ArrayXd McEngine::barrier_pathwise_european() const {
    const auto& spec = std::get<BarrierSpec>(state_.spec);
    const SimulationResult& sim = simulation();
    const Index end =
        resolve_time_index(sim.dates, sim.time_grid, state_.maturity(), "maturity", state_);
    const auto& curve = state_.discount_curve();

    const ArrayXd payoff = payoff_at(sim.paths.col(end).array());

    ArrayXd grid_dfs(sim.time_grid.size());
    for (Index i = 0; i < sim.time_grid.size(); ++i) {
        grid_dfs[i] = curve.df(sim.time_grid[i]);
    }

    const BarrierWeights legs = spec.monitoring() == BarrierMonitoring::Continuous
                                    ? continuous_weights(sim, end, grid_dfs)
                                    : discrete_weights(sim, end, grid_dfs);

    return curve.df(state_.maturity_year_fraction()) * payoff * legs.weight + legs.rebate_pv;
}

McEngine::BarrierWeights McEngine::continuous_weights(const SimulationResult& sim,
                                                      Index end,
                                                      const ArrayXd& grid_dfs) const {
    const auto& spec = std::get<BarrierSpec>(state_.spec);
    const Index n_paths = sim.paths.rows();
    const bool is_out = spec.action() == BarrierAction::Out;
    const double rebate = spec.rebate();
    const bool rebate_at_hit = spec.rebate_timing() == RebateTiming::AtHit;

    BarrierWeights legs{.weight = ArrayXd(n_paths), .rebate_pv = ArrayXd::Zero(n_paths)};

    if (detail::barrier_triggered_at_inception(state_)) {
        legs.weight.setConstant(is_out ? 0.0 : 1.0);
        if (rebate > 0.0 && is_out) {
            legs.rebate_pv.setConstant(rebate_at_hit ? rebate : rebate * grid_dfs[end]);
        }
        return legs;
    }

    const auto& curve = state_.discount_curve();
    const double barrier = spec.barrier();
    const bool is_up = spec.direction() == BarrierDirection::Up;
    const double sigma = state_.underlying.volatility();
    const DayCountConvention dcc = state_.day_count_convention();

    ArrayXd survival = ArrayXd::Ones(n_paths);
    for (Index t = 1; t <= end; ++t) {
        const double dt_step = year_fraction(sim.dates[static_cast<std::size_t>(t) - 1],
                                             sim.dates[static_cast<std::size_t>(t)],
                                             dcc);
        const ArrayXd p_hit = brownian_bridge_hit_prob(
            sim.paths.col(t - 1).array(), sim.paths.col(t).array(), barrier, sigma, dt_step, is_up);
        if (rebate > 0.0 && is_out && rebate_at_hit) {
            // The intra-step hit time is unknown: discount to the interval
            // midpoint.
            const double midpoint_df = curve.df(0.5 * (sim.time_grid[t - 1] + sim.time_grid[t]));
            legs.rebate_pv += survival * p_hit * midpoint_df * rebate;
        }
        survival *= 1.0 - p_hit;
    }

    if (is_out) {
        legs.weight = survival;
    } else {
        legs.weight = 1.0 - survival;
    }
    if (rebate > 0.0) {
        if (is_out && !rebate_at_hit) {
            legs.rebate_pv = (1.0 - survival) * rebate * grid_dfs[end];
        } else if (!is_out) {
            legs.rebate_pv = survival * rebate * grid_dfs[end];
        }
    }
    return legs;
}

McEngine::BarrierWeights McEngine::discrete_weights(const SimulationResult& sim,
                                                    Index end,
                                                    const ArrayXd& grid_dfs) const {
    const auto& spec = std::get<BarrierSpec>(state_.spec);
    const Index n_paths = sim.paths.rows();
    const bool is_out = spec.action() == BarrierAction::Out;
    const double rebate = spec.rebate();
    const bool rebate_at_hit = spec.rebate_timing() == RebateTiming::AtHit;
    const bool inception_hit = detail::barrier_triggered_at_inception(state_);

    BoolArray ever_hit = BoolArray::Constant(n_paths, inception_hit);
    IndexArray first_hit_step = IndexArray::Constant(n_paths, inception_hit ? 0 : -1);
    for (const Index idx : barrier_monitoring_indices(sim, end)) {
        if (idx == 0) {
            continue;
        }
        record_hits(ever_hit,
                    first_hit_step,
                    barrier_hit(sim.paths.col(idx).array(),
                                spec.barrier(),
                                spec.direction() == BarrierDirection::Up),
                    idx);
    }

    BarrierWeights legs{.weight = ArrayXd(n_paths), .rebate_pv = ArrayXd::Zero(n_paths)};
    if (is_out) {
        legs.weight = (!ever_hit).cast<double>();
    } else {
        legs.weight = ever_hit.cast<double>();
    }
    if (rebate > 0.0) {
        for (Index p = 0; p < n_paths; ++p) {
            if (is_out && ever_hit[p]) {
                legs.rebate_pv[p] =
                    rebate * (rebate_at_hit ? grid_dfs[first_hit_step[p]] : grid_dfs[end]);
            } else if (!is_out && !ever_hit[p]) {
                legs.rebate_pv[p] = rebate * grid_dfs[end];
            }
        }
    }
    return legs;
}

ArrayXd McEngine::barrier_pathwise_american() const {
    const auto& spec = std::get<BarrierSpec>(state_.spec);
    const SimulationResult& sim = simulation();
    const Index start =
        resolve_time_index(sim.dates, sim.time_grid, state_.pricing_date(), "Pricing date", state_);
    const Index end =
        resolve_time_index(sim.dates, sim.time_grid, state_.maturity(), "maturity", state_);
    const Index n_times = end - start + 1;
    const Index n_paths = sim.paths.rows();
    const bool is_out = spec.action() == BarrierAction::Out;
    const bool inception_hit = detail::barrier_triggered_at_inception(state_);

    const ArrayXd dfs = state_.discount_curve().df(
        (sim.time_grid.segment(start, n_times) - sim.time_grid[start]).eval());

    // KO triggered at inception: dead, no exercise — the rebate PV per its
    // timing rule, bypassing the LSM induction entirely.
    if (is_out && inception_hit) {
        double value = 0.0;
        if (spec.rebate() > 0.0) {
            value = spec.rebate_timing() == RebateTiming::AtHit ? spec.rebate()
                                                                : spec.rebate() * dfs[n_times - 1];
        }
        return ArrayXd::Constant(n_paths, value);
    }

    const BarrierPathState path_state = american_barrier_state(sim, start, n_times);
    ArrayXd values =
        initial_barrier_values(payoff_at(sim.paths.col(end).array()), path_state.ever_hit.back());

    for (Index t = n_times - 2; t >= 1; --t) {
        const ArrayXd spots = sim.paths.col(start + t).array();
        const ArrayXd intrinsic = payoff_at(spots);
        const ArrayXd discounted_next = (dfs[t + 1] / dfs[t]) * values;
        values = is_out
                     ? knock_out_step_values(t, spots, intrinsic, discounted_next, path_state, dfs)
                     : knock_in_step_values(spots,
                                            intrinsic,
                                            discounted_next,
                                            path_state.ever_hit[static_cast<std::size_t>(t)]);
    }

    // t = 0: a knock-in that is not yet triggered has no immediate-exercise
    // right — pure continuation.
    std::optional<double> immediate;
    if (is_out || inception_hit) {
        immediate = payoff_at(sim.paths.col(start).array()).mean();
    }
    return time_zero_lsm_value((dfs[1] / dfs[0]) * values, immediate);
}

McEngine::BarrierPathState McEngine::american_barrier_state(const SimulationResult& sim,
                                                            Index start,
                                                            Index n_times) const {
    const auto& spec = std::get<BarrierSpec>(state_.spec);
    const Index n_paths = sim.paths.rows();
    const double barrier = spec.barrier();
    const bool is_up = spec.direction() == BarrierDirection::Up;
    const bool continuous = spec.monitoring() == BarrierMonitoring::Continuous;
    const bool inception_hit = detail::barrier_triggered_at_inception(state_);

    BarrierPathState path_state{
        .ever_hit = {}, .first_hit_step = IndexArray::Constant(n_paths, inception_hit ? 0 : -1)};
    path_state.ever_hit.reserve(static_cast<std::size_t>(n_times));
    path_state.ever_hit.push_back(BoolArray::Constant(n_paths, inception_hit));

    std::set<Index> monitoring_set;
    for (const Index idx : barrier_monitoring_indices(sim, start + n_times - 1)) {
        monitoring_set.insert(idx);
    }

    // Independent stream for the Bernoulli bridge sampling.
    const auto& seed = mc_params(state_).random_seed;
    pcg64 bridge_rng = seed.has_value() ? pcg64(*seed, kBridgeStream)
                                        : pcg64(pcg_extras::seed_seq_from<std::random_device>{});
    const double sigma = state_.underlying.volatility();
    const DayCountConvention dcc = state_.day_count_convention();

    for (Index t = 1; t < n_times; ++t) {
        path_state.ever_hit.push_back(path_state.ever_hit.back());
        if (!continuous && !monitoring_set.contains(start + t)) {
            continue;
        }
        const auto spots_curr = sim.paths.col(start + t).array();

        BoolArray step_hit(n_paths);
        if (continuous) {
            const double dt_step = year_fraction(sim.dates[static_cast<std::size_t>(start + t) - 1],
                                                 sim.dates[static_cast<std::size_t>(start + t)],
                                                 dcc);
            const ArrayXd p_hit = brownian_bridge_hit_prob(
                sim.paths.col(start + t - 1).array(), spots_curr, barrier, sigma, dt_step, is_up);
            for (Index p = 0; p < n_paths; ++p) {
                step_hit[p] = open_uniform(bridge_rng) < p_hit[p];
            }
        } else {
            step_hit = barrier_hit(spots_curr, barrier, is_up);
        }
        record_hits(path_state.ever_hit.back(), path_state.first_hit_step, step_hit, t);
    }
    return path_state;
}

ArrayXd McEngine::initial_barrier_values(const ArrayXd& intrinsic_end,
                                         const BoolArray& hit_end) const {
    const auto& spec = std::get<BarrierSpec>(state_.spec);
    const double rebate = spec.rebate();
    if (spec.action() == BarrierAction::Out) {
        if (rebate > 0.0 && spec.rebate_timing() == RebateTiming::AtExpiry) {
            return hit_end.select(rebate, intrinsic_end);
        }
        return (!hit_end).cast<double>() * intrinsic_end;
    }
    ArrayXd values = hit_end.cast<double>() * intrinsic_end;
    if (rebate > 0.0) {
        values += rebate * (!hit_end).cast<double>();
    }
    return values;
}

ArrayXd McEngine::knock_out_step_values(Index t,
                                        const ArrayXd& spots,
                                        const ArrayXd& intrinsic,
                                        const ArrayXd& discounted_next,
                                        const BarrierPathState& state,
                                        const ArrayXd& dfs) const {
    const auto& spec = std::get<BarrierSpec>(state_.spec);
    const MonteCarloParams& params = mc_params(state_);
    const BoolArray& hit_t = state.ever_hit[static_cast<std::size_t>(t)];
    const BoolArray alive = !hit_t;
    const BoolArray itm = (intrinsic > 0.0) && alive;

    const ArrayXd continuation =
        params.barrier_aware_basis
            ? ridge_barrier_lsm_continuation(spots,
                                             discounted_next,
                                             itm,
                                             spec.strike(),
                                             spec.barrier(),
                                             spec.direction() == BarrierDirection::Up,
                                             params)
            : ridge_lsm_continuation(spots, discounted_next, itm, spec.strike(), params);
    ArrayXd values = (alive && (intrinsic > continuation)).select(intrinsic, discounted_next);

    const double rebate = spec.rebate();
    if (rebate > 0.0 && spec.rebate_timing() == RebateTiming::AtHit) {
        for (Index p = 0; p < values.size(); ++p) {
            if (state.first_hit_step[p] == t) {
                values[p] = rebate;
            } else if (hit_t[p]) {
                values[p] = 0.0;
            }
        }
        return values;
    }
    if (rebate > 0.0) {
        const double rebate_hold = rebate * (dfs[dfs.size() - 1] / dfs[t]);
        return hit_t.select(rebate_hold, values);
    }
    return hit_t.select(0.0, values);
}

ArrayXd McEngine::knock_in_step_values(const ArrayXd& spots,
                                       const ArrayXd& intrinsic,
                                       const ArrayXd& discounted_next,
                                       const BoolArray& hit_t) const {
    const BoolArray itm_active = (intrinsic > 0.0) && hit_t;
    const ArrayXd continuation = ridge_lsm_continuation(spots,
                                                        discounted_next,
                                                        itm_active,
                                                        std::get<BarrierSpec>(state_.spec).strike(),
                                                        mc_params(state_));
    return (hit_t && (intrinsic > continuation)).select(intrinsic, discounted_next);
}

ArrayXd McEngine::present_value_pathwise() const {
    if (std::holds_alternative<BarrierSpec>(state_.spec)) {
        return state_.exercise_type() == ExerciseType::European ? barrier_pathwise_european()
                                                                : barrier_pathwise_american();
    }
    if (state_.exercise_type() == ExerciseType::European) {
        const Terminal at_expiry = terminal();
        return at_expiry.df * payoff_at(at_expiry.spot);
    }

    // Longstaff-Schwartz backward induction from maturity to the pricing date.
    const SimulationResult& sim = simulation();
    const Index start =
        resolve_time_index(sim.dates, sim.time_grid, state_.pricing_date(), "Pricing date", state_);
    const Index end =
        resolve_time_index(sim.dates, sim.time_grid, state_.maturity(), "maturity", state_);
    const MonteCarloParams& params = mc_params(state_);
    const auto strike = state_.strike();

    const ArrayXd dfs = state_.discount_curve().df(
        (sim.time_grid.segment(start, end - start + 1) - sim.time_grid[start]).eval());

    ArrayXd values = payoff_at(sim.paths.col(end).array());
    for (Index t = end - start - 1; t >= 1; --t) {
        const double df_step = dfs[t + 1] / dfs[t];
        const ArrayXd spots = sim.paths.col(start + t).array();
        const ArrayXd intrinsic = payoff_at(spots);
        const Eigen::Array<bool, Eigen::Dynamic, 1> itm = intrinsic > 0.0;
        const ArrayXd discounted_next = df_step * values;
        const ArrayXd continuation =
            ridge_lsm_continuation(spots, discounted_next, itm, strike, params);
        values = (intrinsic > continuation).select(intrinsic, discounted_next);
    }

    return time_zero_lsm_value((dfs[1] / dfs[0]) * values,
                               payoff_at(sim.paths.col(start).array()).mean());
}

double McEngine::present_value() const {
    const ArrayXd pv_pathwise = present_value_pathwise();
    const double pv = pv_pathwise.mean();
    const bool european = state_.exercise_type() == ExerciseType::European;
    const bool barrier = std::holds_alternative<BarrierSpec>(state_.spec);
    std::string_view label = european ? "European" : "American";
    if (barrier) {
        label = european ? "Barrier European" : "Barrier American";
    }
    warn_if_high_std_error(pv_pathwise, pv, mc_params(state_), label);
    return pv;
}

// ── Pathwise / likelihood-ratio Greeks (European vanillas; the facade
//    validates eligibility before dispatching here) ──────────────────

std::pair<double, double> McEngine::risk_free_and_div_rates(double ttm) const {
    const double r = -std::log(state_.discount_curve().df(ttm)) / ttm;
    const auto& q_curve = state_.underlying.dividend_curve();
    const double q = q_curve.has_value() ? -std::log(q_curve->df(ttm)) / ttm : 0.0;
    return {r, q};
}

double McEngine::delta_pathwise() const {
    const Terminal at_expiry = terminal();
    const double s0 = state_.underlying.initial_value();
    const double strike = state_.strike().value_or(0.0);
    if (state_.option_type == OptionType::Call) {
        return (at_expiry.df * (at_expiry.spot > strike).cast<double>() * (at_expiry.spot / s0))
            .mean();
    }
    return (-at_expiry.df * (at_expiry.spot < strike).cast<double>() * (at_expiry.spot / s0))
        .mean();
}

double McEngine::vega_pathwise() const {
    const Terminal at_expiry = terminal();
    const ArrayXd z = effective_terminal_z(at_expiry.index, at_expiry.ttm);
    const double sigma = state_.underlying.volatility();
    const double strike = state_.strike().value_or(0.0);
    const ArrayXd sensitivity =
        at_expiry.spot * (-sigma * at_expiry.ttm + std::sqrt(at_expiry.ttm) * z);
    if (state_.option_type == OptionType::Call) {
        return (at_expiry.df * (at_expiry.spot > strike).cast<double>() * sensitivity).mean() /
               100.0;
    }
    return (-at_expiry.df * (at_expiry.spot < strike).cast<double>() * sensitivity).mean() / 100.0;
}

double McEngine::theta_pathwise() const {
    const Terminal at_expiry = terminal();
    const ArrayXd z = effective_terminal_z(at_expiry.index, at_expiry.ttm);
    const double sigma = state_.underlying.volatility();
    const double strike = state_.strike().value_or(0.0);
    const auto [r, q] = risk_free_and_div_rates(at_expiry.ttm);
    const double a = r - q - 0.5 * sigma * sigma;

    const ArrayXd payoff = payoff_at(at_expiry.spot);
    const ArrayXd indicator = state_.option_type == OptionType::Call
                                  ? (at_expiry.spot > strike).cast<double>().eval()
                                  : (-(at_expiry.spot < strike).cast<double>()).eval();
    const ArrayXd sensitivity =
        indicator * at_expiry.spot * (a + sigma * z / (2.0 * std::sqrt(at_expiry.ttm)));
    return (-at_expiry.df * (sensitivity - r * payoff)).mean() / kDaysPerYear;
}

double McEngine::rho_pathwise() const {
    const Terminal at_expiry = terminal();
    const double strike = state_.strike().value_or(0.0);
    if (state_.option_type == OptionType::Call) {
        return (at_expiry.df * at_expiry.ttm * strike * (at_expiry.spot > strike).cast<double>())
                   .mean() *
               0.01;
    }
    return (-at_expiry.df * at_expiry.ttm * strike * (at_expiry.spot < strike).cast<double>())
               .mean() *
           0.01;
}

namespace {
void require_positive_sigma_for_lr(double sigma, std::string_view greek) {
    if (sigma <= 0.0) {
        throw UnsupportedFeatureError(std::format(
            "Likelihood-ratio {} is undefined at sigma=0 (the lognormal density degenerates "
            "to a Dirac at the forward, so the score function diverges). Use "
            "GreekCalculationMethod.NUMERICAL or PATHWISE instead, or call BSM analytical "
            "greeks.",
            greek));
    }
}
}  // namespace

double McEngine::delta_lr() const {
    const double sigma = state_.underlying.volatility();
    require_positive_sigma_for_lr(sigma, "delta");
    const Terminal at_expiry = terminal();
    const ArrayXd z = effective_terminal_z(at_expiry.index, at_expiry.ttm);
    const double s0 = state_.underlying.initial_value();
    const ArrayXd payoff = payoff_at(at_expiry.spot);
    return (at_expiry.df * payoff * z / (sigma * std::sqrt(at_expiry.ttm) * s0)).mean();
}

double McEngine::vega_lr() const {
    const double sigma = state_.underlying.volatility();
    require_positive_sigma_for_lr(sigma, "vega");
    const Terminal at_expiry = terminal();
    const ArrayXd z = effective_terminal_z(at_expiry.index, at_expiry.ttm);
    const ArrayXd payoff = payoff_at(at_expiry.spot);
    const ArrayXd score = (z.square() - 1.0) / sigma - z * std::sqrt(at_expiry.ttm);
    return (at_expiry.df * payoff * score).mean() / 100.0;
}

double McEngine::theta_lr() const {
    const double sigma = state_.underlying.volatility();
    require_positive_sigma_for_lr(sigma, "theta");
    const Terminal at_expiry = terminal();
    const ArrayXd z = effective_terminal_z(at_expiry.index, at_expiry.ttm);
    const auto [r, q] = risk_free_and_div_rates(at_expiry.ttm);
    const double a = r - q - 0.5 * sigma * sigma;
    const ArrayXd payoff = payoff_at(at_expiry.spot);
    const ArrayXd score =
        (z.square() - 1.0) / (2.0 * at_expiry.ttm) + a * z / (sigma * std::sqrt(at_expiry.ttm)) - r;
    return (-at_expiry.df * payoff * score).mean() / kDaysPerYear;
}

double McEngine::rho_lr() const {
    const double sigma = state_.underlying.volatility();
    require_positive_sigma_for_lr(sigma, "rho");
    const Terminal at_expiry = terminal();
    const ArrayXd z = effective_terminal_z(at_expiry.index, at_expiry.ttm);
    const ArrayXd payoff = payoff_at(at_expiry.spot);
    const ArrayXd score = std::sqrt(at_expiry.ttm) / sigma * z - at_expiry.ttm;
    return (at_expiry.df * payoff * score).mean() * 0.01;
}

double McEngine::gamma_pathwise_fd(std::optional<double> epsilon) const {
    const Terminal at_expiry = terminal();
    const double s0 = state_.underlying.initial_value();
    const double strike = state_.strike().value_or(0.0);
    const bool is_call = state_.option_type == OptionType::Call;
    const double h = epsilon.value_or(s0 / 100.0);

    const auto bumped_delta = [&](double s0_bumped) {
        const ArrayXd spot_bumped = at_expiry.spot * (s0_bumped / s0);
        if (is_call) {
            return (at_expiry.df * (spot_bumped > strike).cast<double>() *
                    (spot_bumped / s0_bumped))
                .mean();
        }
        return (-at_expiry.df * (spot_bumped < strike).cast<double>() * (spot_bumped / s0_bumped))
            .mean();
    };
    return (bumped_delta(s0 + h) - bumped_delta(s0 - h)) / (2.0 * h);
}

}  // namespace dp::engines
