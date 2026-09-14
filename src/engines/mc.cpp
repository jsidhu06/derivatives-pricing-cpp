#include "engines/mc.hpp"

#include "dp/dates.hpp"
#include "dp/exceptions.hpp"
#include "dp/params.hpp"

#include "detail/log.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <variant>

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

}  // namespace

McEngine::McEngine(const detail::ValuationState& state) : state_(state) {}

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

ArrayXd McEngine::present_value_pathwise() const {
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

    // t = 0: spot is constant across paths, so the exercise decision compares
    // two scalars.
    const double df0 = dfs[1] / dfs[0];
    ArrayXd discounted = df0 * values;
    const double immediate = payoff_at(sim.paths.col(start).array()).mean();
    if (immediate > discounted.mean()) {
        return ArrayXd::Constant(discounted.size(), immediate);
    }
    return discounted;
}

double McEngine::present_value() const {
    const ArrayXd pv_pathwise = present_value_pathwise();
    const double pv = pv_pathwise.mean();
    const bool european = state_.exercise_type() == ExerciseType::European;
    warn_if_high_std_error(pv_pathwise, pv, mc_params(state_), european ? "European" : "American");
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
