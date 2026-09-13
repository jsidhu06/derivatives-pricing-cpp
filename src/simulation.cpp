#include "dp/simulation.hpp"

#include "dp/diagnostics.hpp"
#include "dp/exceptions.hpp"
#include "dp/math/normal.hpp"

#include "detail/validate.hpp"

#include <pcg_random.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <random>
#include <set>
#include <utility>

namespace dp {

namespace {

using Eigen::Index;

/// Uniform in the open interval (0, 1): 53 random bits centred in the cell,
/// so the inverse normal CDF never sees 0 or 1.
double open_uniform(pcg64& rng) {
    constexpr double kScale = 0x1.0p-53;
    return (static_cast<double>(rng() >> 11) + 0.5) * kScale;
}

void validate_ascending_grid(const std::vector<DateTime>& grid, DateTime pricing_date) {
    if (grid.empty()) {
        throw ValidationError("SimulationConfig.time_grid must be non-empty when provided");
    }
    for (std::size_t i = 1; i < grid.size(); ++i) {
        if (grid[i] <= grid[i - 1]) {
            throw ValidationError("SimulationConfig.time_grid must be strictly ascending");
        }
    }
    if (grid.front() < pricing_date) {
        throw ValidationError(std::format(
            "Explicit time_grid starts at {}, which is before pricing_date ({}). All grid "
            "dates must be >= pricing_date.",
            to_string(grid.front()),
            to_string(pricing_date)));
    }
}

}  // namespace

void SimulationConfig::validate() const {
    if (paths <= 0) {
        throw ValidationError("SimulationConfig.paths must be a positive integer");
    }
    const bool has_grid = time_grid.has_value();
    if (has_grid) {
        if (end_date.has_value() || num_steps.has_value()) {
            throw ValidationError(
                "When time_grid is provided, end_date and num_steps must be omitted.");
        }
        if (time_grid->empty()) {
            throw ValidationError("SimulationConfig.time_grid must be non-empty when provided");
        }
        return;
    }
    if (!end_date.has_value()) {
        throw ValidationError(
            "SimulationConfig.end_date must be provided when time_grid is not set");
    }
    if (!num_steps.has_value()) {
        throw ValidationError("SimulationConfig.num_steps must be provided when end_date is set");
    }
    if (*num_steps <= 0) {
        throw ValidationError("SimulationConfig.num_steps must be a positive integer");
    }
}

GbmProcess::GbmProcess(MarketData market_data, GbmParams params, SimulationConfig config)
    : market_data_(std::move(market_data)), params_(std::move(params)), config_(std::move(config)) {
    detail::require_positive(params_.initial_value, "GbmParams.initial_value");
    detail::require_non_negative(params_.volatility, "GbmParams.volatility");
    params_.discrete_dividends = normalize_dividends(std::move(params_.discrete_dividends));
    if (params_.dividend_curve.has_value() && !params_.discrete_dividends.empty()) {
        warn(
            "GbmParams: both dividend_curve and discrete_dividends provided. The continuous "
            "yield will enter the drift and discrete dividends will be subtracted at each "
            "ex-date.");
    }
    config_.validate();
    if (config_.time_grid.has_value()) {
        validate_ascending_grid(*config_.time_grid, market_data_.pricing_date());
    }
}

DateTime GbmProcess::end_date() const {
    if (config_.time_grid.has_value()) {
        return config_.time_grid->back();
    }
    if (!config_.end_date.has_value()) {
        throw ConfigurationError("SimulationConfig has neither time_grid nor end_date");
    }
    return *config_.end_date;
}

std::vector<DateTime> GbmProcess::build_dates() const {
    if (config_.time_grid.has_value()) {
        return *config_.time_grid;
    }
    const DateTime pricing = market_data_.pricing_date();
    const DateTime dense_start = config_.grid_start.value_or(pricing);
    const DateTime end = end_date();

    if (!config_.num_steps.has_value()) {
        throw ConfigurationError("SimulationConfig.num_steps is required to build a dense grid");
    }
    std::set<DateTime> all;
    for (const DateTime d :
         date_range(dense_start, end, static_cast<std::size_t>(*config_.num_steps) + 1)) {
        all.insert(d);
    }
    all.insert(pricing);
    all.insert(dense_start);
    all.insert(end);
    // Observation dates before the pricing date are excluded: the diffusion
    // cannot start before the valuation date.
    for (const DateTime d : config_.observation_dates) {
        if (d >= pricing) {
            all.insert(d);
        }
    }
    for (const auto& dividend : params_.discrete_dividends) {
        if (dividend.ex_date >= pricing) {
            all.insert(dividend.ex_date);
        }
    }
    std::vector<DateTime> dates(all.begin(), all.end());
    if (dates.front() != pricing) {
        throw ValidationError(std::format(
            "Time grid must start at pricing_date ({}), but starts at {}. Check grid_start / "
            "observation_dates.",
            to_string(pricing),
            to_string(dates.front())));
    }
    return dates;
}

SimulationResult GbmProcess::simulate(std::optional<std::uint64_t> seed) const {
    std::vector<DateTime> dates = build_dates();
    const auto num_dates = static_cast<Index>(dates.size());
    const Index steps = num_dates - 1;
    const Index paths = config_.paths;
    const DateTime pricing = market_data_.pricing_date();
    const DayCountConvention dcc = market_data_.day_count_convention();

    Eigen::ArrayXd time_grid(num_dates);
    for (Index i = 0; i < num_dates; ++i) {
        time_grid[i] = year_fraction(pricing, dates[static_cast<std::size_t>(i)], dcc);
    }
    Eigen::ArrayXd time_deltas(std::max<Index>(steps, 0));
    for (Index t = 0; t < steps; ++t) {
        time_deltas[t] = year_fraction(
            dates[static_cast<std::size_t>(t)], dates[static_cast<std::size_t>(t) + 1], dcc);
    }

    // ── Standard-normal shocks ─────────────────────────────────────
    pcg64 rng =
        seed.has_value() ? pcg64(*seed) : pcg64(pcg_extras::seed_seq_from<std::random_device>{});

    bool antithetic = config_.antithetic;
    if (antithetic && paths % 2 != 0) {
        warn(std::format(
            "antithetic=true but paths={} is odd; antithetic variates require an even number "
            "of paths. Falling back to plain random sampling.",
            paths));
        antithetic = false;
    }

    Eigen::MatrixXd normals(paths, steps);
    if (antithetic) {
        const Index half = paths / 2;
        for (Index t = 0; t < steps; ++t) {
            for (Index p = 0; p < half; ++p) {
                const double z = math::normal_quantile(open_uniform(rng));
                normals(p, t) = z;
                normals(half + p, t) = -z;
            }
        }
    } else {
        for (Index t = 0; t < steps; ++t) {
            for (Index p = 0; p < paths; ++p) {
                normals(p, t) = math::normal_quantile(open_uniform(rng));
            }
        }
    }
    if (config_.moment_matching && paths > 1) {
        for (Index t = 0; t < steps; ++t) {
            auto column = normals.col(t).array();
            const double mean = column.mean();
            const double stddev = std::sqrt((column - mean).square().mean());
            column = (column - mean) / stddev;
        }
    }

    // ── Risk-neutral drift per step ────────────────────────────────
    const Eigen::ArrayXd r_steps = market_data_.discount_curve().step_forward_rates(time_grid);
    Eigen::ArrayXd q_steps = Eigen::ArrayXd::Zero(std::max<Index>(steps, 0));
    if (params_.dividend_curve.has_value()) {
        q_steps = params_.dividend_curve->step_forward_rates(time_grid);
    }
    const double sigma = params_.volatility;

    // ── Paths (exact log-space GBM) ────────────────────────────────
    Eigen::MatrixXd paths_matrix(paths, num_dates);
    paths_matrix.col(0).setConstant(params_.initial_value);

    // Discrete dividends: cash comes off the path at each ex-date on the grid.
    std::map<Index, double> dividend_at;
    for (const auto& dividend : params_.discrete_dividends) {
        if (dividend.ex_date < pricing || dividend.ex_date > dates.back()) {
            continue;
        }
        const auto it = std::lower_bound(dates.begin(), dates.end(), dividend.ex_date);
        if (it == dates.end() || *it != dividend.ex_date) {
            throw ConfigurationError(std::format(
                "Discrete dividend ex_date {} is within [pricing_date, maturity] but missing "
                "from the simulation time grid. Ensure the time grid includes all ex-dividend "
                "dates.",
                to_string(dividend.ex_date)));
        }
        dividend_at[std::distance(dates.begin(), it)] += dividend.amount;
    }
    if (const auto it = dividend_at.find(0); it != dividend_at.end()) {
        // The input spot is cum-dividend: it goes ex immediately at t=0.
        paths_matrix.col(0) = (paths_matrix.col(0).array() - it->second).cwiseMax(0.0).matrix();
    }

    for (Index t = 0; t < steps; ++t) {
        const double dt_step = time_deltas[t];
        const double drift = (r_steps[t] - q_steps[t] - 0.5 * sigma * sigma) * dt_step;
        const double vol_step = sigma * std::sqrt(dt_step);
        paths_matrix.col(t + 1) =
            paths_matrix.col(t).array() * (drift + vol_step * normals.col(t).array()).exp();
        if (const auto it = dividend_at.find(t + 1); it != dividend_at.end()) {
            paths_matrix.col(t + 1) =
                (paths_matrix.col(t + 1).array() - it->second).cwiseMax(0.0).matrix();
        }
    }

    return {.dates = std::move(dates),
            .time_grid = std::move(time_grid),
            .paths = std::move(paths_matrix),
            .normals = std::move(normals)};
}

GbmProcess GbmProcess::with_initial_value(double initial_value) const {
    GbmParams params = params_;
    params.initial_value = initial_value;
    return {market_data_, std::move(params), config_};
}

GbmProcess GbmProcess::with_volatility(double volatility) const {
    GbmParams params = params_;
    params.volatility = volatility;
    return {market_data_, std::move(params), config_};
}

GbmProcess GbmProcess::with_market_data(MarketData market_data) const {
    return {std::move(market_data), params_, config_};
}

GbmProcess GbmProcess::with_pricing_date(DateTime pricing_date) const {
    return with_market_data(market_data_.with_pricing_date(pricing_date));
}

GbmProcess GbmProcess::with_discount_curve(DiscountCurve discount_curve) const {
    return with_market_data(market_data_.with_discount_curve(std::move(discount_curve)));
}

}  // namespace dp
