#include "dp/simulation.hpp"

#include "dp/diagnostics.hpp"
#include "dp/exceptions.hpp"
#include "dp/math/normal.hpp"

#include "detail/validate.hpp"

#include <Eigen/Cholesky>
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

/// PCG64 stream selector for the jump components of `JdProcess`, so the jump
/// draws are independent of the diffusion draws made from the same seed
/// (mirrors numpy's `SeedSequence.spawn` in the Python reference).
constexpr std::uint64_t kJumpStream = 0x4a554d50;  // "JUMP"

/// Uniform in the open interval (0, 1): 53 random bits centred in the cell,
/// so the inverse normal CDF never sees 0 or 1.
double open_uniform(pcg64& rng) {
    constexpr double kScale = 0x1.0p-53;
    return (static_cast<double>(rng() >> 11) + 0.5) * kScale;
}

/// Poisson sample via Knuth's multiplication method — exact, O(1 + mean)
/// uniforms per draw. Used instead of `std::poisson_distribution`, whose
/// algorithm is implementation-defined and therefore not portable across
/// standard libraries (the per-step mean lambda*dt is small, so the loop is
/// short in practice).
int poisson_knuth(pcg64& rng, double mean) {
    if (mean <= 0.0) {
        return 0;
    }
    const double limit = std::exp(-mean);
    int count = 0;
    double product = open_uniform(rng);
    while (product > limit) {
        ++count;
        product *= open_uniform(rng);
    }
    return count;
}

pcg64 seeded_rng(std::optional<std::uint64_t> seed) {
    return seed.has_value() ? pcg64(*seed) : pcg64(pcg_extras::seed_seq_from<std::random_device>{});
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

/// Grid index -> total cash dividend at that date. Throws when an in-horizon
/// ex-date is missing from the grid.
std::map<Index, double> dividend_indices(const std::vector<DateTime>& dates,
                                         const DividendSchedule& dividends,
                                         DateTime pricing_date) {
    std::map<Index, double> dividend_at;
    for (const auto& dividend : dividends) {
        if (dividend.ex_date < pricing_date || dividend.ex_date > dates.back()) {
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
    return dividend_at;
}

/// Subtract the column's dividend (if any) and floor at zero.
void apply_dividend(Eigen::MatrixXd& paths,
                    const std::map<Index, double>& dividend_at,
                    Index column) {
    if (const auto it = dividend_at.find(column); it != dividend_at.end()) {
        paths.col(column) = (paths.col(column).array() - it->second).cwiseMax(0.0).matrix();
    }
}

/// Per-step risk-neutral forward rates minus the dividend yield.
Eigen::ArrayXd net_forward_rates(const DiscountCurve& discount_curve,
                                 const std::optional<DiscountCurve>& dividend_curve,
                                 const Eigen::ArrayXd& time_grid) {
    Eigen::ArrayXd rates = discount_curve.step_forward_rates(time_grid);
    if (dividend_curve.has_value()) {
        rates -= dividend_curve->step_forward_rates(time_grid);
    }
    return rates;
}

void warn_on_mixed_dividends(std::string_view params_name,
                             const std::optional<DiscountCurve>& dividend_curve,
                             const DividendSchedule& discrete_dividends) {
    if (dividend_curve.has_value() && !discrete_dividends.empty()) {
        warn(std::format(
            "{}: both dividend_curve and discrete_dividends provided. The continuous "
            "yield will enter the drift and discrete dividends will be subtracted at each "
            "ex-date.",
            params_name));
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

// ── CorrelationContext ─────────────────────────────────────────────

CorrelationContext::CorrelationContext(Eigen::MatrixXd correlation_matrix,
                                       std::vector<Eigen::MatrixXd> random_numbers,
                                       std::vector<std::string> asset_names)
    : correlation_matrix_(std::move(correlation_matrix)),
      random_numbers_(std::move(random_numbers)),
      asset_names_(std::move(asset_names)) {
    const Index n = correlation_matrix_.rows();
    if (n == 0 || correlation_matrix_.cols() != n) {
        throw ValidationError(std::format("correlation_matrix must be square, got shape ({}, {})",
                                          correlation_matrix_.rows(),
                                          correlation_matrix_.cols()));
    }
    if (std::cmp_not_equal(asset_names_.size(), n)) {
        throw ValidationError(
            std::format("asset_names length ({}) must match correlation_matrix dimension ({})",
                        asset_names_.size(),
                        n));
    }
    if (std::set<std::string>(asset_names_.begin(), asset_names_.end()).size() !=
        asset_names_.size()) {
        throw ValidationError("asset_names must contain unique entries");
    }
    constexpr double kTolerance = 1e-12;
    if (((correlation_matrix_ - correlation_matrix_.transpose()).array().abs() > kTolerance)
            .any()) {
        throw ValidationError("correlation_matrix must be symmetric");
    }
    if (((correlation_matrix_.diagonal().array() - 1.0).abs() > kTolerance).any()) {
        throw ValidationError(
            "correlation_matrix diagonal must be 1 (it is a correlation matrix, not covariance)");
    }
    const Eigen::LLT<Eigen::MatrixXd> llt(correlation_matrix_);
    if (llt.info() != Eigen::Success) {
        throw ValidationError("correlation_matrix is not positive-definite");
    }
    cholesky_ = llt.matrixL();

    if (std::cmp_not_equal(random_numbers_.size(), n)) {
        throw ValidationError(
            std::format("random_numbers count ({}) must match number of assets ({})",
                        random_numbers_.size(),
                        n));
    }
    for (const auto& matrix : random_numbers_) {
        if (matrix.rows() != random_numbers_.front().rows() ||
            matrix.cols() != random_numbers_.front().cols() || matrix.size() == 0) {
            throw ValidationError(
                "random_numbers matrices must be non-empty and identically shaped "
                "(paths x steps)");
        }
    }
}

std::size_t CorrelationContext::asset_index(const std::string& name) const {
    const auto it = std::find(asset_names_.begin(), asset_names_.end(), name);
    if (it == asset_names_.end()) {
        std::string names;
        for (const auto& asset : asset_names_) {
            names += (names.empty() ? "" : ", ") + asset;
        }
        throw ValidationError(std::format("'{}' not found in asset_names: [{}]", name, names));
    }
    return static_cast<std::size_t>(std::distance(asset_names_.begin(), it));
}

std::vector<Eigen::MatrixXd> CorrelationContext::standard_normals(
    std::size_t n_assets,
    Eigen::Index paths,
    Eigen::Index steps,
    std::optional<std::uint64_t> seed) {
    pcg64 rng = seeded_rng(seed);
    std::vector<Eigen::MatrixXd> result;
    result.reserve(n_assets);
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        Eigen::MatrixXd draws(paths, steps);
        for (Index t = 0; t < steps; ++t) {
            for (Index p = 0; p < paths; ++p) {
                draws(p, t) = math::normal_quantile_fast(open_uniform(rng));
            }
        }
        result.push_back(std::move(draws));
    }
    return result;
}

// ── PathProcess ────────────────────────────────────────────────────

PathProcess::PathProcess(MarketData market_data,
                         SimulationConfig config,
                         std::shared_ptr<const CorrelationContext> corr,
                         std::optional<std::string> name)
    : market_data_(std::move(market_data)),
      config_(std::move(config)),
      corr_(std::move(corr)),
      name_(std::move(name)) {
    config_.validate();
    if (config_.time_grid.has_value()) {
        validate_ascending_grid(*config_.time_grid, market_data_.pricing_date());
    }
    if (corr_ != nullptr) {
        if (!name_.has_value()) {
            throw ValidationError(
                "name is required when using CorrelationContext (used to look up the asset "
                "index)");
        }
        // Validate early: fail now rather than at simulation time.
        rn_index_ = corr_->asset_index(*name_);
    }
}

void PathProcess::register_grid_dates(const DividendSchedule& dividends) {
    for (const auto& dividend : dividends) {
        dividend_grid_dates_.push_back(dividend.ex_date);
    }
}

DateTime PathProcess::end_date() const {
    if (config_.time_grid.has_value()) {
        return config_.time_grid->back();
    }
    if (!config_.end_date.has_value()) {
        throw ConfigurationError("SimulationConfig has neither time_grid nor end_date");
    }
    return *config_.end_date;
}

std::vector<DateTime> PathProcess::build_dates() const {
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
    for (const DateTime d : dividend_grid_dates_) {
        if (d >= pricing) {
            all.insert(d);
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

PathProcess::GridBundle PathProcess::prepare_grid() const {
    GridBundle grid;
    grid.dates = build_dates();
    const auto num_dates = static_cast<Index>(grid.dates.size());
    const Index steps = num_dates - 1;
    const DateTime pricing = market_data_.pricing_date();
    const DayCountConvention dcc = market_data_.day_count_convention();

    grid.time_grid.resize(num_dates);
    for (Index i = 0; i < num_dates; ++i) {
        grid.time_grid[i] = year_fraction(pricing, grid.dates[static_cast<std::size_t>(i)], dcc);
    }
    grid.time_deltas.resize(std::max<Index>(steps, 0));
    for (Index t = 0; t < steps; ++t) {
        grid.time_deltas[t] = year_fraction(grid.dates[static_cast<std::size_t>(t)],
                                            grid.dates[static_cast<std::size_t>(t) + 1],
                                            dcc);
    }
    return grid;
}

Eigen::MatrixXd PathProcess::draw_normals(std::optional<std::uint64_t> seed, Index steps) const {
    const Index paths = config_.paths;

    if (corr_ != nullptr) {
        // Correlated multi-asset draws: this asset's linear combination of the
        // shared pre-generated numbers. antithetic / moment_matching do not
        // apply — the context owns the raw draws.
        const std::vector<Eigen::MatrixXd>& raw = corr_->random_numbers();
        if (raw.front().rows() != paths) {
            throw ConfigurationError(
                std::format("CorrelationContext random_numbers have {} paths but the "
                            "simulation is configured for {} paths",
                            raw.front().rows(),
                            paths));
        }
        if (raw.front().cols() < steps) {
            throw ConfigurationError(
                std::format("CorrelationContext random_numbers cover {} steps but the "
                            "simulation grid requires {} steps",
                            raw.front().cols(),
                            steps));
        }
        const Eigen::MatrixXd& chol = corr_->cholesky_matrix();
        Eigen::MatrixXd correlated = Eigen::MatrixXd::Zero(paths, steps);
        const auto row = static_cast<Index>(rn_index_);
        // The factor is lower-triangular: only assets up to rn_index_ load.
        for (Index j = 0; j <= row; ++j) {
            if (chol(row, j) != 0.0) {
                correlated += chol(row, j) * raw[static_cast<std::size_t>(j)].leftCols(steps);
            }
        }
        return correlated;
    }

    pcg64 rng = seeded_rng(seed);

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
                const double z = math::normal_quantile_fast(open_uniform(rng));
                normals(p, t) = z;
                normals(half + p, t) = -z;
            }
        }
    } else {
        for (Index t = 0; t < steps; ++t) {
            for (Index p = 0; p < paths; ++p) {
                normals(p, t) = math::normal_quantile_fast(open_uniform(rng));
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
    return normals;
}

// ── GbmProcess ─────────────────────────────────────────────────────

GbmProcess::GbmProcess(MarketData market_data,
                       GbmParams params,
                       SimulationConfig config,
                       std::shared_ptr<const CorrelationContext> corr,
                       std::optional<std::string> name)
    : PathProcess(std::move(market_data), std::move(config), std::move(corr), std::move(name)),
      params_(std::move(params)) {
    detail::require_positive(params_.initial_value, "GbmParams.initial_value");
    detail::require_non_negative(params_.volatility, "GbmParams.volatility");
    params_.discrete_dividends = normalize_dividends(std::move(params_.discrete_dividends));
    warn_on_mixed_dividends("GbmParams", params_.dividend_curve, params_.discrete_dividends);
    register_grid_dates(params_.discrete_dividends);
}

SimulationResult GbmProcess::simulate(std::optional<std::uint64_t> seed) const {
    GridBundle grid = prepare_grid();
    const auto num_dates = static_cast<Index>(grid.dates.size());
    const Index steps = num_dates - 1;
    const Index paths = config().paths;

    Eigen::MatrixXd normals = draw_normals(seed, steps);

    // ── Risk-neutral drift per step ────────────────────────────────
    const Eigen::ArrayXd net_rates =
        net_forward_rates(discount_curve(), params_.dividend_curve, grid.time_grid);
    const double sigma = params_.volatility;

    // ── Paths (exact log-space GBM) ────────────────────────────────
    Eigen::MatrixXd paths_matrix(paths, num_dates);
    paths_matrix.col(0).setConstant(params_.initial_value);

    // Discrete dividends: cash comes off the path at each ex-date on the grid.
    const std::map<Index, double> dividend_at =
        dividend_indices(grid.dates, params_.discrete_dividends, pricing_date());
    // The input spot is cum-dividend: it goes ex immediately at t=0.
    apply_dividend(paths_matrix, dividend_at, 0);

    for (Index t = 0; t < steps; ++t) {
        const double dt_step = grid.time_deltas[t];
        const double drift = (net_rates[t] - 0.5 * sigma * sigma) * dt_step;
        const double vol_step = sigma * std::sqrt(dt_step);
        paths_matrix.col(t + 1) =
            paths_matrix.col(t).array() * (drift + vol_step * normals.col(t).array()).exp();
        apply_dividend(paths_matrix, dividend_at, t + 1);
    }

    return {.dates = std::move(grid.dates),
            .time_grid = std::move(grid.time_grid),
            .paths = std::move(paths_matrix),
            .normals = std::move(normals)};
}

GbmProcess GbmProcess::with_initial_value(double initial_value) const {
    GbmParams params = params_;
    params.initial_value = initial_value;
    return {market_data(), std::move(params), config(), correlation_context(), name()};
}

GbmProcess GbmProcess::with_volatility(double volatility) const {
    GbmParams params = params_;
    params.volatility = volatility;
    return {market_data(), std::move(params), config(), correlation_context(), name()};
}

GbmProcess GbmProcess::with_market_data(MarketData market_data) const {
    return {std::move(market_data), params_, config(), correlation_context(), name()};
}

GbmProcess GbmProcess::with_observation_dates(std::vector<DateTime> dates) const {
    SimulationConfig new_config = config();
    new_config.observation_dates.insert(
        new_config.observation_dates.end(), dates.begin(), dates.end());
    return {market_data(), params_, std::move(new_config), correlation_context(), name()};
}

GbmProcess GbmProcess::with_pricing_date(DateTime pricing_date) const {
    return with_market_data(market_data().with_pricing_date(pricing_date));
}

GbmProcess GbmProcess::with_discount_curve(DiscountCurve discount_curve) const {
    return with_market_data(market_data().with_discount_curve(std::move(discount_curve)));
}

// ── JdProcess ──────────────────────────────────────────────────────

JdProcess::JdProcess(MarketData market_data,
                     JdParams params,
                     SimulationConfig config,
                     std::shared_ptr<const CorrelationContext> corr,
                     std::optional<std::string> name)
    : PathProcess(std::move(market_data), std::move(config), std::move(corr), std::move(name)),
      params_(std::move(params)) {
    detail::require_positive(params_.initial_value, "JdParams.initial_value");
    detail::require_non_negative(params_.volatility, "JdParams.volatility");
    detail::require_non_negative(params_.lambd, "JdParams.lambd");
    detail::require_finite(params_.mu, "JdParams.mu");
    detail::require_non_negative(params_.delta, "JdParams.delta");
    params_.discrete_dividends = normalize_dividends(std::move(params_.discrete_dividends));
    warn_on_mixed_dividends("JdParams", params_.dividend_curve, params_.discrete_dividends);
    register_grid_dates(params_.discrete_dividends);
}

SimulationResult JdProcess::simulate(std::optional<std::uint64_t> seed) const {
    GridBundle grid = prepare_grid();
    const auto num_dates = static_cast<Index>(grid.dates.size());
    const Index steps = num_dates - 1;
    const Index paths = config().paths;

    // Diffusion shocks share the GBM stream layout, so lambda == 0 reproduces
    // the GBM paths bit-for-bit; jumps draw from an independent PCG64 stream.
    Eigen::MatrixXd normals = draw_normals(seed, steps);
    pcg64 jump_rng = seed.has_value() ? pcg64(*seed, kJumpStream)
                                      : pcg64(pcg_extras::seed_seq_from<std::random_device>{});

    const Eigen::ArrayXd net_rates =
        net_forward_rates(discount_curve(), params_.dividend_curve, grid.time_grid);
    const double sigma = params_.volatility;
    const double lambda = params_.lambd;
    const double mu_jump = params_.mu;
    const double delta_jump = params_.delta;
    // Drift compensator kbar = E[e^Y - 1] keeps the process a martingale.
    const double kbar = std::exp(mu_jump + 0.5 * delta_jump * delta_jump) - 1.0;

    Eigen::MatrixXd paths_matrix(paths, num_dates);
    paths_matrix.col(0).setConstant(params_.initial_value);

    const std::map<Index, double> dividend_at =
        dividend_indices(grid.dates, params_.discrete_dividends, pricing_date());
    // The input spot is cum-dividend: it goes ex immediately at t=0. (The
    // Python reference skips this adjustment for JDProcess — an inconsistency
    // with its GBMProcess; we apply it for both.)
    apply_dividend(paths_matrix, dividend_at, 0);

    for (Index t = 0; t < steps; ++t) {
        const double dt_step = grid.time_deltas[t];
        const double drift = (net_rates[t] - lambda * kbar - 0.5 * sigma * sigma) * dt_step;
        const double vol_step = sigma * std::sqrt(dt_step);
        paths_matrix.col(t + 1) =
            paths_matrix.col(t).array() * (drift + vol_step * normals.col(t).array()).exp();

        if (lambda > 0.0) {
            // Sum of N iid N(mu, delta^2) jumps is N(N mu, N delta^2):
            // sample the sum directly as N mu + sqrt(N) delta Z_J.
            const double jump_mean = lambda * dt_step;
            for (Index p = 0; p < paths; ++p) {
                const int jumps = poisson_knuth(jump_rng, jump_mean);
                if (jumps > 0) {
                    const double z_jump = math::normal_quantile_fast(open_uniform(jump_rng));
                    paths_matrix(p, t + 1) *=
                        std::exp(jumps * mu_jump + std::sqrt(jumps) * delta_jump * z_jump);
                }
            }
        }
        apply_dividend(paths_matrix, dividend_at, t + 1);
    }

    return {.dates = std::move(grid.dates),
            .time_grid = std::move(grid.time_grid),
            .paths = std::move(paths_matrix),
            .normals = std::move(normals)};
}

JdProcess JdProcess::with_initial_value(double initial_value) const {
    JdParams params = params_;
    params.initial_value = initial_value;
    return {market_data(), std::move(params), config(), correlation_context(), name()};
}

JdProcess JdProcess::with_volatility(double volatility) const {
    JdParams params = params_;
    params.volatility = volatility;
    return {market_data(), std::move(params), config(), correlation_context(), name()};
}

JdProcess JdProcess::with_market_data(MarketData market_data) const {
    return {std::move(market_data), params_, config(), correlation_context(), name()};
}

JdProcess JdProcess::with_observation_dates(std::vector<DateTime> dates) const {
    SimulationConfig new_config = config();
    new_config.observation_dates.insert(
        new_config.observation_dates.end(), dates.begin(), dates.end());
    return {market_data(), params_, std::move(new_config), correlation_context(), name()};
}

JdProcess JdProcess::with_pricing_date(DateTime pricing_date) const {
    return with_market_data(market_data().with_pricing_date(pricing_date));
}

JdProcess JdProcess::with_discount_curve(DiscountCurve discount_curve) const {
    return with_market_data(market_data().with_discount_curve(std::move(discount_curve)));
}

// ── SrdProcess ─────────────────────────────────────────────────────

SrdProcess::SrdProcess(MarketData market_data,
                       SrdParams params,
                       SimulationConfig config,
                       std::shared_ptr<const CorrelationContext> corr,
                       std::optional<std::string> name)
    : PathProcess(std::move(market_data), std::move(config), std::move(corr), std::move(name)),
      params_(params) {
    detail::require_finite(params_.initial_value, "SrdParams.initial_value");
    detail::require_non_negative(params_.volatility, "SrdParams.volatility");
    detail::require_non_negative(params_.kappa, "SrdParams.kappa");
    detail::require_non_negative(params_.theta, "SrdParams.theta");
    const double feller = 2.0 * params_.kappa * params_.theta;
    const double sigma_sq = params_.volatility * params_.volatility;
    if (feller < sigma_sq) {
        warn(
            std::format("Feller condition violated: 2·κ·θ = {:.6g} < σ² = {:.6g}. The CIR "
                        "process may frequently touch zero.",
                        feller,
                        sigma_sq));
    }
}

namespace {
const DividendSchedule kEmptyDividends;
const std::optional<DiscountCurve> kNoDividendCurve;
}  // namespace

// Instance methods (not static) for interface parity: they are called through
// the variant underlying's generic accessors alongside GBM/JD.
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
const DividendSchedule& SrdProcess::discrete_dividends() const noexcept {
    return kEmptyDividends;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
const std::optional<DiscountCurve>& SrdProcess::dividend_curve() const noexcept {
    return kNoDividendCurve;
}

SimulationResult SrdProcess::simulate(std::optional<std::uint64_t> seed) const {
    GridBundle grid = prepare_grid();
    const auto num_dates = static_cast<Index>(grid.dates.size());
    const Index steps = num_dates - 1;
    const Index paths = config().paths;

    Eigen::MatrixXd normals = draw_normals(seed, steps);

    const double kappa = params_.kappa;
    const double theta = params_.theta;
    const double sigma = params_.volatility;

    // Full truncation: the raw Euler state accumulates unfloored, while the
    // returned paths (and the coefficients) use the value floored at zero.
    Eigen::MatrixXd paths_matrix(paths, num_dates);
    paths_matrix.col(0).setConstant(params_.initial_value);
    Eigen::ArrayXd euler_raw = Eigen::ArrayXd::Constant(paths, params_.initial_value);

    for (Index t = 0; t < steps; ++t) {
        const double dt_step = grid.time_deltas[t];
        const auto x_prev = paths_matrix.col(t).array();
        euler_raw += kappa * (theta - x_prev) * dt_step +
                     sigma * (x_prev * dt_step).sqrt() * normals.col(t).array();
        paths_matrix.col(t + 1) = euler_raw.max(0.0).matrix();
    }

    return {.dates = std::move(grid.dates),
            .time_grid = std::move(grid.time_grid),
            .paths = std::move(paths_matrix),
            .normals = std::move(normals)};
}

SrdProcess SrdProcess::with_initial_value(double initial_value) const {
    SrdParams params = params_;
    params.initial_value = initial_value;
    return {market_data(), params, config(), correlation_context(), name()};
}

SrdProcess SrdProcess::with_volatility(double volatility) const {
    SrdParams params = params_;
    params.volatility = volatility;
    return {market_data(), params, config(), correlation_context(), name()};
}

SrdProcess SrdProcess::with_market_data(MarketData market_data) const {
    return {std::move(market_data), params_, config(), correlation_context(), name()};
}

SrdProcess SrdProcess::with_observation_dates(std::vector<DateTime> dates) const {
    SimulationConfig new_config = config();
    new_config.observation_dates.insert(
        new_config.observation_dates.end(), dates.begin(), dates.end());
    return {market_data(), params_, std::move(new_config), correlation_context(), name()};
}

SrdProcess SrdProcess::with_pricing_date(DateTime pricing_date) const {
    return with_market_data(market_data().with_pricing_date(pricing_date));
}

SrdProcess SrdProcess::with_discount_curve(DiscountCurve discount_curve) const {
    return with_market_data(market_data().with_discount_curve(std::move(discount_curve)));
}

}  // namespace dp
