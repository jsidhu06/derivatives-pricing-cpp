#pragma once

#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/dividends.hpp"
#include "dp/enums.hpp"
#include "dp/market_data.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

/// Path simulation for Monte Carlo pricing.
namespace dp {

/// Configuration of a simulation run. The start date is the market's pricing
/// date; exactly one of {`end_date` + `num_steps`} or an explicit `time_grid`
/// must be supplied.
struct SimulationConfig {
    int paths;
    std::optional<DateTime> end_date = std::nullopt;
    std::optional<int> num_steps = std::nullopt;
    /// Explicit, strictly ascending grid (overrides end_date/num_steps).
    std::optional<std::vector<DateTime>> time_grid = std::nullopt;
    /// Extra dates merged into the generated grid (fixings, ex-div dates...).
    // NOLINTNEXTLINE(readability-redundant-member-init): see ForwardInputs.
    std::vector<DateTime> observation_dates = {};
    /// Start of the dense uniform region (defaults to the pricing date).
    std::optional<DateTime> grid_start = std::nullopt;
    /// Mirror half the normal draws (requires an even path count).
    bool antithetic = true;
    /// Standardise the draws to exact N(0,1) moments per time step.
    bool moment_matching = true;

    void validate() const;
};

/// Parameters of a geometric Brownian motion underlying.
struct GbmParams {
    double initial_value;
    double volatility;
    // NOLINTNEXTLINE(readability-redundant-member-init): see ForwardInputs.
    DividendSchedule discrete_dividends = {};
    std::optional<DiscountCurve> dividend_curve = std::nullopt;
};

/// Parameters of the Merton (1976) jump-diffusion process: GBM diffusion plus
/// compound-Poisson jumps with log-normal sizes Y ~ N(mu, delta^2) arriving at
/// intensity `lambd` per year.
struct JdParams {
    double initial_value;
    double volatility;
    /// Jump intensity — expected number of jumps per year (>= 0).
    double lambd;
    /// Mean of the log-jump size distribution.
    double mu;
    /// Standard deviation of the log-jump size distribution (>= 0).
    double delta;
    // NOLINTNEXTLINE(readability-redundant-member-init): see ForwardInputs.
    DividendSchedule discrete_dividends = {};
    std::optional<DiscountCurve> dividend_curve = std::nullopt;
};

/// Parameters of the Cox-Ingersoll-Ross square-root diffusion
/// dX = kappa (theta - X) dt + sigma sqrt(X) dW. The Feller condition
/// 2*kappa*theta >= sigma^2 keeps the process strictly positive; a violation
/// is allowed but logged (paths may frequently touch zero).
struct SrdParams {
    double initial_value;
    double volatility;
    /// Mean-reversion speed (>= 0).
    double kappa;
    /// Long-run mean level (>= 0).
    double theta;
};

/// Everything one simulation run produces. Returned by value: the process is
/// immutable and holds no run state (a deliberate departure from the Python
/// reference, which cached `time_grid` / `last_normals` on the instance and
/// therefore needed defensive copies).
struct SimulationResult {
    std::vector<DateTime> dates;  ///< Simulation grid (starts at the pricing date).
    Eigen::ArrayXd time_grid;     ///< Year fractions of `dates` from the pricing date.
    /// Simulated spots, one row per path; column t is the whole cross-section
    /// at `dates[t]` (contiguous, which is what the engines iterate over).
    Eigen::MatrixXd paths;
    /// Standard-normal shocks used, (paths x steps); kept for pathwise/LR Greeks.
    Eigen::MatrixXd normals;
};

/// Shared correlation context for multi-asset simulation. Holds a validated
/// correlation matrix (with its Cholesky factor) and pre-generated independent
/// standard-normal draws, one `(paths x steps)` matrix per asset. Processes
/// built with the same context draw correlated shocks from these numbers, so
/// `antithetic` / `moment_matching` do not apply and `simulate` consumes no
/// randomness of its own.
///
/// Share one instance between processes via `std::shared_ptr`; the context is
/// immutable after construction.
class CorrelationContext {
public:
    /// Throws `ValidationError` when the matrix is not square/symmetric/
    /// unit-diagonal/positive-definite, when names are not unique or their
    /// count differs from the matrix dimension, or when the random-number
    /// matrices differ in count or shape.
    CorrelationContext(Eigen::MatrixXd correlation_matrix,
                       std::vector<Eigen::MatrixXd> random_numbers,
                       std::vector<std::string> asset_names);

    [[nodiscard]] const Eigen::MatrixXd& correlation_matrix() const noexcept {
        return correlation_matrix_;
    }
    /// Lower-triangular Cholesky factor L with L * L^T = correlation matrix.
    [[nodiscard]] const Eigen::MatrixXd& cholesky_matrix() const noexcept { return cholesky_; }
    /// Independent standard-normal draws, `asset_names()[i]` -> matrix i.
    [[nodiscard]] const std::vector<Eigen::MatrixXd>& random_numbers() const noexcept {
        return random_numbers_;
    }
    [[nodiscard]] const std::vector<std::string>& asset_names() const noexcept {
        return asset_names_;
    }
    /// Index of `name` in `asset_names()`; throws `ValidationError` if absent.
    [[nodiscard]] std::size_t asset_index(const std::string& name) const;

    /// Convenience generator for `random_numbers`: `n_assets` matrices of
    /// independent N(0,1) draws, each `(paths x steps)`. Same RNG scheme as
    /// the single-asset processes (PCG64 + inverse CDF).
    [[nodiscard]] static std::vector<Eigen::MatrixXd> standard_normals(
        std::size_t n_assets,
        Eigen::Index paths,
        Eigen::Index steps,
        std::optional<std::uint64_t> seed);

private:
    Eigen::MatrixXd correlation_matrix_;
    Eigen::MatrixXd cholesky_;
    std::vector<Eigen::MatrixXd> random_numbers_;
    std::vector<std::string> asset_names_;
};

/// Common state and machinery shared by the path processes: market data, the
/// simulation config, grid construction, and standard-normal shock generation
/// (own PCG64 stream, or correlated draws from a `CorrelationContext`).
///
/// Not polymorphic — concrete processes are dispatched via `std::variant`
/// (see `detail::Underlying`), so there are no virtuals and the destructor is
/// protected and non-virtual.
///
/// RNG: PCG64 seeded from `MonteCarloParams.random_seed`, normals via the
/// inverse CDF (one uniform per normal, so stream positions are predictable
/// and the layout is QMC-ready). Streams are portable across platforms but
/// intentionally not numpy-compatible.
class PathProcess {
public:
    [[nodiscard]] const MarketData& market_data() const noexcept { return market_data_; }
    [[nodiscard]] const SimulationConfig& config() const noexcept { return config_; }
    /// Correlation context for multi-asset simulation (null when unused).
    [[nodiscard]] const std::shared_ptr<const CorrelationContext>& correlation_context()
        const noexcept {
        return corr_;
    }
    /// Process identifier; required (and used) only with a correlation context.
    [[nodiscard]] const std::optional<std::string>& name() const noexcept { return name_; }

    [[nodiscard]] DateTime pricing_date() const noexcept { return market_data_.pricing_date(); }
    [[nodiscard]] const DiscountCurve& discount_curve() const noexcept {
        return market_data_.discount_curve();
    }
    [[nodiscard]] const std::string& currency() const noexcept { return market_data_.currency(); }
    [[nodiscard]] DayCountConvention day_count_convention() const noexcept {
        return market_data_.day_count_convention();
    }
    /// Simulation horizon end (max of the explicit grid, or `config.end_date`).
    [[nodiscard]] DateTime end_date() const;

    /// The full simulation grid: uniform dense region merged with the pricing
    /// date, observation dates and dividend ex-dates.
    [[nodiscard]] std::vector<DateTime> build_dates() const;

protected:
    PathProcess(MarketData market_data,
                SimulationConfig config,
                std::shared_ptr<const CorrelationContext> corr,
                std::optional<std::string> name);
    ~PathProcess() = default;
    PathProcess(const PathProcess&) = default;
    PathProcess(PathProcess&&) = default;
    PathProcess& operator=(const PathProcess&) = default;
    PathProcess& operator=(PathProcess&&) = default;

    /// Merge dividend ex-dates into the generated grid (call from the derived
    /// constructor after normalising the schedule).
    void register_grid_dates(const DividendSchedule& dividends);

    /// Grid pieces shared by every `simulate` implementation.
    struct GridBundle {
        std::vector<DateTime> dates;
        Eigen::ArrayXd time_grid;    ///< Year fractions from the pricing date.
        Eigen::ArrayXd time_deltas;  ///< Year fractions between grid neighbours.
    };
    [[nodiscard]] GridBundle prepare_grid() const;

    /// Standard-normal shocks, `(paths x steps)`. Without a correlation
    /// context: fresh PCG64 draws with antithetic mirroring and per-step
    /// moment matching per the config. With one: this asset's correlated
    /// slice of the shared pre-generated numbers (config flags ignored).
    [[nodiscard]] Eigen::MatrixXd draw_normals(std::optional<std::uint64_t> seed,
                                               Eigen::Index steps) const;

private:
    MarketData market_data_;
    SimulationConfig config_;
    std::shared_ptr<const CorrelationContext> corr_;
    std::optional<std::string> name_;
    std::size_t rn_index_ = 0;
    std::vector<DateTime> dividend_grid_dates_;
};

/// Geometric Brownian motion path generator (risk-neutral, log-exact).
class GbmProcess : public PathProcess {
public:
    GbmProcess(MarketData market_data,
               GbmParams params,
               SimulationConfig config,
               std::shared_ptr<const CorrelationContext> corr = nullptr,
               std::optional<std::string> name = std::nullopt);

    [[nodiscard]] const GbmParams& params() const noexcept { return params_; }
    [[nodiscard]] double initial_value() const noexcept { return params_.initial_value; }
    [[nodiscard]] double volatility() const noexcept { return params_.volatility; }
    [[nodiscard]] const DividendSchedule& discrete_dividends() const noexcept {
        return params_.discrete_dividends;
    }
    [[nodiscard]] const std::optional<DiscountCurve>& dividend_curve() const noexcept {
        return params_.dividend_curve;
    }

    /// Generate paths. `seed == nullopt` draws fresh entropy; a fixed seed
    /// reproduces the run bit-for-bit on any platform.
    [[nodiscard]] SimulationResult simulate(std::optional<std::uint64_t> seed) const;

    [[nodiscard]] GbmProcess with_initial_value(double initial_value) const;
    [[nodiscard]] GbmProcess with_volatility(double volatility) const;
    [[nodiscard]] GbmProcess with_market_data(MarketData market_data) const;
    [[nodiscard]] GbmProcess with_pricing_date(DateTime pricing_date) const;
    [[nodiscard]] GbmProcess with_discount_curve(DiscountCurve discount_curve) const;

private:
    GbmParams params_;
};

/// Merton (1976) jump-diffusion path generator. Per step:
///
///     S_{t+dt} = S_t * exp((r - q - lambda*kbar - sigma^2/2) dt + sigma sqrt(dt) Z)
///                    * exp(N mu + sqrt(N) delta Z_J),  N ~ Poisson(lambda dt)
///
/// with kbar = exp(mu + delta^2/2) - 1 compensating the drift. The jump
/// components use a dedicated PCG64 stream derived from the same seed, so a
/// zero-intensity process reproduces the GBM paths bit-for-bit.
class JdProcess : public PathProcess {
public:
    JdProcess(MarketData market_data,
              JdParams params,
              SimulationConfig config,
              std::shared_ptr<const CorrelationContext> corr = nullptr,
              std::optional<std::string> name = std::nullopt);

    [[nodiscard]] const JdParams& params() const noexcept { return params_; }
    [[nodiscard]] double initial_value() const noexcept { return params_.initial_value; }
    [[nodiscard]] double volatility() const noexcept { return params_.volatility; }
    [[nodiscard]] double lambd() const noexcept { return params_.lambd; }
    [[nodiscard]] double mu() const noexcept { return params_.mu; }
    [[nodiscard]] double delta() const noexcept { return params_.delta; }
    [[nodiscard]] const DividendSchedule& discrete_dividends() const noexcept {
        return params_.discrete_dividends;
    }
    [[nodiscard]] const std::optional<DiscountCurve>& dividend_curve() const noexcept {
        return params_.dividend_curve;
    }

    [[nodiscard]] SimulationResult simulate(std::optional<std::uint64_t> seed) const;

    [[nodiscard]] JdProcess with_initial_value(double initial_value) const;
    [[nodiscard]] JdProcess with_volatility(double volatility) const;
    [[nodiscard]] JdProcess with_market_data(MarketData market_data) const;
    [[nodiscard]] JdProcess with_pricing_date(DateTime pricing_date) const;
    [[nodiscard]] JdProcess with_discount_curve(DiscountCurve discount_curve) const;

private:
    JdParams params_;
};

/// Cox-Ingersoll-Ross square-root diffusion path generator. Uses the
/// full-truncation Euler scheme (Lord, Koekkoek & van Dijk 2010): drift and
/// diffusion are evaluated at max(X, 0) and the returned paths are floored at
/// zero, while the raw Euler state accumulates untruncated.
class SrdProcess : public PathProcess {
public:
    SrdProcess(MarketData market_data,
               SrdParams params,
               SimulationConfig config,
               std::shared_ptr<const CorrelationContext> corr = nullptr,
               std::optional<std::string> name = std::nullopt);

    [[nodiscard]] const SrdParams& params() const noexcept { return params_; }
    [[nodiscard]] double initial_value() const noexcept { return params_.initial_value; }
    [[nodiscard]] double volatility() const noexcept { return params_.volatility; }
    [[nodiscard]] double kappa() const noexcept { return params_.kappa; }
    [[nodiscard]] double theta() const noexcept { return params_.theta; }
    /// Mean-reverting processes carry no dividends; empty for interface parity.
    [[nodiscard]] const DividendSchedule& discrete_dividends() const noexcept;
    [[nodiscard]] const std::optional<DiscountCurve>& dividend_curve() const noexcept;

    [[nodiscard]] SimulationResult simulate(std::optional<std::uint64_t> seed) const;

    [[nodiscard]] SrdProcess with_initial_value(double initial_value) const;
    [[nodiscard]] SrdProcess with_volatility(double volatility) const;
    [[nodiscard]] SrdProcess with_market_data(MarketData market_data) const;
    [[nodiscard]] SrdProcess with_pricing_date(DateTime pricing_date) const;
    [[nodiscard]] SrdProcess with_discount_curve(DiscountCurve discount_curve) const;

private:
    SrdParams params_;
};

}  // namespace dp
