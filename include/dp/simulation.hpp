#pragma once

#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/dividends.hpp"
#include "dp/enums.hpp"
#include "dp/market_data.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <optional>
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

/// Geometric Brownian motion path generator (risk-neutral, log-exact).
///
/// RNG: PCG64 seeded from `MonteCarloParams.random_seed`, normals via the
/// inverse CDF (one uniform per normal, so stream positions are predictable
/// and the layout is QMC-ready). Streams are portable across platforms but
/// intentionally not numpy-compatible.
class GbmProcess {
public:
    GbmProcess(MarketData market_data, GbmParams params, SimulationConfig config);

    [[nodiscard]] const MarketData& market_data() const noexcept { return market_data_; }
    [[nodiscard]] const GbmParams& params() const noexcept { return params_; }
    [[nodiscard]] const SimulationConfig& config() const noexcept { return config_; }

    [[nodiscard]] double initial_value() const noexcept { return params_.initial_value; }
    [[nodiscard]] double volatility() const noexcept { return params_.volatility; }
    [[nodiscard]] const DividendSchedule& discrete_dividends() const noexcept {
        return params_.discrete_dividends;
    }
    [[nodiscard]] const std::optional<DiscountCurve>& dividend_curve() const noexcept {
        return params_.dividend_curve;
    }
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

    /// Generate paths. `seed == nullopt` draws fresh entropy; a fixed seed
    /// reproduces the run bit-for-bit on any platform.
    [[nodiscard]] SimulationResult simulate(std::optional<std::uint64_t> seed) const;

    [[nodiscard]] GbmProcess with_initial_value(double initial_value) const;
    [[nodiscard]] GbmProcess with_volatility(double volatility) const;
    [[nodiscard]] GbmProcess with_market_data(MarketData market_data) const;
    [[nodiscard]] GbmProcess with_pricing_date(DateTime pricing_date) const;
    [[nodiscard]] GbmProcess with_discount_curve(DiscountCurve discount_curve) const;

private:
    MarketData market_data_;
    GbmParams params_;
    SimulationConfig config_;
};

}  // namespace dp
