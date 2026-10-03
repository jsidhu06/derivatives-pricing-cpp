// Path-simulation mechanics: the variance-reduction switches, shock
// invariance under parameter bumps (what makes bump-and-revalue Greeks
// common-random-number), grid construction rules, and jump-diffusion
// behaviour. Ports the corresponding tests of the Python suite's
// test_stochastic_processes.py.

#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/exceptions.hpp"
#include "dp/market_data.hpp"
#include "dp/simulation.hpp"

#include "test_helpers.hpp"

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <vector>

using namespace dp;
using dp::test::close_to;
using dp::test::kMaturity;
using dp::test::kPricingDate;
using dp::test::message_contains;
using dp::test::WarningCapture;

namespace {

using std::chrono::days;

MarketData md(double rate = 0.05) {
    return {kPricingDate, DiscountCurve::flat(rate), "USD"};
}

SimulationConfig config(int paths, bool antithetic = true, bool moment_matching = true) {
    return {.paths = paths,
            .end_date = kMaturity,
            .num_steps = 12,
            .antithetic = antithetic,
            .moment_matching = moment_matching};
}

GbmProcess gbm(SimulationConfig cfg, double spot = 100.0, double vol = 0.2) {
    return {md(), {.initial_value = spot, .volatility = vol}, std::move(cfg)};
}

/// Population standard deviation of each column.
Eigen::ArrayXd column_stddev(const Eigen::MatrixXd& m) {
    const Eigen::RowVectorXd mean = m.colwise().mean();
    return ((m.rowwise() - mean).array().square().colwise().mean()).sqrt().transpose();
}

/// Sample standard deviation of log terminal spots.
double log_terminal_stddev(const Eigen::MatrixXd& paths) {
    const Eigen::ArrayXd log_st = paths.col(paths.cols() - 1).array().log();
    return std::sqrt((log_st - log_st.mean()).square().sum() /
                     static_cast<double>(log_st.size() - 1));
}

}  // namespace

// ---------------------------------------------------------------------------
// Variance-reduction switches
// ---------------------------------------------------------------------------

TEST_CASE("moment matching standardises each step's shocks", "[simulation][variance_reduction]") {
    const Eigen::MatrixXd matched = gbm(config(1001, false, true)).simulate(7).normals;
    CHECK(matched.colwise().mean().cwiseAbs().maxCoeff() < 1e-12);
    CHECK((column_stddev(matched) - 1.0).abs().maxCoeff() < 1e-12);

    // Without it the raw draws carry ordinary sampling error.
    const Eigen::MatrixXd raw = gbm(config(1001, false, false)).simulate(7).normals;
    CHECK(raw.colwise().mean().cwiseAbs().maxCoeff() > 1e-6);
    CHECK((column_stddev(raw) - 1.0).abs().maxCoeff() > 1e-6);
}

TEST_CASE("antithetic draws mirror the shocks across the two halves",
          "[simulation][variance_reduction]") {
    const Eigen::MatrixXd mirrored = gbm(config(1000)).simulate(7).normals;
    const Eigen::Index half = mirrored.rows() / 2;
    // Moment matching rescales after mirroring, which keeps the symmetry up
    // to the rounding of the (near-zero) column mean.
    CHECK((mirrored.topRows(half) + mirrored.bottomRows(half)).cwiseAbs().maxCoeff() < 1e-12);

    const Eigen::MatrixXd plain = gbm(config(1000, false)).simulate(7).normals;
    CHECK((plain.topRows(half) + plain.bottomRows(half)).cwiseAbs().maxCoeff() > 0.1);
}

TEST_CASE("simulation with both switches off still produces valid paths",
          "[simulation][variance_reduction]") {
    const SimulationResult sim = gbm(config(20'000, false, false)).simulate(7);
    CHECK(sim.paths.allFinite());
    CHECK((sim.paths.array() > 0.0).all());
    // Terminal mean near the risk-neutral forward S e^{rT} (~4 standard
    // errors at 20k paths).
    const double forward = 100.0 * std::exp(0.05 * sim.time_grid[sim.time_grid.size() - 1]);
    CHECK_THAT(sim.paths.col(sim.paths.cols() - 1).mean(), close_to(forward, 0.0, 0.6));
}

TEST_CASE("antithetic sampling reduces estimator variance", "[simulation][variance_reduction]") {
    // Spread of the terminal-spot mean across seeds, with and without
    // mirroring (moment matching off so it does not mask the effect).
    const auto spread = [](bool antithetic) {
        std::vector<double> means;
        for (std::uint64_t seed = 1; seed <= 20; ++seed) {
            const auto sim = gbm(config(2000, antithetic, false)).simulate(seed);
            means.push_back(sim.paths.col(sim.paths.cols() - 1).mean());
        }
        const Eigen::Map<const Eigen::ArrayXd> m(means.data(),
                                                 static_cast<Eigen::Index>(means.size()));
        return std::sqrt((m - m.mean()).square().mean());
    };
    CHECK(spread(true) < 0.5 * spread(false));
}

TEST_CASE("an odd path count falls back to plain draws", "[simulation][variance_reduction]") {
    const WarningCapture warnings;
    const SimulationResult sim = gbm(config(1001)).simulate(7);
    CHECK(warnings.any_contains("antithetic"));
    CHECK(sim.paths.rows() == 1001);
    CHECK(sim.normals.rows() == 1001);
    CHECK(sim.paths.allFinite());
    CHECK(sim.normals.colwise().mean().cwiseAbs().maxCoeff() < 1e-12);  // still moment-matched
}

// ---------------------------------------------------------------------------
// Shock invariance — common random numbers for bump-and-revalue Greeks
// ---------------------------------------------------------------------------

TEST_CASE("shocks are unchanged when a parameter is bumped", "[simulation][crn]") {
    const GbmProcess base = gbm(config(2000));
    const Eigen::MatrixXd shocks = base.simulate(11).normals;

    CHECK(base.with_initial_value(101.0).simulate(11).normals == shocks);
    CHECK(base.with_volatility(0.21).simulate(11).normals == shocks);
    CHECK(base.with_discount_curve(DiscountCurve::flat(0.06)).simulate(11).normals == shocks);
    CHECK(base.with_pricing_date(kPricingDate + days{1}).simulate(11).normals == shocks);

    const GbmProcess with_dividend_yield(
        md(),
        {.initial_value = 100.0, .volatility = 0.2, .dividend_curve = DiscountCurve::flat(0.02)},
        config(2000));
    CHECK(with_dividend_yield.simulate(11).normals == shocks);
}

// ---------------------------------------------------------------------------
// Grid construction
// ---------------------------------------------------------------------------

TEST_CASE("observation dates before the pricing date are dropped", "[simulation][grid]") {
    SimulationConfig with_past = config(100);
    with_past.observation_dates = {kPricingDate - days{10}, kPricingDate + days{100}};
    SimulationConfig future_only = config(100);
    future_only.observation_dates = {kPricingDate + days{100}};

    const SimulationResult past = gbm(with_past).simulate(3);
    const SimulationResult future = gbm(future_only).simulate(3);
    CHECK(past.dates.front() == kPricingDate);
    CHECK(past.dates == future.dates);
    CHECK(std::ranges::find(past.dates, kPricingDate + days{100}) != past.dates.end());
    CHECK(past.paths == future.paths);  // the excluded date leaves the simulation intact
}

TEST_CASE("grids starting before the pricing date are rejected", "[simulation][grid]") {
    SECTION("explicit time grid") {
        const SimulationConfig cfg{
            .paths = 100,
            .time_grid =
                std::vector<DateTime>{kPricingDate - days{1}, kPricingDate + days{100}, kMaturity}};
        CHECK_THROWS_MATCHES(
            gbm(cfg).simulate(1), ValidationError, message_contains("before pricing_date"));
    }
    SECTION("grid_start") {
        SimulationConfig cfg = config(100);
        cfg.grid_start = kPricingDate - days{5};
        CHECK_THROWS_MATCHES(gbm(cfg).simulate(1),
                             ValidationError,
                             message_contains("Time grid must start at pricing_date"));
    }
}

// ---------------------------------------------------------------------------
// Dispersion and jump behaviour
// ---------------------------------------------------------------------------

TEST_CASE("GBM dispersion grows with volatility", "[simulation][gbm]") {
    const double low = log_terminal_stddev(gbm(config(20'000), 100.0, 0.10).simulate(5).paths);
    const double high = log_terminal_stddev(gbm(config(20'000), 100.0, 0.40).simulate(5).paths);
    CHECK(high > 3.0 * low);  // ~4x in theory (sigma sqrt(T))
}

TEST_CASE("jump-diffusion behaviour", "[simulation][jd]") {
    const auto jd = [](double lambd, double mu, double delta) {
        return JdProcess(
            md(),
            {.initial_value = 100.0, .volatility = 0.2, .lambd = lambd, .mu = mu, .delta = delta},
            config(20'000));
    };

    SECTION("zero-size jumps reproduce GBM even with a positive intensity") {
        // mu = delta = 0: every jump multiplies by e^0 = 1 and the
        // compensator lambda (e^{mu + delta^2/2} - 1) vanishes.
        const Eigen::MatrixXd jumpless = jd(0.5, 0.0, 0.0).simulate(9).paths;
        const Eigen::MatrixXd plain = gbm(config(20'000)).simulate(9).paths;
        CHECK(((jumpless - plain).cwiseAbs().array() / plain.array()).maxCoeff() < 1e-12);
    }
    SECTION("dispersion grows with jump intensity") {
        const double rare = log_terminal_stddev(jd(0.1, -0.1, 0.15).simulate(9).paths);
        const double frequent = log_terminal_stddev(jd(1.5, -0.1, 0.15).simulate(9).paths);
        CHECK(frequent > rare);
    }
    SECTION("paths stay positive") {
        CHECK((jd(1.0, -0.3, 0.3).simulate(9).paths.array() > 0.0).all());
    }
}
