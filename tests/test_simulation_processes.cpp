#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/exceptions.hpp"
#include "dp/market_data.hpp"
#include "dp/params.hpp"
#include "dp/simulation.hpp"
#include "dp/underlying.hpp"
#include "dp/valuation.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace dp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using dp::test::kMaturity;
using dp::test::kPricingDate;
using dp::test::message_contains;
using dp::test::WarningCapture;

namespace {

constexpr double kRate = 0.05;

MarketData md(double rate = kRate) {
    return {kPricingDate, DiscountCurve::flat(rate), "USD"};
}

SimulationConfig config(int paths, int num_steps = 26) {
    return {.paths = paths, .end_date = kMaturity, .num_steps = num_steps};
}

JdParams jd_params(double jump_intensity = 0.4, double mu = -0.1, double delta = 0.15) {
    return {.initial_value = 100.0,
            .volatility = 0.20,
            .jump_intensity = jump_intensity,
            .mu = mu,
            .delta = delta};
}

SrdParams srd_params() {
    return {.initial_value = 0.04, .volatility = 0.15, .kappa = 1.5, .theta = 0.06};
}

VanillaSpec spec(OptionType type, ExerciseType exercise = ExerciseType::European) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = exercise,
                        .strike = 100.0,
                        .maturity = kMaturity,
                        .currency = "USD"});
}

/// Merton (1976) closed-form European price: a Poisson-weighted sum of BSM
/// prices with jump-adjusted rate and volatility per jump count.
double merton_price(OptionType type,
                    double spot,
                    double sigma,
                    double rate,
                    double jump_intensity,
                    double mu,
                    double delta,
                    double maturity) {
    const double kbar = std::exp(mu + 0.5 * delta * delta) - 1.0;
    const double lambda_prime = jump_intensity * (1.0 + kbar);
    double weight = std::exp(-lambda_prime * maturity);
    double price = 0.0;
    for (int n = 0; n < 60; ++n) {
        if (n > 0) {
            weight *= lambda_prime * maturity / n;
        }
        const double sigma_n = std::sqrt(sigma * sigma + n * delta * delta / maturity);
        const double rate_n = rate - jump_intensity * kbar + n * std::log1p(kbar) / maturity;
        const UnderlyingData leg(spot, sigma_n, md(rate_n));
        price +=
            weight * OptionValuation(leg, spec(type), PricingMethod::Analytical).present_value();
    }
    return price;
}

/// Uniform grid from the pricing date to maturity with `steps` intervals.
std::vector<DateTime> uniform_grid(int steps) {
    return shared_time_grid(kPricingDate, kMaturity, steps);
}

std::shared_ptr<const CorrelationContext> two_asset_context(double rho,
                                                            Eigen::Index paths,
                                                            int steps,
                                                            std::uint64_t seed = 7) {
    Eigen::MatrixXd corr(2, 2);
    corr << 1.0, rho, rho, 1.0;
    return std::make_shared<const CorrelationContext>(CorrelationContext::from_seed(
        corr, {"asset_a", "asset_b"}, uniform_grid(steps), paths, seed));
}

/// Config for a process on `context`'s shared grid.
SimulationConfig shared_config(const std::shared_ptr<const CorrelationContext>& context) {
    return {.paths = static_cast<int>(context->paths()), .time_grid = context->time_grid()};
}

}  // namespace

// ---------------------------------------------------------------------------
// Jump diffusion (Merton)
// ---------------------------------------------------------------------------

TEST_CASE("JdParams validation", "[processes][jd]") {
    CHECK_THROWS_MATCHES(JdProcess(md(), jd_params(-0.1), config(100)),
                         ValidationError,
                         message_contains("JdParams.jump_intensity"));
    CHECK_THROWS_MATCHES(JdProcess(md(), jd_params(0.4, -0.1, -0.2), config(100)),
                         ValidationError,
                         message_contains("JdParams.delta"));
    CHECK_THROWS_MATCHES(JdProcess(md(), jd_params(0.4, dp::test::kNaN), config(100)),
                         ValidationError,
                         message_contains("JdParams.mu"));
    CHECK_THROWS_MATCHES(
        JdProcess(
            md(),
            {.initial_value = -5.0, .volatility = 0.2, .jump_intensity = 0, .mu = 0, .delta = 0},
            config(100)),
        ValidationError,
        message_contains("JdParams.initial_value"));
}

TEST_CASE("JD with zero intensity reproduces GBM bit-for-bit", "[processes][jd]") {
    const GbmProcess gbm(md(), {.initial_value = 100.0, .volatility = 0.20}, config(1000, 12));
    const JdProcess jd(md(), jd_params(0.0), config(1000, 12));

    const SimulationResult g = gbm.simulate(42);
    const SimulationResult j = jd.simulate(42);
    REQUIRE(g.paths.rows() == j.paths.rows());
    REQUIRE(g.paths.cols() == j.paths.cols());
    CHECK((g.paths - j.paths).cwiseAbs().maxCoeff() == 0.0);
    CHECK((g.normals - j.normals).cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("JD paths are reproducible per seed and vary across seeds", "[processes][jd]") {
    const JdProcess jd(md(), jd_params(3.0), config(500, 12));
    const SimulationResult a = jd.simulate(123);
    const SimulationResult b = jd.simulate(123);
    const SimulationResult c = jd.simulate(124);
    CHECK((a.paths - b.paths).cwiseAbs().maxCoeff() == 0.0);
    CHECK((a.paths - c.paths).cwiseAbs().maxCoeff() > 0.0);
}

TEST_CASE("JD terminal mean matches the risk-neutral forward", "[processes][jd][statistical]") {
    const JdProcess jd(md(), jd_params(), config(200'000, 26));
    const SimulationResult result = jd.simulate(42);
    const double forward = 100.0 * std::exp(kRate * result.time_grid[result.time_grid.size() - 1]);
    const double mean = result.paths.col(result.paths.cols() - 1).mean();
    CHECK_THAT(mean, WithinRel(forward, 1e-2));
}

TEST_CASE("JD European price matches the Merton closed form", "[processes][jd][statistical]") {
    // High-order series legs use negative jump-adjusted rates; the discount
    // curve warns about them, which is expected here.
    const WarningCapture warnings;
    const double maturity = year_fraction(kPricingDate, kMaturity, DayCountConvention::Act365F);
    const double reference =
        merton_price(OptionType::Call, 100.0, 0.20, kRate, 0.4, -0.1, 0.15, maturity);

    const JdProcess jd(md(), jd_params(), config(200'000, 26));
    const OptionValuation mc(jd, spec(OptionType::Call), PricingMethod::MonteCarlo);
    CHECK_THAT(mc.present_value(), WithinRel(reference, 1.5e-2));
}

TEST_CASE("JD American put is worth at least the European put", "[processes][jd]") {
    const JdProcess jd(md(), jd_params(), config(20'000, 26));
    const double european =
        OptionValuation(jd, spec(OptionType::Put), PricingMethod::MonteCarlo).present_value();
    const double american =
        OptionValuation(
            jd, spec(OptionType::Put, ExerciseType::American), PricingMethod::MonteCarlo)
            .present_value();
    CHECK(american >= european - 0.05);
    CHECK(american > 0.0);
}

// ---------------------------------------------------------------------------
// Square-root diffusion (CIR)
// ---------------------------------------------------------------------------

TEST_CASE("SrdParams validation and Feller warning", "[processes][srd]") {
    CHECK_THROWS_MATCHES(
        SrdProcess(md(),
                   {.initial_value = 0.04, .volatility = 0.15, .kappa = -1.0, .theta = 0.06},
                   config(100)),
        ValidationError,
        message_contains("SrdParams.kappa"));
    CHECK_THROWS_MATCHES(
        SrdProcess(md(),
                   {.initial_value = 0.04, .volatility = -0.15, .kappa = 1.0, .theta = 0.06},
                   config(100)),
        ValidationError,
        message_contains("SrdParams.volatility"));

    const WarningCapture warnings;
    // 2*kappa*theta = 0.004 < sigma^2 = 0.09 violates the Feller condition.
    const SrdProcess violating(
        md(),
        {.initial_value = 0.04, .volatility = 0.30, .kappa = 0.1, .theta = 0.02},
        config(100));
    REQUIRE_FALSE(warnings.empty());
    CHECK(warnings.messages().front().find("Feller condition violated") != std::string::npos);
}

TEST_CASE("SRD paths are non-negative and reproducible", "[processes][srd]") {
    const SrdProcess srd(md(), srd_params(), config(2000, 26));
    const SimulationResult a = srd.simulate(99);
    const SimulationResult b = srd.simulate(99);
    CHECK(a.paths.minCoeff() >= 0.0);
    CHECK((a.paths - b.paths).cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("SRD terminal mean matches the CIR expectation", "[processes][srd][statistical]") {
    const SrdProcess srd(md(), srd_params(), config(100'000, 50));
    const SimulationResult result = srd.simulate(42);
    const double maturity = result.time_grid[result.time_grid.size() - 1];
    // E[X_T] = theta + (X_0 - theta) e^{-kappa T} for the CIR process.
    const SrdParams p = srd_params();
    const double expected = p.theta + (p.initial_value - p.theta) * std::exp(-p.kappa * maturity);
    const double mean = result.paths.col(result.paths.cols() - 1).mean();
    CHECK_THAT(mean, WithinRel(expected, 2e-2));
}

TEST_CASE("SRD mean-reverts towards theta", "[processes][srd][statistical]") {
    // Very high kappa pulls the terminal mean close to theta.
    const SrdProcess fast(md(),
                          {.initial_value = 0.03, .volatility = 0.01, .kappa = 20.0, .theta = 0.08},
                          config(5000, 50));
    const SimulationResult fast_result = fast.simulate(8);
    CHECK_THAT(fast_result.paths.col(fast_result.paths.cols() - 1).mean(), WithinAbs(0.08, 0.005));

    // Nearly-zero kappa leaves the process close to its start (and violates
    // the Feller condition, which warns by design).
    const WarningCapture warnings;
    const SrdProcess slow(
        md(),
        {.initial_value = 0.03, .volatility = 0.01, .kappa = 1.0e-4, .theta = 0.20},
        config(5000, 50));
    const SimulationResult slow_result = slow.simulate(7);
    CHECK_THAT(slow_result.paths.col(slow_result.paths.cols() - 1).mean(), WithinAbs(0.03, 0.005));
}

// ---------------------------------------------------------------------------
// CorrelationContext
// ---------------------------------------------------------------------------

TEST_CASE("CorrelationContext validation", "[processes][correlation]") {
    const auto rn = [](std::size_t n) { return CorrelationContext::standard_normals(n, 4, 3, 1); };
    const std::vector<std::string> names{"a", "b"};

    CHECK_THROWS_MATCHES(
        CorrelationContext(Eigen::MatrixXd::Ones(2, 3), rn(2), names, uniform_grid(3)),
        ValidationError,
        message_contains("must be square"));

    Eigen::MatrixXd asymmetric(2, 2);
    asymmetric << 1.0, 0.5, 0.3, 1.0;
    CHECK_THROWS_MATCHES(CorrelationContext(asymmetric, rn(2), names, uniform_grid(3)),
                         ValidationError,
                         message_contains("must be symmetric"));

    Eigen::MatrixXd bad_diagonal(2, 2);
    bad_diagonal << 1.1, 0.0, 0.0, 1.0;
    CHECK_THROWS_MATCHES(CorrelationContext(bad_diagonal, rn(2), names, uniform_grid(3)),
                         ValidationError,
                         message_contains("diagonal must be 1"));

    Eigen::MatrixXd not_pd(2, 2);
    not_pd << 1.0, 1.5, 1.5, 1.0;
    CHECK_THROWS_MATCHES(CorrelationContext(not_pd, rn(2), names, uniform_grid(3)),
                         ValidationError,
                         message_contains("not positive-definite"));

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(2, 2);
    CHECK_THROWS_MATCHES(CorrelationContext(identity, rn(2), {"a", "a"}, uniform_grid(3)),
                         ValidationError,
                         message_contains("unique"));
    CHECK_THROWS_MATCHES(CorrelationContext(identity, rn(2), {"a", "b", "c"}, uniform_grid(3)),
                         ValidationError,
                         message_contains("asset_names length"));
    CHECK_THROWS_MATCHES(CorrelationContext(identity, rn(3), names, uniform_grid(3)),
                         ValidationError,
                         message_contains("random_numbers count"));

    auto mismatched = rn(2);
    mismatched[1].resize(4, 5);
    CHECK_THROWS_MATCHES(
        CorrelationContext(identity, std::move(mismatched), names, uniform_grid(3)),
        ValidationError,
        message_contains("identically shaped"));

    const CorrelationContext valid(identity, rn(2), names, uniform_grid(3));
    CHECK(valid.asset_index("b") == 1);
    CHECK_THROWS_MATCHES(
        valid.asset_index("zzz"), ValidationError, message_contains("not found in asset_names"));
}

TEST_CASE("GbmParams validation", "[processes][gbm]") {
    CHECK_THROWS_MATCHES(GbmProcess(md(), {.initial_value = 0.0, .volatility = 0.2}, config(10)),
                         ValidationError,
                         message_contains("GbmParams.initial_value must be > 0"));
    CHECK_THROWS_MATCHES(GbmProcess(md(), {.initial_value = 100.0, .volatility = -0.1}, config(10)),
                         ValidationError,
                         message_contains("GbmParams.volatility must be >= 0"));
    // Zero volatility is a valid (deterministic) process.
    CHECK_NOTHROW(GbmProcess(md(), {.initial_value = 100.0, .volatility = 0.0}, config(10)));
}

TEST_CASE("singular and near-singular correlation matrices", "[processes][correlation]") {
    const auto rn = [](std::size_t n) { return CorrelationContext::standard_normals(n, 4, 3, 1); };

    // Perfectly correlated: positive semi-definite but singular, so there is
    // no Cholesky factor.
    const Eigen::MatrixXd singular = Eigen::MatrixXd::Ones(2, 2);
    CHECK_THROWS_MATCHES(CorrelationContext(singular, rn(2), {"a", "b"}, uniform_grid(3)),
                         ValidationError,
                         message_contains("not positive-definite"));

    // Highly but not perfectly correlated assets must still be accepted.
    Eigen::MatrixXd near_singular(2, 2);
    near_singular << 1.0, 0.999, 0.999, 1.0;
    const CorrelationContext context(near_singular, rn(2), {"a", "b"}, uniform_grid(3));
    const Eigen::MatrixXd product =
        context.cholesky_matrix() * context.cholesky_matrix().transpose();
    CHECK((product - near_singular).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("Cholesky factor reconstructs the correlation matrix", "[processes][correlation]") {
    Eigen::MatrixXd corr(3, 3);
    corr << 1.0, 0.6, 0.3, 0.6, 1.0, 0.0, 0.3, 0.0, 1.0;
    const CorrelationContext context(
        corr, CorrelationContext::standard_normals(3, 2, 2, 1), {"a", "b", "c"}, uniform_grid(2));
    const Eigen::MatrixXd product =
        context.cholesky_matrix() * context.cholesky_matrix().transpose();
    CHECK((product - corr).cwiseAbs().maxCoeff() < 1e-12);

    // Four assets with zero and negative correlations.
    Eigen::MatrixXd corr4(4, 4);
    corr4 << 1.0, 0.3, -0.2, 0.0,  //
        0.3, 1.0, 0.0, -0.4,       //
        -0.2, 0.0, 1.0, 0.1,       //
        0.0, -0.4, 0.1, 1.0;
    const CorrelationContext context4(corr4,
                                      CorrelationContext::standard_normals(4, 2, 2, 1),
                                      {"a", "b", "c", "d"},
                                      uniform_grid(2));
    const Eigen::MatrixXd product4 =
        context4.cholesky_matrix() * context4.cholesky_matrix().transpose();
    CHECK((product4 - corr4).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("correlated processes require a registered name", "[processes][correlation]") {
    const auto context = two_asset_context(0.7, 100, 26);
    CHECK_THROWS_MATCHES(
        GbmProcess(md(), {.initial_value = 100.0, .volatility = 0.2}, config(100), context),
        ValidationError,
        message_contains("name is required when using CorrelationContext"));
    CHECK_THROWS_MATCHES(
        GbmProcess(
            md(), {.initial_value = 100.0, .volatility = 0.2}, config(100), context, "unknown"),
        ValidationError,
        message_contains("not found in asset_names"));
}

TEST_CASE("correlated shocks reproduce the requested correlation",
          "[processes][correlation][statistical]") {
    const double rho = 0.7;
    const auto context = two_asset_context(rho, 20'000, 26);
    const GbmProcess a(md(),
                       {.initial_value = 100.0, .volatility = 0.2},
                       shared_config(context),
                       context,
                       "asset_a");
    const GbmProcess b(md(),
                       {.initial_value = 80.0, .volatility = 0.3},
                       shared_config(context),
                       context,
                       "asset_b");

    const SimulationResult ra = a.simulate(std::nullopt);
    const SimulationResult rb = b.simulate(std::nullopt);

    // The stored shocks are the correlated draws; their sample correlation
    // over paths x steps should recover rho. (Plain loop: GCC's
    // -Wnull-dereference false-positives on Eigen reductions here.)
    const auto n = static_cast<std::size_t>(ra.normals.size());
    REQUIRE(n > 0);
    const std::span<const double> za{ra.normals.data(), n};
    const std::span<const double> zb{rb.normals.data(), n};
    double sum_a = 0.0;
    double sum_b = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum_a += za[i];
        sum_b += zb[i];
    }
    const double mean_a = sum_a / static_cast<double>(n);
    const double mean_b = sum_b / static_cast<double>(n);
    double covariance = 0.0;
    double var_a = 0.0;
    double var_b = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        covariance += (za[i] - mean_a) * (zb[i] - mean_b);
        var_a += (za[i] - mean_a) * (za[i] - mean_a);
        var_b += (zb[i] - mean_b) * (zb[i] - mean_b);
    }
    const double corr = covariance / std::sqrt(var_a * var_b);
    CHECK_THAT(corr, WithinAbs(rho, 0.01));

    // The context owns all randomness: repeated simulation is deterministic.
    const SimulationResult ra2 = a.simulate(std::nullopt);
    CHECK((ra.paths - ra2.paths).cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("correlated processes must simulate on the context's shared grid",
          "[processes][correlation]") {
    const auto context = two_asset_context(0.5, 100, 5);
    const GbmParams params{.initial_value = 100.0, .volatility = 0.2};

    // A generated grid (end_date + num_steps), even an identical one, is
    // rejected: the grid must come from the context.
    CHECK_THROWS_MATCHES(GbmProcess(md(), params, config(100, 5), context, "asset_a"),
                         ValidationError,
                         message_contains("must simulate on the context's shared grid"));

    SimulationConfig other = shared_config(context);
    other.time_grid = uniform_grid(6);
    CHECK_THROWS_MATCHES(GbmProcess(md(), params, other, context, "asset_a"),
                         ValidationError,
                         message_contains("differs from its CorrelationContext's shared grid"));

    SimulationConfig wrong_paths = shared_config(context);
    wrong_paths.paths = 200;
    CHECK_THROWS_MATCHES(GbmProcess(md(), params, wrong_paths, context, "asset_a"),
                         ValidationError,
                         message_contains("draws cover 100"));

    const DateTime off_grid = kPricingDate + std::chrono::days{50};
    GbmParams with_dividend = params;
    with_dividend.discrete_dividends = {{off_grid, 1.0}};
    CHECK_THROWS_MATCHES(
        GbmProcess(md(), with_dividend, shared_config(context), context, "asset_a"),
        ValidationError,
        message_contains("missing from its CorrelationContext's shared grid"));

    SimulationConfig with_observation = shared_config(context);
    with_observation.observation_dates = {off_grid};
    CHECK_THROWS_MATCHES(GbmProcess(md(), params, with_observation, context, "asset_a"),
                         ValidationError,
                         message_contains("missing from its CorrelationContext's shared grid"));

    const MarketData later(kPricingDate + std::chrono::days{1}, DiscountCurve::flat(kRate), "USD");
    CHECK_THROWS_MATCHES(GbmProcess(later, params, shared_config(context), context, "asset_a"),
                         ValidationError,
                         message_contains("must start at the pricing date"));
}

TEST_CASE("CorrelationContext validates its time grid", "[processes][correlation]") {
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(2, 2);
    const std::vector<std::string> names{"a", "b"};
    const auto rn = CorrelationContext::standard_normals(2, 4, 3, 1);

    CHECK_THROWS_MATCHES(CorrelationContext(identity, rn, names, uniform_grid(4)),
                         ValidationError,
                         message_contains("step count (3) must equal the number of time_grid "
                                          "intervals (4)"));
    auto unsorted = uniform_grid(3);
    std::swap(unsorted[1], unsorted[2]);
    CHECK_THROWS_MATCHES(CorrelationContext(identity, rn, names, unsorted),
                         ValidationError,
                         message_contains("strictly ascending"));
    CHECK_THROWS_MATCHES(CorrelationContext(identity, rn, names, {kPricingDate}),
                         ValidationError,
                         message_contains("at least two dates"));

    const auto a = CorrelationContext::from_seed(identity, names, uniform_grid(12), 64, 7);
    const auto b = CorrelationContext::from_seed(identity, names, uniform_grid(12), 64, 7);
    CHECK(a.paths() == 64);
    CHECK(a.random_numbers().front().cols() == 12);
    CHECK(a.random_numbers().front() == b.random_numbers().front());
    CHECK(a.time_grid() == uniform_grid(12));
}

TEST_CASE("shared_time_grid merges event dates into the uniform grid", "[processes][grid]") {
    const DateTime ex_date = kPricingDate + std::chrono::days{136};
    const DateTime early = kPricingDate - std::chrono::days{5};
    const std::vector<DateTime> extra{ex_date, early};
    const auto grid = shared_time_grid(kPricingDate, kMaturity, 12, {.extra_dates = extra});
    CHECK(grid.front() == kPricingDate);
    CHECK(grid.back() == kMaturity);
    CHECK(std::binary_search(grid.begin(), grid.end(), ex_date));
    CHECK_FALSE(std::binary_search(grid.begin(), grid.end(), early));
    CHECK(grid.size() == 14);

    // A process built from end_date + num_steps uses the same construction.
    const GbmProcess process(
        md(),
        {.initial_value = 100.0, .volatility = 0.2, .discrete_dividends = {{ex_date, 1.0}}},
        config(8, 12));
    const std::vector<DateTime> ex_dates{ex_date};
    CHECK(process.build_dates() ==
          shared_time_grid(kPricingDate, kMaturity, 12, {.extra_dates = ex_dates}));
}

TEST_CASE("GridEvents collects contract and ex-dividend dates", "[processes][grid]") {
    const BarrierSpec barrier({.option_type = OptionType::Call,
                               .exercise_type = ExerciseType::European,
                               .strike = 95.0,
                               .maturity = kMaturity,
                               .barrier = 120.0,
                               .direction = BarrierDirection::Up,
                               .action = BarrierAction::Out,
                               .monitoring = BarrierMonitoring::Discrete,
                               .currency = "USD",
                               .num_observations = 5});
    const DateTime ex_date = kPricingDate + std::chrono::days{136};
    const DateTime late_ex = kMaturity + std::chrono::days{30};
    const GbmParams params_a{.initial_value = 100.0,
                             .volatility = 0.2,
                             .discrete_dividends = {{ex_date, 2.5}, {late_ex, 1.0}}};
    const GbmParams params_b{.initial_value = 90.0, .volatility = 0.3};

    const auto grid =
        shared_time_grid(kPricingDate,
                         kMaturity,
                         12,
                         {.specs = {spec(OptionType::Call), barrier},
                          .dividends = {params_a.discrete_dividends, params_b.discrete_dividends}});
    for (const DateTime d : contract_event_dates(barrier, kPricingDate)) {
        CHECK(std::binary_search(grid.begin(), grid.end(), d));
    }
    CHECK(std::binary_search(grid.begin(), grid.end(), ex_date));
    // After end_date: never affects the simulation, so not on the grid.
    CHECK_FALSE(std::binary_search(grid.begin(), grid.end(), late_ex));
    CHECK(grid.back() == kMaturity);

    // The intended workflow: one grid serves every asset and contract.
    Eigen::MatrixXd corr(2, 2);
    corr << 1.0, 0.5, 0.5, 1.0;
    const auto context = std::make_shared<const CorrelationContext>(
        CorrelationContext::from_seed(corr, {"asset_a", "asset_b"}, grid, 2000, 5));
    const GbmProcess a(md(), params_a, shared_config(context), context, "asset_a");
    const GbmProcess b(md(), params_b, shared_config(context), context, "asset_b");
    for (const GbmProcess* process : {&a, &b}) {
        CHECK(OptionValuation(*process, spec(OptionType::Call), PricingMethod::MonteCarlo)
                  .present_value() > 0.0);
        CHECK(OptionValuation(*process, barrier, PricingMethod::MonteCarlo).present_value() > 0.0);
    }
}

TEST_CASE("contract_event_dates lists maturity and monitoring dates", "[processes][grid]") {
    CHECK(contract_event_dates(spec(OptionType::Call), kPricingDate) ==
          std::vector<DateTime>{kMaturity});
    const BarrierSpec barrier({.option_type = OptionType::Call,
                               .exercise_type = ExerciseType::European,
                               .strike = 100.0,
                               .maturity = kMaturity,
                               .barrier = 120.0,
                               .direction = BarrierDirection::Up,
                               .action = BarrierAction::Out,
                               .monitoring = BarrierMonitoring::Discrete,
                               .currency = "USD",
                               .num_observations = 4});
    const auto dates = contract_event_dates(barrier, kPricingDate);
    REQUIRE(dates.size() == 4);
    CHECK(dates.back() == kMaturity);
}

TEST_CASE("increments stay correlated across one asset's ex-dividend date",
          "[processes][correlation][statistical]") {
    // Only asset_a pays a dividend. Before the grid lived on the context, its
    // extra ex-date column shifted its draws against asset_b's and the
    // realised correlation of later increments fell from 0.9 to about 0.
    constexpr double kRho = 0.9;
    const DateTime ex_date = kPricingDate + std::chrono::days{136};
    const std::vector<DateTime> ex_dates{ex_date};
    const auto grid = shared_time_grid(kPricingDate, kMaturity, 12, {.extra_dates = ex_dates});
    Eigen::MatrixXd corr(2, 2);
    corr << 1.0, kRho, kRho, 1.0;
    const auto context = std::make_shared<const CorrelationContext>(
        CorrelationContext::from_seed(corr, {"asset_a", "asset_b"}, grid, 20'000, 11));
    const GbmProcess a(
        md(),
        {.initial_value = 100.0, .volatility = 0.2, .discrete_dividends = {{ex_date, 2.0}}},
        shared_config(context),
        context,
        "asset_a");
    const GbmProcess b(md(),
                       {.initial_value = 100.0, .volatility = 0.2},
                       shared_config(context),
                       context,
                       "asset_b");
    const Eigen::MatrixXd pa = a.simulate(std::nullopt).paths;
    const Eigen::MatrixXd pb = b.simulate(std::nullopt).paths;

    const auto ex_step = std::distance(grid.begin(), std::find(grid.begin(), grid.end(), ex_date));
    for (Eigen::Index step = 1; step < pa.cols(); ++step) {
        if (step == ex_step) {
            continue;  // the dividend drop is not a diffusion increment
        }
        const Eigen::ArrayXd za = (pa.col(step).array() / pa.col(step - 1).array()).log();
        const Eigen::ArrayXd zb = (pb.col(step).array() / pb.col(step - 1).array()).log();
        const Eigen::ArrayXd da = za - za.mean();
        const Eigen::ArrayXd db = zb - zb.mean();
        const double rho = (da * db).sum() / std::sqrt(da.square().sum() * db.square().sum());
        CAPTURE(step, rho);
        CHECK_THAT(rho, WithinAbs(kRho, 0.02));
    }
}

TEST_CASE("valuations on a correlated process respect the shared grid",
          "[processes][correlation][valuation]") {
    const BarrierSpec barrier({.option_type = OptionType::Call,
                               .exercise_type = ExerciseType::European,
                               .strike = 100.0,
                               .maturity = kMaturity,
                               .barrier = 130.0,
                               .direction = BarrierDirection::Up,
                               .action = BarrierAction::Out,
                               .monitoring = BarrierMonitoring::Discrete,
                               .currency = "USD",
                               .num_observations = 5});
    const GbmParams params{.initial_value = 100.0, .volatility = 0.2};

    SECTION("contract dates missing from the grid are rejected") {
        const auto context = two_asset_context(0.5, 1000, 12);
        const GbmProcess process(md(), params, shared_config(context), context, "asset_a");
        CHECK_THROWS_MATCHES(OptionValuation(process, barrier, PricingMethod::MonteCarlo),
                             ValidationError,
                             message_contains("missing from the CorrelationContext"));
    }
    SECTION("a grid holding the contract dates prices") {
        Eigen::MatrixXd corr(2, 2);
        corr << 1.0, 0.5, 0.5, 1.0;
        const auto context =
            std::make_shared<const CorrelationContext>(CorrelationContext::from_seed(
                corr,
                {"asset_a", "asset_b"},
                shared_time_grid(kPricingDate, kMaturity, 12, {.specs = {barrier}}),
                1000,
                3));
        const GbmProcess process(md(), params, shared_config(context), context, "asset_a");
        CHECK(OptionValuation(process, barrier, PricingMethod::MonteCarlo).present_value() > 0.0);
    }
    SECTION("numerical theta is rejected; pathwise theta works") {
        const auto context = two_asset_context(0.5, 1000, 12);
        const GbmProcess process(md(), params, shared_config(context), context, "asset_a");
        const OptionValuation valuation(process, spec(OptionType::Call), PricingMethod::MonteCarlo);
        CHECK_THROWS_MATCHES(valuation.theta({.method = GreekCalculationMethod::Numerical}),
                             UnsupportedFeatureError,
                             message_contains("Numerical theta moves the pricing date"));
        CHECK(valuation.theta({.method = GreekCalculationMethod::Pathwise}) < 0.0);
    }
}

TEST_CASE("correlated GBM prices European options like independent GBM",
          "[processes][correlation][statistical]") {
    const auto context = two_asset_context(0.6, 100'000, 26);
    const GbmProcess correlated(md(),
                                {.initial_value = 100.0, .volatility = 0.2},
                                shared_config(context),
                                context,
                                "asset_a");
    const UnderlyingData analytic(100.0, 0.2, md());

    const double mc = OptionValuation(correlated, spec(OptionType::Call), PricingMethod::MonteCarlo)
                          .present_value();
    const double bsm = OptionValuation(analytic, spec(OptionType::Call), PricingMethod::Analytical)
                           .present_value();
    CHECK_THAT(mc, WithinRel(bsm, 2e-2));
}

// ---------------------------------------------------------------------------
// Facade gating for non-GBM processes
// ---------------------------------------------------------------------------

TEST_CASE("non-GBM processes are rejected where GBM is required", "[processes][valuation]") {
    const JdProcess jd(md(), jd_params(), config(1000, 12));
    const SrdProcess srd(md(), srd_params(), config(1000, 12));

    // Deterministic methods refuse any path process, naming the actual type.
    CHECK_THROWS_MATCHES(OptionValuation(jd, spec(OptionType::Call), PricingMethod::Binomial),
                         ConfigurationError,
                         message_contains("got JdProcess"));
    CHECK_THROWS_MATCHES(OptionValuation(srd, spec(OptionType::Call), PricingMethod::Analytical),
                         ConfigurationError,
                         message_contains("got SrdProcess"));

    // Pathwise / LR Greeks are GBM-only.
    const OptionValuation jd_mc(jd, spec(OptionType::Call), PricingMethod::MonteCarlo);
    CHECK_THROWS_MATCHES(
        jd_mc.delta({.method = GreekCalculationMethod::Pathwise}),
        ValidationError,
        message_contains("MC greeks are only available for GBMProcess underlying"));

    // The BSM control-variate leg cannot be built from a jump process.
    const MonteCarloParams cv{.control_variate_european = true};
    CHECK_THROWS_MATCHES(
        OptionValuation(
            jd, spec(OptionType::Put, ExerciseType::American), PricingMethod::MonteCarlo, cv)
            .present_value(),
        ConfigurationError,
        message_contains("Expected UnderlyingData or GbmProcess, got JdProcess."));
    CHECK_THROWS_MATCHES(
        OptionValuation(
            srd, spec(OptionType::Put, ExerciseType::American), PricingMethod::MonteCarlo, cv)
            .present_value(),
        ConfigurationError,
        message_contains("Expected UnderlyingData or GbmProcess, got SrdProcess."));
}

TEST_CASE("numerical Greeks work for JD and SRD underlyings", "[processes][valuation]") {
    const JdProcess jd(md(), jd_params(), config(20'000, 12));
    const OptionValuation call(jd, spec(OptionType::Call), PricingMethod::MonteCarlo);
    const double delta = call.delta();
    CHECK(delta > 0.0);
    CHECK(delta < 1.0);

    const SrdProcess srd(md(), srd_params(), config(20'000, 12));
    const VanillaSpec rate_call({.option_type = OptionType::Call,
                                 .exercise_type = ExerciseType::European,
                                 .strike = 0.05,
                                 .maturity = kMaturity,
                                 .currency = "USD"});
    const OptionValuation srd_call(srd, rate_call, PricingMethod::MonteCarlo);
    CHECK(srd_call.present_value() > 0.0);
    CHECK(srd_call.delta() > 0.0);
}

TEST_CASE("contracts maturing before the end of a shared grid price at maturity",
          "[processes][correlation][grid][statistical]") {
    // A shared grid usually runs past a short-dated contract's maturity. Every
    // MC pricer must stop at the maturity column: the extra columns are
    // simulated but must not touch the payoff, exercise or monitoring.
    const DateTime short_maturity = make_datetime(2025, 7, 1);
    const std::vector<DateTime> short_dates{short_maturity};
    const auto grid = shared_time_grid(kPricingDate, kMaturity, 24, {.extra_dates = short_dates});
    REQUIRE(grid.back() > short_maturity);
    const auto context = std::make_shared<const CorrelationContext>(CorrelationContext::from_seed(
        Eigen::MatrixXd::Identity(2, 2), {"asset_a", "asset_b"}, grid, 100'000, 3));
    const GbmProcess process(md(),
                             {.initial_value = 100.0, .volatility = 0.2},
                             shared_config(context),
                             context,
                             "asset_a");
    const UnderlyingData ud(100.0, 0.2, md());

    const auto within_4_se = [&](const OptionSpec& contract, double reference) {
        const OptionValuation mc(process, contract, PricingMethod::MonteCarlo);
        const auto se = mc.std_error();
        REQUIRE(se.has_value());
        CAPTURE(mc.present_value(), reference, se.value_or(0.0));
        CHECK(std::abs(mc.present_value() - reference) < 4.0 * se.value_or(0.0));
    };
    const auto vanilla = [&](OptionType type, ExerciseType exercise) {
        return VanillaSpec({.option_type = type,
                            .exercise_type = exercise,
                            .strike = 100.0,
                            .maturity = short_maturity,
                            .currency = "USD"});
    };

    SECTION("European") {
        const auto call = vanilla(OptionType::Call, ExerciseType::European);
        within_4_se(call, OptionValuation(ud, call, PricingMethod::Analytical).present_value());
    }
    SECTION("American") {
        const auto put = vanilla(OptionType::Put, ExerciseType::American);
        within_4_se(
            put,
            OptionValuation(ud, put, PricingMethod::Binomial, BinomialParams{.num_steps = 2000})
                .present_value());
    }
    SECTION("continuous barrier") {
        const BarrierSpec barrier({.option_type = OptionType::Call,
                                   .exercise_type = ExerciseType::European,
                                   .strike = 100.0,
                                   .maturity = short_maturity,
                                   .barrier = 120.0,
                                   .direction = BarrierDirection::Up,
                                   .action = BarrierAction::Out,
                                   .currency = "USD"});
        within_4_se(barrier,
                    OptionValuation(ud, barrier, PricingMethod::Analytical).present_value());
    }
    SECTION("pathwise greeks use the maturity column") {
        const auto call = vanilla(OptionType::Call, ExerciseType::European);
        const OptionValuation mc(process, call, PricingMethod::MonteCarlo);
        const OptionValuation bsm(ud, call, PricingMethod::Analytical);
        constexpr GreekOptions kPathwiseOptions{.method = GreekCalculationMethod::Pathwise};
        CHECK_THAT(mc.delta(kPathwiseOptions), WithinRel(bsm.delta(), 1e-2));
        CHECK_THAT(mc.vega(kPathwiseOptions), WithinRel(bsm.vega(), 2e-2));
    }
}
