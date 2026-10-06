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

JdParams jd_params(double lambd = 0.4, double mu = -0.1, double delta = 0.15) {
    return {.initial_value = 100.0, .volatility = 0.20, .lambd = lambd, .mu = mu, .delta = delta};
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
                    double lambd,
                    double mu,
                    double delta,
                    double maturity) {
    const double kbar = std::exp(mu + 0.5 * delta * delta) - 1.0;
    const double lambda_prime = lambd * (1.0 + kbar);
    double weight = std::exp(-lambda_prime * maturity);
    double price = 0.0;
    for (int n = 0; n < 60; ++n) {
        if (n > 0) {
            weight *= lambda_prime * maturity / n;
        }
        const double sigma_n = std::sqrt(sigma * sigma + n * delta * delta / maturity);
        const double rate_n = rate - lambd * kbar + n * std::log1p(kbar) / maturity;
        const UnderlyingData leg(spot, sigma_n, md(rate_n));
        price +=
            weight * OptionValuation(leg, spec(type), PricingMethod::Analytical).present_value();
    }
    return price;
}

std::shared_ptr<const CorrelationContext> two_asset_context(double rho,
                                                            Eigen::Index paths,
                                                            Eigen::Index steps,
                                                            std::uint64_t seed = 7) {
    Eigen::MatrixXd corr(2, 2);
    corr << 1.0, rho, rho, 1.0;
    return std::make_shared<const CorrelationContext>(
        corr,
        CorrelationContext::standard_normals(2, paths, steps, seed),
        std::vector<std::string>{"asset_a", "asset_b"});
}

}  // namespace

// ---------------------------------------------------------------------------
// Jump diffusion (Merton)
// ---------------------------------------------------------------------------

TEST_CASE("JdParams validation", "[processes][jd]") {
    CHECK_THROWS_MATCHES(JdProcess(md(), jd_params(-0.1), config(100)),
                         ValidationError,
                         message_contains("JdParams.lambd"));
    CHECK_THROWS_MATCHES(JdProcess(md(), jd_params(0.4, -0.1, -0.2), config(100)),
                         ValidationError,
                         message_contains("JdParams.delta"));
    CHECK_THROWS_MATCHES(JdProcess(md(), jd_params(0.4, dp::test::kNaN), config(100)),
                         ValidationError,
                         message_contains("JdParams.mu"));
    CHECK_THROWS_MATCHES(
        JdProcess(md(),
                  {.initial_value = -5.0, .volatility = 0.2, .lambd = 0, .mu = 0, .delta = 0},
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

    CHECK_THROWS_MATCHES(CorrelationContext(Eigen::MatrixXd::Ones(2, 3), rn(2), names),
                         ValidationError,
                         message_contains("must be square"));

    Eigen::MatrixXd asymmetric(2, 2);
    asymmetric << 1.0, 0.5, 0.3, 1.0;
    CHECK_THROWS_MATCHES(CorrelationContext(asymmetric, rn(2), names),
                         ValidationError,
                         message_contains("must be symmetric"));

    Eigen::MatrixXd bad_diagonal(2, 2);
    bad_diagonal << 1.1, 0.0, 0.0, 1.0;
    CHECK_THROWS_MATCHES(CorrelationContext(bad_diagonal, rn(2), names),
                         ValidationError,
                         message_contains("diagonal must be 1"));

    Eigen::MatrixXd not_pd(2, 2);
    not_pd << 1.0, 1.5, 1.5, 1.0;
    CHECK_THROWS_MATCHES(CorrelationContext(not_pd, rn(2), names),
                         ValidationError,
                         message_contains("not positive-definite"));

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(2, 2);
    CHECK_THROWS_MATCHES(CorrelationContext(identity, rn(2), {"a", "a"}),
                         ValidationError,
                         message_contains("unique"));
    CHECK_THROWS_MATCHES(CorrelationContext(identity, rn(2), {"a", "b", "c"}),
                         ValidationError,
                         message_contains("asset_names length"));
    CHECK_THROWS_MATCHES(CorrelationContext(identity, rn(3), names),
                         ValidationError,
                         message_contains("random_numbers count"));

    auto mismatched = rn(2);
    mismatched[1].resize(4, 5);
    CHECK_THROWS_MATCHES(CorrelationContext(identity, std::move(mismatched), names),
                         ValidationError,
                         message_contains("identically shaped"));

    const CorrelationContext valid(identity, rn(2), names);
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
    CHECK_THROWS_MATCHES(CorrelationContext(singular, rn(2), {"a", "b"}),
                         ValidationError,
                         message_contains("not positive-definite"));

    // Highly but not perfectly correlated assets must still be accepted.
    Eigen::MatrixXd near_singular(2, 2);
    near_singular << 1.0, 0.999, 0.999, 1.0;
    const CorrelationContext context(near_singular, rn(2), {"a", "b"});
    const Eigen::MatrixXd product =
        context.cholesky_matrix() * context.cholesky_matrix().transpose();
    CHECK((product - near_singular).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("Cholesky factor reconstructs the correlation matrix", "[processes][correlation]") {
    Eigen::MatrixXd corr(3, 3);
    corr << 1.0, 0.6, 0.3, 0.6, 1.0, 0.0, 0.3, 0.0, 1.0;
    const CorrelationContext context(
        corr, CorrelationContext::standard_normals(3, 2, 2, 1), {"a", "b", "c"});
    const Eigen::MatrixXd product =
        context.cholesky_matrix() * context.cholesky_matrix().transpose();
    CHECK((product - corr).cwiseAbs().maxCoeff() < 1e-12);

    // Four assets with zero and negative correlations.
    Eigen::MatrixXd corr4(4, 4);
    corr4 << 1.0, 0.3, -0.2, 0.0,  //
        0.3, 1.0, 0.0, -0.4,       //
        -0.2, 0.0, 1.0, 0.1,       //
        0.0, -0.4, 0.1, 1.0;
    const CorrelationContext context4(
        corr4, CorrelationContext::standard_normals(4, 2, 2, 1), {"a", "b", "c", "d"});
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
    const GbmProcess a(
        md(), {.initial_value = 100.0, .volatility = 0.2}, config(20'000), context, "asset_a");
    const GbmProcess b(
        md(), {.initial_value = 80.0, .volatility = 0.3}, config(20'000), context, "asset_b");

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

TEST_CASE("correlated shock matrices must cover the simulation shape", "[processes][correlation]") {
    const auto context = two_asset_context(0.5, 100, 5);
    const GbmProcess wrong_paths(
        md(), {.initial_value = 100.0, .volatility = 0.2}, config(200, 5), context, "asset_a");
    CHECK_THROWS_MATCHES(
        wrong_paths.simulate(std::nullopt), ConfigurationError, message_contains("paths"));
    const GbmProcess wrong_steps(
        md(), {.initial_value = 100.0, .volatility = 0.2}, config(100, 10), context, "asset_a");
    CHECK_THROWS_MATCHES(
        wrong_steps.simulate(std::nullopt), ConfigurationError, message_contains("steps"));
}

TEST_CASE("correlated GBM prices European options like independent GBM",
          "[processes][correlation][statistical]") {
    const auto context = two_asset_context(0.6, 100'000, 26);
    const GbmProcess correlated(
        md(), {.initial_value = 100.0, .volatility = 0.2}, config(100'000), context, "asset_a");
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
