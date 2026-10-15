#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/market_data.hpp"
#include "dp/params.hpp"
#include "dp/simulation.hpp"
#include "dp/underlying.hpp"
#include "dp/valuation.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstddef>
#include <span>
#include <tuple>
#include <vector>

// Convergence-order tests: fit the log-log slope of the error against the
// discretisation parameter, so a regression to a lower order fails even
// when every individual price is still within a loose absolute tolerance.

using namespace dp;
using dp::test::kMaturity;
using dp::test::kPricingDate;

namespace {

MarketData md() {
    return {kPricingDate, DiscountCurve::flat(0.05), "USD"};
}

UnderlyingData ud() {
    return {100.0, 0.20, md()};
}

VanillaSpec spec(OptionType type, double strike) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = ExerciseType::European,
                        .strike = strike,
                        .maturity = kMaturity,
                        .currency = "USD"});
}

/// Least-squares slope of log(y) against log(x).
double log_log_slope(std::span<const double> x, std::span<const double> y) {
    const auto n = static_cast<double>(x.size());
    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double lx = std::log(x[i]);
        const double ly = std::log(y[i]);
        sx += lx;
        sy += ly;
        sxx += lx * lx;
        sxy += lx * ly;
    }
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

}  // namespace

TEST_CASE("CRR tree converges at first order", "[convergence][binomial][slow]") {
    // At the money with an even step count the strike sits on a node and the
    // CRR error is a clean c / N. Away from the money the coefficient
    // oscillates with the strike's position between nodes, which would make
    // a fitted slope meaningless.
    const auto type = GENERATE(OptionType::Call, OptionType::Put);
    CAPTURE(type);
    const double exact =
        OptionValuation(ud(), spec(type, 100.0), PricingMethod::Analytical).present_value();
    std::vector<double> steps;
    std::vector<double> errors;
    for (const int n : {100, 200, 400, 800, 1600}) {
        const double pv =
            OptionValuation(
                ud(), spec(type, 100.0), PricingMethod::Binomial, BinomialParams{.num_steps = n})
                .present_value();
        steps.push_back(n);
        errors.push_back(std::abs(pv - exact));
    }
    const double slope = log_log_slope(steps, errors);
    CAPTURE(errors, slope);  // measured: -1.000
    // Expected order -1. Flatter fails a sub-linear or plateauing error (a
    // wrong reference or a fixed bias); steeper means the fit is invalid.
    CHECK(slope < -0.9);
    CHECK(slope > -1.1);
}

TEST_CASE("Crank-Nicolson PDE converges at second order", "[convergence][pde][slow]") {
    // smax_mult = 4 with spot_steps = time_steps = n: the spot and the strike
    // are grid nodes at every n, so the payoff kink adds no oscillating
    // interpolation error and Rannacher-smoothed Crank-Nicolson is
    // O(h^2 + dt^2). The uniform Spot grid on [0, 400] has nodes at 80 and
    // 120; the LogSpot grid on [25, 400] is centred on ln(100) with
    // ln-spacing ln(16) / n, so 100 / sqrt(2) and 100 * sqrt(2) are nodes
    // whenever n is a multiple of 8. With the default fitted domain the
    // strike falls between nodes and the error is second order only on
    // average.
    const double root2 = std::sqrt(2.0);
    const auto [grid, type, strike] =
        GENERATE_COPY(std::tuple{PdeSpaceGrid::Spot, OptionType::Call, 100.0},
                      std::tuple{PdeSpaceGrid::Spot, OptionType::Put, 100.0},
                      std::tuple{PdeSpaceGrid::Spot, OptionType::Put, 80.0},
                      std::tuple{PdeSpaceGrid::Spot, OptionType::Call, 120.0},
                      std::tuple{PdeSpaceGrid::LogSpot, OptionType::Call, 100.0},
                      std::tuple{PdeSpaceGrid::LogSpot, OptionType::Put, 100.0},
                      std::tuple{PdeSpaceGrid::LogSpot, OptionType::Put, 100.0 / root2},
                      std::tuple{PdeSpaceGrid::LogSpot, OptionType::Call, 100.0 * root2});
    CAPTURE(grid, type, strike);
    const double exact =
        OptionValuation(ud(), spec(type, strike), PricingMethod::Analytical).present_value();
    std::vector<double> sizes;
    std::vector<double> errors;
    for (const int n : {200, 400, 800, 1600}) {
        const double pv =
            OptionValuation(
                ud(),
                spec(type, strike),
                PricingMethod::PdeFd,
                PdeParams{.smax_mult = 4.0, .spot_steps = n, .time_steps = n, .space_grid = grid})
                .present_value();
        sizes.push_back(n);
        errors.push_back(std::abs(pv - exact));
    }
    const double slope = log_log_slope(sizes, errors);
    CAPTURE(errors, slope);  // measured: -2.00 for every case on both grids
    // Expected order -2. Flatter fails a first- or mixed-order scheme (fully
    // implicit gives -1.4 to -1.7) or an error floor such as domain
    // truncation; steeper means the error crosses zero inside the range.
    CHECK(slope < -1.8);
    CHECK(slope > -2.2);
}

TEST_CASE("Monte Carlo error falls as one over root paths", "[convergence][mc][slow]") {
    // The reported standard error must scale as N^(-1/2), and the price must
    // stay within 4 standard errors of the closed form at every N. Plain
    // sampling (no antithetic pairs or moment matching) keeps the estimator
    // an average of independent draws.
    const double exact =
        OptionValuation(ud(), spec(OptionType::Call, 100.0), PricingMethod::Analytical)
            .present_value();
    std::vector<double> paths;
    std::vector<double> std_errors;
    for (const int n : {10'000, 40'000, 160'000, 640'000}) {
        const GbmProcess process(md(),
                                 {.initial_value = 100.0, .volatility = 0.20},
                                 {.paths = n,
                                  .end_date = kMaturity,
                                  .num_steps = 4,
                                  .antithetic = false,
                                  .moment_matching = false});
        const OptionValuation mc(process, spec(OptionType::Call, 100.0), PricingMethod::MonteCarlo);
        const auto std_error = mc.std_error();
        REQUIRE(std_error.has_value());
        const double se = std_error.value_or(0.0);
        CAPTURE(n, se, mc.present_value());
        CHECK(std::abs(mc.present_value() - exact) < 4.0 * se);
        paths.push_back(n);
        std_errors.push_back(se);
    }
    const double slope = log_log_slope(paths, std_errors);
    CAPTURE(std_errors, slope);
    // Expected order -1/2 (central limit theorem). Flatter fails dependent
    // samples (e.g. reused random streams) or a wrong SE formula.
    CHECK(slope < -0.45);
    CHECK(slope > -0.55);
}
