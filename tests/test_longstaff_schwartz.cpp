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

#include <array>
#include <cmath>

// External American benchmark: Longstaff & Schwartz (2001), "Valuing American
// Options by Simulation: A Simple Least-Squares Approach", Review of Financial
// Studies 14(1), Table 1 (p. 127). American put, K = 40, r = 0.06, exercisable
// 50 times per year. The paper's finite-difference column is therefore a
// Bermudan price (50 exercise dates a year), not continuous exercise.

using namespace dp;
using dp::test::kPricingDate;

namespace {

constexpr double kStrike = 40.0;
constexpr double kRate = 0.06;
/// Half a unit in the table's third decimal place.
constexpr double kRounding = 0.0005;

struct Row {
    double spot, vol;
    int years;
    double fd_bermudan, european, lsm, lsm_se;
};

constexpr std::array<Row, 20> kTable1{{
    {36, .20, 1, 4.478, 3.844, 4.472, .010}, {36, .20, 2, 4.840, 3.763, 4.821, .012},
    {36, .40, 1, 7.101, 6.711, 7.091, .020}, {36, .40, 2, 8.508, 7.700, 8.488, .024},
    {38, .20, 1, 3.250, 2.852, 3.244, .009}, {38, .20, 2, 3.745, 2.991, 3.735, .011},
    {38, .40, 1, 6.148, 5.834, 6.139, .019}, {38, .40, 2, 7.670, 6.979, 7.669, .022},
    {40, .20, 1, 2.314, 2.066, 2.313, .009}, {40, .20, 2, 2.885, 2.356, 2.879, .010},
    {40, .40, 1, 5.312, 5.060, 5.308, .018}, {40, .40, 2, 6.920, 6.326, 6.921, .022},
    {42, .20, 1, 1.617, 1.465, 1.617, .007}, {42, .20, 2, 2.212, 1.841, 2.206, .010},
    {42, .40, 1, 4.582, 4.379, 4.588, .017}, {42, .40, 2, 6.248, 5.736, 6.243, .021},
    {44, .20, 1, 1.110, 1.017, 1.118, .007}, {44, .20, 2, 1.690, 1.429, 1.675, .009},
    {44, .40, 1, 3.948, 3.783, 3.957, .017}, {44, .40, 2, 5.647, 5.202, 5.622, .021},
}};

DateTime maturity(const Row& row) {
    return make_datetime(2025 + row.years, 1, 1);  // T = 1 or 2 years (Act/365F)
}

MarketData market() {
    return {kPricingDate, DiscountCurve::flat(kRate), "USD"};
}

VanillaSpec put(const Row& row, ExerciseType exercise) {
    return VanillaSpec({.option_type = OptionType::Put,
                        .exercise_type = exercise,
                        .strike = kStrike,
                        .maturity = maturity(row),
                        .currency = "USD"});
}

}  // namespace

TEST_CASE("Longstaff-Schwartz Table 1: closed-form European column", "[benchmark][bsm]") {
    for (const auto& row : kTable1) {
        CAPTURE(row.spot, row.vol, row.years);
        const UnderlyingData ud(row.spot, row.vol, market());
        const double european =
            OptionValuation(ud, put(row, ExerciseType::European), PricingMethod::Analytical)
                .present_value();
        CHECK(std::abs(european - row.european) <= kRounding + 1e-9);
    }
}

TEST_CASE("Longstaff-Schwartz Table 1: tree and PDE American bound the Bermudan column",
          "[benchmark][binomial][pde][slow]") {
    // The tree and PDE exercise continuously, so they must not price below the
    // paper's 50-dates-a-year Bermudan value (allowing for its rounding and
    // our discretisation error). The excess is the value of exercising between
    // those dates: measured up to 0.009, and within the table's rounding for
    // S = 44, sigma = 0.4, T = 2. The 0.015 cap catches overpricing. The PDE
    // uses the log-spot grid: the uniform grid's capped domain sits only ~2.3
    // standard deviations out at sigma = 0.4, T = 2.
    constexpr double kDiscretisation = 0.0005;
    constexpr double kMaxExcess = 0.015;
    for (const auto& row : kTable1) {
        CAPTURE(row.spot, row.vol, row.years);
        const UnderlyingData ud(row.spot, row.vol, market());
        const double tree = OptionValuation(ud,
                                            put(row, ExerciseType::American),
                                            PricingMethod::Binomial,
                                            BinomialParams{.num_steps = 5000})
                                .present_value();
        const double pde = OptionValuation(ud,
                                           put(row, ExerciseType::American),
                                           PricingMethod::PdeFd,
                                           PdeParams{.spot_steps = 800,
                                                     .time_steps = 800,
                                                     .space_grid = PdeSpaceGrid::LogSpot})
                               .present_value();
        for (const double american : {tree, pde}) {
            CAPTURE(american, row.fd_bermudan);
            CHECK(american > row.fd_bermudan - kRounding - kDiscretisation);
            CHECK(american < row.fd_bermudan + kMaxExcess);
        }
    }
}

TEST_CASE("Longstaff-Schwartz Table 1: LSM matches the Bermudan column", "[benchmark][mc][slow]") {
    // Same setting as the paper: 100,000 paths (50,000 plus antithetic) and 50
    // exercise dates a year, so LSM prices exactly the Bermudan option in the
    // finite-difference column, which has no sampling error. The regression
    // differs: the paper uses a constant plus three exponentially weighted
    // Laguerre polynomials of S; we use Laguerre polynomials of S/K with a
    // small ridge penalty. For a one-factor put both bases give nearly the
    // same exercise policy, and the 0.005 allowance covers the difference.
    // Measured: |z| <= 2.0 over the 20 rows, mean difference -0.0004.
    //
    // Each row alone allows 2-4 cents. Each row gets its own seed, so the 20
    // sampling errors are independent and the mean difference has standard
    // error sqrt(sum se_i^2) / 20 (~0.002): the aggregate check fails a
    // systematic bias of about a cent. (With one shared seed, rows with the
    // same T would reuse the same shocks and their errors would correlate.)
    constexpr double kBasisAllowance = 0.005;
    double sum_diff = 0.0;
    double sum_var = 0.0;
    unsigned long long seed = 1;
    for (const auto& row : kTable1) {
        CAPTURE(row.spot, row.vol, row.years, seed);
        const GbmProcess process(market(),
                                 {.initial_value = row.spot, .volatility = row.vol},
                                 {.paths = 100'000,
                                  .end_date = maturity(row),
                                  .num_steps = 50 * row.years,
                                  .antithetic = true,
                                  .moment_matching = false});
        const OptionValuation lsm(process,
                                  put(row, ExerciseType::American),
                                  PricingMethod::MonteCarlo,
                                  MonteCarloParams{.random_seed = seed++});
        const auto std_error = lsm.std_error();
        REQUIRE(std_error.has_value());
        CAPTURE(lsm.present_value(), row.fd_bermudan, std_error);
        const double se = std_error.value_or(0.0);
        const double diff = lsm.present_value() - row.fd_bermudan;
        CHECK(std::abs(diff) < 3.0 * se + kBasisAllowance + kRounding);
        sum_diff += diff;
        sum_var += se * se;
    }
    const auto n = static_cast<double>(kTable1.size());
    const double mean_diff = sum_diff / n;
    const double se_of_mean = std::sqrt(sum_var) / n;
    CAPTURE(mean_diff, se_of_mean);
    CHECK(std::abs(mean_diff) < 3.0 * se_of_mean + kBasisAllowance);
}
