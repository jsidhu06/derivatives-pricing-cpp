#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/exceptions.hpp"
#include "dp/forward.hpp"
#include "dp/market_data.hpp"
#include "dp/params.hpp"
#include "dp/simulation.hpp"
#include "dp/underlying.hpp"
#include "dp/valuation.hpp"

#include "test_helpers.hpp"

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <optional>
#include <string_view>

using namespace dp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using dp::test::close_to;
using dp::test::kMaturity;
using dp::test::kPricingDate;
using dp::test::message_contains;

namespace {

MarketData md(double rate = 0.05) {
    return {kPricingDate, DiscountCurve::flat(rate), "USD"};
}

GbmProcess gbm(int paths = 100'000,
               double spot = 100.0,
               double vol = 0.20,
               GbmParams extra = {.initial_value = 0, .volatility = 0}) {
    GbmParams params = std::move(extra);
    params.initial_value = spot;
    params.volatility = vol;
    return {md(), std::move(params), {.paths = paths, .end_date = kMaturity, .num_steps = 26}};
}

VanillaSpec spec(OptionType type,
                 ExerciseType exercise = ExerciseType::European,
                 double strike = 100.0) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = exercise,
                        .strike = strike,
                        .maturity = kMaturity,
                        .currency = "USD"});
}

constexpr GreekOptions kPathwise{.method = GreekCalculationMethod::Pathwise};
constexpr GreekOptions kLikelihoodRatio{.method = GreekCalculationMethod::LikelihoodRatio};

}  // namespace

// ---------------------------------------------------------------------------
// Simulation layer
// ---------------------------------------------------------------------------

TEST_CASE("SimulationConfig validation", "[mc][simulation]") {
    CHECK_THROWS_MATCHES(SimulationConfig{.paths = 0}.validate(),
                         ValidationError,
                         message_contains("paths must be a positive integer"));
    CHECK_THROWS_MATCHES(SimulationConfig{.paths = 10}.validate(),
                         ValidationError,
                         message_contains("end_date must be provided"));
    CHECK_THROWS_MATCHES((SimulationConfig{.paths = 10, .end_date = kMaturity}.validate()),
                         ValidationError,
                         message_contains("num_steps must be provided"));
    CHECK_THROWS_MATCHES((SimulationConfig{.paths = 10,
                                           .end_date = kMaturity,
                                           .num_steps = 12,
                                           .time_grid = std::vector{kPricingDate, kMaturity}}
                              .validate()),
                         ValidationError,
                         message_contains("end_date and num_steps must be omitted"));
    CHECK_NOTHROW((SimulationConfig{.paths = 10, .time_grid = std::vector{kPricingDate, kMaturity}}
                       .validate()));
}

TEST_CASE("simulation grid starts at pricing date and snaps ex-div dates", "[mc][simulation]") {
    const auto ex_date = kPricingDate + std::chrono::days{100};
    const GbmProcess process(
        md(),
        {.initial_value = 100.0, .volatility = 0.2, .discrete_dividends = {{ex_date, 1.0}}},
        {.paths = 10, .end_date = kMaturity, .num_steps = 12});
    const auto dates = process.build_dates();
    CHECK(dates.front() == kPricingDate);
    CHECK(dates.back() == kMaturity);
    CHECK(std::find(dates.begin(), dates.end(), ex_date) != dates.end());
    CHECK(dates.size() >= 14);
}

TEST_CASE("simulation is reproducible per seed", "[mc][simulation]") {
    const auto process = gbm(1000);
    const auto a = process.simulate(7);
    const auto b = process.simulate(7);
    const auto c = process.simulate(8);
    CHECK(a.paths == b.paths);
    CHECK(a.paths != c.paths);
    // Antithetic layout: the second half mirrors the first half's shocks.
    CHECK(a.normals(0, 0) == -a.normals(500, 0));
}

TEST_CASE("terminal spots are risk-neutral on average", "[mc][simulation]") {
    const auto process = gbm(200'000);
    const auto result = process.simulate(42);
    const double mean_terminal = result.paths.col(result.paths.cols() - 1).mean();
    const auto& curve = process.discount_curve();
    const double fwd = forward_price({.spot = 100.0,
                                      .pricing_date = kPricingDate,
                                      .maturity = kMaturity,
                                      .discount_curve = &curve});
    CHECK_THAT(mean_terminal, WithinRel(fwd, 5e-3));
}

TEST_CASE("odd path count downgrades antithetic with a warning", "[mc][simulation]") {
    const dp::test::WarningCapture capture;
    const GbmProcess process(md(),
                             {.initial_value = 100.0, .volatility = 0.2},
                             {.paths = 101, .end_date = kMaturity, .num_steps = 4});
    (void)process.simulate(1);
    CHECK(capture.any_contains("antithetic"));
}

// ---------------------------------------------------------------------------
// European pricing and Greeks vs BSM (statistical agreement)
// ---------------------------------------------------------------------------

TEST_CASE("MC European agrees with BSM", "[mc]") {
    const auto type = GENERATE(OptionType::Call, OptionType::Put);
    const UnderlyingData ud(100.0, 0.20, md());
    const OptionValuation bsm(ud, spec(type), PricingMethod::Bsm);
    const OptionValuation mc(gbm(200'000), spec(type), PricingMethod::MonteCarlo);
    CHECK_THAT(mc.present_value(), WithinRel(bsm.present_value(), 1e-2));
}

TEST_CASE("pathwise and LR Greeks agree with BSM analytical", "[mc][greeks]") {
    const auto type = GENERATE(OptionType::Call, OptionType::Put);
    const UnderlyingData ud(100.0, 0.20, md());
    const OptionValuation bsm(ud, spec(type), PricingMethod::Bsm);
    const OptionValuation mc(gbm(200'000), spec(type), PricingMethod::MonteCarlo);

    CHECK_THAT(mc.delta(kPathwise), WithinRel(bsm.delta(), 2e-2));
    CHECK_THAT(mc.vega(kPathwise), WithinRel(bsm.vega(), 2e-2));
    CHECK_THAT(mc.theta(kPathwise), WithinRel(bsm.theta(), 2e-2));
    CHECK_THAT(mc.rho(kPathwise), WithinRel(bsm.rho(), 2e-2));
    CHECK_THAT(mc.gamma(kPathwise), WithinRel(bsm.gamma(), 5e-2));

    CHECK_THAT(mc.delta(kLikelihoodRatio), WithinRel(bsm.delta(), 2e-2));
    CHECK_THAT(mc.vega(kLikelihoodRatio), WithinRel(bsm.vega(), 5e-2));
    CHECK_THAT(mc.theta(kLikelihoodRatio), WithinRel(bsm.theta(), 5e-2));
    CHECK_THAT(mc.rho(kLikelihoodRatio), WithinRel(bsm.rho(), 2e-2));
}

TEST_CASE("pathwise is the auto-selected Greek method for eligible MC", "[mc][greeks]") {
    const OptionValuation mc(gbm(20'000), spec(OptionType::Call), PricingMethod::MonteCarlo);
    CHECK(mc.delta() == mc.delta(kPathwise));
    CHECK(mc.vega() == mc.vega(kPathwise));
}

TEST_CASE("numerical MC Greeks share common random numbers", "[mc][greeks]") {
    // The fixed default seed makes bump-and-revalue smooth; a numerical delta
    // should sit close to the pathwise one.
    const OptionValuation mc(gbm(100'000), spec(OptionType::Call), PricingMethod::MonteCarlo);
    const double numerical = mc.delta({.method = GreekCalculationMethod::Numerical});
    CHECK_THAT(numerical, WithinRel(mc.delta(kPathwise), 2e-2));
}

// ---------------------------------------------------------------------------
// American (Longstaff-Schwartz)
// ---------------------------------------------------------------------------

TEST_CASE("LSM American put sits between European and the tree price", "[mc][american]") {
    const auto process = gbm(200'000);
    const double european =
        OptionValuation(process, spec(OptionType::Put), PricingMethod::MonteCarlo).present_value();
    const double american =
        OptionValuation(
            process, spec(OptionType::Put, ExerciseType::American), PricingMethod::MonteCarlo)
            .present_value();
    const UnderlyingData ud(100.0, 0.20, md());
    const double tree = OptionValuation(ud,
                                        spec(OptionType::Put, ExerciseType::American),
                                        PricingMethod::Binomial,
                                        BinomialParams{.num_steps = 2000})
                            .present_value();
    CHECK(american > european);
    CHECK_THAT(american, WithinRel(tree, 1e-2));
}

TEST_CASE("MC control variate equals the explicit adjustment identity", "[mc][cv]") {
    const auto process = gbm(50'000, 100.0, 0.25);
    const auto amer = spec(OptionType::Put, ExerciseType::American, 110.0);
    const auto euro = spec(OptionType::Put, ExerciseType::European, 110.0);
    const MonteCarloParams base{};
    const MonteCarloParams cv{.control_variate_european = true};

    const double american_raw =
        OptionValuation(process, amer, PricingMethod::MonteCarlo, base).present_value();
    const double european_mc =
        OptionValuation(process, euro, PricingMethod::MonteCarlo, base).present_value();
    const UnderlyingData ud(100.0, 0.25, md());
    const double european_bsm = OptionValuation(ud, euro, PricingMethod::Bsm).present_value();
    const double american_cv =
        OptionValuation(process, amer, PricingMethod::MonteCarlo, cv).present_value();
    CHECK_THAT(american_cv, WithinAbs(american_raw + (european_bsm - european_mc), 1e-8));
}

// ---------------------------------------------------------------------------
// Facade pairing and Greek-method validation
// ---------------------------------------------------------------------------

TEST_CASE("pricing method and underlying type must pair up", "[mc][valuation]") {
    const UnderlyingData ud(100.0, 0.20, md());
    CHECK_THROWS_MATCHES(OptionValuation(ud, spec(OptionType::Call), PricingMethod::MonteCarlo),
                         ConfigurationError,
                         message_contains("requires underlying to be a path process"));
    CHECK_THROWS_MATCHES(OptionValuation(gbm(100), spec(OptionType::Call), PricingMethod::Bsm),
                         ConfigurationError,
                         message_contains("BSM pricing requires an UnderlyingData instance"));
    CHECK_THROWS_MATCHES(OptionValuation(gbm(100), spec(OptionType::Call), PricingMethod::Binomial),
                         ConfigurationError,
                         message_contains("BINOMIAL pricing requires an UnderlyingData instance"));
    CHECK_THROWS_MATCHES(OptionValuation(gbm(100), spec(OptionType::Call), PricingMethod::PdeFd),
                         ConfigurationError,
                         message_contains("PDE_FD pricing requires an UnderlyingData instance"));
}

TEST_CASE("pathwise/LR eligibility is enforced", "[mc][greeks]") {
    const OptionValuation american(
        gbm(1000), spec(OptionType::Put, ExerciseType::American), PricingMethod::MonteCarlo);
    CHECK_THROWS_MATCHES(
        american.delta(kPathwise), ValidationError, message_contains("vanilla European options"));

    const PayoffSpec payoff({.exercise_type = ExerciseType::European,
                             .maturity = kMaturity,
                             .payoff_fn = [](double s) { return std::max(s - 95.0, 0.0); },
                             .currency = "USD"});
    const OptionValuation custom(gbm(1000), payoff, PricingMethod::MonteCarlo);
    CHECK_THROWS_MATCHES(custom.delta(kLikelihoodRatio),
                         ValidationError,
                         message_contains("vanilla European options"));
    CHECK(std::isfinite(custom.delta({.method = GreekCalculationMethod::Numerical})));

    const OptionValuation vanilla(gbm(1000), spec(OptionType::Call), PricingMethod::MonteCarlo);
    CHECK_THROWS_MATCHES(vanilla.gamma(kLikelihoodRatio),
                         ValidationError,
                         message_contains("likelihood_ratio is not available for gamma"));

    const GbmProcess with_div(
        md(),
        {.initial_value = 100.0,
         .volatility = 0.2,
         .discrete_dividends = {{kPricingDate + std::chrono::days{100}, 1.0}}},
        {.paths = 1000, .end_date = kMaturity, .num_steps = 12});
    const OptionValuation div_ov(with_div, spec(OptionType::Call), PricingMethod::MonteCarlo);
    CHECK_THROWS_MATCHES(
        div_ov.delta(kPathwise), UnsupportedFeatureError, message_contains("discrete dividends"));
    CHECK_THROWS_MATCHES(div_ov.delta(kLikelihoodRatio),
                         UnsupportedFeatureError,
                         message_contains("discrete dividends"));
    // Auto-selection therefore falls back to NUMERICAL and still works.
    CHECK(std::isfinite(div_ov.delta()));
}

TEST_CASE("LR estimators reject zero volatility", "[mc][greeks][edge]") {
    const OptionValuation ov(
        gbm(1000, 100.0, 0.0), spec(OptionType::Call), PricingMethod::MonteCarlo);
    CHECK_THROWS_MATCHES(ov.delta(kLikelihoodRatio),
                         UnsupportedFeatureError,
                         message_contains("Likelihood-ratio delta is undefined at sigma=0"));
    CHECK_THROWS_MATCHES(ov.vega(kLikelihoodRatio),
                         UnsupportedFeatureError,
                         message_contains("Likelihood-ratio vega is undefined at sigma=0"));
    CHECK_THROWS_MATCHES(ov.theta(kLikelihoodRatio),
                         UnsupportedFeatureError,
                         message_contains("Likelihood-ratio theta is undefined at sigma=0"));
    CHECK_THROWS_MATCHES(ov.rho(kLikelihoodRatio),
                         UnsupportedFeatureError,
                         message_contains("Likelihood-ratio rho is undefined at sigma=0"));
}

TEST_CASE("discrete dividends reduce the MC call price", "[mc]") {
    const double base =
        OptionValuation(gbm(100'000), spec(OptionType::Call), PricingMethod::MonteCarlo)
            .present_value();
    const GbmProcess with_div(
        md(),
        {.initial_value = 100.0,
         .volatility = 0.2,
         .discrete_dividends = {{kPricingDate + std::chrono::days{180}, 2.0}}},
        {.paths = 100'000, .end_date = kMaturity, .num_steps = 26});
    const double with_div_pv =
        OptionValuation(with_div, spec(OptionType::Call), PricingMethod::MonteCarlo)
            .present_value();
    CHECK(with_div_pv < base);
}

// ---------------------------------------------------------------------------
// Custom payoffs
// ---------------------------------------------------------------------------

namespace {

double bull_spread(double s) {
    return std::max(s - 95.0, 0.0) - std::max(s - 115.0, 0.0);
}

PayoffSpec bull_spread_spec(ExerciseType exercise) {
    return PayoffSpec({.exercise_type = exercise,
                       .maturity = kMaturity,
                       .payoff_fn = bull_spread,
                       .currency = "USD"});
}

}  // namespace

TEST_CASE("MC prices a custom payoff like its replication", "[mc][payoff]") {
    // European bull spread = C(95) - C(115) in BSM.
    const UnderlyingData ud(100.0, 0.20, md());
    const auto call = [&](double strike) {
        return OptionValuation(
                   ud, spec(OptionType::Call, ExerciseType::European, strike), PricingMethod::Bsm)
            .present_value();
    };
    const double replication = call(95.0) - call(115.0);
    const double mc =
        OptionValuation(gbm(), bull_spread_spec(ExerciseType::European), PricingMethod::MonteCarlo)
            .present_value();
    CHECK_THAT(mc, close_to(replication, 1.5e-2, 0.0));

    SECTION("American exercise via LSM is a bounded Bermudan approximation") {
        // MC exercises only on its grid dates (a Bermudan), and LSM's
        // regression exercise rule is suboptimal, so it is a lower bound on
        // the American value. This capped payoff (exercise once spot passes
        // 115) is hard for a degree-3 basis: LSM plateaus ~2.6% below the
        // converged tree from ~100 dates on. Checked here: below the tree up
        // to noise, most of the early-exercise premium captured (~84% at 26
        // dates), and a gain from 26 to 52 dates.
        const auto lsm = [](int num_steps) {
            const GbmProcess process(
                md(),
                {.initial_value = 100.0, .volatility = 0.20},
                {.paths = 100'000, .end_date = kMaturity, .num_steps = num_steps});
            return OptionValuation(
                       process, bull_spread_spec(ExerciseType::American), PricingMethod::MonteCarlo)
                .present_value();
        };
        const double bermudan_26 = lsm(26);
        const double tree = OptionValuation(ud,
                                            bull_spread_spec(ExerciseType::American),
                                            PricingMethod::Binomial,
                                            BinomialParams{.num_steps = 2000})
                                .present_value();
        CHECK(bermudan_26 < tree + 0.02);
        CHECK((bermudan_26 - replication) / (tree - replication) > 0.75);  // measured ~84%
        CHECK(lsm(52) > bermudan_26);
    }
}

// ---------------------------------------------------------------------------
// Longstaff-Schwartz options
// ---------------------------------------------------------------------------

TEST_CASE("LSM is stable across basis degrees", "[mc][american]") {
    const auto american_put = [](int deg) {
        return OptionValuation(gbm(20'000),
                               spec(OptionType::Put, ExerciseType::American),
                               PricingMethod::MonteCarlo,
                               MonteCarloParams{.deg = deg})
            .present_value();
    };
    const double deg2 = american_put(2);
    const double deg5 = american_put(5);
    const double tree = OptionValuation(UnderlyingData(100.0, 0.20, md()),
                                        spec(OptionType::Put, ExerciseType::American),
                                        PricingMethod::Binomial,
                                        BinomialParams{.num_steps = 500})
                            .present_value();
    // Observed gaps: ~0.1% between degrees, ~0.3% to the tree.
    CHECK_THAT(deg5, close_to(deg2, 2e-2, 0.0));
    CHECK_THAT(deg2, close_to(tree, 2e-2, 0.0));
}

TEST_CASE("LSM exercises a deep in-the-money put immediately", "[mc][american]") {
    // Spot 40 against strike 100 with tiny vol and a 20% rate: holding is
    // strictly worse, so the value is the intrinsic K - S = 60 exactly.
    const GbmProcess process(md(0.20),
                             {.initial_value = 40.0, .volatility = 0.01},
                             {.paths = 20'000, .end_date = kMaturity, .num_steps = 26});
    const double pv = OptionValuation(process,
                                      spec(OptionType::Put, ExerciseType::American),
                                      PricingMethod::MonteCarlo,
                                      MonteCarloParams{.deg = 2})
                          .present_value();
    CHECK_THAT(pv, close_to(60.0, 0.0, 0.01));
}

// ---------------------------------------------------------------------------
// Greek estimators across market setups (dividends, strikes, term curves)
// ---------------------------------------------------------------------------

TEST_CASE("pathwise, likelihood-ratio and numerical Greeks agree", "[mc][greeks][slow]") {
    // Ports the Python suite's method-agreement scenarios (500k paths,
    // 12 steps), and additionally holds pathwise to BSM analytical.
    struct Scenario {
        const char* id;
        OptionType type;
        double strike;
        double vol;
        bool term_curves;  ///< else flat 5% rates
        bool dividends;    ///< flat 2% (or the term dividend curve)
    };
    const std::array<Scenario, 4> scenarios{
        {{"otm_call", OptionType::Call, 120.0, 0.20, false, true},
         {"itm_put_high_vol", OptionType::Put, 120.0, 0.25, false, true},
         {"atm_no_div", OptionType::Call, 100.0, 0.20, false, false},
         {"non_flat_both_curves", OptionType::Call, 110.0, 0.25, true, true}}};
    for (const auto& s : scenarios) {
        Eigen::ArrayXd times(3);
        times << 0.0, 0.5, 1.0;
        Eigen::ArrayXd rate_fwd(2);
        rate_fwd << 0.04, 0.06;
        Eigen::ArrayXd div_fwd(2);
        div_fwd << 0.01, 0.03;
        const DiscountCurve rates = s.term_curves
                                        ? DiscountCurve::from_forwards(times, rate_fwd)
                                        : dp::test::flat_curve(kPricingDate, kMaturity, 0.05);
        std::optional<DiscountCurve> q_curve;
        if (s.dividends) {
            q_curve = s.term_curves ? DiscountCurve::from_forwards(times, div_fwd)
                                    : dp::test::flat_curve(kPricingDate, kMaturity, 0.02);
        }
        const MarketData market(kPricingDate, rates, "USD");
        const GbmProcess process(
            market,
            {.initial_value = 100.0, .volatility = s.vol, .dividend_curve = q_curve},
            {.paths = 500'000, .end_date = kMaturity, .num_steps = 12});
        const VanillaSpec vanilla = spec(s.type, ExerciseType::European, s.strike);
        const OptionValuation mc(process, vanilla, PricingMethod::MonteCarlo);
        const OptionValuation bsm(
            UnderlyingData(100.0, s.vol, market, {}, q_curve), vanilla, PricingMethod::Bsm);

        using Greek = double (OptionValuation::*)(GreekOptions) const;
        const std::array<std::pair<const char*, Greek>, 5> greeks{
            {{"delta", &OptionValuation::delta},
             {"gamma", &OptionValuation::gamma},
             {"vega", &OptionValuation::vega},
             {"theta", &OptionValuation::theta},
             {"rho", &OptionValuation::rho}}};
        for (const auto& [name, greek] : greeks) {
            INFO(s.id << " " << name);
            const double pw = (mc.*greek)(kPathwise);
            const double num = (mc.*greek)({.method = GreekCalculationMethod::Numerical});
            CHECK_THAT(pw, close_to(num, 1e-2, 5e-3));
            CHECK_THAT(pw, close_to((bsm.*greek)({}), 1e-2, 5e-3));
            if (std::string_view(name) != "gamma") {  // no LR gamma
                CHECK_THAT(pw, close_to((mc.*greek)(kLikelihoodRatio), 1e-2, 5e-3));
            }
        }
    }
}
