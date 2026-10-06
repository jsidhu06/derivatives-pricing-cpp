// Barrier cross-engine sweeps (ported from test_barrier.py and test_pde.py):
// in/out parity beyond the base cases, the Broadie-Glasserman-Kou discrete
// correction against continuous monitoring and pathwise MC, binomial
// half-step alignment against the PDE, PDE scheme/grid/solver equivalence,
// and American barriers with discrete dividends (PDE vs MC). The sweeps are
// [slow]: minutes under the sanitizer build.

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

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>

using namespace dp;
using Catch::Matchers::WithinRel;
using dp::test::close_to;
using dp::test::flat_curve;
using dp::test::kMaturity;
using dp::test::kPricingDate;

namespace {

MarketData md(double rate = 0.05) {
    return {kPricingDate, flat_curve(kPricingDate, kMaturity, rate), "USD"};
}

/// The Python barrier suite's default market: S = 100, sigma = 20%, r = 5%, q = 0.
UnderlyingData ud(double vol = 0.20) {
    return {100.0, vol, md()};
}

BarrierSpec::Params barrier(double level,
                            BarrierDirection direction,
                            BarrierAction action,
                            OptionType type = OptionType::Call) {
    return {.option_type = type,
            .exercise_type = ExerciseType::European,
            .strike = 100.0,
            .maturity = kMaturity,
            .barrier = level,
            .direction = direction,
            .action = action,
            .currency = "USD"};
}

BarrierSpec::Params discrete(BarrierSpec::Params params, int observations) {
    params.monitoring = BarrierMonitoring::Discrete;
    params.num_observations = observations;
    return params;
}

BarrierSpec::Params with_action(BarrierSpec::Params params, BarrierAction action) {
    params.action = action;
    return params;
}

VanillaSpec vanilla(OptionType type) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = ExerciseType::European,
                        .strike = 100.0,
                        .maturity = kMaturity,
                        .currency = "USD"});
}

double bsm_pv(const UnderlyingData& u, const BarrierSpec::Params& params) {
    return OptionValuation(u, BarrierSpec(params), PricingMethod::Analytical).present_value();
}

struct Parity {
    double knock_in = 0.0;
    double knock_out = 0.0;
    double vanilla = 0.0;
};

/// Knock-in, knock-out and vanilla on one engine.
Parity parity(const UnderlyingData& u,
              const BarrierSpec::Params& params,
              PricingMethod method,
              const std::optional<ValuationParams>& engine_params = std::nullopt) {
    const auto value = [&](const OptionSpec& s) {
        return OptionValuation(u, s, method, engine_params).present_value();
    };
    return {value(BarrierSpec(with_action(params, BarrierAction::In))),
            value(BarrierSpec(with_action(params, BarrierAction::Out))),
            value(vanilla(params.option_type))};
}

Eigen::ArrayXd array(std::initializer_list<double> values) {
    Eigen::ArrayXd out(static_cast<Eigen::Index>(values.size()));
    std::copy(values.begin(), values.end(), out.begin());
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// In/out parity: knock-in + knock-out == vanilla
// ---------------------------------------------------------------------------

TEST_CASE("analytical in/out parity holds beyond the base cases", "[barrier][parity]") {
    SECTION("barriers already breached at inception") {
        // Down at 110 / up at 90 with spot 100: the knock-in is the vanilla
        // and the knock-out is dead, so parity holds through the
        // short-circuits rather than the closed form.
        const auto type = GENERATE(OptionType::Call, OptionType::Put);
        const auto direction = GENERATE(BarrierDirection::Down, BarrierDirection::Up);
        const double level = direction == BarrierDirection::Down ? 110.0 : 90.0;
        CAPTURE(to_string(type), level);
        const auto p = parity(
            ud(), barrier(level, direction, BarrierAction::In, type), PricingMethod::Analytical);
        CHECK_THAT(p.knock_in + p.knock_out, close_to(p.vanilla, 1e-10, 0.0));
    }
    SECTION("across volatilities") {
        const double vol = GENERATE(0.10, 0.30, 0.50);
        CAPTURE(vol);
        const auto p = parity(ud(vol),
                              barrier(120.0, BarrierDirection::Up, BarrierAction::In),
                              PricingMethod::Analytical);
        CHECK_THAT(p.knock_in + p.knock_out, close_to(p.vanilla, 1e-10, 0.0));
    }
    SECTION("under the BGK discrete correction") {
        const auto p = parity(ud(),
                              discrete(barrier(120.0, BarrierDirection::Up, BarrierAction::In), 50),
                              PricingMethod::Analytical);
        CHECK_THAT(p.knock_in + p.knock_out, close_to(p.vanilla, 1e-10, 0.0));
    }
}

TEST_CASE("in/out parity holds on the tree and PDE", "[barrier][parity][cross_engine]") {
    struct Case {
        OptionType type = OptionType::Call;
        BarrierDirection direction = BarrierDirection::Up;
        double level = 0.0;
    };
    const auto c = GENERATE(Case{OptionType::Call, BarrierDirection::Down, 80.0},
                            Case{OptionType::Call, BarrierDirection::Up, 120.0},
                            Case{OptionType::Put, BarrierDirection::Down, 80.0},
                            Case{OptionType::Put, BarrierDirection::Up, 120.0});
    CAPTURE(to_string(c.type), c.level);
    const auto params = barrier(c.level, c.direction, BarrierAction::In, c.type);

    SECTION("PDE (log-spot grid)") {
        const auto p = parity(
            ud(),
            params,
            PricingMethod::PdeFd,
            PdeParams{.spot_steps = 400, .time_steps = 400, .space_grid = PdeSpaceGrid::LogSpot});
        // The knock-in is vanilla - knock-out on the barrier's grid, while the
        // standalone vanilla uses its own auto (sigma*sqrt(T)) domain, so the
        // residual is the difference of two discretisations: measured <= 1.4e-4
        // (Python allows 2e-3).
        CHECK_THAT(p.knock_in + p.knock_out, close_to(p.vanilla, 5e-4, 1e-4));
    }
    SECTION("binomial tree") {
        const auto p =
            parity(ud(), params, PricingMethod::Binomial, BinomialParams{.num_steps = 600});
        // Boyle-Lau alignment re-grids the barrier trees away from the
        // vanilla's 600 steps: measured up to 1.1e-3 on down barriers,
        // ~3e-5 on up barriers (Python allows 5e-3).
        CHECK_THAT(p.knock_in + p.knock_out, close_to(p.vanilla, 2.5e-3, 1e-4));
    }
}

// ---------------------------------------------------------------------------
// Broadie-Glasserman-Kou discrete monitoring correction
// ---------------------------------------------------------------------------

TEST_CASE("BGK discrete monitoring brackets the continuous price", "[barrier][discrete]") {
    SECTION("very frequent monitoring converges to continuous") {
        const auto up_out = barrier(120.0, BarrierDirection::Up, BarrierAction::Out);
        CHECK_THAT(bsm_pv(ud(), discrete(up_out, 50'000)), WithinRel(bsm_pv(ud(), up_out), 2e-2));
    }
    SECTION("monthly monitoring lifts a down-and-out put") {
        // The shift moves the effective barrier away from spot in either
        // direction; the up-and-out case is in test_valuation_barrier.cpp.
        const auto down_out =
            barrier(80.0, BarrierDirection::Down, BarrierAction::Out, OptionType::Put);
        CHECK(bsm_pv(ud(), discrete(down_out, 12)) > bsm_pv(ud(), down_out));
    }
}

TEST_CASE("BGK closed form agrees with pathwise Monte Carlo", "[barrier][discrete][mc][slow]") {
    // The two are structurally unrelated: BSM shifts the barrier by
    // beta * sigma * sqrt(dt); MC checks it on simulated paths at each
    // monitoring date. A broken correction would miss by > 10%.
    struct Case {
        OptionType type = OptionType::Call;
        BarrierDirection direction = BarrierDirection::Up;
        BarrierAction action = BarrierAction::Out;
        double level = 0.0;
        int observations = 0;
    };
    const auto c =
        GENERATE(Case{OptionType::Call, BarrierDirection::Up, BarrierAction::Out, 120.0, 60},
                 Case{OptionType::Put, BarrierDirection::Down, BarrierAction::Out, 80.0, 60},
                 Case{OptionType::Call, BarrierDirection::Up, BarrierAction::In, 120.0, 50});
    CAPTURE(to_string(c.type), c.level, c.observations);
    const auto params = discrete(barrier(c.level, c.direction, c.action, c.type), c.observations);

    const GbmProcess process(
        md(),
        {.initial_value = 100.0, .volatility = 0.20},
        {.paths = 50'000, .end_date = kMaturity, .num_steps = std::max(200, 10 * c.observations)});
    const double mc =
        OptionValuation(process, BarrierSpec(params), PricingMethod::MonteCarlo).present_value();
    // BGK is O(dt^1.5) accurate, a few percent at monthly monitoring, with
    // MC noise on top: measured <= 2.6% over five seeds.
    CHECK_THAT(bsm_pv(ud(), params), close_to(mc, 5e-2, 5e-3));
}

// ---------------------------------------------------------------------------
// Binomial half-step alignment on hard discrete cases
// ---------------------------------------------------------------------------

TEST_CASE("binomial half-step alignment tracks the PDE on discrete barriers",
          "[barrier][binomial][discrete][slow]") {
    // Discrete monitoring places H midway between CRR layers. Without it
    // (Python's monkeypatched no-align baseline, 5000 steps), the regular
    // knock-ins and the reverse knock-out below missed the PDE by 3-4.5%.
    // Measured with alignment: DOP 2.3%, the others <= 0.2%.
    const auto maturity = kPricingDate + std::chrono::days{73};  // T = 0.2
    const double ttm = year_fraction(kPricingDate, maturity, DayCountConvention::Act365F);
    const UnderlyingData u(100.0,
                           0.25,
                           MarketData(kPricingDate, DiscountCurve::flat(0.05, ttm), "USD"),
                           {},
                           DiscountCurve::flat(0.02, ttm));
    struct Case {
        const char* id = nullptr;
        double level = 0.0;
        BarrierDirection direction = BarrierDirection::Up;
        BarrierAction action = BarrierAction::Out;
        OptionType type = OptionType::Call;
        double rtol = 0.0;
    };
    const auto c = GENERATE(Case{"DIC tighter (hard)",
                                 95.0,
                                 BarrierDirection::Down,
                                 BarrierAction::In,
                                 OptionType::Call,
                                 2.5e-2},
                            Case{"DOP tighter (hard)",
                                 95.0,
                                 BarrierDirection::Down,
                                 BarrierAction::Out,
                                 OptionType::Put,
                                 2.5e-2},
                            Case{"UIP wider (hard)",
                                 110.0,
                                 BarrierDirection::Up,
                                 BarrierAction::In,
                                 OptionType::Put,
                                 2.5e-2},
                            Case{"DOC wider (easy)",
                                 90.0,
                                 BarrierDirection::Down,
                                 BarrierAction::Out,
                                 OptionType::Call,
                                 1e-2});
    CAPTURE(c.id);
    auto params = discrete(barrier(c.level, c.direction, c.action, c.type), 50);
    params.maturity = maturity;

    const double pde =
        OptionValuation(u, BarrierSpec(params), PricingMethod::PdeFd).present_value();
    const double tree =
        OptionValuation(
            u, BarrierSpec(params), PricingMethod::Binomial, BinomialParams{.num_steps = 3000})
            .present_value();
    CHECK_THAT(tree, WithinRel(pde, c.rtol));
}

// ---------------------------------------------------------------------------
// PDE scheme / grid / solver equivalence
// ---------------------------------------------------------------------------

namespace {

struct PdeScenario {
    const char* id = nullptr;
    double spot = 100.0;
    double strike = 100.0;
    double vol = 0.20;
    double rate = 0.05;
    OptionType type = OptionType::Call;
    ExerciseType exercise = ExerciseType::European;
    BarrierDirection direction = BarrierDirection::Up;
    BarrierAction action = BarrierAction::Out;
    double level = 0.0;
    BarrierMonitoring monitoring = BarrierMonitoring::Continuous;
    double rebate = 0.0;
    RebateTiming rebate_timing = RebateTiming::AtHit;
    std::optional<int> observations = std::nullopt;
    std::optional<DiscountCurve> r_curve = std::nullopt;
    std::optional<DiscountCurve> q_curve = std::nullopt;
};

double scenario_pv(const PdeScenario& s, const PdeParams& params) {
    const MarketData market(
        kPricingDate, s.r_curve.value_or(flat_curve(kPricingDate, kMaturity, s.rate)), "USD");
    const UnderlyingData u(s.spot, s.vol, market, {}, s.q_curve);
    const BarrierSpec spec({.option_type = s.type,
                            .exercise_type = s.exercise,
                            .strike = s.strike,
                            .maturity = kMaturity,
                            .barrier = s.level,
                            .direction = s.direction,
                            .action = s.action,
                            .monitoring = s.monitoring,
                            .rebate = s.rebate,
                            .rebate_timing = s.rebate_timing,
                            .currency = "USD",
                            .num_observations = s.observations});
    return OptionValuation(u, spec, PricingMethod::PdeFd, params).present_value();
}

DiscountCurve q_flat(double rate) {
    return flat_curve(kPricingDate, kMaturity, rate);
}

}  // namespace

TEST_CASE("European PDE barriers agree across schemes and grids", "[barrier][pde][slow]") {
    const auto s =
        GENERATE(PdeScenario{.id = "down-out call, continuous",
                             .spot = 104.0,
                             .vol = 0.24,
                             .rate = 0.045,
                             .direction = BarrierDirection::Down,
                             .level = 86.0,
                             .q_curve = q_flat(0.012)},
                 PdeScenario{.id = "up-in put, at-expiry rebate, continuous",
                             .spot = 96.0,
                             .strike = 101.0,
                             .vol = 0.28,
                             .rate = 0.035,
                             .type = OptionType::Put,
                             .action = BarrierAction::In,
                             .level = 114.0,
                             .rebate = 1.75,
                             .rebate_timing = RebateTiming::AtExpiry},
                 PdeScenario{.id = "down-in call, discrete",
                             .spot = 101.0,
                             .strike = 97.0,
                             .vol = 0.22,
                             .rate = 0.04,
                             .direction = BarrierDirection::Down,
                             .action = BarrierAction::In,
                             .level = 89.0,
                             .monitoring = BarrierMonitoring::Discrete,
                             .rebate = 1.25,
                             .rebate_timing = RebateTiming::AtExpiry,
                             .observations = 12,
                             .q_curve = q_flat(0.02)},
                 PdeScenario{.id = "up-out put, discrete, rebate",
                             .spot = 93.0,
                             .strike = 99.0,
                             .vol = 0.26,
                             .rate = 0.03,
                             .type = OptionType::Put,
                             .level = 117.0,
                             .monitoring = BarrierMonitoring::Discrete,
                             .rebate = 2.0,
                             .observations = 10,
                             .q_curve = q_flat(0.005)},
                 PdeScenario{.id = "up-out call, non-flat curves",
                             .spot = 108.0,
                             .strike = 102.0,
                             .vol = 0.21,
                             .level = 124.0,
                             .r_curve = DiscountCurve::from_forwards(array({0.0, 0.2, 0.6, 1.0}),
                                                                     array({0.035, 0.055, 0.04})),
                             .q_curve = DiscountCurve::from_forwards(array({0.0, 0.4, 0.8, 1.0}),
                                                                     array({0.01, 0.018, 0.008}))});
    CAPTURE(s.id);
    const double baseline =
        scenario_pv(s, {.spot_steps = 800, .time_steps = 800, .space_grid = PdeSpaceGrid::LogSpot});
    for (const auto method : {PdeMethod::Implicit, PdeMethod::CrankNicolson}) {
        for (const auto grid : {PdeSpaceGrid::Spot, PdeSpaceGrid::LogSpot}) {
            CAPTURE(to_string(method), to_string(grid));
            const double pv = scenario_pv(
                s, {.spot_steps = 800, .time_steps = 800, .method = method, .space_grid = grid});
            CHECK_THAT(pv, WithinRel(baseline, 5e-3));  // Python: 1.5%; measured <= 0.2%
        }
    }
}

TEST_CASE("American PDE barriers agree across schemes, grids and solvers",
          "[barrier][pde][american][slow]") {
    // Explicit schemes are left out: a free boundary plus a barrier boundary
    // is materially less robust on them (as in Python).
    const auto s =
        GENERATE(PdeScenario{.id = "down-out put, continuous",
                             .spot = 94.0,
                             .vol = 0.23,
                             .rate = 0.042,
                             .type = OptionType::Put,
                             .exercise = ExerciseType::American,
                             .direction = BarrierDirection::Down,
                             .level = 82.0},
                 PdeScenario{.id = "up-out call, rebate, continuous",
                             .spot = 101.0,
                             .strike = 99.0,
                             .vol = 0.27,
                             .rate = 0.038,
                             .exercise = ExerciseType::American,
                             .level = 119.0,
                             .rebate = 3.0,
                             .q_curve = q_flat(0.015)},
                 PdeScenario{.id = "down-in call, continuous",
                             .spot = 103.0,
                             .strike = 98.0,
                             .vol = 0.25,
                             .rate = 0.04,
                             .exercise = ExerciseType::American,
                             .direction = BarrierDirection::Down,
                             .action = BarrierAction::In,
                             .level = 88.0,
                             .rebate = 1.5,
                             .rebate_timing = RebateTiming::AtExpiry,
                             .q_curve = q_flat(0.01)},
                 PdeScenario{.id = "up-in put, continuous",
                             .spot = 97.0,
                             .vol = 0.29,
                             .rate = 0.033,
                             .type = OptionType::Put,
                             .exercise = ExerciseType::American,
                             .action = BarrierAction::In,
                             .level = 113.0},
                 PdeScenario{.id = "down-in put, discrete, non-flat curves",
                             .spot = 98.0,
                             .strike = 102.0,
                             .vol = 0.24,
                             .type = OptionType::Put,
                             .exercise = ExerciseType::American,
                             .direction = BarrierDirection::Down,
                             .action = BarrierAction::In,
                             .level = 87.0,
                             .monitoring = BarrierMonitoring::Discrete,
                             .rebate = 1.25,
                             .rebate_timing = RebateTiming::AtExpiry,
                             .observations = 9,
                             .r_curve = DiscountCurve::from_forwards(array({0.0, 0.3, 0.7, 1.0}),
                                                                     array({0.03, 0.05, 0.042})),
                             .q_curve = DiscountCurve::from_forwards(
                                 array({0.0, 0.25, 0.75, 1.0}), array({0.008, 0.015, 0.005}))});
    CAPTURE(s.id);
    for (const auto solver : {PdeEarlyExercise::Intrinsic, PdeEarlyExercise::GaussSeidel}) {
        const double baseline = scenario_pv(s,
                                            {.spot_steps = 600,
                                             .time_steps = 600,
                                             .space_grid = PdeSpaceGrid::LogSpot,
                                             .american_solver = solver});
        for (const auto method : {PdeMethod::Implicit, PdeMethod::CrankNicolson}) {
            for (const auto grid : {PdeSpaceGrid::Spot, PdeSpaceGrid::LogSpot}) {
                CAPTURE(to_string(solver), to_string(method), to_string(grid));
                const double pv = scenario_pv(s,
                                              {.spot_steps = 600,
                                               .time_steps = 600,
                                               .method = method,
                                               .space_grid = grid,
                                               .american_solver = solver});
                CHECK_THAT(pv, WithinRel(baseline, 3e-2));  // Python: 5%; measured <= 2.4%
            }
        }
    }
}

// ---------------------------------------------------------------------------
// American barriers with discrete dividends: PDE vs MC
// ---------------------------------------------------------------------------

TEST_CASE("American barriers with discrete dividends agree on the PDE and MC",
          "[barrier][american][dividends][mc][slow]") {
    // The tree rejects barriers with discrete dividends (its escrowed
    // adjustment smears the ex-date jump), so the PDE (ex-date grid shift)
    // and MC (pathwise jump) are the only two engines that price this.
    const MarketData market(kPricingDate, DiscountCurve::flat(0.05, 2.0), "USD");
    const DividendSchedule dividends{{kPricingDate + std::chrono::days{90}, 2.0},
                                     {kPricingDate + std::chrono::days{270}, 2.0}};
    struct Case {
        OptionType type = OptionType::Call;
        BarrierDirection direction = BarrierDirection::Up;
        BarrierAction action = BarrierAction::Out;
        double level = 0.0;
        RebateTiming timing = RebateTiming::AtHit;
    };
    const auto c = GENERATE(
        Case{OptionType::Call,
             BarrierDirection::Down,
             BarrierAction::Out,
             85.0,
             RebateTiming::AtHit},
        Case{OptionType::Put, BarrierDirection::Up, BarrierAction::Out, 120.0, RebateTiming::AtHit},
        Case{OptionType::Call,
             BarrierDirection::Down,
             BarrierAction::In,
             85.0,
             RebateTiming::AtExpiry},
        Case{OptionType::Put,
             BarrierDirection::Up,
             BarrierAction::In,
             120.0,
             RebateTiming::AtExpiry});
    CAPTURE(to_string(c.type), c.level);
    auto params = barrier(c.level, c.direction, c.action, c.type);
    params.exercise_type = ExerciseType::American;
    params.rebate_timing = c.timing;
    const BarrierSpec spec(params);

    const double pde =
        OptionValuation(UnderlyingData(100.0, 0.20, market, dividends), spec, PricingMethod::PdeFd)
            .present_value();
    const GbmProcess process(
        market,
        {.initial_value = 100.0, .volatility = 0.20, .discrete_dividends = dividends},
        {.paths = 100'000, .end_date = kMaturity, .num_steps = 200});
    const dp::test::WarningCapture capture;  // reverse knock-outs warn about LSM bias
    const double mc = OptionValuation(process, spec, PricingMethod::MonteCarlo).present_value();
    // Measured over four seeds: knock-outs <= 0.35%; the small knock-ins
    // (PV 0.26-0.44) up to 3% relative, inside the absolute floor.
    CHECK_THAT(mc, close_to(pde, 2.5e-2, 1e-2));
}

// ---------------------------------------------------------------------------
// Regression: knock-out at-hit rebate on a maturity hit
// ---------------------------------------------------------------------------

TEST_CASE("discrete knock-outs pay the rebate when first hit at maturity",
          "[barrier][rebate][discrete][cross_engine]") {
    // Bug (shared with the Python reference until fixed): the PDE zeroed
    // the knocked-out nodes at a maturity observation instead of paying the
    // rebate, and American LSM dropped it for paths first hit on the final
    // step. The PDE sat ~3% below the tree and MC, and American MC below
    // European. Fixed in both engines; the tree and European MC were right.
    struct Case {
        OptionType type = OptionType::Call;
        BarrierDirection direction = BarrierDirection::Down;
        double level = 0.0;
    };
    const auto c = GENERATE(Case{OptionType::Call, BarrierDirection::Down, 90.0},
                            Case{OptionType::Put, BarrierDirection::Up, 115.0});
    const int observations = GENERATE(4, 12);
    CAPTURE(to_string(c.type), observations);
    const UnderlyingData u(100.0, 0.30, md());
    auto params = discrete(barrier(c.level, c.direction, BarrierAction::Out, c.type), observations);
    params.rebate = 10.0;

    const GbmProcess process(md(),
                             {.initial_value = 100.0, .volatility = 0.30},
                             {.paths = 50'000, .end_date = kMaturity, .num_steps = 52});
    for (const auto exercise : {ExerciseType::European, ExerciseType::American}) {
        CAPTURE(to_string(exercise));
        params.exercise_type = exercise;
        const BarrierSpec spec(params);
        const double tree = OptionValuation(u, spec, PricingMethod::Binomial).present_value();
        const double pde = OptionValuation(u, spec, PricingMethod::PdeFd).present_value();
        const double mc = OptionValuation(process, spec, PricingMethod::MonteCarlo).present_value();
        CHECK_THAT(pde, WithinRel(tree, 1e-3));   // measured <= 5e-4; the bug was ~3%
        CHECK_THAT(mc, WithinRel(tree, 1.5e-2));  // 50k paths: measured <= 0.7%; the bug was ~3%
    }
}
