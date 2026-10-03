// Market conventions through the deterministic engines, against the Python
// reference: term-structure rate / dividend curves, ACT/360, ACT/365.25 and
// 30/360 day counts, intraday timestamps, and cash dividends on a non-flat
// curve (see tools/gen_conventions_golden.py).

#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/exceptions.hpp"
#include "dp/market_data.hpp"
#include "dp/params.hpp"
#include "dp/underlying.hpp"
#include "dp/valuation.hpp"

#include "test_helpers.hpp"

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

using namespace dp;
using dp::test::close_to;
using dp::test::kPricingDate;

namespace {

// Field order mirrors the generator's column order; padding is irrelevant
// for a test fixture.
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct ConventionsGoldenRow {
    const char* label = nullptr;
    const char* engine = nullptr;
    const char* curve = nullptr;
    const char* q_curve = nullptr;
    const char* dcc = nullptr;
    int pricing_minutes = 0;
    int maturity_days = 0;
    int maturity_minutes = 0;
    double div_amount = 0.0;
    int div_day = 0;
    bool is_call = false;
    bool is_american = false;
    double barrier = 0.0;  ///< 0 for vanillas.
    const char* direction = nullptr;
    const char* action = nullptr;
    int observations = 0;  ///< 0 = continuous monitoring.
    std::optional<double> pv;
    std::optional<double> delta;
    std::optional<double> gamma;
    std::optional<double> theta;
    std::optional<double> vega;
    std::optional<double> rho;
};

#include "golden/conventions_reference.inc"

/// The generator's barrier engine settings (see the .inc header).
constexpr int kTreeBarrierSteps = 400;
constexpr int kPdeBarrierSteps = 300;

bool is(const char* value, std::string_view expected) {
    return std::string_view(value) == expected;
}

template <std::size_t N>
Eigen::ArrayXd to_eigen(const std::array<double, N>& values) {
    return Eigen::Map<const Eigen::ArrayXd>(values.data(), static_cast<Eigen::Index>(N));
}

std::optional<DiscountCurve> curve(const char* name) {
    const Eigen::ArrayXd times = to_eigen(kCurveTimes);
    if (is(name, "up")) {
        return DiscountCurve::from_forwards(times, to_eigen(kForwards_up));
    }
    if (is(name, "inv")) {
        return DiscountCurve::from_forwards(times, to_eigen(kForwards_inv));
    }
    if (is(name, "q_up")) {
        return DiscountCurve::from_forwards(times, to_eigen(kForwards_q_up));
    }
    return std::nullopt;
}

/// A curve every row is expected to name (rate curves are never "none").
DiscountCurve required_curve(const char* name) {
    std::optional<DiscountCurve> found = curve(name);
    if (!found.has_value()) {
        throw std::logic_error("unknown curve in conventions golden row");
    }
    return std::move(*found);
}

PricingMethod engine_of(const char* code) {
    if (is(code, "TREE")) {
        return PricingMethod::Binomial;
    }
    if (is(code, "PDE")) {
        return PricingMethod::PdeFd;
    }
    return PricingMethod::Bsm;
}

DayCountConvention dcc_of(const char* code) {
    if (is(code, "ACT360")) {
        return DayCountConvention::Act360;
    }
    if (is(code, "ACT365_25")) {
        return DayCountConvention::Act365_25;
    }
    if (is(code, "THIRTY360")) {
        return DayCountConvention::Thirty360Us;
    }
    return DayCountConvention::Act365F;
}

OptionValuation replay(const ConventionsGoldenRow& row) {
    using std::chrono::days;
    using std::chrono::minutes;
    const DateTime pricing = kPricingDate + minutes{row.pricing_minutes};
    const DateTime maturity =
        kPricingDate + days{row.maturity_days} + minutes{row.maturity_minutes};
    const MarketData market(pricing, required_curve(row.curve), "USD", dcc_of(row.dcc));
    DividendSchedule dividends;
    if (row.div_amount != 0.0) {
        dividends.push_back({kPricingDate + days{row.div_day}, row.div_amount});
    }
    const UnderlyingData underlying(100.0, 0.25, market, std::move(dividends), curve(row.q_curve));
    const OptionType type = row.is_call ? OptionType::Call : OptionType::Put;
    const ExerciseType exercise = row.is_american ? ExerciseType::American : ExerciseType::European;
    const PricingMethod method = engine_of(row.engine);
    if (row.barrier == 0.0) {
        return {underlying,
                VanillaSpec({.option_type = type,
                             .exercise_type = exercise,
                             .strike = 100.0,
                             .maturity = maturity,
                             .currency = "USD"}),
                method};
    }

    const BarrierMonitoring monitoring =
        row.observations > 0 ? BarrierMonitoring::Discrete : BarrierMonitoring::Continuous;
    BarrierSpec::Params params{
        .option_type = type,
        .exercise_type = exercise,
        .strike = 100.0,
        .maturity = maturity,
        .barrier = row.barrier,
        .direction = is(row.direction, "U") ? BarrierDirection::Up : BarrierDirection::Down,
        .action = is(row.action, "I") ? BarrierAction::In : BarrierAction::Out,
        .monitoring = monitoring,
        .currency = "USD"};
    if (row.observations > 0) {
        params.num_observations = row.observations;
    }
    std::optional<ValuationParams> engine_params;
    if (method == PricingMethod::Binomial) {
        engine_params = BinomialParams{.num_steps = kTreeBarrierSteps};
    } else if (method == PricingMethod::PdeFd) {
        PdeParams pde = PdeParams::for_barriers(monitoring);
        pde.spot_steps = kPdeBarrierSteps;
        pde.time_steps = kPdeBarrierSteps;
        pde.method = PdeMethod::CrankNicolson;
        pde.american_solver = PdeEarlyExercise::GaussSeidel;
        engine_params = pde;
    }
    return {underlying, BarrierSpec(std::move(params)), method, engine_params};
}

/// Per-engine numpy.isclose tolerances: closed forms tightest, the lattice
/// and grid engines at their golden-table parity levels.
std::pair<double, double> tolerance(const char* engine) {
    if (is(engine, "BSM")) {
        return {1e-10, 1e-12};
    }
    if (is(engine, "TREE")) {
        return {1e-9, 1e-11};
    }
    return {1e-8, 1e-10};
}

}  // namespace

TEST_CASE("market conventions match the Python reference implementation",
          "[valuation][conventions][golden]") {
    for (const auto& row : std::span(kConventionsGolden)) {
        INFO("row " << row.label);
        if (!row.pv.has_value()) {
            // The whole valuation is refused (e.g. a tree on 30/360).
            CHECK_THROWS_AS(replay(row).present_value(), DerivativesPricingError);
            continue;
        }
        const OptionValuation ov = replay(row);
        const auto [rtol, atol] = tolerance(row.engine);
        const std::array<std::pair<const char*, std::function<double()>>, 6> values{{
            {"pv", [&] { return ov.present_value(); }},
            {"delta", [&] { return ov.delta(); }},
            {"gamma", [&] { return ov.gamma(); }},
            {"theta", [&] { return ov.theta(); }},
            {"vega", [&] { return ov.vega(); }},
            {"rho", [&] { return ov.rho(); }},
        }};
        const std::array<std::optional<double>, 6> expected{
            row.pv, row.delta, row.gamma, row.theta, row.vega, row.rho};
        for (std::size_t i = 0; i < values.size(); ++i) {
            INFO(values[i].first);
            if (const auto& e = expected[i]; e.has_value()) {
                CHECK_THAT(values[i].second(), close_to(*e, rtol, atol));
            } else {
                CHECK_THROWS_AS(values[i].second(), DerivativesPricingError);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Independent checks: the golden table proves parity with Python, these
// would also catch a convention bug the two libraries shared.
// ---------------------------------------------------------------------------

namespace {

/// 18-month vanilla on 2025-01-01 with the named rate / dividend curves.
OptionValuation term_vanilla(PricingMethod method,
                             const char* rate_curve,
                             const char* q_curve,
                             OptionType type,
                             ExerciseType exercise = ExerciseType::European,
                             std::optional<ValuationParams> params = std::nullopt,
                             DayCountConvention dcc = DayCountConvention::Act365F) {
    const MarketData market(kPricingDate, required_curve(rate_curve), "USD", dcc);
    const UnderlyingData underlying(100.0, 0.25, market, {}, curve(q_curve));
    return {underlying,
            VanillaSpec({.option_type = type,
                         .exercise_type = exercise,
                         .strike = 100.0,
                         .maturity = kPricingDate + std::chrono::days{548},
                         .currency = "USD"}),
            method,
            params};
}

// Refined enough that discretisation error sits well inside the tolerance:
// measured gaps to BSM are ~7.5e-4 (tree, O(1/N)) and ~3.9e-4 (PDE, O(h^2)).
const BinomialParams kFineTree{.num_steps = 4000};
const PdeParams kFinePde{.spot_steps = 800, .time_steps = 800};

}  // namespace

TEST_CASE("tree and PDE converge to BSM under term-structure curves", "[valuation][conventions]") {
    struct Case {
        const char* rates;
        const char* dividends;
        OptionType type;
    };
    for (const Case c : {Case{"up", "none", OptionType::Call},
                         Case{"up", "q_up", OptionType::Put},
                         Case{"inv", "none", OptionType::Put},
                         Case{"inv", "q_up", OptionType::Call}}) {
        INFO(c.rates << " / " << c.dividends << " " << to_string(c.type));
        const double bsm =
            term_vanilla(PricingMethod::Bsm, c.rates, c.dividends, c.type).present_value();
        const double tree = term_vanilla(PricingMethod::Binomial,
                                         c.rates,
                                         c.dividends,
                                         c.type,
                                         ExerciseType::European,
                                         kFineTree)
                                .present_value();
        const double pde = term_vanilla(PricingMethod::PdeFd,
                                        c.rates,
                                        c.dividends,
                                        c.type,
                                        ExerciseType::European,
                                        kFinePde)
                               .present_value();
        CHECK_THAT(tree, close_to(bsm, 0.0, 1.5e-3));
        CHECK_THAT(pde, close_to(bsm, 0.0, 1.5e-3));
    }

    SECTION("American: tree and PDE agree with each other") {
        const double tree = term_vanilla(PricingMethod::Binomial,
                                         "up",
                                         "q_up",
                                         OptionType::Put,
                                         ExerciseType::American,
                                         kFineTree)
                                .present_value();
        const double pde = term_vanilla(PricingMethod::PdeFd,
                                        "up",
                                        "q_up",
                                        OptionType::Put,
                                        ExerciseType::American,
                                        kFinePde)
                               .present_value();
        CHECK_THAT(tree, close_to(pde, 0.0, 2e-3));
    }
}

TEST_CASE("Monte Carlo integrates a term curve under ACT/360 like BSM",
          "[valuation][conventions][mc]") {
    // MC discounts and drifts step by step along the simulation grid — a
    // separate code path from BSM's single df(T) — so agreement checks the
    // curve integration and the ACT/360 year fractions end to end.
    const MarketData market(kPricingDate, required_curve("up"), "USD", DayCountConvention::Act360);
    const DateTime maturity = kPricingDate + std::chrono::days{548};
    const VanillaSpec spec({.option_type = OptionType::Call,
                            .exercise_type = ExerciseType::European,
                            .strike = 100.0,
                            .maturity = maturity,
                            .currency = "USD"});
    const double bsm =
        OptionValuation(
            UnderlyingData(100.0, 0.25, market, {}, curve("q_up")), spec, PricingMethod::Bsm)
            .present_value();
    const GbmProcess gbm(
        market,
        {.initial_value = 100.0, .volatility = 0.25, .dividend_curve = curve("q_up")},
        {.paths = 200'000, .end_date = maturity, .num_steps = 18});
    const double mc = OptionValuation(gbm, spec, PricingMethod::MonteCarlo).present_value();
    // 1% is ~3.2 standard deviations of the estimator (measured over 30
    // seeds; this seed sits 0.3% from BSM). Replacing the term curve with a
    // flat 4.5% would move the price ~3.4%, roughly 11 standard deviations.
    CHECK_THAT(mc, close_to(bsm, 1e-2, 0.0));
}

TEST_CASE("ACT/360 prices above ACT/365F on the same dates", "[valuation][conventions]") {
    // The same calendar span is a longer year fraction under ACT/360, so an
    // at-the-money option carries more time value.
    for (const auto method : {PricingMethod::Bsm, PricingMethod::Binomial, PricingMethod::PdeFd}) {
        for (const auto type : {OptionType::Call, OptionType::Put}) {
            INFO(to_string(method) << " " << to_string(type));
            const auto pv = [&](DayCountConvention dcc) {
                return term_vanilla(
                           method, "up", "none", type, ExerciseType::European, std::nullopt, dcc)
                    .present_value();
            };
            CHECK(pv(DayCountConvention::Act360) > pv(DayCountConvention::Act365F));
        }
    }
}
