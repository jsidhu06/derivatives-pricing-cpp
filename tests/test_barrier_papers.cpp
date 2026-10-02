// Barrier engines against published academic tables.
//
// Ports the paper-comparison classes of the Python suite (test_barrier.py):
// the same scenarios, paper values, errata and per-engine tolerances. Every
// engine runs with the parameters the facade resolves for barriers, as in
// the Python tests.
//
// Run with `-v high` to print one summary line per paper row, in the Python
// suite's live-log format (paper value next to each engine's), e.g.
//   dp_tests "[paper]" -v high
//
// Cases tagged [slow] take tens of seconds to minutes under the debug
// (ASan/UBSan, -O0) build; the debug test presets exclude them, the release
// presets run everything.
//
//   Boyle & Tian (1998) "An explicit finite difference approach to the
//   pricing of barrier options", Applied Mathematical Finance 5:1, 17-43 —
//   Tables 3, 6, 8 and 9.
//   Broadie & Glasserman (1997) "A Continuity Correction for Discrete
//   Barrier Options", Mathematical Finance 7(4), 325-349 — Tables 2.1, 2.4.
//   Dai & Kwok (2004) "Knock-in American options", Journal of Futures
//   Markets 24(2), 179-192.

#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/exceptions.hpp"
#include "dp/market_data.hpp"
#include "dp/underlying.hpp"
#include "dp/valuation.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/internal/catch_context.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <format>
#include <iostream>
#include <optional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

using namespace dp;
using dp::test::close_to;
using dp::test::kPricingDate;
using dp::test::WarningCapture;

namespace {

using std::chrono::days;
using std::chrono::hours;

/// numpy.isclose tolerance pair.
struct Tol {
    double rtol = 0.0;
    double atol = 0.0;
};

/// The papers' market: flat 10% rate, ACT/365F, optional flat dividend yield.
UnderlyingData paper_underlying(double spot,
                                double vol,
                                std::optional<double> div_yield = std::nullopt,
                                double curve_end = 2.0) {
    const MarketData market(kPricingDate, DiscountCurve::flat(0.10, curve_end), "USD");
    std::optional<DiscountCurve> q_curve;
    if (div_yield.has_value()) {
        q_curve = DiscountCurve::flat(*div_yield, 2.0);
    }
    return {spot, vol, market, {}, std::move(q_curve)};
}

/// European down-and-out call; discrete monitoring when `observations` is set.
BarrierSpec doc_spec(double strike,
                     double barrier,
                     DateTime maturity,
                     std::optional<int> observations = std::nullopt) {
    return BarrierSpec({.option_type = OptionType::Call,
                        .exercise_type = ExerciseType::European,
                        .strike = strike,
                        .maturity = maturity,
                        .barrier = barrier,
                        .direction = BarrierDirection::Down,
                        .action = BarrierAction::Out,
                        .monitoring = observations.has_value() ? BarrierMonitoring::Discrete
                                                               : BarrierMonitoring::Continuous,
                        .currency = "USD",
                        .num_observations = observations});
}

constexpr std::array kAllEngines{PricingMethod::Bsm, PricingMethod::Binomial, PricingMethod::PdeFd};
constexpr std::array kNumericalEngines{PricingMethod::Binomial, PricingMethod::PdeFd};

/// Index of `method` in kAllEngines (tolerance tables follow that order).
std::size_t engine_slot(PricingMethod method) {
    switch (method) {
        case PricingMethod::Bsm:
            return 0;
        case PricingMethod::Binomial:
            return 1;
        default:
            return 2;
    }
}

enum class Greek { Delta, Gamma, Theta };

/// The paper's value for `greek` out of a (delta, gamma, theta) triple.
double pick(Greek greek, double delta, double gamma, double theta) {
    switch (greek) {
        case Greek::Delta:
            return delta;
        case Greek::Gamma:
            return gamma;
        case Greek::Theta:
            return theta;
    }
    return 0.0;
}

/// Greek on the paper's reporting basis: theta annualised (library theta is
/// per calendar day).
double paper_greek(const OptionValuation& ov, Greek greek) {
    switch (greek) {
        case Greek::Delta:
            return ov.delta();
        case Greek::Gamma:
            return ov.gamma();
        case Greek::Theta:
            return ov.theta() * 365.0;
    }
    return 0.0;
}

std::string_view greek_name(Greek greek) {
    switch (greek) {
        case Greek::Delta:
            return "delta";
        case Greek::Gamma:
            return "gamma";
        case Greek::Theta:
            return "theta";
    }
    return "";
}

/// Engine results for one paper row, in kAllEngines order; empty when the
/// engine is not run or its value is skipped.
using EngineValues = std::array<std::optional<double>, 3>;

/// An engine result every row is expected to have; throws (failing the test
/// case) if the engine was unexpectedly not run.
double engine_value(const std::optional<double>& value) {
    if (!value.has_value()) {
        throw std::logic_error("engine value missing for a paper row");
    }
    return *value;
}

/// `value` to `precision` decimals, or "skipped".
std::string fmt(std::optional<double> value, int precision) {
    return value.has_value() ? std::format("{:.{}f}", *value, precision) : "skipped";
}

/// The Python suite's engine columns: " dp_bsm=... dp_bn=... dp_fd=...",
/// omitting engines a table does not run.
std::string engine_columns(const EngineValues& values, int precision, bool with_bsm = true) {
    std::string out;
    if (with_bsm) {
        out += " dp_bsm=" + fmt(values[0], precision);
    }
    out += " dp_bn=" + fmt(values[1], precision);
    out += " dp_fd=" + fmt(values[2], precision);
    return out;
}

/// Print a paper-vs-engines summary line when the run asks for high
/// verbosity (`-v high`) — the analogue of the Python suite's live log.
void paper_log(std::string_view message,
               std::source_location where = std::source_location::current()) {
    if (Catch::getCurrentContext().getConfig()->verbosity() != Catch::Verbosity::High) {
        return;
    }
    const std::string_view file = where.file_name();
    std::cout << "INFO     " << file.substr(file.find_last_of('/') + 1) << ':' << where.line()
              << ' ' << message << '\n';
}

/// "—" for continuous monitoring, else the observation count.
std::string observations_label(std::optional<int> observations) {
    return observations.has_value() ? std::to_string(*observations) : "\u2014";
}

// Paper maturities. Python builds them as `timedelta(days=...)`; these are
// the same instants (all whole hours).
const DateTime kOneMonth = kPricingDate + days{30} + hours{10};   // 365 / 12 days
const DateTime kHalfMonth = kPricingDate + days{15} + hours{5};   // 365 / 24 days
const DateTime kHalfYear = kPricingDate + days{182} + hours{12};  // 0.5 * 365 days
const DateTime kFifthYear = kPricingDate + days{73};              // 0.2 * 365 days
const DateTime kOneYear = kPricingDate + days{365};

}  // namespace

// ---------------------------------------------------------------------------
// Boyle-Tian (1998) Table 3 — continuous single barriers, closed-form column
// ---------------------------------------------------------------------------

TEST_CASE("Boyle-Tian Table 3: continuous barrier PVs", "[barrier][paper]") {
    // Paper errata: Table 3 uses 25% vol (not 20%), and Case IV's second
    // maturity is 0.5 months (not 1 month). The double knock-out case is
    // omitted.
    struct Case {
        const char* id = nullptr;
        OptionType type = OptionType::Call;
        double strike = 0.0;
        DateTime maturity;
        double barrier = 0.0;
        BarrierDirection direction = BarrierDirection::Down;
        double rebate = 0.0;
        double paper_pv = 0.0;
    };
    const std::array<Case, 6> cases{{
        {"table3_case_ii_1_month",
         OptionType::Call,
         95.0,
         kOneMonth,
         97.5,
         BarrierDirection::Down,
         0.0,
         3.6061},
        {"table3_case_ii_0_5_month",
         OptionType::Call,
         95.0,
         kHalfMonth,
         97.5,
         BarrierDirection::Down,
         0.0,
         3.7287},
        {"table3_case_iii_1_month",
         OptionType::Put,
         110.0,
         kOneMonth,
         105.0,
         BarrierDirection::Up,
         0.0,
         6.7530},
        {"table3_case_iii_0_5_month",
         OptionType::Put,
         110.0,
         kHalfMonth,
         105.0,
         BarrierDirection::Up,
         0.0,
         7.8392},
        {"table3_case_iv_6_months_rebate",
         OptionType::Call,
         100.0,
         kHalfYear,
         90.0,
         BarrierDirection::Down,
         1.0,
         8.8485},
        {"table3_case_iv_0_5_month_rebate",
         OptionType::Call,
         100.0,
         kHalfMonth,
         90.0,
         BarrierDirection::Down,
         1.0,
         2.2806},
    }};
    constexpr std::array<Tol, 3> kTols{{{1.0e-4, 7.5e-4}, {5.0e-4, 2.0e-3}, {2.0e-4, 1.0e-3}}};

    const auto& c = cases[GENERATE(range(std::size_t{0}, std::size_t{6}))];
    const BarrierSpec spec({.option_type = c.type,
                            .exercise_type = ExerciseType::European,
                            .strike = c.strike,
                            .maturity = c.maturity,
                            .barrier = c.barrier,
                            .direction = c.direction,
                            .action = BarrierAction::Out,
                            .rebate = c.rebate,
                            .rebate_timing = RebateTiming::AtHit,
                            .currency = "USD"});
    EngineValues pvs;
    for (const auto method : kAllEngines) {
        pvs[engine_slot(method)] =
            OptionValuation(paper_underlying(100.0, 0.25), spec, method).present_value();
    }
    paper_log(
        std::format("BT98 Table3 {} | paper={:.4f}{}", c.id, c.paper_pv, engine_columns(pvs, 4)));

    for (const auto method : kAllEngines) {
        INFO(c.id << " " << to_string(method));
        const Tol tol = kTols[engine_slot(method)];
        CHECK_THAT(engine_value(pvs[engine_slot(method)]),
                   close_to(c.paper_pv, tol.rtol, tol.atol));
    }
}

// ---------------------------------------------------------------------------
// Boyle-Tian (1998) Table 6 — DOC greeks as spot approaches the barrier
// ---------------------------------------------------------------------------

TEST_CASE("Boyle-Tian Table 6: DOC greeks near the barrier", "[barrier][paper][greeks][slow]") {
    // K=100, H=90, sigma=25%, r=10%, q=0, T=1y; the paper's starred
    // analytical (delta, gamma, annualised theta).
    struct Row {
        double spot = 0.0;
        double delta = 0.0;
        double gamma = 0.0;
        double theta = 0.0;
    };
    const std::array<Row, 7> rows{{{95.0, 1.1192, -0.0262, -2.6468},
                                   {92.0, 1.2132, -0.0370, -1.1229},
                                   {91.0, 1.2524, -0.0413, -0.5720},
                                   {90.5, 1.2736, -0.0437, -0.2886},
                                   {90.4, 1.2780, -0.0441, -0.2313},
                                   {90.3, 1.2825, -0.0446, -0.1738},
                                   {90.2, 1.2869, -0.0451, -0.1161}}};
    // Same tolerance for all three engines, per greek.
    constexpr std::array<Tol, 3> kTols{{{1.0e-2, 1.0e-2}, {1.5e-2, 5.0e-4}, {2.0e-2, 3.0e-2}}};

    const auto& row = rows[GENERATE(range(std::size_t{0}, std::size_t{7}))];
    const BarrierSpec spec = doc_spec(100.0, 90.0, kOneYear);
    for (const Greek greek : {Greek::Delta, Greek::Gamma, Greek::Theta}) {
        const double paper = pick(greek, row.delta, row.gamma, row.theta);
        // Values an engine itself flags as untrustworthy stay empty: a
        // Boyle-Lau alignment past the step cap (O(1/sqrt(n)) accuracy) or
        // a tree stencil straddling the barrier (refused outright).
        EngineValues values;
        for (const auto method : kAllEngines) {
            const WarningCapture warnings;
            try {
                const OptionValuation ov(paper_underlying(row.spot, 0.25, 0.0), spec, method);
                const double value = paper_greek(ov, greek);
                if (!warnings.any_contains("Boyle-Lau step alignment")) {
                    values[engine_slot(method)] = value;
                }
            } catch (const UnsupportedFeatureError&) {
                values[engine_slot(method)] = std::nullopt;  // refused by the engine
            }
        }
        paper_log(std::format("BT98 Table6 DOC spot={:.2f} {} | paper={:.6f}{}",
                              row.spot,
                              greek_name(greek),
                              paper,
                              engine_columns(values, 6)));

        const Tol tol = kTols[static_cast<std::size_t>(greek)];
        for (const auto method : kAllEngines) {
            const auto& value = values[engine_slot(method)];
            if (!value.has_value()) {
                continue;
            }
            INFO("spot " << row.spot << " " << greek_name(greek) << " " << to_string(method));
            CHECK_THAT(*value, close_to(paper, tol.rtol, tol.atol));
        }
    }
}

// ---------------------------------------------------------------------------
// Boyle-Tian (1998) Table 8 — discrete DOC across monitoring frequencies
// ---------------------------------------------------------------------------

TEST_CASE("Boyle-Tian Table 8: discrete DOC PVs by monitoring frequency", "[barrier][paper]") {
    // Cheuk-Vorst frequencies (1y = 4q = 12m = 52w = 250d, hourly = 1000/y)
    // scaled to the half-year maturity. S=K=100, H=95, sigma=20%, r=10%.
    // BSM's BGK correction degrades at low N; the hourly paper value sits ~1%
    // below the cross-engine consensus (~5.945, likely an artefact of the
    // paper's explicit FD), so PDE gets a wider tolerance on that row.
    struct Row {
        const char* label = nullptr;
        std::optional<int> observations;
        double paper_pv = 0.0;
    };
    const std::array<Row, 6> rows{{{"continuous", std::nullopt, 5.7163},
                                   {"hourly", 500, 5.8881},
                                   {"daily", 125, 6.1669},
                                   {"weekly", 26, 6.6176},
                                   {"monthly", 6, 7.3100},
                                   {"quarterly", 2, 7.9759}}};
    constexpr std::array<Tol, 3> kTols{{{0.022, 1.0e-4}, {0.013, 1.0e-4}, {0.002, 1.0e-4}}};
    constexpr Tol kPdeHourly{0.020, 1.0e-4};

    const auto& row = rows[GENERATE(range(std::size_t{0}, std::size_t{6}))];
    const BarrierSpec spec = doc_spec(100.0, 95.0, kHalfYear, row.observations);
    EngineValues pvs;
    for (const auto method : kAllEngines) {
        pvs[engine_slot(method)] =
            OptionValuation(paper_underlying(100.0, 0.20), spec, method).present_value();
    }
    paper_log(std::format("BT98 Table8 DOC freq={} N={} | paper={:.4f}{}",
                          row.label,
                          observations_label(row.observations),
                          row.paper_pv,
                          engine_columns(pvs, 4)));

    for (const auto method : kAllEngines) {
        INFO(row.label << " " << to_string(method));
        const bool pde_hourly = method == PricingMethod::PdeFd && row.observations == 500;
        const Tol tol = pde_hourly ? kPdeHourly : kTols[engine_slot(method)];
        CHECK_THAT(engine_value(pvs[engine_slot(method)]),
                   close_to(row.paper_pv, tol.rtol, tol.atol));
    }
}

// ---------------------------------------------------------------------------
// Boyle-Tian (1998) Table 9 — discrete DOC greeks by monitoring frequency
// ---------------------------------------------------------------------------

TEST_CASE("Boyle-Tian Table 9: discrete DOC greeks by monitoring frequency",
          "[barrier][paper][greeks]") {
    // Same contract as Table 8. BSM is excluded: the BGK correction is an
    // asymptotic series in 1/sqrt(N) whose derivatives break down at small N
    // (gamma even flips sign at monthly/quarterly).
    struct Row {
        const char* label = nullptr;
        std::optional<int> observations;
        double delta = 0.0;
        double gamma = 0.0;
        double theta = 0.0;
    };
    const std::array<Row, 5> rows{{{"continuous", std::nullopt, 1.0474, -0.0272, -4.4572},
                                   {"daily", 125, 0.9895, -0.0208, -5.1264},
                                   {"weekly", 26, 0.9312, -0.0134, -5.9688},
                                   {"monthly", 6, 0.8055, 0.0155, -10.4166},
                                   {"quarterly", 2, 0.6955, 0.0250, -11.1626}}};
    // [greek][binomial, PDE]. The binomial gamma tolerance absorbs
    // tree-to-monitoring-date rounding noise at weekly frequency.
    constexpr std::array<std::array<Tol, 2>, 3> kTols{{
        {{{1.5e-2, 1.0e-3}, {1.5e-2, 1.0e-3}}},
        {{{1.5e-1, 2.0e-3}, {5.0e-2, 2.0e-3}}},
        {{{7.0e-2, 2.0e-2}, {3.0e-2, 2.0e-2}}},
    }};

    const auto& row = rows[GENERATE(range(std::size_t{0}, std::size_t{5}))];
    const BarrierSpec spec = doc_spec(100.0, 95.0, kHalfYear, row.observations);
    const OptionValuation tree(paper_underlying(100.0, 0.20), spec, PricingMethod::Binomial);
    const OptionValuation pde(paper_underlying(100.0, 0.20), spec, PricingMethod::PdeFd);
    for (const Greek greek : {Greek::Delta, Greek::Gamma, Greek::Theta}) {
        const double paper = pick(greek, row.delta, row.gamma, row.theta);
        const EngineValues values{std::nullopt, paper_greek(tree, greek), paper_greek(pde, greek)};
        paper_log(std::format("BT98 Table9 DOC freq={} N={} {} | paper={:.6f}{}",
                              row.label,
                              observations_label(row.observations),
                              greek_name(greek),
                              paper,
                              engine_columns(values, 6, /*with_bsm=*/false)));

        const auto& tols = kTols[static_cast<std::size_t>(greek)];
        INFO(row.label << " " << greek_name(greek));
        CHECK_THAT(engine_value(values[1]), close_to(paper, tols[0].rtol, tols[0].atol));
        CHECK_THAT(engine_value(values[2]), close_to(paper, tols[1].rtol, tols[1].atol));
    }
}

// ---------------------------------------------------------------------------
// Broadie-Glasserman (1997) Tables 2.1 / 2.4 — discrete DOC, m = 50
// ---------------------------------------------------------------------------

namespace {

/// S=K=100, sigma=30%, r=10%, T=0.2y, m=50, H swept 85..99. "True" is the
/// paper's trinomial ground truth for the discrete problem; "Corrected" is
/// what the BGK closed-form correction produces — the BSM engine implements
/// BGK, so it is held to Corrected while the numerical engines are held to
/// True.
struct BgRow {
    double barrier = 0.0;
    double pv_true = 0.0;
    double pv_corrected = 0.0;
    double delta_true = 0.0;
    double delta_corrected = 0.0;
};
const std::array<BgRow, 15> kBroadieGlasserman{{{85.0, 6.322, 6.322, 0.591, 0.591},
                                                {86.0, 6.306, 6.306, 0.594, 0.594},
                                                {87.0, 6.281, 6.281, 0.600, 0.600},
                                                {88.0, 6.242, 6.242, 0.607, 0.607},
                                                {89.0, 6.184, 6.184, 0.618, 0.618},
                                                {90.0, 6.098, 6.098, 0.633, 0.633},
                                                {91.0, 5.977, 5.977, 0.653, 0.653},
                                                {92.0, 5.810, 5.810, 0.678, 0.678},
                                                {93.0, 5.584, 5.585, 0.711, 0.710},
                                                {94.0, 5.288, 5.288, 0.750, 0.750},
                                                {95.0, 4.907, 4.907, 0.798, 0.798},
                                                {96.0, 4.427, 4.428, 0.854, 0.853},
                                                {97.0, 3.834, 3.836, 0.917, 0.917},
                                                {98.0, 3.126, 3.121, 0.966, 0.988},
                                                {99.0, 2.337, 2.271, 0.958, 1.066}}};
constexpr int kBgObservations = 50;

}  // namespace

TEST_CASE("Broadie-Glasserman Table 2.1: discrete DOC PVs", "[barrier][paper][slow]") {
    constexpr std::array<Tol, 3> kTols{{{0.0, 1.5e-3}, {0.035, 1.0e-3}, {0.004, 1.0e-3}}};

    const auto& row =
        kBroadieGlasserman[GENERATE(range(std::size_t{0}, kBroadieGlasserman.size()))];
    const BarrierSpec spec = doc_spec(100.0, row.barrier, kFifthYear, kBgObservations);
    EngineValues pvs;
    for (const auto method : kAllEngines) {
        pvs[engine_slot(method)] =
            OptionValuation(paper_underlying(100.0, 0.30, std::nullopt, 3.0), spec, method)
                .present_value();
    }
    paper_log(std::format("BG97 DOC PV H={:.0f} m={} | true={:.3f} corrected={:.3f} |{}",
                          row.barrier,
                          kBgObservations,
                          row.pv_true,
                          row.pv_corrected,
                          engine_columns(pvs, 3)));

    for (const auto method : kAllEngines) {
        INFO("H " << row.barrier << " " << to_string(method));
        const double reference = method == PricingMethod::Bsm ? row.pv_corrected : row.pv_true;
        const Tol tol = kTols[engine_slot(method)];
        CHECK_THAT(engine_value(pvs[engine_slot(method)]), close_to(reference, tol.rtol, tol.atol));
    }
}

TEST_CASE("Broadie-Glasserman Table 2.4: discrete DOC deltas", "[barrier][paper][greeks][slow]") {
    constexpr std::array<Tol, 3> kTols{{{0.0, 1.5e-3}, {0.025, 2.0e-3}, {0.010, 2.0e-3}}};

    const auto& row =
        kBroadieGlasserman[GENERATE(range(std::size_t{0}, kBroadieGlasserman.size()))];
    const BarrierSpec spec = doc_spec(100.0, row.barrier, kFifthYear, kBgObservations);
    EngineValues deltas;
    for (const auto method : kAllEngines) {
        deltas[engine_slot(method)] =
            OptionValuation(paper_underlying(100.0, 0.30), spec, method).delta();
    }
    paper_log(std::format("BG97 DOC Δ H={:.0f} m={} | true={:.3f} corrected={:.3f} |{}",
                          row.barrier,
                          kBgObservations,
                          row.delta_true,
                          row.delta_corrected,
                          engine_columns(deltas, 3)));

    for (const auto method : kAllEngines) {
        INFO("H " << row.barrier << " " << to_string(method));
        const double reference =
            method == PricingMethod::Bsm ? row.delta_corrected : row.delta_true;
        const Tol tol = kTols[engine_slot(method)];
        CHECK_THAT(engine_value(deltas[engine_slot(method)]),
                   close_to(reference, tol.rtol, tol.atol));
    }
}

// ---------------------------------------------------------------------------
// Dai-Kwok (2004) — American down-and-in calls
// ---------------------------------------------------------------------------

TEST_CASE("Dai-Kwok: American down-and-in call PVs", "[barrier][paper][slow]") {
    // K=100, sigma=30%, r=10%, q=9%, T=1y; paper values from a 10,000-step
    // binomial scheme. The PDE two-surface solver matches to ~3dp; the tree
    // gets ~1% for Boyle-Lau cap-bind rows where H sits within 1 of spot
    // (their alignment warnings are expected and swallowed).
    struct Row {
        double barrier = 0.0;
        double spot = 0.0;
        double paper_pv = 0.0;
    };
    const std::array<Row, 11> rows{{{99.0, 99.5, 10.7432},
                                    {99.0, 110.5, 6.8224},
                                    {110.0, 110.5, 17.2062},
                                    {110.0, 120.5, 12.5409},
                                    {110.0, 140.5, 6.3553},
                                    {110.0, 160.5, 3.0667},
                                    {130.0, 130.5, 32.1285},
                                    {130.0, 140.5, 25.6659},
                                    {130.0, 150.5, 20.1773},
                                    {170.0, 170.5, 69.4759},
                                    {170.0, 180.5, 59.3874}}};
    constexpr Tol kBinomialTol{1.0e-2, 2.0e-3};
    constexpr Tol kPdeTol{1.0e-3, 2.0e-3};

    const auto& row = rows[GENERATE(range(std::size_t{0}, std::size_t{11}))];
    const BarrierSpec spec({.option_type = OptionType::Call,
                            .exercise_type = ExerciseType::American,
                            .strike = 100.0,
                            .maturity = kOneYear,
                            .barrier = row.barrier,
                            .direction = BarrierDirection::Down,
                            .action = BarrierAction::In,
                            .currency = "USD"});
    EngineValues pvs;
    for (const auto method : kNumericalEngines) {
        const WarningCapture warnings;
        pvs[engine_slot(method)] =
            OptionValuation(paper_underlying(row.spot, 0.30, 0.09), spec, method).present_value();
    }
    paper_log(std::format("DK04 AmKI H={:g} S={:g} | true={:.4f}{}",
                          row.barrier,
                          row.spot,
                          row.paper_pv,
                          engine_columns(pvs, 4, /*with_bsm=*/false)));

    INFO("H " << row.barrier << " S " << row.spot);
    CHECK_THAT(engine_value(pvs[1]), close_to(row.paper_pv, kBinomialTol.rtol, kBinomialTol.atol));
    CHECK_THAT(engine_value(pvs[2]), close_to(row.paper_pv, kPdeTol.rtol, kPdeTol.atol));
}
