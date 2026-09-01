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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <cmath>

using namespace dp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using dp::test::close_to;
using dp::test::flat_curve;
using dp::test::kMaturity;
using dp::test::kPricingDate;
using dp::test::message_contains;

namespace {

constexpr double kSpot = 100.0;
constexpr double kStrike = 100.0;
constexpr double kRate = 0.05;
constexpr double kVol = 0.20;
// BSM reference for S=100, K=100, r=0.05, sigma=0.2, T=1.
constexpr double kBsmAtmCall = 10.4506;

MarketData md(double rate = kRate,
              DayCountConvention dcc = DayCountConvention::Act365F,
              DateTime pricing_date = kPricingDate) {
    return {pricing_date, DiscountCurve::flat(rate), "USD", dcc};
}

UnderlyingData ud(double spot = kSpot,
                  double vol = kVol,
                  MarketData market = md(),
                  DividendSchedule dividends = {},
                  std::optional<DiscountCurve> q_curve = std::nullopt) {
    return {spot, vol, std::move(market), std::move(dividends), std::move(q_curve)};
}

VanillaSpec spec(OptionType type = OptionType::Call,
                 ExerciseType exercise = ExerciseType::European,
                 double strike = kStrike,
                 DateTime maturity = kMaturity) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = exercise,
                        .strike = strike,
                        .maturity = maturity,
                        .currency = "USD"});
}

double tree_pv(const UnderlyingData& u, const OptionSpec& s, BinomialParams params = {}) {
    return OptionValuation(u, s, PricingMethod::Binomial, params).present_value();
}

constexpr GreekOptions kTree{.method = GreekCalculationMethod::Tree};
constexpr GreekOptions kNumerical{.method = GreekCalculationMethod::Numerical};

}  // namespace

// ---------------------------------------------------------------------------
// Present value
// ---------------------------------------------------------------------------

TEST_CASE("binomial European ATM call converges to BSM", "[binomial]") {
    CHECK_THAT(tree_pv(ud(), spec(), {.num_steps = 500}), WithinRel(kBsmAtmCall, 5e-3));
}

TEST_CASE("American call without dividends equals European", "[binomial]") {
    const double eu = tree_pv(ud(), spec(), {.num_steps = 500});
    const double am =
        tree_pv(ud(), spec(OptionType::Call, ExerciseType::American), {.num_steps = 500});
    CHECK_THAT(am, WithinRel(eu, 5e-3));
    CHECK(am >= eu - 1e-8);
}

TEST_CASE("discrete dividends reduce the European call price", "[binomial]") {
    const double base = tree_pv(ud(), spec());
    const double with_div =
        tree_pv(ud(kSpot, kVol, md(), {{kPricingDate + std::chrono::days{180}, 1.0}}), spec());
    CHECK(with_div < base);
}

TEST_CASE("American put carries an early-exercise premium", "[binomial]") {
    const BinomialParams params{.num_steps = 100};
    const double eu = tree_pv(ud(), spec(OptionType::Put), params);
    const double am = tree_pv(ud(), spec(OptionType::Put, ExerciseType::American), params);
    CHECK(am > eu);
}

TEST_CASE("more steps converge towards BSM", "[binomial]") {
    const double pv_100 = tree_pv(ud(), spec(), {.num_steps = 100});
    const double pv_200 = tree_pv(ud(), spec(), {.num_steps = 200});
    CHECK(std::abs(pv_200 - kBsmAtmCall) < std::abs(pv_100 - kBsmAtmCall));
    CHECK_THAT(pv_100, WithinRel(kBsmAtmCall, 1e-2));
    CHECK_THAT(pv_200, WithinRel(kBsmAtmCall, 5e-3));
}

TEST_CASE("dividend curve enters the drift", "[binomial]") {
    const auto q_curve = flat_curve(kPricingDate, kMaturity, 0.03);
    const double with_q = tree_pv(ud(kSpot, kVol, md(), {}, q_curve), spec());
    CHECK(with_q < tree_pv(ud(), spec()));
}

// ---------------------------------------------------------------------------
// Arbitrage guard / validation
// ---------------------------------------------------------------------------

TEST_CASE("zero volatility violates the CRR no-arbitrage bounds", "[binomial][edge]") {
    CHECK_THROWS_MATCHES(tree_pv(ud(kSpot, 0.0),
                                 spec(OptionType::Call, ExerciseType::European, 90.0),
                                 {.num_steps = 100}),
                         ArbitrageViolationError,
                         message_contains("No-arbitrage condition"));
}

TEST_CASE("near-zero volatility raises cleanly", "[binomial][edge]") {
    const double vol = GENERATE(1.0e-4, 5.0e-4, 1.0e-3);
    CHECK_THROWS_AS(tree_pv(ud(50.0, vol),
                            spec(OptionType::Put, ExerciseType::American, 200.0),
                            {.num_steps = 500}),
                    ArbitrageViolationError);
}

TEST_CASE("30/360 day count is rejected for CRR trees", "[binomial][edge]") {
    const UnderlyingData u(kSpot, kVol, md(kRate, DayCountConvention::Thirty360Us));
    CHECK_THROWS_MATCHES(
        tree_pv(u, spec()), UnsupportedFeatureError, message_contains("equal time steps"));
}

TEST_CASE("negative rates still price on the tree", "[binomial][edge]") {
    const dp::test::WarningCapture capture;
    const UnderlyingData u(kSpot, kVol, md(-0.02));
    for (const auto type : {OptionType::Call, OptionType::Put}) {
        const double pv = tree_pv(u, spec(type), {.num_steps = 200});
        CHECK(std::isfinite(pv));
        CHECK(pv > 0.0);
    }
}

// ---------------------------------------------------------------------------
// Tree Greeks
// ---------------------------------------------------------------------------

TEST_CASE("tree Greeks are the default for binomial and look sane", "[binomial][greeks]") {
    const OptionValuation call(ud(), spec(), PricingMethod::Binomial, BinomialParams{});
    const OptionValuation put(
        ud(), spec(OptionType::Put), PricingMethod::Binomial, BinomialParams{});
    // Auto-selection resolves to TREE: explicit TREE must agree exactly.
    CHECK(call.delta() == call.delta(kTree));
    CHECK(call.gamma() == call.gamma(kTree));
    CHECK(call.theta() == call.theta(kTree));

    CHECK((call.delta() > 0.5 && call.delta() < 1.0));
    CHECK(put.delta() < 0.0);
    CHECK(call.gamma() > 0.0);
    CHECK(call.theta() < 0.0);
    // Tree delta close to BSM analytical delta.
    const OptionValuation bsm(ud(), spec(), PricingMethod::Bsm);
    CHECK_THAT(call.delta(), WithinRel(bsm.delta(), 1e-2));
    CHECK_THAT(call.gamma(), WithinRel(bsm.gamma(), 1e-2));
}

TEST_CASE("tree theta matches numerical theta at a 2*dt horizon", "[binomial][greeks]") {
    const int num_steps = 200;
    const OptionValuation ov(
        ud(), spec(), PricingMethod::Binomial, BinomialParams{.num_steps = num_steps});
    const double ttm = ov.maturity_year_fraction();
    const double bump_days = 2.0 * (ttm / num_steps) * 365.0;
    const double tree = ov.theta(kTree);
    const double numerical =
        ov.theta({.method = GreekCalculationMethod::Numerical, .bump = bump_days});
    CHECK_THAT(tree, close_to(numerical, 2e-3, 5e-5));
}

TEST_CASE("tree extraction is rejected for vega and rho", "[binomial][greeks]") {
    const OptionValuation ov(ud(), spec(), PricingMethod::Binomial, BinomialParams{});
    CHECK_THROWS_MATCHES(
        ov.vega(kTree), ValidationError, message_contains("Tree extraction is not available"));
    CHECK_THROWS_MATCHES(
        ov.rho(kTree), ValidationError, message_contains("Tree extraction is not available"));
    // Numerical vega/rho remain available (and are the auto-selected method).
    CHECK(ov.vega() > 0.0);
    CHECK(ov.rho() > 0.0);
}

TEST_CASE("tree gamma and theta need at least two steps", "[binomial][greeks]") {
    const OptionValuation ov(ud(), spec(), PricingMethod::Binomial, BinomialParams{.num_steps = 1});
    CHECK_THROWS_MATCHES(
        ov.gamma(kTree), ValidationError, message_contains("Tree gamma requires num_steps >= 2"));
    CHECK_THROWS_MATCHES(
        ov.theta(kTree), ValidationError, message_contains("Tree theta requires num_steps >= 2"));
    CHECK_NOTHROW(ov.delta(kTree));
}

TEST_CASE("tree Greeks are finite on the smallest admissible trees", "[binomial][greeks][edge]") {
    // For n == 1 (delta) and n == 2 (gamma/theta) the nodes the Greeks read
    // sit on the terminal layer, which the backward-induction loop never
    // revisits; regression guard against those nodes staying uncaptured.
    const OptionValuation one(
        ud(), spec(), PricingMethod::Binomial, BinomialParams{.num_steps = 1});
    const double delta_one = one.delta(kTree);
    CHECK(std::isfinite(delta_one));
    CHECK((delta_one > 0.0 && delta_one < 1.0));

    const OptionValuation two(
        ud(), spec(), PricingMethod::Binomial, BinomialParams{.num_steps = 2});
    const double delta_two = two.delta(kTree);
    CHECK(std::isfinite(delta_two));
    CHECK((delta_two > 0.0 && delta_two < 1.0));
    CHECK(std::isfinite(two.gamma(kTree)));
    CHECK(two.gamma(kTree) > 0.0);
    CHECK(std::isfinite(two.theta(kTree)));
    CHECK(two.theta(kTree) < 0.0);
}

TEST_CASE("binomial greeks do not mutate the underlying", "[binomial][purity]") {
    const UnderlyingData underlying = ud();
    const OptionValuation ov(
        underlying, spec(), PricingMethod::Binomial, BinomialParams{.num_steps = 200});
    (void)ov.delta();
    (void)ov.gamma();
    (void)ov.vega();
    (void)ov.theta(kNumerical);
    (void)ov.rho();
    CHECK(underlying.initial_value() == kSpot);
    CHECK(underlying.volatility() == kVol);
    CHECK(underlying.pricing_date() == kPricingDate);
}

// ---------------------------------------------------------------------------
// Control variate
// ---------------------------------------------------------------------------

TEST_CASE("control variate equals the explicit adjustment identity", "[binomial][cv]") {
    const UnderlyingData u(100.0, 0.25, md(0.03));
    const auto amer = spec(OptionType::Put, ExerciseType::American, 110.0);
    const auto euro = spec(OptionType::Put, ExerciseType::European, 110.0);
    const BinomialParams base{.num_steps = 300};
    const BinomialParams cv{.num_steps = 300, .control_variate_european = true};

    const double american_raw = tree_pv(u, amer, base);
    const double european_num = tree_pv(u, euro, base);
    const double european_bsm = OptionValuation(u, euro, PricingMethod::Bsm).present_value();
    const double american_cv = tree_pv(u, amer, cv);

    CHECK_THAT(american_cv, WithinAbs(american_raw + (european_bsm - european_num), 1e-8));
}

TEST_CASE("control variate demands American exercise", "[binomial][cv]") {
    const OptionValuation ov(
        ud(), spec(), PricingMethod::Binomial, BinomialParams{.control_variate_european = true});
    CHECK_THROWS_MATCHES(ov.present_value(),
                         ValidationError,
                         message_contains("only valid for options with American exercise"));
}

TEST_CASE("control variate rejects PayoffSpec", "[binomial][cv]") {
    const PayoffSpec payoff({.exercise_type = ExerciseType::American,
                             .maturity = kMaturity,
                             .payoff_fn = [](double s) { return std::max(s - 95.0, 0.0); },
                             .currency = "USD"});
    const OptionValuation ov(
        ud(), payoff, PricingMethod::Binomial, BinomialParams{.control_variate_european = true});
    CHECK_THROWS_MATCHES(ov.present_value(),
                         UnsupportedFeatureError,
                         message_contains("not supported for PayoffSpec"));
}

// ---------------------------------------------------------------------------
// Custom payoffs
// ---------------------------------------------------------------------------

namespace {

double bull_call_spread(double s) {
    return std::max(s - 95.0, 0.0) - std::max(s - 115.0, 0.0);
}

PayoffSpec payoff_spec(ExerciseType exercise = ExerciseType::European) {
    return PayoffSpec({.exercise_type = exercise,
                       .maturity = kMaturity,
                       .payoff_fn = bull_call_spread,
                       .currency = "USD"});
}

}  // namespace

TEST_CASE("bull spread payoff prices within its bounds", "[binomial][payoff]") {
    const double pv = tree_pv(ud(), payoff_spec());
    CHECK(pv > 0.0);
    CHECK(pv < 20.0);  // max payoff of the spread
}

TEST_CASE("American custom payoff is worth at least the European", "[binomial][payoff]") {
    CHECK(tree_pv(ud(), payoff_spec(ExerciseType::American)) >=
          tree_pv(ud(), payoff_spec()) - 1e-10);
}

TEST_CASE("custom payoff replicates its vanilla legs", "[binomial][payoff]") {
    const double spread = tree_pv(ud(), payoff_spec());
    const double long_leg = tree_pv(ud(), spec(OptionType::Call, ExerciseType::European, 95.0));
    const double short_leg = tree_pv(ud(), spec(OptionType::Call, ExerciseType::European, 115.0));
    CHECK_THAT(spread, WithinRel(long_leg - short_leg, 1e-10));
}

TEST_CASE("custom payoff tree delta matches numerical delta", "[binomial][payoff][greeks]") {
    const OptionValuation ov(
        ud(), payoff_spec(), PricingMethod::Binomial, BinomialParams{.num_steps = 500});
    CHECK_THAT(ov.delta(kTree), close_to(ov.delta(kNumerical), 5e-2, 1e-3));
    CHECK((ov.delta(kTree) > 0.0 && ov.delta(kTree) < 1.0));
}

// ---------------------------------------------------------------------------
// Golden values from the Python reference implementation
// ---------------------------------------------------------------------------

namespace {

struct BinomialGoldenRow {
    const char* label;
    double spot;
    double strike;
    double rate;
    double div_yield;
    double vol;
    int maturity_days;
    int num_steps;
    double div_amount;
    int div_day;
    bool is_call;
    bool is_american;
    bool control_variate;
    double pv;
    double delta_tree;
    double gamma_tree;
    double theta_tree;
    double vega_num;
    double rho_num;
};

#include "golden/binomial_reference.inc"

OptionValuation replay(const BinomialGoldenRow& row) {
    const DateTime maturity = kPricingDate + std::chrono::days{row.maturity_days};
    const MarketData market(kPricingDate, DiscountCurve::flat(row.rate), "USD");
    std::optional<DiscountCurve> q_curve;
    if (row.div_yield != 0.0) {
        q_curve = DiscountCurve::flat(row.div_yield);
    }
    DividendSchedule dividends;
    if (row.div_amount != 0.0) {
        dividends.push_back({kPricingDate + std::chrono::days{row.div_day}, row.div_amount});
    }
    const UnderlyingData u(row.spot, row.vol, market, std::move(dividends), std::move(q_curve));
    const VanillaSpec s(
        {.option_type = row.is_call ? OptionType::Call : OptionType::Put,
         .exercise_type = row.is_american ? ExerciseType::American : ExerciseType::European,
         .strike = row.strike,
         .maturity = maturity});
    return {u,
            s,
            PricingMethod::Binomial,
            BinomialParams{.num_steps = row.num_steps,
                           .control_variate_european = row.control_variate}};
}

}  // namespace

TEST_CASE("binomial matches the Python reference implementation", "[binomial][golden]") {
    for (const auto& row : kBinomialGolden) {
        INFO("row " << row.label);
        const auto ov = replay(row);
        CHECK_THAT(ov.present_value(), close_to(row.pv, 1e-10, 1e-12));
        if (row.control_variate) {
            continue;  // CV rows pin PV only.
        }
        CHECK_THAT(ov.delta(kTree), close_to(row.delta_tree, 1e-10, 1e-12));
        CHECK_THAT(ov.gamma(kTree), close_to(row.gamma_tree, 1e-10, 1e-12));
        CHECK_THAT(ov.theta(kTree), close_to(row.theta_tree, 1e-10, 1e-12));
        CHECK_THAT(ov.vega(kNumerical), close_to(row.vega_num, 1e-8, 1e-10));
        CHECK_THAT(ov.rho(kNumerical), close_to(row.rho_num, 1e-8, 1e-10));
    }
}
