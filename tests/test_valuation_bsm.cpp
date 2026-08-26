#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/forward.hpp"
#include "dp/market_data.hpp"
#include "dp/underlying.hpp"
#include "dp/valuation.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <cmath>
#include <string>

using namespace dp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using dp::test::close_to;
using dp::test::flat_curve;
using dp::test::kMaturity;
using dp::test::kPricingDate;

namespace {

constexpr double kSpot = 100.0;
constexpr double kStrike = 100.0;
constexpr double kRate = 0.05;
constexpr double kVol = 0.20;

MarketData md(DateTime pricing_date = kPricingDate,
              DateTime maturity = kMaturity,
              double rate = kRate) {
    return {pricing_date, flat_curve(pricing_date, maturity, rate), "USD"};
}

UnderlyingData ud(double spot = kSpot,
                  double vol = kVol,
                  std::optional<DiscountCurve> q_curve = std::nullopt,
                  DividendSchedule dividends = {}) {
    return {spot, vol, md(), std::move(dividends), std::move(q_curve)};
}

VanillaSpec spec(OptionType type = OptionType::Call,
                 double strike = kStrike,
                 DateTime maturity = kMaturity) {
    return VanillaSpec({.option_type = type,
                        .exercise_type = ExerciseType::European,
                        .strike = strike,
                        .maturity = maturity,
                        .currency = "USD"});
}

double pv(const UnderlyingData& u, const VanillaSpec& s) {
    return OptionValuation(u, s, PricingMethod::Bsm).present_value();
}

OptionValuation bsm(const UnderlyingData& u, const VanillaSpec& s) {
    return {u, s, PricingMethod::Bsm};
}

constexpr GreekOptions kNumerical{.method = GreekCalculationMethod::Numerical};
constexpr GreekOptions kAnalytical{.method = GreekCalculationMethod::Analytical};

}  // namespace

// ---------------------------------------------------------------------------
// Reference values (S=100, K=100, r=0.05, sigma=0.20, T=1, no dividends)
// ---------------------------------------------------------------------------

TEST_CASE("BSM ATM call and put match textbook values", "[bsm]") {
    CHECK_THAT(pv(ud(), spec(OptionType::Call)), WithinRel(10.4506, 1e-4));
    CHECK_THAT(pv(ud(), spec(OptionType::Put)), WithinRel(5.5735, 1e-4));
}

TEST_CASE("BSM ITM call and OTM put at S=110", "[bsm]") {
    const double call = pv(ud(110.0), spec(OptionType::Call));
    CHECK(call > 110.0 - kStrike * std::exp(-kRate));
    CHECK_THAT(call, WithinRel(17.6630, 1e-4));
    CHECK_THAT(pv(ud(110.0), spec(OptionType::Put)), WithinRel(2.7859, 1e-4));
}

TEST_CASE("dividends reduce the call price", "[bsm]") {
    const double base = pv(ud(), spec());
    CHECK(pv(ud(kSpot, kVol, std::nullopt, {{kPricingDate + std::chrono::days{180}, 1.0}}),
             spec()) < base);
    CHECK(pv(ud(kSpot, kVol, flat_curve(kPricingDate, kMaturity, 0.03)), spec()) < base);
}

TEST_CASE("BSM satisfies put-call parity", "[bsm][parity]") {
    SECTION("no dividends") {
        const double c = pv(ud(), spec(OptionType::Call));
        const double p = pv(ud(), spec(OptionType::Put));
        CHECK_THAT(c - p, WithinRel(kSpot - kStrike * std::exp(-kRate), 1e-10));
    }
    SECTION("with dividend curve, against put_call_parity_rhs") {
        const auto r_curve = flat_curve(kPricingDate, kMaturity, 0.04);
        const auto q_curve = flat_curve(kPricingDate, kMaturity, 0.02);
        const UnderlyingData u(kSpot, 0.25, {kPricingDate, r_curve, "USD"}, {}, q_curve);
        const double c = pv(u, spec(OptionType::Call, 105.0));
        const double p = pv(u, spec(OptionType::Put, 105.0));
        const double rhs = put_call_parity_rhs({.spot = kSpot,
                                                .pricing_date = kPricingDate,
                                                .maturity = kMaturity,
                                                .discount_curve = &r_curve,
                                                .dividend_curve = &q_curve},
                                               105.0);
        CHECK_THAT(c - p, WithinRel(rhs, 1e-10));
    }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST_CASE("zero volatility prices discounted intrinsic", "[bsm][edge]") {
    const auto maturity = make_datetime(2025, 7, 3);  // ~0.5y
    const MarketData market(kPricingDate, flat_curve(kPricingDate, maturity, kRate), "USD");
    const UnderlyingData u(100.0, 0.0, market);
    const double df = market.discount_curve().df(0.5);

    CHECK_THAT(pv(u, spec(OptionType::Call, 90.0, maturity)), WithinRel(100.0 - 90.0 * df, 1e-2));
    CHECK_THAT(pv(u, spec(OptionType::Put, 110.0, maturity)), WithinRel(110.0 * df - 100.0, 1e-2));
    CHECK_THAT(pv(u, spec(OptionType::Call, 110.0, maturity)), WithinAbs(0.0, 1e-10));
    CHECK_THAT(pv(u, spec(OptionType::Put, 90.0, maturity)), WithinAbs(0.0, 1e-10));
    for (const auto type : {OptionType::Call, OptionType::Put}) {
        CHECK(std::abs(pv(u, spec(type, 100.0, maturity))) < 5.0);
    }
}

TEST_CASE("tiny volatility converges to discounted intrinsic", "[bsm][edge]") {
    const auto maturity = make_datetime(2025, 7, 3);
    const MarketData market(kPricingDate, flat_curve(kPricingDate, maturity, kRate), "USD");
    const UnderlyingData u(100.0, 1e-4, market);
    const double df = market.discount_curve().df(0.5);
    CHECK_THAT(pv(u, spec(OptionType::Call, 80.0, maturity)), WithinRel(100.0 - 80.0 * df, 1e-2));
    CHECK_THAT(pv(u, spec(OptionType::Put, 120.0, maturity)), WithinRel(120.0 * df - 100.0, 1e-2));
    CHECK(pv(u, spec(OptionType::Call, 120.0, maturity)) < 0.01);
    CHECK(pv(u, spec(OptionType::Put, 80.0, maturity)) < 0.01);
}

TEST_CASE("one-day expiry is near intrinsic", "[bsm][edge]") {
    const auto maturity = kPricingDate + std::chrono::days{1};
    const UnderlyingData u(100.0, kVol, md(kPricingDate, maturity));
    CHECK_THAT(pv(u, spec(OptionType::Call, 90.0, maturity)), WithinAbs(10.0, 0.5));
    CHECK_THAT(pv(u, spec(OptionType::Put, 110.0, maturity)), WithinAbs(10.0, 0.5));
    CHECK_THAT(pv(u, spec(OptionType::Call, 110.0, maturity)), WithinAbs(0.0, 0.5));
    CHECK_THAT(pv(u, spec(OptionType::Put, 90.0, maturity)), WithinAbs(0.0, 0.5));
}

TEST_CASE("extreme rates still price", "[bsm][edge]") {
    const dp::test::WarningCapture capture;  // negative rates warn; that's fine here
    for (const double rate : {-0.02, 0.25, 0.50}) {
        const UnderlyingData u(100.0, kVol, md(kPricingDate, kMaturity, rate));
        for (const auto type : {OptionType::Call, OptionType::Put}) {
            const double value = pv(u, spec(type));
            CHECK(std::isfinite(value));
            CHECK(value > 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Greek properties
// ---------------------------------------------------------------------------

TEST_CASE("delta sign and magnitude", "[bsm][greeks]") {
    CHECK(bsm(ud(), spec(OptionType::Call)).delta() > 0.0);
    CHECK(bsm(ud(), spec(OptionType::Put)).delta() < 0.0);
    const double atm = bsm(ud(), spec(OptionType::Call)).delta();
    CHECK((atm > 0.5 && atm < 1.0));  // forward above spot with r > 0
    CHECK(bsm(ud(150.0), spec(OptionType::Call)).delta() > 0.95);
    CHECK(bsm(ud(50.0), spec(OptionType::Call)).delta() < 0.05);
    CHECK(bsm(ud(50.0), spec(OptionType::Put)).delta() < -0.95);
    CHECK(bsm(ud(150.0), spec(OptionType::Put)).delta() > -0.05);
    const double custom =
        bsm(ud(), spec()).delta({.method = GreekCalculationMethod::Numerical, .bump = 0.5});
    CHECK((custom > 0.0 && custom < 1.0));
}

TEST_CASE("gamma is positive, peaks ATM and grows near expiry", "[bsm][greeks]") {
    CHECK(bsm(ud(), spec(OptionType::Call)).gamma() > 0.0);
    CHECK(bsm(ud(), spec(OptionType::Put)).gamma() > 0.0);
    CHECK(bsm(ud(), spec()).gamma() > bsm(ud(110.0), spec()).gamma());
    CHECK(bsm(ud(), spec(OptionType::Call, kStrike, make_datetime(2025, 3, 1))).gamma() >
          bsm(ud(), spec()).gamma());
}

TEST_CASE("vega is positive, peaks ATM and grows with maturity", "[bsm][greeks]") {
    CHECK(bsm(ud(), spec(OptionType::Call)).vega() > 0.0);
    CHECK(bsm(ud(), spec(OptionType::Put)).vega() > 0.0);
    CHECK(bsm(ud(), spec()).vega() > bsm(ud(110.0), spec()).vega());
    CHECK(bsm(ud(), spec()).vega() >
          bsm(ud(), spec(OptionType::Call, kStrike, make_datetime(2025, 3, 1))).vega());
    CHECK(bsm(ud(), spec()).vega({.method = GreekCalculationMethod::Numerical, .bump = 0.02}) >
          0.0);
}

TEST_CASE("rho sign and monotonicity", "[bsm][greeks]") {
    CHECK(bsm(ud(), spec(OptionType::Call)).rho() > 0.0);
    CHECK(bsm(ud(), spec(OptionType::Put)).rho() < 0.0);
    const double rho_short =
        bsm(ud(), spec(OptionType::Call, kStrike, make_datetime(2025, 4, 1))).rho();
    const double rho_long = bsm(ud(), spec()).rho();
    CHECK((rho_long > rho_short && rho_short > 0.0));
    CHECK(bsm(ud(), spec(OptionType::Call, 80.0)).rho() >
          bsm(ud(), spec(OptionType::Call, 120.0)).rho());
}

TEST_CASE("theta sign and behaviour", "[bsm][greeks]") {
    CHECK(bsm(ud(), spec(OptionType::Call)).theta() < 0.0);
    CHECK(bsm(ud(), spec(OptionType::Put)).theta() < 0.0);

    const double theta_short =
        bsm(ud(), spec(OptionType::Call, kStrike, make_datetime(2025, 2, 1))).theta();
    const double theta_long = bsm(ud(), spec()).theta();
    CHECK((theta_short < theta_long && theta_long < 0.0));

    CHECK(bsm(ud(), spec()).theta() < bsm(ud(70.0), spec()).theta());

    const double no_div = bsm(ud(), spec()).theta();
    const double with_div =
        bsm(ud(kSpot, kVol, flat_curve(kPricingDate, kMaturity, 0.03)), spec()).theta();
    CHECK(no_div < 0.0);
    CHECK(with_div < 0.0);
    CHECK_FALSE(std::abs(no_div - with_div) < 1e-8);

    const auto near = kPricingDate + std::chrono::days{2};
    const UnderlyingData u(kSpot, kVol, md(kPricingDate, near));
    CHECK(bsm(u, spec(OptionType::Call, kStrike, near)).theta() < -0.05);
}

TEST_CASE("numerical theta is zero when the bump crosses maturity", "[bsm][greeks]") {
    const auto near = kPricingDate + std::chrono::days{1};
    const UnderlyingData u(kSpot, kVol, md(kPricingDate, near));
    CHECK(bsm(u, spec(OptionType::Call, kStrike, near))
              .theta({.method = GreekCalculationMethod::Numerical, .bump = 2.0}) == 0.0);
}

TEST_CASE("delta with a dividend curve", "[bsm][greeks]") {
    const auto q = flat_curve(kPricingDate, kMaturity, 0.03);
    CHECK(bsm(ud(kSpot, kVol, q), spec(OptionType::Call)).delta() <
          bsm(ud(), spec(OptionType::Call)).delta());
    CHECK(bsm(ud(kSpot, kVol, q), spec(OptionType::Put)).delta() <
          bsm(ud(), spec(OptionType::Put)).delta());
}

// ---------------------------------------------------------------------------
// Analytical vs numerical
// ---------------------------------------------------------------------------

TEST_CASE("analytical and numerical Greeks agree", "[bsm][greeks]") {
    const auto type = GENERATE(OptionType::Call, OptionType::Put);
    const auto ov = bsm(ud(), spec(type));
    CHECK_THAT(ov.delta(kNumerical), WithinRel(ov.delta(kAnalytical), 1e-3));
    CHECK_THAT(ov.gamma(kNumerical), WithinRel(ov.gamma(kAnalytical), 1e-3));
    CHECK_THAT(ov.vega(kNumerical), WithinRel(ov.vega(kAnalytical), 1e-3));
    CHECK_THAT(ov.theta(kNumerical), WithinRel(ov.theta(kAnalytical), 0.02));
    CHECK_THAT(ov.rho(kNumerical), WithinRel(ov.rho(kAnalytical), 0.01));
}

// ---------------------------------------------------------------------------
// Golden values from the Python reference implementation
// ---------------------------------------------------------------------------

namespace {

struct BsmGoldenRow {
    const char* label;
    double spot;
    double strike;
    double rate;
    double div_yield;
    double vol;
    int maturity_days;
    double div_amount;
    int div_day;
    bool is_call;
    double pv;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
    double delta_num;
    double gamma_num;
    double vega_num;
    double theta_num;
    double rho_num;
};

#include "golden/bsm_reference.inc"

OptionValuation replay(const BsmGoldenRow& row) {
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
    const VanillaSpec s({.option_type = row.is_call ? OptionType::Call : OptionType::Put,
                         .exercise_type = ExerciseType::European,
                         .strike = row.strike,
                         .maturity = maturity});
    return {u, s, PricingMethod::Bsm};
}

}  // namespace

TEST_CASE("BSM matches the Python reference implementation", "[bsm][golden]") {
    const dp::test::WarningCapture capture;  // rate bumps can go negative on r=0 rows
    for (const auto& row : kBsmGolden) {
        INFO("row " << row.label);
        const auto ov = replay(row);
        CHECK_THAT(ov.present_value(), close_to(row.pv, 1e-11, 1e-13));
        CHECK_THAT(ov.delta(), close_to(row.delta, 1e-11, 1e-13));
        CHECK_THAT(ov.gamma(), close_to(row.gamma, 1e-11, 1e-13));
        CHECK_THAT(ov.vega(), close_to(row.vega, 1e-11, 1e-13));
        CHECK_THAT(ov.theta(), close_to(row.theta, 1e-11, 1e-13));
        CHECK_THAT(ov.rho(), close_to(row.rho, 1e-11, 1e-13));
        CHECK_THAT(ov.delta(kNumerical), close_to(row.delta_num, 1e-8, 1e-10));
        CHECK_THAT(ov.gamma(kNumerical), close_to(row.gamma_num, 1e-8, 1e-10));
        CHECK_THAT(ov.vega(kNumerical), close_to(row.vega_num, 1e-8, 1e-10));
        CHECK_THAT(ov.theta(kNumerical), close_to(row.theta_num, 1e-8, 1e-10));
        CHECK_THAT(ov.rho(kNumerical), close_to(row.rho_num, 1e-8, 1e-10));
    }
}
