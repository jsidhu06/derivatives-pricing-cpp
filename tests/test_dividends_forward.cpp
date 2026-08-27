#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/dividends.hpp"
#include "dp/exceptions.hpp"
#include "dp/forward.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using dp::DividendSchedule;
using dp::ForwardInputs;
using dp::make_datetime;
using dp::test::flat_curve;
using dp::test::kMaturity;
using dp::test::kPricingDate;

// ---------------------------------------------------------------------------
// Dividends
// ---------------------------------------------------------------------------

TEST_CASE("normalize_dividends sorts by ex-date and validates amounts", "[dividends]") {
    const DividendSchedule divs{{make_datetime(2025, 9, 1), 1.0}, {make_datetime(2025, 3, 1), 0.5}};
    const auto sorted = dp::normalize_dividends(divs);
    REQUIRE(sorted.size() == 2);
    CHECK(sorted[0].ex_date == make_datetime(2025, 3, 1));
    CHECK(sorted[1].ex_date == make_datetime(2025, 9, 1));

    CHECK_THROWS_AS(dp::normalize_dividends({{make_datetime(2025, 3, 1), dp::test::kNaN}}),
                    dp::ValidationError);
}

TEST_CASE("pv_discrete_dividends discounts to the curve date", "[dividends]") {
    const double r = 0.05;
    const auto curve = flat_curve(kPricingDate, kMaturity, r);
    const auto ex = make_datetime(2025, 7, 1);
    const DividendSchedule divs{{ex, 0.5}};

    const double t_ex = dp::year_fraction(kPricingDate, ex);
    const double expected = 0.5 * std::exp(-r * t_ex);
    CHECK_THAT(dp::pv_discrete_dividends(divs, kPricingDate, kMaturity, curve),
               WithinRel(expected, 1e-12));
}

TEST_CASE("pv_discrete_dividends is empty-safe and respects the window", "[dividends]") {
    const auto curve = flat_curve(kPricingDate, kMaturity, 0.05);
    CHECK(dp::pv_discrete_dividends({}, kPricingDate, kMaturity, curve) == 0.0);

    // After maturity: excluded. Before pricing date: excluded.
    const DividendSchedule outside{{make_datetime(2024, 7, 1), 1.0},
                                   {make_datetime(2026, 7, 1), 1.0}};
    CHECK(dp::pv_discrete_dividends(outside, kPricingDate, kMaturity, curve) == 0.0);

    // Exactly at maturity: included (closed on the right).
    const DividendSchedule at_maturity{{kMaturity, 1.0}};
    CHECK(dp::pv_discrete_dividends(at_maturity, kPricingDate, kMaturity, curve) > 0.0);
}

TEST_CASE("pv_discrete_dividends include_start semantics", "[dividends]") {
    const auto curve = flat_curve(kPricingDate, kMaturity, 0.05);
    const DividendSchedule at_start{{kPricingDate, 2.0}};

    // Closed on the left: dividend at exactly start_date counts, DF = 1.
    CHECK_THAT(dp::pv_discrete_dividends(at_start,
                                         kPricingDate,
                                         kMaturity,
                                         curve,
                                         std::nullopt,
                                         dp::DayCountConvention::Act365F,
                                         true),
               WithinAbs(2.0, 1e-15));
    // Open on the left: it has already gone ex.
    CHECK(dp::pv_discrete_dividends(at_start,
                                    kPricingDate,
                                    kMaturity,
                                    curve,
                                    std::nullopt,
                                    dp::DayCountConvention::Act365F,
                                    false) == 0.0);
}

TEST_CASE("pv_discrete_dividends forward-discounts to start_date", "[dividends]") {
    const double r = 0.05;
    const auto curve = flat_curve(kPricingDate, kMaturity, r);
    const auto start = make_datetime(2025, 4, 1);
    const auto ex = make_datetime(2025, 7, 1);
    const DividendSchedule divs{{ex, 1.0}};

    const double t_start = dp::year_fraction(kPricingDate, start);
    const double t_ex = dp::year_fraction(kPricingDate, ex);
    const double expected = std::exp(-r * (t_ex - t_start));
    CHECK_THAT(dp::pv_discrete_dividends(divs, kPricingDate, kMaturity, curve, start),
               WithinRel(expected, 1e-12));
}

// ---------------------------------------------------------------------------
// Forward price
// ---------------------------------------------------------------------------

TEST_CASE("forward price with continuous dividend curve", "[forward]") {
    const double spot = 100.0;
    const double r = 0.05;
    const double q = 0.02;
    const auto r_curve = flat_curve(kPricingDate, kMaturity, r);
    const auto q_curve = flat_curve(kPricingDate, kMaturity, q);
    const double ttm = dp::year_fraction(kPricingDate, kMaturity);

    const ForwardInputs inputs{
        .spot = spot,
        .pricing_date = kPricingDate,
        .maturity = kMaturity,
        .discount_curve = &r_curve,
        .dividend_curve = &q_curve,
    };
    CHECK_THAT(dp::forward_price(inputs), WithinRel(spot * std::exp((r - q) * ttm), 1e-12));
}

TEST_CASE("forward price with discrete dividends", "[forward]") {
    const double spot = 100.0;
    const double r = 0.05;
    const auto curve = flat_curve(kPricingDate, kMaturity, r);
    const auto ex = make_datetime(2025, 7, 1);
    const double ttm = dp::year_fraction(kPricingDate, kMaturity);
    const double t_div = dp::year_fraction(kPricingDate, ex);

    const ForwardInputs inputs{
        .spot = spot,
        .pricing_date = kPricingDate,
        .maturity = kMaturity,
        .discount_curve = &curve,
        .discrete_dividends = {{ex, 0.5}},
    };
    const double pv_div = 0.5 * std::exp(-r * t_div);
    CHECK_THAT(dp::forward_price(inputs), WithinRel((spot - pv_div) * std::exp(r * ttm), 1e-12));
}

TEST_CASE("forward price warns when both dividend inputs are supplied", "[forward]") {
    const auto r_curve = flat_curve(kPricingDate, kMaturity, 0.05);
    const auto q_curve = flat_curve(kPricingDate, kMaturity, 0.02);
    const ForwardInputs inputs{
        .spot = 100.0,
        .pricing_date = kPricingDate,
        .maturity = kMaturity,
        .discount_curve = &r_curve,
        .dividend_curve = &q_curve,
        .discrete_dividends = {{make_datetime(2025, 7, 1), 0.5}},
    };
    const dp::test::WarningCapture capture;
    (void)dp::forward_price(inputs);
    CHECK(capture.any_contains("Both dividend_curve and discrete_dividends"));
}

TEST_CASE("forward price validation", "[forward]") {
    const auto curve = flat_curve(kPricingDate, kMaturity, 0.05);
    SECTION("maturity must be after pricing date") {
        const ForwardInputs inputs{.spot = 100.0,
                                   .pricing_date = kMaturity,
                                   .maturity = kPricingDate,
                                   .discount_curve = &curve};
        CHECK_THROWS_MATCHES(dp::forward_price(inputs),
                             dp::ValidationError,
                             dp::test::message_contains("maturity must be after pricing_date"));
    }
    SECTION("discount curve is required") {
        const ForwardInputs inputs{.spot = 100.0,
                                   .pricing_date = kPricingDate,
                                   .maturity = kMaturity,
                                   .discount_curve = nullptr};
        CHECK_THROWS_AS(dp::forward_price(inputs), dp::ValidationError);
    }
    SECTION("spot must be positive") {
        const ForwardInputs inputs{.spot = -1.0,
                                   .pricing_date = kPricingDate,
                                   .maturity = kMaturity,
                                   .discount_curve = &curve};
        CHECK_THROWS_AS(dp::forward_price(inputs), dp::ValidationError);
    }
}

// ---------------------------------------------------------------------------
// Put-call parity
// ---------------------------------------------------------------------------

TEST_CASE("put-call parity rhs and gap are consistent", "[forward][parity]") {
    const auto curve = flat_curve(kPricingDate, kMaturity, 0.05);
    const ForwardInputs inputs{.spot = 100.0,
                               .pricing_date = kPricingDate,
                               .maturity = kMaturity,
                               .discount_curve = &curve};
    const double strike = 100.0;
    const double rhs = dp::put_call_parity_rhs(inputs, strike);
    const double call_price = 10.0;
    const double put_price = call_price - rhs;
    CHECK_THAT(dp::put_call_parity_gap(inputs, strike, call_price, put_price),
               WithinAbs(0.0, 1e-12));
}

TEST_CASE("put-call parity rhs equals S - K e^{-rT} without dividends", "[forward][parity]") {
    const double r = 0.05;
    const double spot = 100.0;
    const double strike = 95.0;
    const auto curve = flat_curve(kPricingDate, kMaturity, r);
    const double ttm = dp::year_fraction(kPricingDate, kMaturity);
    const ForwardInputs inputs{.spot = spot,
                               .pricing_date = kPricingDate,
                               .maturity = kMaturity,
                               .discount_curve = &curve};
    CHECK_THAT(dp::put_call_parity_rhs(inputs, strike),
               WithinRel(spot - strike * std::exp(-r * ttm), 1e-12));
}
