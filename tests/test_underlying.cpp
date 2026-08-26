#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/exceptions.hpp"
#include "dp/market_data.hpp"
#include "dp/underlying.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

using namespace dp;
using Catch::Matchers::WithinRel;
using dp::test::flat_curve;
using dp::test::kMaturity;
using dp::test::kPricingDate;
using dp::test::message_contains;

namespace {

MarketData md() {
    return {kPricingDate, flat_curve(kPricingDate, kMaturity, 0.05), "USD"};
}

}  // namespace

TEST_CASE("UnderlyingData creation and pass-throughs", "[underlying]") {
    const UnderlyingData ud(100.0, 0.2, md());
    CHECK(ud.initial_value() == 100.0);
    CHECK(ud.volatility() == 0.2);
    CHECK(ud.pricing_date() == kPricingDate);
    CHECK(ud.currency() == "USD");
    CHECK(ud.day_count_convention() == DayCountConvention::Act365F);
    CHECK(ud.discrete_dividends().empty());
    CHECK_FALSE(ud.dividend_curve().has_value());
    CHECK_THAT(ud.discount_curve().df(0.5), WithinRel(std::exp(-0.025), 1e-12));
}

TEST_CASE("UnderlyingData rejects negative volatility, allows zero", "[underlying]") {
    const double vol = GENERATE(-1e-12, -0.2, -1.0);
    CHECK_THROWS_MATCHES(UnderlyingData(100.0, vol, md()),
                         ValidationError,
                         message_contains("volatility must be >= 0"));
    CHECK_NOTHROW(UnderlyingData(100.0, 0.0, md()));
}

TEST_CASE("UnderlyingData rejects NaN volatility", "[underlying]") {
    CHECK_THROWS_MATCHES(UnderlyingData(100.0, NAN, md()),
                         ValidationError,
                         message_contains("volatility must be finite"));
}

TEST_CASE("UnderlyingData rejects non-positive spot", "[underlying]") {
    const double spot = GENERATE(0.0, -1e-12, -100.0);
    CHECK_THROWS_MATCHES(UnderlyingData(spot, 0.2, md()),
                         ValidationError,
                         message_contains("initial_value must be > 0"));
}

TEST_CASE("UnderlyingData sorts dividends and validates amounts", "[underlying]") {
    const UnderlyingData ud(
        100.0, 0.2, md(), {{make_datetime(2025, 9, 1), 1.0}, {make_datetime(2025, 3, 1), 0.5}});
    REQUIRE(ud.discrete_dividends().size() == 2);
    CHECK(ud.discrete_dividends()[0].ex_date == make_datetime(2025, 3, 1));
    CHECK(ud.discrete_dividends()[1].amount == 1.0);
    CHECK_THROWS_AS(UnderlyingData(100.0, 0.2, md(), {{make_datetime(2025, 3, 1), NAN}}),
                    ValidationError);
}

TEST_CASE("UnderlyingData warns when both dividend inputs are given", "[underlying]") {
    const dp::test::WarningCapture capture;
    const UnderlyingData ud(100.0,
                            0.2,
                            md(),
                            {{make_datetime(2025, 6, 1), 1.0}},
                            flat_curve(kPricingDate, kMaturity, 0.02));
    CHECK(capture.any_contains("both dividend_curve and discrete_dividends"));
    CHECK(ud.dividend_curve().has_value());
}

TEST_CASE("UnderlyingData with_* return modified copies", "[underlying]") {
    const UnderlyingData ud(100.0, 0.2, md(), {{make_datetime(2025, 6, 1), 1.0}});

    const auto vol_bumped = ud.with_volatility(0.3);
    CHECK(vol_bumped.volatility() == 0.3);
    CHECK(ud.volatility() == 0.2);
    CHECK(vol_bumped.initial_value() == ud.initial_value());
    CHECK(vol_bumped.discrete_dividends() == ud.discrete_dividends());

    const auto spot_bumped = ud.with_initial_value(110.0);
    CHECK(spot_bumped.initial_value() == 110.0);
    CHECK(ud.initial_value() == 100.0);

    const auto later = ud.with_pricing_date(make_datetime(2025, 2, 1));
    CHECK(later.pricing_date() == make_datetime(2025, 2, 1));
    CHECK(later.currency() == "USD");
    CHECK(ud.pricing_date() == kPricingDate);

    const auto recurved = ud.with_discount_curve(DiscountCurve::flat(0.10, 1.0));
    CHECK_THAT(recurved.discount_curve().df(1.0), WithinRel(std::exp(-0.10), 1e-12));
    CHECK_THAT(ud.discount_curve().df(1.0), WithinRel(std::exp(-0.05), 1e-12));

    // Bumps still validate.
    CHECK_THROWS_AS(ud.with_initial_value(-1.0), ValidationError);
    CHECK_THROWS_AS(ud.with_volatility(-0.1), ValidationError);
}
