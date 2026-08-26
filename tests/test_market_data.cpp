#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/exceptions.hpp"
#include "dp/market_data.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinRel;
using dp::DayCountConvention;
using dp::MarketData;
using dp::test::message_contains;

namespace {
const auto kCurve = dp::DiscountCurve::flat(0.05, 1.0);
}

TEST_CASE("MarketData valid construction", "[market_data]") {
    const MarketData md(dp::test::kPricingDate, kCurve, "USD");
    CHECK(md.pricing_date() == dp::test::kPricingDate);
    CHECK(md.currency() == "USD");
    CHECK(md.day_count_convention() == DayCountConvention::Act365F);
    CHECK_THAT(md.discount_curve().df(1.0), WithinRel(kCurve.df(1.0), 1e-15));
}

TEST_CASE("MarketData custom day count", "[market_data]") {
    const MarketData md(dp::test::kPricingDate, kCurve, "USD", DayCountConvention::Act360);
    CHECK(md.day_count_convention() == DayCountConvention::Act360);
    CHECK_THAT(md.year_fraction_to(dp::make_datetime(2025, 4, 1)), WithinRel(0.25, 1e-12));
}

TEST_CASE("MarketData rejects empty currency", "[market_data]") {
    CHECK_THROWS_MATCHES(MarketData(dp::test::kPricingDate, kCurve, ""),
                         dp::ValidationError,
                         message_contains("currency must be a non-empty string"));
}

TEST_CASE("MarketData is copyable and independent of the source curve", "[market_data]") {
    const MarketData md(dp::test::kPricingDate, dp::DiscountCurve::flat(0.03, 2.0), "EUR");
    const MarketData copy = md;  // NOLINT(performance-unnecessary-copy-initialization)
    CHECK(copy.currency() == "EUR");
    CHECK_THAT(copy.discount_curve().df(2.0), WithinRel(md.discount_curve().df(2.0), 1e-15));
}
