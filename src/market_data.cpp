#include "dp/market_data.hpp"

#include "dp/exceptions.hpp"

#include <string>
#include <utility>

namespace dp {

MarketData::MarketData(DateTime pricing_date,
                       DiscountCurve discount_curve,
                       std::string currency,
                       DayCountConvention day_count_convention)
    : pricing_date_(pricing_date),
      discount_curve_(std::move(discount_curve)),
      currency_(std::move(currency)),
      day_count_convention_(day_count_convention) {
    if (currency_.empty()) {
        throw ValidationError("currency must be a non-empty string");
    }
}

double MarketData::year_fraction_to(DateTime date) const {
    return year_fraction(pricing_date_, date, day_count_convention_);
}

MarketData MarketData::with_pricing_date(DateTime pricing_date) const {
    return {pricing_date, discount_curve_, currency_, day_count_convention_};
}

MarketData MarketData::with_discount_curve(DiscountCurve discount_curve) const {
    return {pricing_date_, std::move(discount_curve), currency_, day_count_convention_};
}

}  // namespace dp
