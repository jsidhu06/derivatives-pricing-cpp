#include "dp/underlying.hpp"

#include "dp/diagnostics.hpp"

#include "detail/validate.hpp"

#include <utility>

namespace dp {

UnderlyingData::UnderlyingData(double initial_value,
                               double volatility,
                               MarketData market_data,
                               DividendSchedule discrete_dividends,
                               std::optional<DiscountCurve> dividend_curve)
    : initial_value_(initial_value),
      volatility_(volatility),
      market_data_(std::move(market_data)),
      discrete_dividends_(normalize_dividends(std::move(discrete_dividends))),
      dividend_curve_(std::move(dividend_curve)) {
    detail::require_positive(initial_value_, "UnderlyingData.initial_value");
    detail::require_non_negative(volatility_, "UnderlyingData.volatility");

    if (dividend_curve_.has_value() && !discrete_dividends_.empty()) {
        warn(
            "UnderlyingData: both dividend_curve and discrete_dividends provided. The "
            "continuous yield will enter the drift and discrete dividends will be "
            "subtracted at each ex-date.");
    }
}

UnderlyingData UnderlyingData::with_initial_value(double initial_value) const {
    return {initial_value, volatility_, market_data_, discrete_dividends_, dividend_curve_};
}

UnderlyingData UnderlyingData::with_volatility(double volatility) const {
    return {initial_value_, volatility, market_data_, discrete_dividends_, dividend_curve_};
}

UnderlyingData UnderlyingData::with_market_data(MarketData market_data) const {
    return {
        initial_value_, volatility_, std::move(market_data), discrete_dividends_, dividend_curve_};
}

UnderlyingData UnderlyingData::with_pricing_date(DateTime pricing_date) const {
    return with_market_data(market_data_.with_pricing_date(pricing_date));
}

UnderlyingData UnderlyingData::with_discount_curve(DiscountCurve discount_curve) const {
    return with_market_data(market_data_.with_discount_curve(std::move(discount_curve)));
}

}  // namespace dp
