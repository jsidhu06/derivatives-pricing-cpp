#pragma once

#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"

#include <string>

/// Market data containers for valuation and simulation.
namespace dp {

/// Market context required for valuation: pricing date, risk-free curve,
/// currency and day-count basis. Immutable after construction.
class MarketData {
public:
    MarketData(DateTime pricing_date,
               DiscountCurve discount_curve,
               std::string currency,
               DayCountConvention day_count_convention = DayCountConvention::Act365F);

    [[nodiscard]] DateTime pricing_date() const noexcept { return pricing_date_; }
    [[nodiscard]] const DiscountCurve& discount_curve() const noexcept { return discount_curve_; }
    [[nodiscard]] const std::string& currency() const noexcept { return currency_; }
    [[nodiscard]] DayCountConvention day_count_convention() const noexcept {
        return day_count_convention_;
    }

    /// Year fraction from the pricing date to `date` under this market's
    /// day-count convention.
    [[nodiscard]] double year_fraction_to(DateTime date) const;

private:
    DateTime pricing_date_;
    DiscountCurve discount_curve_;
    std::string currency_;
    DayCountConvention day_count_convention_;
};

}  // namespace dp
