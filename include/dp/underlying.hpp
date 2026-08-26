#pragma once

#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/dividends.hpp"
#include "dp/enums.hpp"
#include "dp/market_data.hpp"

#include <optional>
#include <string>

/// Underlying asset description for deterministic engines.
namespace dp {

/// Black-Scholes-Merton (GBM) parameterisation of an underlying: spot,
/// volatility, market context and optional dividends. Used by the analytical,
/// tree and PDE engines. Immutable; `with_*` return modified copies (used by
/// bump-and-revalue Greeks).
class UnderlyingData {
public:
    UnderlyingData(double initial_value,
                   double volatility,
                   MarketData market_data,
                   DividendSchedule discrete_dividends = {},
                   std::optional<DiscountCurve> dividend_curve = std::nullopt);

    [[nodiscard]] double initial_value() const noexcept { return initial_value_; }
    [[nodiscard]] double volatility() const noexcept { return volatility_; }
    [[nodiscard]] const MarketData& market_data() const noexcept { return market_data_; }
    /// Cash dividends sorted by ex-date (empty when none).
    [[nodiscard]] const DividendSchedule& discrete_dividends() const noexcept {
        return discrete_dividends_;
    }
    [[nodiscard]] const std::optional<DiscountCurve>& dividend_curve() const noexcept {
        return dividend_curve_;
    }

    // Convenience pass-throughs to `market_data()`.
    [[nodiscard]] DateTime pricing_date() const noexcept { return market_data_.pricing_date(); }
    [[nodiscard]] const DiscountCurve& discount_curve() const noexcept {
        return market_data_.discount_curve();
    }
    [[nodiscard]] const std::string& currency() const noexcept { return market_data_.currency(); }
    [[nodiscard]] DayCountConvention day_count_convention() const noexcept {
        return market_data_.day_count_convention();
    }

    [[nodiscard]] UnderlyingData with_initial_value(double initial_value) const;
    [[nodiscard]] UnderlyingData with_volatility(double volatility) const;
    [[nodiscard]] UnderlyingData with_market_data(MarketData market_data) const;
    [[nodiscard]] UnderlyingData with_pricing_date(DateTime pricing_date) const;
    [[nodiscard]] UnderlyingData with_discount_curve(DiscountCurve discount_curve) const;

private:
    double initial_value_;
    double volatility_;
    MarketData market_data_;
    DividendSchedule discrete_dividends_;
    std::optional<DiscountCurve> dividend_curve_;
};

}  // namespace dp
