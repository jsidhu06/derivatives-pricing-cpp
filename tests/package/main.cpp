// Prices one option through the public API, to check that a consumer compiles and links.
#include "dp/dp.hpp"

#include <cmath>
#include <iostream>

int main() {
    using namespace dp;
    const DateTime pricing_date = make_datetime(2025, 1, 1);
    const UnderlyingData underlying(
        100.0, 0.2, MarketData(pricing_date, DiscountCurve::flat(0.05), "USD"));
    const VanillaSpec put({.option_type = OptionType::Put,
                           .exercise_type = ExerciseType::European,
                           .strike = 100.0,
                           .maturity = make_datetime(2026, 1, 1)});

    const double analytical =
        OptionValuation(underlying, put, PricingMethod::Analytical).present_value();
    const double pde = OptionValuation(underlying, put, PricingMethod::PdeFd).present_value();
    std::cout << "derivatives_pricing " << kVersionString << ": analytical " << analytical
              << ", PDE " << pde << '\n';
    return std::abs(pde - analytical) < 1e-2 ? 0 : 1;
}
