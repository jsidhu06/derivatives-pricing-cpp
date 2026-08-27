/// Quick start: price a European call with Black-Scholes-Merton and query
/// its Greeks. Build with `cmake --build --preset debug --target quick_start`
/// and run `build/debug/examples/quick_start`.

#include "dp/dp.hpp"

#include <format>
#include <iostream>

using namespace dp;

int main() {
    const auto pricing_date = make_datetime(2025, 1, 1);
    const auto maturity = make_datetime(2026, 1, 1);

    const MarketData market(pricing_date, DiscountCurve::flat(0.05), "USD");
    const UnderlyingData underlying(100.0, 0.20, market);

    const VanillaSpec spec({
        .option_type = OptionType::Call,
        .exercise_type = ExerciseType::European,
        .strike = 100.0,
        .maturity = maturity,
        .currency = "USD",
    });

    std::cout << to_string(spec.option_type()) << " K=" << spec.strike()
              << " T=" << to_string(spec.maturity()) << " " << to_string(spec.exercise_type())
              << '\n';

    const OptionValuation val(underlying, spec, PricingMethod::Bsm);
    std::cout << val.describe() << '\n';
    std::cout << std::format("{:<8}{:>10.4f}\n", "PV:", val.present_value());
    std::cout << std::format("{:<8}{:>10.4f}\n", "Delta:", val.delta());
    std::cout << std::format("{:<8}{:>10.4f}\n", "Gamma:", val.gamma());
    std::cout << std::format("{:<8}{:>10.4f}\n", "Vega:", val.vega());
    std::cout << std::format("{:<8}{:>10.4f}\n", "Theta:", val.theta());
    std::cout << std::format("{:<8}{:>10.4f}\n", "Rho:", val.rho());

    // Bump-and-revalue instead of the closed form, with an explicit spot bump.
    const double numerical_delta =
        val.delta({.method = GreekCalculationMethod::Numerical, .bump = 0.5});
    std::cout << std::format("{:<8}{:>10.4f}  (numerical, eps=0.5)\n", "Delta:", numerical_delta);
    return 0;
}
