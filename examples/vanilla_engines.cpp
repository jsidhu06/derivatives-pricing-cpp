// One contract, four engines.
//
// Prices a European call and an American put on an underlying with a
// continuous dividend yield and a term structure of interest rates, and
// compares the engines' present values and Greeks.

#include "dp/dp.hpp"

#include <format>
#include <iostream>
#include <string_view>

using namespace dp;

namespace {

void print_row(std::string_view label, const OptionValuation& valuation) {
    std::cout << std::format("  {:<12} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f}\n",
                             label,
                             valuation.present_value(),
                             valuation.delta(),
                             valuation.gamma(),
                             valuation.vega(),
                             valuation.theta(),
                             valuation.rho());
}

void print_header(std::string_view title) {
    std::cout << std::format("{}\n  {:<12} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}\n",
                             title,
                             "engine",
                             "PV",
                             "delta",
                             "gamma",
                             "vega",
                             "theta",
                             "rho");
}

}  // namespace

int main() {
    const DateTime pricing_date = make_datetime(2025, 1, 1);
    const DateTime maturity = make_datetime(2026, 1, 1);

    // Zero rates of 4.0%, 4.5% and 5.0% at 3 months, 1 year and 2 years.
    Eigen::ArrayXd times(3);
    Eigen::ArrayXd zero_rates(3);
    times << 0.25, 1.0, 2.0;
    zero_rates << 0.040, 0.045, 0.050;
    const MarketData market(pricing_date, DiscountCurve::from_zero_rates(times, zero_rates), "USD");

    // Spot 100, volatility 20%, continuous dividend yield 2%.
    const double spot = 100.0;
    const double volatility = 0.20;
    const DiscountCurve dividend_curve = DiscountCurve::flat(0.02);
    const UnderlyingData underlying(spot, volatility, market, {}, dividend_curve);

    // Monte Carlo prices from simulated paths: the same market as a process.
    const GbmProcess process(
        market,
        {.initial_value = spot, .volatility = volatility, .dividend_curve = dividend_curve},
        {.paths = 100'000, .end_date = maturity, .num_steps = 50});

    const VanillaSpec call({.option_type = OptionType::Call,
                            .exercise_type = ExerciseType::European,
                            .strike = 105.0,
                            .maturity = maturity});
    print_header("European call, K = 105");
    print_row("analytical", OptionValuation(underlying, call, PricingMethod::Analytical));
    print_row("binomial", OptionValuation(underlying, call, PricingMethod::Binomial));
    print_row("pde", OptionValuation(underlying, call, PricingMethod::PdeFd));
    print_row("monte carlo", OptionValuation(process, call, PricingMethod::MonteCarlo));

    // American exercise has no closed form: tree, PDE and Longstaff-Schwartz.
    // The Monte Carlo Greeks here are bump-and-revalue through the exercise
    // regression, so they are noisier than the tree's and the PDE's.
    const VanillaSpec put({.option_type = OptionType::Put,
                           .exercise_type = ExerciseType::American,
                           .strike = 105.0,
                           .maturity = maturity});
    print_header("\nAmerican put, K = 105");
    print_row("binomial", OptionValuation(underlying, put, PricingMethod::Binomial));
    print_row("pde", OptionValuation(underlying, put, PricingMethod::PdeFd));
    print_row("monte carlo", OptionValuation(process, put, PricingMethod::MonteCarlo));
}
