// Implied volatility: recover a volatility smile from option quotes.
//
// The quotes are American puts priced on a binomial tree from a known smile,
// so the solver's output can be checked against it.

#include "dp/dp.hpp"

#include <format>
#include <iostream>
#include <string_view>

using namespace dp;

namespace {

/// A skewed smile: lower strikes trade at higher volatility.
double smile(double strike) {
    return 0.20 + 0.002 * (100.0 - strike);
}

}  // namespace

int main() {
    // The library reports recoverable conditions through `dp::warn`, which
    // logs by default. Silenced here: the last row triggers a warning, and the
    // `well_determined` flag it comes with is printed in the table.
    set_warning_handler([](std::string_view /*message*/) {});

    const DateTime pricing_date = make_datetime(2025, 1, 1);
    const DateTime maturity = make_datetime(2025, 7, 1);
    const MarketData market(pricing_date, DiscountCurve::flat(0.05), "USD");
    const double spot = 100.0;

    std::cout << "American puts, priced and inverted on a 500-step binomial tree\n";
    std::cout << std::format("  {:>6} {:>9} {:>10} {:>10} {:>9} {:>16}\n",
                             "strike",
                             "quote",
                             "true vol",
                             "implied",
                             "pricings",
                             "well determined");
    for (const double strike : {80.0, 90.0, 100.0, 110.0, 120.0}) {
        const VanillaSpec put({.option_type = OptionType::Put,
                               .exercise_type = ExerciseType::American,
                               .strike = strike,
                               .maturity = maturity});

        // The "market" quote: the model price at the smile's volatility.
        const double quote =
            OptionValuation(
                UnderlyingData(spot, smile(strike), market), put, PricingMethod::Binomial)
                .present_value();

        // The valuation passed to the solver fixes the contract, market and
        // engine; its own volatility (here 30%) is replaced during the search.
        const OptionValuation valuation(
            UnderlyingData(spot, 0.30, market), put, PricingMethod::Binomial);
        const ImpliedVolResult result = implied_volatility(quote, valuation);

        std::cout << std::format("  {:>6.0f} {:>9.4f} {:>9.2f}% {:>9.2f}% {:>9} {:>16}\n",
                                 strike,
                                 quote,
                                 100.0 * smile(strike),
                                 100.0 * result.implied_vol,
                                 result.function_evaluations,
                                 result.well_determined);
    }

    // The K = 120 put is worth exactly its intrinsic value of 20: early
    // exercise is optimal at any low volatility, so the quote does not pin the
    // volatility down. The solver flags this with `well_determined = false`
    // (and a warning) instead of returning a number that looks precise.
}
