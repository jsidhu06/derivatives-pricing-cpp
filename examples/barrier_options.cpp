// Barrier options: monitoring style and knock-in / knock-out parity.
//
// Prices a down-and-out call under continuous and weekly monitoring with each
// engine, then checks that a knock-in and a knock-out sum to the vanilla.

#include "dp/dp.hpp"

#include <format>
#include <iostream>
#include <utility>

using namespace dp;

namespace {

const DateTime kPricingDate = make_datetime(2025, 1, 1);
const DateTime kMaturity = make_datetime(2026, 1, 1);

BarrierSpec down_call(BarrierAction action, BarrierMonitoring monitoring) {
    BarrierSpec::Params params{.option_type = OptionType::Call,
                               .exercise_type = ExerciseType::European,
                               .strike = 100.0,
                               .maturity = kMaturity,
                               .barrier = 90.0,
                               .direction = BarrierDirection::Down,
                               .action = action,
                               .monitoring = monitoring};
    if (monitoring == BarrierMonitoring::Discrete) {
        params.num_observations = 52;  // weekly fixings
    }
    return BarrierSpec(std::move(params));
}

}  // namespace

int main() {
    const MarketData market(kPricingDate, DiscountCurve::flat(0.05), "USD");
    const UnderlyingData underlying(100.0, 0.25, market);

    // Weekly monitoring misses crossings between fixings, so the knock-out is
    // worth more than under continuous monitoring.
    std::cout << "Down-and-out call, K = 100, barrier = 90\n";
    std::cout << std::format("  {:<12} {:>12} {:>12}\n", "engine", "continuous", "weekly");
    for (const auto& [label, method] : {std::pair{"analytical", PricingMethod::Analytical},
                                        std::pair{"binomial", PricingMethod::Binomial},
                                        std::pair{"pde", PricingMethod::PdeFd}}) {
        std::cout << std::format("  {:<12}", label);
        for (const auto monitoring : {BarrierMonitoring::Continuous, BarrierMonitoring::Discrete}) {
            // No engine parameters: the defaults are tuned to the contract
            // (for a barrier, to its monitoring style).
            const OptionValuation valuation(
                underlying, down_call(BarrierAction::Out, monitoring), method);
            std::cout << std::format(" {:>12.4f}", valuation.present_value());
        }
        std::cout << '\n';
    }

    // Monte Carlo: weekly fixings are checked on the simulated path; continuous
    // monitoring uses the Brownian-bridge probability of crossing between steps.
    const GbmProcess process(market,
                             {.initial_value = 100.0, .volatility = 0.25},
                             {.paths = 100'000, .end_date = kMaturity, .num_steps = 52});
    std::cout << std::format("  {:<12}", "monte carlo");
    double std_error = 0.0;
    for (const auto monitoring : {BarrierMonitoring::Continuous, BarrierMonitoring::Discrete}) {
        const OptionValuation valuation(
            process, down_call(BarrierAction::Out, monitoring), PricingMethod::MonteCarlo);
        std::cout << std::format(" {:>12.4f}", valuation.present_value());
        std_error = valuation.std_error().value_or(0.0);
    }
    std::cout << std::format("   (std error {:.2f})\n\n", std_error);

    // In-out parity: knock-in + knock-out = vanilla (no rebate).
    const auto price = [&](const OptionSpec& spec) {
        return OptionValuation(underlying, spec, PricingMethod::Analytical).present_value();
    };
    const double knock_in = price(down_call(BarrierAction::In, BarrierMonitoring::Continuous));
    const double knock_out = price(down_call(BarrierAction::Out, BarrierMonitoring::Continuous));
    const double vanilla = price(VanillaSpec({.option_type = OptionType::Call,
                                              .exercise_type = ExerciseType::European,
                                              .strike = 100.0,
                                              .maturity = kMaturity}));
    std::cout << std::format("In-out parity: {:.4f} (in) + {:.4f} (out) = {:.4f}; vanilla {:.4f}\n",
                             knock_in,
                             knock_out,
                             knock_in + knock_out,
                             vanilla);
}
