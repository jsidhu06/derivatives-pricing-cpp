// Monte Carlo: standard errors, American exercise and jump diffusion.

#include "dp/dp.hpp"

#include <format>
#include <iostream>

using namespace dp;

int main() {
    const DateTime pricing_date = make_datetime(2025, 1, 1);
    const DateTime maturity = make_datetime(2026, 1, 1);
    const MarketData market(pricing_date, DiscountCurve::flat(0.05), "USD");
    const UnderlyingData underlying(100.0, 0.20, market);

    const VanillaSpec call({.option_type = OptionType::Call,
                            .exercise_type = ExerciseType::European,
                            .strike = 100.0,
                            .maturity = maturity});
    const VanillaSpec american_put({.option_type = OptionType::Put,
                                    .exercise_type = ExerciseType::American,
                                    .strike = 100.0,
                                    .maturity = maturity});

    // 1. The standard error shrinks like 1 / sqrt(paths).
    const double exact =
        OptionValuation(underlying, call, PricingMethod::Analytical).present_value();
    std::cout << std::format("European call, analytical PV {:.4f}\n", exact);
    std::cout << std::format("  {:>9} {:>9} {:>10} {:>9}\n", "paths", "PV", "std error", "error");
    for (const int paths : {1'000, 10'000, 100'000}) {
        // Plain sampling, so the standard error is the textbook one. The
        // defaults (antithetic paths, moment matching) reduce it further.
        const GbmProcess process(market,
                                 {.initial_value = 100.0, .volatility = 0.20},
                                 {.paths = paths,
                                  .end_date = maturity,
                                  .num_steps = 1,
                                  .antithetic = false,
                                  .moment_matching = false});
        const OptionValuation mc(process, call, PricingMethod::MonteCarlo);
        std::cout << std::format("  {:>9} {:>9.4f} {:>10.4f} {:>9.4f}\n",
                                 paths,
                                 mc.present_value(),
                                 mc.std_error().value_or(0.0),
                                 mc.present_value() - exact);
    }

    // 2. American put by Longstaff-Schwartz, against a binomial tree. The
    //    simulation can exercise on its 50 time steps only, and with an
    //    estimated rule, so it comes out slightly below the tree.
    const GbmProcess process(market,
                             {.initial_value = 100.0, .volatility = 0.20},
                             {.paths = 100'000, .end_date = maturity, .num_steps = 50});
    const OptionValuation lsm(process, american_put, PricingMethod::MonteCarlo);
    const double tree =
        OptionValuation(
            underlying, american_put, PricingMethod::Binomial, BinomialParams{.num_steps = 2000})
            .present_value();
    std::cout << std::format(
        "\nAmerican put: Longstaff-Schwartz {:.4f} (std error {:.4f}), "
        "binomial {:.4f}\n",
        lsm.present_value(),
        lsm.std_error().value_or(0.0),
        tree);

    // 3. Merton jump diffusion: on average one jump every two years, with a
    //    mean log-jump of -10%. The jumps add variance on top of the
    //    diffusion, so the call is worth more than the Black-Scholes price at
    //    the same volatility.
    const JdProcess jumps(market,
                          {.initial_value = 100.0,
                           .volatility = 0.20,
                           .jump_intensity = 0.5,
                           .mu = -0.10,
                           .delta = 0.15},
                          {.paths = 100'000, .end_date = maturity, .num_steps = 50});
    const OptionValuation jump_call(jumps, call, PricingMethod::MonteCarlo);
    std::cout << std::format(
        "\nEuropean call under jump diffusion: {:.4f} (std error {:.4f}); "
        "Black-Scholes {:.4f}\n",
        jump_call.present_value(),
        jump_call.std_error().value_or(0.0),
        exact);
}
