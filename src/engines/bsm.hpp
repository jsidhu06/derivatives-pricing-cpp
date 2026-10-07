#pragma once

#include "engines/engine.hpp"
#include "valuation_state.hpp"

/// Internal: closed-form Black-Scholes-Merton engine for European vanillas
/// with continuous dividend yield and/or escrowed discrete dividends.
namespace dp::engines {

class BsmEngine final : public PricingEngine {
public:
    explicit BsmEngine(const detail::ValuationState& state);

    [[nodiscard]] std::string_view name() const noexcept override { return "BSM"; }
    [[nodiscard]] double present_value() const override;
    [[nodiscard]] double delta() const override;
    [[nodiscard]] double gamma() const override;
    /// Per 1 vol-point (1%) change.
    [[nodiscard]] double vega() const override;
    /// Per calendar day (annual theta / 365), with the zero rates to maturity
    /// (see `OptionValuation::theta`).
    [[nodiscard]] double theta() const override;
    /// Per 1% change in the continuously-compounded rate.
    [[nodiscard]] double rho() const override;

private:
    /// Inputs shared by the price and every Greek.
    struct Inputs {
        double spot;  ///< Dividend-adjusted spot.
        double strike;
        double volatility;
        double time_to_maturity;
        double df_r;
        double df_q;
        double d1;
        double d2;
        bool is_call;
    };

    [[nodiscard]] Inputs inputs() const;
    [[nodiscard]] double adjusted_spot() const;

    const detail::ValuationState& state_;
};

}  // namespace dp::engines
