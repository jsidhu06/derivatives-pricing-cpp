#pragma once

#include "engines/engine.hpp"
#include "valuation_state.hpp"

/// Internal: closed-form Reiner-Rubinstein (1991) barrier engine.
namespace dp::engines {

/// Analytical barrier pricing for all 8 knock-in/knock-out combinations of
/// {up, down} x {in, out} x {call, put} under GBM, European exercise only.
/// Continuous monitoring natively; equally spaced discrete monitoring via
/// the Broadie-Glasserman-Kou continuity correction. Rebates: knock-out paid
/// at hit or at expiry, knock-in paid at expiry if never hit. Non-flat rate
/// and dividend curves enter through flat-equivalent rates at maturity.
///
/// Dispatched by `OptionValuation` for `PricingMethod::Bsm` + `BarrierSpec`.
class AnalyticalBarrierEngine final : public PricingEngine {
public:
    /// Throws `UnsupportedFeatureError` for discrete dividends.
    explicit AnalyticalBarrierEngine(const detail::ValuationState& state);

    [[nodiscard]] std::string_view name() const noexcept override { return "AnalyticalBarrier"; }
    [[nodiscard]] double present_value() const override;

    /// Barrier theta via the Black-Scholes PDE identity
    /// `theta = r V - (r - q) S delta - sigma^2 S^2 gamma / 2` (per calendar
    /// day), with delta/gamma from barrier-clamped central differences of the
    /// closed form. The identity holds in the continuation region only; the
    /// facade short-circuits inception-triggered barriers before calling it.
    [[nodiscard]] double theta() const override;

private:
    /// Flat-equivalent market inputs at maturity, shared by PV and theta.
    struct Inputs {
        double strike;
        double barrier;
        double sigma;
        double maturity;
        double df_r;
        double df_q;
        double rate;
        double yield;
    };
    [[nodiscard]] Inputs inputs() const;
    /// Barrier PV at `spot` — monitoring adjustment, sigma->0 fallback,
    /// rebate leg, and floor; no inception-trigger handling.
    [[nodiscard]] double price_at_spot(const Inputs& in, double spot) const;
    [[nodiscard]] double triggered_at_inception_value(double df_r) const;

    const detail::ValuationState& state_;
};

}  // namespace dp::engines
