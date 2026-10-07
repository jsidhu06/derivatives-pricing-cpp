#pragma once

#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/dividends.hpp"
#include "dp/enums.hpp"

/// Forward prices and put-call parity for equity-style underlyings.
namespace dp {

/// Inputs shared by the forward-price and parity helpers.
///
/// `dividend_curve` is an optional non-owning pointer to a continuous
/// dividend-yield curve; `discrete_dividends` may be empty. When both are
/// provided the continuous yield enters the cost-of-carry and the PV of the
/// discrete dividends, discounted at `r - q`, is subtracted from spot (a
/// warning is emitted).
struct ForwardInputs {
    double spot;
    DateTime pricing_date;
    DateTime maturity;
    const DiscountCurve* discount_curve;
    const DiscountCurve* dividend_curve = nullptr;
    // Explicit default keeps GCC's -Wmissing-field-initializers quiet when
    // the member is omitted from a designated initializer.
    DividendSchedule discrete_dividends = {};  // NOLINT(readability-redundant-member-init)
    DayCountConvention day_count_convention = DayCountConvention::Act365F;
};

/// No-arbitrage forward price `F = (S - PV(divs)) * D_q(T) / D_r(T)`, with
/// `PV` at the cost of carry (see `pv_discrete_dividends`). This is QuantLib's
/// forward and the expected terminal spot under the spot-drop model.
/// Throws `ValidationError` unless `maturity > pricing_date`.
[[nodiscard]] double forward_price(const ForwardInputs& inputs);

/// Right-hand side of European put-call parity: `C - P = D_r(T) (F - K)`.
[[nodiscard]] double put_call_parity_rhs(const ForwardInputs& inputs, double strike);

/// Parity residual `(C - P) - D_r(T) (F - K)`; zero for arbitrage-free prices.
[[nodiscard]] double put_call_parity_gap(const ForwardInputs& inputs,
                                         double strike,
                                         double call_price,
                                         double put_price);

}  // namespace dp
