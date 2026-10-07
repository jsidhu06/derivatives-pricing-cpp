#include "dp/forward.hpp"

#include "dp/diagnostics.hpp"
#include "dp/exceptions.hpp"

#include "detail/validate.hpp"

namespace dp {

namespace {

double maturity_year_fraction(const ForwardInputs& inputs) {
    const double t =
        year_fraction(inputs.pricing_date, inputs.maturity, inputs.day_count_convention);
    if (t <= 0.0) {
        throw ValidationError("maturity must be after pricing_date");
    }
    return t;
}

}  // namespace

double forward_price(const ForwardInputs& inputs) {
    if (inputs.discount_curve == nullptr) {
        throw ValidationError("discount_curve must be provided");
    }
    detail::require_positive(inputs.spot, "spot");

    if (inputs.dividend_curve != nullptr && !inputs.discrete_dividends.empty()) {
        warn(
            "Both dividend_curve and discrete_dividends provided to forward_price. "
            "The continuous yield will enter the cost-of-carry and discrete "
            "dividends will be subtracted from spot.");
    }

    const double t = maturity_year_fraction(inputs);

    const double pv_divs = pv_discrete_dividends(inputs.discrete_dividends,
                                                 inputs.pricing_date,
                                                 inputs.maturity,
                                                 *inputs.discount_curve,
                                                 std::nullopt,
                                                 inputs.day_count_convention,
                                                 /*include_start=*/true,
                                                 inputs.dividend_curve);

    const double df_q = inputs.dividend_curve == nullptr ? 1.0 : inputs.dividend_curve->df(t);
    const double df_r = inputs.discount_curve->df(t);

    return (inputs.spot - pv_divs) * df_q / df_r;
}

double put_call_parity_rhs(const ForwardInputs& inputs, double strike) {
    detail::require_non_negative(strike, "strike");
    const double fwd = forward_price(inputs);
    const double t = maturity_year_fraction(inputs);
    const double df_r = inputs.discount_curve->df(t);
    return df_r * (fwd - strike);
}

double put_call_parity_gap(const ForwardInputs& inputs,
                           double strike,
                           double call_price,
                           double put_price) {
    return call_price - put_price - put_call_parity_rhs(inputs, strike);
}

}  // namespace dp
