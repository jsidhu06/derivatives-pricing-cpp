#include "engines/bsm.hpp"

#include "dp/dividends.hpp"
#include "dp/exceptions.hpp"
#include "dp/math/normal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dp::engines {

using math::normal_cdf;
using math::normal_pdf;

namespace {

constexpr double kDaysPerYear = 365.0;
constexpr double kZeroVolThreshold = 1e-300;

/// `(d1, d2)`. At zero vol the world is deterministic: both are +inf when the
/// forward is above the strike, -inf below, and 0 at the money.
std::pair<double, double> d_values(double spot,
                                   double strike,
                                   double time_to_maturity,
                                   double volatility,
                                   double df_r,
                                   double df_q) {
    if (time_to_maturity <= 0.0) {
        throw ValidationError("time_to_maturity must be positive");
    }
    const double forward = spot * df_q / df_r;
    const double denominator = volatility * std::sqrt(time_to_maturity);
    if (denominator < kZeroVolThreshold) {
        constexpr double kInfinity = std::numeric_limits<double>::infinity();
        if (forward > strike) {
            return {kInfinity, kInfinity};
        }
        if (forward < strike) {
            return {-kInfinity, -kInfinity};
        }
        return {0.0, 0.0};
    }
    const double numerator =
        std::log(forward / strike) + 0.5 * volatility * volatility * time_to_maturity;
    const double d1 = numerator / denominator;
    return {d1, d1 - denominator};
}

double implied_rate(double df, double time_to_maturity) noexcept {
    return -std::log(df) / time_to_maturity;
}

}  // namespace

BsmEngine::BsmEngine(const detail::ValuationState& state) : state_(state) {}

double BsmEngine::adjusted_spot() const {
    const double spot = state_.underlying.initial_value();
    const auto& dividends = state_.underlying.discrete_dividends();
    if (dividends.empty()) {
        return spot;
    }
    const DiscountCurve* q_curve = detail::curve_or_null(state_.underlying.dividend_curve());
    const double pv_divs = pv_discrete_dividends(dividends,
                                                 state_.pricing_date(),
                                                 state_.maturity(),
                                                 state_.discount_curve(),
                                                 std::nullopt,
                                                 state_.day_count_convention(),
                                                 /*include_start=*/true,
                                                 q_curve);
    return std::max(spot - pv_divs, 0.0);
}

BsmEngine::Inputs BsmEngine::inputs() const {
    const auto strike_opt = state_.strike();
    if (!strike_opt.has_value() || !state_.option_type.has_value()) {
        throw ConfigurationError("BSM engine requires a spec with a strike and a call/put type");
    }
    const double spot = adjusted_spot();
    const double strike = *strike_opt;
    const double volatility = state_.underlying.volatility();
    const double t = state_.maturity_year_fraction();
    const double df_r = state_.discount_curve().df(t);
    const auto& q_curve = state_.underlying.dividend_curve();
    const double df_q = q_curve.has_value() ? q_curve->df(t) : 1.0;
    const auto [d1, d2] = d_values(spot, strike, t, volatility, df_r, df_q);
    return {
        .spot = spot,
        .strike = strike,
        .volatility = volatility,
        .time_to_maturity = t,
        .df_r = df_r,
        .df_q = df_q,
        .d1 = d1,
        .d2 = d2,
        .is_call = *state_.option_type == OptionType::Call,
    };
}

double BsmEngine::present_value() const {
    const Inputs in = inputs();
    if (in.is_call) {
        return in.spot * in.df_q * normal_cdf(in.d1) - in.strike * in.df_r * normal_cdf(in.d2);
    }
    return in.strike * in.df_r * normal_cdf(-in.d2) - in.spot * in.df_q * normal_cdf(-in.d1);
}

double BsmEngine::delta() const {
    const Inputs in = inputs();
    if (in.is_call) {
        return in.df_q * normal_cdf(in.d1);
    }
    return in.df_q * (normal_cdf(in.d1) - 1.0);
}

double BsmEngine::gamma() const {
    const Inputs in = inputs();
    return in.df_q * normal_pdf(in.d1) / (in.spot * in.volatility * std::sqrt(in.time_to_maturity));
}

double BsmEngine::vega() const {
    const Inputs in = inputs();
    return in.spot * in.df_q * normal_pdf(in.d1) * std::sqrt(in.time_to_maturity) / 100.0;
}

double BsmEngine::theta() const {
    const Inputs in = inputs();
    const double r = implied_rate(in.df_r, in.time_to_maturity);
    const double q = implied_rate(in.df_q, in.time_to_maturity);
    const double n_prime_d1 = normal_pdf(in.d1);

    const double term1 =
        -(in.spot * in.df_q * n_prime_d1 * in.volatility / (2.0 * std::sqrt(in.time_to_maturity)));
    double theta_annual = 0.0;
    if (in.is_call) {
        const double term2 = -r * in.strike * in.df_r * normal_cdf(in.d2);
        const double term3 = q * in.spot * in.df_q * normal_cdf(in.d1);
        theta_annual = term1 + term2 + term3;
    } else {
        const double term2 = r * in.strike * in.df_r * normal_cdf(-in.d2);
        const double term3 = -q * in.spot * in.df_q * normal_cdf(-in.d1);
        theta_annual = term1 + term2 + term3;
    }
    return theta_annual / kDaysPerYear;
}

double BsmEngine::rho() const {
    const Inputs in = inputs();
    if (in.is_call) {
        return in.strike * in.time_to_maturity * in.df_r * normal_cdf(in.d2) / 100.0;
    }
    return -in.strike * in.time_to_maturity * in.df_r * normal_cdf(-in.d2) / 100.0;
}

}  // namespace dp::engines
