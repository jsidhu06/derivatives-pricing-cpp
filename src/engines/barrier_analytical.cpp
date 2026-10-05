#include "engines/barrier_analytical.hpp"

#include "dp/exceptions.hpp"
#include "dp/math/normal.hpp"

#include "detail/barrier.hpp"
#include "engines/bsm.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <variant>

namespace dp::engines {

namespace {

using math::normal_cdf;

/// Broadie-Glasserman-Kou constant for the discrete-monitoring correction.
constexpr double kBgBeta = 0.5826;

/// Continuity correction for a barrier observed at m equally spaced dates:
/// `H_adj = H * exp(±beta * sigma * sqrt(T/m))`, shifted away from the spot
/// side (+ for UP, - for DOWN), so the continuous-monitoring formula at the
/// adjusted level approximates the discretely monitored price.
double broadie_glasserman_adjustment(
    double barrier, double sigma, double maturity, int observations, BarrierDirection direction) {
    const double sign = direction == BarrierDirection::Up ? 1.0 : -1.0;
    return barrier * std::exp(sign * kBgBeta * sigma * std::sqrt(maturity / observations));
}

/// PV of a rebate paid instantly when a knock-out barrier is hit
/// (one-touch cash-or-nothing digital, Reiner-Rubinstein / Haug).
double rebate_knock_out_at_hit(double rebate,
                               double spot,
                               double barrier,
                               double rate,
                               double yield,
                               double sigma,
                               double maturity,
                               BarrierDirection direction) {
    if (rebate == 0.0) {
        return 0.0;
    }
    const double sigma2 = sigma * sigma;
    const double a = (rate - yield - sigma2 / 2.0) / sigma2;
    const double b = std::sqrt(a * a + 2.0 * rate / sigma2);
    const double sigma_sqrt_t = sigma * std::sqrt(maturity);

    const double eta = direction == BarrierDirection::Down ? 1.0 : -1.0;
    const double log_hs = std::log(barrier / spot);

    const double z1 = log_hs / sigma_sqrt_t + b * sigma_sqrt_t;
    const double z2 = log_hs / sigma_sqrt_t - b * sigma_sqrt_t;

    return rebate * (std::pow(barrier / spot, a + b) * normal_cdf(eta * z1) +
                     std::pow(barrier / spot, a - b) * normal_cdf(eta * z2));
}

/// PV of a rebate paid at expiry when a knock-out barrier is hit:
/// `R * df_r * P(tau_B <= T)` with the reflection-principle first-passage
/// probability of the drifted log-spot.
double rebate_knock_out_at_expiry(double rebate,
                                  double spot,
                                  double barrier,
                                  double rate,
                                  double yield,
                                  double sigma,
                                  double maturity,
                                  double df_r,
                                  BarrierDirection direction) {
    if (rebate == 0.0) {
        return 0.0;
    }
    const double sigma2 = sigma * sigma;
    const double mu = rate - yield - sigma2 / 2.0;
    const double sigma_sqrt_t = sigma * std::sqrt(maturity);

    const double eta = direction == BarrierDirection::Down ? 1.0 : -1.0;
    const double log_hs = std::log(barrier / spot);

    // P(barrier hit on [0, T]) = N(eta a) + (H/S)^{2 mu / sigma^2} N(eta b),
    // a = (ln(H/S) - mu T)/(sigma sqrt(T)), b = (ln(H/S) + mu T)/(sigma sqrt(T)).
    const double a = (log_hs - mu * maturity) / sigma_sqrt_t;
    const double b = (log_hs + mu * maturity) / sigma_sqrt_t;
    const double hit_prob =
        normal_cdf(eta * a) + std::pow(barrier / spot, 2.0 * mu / sigma2) * normal_cdf(eta * b);

    return rebate * df_r * hit_prob;
}

/// PV of a rebate paid at expiry if a knock-in barrier is never hit
/// (no-touch): `R * df_r * (1 - P(tau_B <= T))`.
double rebate_knock_in_at_expiry(double rebate,
                                 double spot,
                                 double barrier,
                                 double rate,
                                 double yield,
                                 double sigma,
                                 double maturity,
                                 double df_r,
                                 BarrierDirection direction) {
    if (rebate == 0.0) {
        return 0.0;
    }
    const double hit_pv = rebate_knock_out_at_expiry(
        rebate, spot, barrier, rate, yield, sigma, maturity, df_r, direction);
    return rebate * df_r - hit_pv;
}

/// The four Reiner-Rubinstein building-block terms A, B, C, D.
/// `phi`: +1 call / -1 put; `eta`: +1 down / -1 up;
/// `lam` = (r - q + sigma^2/2) / sigma^2.
struct RrTerms {
    double a;
    double b;
    double c;
    double d;
};

RrTerms barrier_formula_terms(double spot,
                              double strike,
                              double barrier,
                              double sigma,
                              double maturity,
                              double df_r,
                              double df_q,
                              double phi,
                              double eta,
                              double lam) {
    const double sigma_sqrt_t = sigma * std::sqrt(maturity);

    const double d1 = std::log(spot / strike) / sigma_sqrt_t + lam * sigma_sqrt_t;
    const double x1 = std::log(spot / barrier) / sigma_sqrt_t + lam * sigma_sqrt_t;
    const double y =
        std::log(barrier * barrier / (spot * strike)) / sigma_sqrt_t + lam * sigma_sqrt_t;
    const double y1 = std::log(barrier / spot) / sigma_sqrt_t + lam * sigma_sqrt_t;

    const double reflect_s = std::pow(barrier / spot, 2.0 * lam);
    const double reflect_k = std::pow(barrier / spot, 2.0 * lam - 2.0);

    RrTerms terms{};
    // A: vanilla-like.
    terms.a = phi * spot * df_q * normal_cdf(phi * d1) -
              phi * strike * df_r * normal_cdf(phi * (d1 - sigma_sqrt_t));
    // B: truncated at the barrier.
    terms.b = phi * spot * df_q * normal_cdf(phi * x1) -
              phi * strike * df_r * normal_cdf(phi * (x1 - sigma_sqrt_t));
    // C: reflected vanilla.
    terms.c = phi * spot * df_q * reflect_s * normal_cdf(eta * y) -
              phi * strike * df_r * reflect_k * normal_cdf(eta * (y - sigma_sqrt_t));
    // D: reflected truncated.
    terms.d = phi * spot * df_q * reflect_s * normal_cdf(eta * y1) -
              phi * strike * df_r * reflect_k * normal_cdf(eta * (y1 - sigma_sqrt_t));
    return terms;
}

/// No-rebate barrier price: the Reiner-Rubinstein 16-case A/B/C/D table.
/// May be slightly negative from floating-point cancellation; caller floors.
double barrier_price_no_rebate(double spot,
                               double strike,
                               double barrier,
                               double rate,
                               double yield,
                               double sigma,
                               double maturity,
                               double df_r,
                               double df_q,
                               OptionType option_type,
                               BarrierDirection direction,
                               BarrierAction action) {
    const double phi = option_type == OptionType::Call ? 1.0 : -1.0;
    const double eta = direction == BarrierDirection::Down ? 1.0 : -1.0;
    const double sigma2 = sigma * sigma;
    const double lam = (rate - yield + sigma2 / 2.0) / sigma2;

    const auto [a, b, c, d] =
        barrier_formula_terms(spot, strike, barrier, sigma, maturity, df_r, df_q, phi, eta, lam);

    const bool is_call = option_type == OptionType::Call;
    const bool is_down = direction == BarrierDirection::Down;
    const bool is_in = action == BarrierAction::In;
    const bool barrier_below_strike = barrier <= strike;

    if (is_down && is_in && is_call) {
        return barrier_below_strike ? c : a - b + d;
    }
    if (is_down && !is_in && is_call) {
        return barrier_below_strike ? a - c : b - d;
    }
    if (!is_down && is_in && is_call) {
        return barrier_below_strike ? a : b - c + d;
    }
    if (!is_down && !is_in && is_call) {
        return barrier_below_strike ? 0.0 : a - b + c - d;
    }
    if (is_down && is_in && !is_call) {
        return barrier_below_strike ? b - c + d : a;
    }
    if (is_down && !is_in && !is_call) {
        return barrier_below_strike ? a - b + c - d : 0.0;
    }
    if (!is_down && is_in && !is_call) {
        return barrier_below_strike ? a - b + d : c;
    }
    // Up-and-out put.
    return barrier_below_strike ? b - d : a - c;
}

/// Barrier price in the sigma -> 0 (deterministic-drift) limit: the spot
/// moves monotonically to the forward, so continuous monitoring reduces to
/// checking whether [S, S_T] crosses the barrier. Caller must have confirmed
/// the barrier is not triggered at inception. Discrete monitoring collapses
/// to continuous here (the monotone path makes intermediate checks
/// equivalent, and the BGK shift collapses to H).
double deterministic_limit_price(double spot,
                                 double strike,
                                 double barrier,
                                 double rate,
                                 double yield,
                                 double maturity,
                                 double df_r,
                                 OptionType option_type,
                                 BarrierDirection direction,
                                 BarrierAction action,
                                 double rebate,
                                 RebateTiming rebate_timing) {
    const double forward = spot * std::exp((rate - yield) * maturity);

    const bool hit = direction == BarrierDirection::Up ? forward >= barrier : forward <= barrier;

    const double intrinsic = option_type == OptionType::Call ? std::max(forward - strike, 0.0)
                                                             : std::max(strike - forward, 0.0);
    const double vanilla_pv = df_r * intrinsic;

    if (action == BarrierAction::Out) {
        if (hit) {
            if (rebate > 0.0) {
                return rebate_timing == RebateTiming::AtHit ? rebate : df_r * rebate;
            }
            return 0.0;
        }
        return vanilla_pv;
    }
    if (hit) {
        return vanilla_pv;
    }
    return rebate > 0.0 ? df_r * rebate : 0.0;
}

const BarrierSpec& barrier_spec(const detail::ValuationState& state) {
    return std::get<BarrierSpec>(state.spec);
}

}  // namespace

AnalyticalBarrierEngine::AnalyticalBarrierEngine(const detail::ValuationState& state)
    : state_(state) {
    if (!state_.underlying.discrete_dividends().empty()) {
        throw UnsupportedFeatureError(
            "Analytical barrier formula does not support discrete dividends. Use PDE_FD or "
            "MONTE_CARLO.");
    }
}

AnalyticalBarrierEngine::Inputs AnalyticalBarrierEngine::inputs() const {
    const auto& spec = barrier_spec(state_);
    Inputs in{};
    in.strike = spec.strike();
    in.barrier = spec.barrier();
    in.sigma = state_.underlying.volatility();
    in.maturity = state_.maturity_year_fraction();
    in.df_r = state_.discount_curve().df(in.maturity);
    const auto& dividend_curve = state_.underlying.dividend_curve();
    in.df_q = dividend_curve.has_value() ? dividend_curve->df(in.maturity) : 1.0;
    // Flat-equivalent rates from the curve discount factors at maturity.
    in.rate = -std::log(in.df_r) / in.maturity;
    in.yield = -std::log(in.df_q) / in.maturity;
    return in;
}

double AnalyticalBarrierEngine::price_at_spot(const Inputs& in, double spot) const {
    const auto& spec = barrier_spec(state_);

    double barrier = in.barrier;
    if (spec.monitoring() == BarrierMonitoring::Discrete) {
        if (spec.is_spot_past_barrier(spot)) {
            // Not triggered (t=0 is not an observation), but the spot is on
            // the far side of H: the closed form only covers spots on the
            // live side, so it would return a silently wrong price.
            throw UnsupportedFeatureError(
                "Analytical discrete barrier pricing requires the spot on the live side of "
                "the barrier; the spot is past it but the pricing date is not an observation "
                "date. Use BINOMIAL, PDE_FD or MONTE_CARLO.");
        }
        const std::optional<int> observations = spec.num_observations();
        if (!observations.has_value()) {
            throw UnsupportedFeatureError(
                "Analytical barrier pricing with DISCRETE monitoring requires num_observations "
                "(equally spaced). Use BINOMIAL, PDE_FD or MONTE_CARLO for explicit "
                "monitoring_dates.");
        }
        barrier = broadie_glasserman_adjustment(
            barrier, in.sigma, in.maturity, *observations, spec.direction());
    }

    // Reiner-Rubinstein divides by sigma*sqrt(T) and raises (H/S)^(2 lambda),
    // so sigma -> 0 produces division by zero or overflow; the Python
    // reference traps those and falls back to the deterministic-forward
    // closed form, which we mirror via the pre-/post-checks below.
    const auto deterministic = [&] {
        return deterministic_limit_price(spot,
                                         in.strike,
                                         barrier,
                                         in.rate,
                                         in.yield,
                                         in.maturity,
                                         in.df_r,
                                         spec.option_type(),
                                         spec.direction(),
                                         spec.action(),
                                         spec.rebate(),
                                         spec.rebate_timing());
    };
    if (in.sigma <= 0.0) {
        return deterministic();
    }
    double value = barrier_price_no_rebate(spot,
                                           in.strike,
                                           barrier,
                                           in.rate,
                                           in.yield,
                                           in.sigma,
                                           in.maturity,
                                           in.df_r,
                                           in.df_q,
                                           spec.option_type(),
                                           spec.direction(),
                                           spec.action());
    if (!std::isfinite(value)) {
        return deterministic();
    }

    if (spec.rebate() > 0.0) {
        if (spec.action() == BarrierAction::Out) {
            value += spec.rebate_timing() == RebateTiming::AtHit
                         ? rebate_knock_out_at_hit(spec.rebate(),
                                                   spot,
                                                   barrier,
                                                   in.rate,
                                                   in.yield,
                                                   in.sigma,
                                                   in.maturity,
                                                   spec.direction())
                         : rebate_knock_out_at_expiry(spec.rebate(),
                                                      spot,
                                                      barrier,
                                                      in.rate,
                                                      in.yield,
                                                      in.sigma,
                                                      in.maturity,
                                                      in.df_r,
                                                      spec.direction());
        } else {
            // Knock-in: rebate paid at expiry if the barrier is never hit.
            value += rebate_knock_in_at_expiry(spec.rebate(),
                                               spot,
                                               barrier,
                                               in.rate,
                                               in.yield,
                                               in.sigma,
                                               in.maturity,
                                               in.df_r,
                                               spec.direction());
        }
    }

    return std::max(value, 0.0);
}

double AnalyticalBarrierEngine::triggered_at_inception_value(double df_r) const {
    const auto& spec = barrier_spec(state_);
    if (spec.action() == BarrierAction::Out) {
        // Knocked out immediately: PV of the rebate only.
        if (spec.rebate() == 0.0) {
            return 0.0;
        }
        return spec.rebate_timing() == RebateTiming::AtHit ? spec.rebate() : spec.rebate() * df_r;
    }

    // Knock-in already activated: price the vanilla equivalent via BSM.
    const VanillaSpec vanilla = detail::vanilla_equivalent_spec(spec);
    const detail::ValuationState vanilla_state{.underlying = state_.underlying,
                                               .spec = OptionSpec{vanilla},
                                               .pricing_method = PricingMethod::Bsm,
                                               .params = std::nullopt,
                                               .currency = state_.currency,
                                               .option_type = spec.option_type()};
    return BsmEngine(vanilla_state).present_value();
}

double AnalyticalBarrierEngine::present_value() const {
    const Inputs in = inputs();
    if (detail::barrier_triggered_at_inception(state_)) {
        return triggered_at_inception_value(in.df_r);
    }
    return price_at_spot(in, state_.underlying.initial_value());
}

}  // namespace dp::engines
