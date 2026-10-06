#include "dp/implied_volatility.hpp"

#include "dp/contracts.hpp"
#include "dp/dividends.hpp"
#include "dp/exceptions.hpp"
#include "dp/params.hpp"
#include "dp/underlying.hpp"

#include "detail/log.hpp"

#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/toms748_solve.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <utility>
#include <variant>

namespace dp {

void ImpliedVolParams::validate() const {
    if (!std::isfinite(vol_lower) || !std::isfinite(vol_upper) || vol_lower <= 0.0 ||
        vol_upper <= 0.0 || vol_lower >= vol_upper) {
        throw ValidationError(std::format(
            "vol bounds must be positive and satisfy vol_lower < vol_upper, got ({}, {})",
            vol_lower,
            vol_upper));
    }
    if (!std::isfinite(tol) || tol <= 0.0) {
        throw ValidationError(std::format("tol must be positive and finite, got {}", tol));
    }
    if (max_iter < 1) {
        throw ValidationError(std::format("max_iter must be >= 1, got {}", max_iter));
    }
    if (initial_vol.has_value() && (!std::isfinite(*initial_vol) || *initial_vol <= 0.0)) {
        throw ValidationError(
            std::format("initial_vol must be positive and finite, got {}", *initial_vol));
    }
}

namespace {

/// Feasibility checks shared by every solver: contract, engine, underlying.
const UnderlyingData& require_supported(const OptionValuation& valuation) {
    if (std::holds_alternative<PayoffSpec>(valuation.spec())) {
        throw UnsupportedFeatureError(
            "Implied volatility is not supported for PayoffSpec: P(sigma) monotonicity depends "
            "on the user-supplied payoff function and cannot be guaranteed in general.");
    }
    if (std::holds_alternative<BarrierSpec>(valuation.spec())) {
        throw UnsupportedFeatureError(
            "Implied volatility is not supported for barrier specs: P(sigma) is not "
            "guaranteed to be monotonic in sigma.");
    }
    switch (valuation.pricing_method()) {
        case PricingMethod::Analytical:
            if (valuation.exercise_type() != ExerciseType::European) {
                throw UnsupportedFeatureError(
                    "ANALYTICAL implied volatility supports European options only.");
            }
            break;
        case PricingMethod::Binomial:
        case PricingMethod::PdeFd:
            break;
        case PricingMethod::MonteCarlo:
            throw UnsupportedFeatureError(
                "Implied volatility supports ANALYTICAL, BINOMIAL, or PDE_FD pricing "
                "methods only.");
    }
    const auto* underlying = std::get_if<UnderlyingData>(&valuation.underlying());
    if (underlying == nullptr) {
        throw ConfigurationError(
            "Implied volatility requires UnderlyingData (not a path process).");
    }
    return *underlying;
}

/// Smallest volatility the CRR tree accepts: below |r - q| sqrt(dt) the
/// no-arbitrage condition d < exp((r - q) dt) < u fails (1% safety margin).
double binomial_vol_floor(const OptionValuation& valuation, const UnderlyingData& underlying) {
    const auto& params = valuation.params();
    const BinomialParams binomial =
        params.has_value() && std::holds_alternative<BinomialParams>(*params)
            ? std::get<BinomialParams>(*params)
            : BinomialParams{};
    const double ttm = valuation.maturity_year_fraction();
    const double dt = ttm / std::max(binomial.num_steps, 1);
    const double r = -std::log(valuation.discount_curve().df(ttm)) / ttm;
    double q = 0.0;
    if (const auto& dividend_curve = underlying.dividend_curve(); dividend_curve.has_value()) {
        q = -std::log(dividend_curve->df(ttm)) / ttm;
    }
    return std::abs(r - q) * std::sqrt(std::max(dt, 1.0e-12)) * 1.01;
}

/// No-arbitrage price interval. Americans: intrinsic to spot (call) or
/// strike (put). Europeans: discounted forward intrinsic to the discounted
/// underlying (call) or discounted strike (put), with discrete dividends
/// stripped from spot.
std::pair<double, double> price_bounds(const OptionValuation& valuation,
                                       const UnderlyingData& underlying) {
    const double ttm = valuation.maturity_year_fraction();
    if (ttm <= 0.0) {
        throw ValidationError("Option maturity must be after pricing date.");
    }
    const double strike = valuation.strike().value_or(0.0);
    const bool is_call = valuation.option_type() == OptionType::Call;

    if (valuation.exercise_type() == ExerciseType::American) {
        const double spot = underlying.initial_value();
        if (is_call) {
            return {std::max(0.0, spot - strike), spot};
        }
        return {std::max(0.0, strike - spot), strike};
    }

    double spot = underlying.initial_value();
    if (!underlying.discrete_dividends().empty()) {
        const double pv_dividends = pv_discrete_dividends(underlying.discrete_dividends(),
                                                          valuation.pricing_date(),
                                                          valuation.maturity(),
                                                          valuation.discount_curve(),
                                                          std::nullopt,
                                                          valuation.day_count_convention());
        spot = std::max(spot - pv_dividends, 0.0);
    }
    double df_q = 1.0;
    if (const auto& dividend_curve = underlying.dividend_curve(); dividend_curve.has_value()) {
        df_q = dividend_curve->df(ttm);
    }
    const double df_r = valuation.discount_curve().df(ttm);
    if (is_call) {
        return {std::max(0.0, spot * df_q - strike * df_r), spot * df_q};
    }
    return {std::max(0.0, strike * df_r - spot * df_q), strike * df_r};
}

/// Starting volatility for non-BSM Newton solves: 20% plus the extrinsic
/// value scaled to a vol-like quantity, clipped to the search interval.
double heuristic_initial_vol(const OptionValuation& valuation,
                             const UnderlyingData& underlying,
                             double target_price,
                             double low,
                             double high) {
    const double spot = underlying.initial_value();
    const double strike = valuation.strike().value_or(0.0);
    const double time_sqrt = std::sqrt(std::max(valuation.maturity_year_fraction(), 1.0e-8));
    const double intrinsic = valuation.option_type() == OptionType::Call
                                 ? std::max(spot - strike, 0.0)
                                 : std::max(strike - spot, 0.0);
    const double extrinsic = std::max(target_price - intrinsic, 0.0);
    const double scale = std::max({spot, strike, 1.0});
    return std::clamp(0.2 + 0.8 * (extrinsic / scale) / time_sqrt, low, high);
}

/// Bracketing interval and the residuals at its ends.
struct Bracket {
    double low;
    double high;
    double f_low;
    double f_high;
};

/// Widen [low, high] until the residual changes sign: halve `low` (floored
/// at 1e-12) while the price there is still too high, double `high` while
/// it is still too low, at most `max_expansions` times each.
template <class Residual>
Bracket bracket_volatility(const Residual& residual, double low, double high) {
    constexpr int kMaxExpansions = 6;
    Bracket b{.low = low, .high = high, .f_low = residual(low), .f_high = residual(high)};
    for (int i = 0; i < kMaxExpansions && b.f_low > 0.0; ++i) {
        b.low = std::max(b.low / 2.0, 1.0e-12);
        b.f_low = residual(b.low);
    }
    for (int i = 0; i < kMaxExpansions && b.f_high < 0.0; ++i) {
        b.high *= 2.0;
        b.f_high = residual(b.high);
    }
    return b;
}

/// Midpoint of a bracketing solver's final interval.
double bracket_root(const std::pair<double, double>& interval) {
    return interval.first + (interval.second - interval.first) / 2.0;
}

}  // namespace

ImpliedVolResult implied_volatility(double target_price,
                                    const OptionValuation& valuation,
                                    const ImpliedVolParams& params) {
    const UnderlyingData& underlying = require_supported(valuation);
    if (!std::isfinite(target_price)) {
        throw ValidationError("target_price must be finite");
    }
    if (target_price < 0.0) {
        throw ValidationError("target_price must be non-negative");
    }
    params.validate();

    const PricingMethod method = valuation.pricing_method();
    const double tol = params.tol;
    double low = params.vol_lower;
    const double high = params.vol_upper;
    if (method == PricingMethod::Binomial) {
        low = std::max(low, binomial_vol_floor(valuation, underlying));
    }

    const auto [min_price, max_price] = price_bounds(valuation, underlying);
    if (target_price < min_price - tol || target_price > max_price + tol) {
        throw ValidationError(
            "target_price is outside no-arbitrage bounds for the provided inputs");
    }

    const auto revalue = [&](double vol) {
        return OptionValuation(
            underlying.with_volatility(vol), valuation.spec(), method, valuation.params());
    };
    // A price within `tol` of the target counts as an exact root: every
    // Boost solver stops on a zero residual, which gives them the same
    // price-tolerance semantics as the Python reference.
    const auto accept_within_tol = [tol](double diff) {
        return std::abs(diff) <= tol ? 0.0 : diff;
    };
    const auto residual = [&](double vol) {
        return accept_within_tol(revalue(vol).present_value() - target_price);
    };
    const auto bracket_converged = [tol](double a, double b) { return std::abs(b - a) <= tol; };

    double initial = 0.0;
    if (params.initial_vol.has_value()) {
        initial = *params.initial_vol;
    } else if (method == PricingMethod::Analytical) {
        initial = underlying.volatility() > 0.0 ? underlying.volatility() : 0.2;
    } else {
        initial = heuristic_initial_vol(valuation, underlying, target_price, low, high);
    }
    initial = std::clamp(initial, low, high);

    const Bracket bracket = bracket_volatility(residual, low, high);
    if (bracket.f_low > 0.0 || bracket.f_high < 0.0) {
        throw ConvergenceError("Price not bracketed by vol_bounds; adjust bounds.");
    }
    initial = std::clamp(initial, bracket.low, bracket.high);

    const auto bisection = [&] {
        auto iterations = static_cast<std::uintmax_t>(params.max_iter);
        const auto interval = boost::math::tools::bisect(
            residual, bracket.low, bracket.high, bracket_converged, iterations);
        return ImpliedVolResult{.implied_vol = bracket_root(interval),
                                .iterations = static_cast<int>(iterations),
                                .converged = bracket_converged(interval.first, interval.second)};
    };

    ImpliedVolResult result{};
    switch (params.method) {
        case ImpliedVolMethod::NewtonRaphson: {
            // Newton on (price residual, dPrice/dsigma): analytical vega for
            // BSM, bump-and-revalue otherwise, rescaled from per-1% to per-1.0.
            // Empty when Newton fails to reprice within tol.
            const auto newton = [&]() -> std::optional<ImpliedVolResult> {
                const GreekCalculationMethod vega_method = method == PricingMethod::Analytical
                                                               ? GreekCalculationMethod::Analytical
                                                               : GreekCalculationMethod::Numerical;
                // Remember the last evaluation: Boost returns the point it
                // last priced when it stops on a zero (within-tol) residual.
                double last_vol = std::numeric_limits<double>::quiet_NaN();
                double last_residual = std::numeric_limits<double>::quiet_NaN();
                const auto residual_and_vega = [&](double vol) {
                    const OptionValuation at_vol = revalue(vol);
                    last_vol = vol;
                    last_residual = accept_within_tol(at_vol.present_value() - target_price);
                    return std::pair{last_residual, at_vol.vega({.method = vega_method}) * 100.0};
                };
                // Tight step tolerance: the within-tol price exit is the real
                // stopping rule; this only bounds work near a converged root.
                constexpr int kNewtonDigits = 40;
                auto iterations = static_cast<std::uintmax_t>(params.max_iter);
                try {
                    const double vol = boost::math::tools::newton_raphson_iterate(residual_and_vega,
                                                                                  initial,
                                                                                  bracket.low,
                                                                                  bracket.high,
                                                                                  kNewtonDigits,
                                                                                  iterations);
                    // Converged only if the result reprices within tol — Boost
                    // also stops on step size or when pinned to a bracket end.
                    const bool converged =
                        vol == last_vol ? last_residual == 0.0 : residual(vol) == 0.0;
                    if (!converged) {
                        return std::nullopt;
                    }
                    return ImpliedVolResult{.implied_vol = vol,
                                            .iterations = static_cast<int>(iterations),
                                            .converged = true};
                } catch (const boost::math::evaluation_error&) {
                    // Boost lost its bracket (e.g. a flat or noisy numerical
                    // vega); the bisection fallback is robust to that.
                    return std::nullopt;
                }
            };
            if (auto newton_result = newton(); newton_result.has_value()) {
                result = *newton_result;
            } else {
                result = bisection();
            }
            break;
        }
        case ImpliedVolMethod::Bisection:
            result = bisection();
            break;
        case ImpliedVolMethod::Brent: {
            auto iterations = static_cast<std::uintmax_t>(params.max_iter);
            const auto interval = boost::math::tools::toms748_solve(residual,
                                                                    bracket.low,
                                                                    bracket.high,
                                                                    bracket.f_low,
                                                                    bracket.f_high,
                                                                    bracket_converged,
                                                                    iterations);
            result = {.implied_vol = bracket_root(interval),
                      .iterations = static_cast<int>(iterations),
                      .converged = bracket_converged(interval.first, interval.second)};
            break;
        }
    }

    detail::logger()->debug("Implied vol method={} converged={} iterations={} implied_vol={:.6g}",
                            to_string(params.method),
                            result.converged,
                            result.iterations,
                            result.implied_vol);
    return result;
}

}  // namespace dp
