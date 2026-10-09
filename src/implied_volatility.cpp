#include "dp/implied_volatility.hpp"

#include "dp/contracts.hpp"
#include "dp/diagnostics.hpp"
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
    if (min_vega.has_value() && !(*min_vega >= 0.0)) {
        throw ValidationError(std::format("min_vega must be >= 0, got {}", *min_vega));
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

/// Smallest volatility the CRR tree accepts (with a 1% margin). The tree
/// needs d < exp((r_i - q_i) dt) < u at every step i, with u = 1/d =
/// exp(sigma sqrt(dt)), which holds exactly when sigma > |r_i - q_i| sqrt(dt)
/// for all i. The bound therefore uses the per-step forward rates the tree
/// uses: on a term curve some steps need more than the average rate suggests.
double binomial_vol_floor(const OptionValuation& valuation, const UnderlyingData& underlying) {
    const auto& params = valuation.params();
    const BinomialParams binomial =
        params.has_value() && std::holds_alternative<BinomialParams>(*params)
            ? std::get<BinomialParams>(*params)
            : BinomialParams{};
    const int steps = std::max(binomial.num_steps, 1);
    const double ttm = valuation.maturity_year_fraction();
    const Eigen::ArrayXd grid = Eigen::ArrayXd::LinSpaced(steps + 1, 0.0, ttm);
    Eigen::ArrayXd net = valuation.discount_curve().step_forward_rates(grid);
    if (const auto& dividend_curve = underlying.dividend_curve(); dividend_curve.has_value()) {
        net -= dividend_curve->step_forward_rates(grid);
    }
    return net.abs().maxCoeff() * std::sqrt(ttm / steps) * 1.01;
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

    const auto& q_curve = underlying.dividend_curve();
    double spot = underlying.initial_value();
    if (!underlying.discrete_dividends().empty()) {
        const double pv_dividends =
            pv_discrete_dividends(underlying.discrete_dividends(),
                                  valuation.pricing_date(),
                                  valuation.maturity(),
                                  valuation.discount_curve(),
                                  std::nullopt,
                                  valuation.day_count_convention(),
                                  /*include_start=*/true,
                                  q_curve.has_value() ? &*q_curve : nullptr);
        spot = std::max(spot - pv_dividends, 0.0);
    }
    const double df_q = q_curve.has_value() ? q_curve->df(ttm) : 1.0;
    const double df_r = valuation.discount_curve().df(ttm);
    if (is_call) {
        return {std::max(0.0, spot * df_q - strike * df_r), spot * df_q};
    }
    return {std::max(0.0, strike * df_r - spot * df_q), strike * df_r};
}

/// The search interval [vol_lower, vol_upper], raised for the CRR tree to
/// the smallest volatility it accepts.
std::pair<double, double> search_interval(const ImpliedVolParams& params,
                                          const OptionValuation& valuation,
                                          const UnderlyingData& underlying) {
    double low = params.vol_lower;
    if (valuation.pricing_method() == PricingMethod::Binomial) {
        low = std::max(low, binomial_vol_floor(valuation, underlying));
    }
    return {low, params.vol_upper};
}

/// Throws unless the quote lies within the no-arbitrage price bounds.
void require_within_bounds(double target_price,
                           const OptionValuation& valuation,
                           const UnderlyingData& underlying) {
    const auto [min_price, max_price] = price_bounds(valuation, underlying);
    // Rounding slack only: a quote outside the bounds has no implied vol.
    const double bound_slack = 1.0e-12 * std::max(1.0, max_price);
    if (target_price < min_price - bound_slack || target_price > max_price + bound_slack) {
        throw ValidationError(
            "target_price is outside no-arbitrage bounds for the provided inputs");
    }
}

/// Starting volatility for non-BSM Newton solves: 20% plus the extrinsic
/// value scaled to a vol-like quantity.
double heuristic_initial_vol(const OptionValuation& valuation,
                             const UnderlyingData& underlying,
                             double target_price) {
    const double spot = underlying.initial_value();
    const double strike = valuation.strike().value_or(0.0);
    const double time_sqrt = std::sqrt(std::max(valuation.maturity_year_fraction(), 1.0e-8));
    const double intrinsic = valuation.option_type() == OptionType::Call
                                 ? std::max(spot - strike, 0.0)
                                 : std::max(strike - spot, 0.0);
    const double extrinsic = std::max(target_price - intrinsic, 0.0);
    const double scale = std::max({spot, strike, 1.0});
    return 0.2 + 0.8 * (extrinsic / scale) / time_sqrt;
}

/// Newton's starting volatility: the caller's `initial_vol`, else the
/// valuation's own volatility for BSM, else the heuristic; clipped to the
/// search interval.
double initial_vol(const ImpliedVolParams& params,
                   const OptionValuation& valuation,
                   const UnderlyingData& underlying,
                   double target_price,
                   double low,
                   double high) {
    double initial = 0.0;
    if (params.initial_vol.has_value()) {
        initial = *params.initial_vol;
    } else if (valuation.pricing_method() == PricingMethod::Analytical) {
        initial = underlying.volatility() > 0.0 ? underlying.volatility() : 0.2;
    } else {
        initial = heuristic_initial_vol(valuation, underlying, target_price);
    }
    return std::clamp(initial, low, high);
}

/// The function every solver finds the root of: P(vol) - target, the model
/// price at `vol` less the quote. Counts every model pricing (numerical-vega
/// bumps included), reported as `function_evaluations`.
///
/// Not copyable: Boost's solvers take their functor by value, and a copy
/// would count pricings into its own counter. Pass it through a lambda that
/// captures it by reference.
class PriceResidual {
public:
    PriceResidual(const OptionValuation& valuation,
                  const UnderlyingData& underlying,
                  double target_price)
        : valuation_(&valuation),
          underlying_(&underlying),
          target_price_(target_price),
          vega_method_(valuation.pricing_method() == PricingMethod::Analytical
                           ? GreekCalculationMethod::Analytical
                           : GreekCalculationMethod::Numerical) {}
    PriceResidual(const PriceResidual&) = delete;
    PriceResidual& operator=(const PriceResidual&) = delete;
    PriceResidual(PriceResidual&&) = delete;
    PriceResidual& operator=(PriceResidual&&) = delete;
    ~PriceResidual() = default;

    /// The valuation with its volatility replaced by `vol`.
    [[nodiscard]] OptionValuation revalue(double vol) const {
        return {underlying_->with_volatility(vol),
                valuation_->spec(),
                valuation_->pricing_method(),
                valuation_->params()};
    }

    /// P(vol) - target.
    double operator()(double vol) {
        ++pricings_;
        return revalue(vol).present_value() - target_price_;
    }

    /// dP/dsigma per 1.0 of vol at an already-built valuation: analytical
    /// for BSM, else a central bump (two more pricings).
    double vega(const OptionValuation& at_vol) {
        pricings_ += vega_method_ == GreekCalculationMethod::Numerical ? 2 : 0;
        return at_vol.vega({.method = vega_method_}) * 100.0;  // vega is per 1% vol
    }

    /// The residual and its vol derivative from one valuation, for Newton.
    std::pair<double, double> with_vega(double vol) {
        ++pricings_;
        const OptionValuation at_vol = revalue(vol);
        return {at_vol.present_value() - target_price_, vega(at_vol)};
    }

    [[nodiscard]] int pricings() const noexcept { return pricings_; }
    [[nodiscard]] const OptionValuation& valuation() const noexcept { return *valuation_; }
    [[nodiscard]] const UnderlyingData& underlying() const noexcept { return *underlying_; }

private:
    // Non-owning: a PriceResidual lives inside one implied_volatility call.
    const OptionValuation* valuation_;
    const UnderlyingData* underlying_;
    double target_price_;
    GreekCalculationMethod vega_method_;
    int pricings_ = 0;
};

/// Bracketing interval and the residuals at its ends.
struct Bracket {
    double low;
    double high;
    double f_low;
    double f_high;
};

/// Widen [low, high] until the residual changes sign: halve `low` (floored
/// at 1e-12) while the price there is still too high, double `high` while
/// it is still too low, at most 6 times each. Throws `ConvergenceError` if
/// the sign change is still missing.
Bracket bracket_implied_vol(PriceResidual& residual, double low, double high) {
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
    if (b.f_low > 0.0 || b.f_high < 0.0) {
        throw ConvergenceError(std::format(
            "Could not bracket the implied volatility: after widening, vol in [{:.6g}, {:.6g}] "
            "gives price residuals [{:.6g}, {:.6g}] (need <= 0 at the low end and >= 0 at the "
            "high end). Widen the vol bounds.",
            b.low,
            b.high,
            b.f_low,
            b.f_high));
    }
    return b;
}

/// Stopping rule for the bracketing solvers: the vol bracket is within
/// `tol`. `tol` is a vol tolerance for every solver.
struct VolTolerance {
    double tol;
    bool operator()(double a, double b) const { return std::abs(b - a) <= tol; }
};

/// Result of a bracketing solver: the midpoint of its final interval.
ImpliedVolResult bracketed_result(const std::pair<double, double>& interval,
                                  std::uintmax_t iterations,
                                  VolTolerance converged) {
    return {.implied_vol = interval.first + (interval.second - interval.first) / 2.0,
            .iterations = static_cast<int>(iterations),
            .converged = converged(interval.first, interval.second)};
}

ImpliedVolResult solve_bisection(PriceResidual& residual,
                                 const Bracket& bracket,
                                 const ImpliedVolParams& params) {
    auto iterations = static_cast<std::uintmax_t>(params.max_iter);
    const auto interval =
        boost::math::tools::bisect([&residual](double vol) { return residual(vol); },
                                   bracket.low,
                                   bracket.high,
                                   VolTolerance{params.tol},
                                   iterations);
    return bracketed_result(interval, iterations, VolTolerance{params.tol});
}

/// TOMS 748 (Alefeld, Potra & Shi): bracketing with inverse cubic
/// interpolation and double-length secant steps; typically fewer evaluations
/// than bisection or Brent's method.
ImpliedVolResult solve_toms748(PriceResidual& residual,
                               const Bracket& bracket,
                               const ImpliedVolParams& params) {
    auto iterations = static_cast<std::uintmax_t>(params.max_iter);
    const auto interval =
        boost::math::tools::toms748_solve([&residual](double vol) { return residual(vol); },
                                          bracket.low,
                                          bracket.high,
                                          bracket.f_low,
                                          bracket.f_high,
                                          VolTolerance{params.tol},
                                          iterations);
    return bracketed_result(interval, iterations, VolTolerance{params.tol});
}

/// Newton on (price residual, dPrice/dsigma), safeguarded by Boost within the
/// bracket. Boost stops on a relative step, so its digits are set for an
/// absolute step of `tol` at the top of the bracket; the result is then
/// certified in vol space. Empty when Newton cannot be certified (the caller
/// falls back to bisection). `initial` must lie in the bracket.
std::optional<ImpliedVolResult> solve_newton(PriceResidual& residual,
                                             const Bracket& bracket,
                                             double initial,
                                             const ImpliedVolParams& params) {
    const double tol = params.tol;
    const int digits = std::clamp(static_cast<int>(std::ceil(1.0 - std::log2(tol / bracket.high))),
                                  1,
                                  std::numeric_limits<double>::digits);
    auto iterations = static_cast<std::uintmax_t>(params.max_iter);
    try {
        const double vol = boost::math::tools::newton_raphson_iterate(
            [&residual](double trial) { return residual.with_vega(trial); },
            initial,
            bracket.low,
            bracket.high,
            digits,
            iterations);
        // The root is within tol of vol iff the residual (price is
        // increasing in vol) changes sign across vol +/- tol.
        const double below = std::max(vol - tol, bracket.low);
        const double above = std::min(vol + tol, bracket.high);
        const bool brackets_root = residual(below) <= 0.0 && residual(above) >= 0.0;
        if (!brackets_root) {  // also rejects NaN residuals
            return std::nullopt;
        }
        return ImpliedVolResult{
            .implied_vol = vol, .iterations = static_cast<int>(iterations), .converged = true};
    } catch (const boost::math::evaluation_error&) {
        // Boost lost its bracket (e.g. a flat or noisy numerical vega); the
        // bisection fallback is robust to that.
        return std::nullopt;
    }
}

/// Shared by every exit: mark the vol well determined only if the solve
/// converged and vega reaches `min_vega` (warning when it does not), record
/// the pricing count, and log the outcome.
ImpliedVolResult finalise(ImpliedVolResult result,
                          PriceResidual& residual,
                          const ImpliedVolParams& params,
                          ImpliedVolMethod solver) {
    if (result.converged) {
        const double floor = params.min_vega.value_or(
            1.0e-6 * std::max(residual.underlying().initial_value(),
                              residual.valuation().strike().value_or(0.0)));
        const double vega = residual.vega(residual.revalue(result.implied_vol));
        result.well_determined = vega >= floor;
        if (!result.well_determined) {
            warn(std::format(
                "Implied vol {:.6g} is not determined by the price: vega {:.3g} per unit vol "
                "is below min_vega={:.3g}, so rounding in the price is a large vol error "
                "(typical far out of the money or near intrinsic). Returning "
                "well_determined=false.",
                result.implied_vol,
                vega,
                floor));
        }
    }
    result.function_evaluations = residual.pricings();
    detail::logger()->debug(
        "Implied vol method={} converged={} well_determined={} iterations={} "
        "function_evaluations={} implied_vol={:.6g}",
        to_string(solver),
        result.converged,
        result.well_determined,
        result.iterations,
        result.function_evaluations,
        result.implied_vol);
    return result;
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

    const auto [low, high] = search_interval(params, valuation, underlying);
    require_within_bounds(target_price, valuation, underlying);

    const ImpliedVolMethod solver = params.method.value_or(
        valuation.pricing_method() == PricingMethod::Analytical ? ImpliedVolMethod::NewtonRaphson
                                                                : ImpliedVolMethod::Toms748);
    // Clipped to [low, high], which the bracket contains (it only widens).
    const double initial = initial_vol(params, valuation, underlying, target_price, low, high);
    PriceResidual residual(valuation, underlying, target_price);
    const Bracket bracket = bracket_implied_vol(residual, low, high);
    if (bracket.f_low == 0.0 || bracket.f_high == 0.0) {
        // The root sits on a bracket end: nothing to solve.
        return finalise({.implied_vol = bracket.f_low == 0.0 ? bracket.low : bracket.high,
                         .iterations = 0,
                         .converged = true},
                        residual,
                        params,
                        solver);
    }

    switch (solver) {
        case ImpliedVolMethod::NewtonRaphson:
            if (const auto newton = solve_newton(residual, bracket, initial, params)) {
                return finalise(*newton, residual, params, solver);
            }
            return finalise(solve_bisection(residual, bracket, params), residual, params, solver);
        case ImpliedVolMethod::Bisection:
            return finalise(solve_bisection(residual, bracket, params), residual, params, solver);
        case ImpliedVolMethod::Toms748:
            return finalise(solve_toms748(residual, bracket, params), residual, params, solver);
    }
    throw ConfigurationError("unknown ImpliedVolMethod");
}

}  // namespace dp
