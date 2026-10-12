#pragma once

#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/params.hpp"
#include "dp/simulation.hpp"
#include "dp/underlying.hpp"

#include <memory>
#include <optional>
#include <string>
#include <variant>

/// Valuation facade: dispatches a contract spec + underlying + pricing method
/// to an engine, and exposes cached present value and Greeks.
namespace dp {

namespace detail {
class Underlying;
}  // namespace detail

/// Options for a Greek request.
///
/// `method` selects the computation method; when unset the best available
/// method for the engine is chosen (ANALYTICAL → TREE → GRID → PATHWISE →
/// NUMERICAL). `bump` is the finite-difference step used by NUMERICAL
/// bump-and-revalue and is rejected for methods that do not bump:
///
/// | Greek | `bump` meaning                          | default      |
/// |-------|------------------------------------------|--------------|
/// | delta | spot bump (central difference)           | spot / 100   |
/// | gamma | spot bump (second central difference)    | spot / 100   |
/// | vega  | volatility bump (absolute)               | 0.01         |
/// | theta | calendar days added to the pricing date  | 1.0          |
/// | rho   | parallel zero-rate bump (absolute)       | 0.01         |
///
/// Differences are central, except vega when the volatility is no larger
/// than its bump, which uses a forward difference.
struct GreekOptions {
    std::optional<GreekCalculationMethod> method = std::nullopt;
    std::optional<double> bump = std::nullopt;
};

/// Single-factor option valuation. Immutable once constructed; results are
/// memoised per instance and the instance is safe to query from multiple
/// threads. Movable, not copyable.
class OptionValuation {
public:
    /// Throws `ConfigurationError` / `ValidationError` / `UnsupportedFeatureError`
    /// on inconsistent inputs; engine construction happens eagerly.
    OptionValuation(UnderlyingData underlying,
                    OptionSpec spec,
                    PricingMethod pricing_method,
                    std::optional<ValuationParams> params = std::nullopt);
    /// Monte Carlo pricing: the underlying is a path process.
    OptionValuation(GbmProcess process,
                    OptionSpec spec,
                    PricingMethod pricing_method,
                    std::optional<ValuationParams> params = std::nullopt);
    OptionValuation(JdProcess process,
                    OptionSpec spec,
                    PricingMethod pricing_method,
                    std::optional<ValuationParams> params = std::nullopt);
    OptionValuation(SrdProcess process,
                    OptionSpec spec,
                    PricingMethod pricing_method,
                    std::optional<ValuationParams> params = std::nullopt);
    ~OptionValuation();

    OptionValuation(OptionValuation&&) noexcept;
    OptionValuation& operator=(OptionValuation&&) noexcept;
    OptionValuation(const OptionValuation&) = delete;
    OptionValuation& operator=(const OptionValuation&) = delete;

    /// Present value per unit of underlying (contract size not applied).
    [[nodiscard]] double present_value() const;
    /// Monte Carlo standard error of `present_value()` (antithetic pairs
    /// averaged before the sample variance; moment matching ignored). Same
    /// simulation as the PV, so no extra cost. NaN with fewer than two
    /// independent samples.
    ///
    /// `nullopt` for the deterministic methods: they have no sampling error
    /// to report. That is "not applicable", not zero error; their
    /// discretisation error is not measured here. Throws
    /// `UnsupportedFeatureError` for Monte Carlo with
    /// `control_variate_european`, where a sampling error exists but needs
    /// the covariance of both legs, which is not computed.
    [[nodiscard]] std::optional<double> std_error() const;

    /// dV/dS.
    [[nodiscard]] double delta(GreekOptions options = {}) const;
    /// d²V/dS².
    [[nodiscard]] double gamma(GreekOptions options = {}) const;
    /// dV/dσ per 1 vol-point (1%) change.
    [[nodiscard]] double vega(GreekOptions options = {}) const;
    /// Value change per calendar day.
    ///
    /// On a flat curve every method agrees. On a sloped curve they use
    /// different rates for the `r V` and carry terms, so they differ:
    /// - Analytical (vanilla and barrier) and Monte Carlo pathwise /
    ///   likelihood-ratio theta use the zero rates to maturity, the
    ///   flat-equivalent rates of the closed-form price (as QuantLib's
    ///   `BlackCalculator::theta`).
    /// - Tree and PDE theta use the forward rates at the start of the curve:
    ///   the one-day decay with today's calendar curve held fixed (as
    ///   QuantLib's FD engines).
    /// - Numerical theta moves the pricing date and keeps the curve, which is
    ///   defined by time from the pricing date, so the curve slides with it:
    ///   each tenor keeps its zero rate, and the decay picks up the forward
    ///   rate at maturity.
    ///
    /// One convention across all methods is planned.
    [[nodiscard]] double theta(GreekOptions options = {}) const;
    /// dV/dr per 1% parallel rate move.
    [[nodiscard]] double rho(GreekOptions options = {}) const;

    /// The underlying as constructed: `UnderlyingData` for deterministic
    /// engines, the path process (`GbmProcess`, `JdProcess` or `SrdProcess`)
    /// for Monte Carlo.
    [[nodiscard]] const std::variant<UnderlyingData, GbmProcess, JdProcess, SrdProcess>&
    underlying() const noexcept;
    [[nodiscard]] const OptionSpec& spec() const noexcept;
    [[nodiscard]] PricingMethod pricing_method() const noexcept;
    [[nodiscard]] const std::optional<ValuationParams>& params() const noexcept;
    [[nodiscard]] std::optional<OptionType> option_type() const noexcept;
    [[nodiscard]] DateTime maturity() const;
    [[nodiscard]] std::optional<double> strike() const;
    [[nodiscard]] const std::string& currency() const noexcept;
    [[nodiscard]] ExerciseType exercise_type() const;
    [[nodiscard]] double contract_size() const;
    [[nodiscard]] DateTime pricing_date() const;
    [[nodiscard]] const DiscountCurve& discount_curve() const;
    [[nodiscard]] DayCountConvention day_count_convention() const;

    /// Time to maturity in years under the valuation day-count convention.
    [[nodiscard]] double maturity_year_fraction() const;

    /// One-line description, e.g.
    /// `OptionValuation(ANALYTICAL, EUROPEAN, CALL, S=100, K=105, T=2025-07-01,
    /// pricing_date=2025-01-01, VanillaSpec)`.
    [[nodiscard]] std::string describe() const;

private:
    struct Impl;
    void init(detail::Underlying underlying,
              OptionSpec spec,
              PricingMethod pricing_method,
              std::optional<ValuationParams> params);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dp
