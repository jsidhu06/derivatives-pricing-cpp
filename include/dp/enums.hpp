#pragma once

#include <string_view>

/// Enumerations shared across contract specifications, market data and engines.
///
/// Each enum has a `to_string` overload returning a stable, human-readable
/// identifier, the same as the companion Python library's enum values.
namespace dp {

/// Call or put option type.
enum class OptionType { Call, Put };

/// European (expiry only) or American (any time) exercise.
enum class ExerciseType { European, American };

/// Numerical method used by the valuation facade to price an option.
/// `Analytical` is closed-form pricing under GBM dynamics: Black-Scholes-Merton
/// for vanillas, Reiner-Rubinstein (with the Broadie-Glasserman-Kou discrete
/// correction) for barriers.
enum class PricingMethod { Analytical, Binomial, PdeFd, MonteCarlo };

/// Algorithm used to compute Greeks. Not every method is available for every
/// pricing engine; the valuation facade validates the combination.
enum class GreekCalculationMethod {
    Analytical,
    Tree,
    Grid,
    Pathwise,
    LikelihoodRatio,
    Numerical,
};

/// Root-finding algorithm for the implied-volatility solver.
enum class ImpliedVolMethod { NewtonRaphson, Bisection, Toms748 };

/// Finite-difference time-stepping scheme for the PDE solver.
enum class PdeMethod { Implicit, Explicit, ExplicitHull, CrankNicolson };

/// Spatial variable for the PDE finite-difference grid.
enum class PdeSpaceGrid { Spot, LogSpot };

/// Early-exercise enforcement algorithm for American PDE pricing.
enum class PdeEarlyExercise { Intrinsic, GaussSeidel };

/// Day-count convention for year-fraction calculations.
enum class DayCountConvention { Act360, Act365F, Act365_25, Thirty360Us };

/// Direction the underlying must move to trigger the barrier.
enum class BarrierDirection { Up, Down };

/// Whether the barrier knocks the option in or out.
enum class BarrierAction { In, Out };

/// How frequently the barrier is monitored.
enum class BarrierMonitoring { Continuous, Discrete };

/// When the rebate is paid upon barrier trigger.
enum class RebateTiming { AtHit, AtExpiry };

[[nodiscard]] std::string_view to_string(OptionType value) noexcept;
[[nodiscard]] std::string_view to_string(ExerciseType value) noexcept;
[[nodiscard]] std::string_view to_string(PricingMethod value) noexcept;
[[nodiscard]] std::string_view to_string(GreekCalculationMethod value) noexcept;
[[nodiscard]] std::string_view to_string(ImpliedVolMethod value) noexcept;
[[nodiscard]] std::string_view to_string(PdeMethod value) noexcept;
[[nodiscard]] std::string_view to_string(PdeSpaceGrid value) noexcept;
[[nodiscard]] std::string_view to_string(PdeEarlyExercise value) noexcept;
[[nodiscard]] std::string_view to_string(DayCountConvention value) noexcept;
[[nodiscard]] std::string_view to_string(BarrierDirection value) noexcept;
[[nodiscard]] std::string_view to_string(BarrierAction value) noexcept;
[[nodiscard]] std::string_view to_string(BarrierMonitoring value) noexcept;
[[nodiscard]] std::string_view to_string(RebateTiming value) noexcept;

/// Upper-case identifier (`ANALYTICAL`, `PDE_FD`, `MONTE_CARLO`, ...), the same
/// as the companion Python library's `Enum.name`. Used in diagnostics.
[[nodiscard]] std::string_view name(PricingMethod value) noexcept;
[[nodiscard]] std::string_view name(ExerciseType value) noexcept;
[[nodiscard]] std::string_view name(GreekCalculationMethod value) noexcept;

}  // namespace dp
