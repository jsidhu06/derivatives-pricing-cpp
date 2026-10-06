#include "dp/enums.hpp"

namespace dp {

std::string_view to_string(OptionType value) noexcept {
    switch (value) {
        case OptionType::Call:
            return "call";
        case OptionType::Put:
            return "put";
    }
    return "unknown";
}

std::string_view to_string(ExerciseType value) noexcept {
    switch (value) {
        case ExerciseType::European:
            return "european";
        case ExerciseType::American:
            return "american";
    }
    return "unknown";
}

std::string_view to_string(PricingMethod value) noexcept {
    switch (value) {
        case PricingMethod::Analytical:
            return "analytical";
        case PricingMethod::Binomial:
            return "binomial";
        case PricingMethod::PdeFd:
            return "pde_fd";
        case PricingMethod::MonteCarlo:
            return "monte_carlo";
    }
    return "unknown";
}

std::string_view to_string(GreekCalculationMethod value) noexcept {
    switch (value) {
        case GreekCalculationMethod::Analytical:
            return "analytical";
        case GreekCalculationMethod::Tree:
            return "tree";
        case GreekCalculationMethod::Grid:
            return "grid";
        case GreekCalculationMethod::Pathwise:
            return "pathwise";
        case GreekCalculationMethod::LikelihoodRatio:
            return "likelihood_ratio";
        case GreekCalculationMethod::Numerical:
            return "numerical";
    }
    return "unknown";
}

std::string_view to_string(ImpliedVolMethod value) noexcept {
    switch (value) {
        case ImpliedVolMethod::NewtonRaphson:
            return "newton_raphson";
        case ImpliedVolMethod::Bisection:
            return "bisection";
        case ImpliedVolMethod::Brent:
            return "brent";
    }
    return "unknown";
}

std::string_view to_string(PdeMethod value) noexcept {
    switch (value) {
        case PdeMethod::Implicit:
            return "implicit";
        case PdeMethod::Explicit:
            return "explicit";
        case PdeMethod::ExplicitHull:
            return "explicit_hull";
        case PdeMethod::CrankNicolson:
            return "crank_nicolson";
    }
    return "unknown";
}

std::string_view to_string(PdeSpaceGrid value) noexcept {
    switch (value) {
        case PdeSpaceGrid::Spot:
            return "spot";
        case PdeSpaceGrid::LogSpot:
            return "log_spot";
    }
    return "unknown";
}

std::string_view to_string(PdeEarlyExercise value) noexcept {
    switch (value) {
        case PdeEarlyExercise::Intrinsic:
            return "intrinsic";
        case PdeEarlyExercise::GaussSeidel:
            return "gauss_seidel";
    }
    return "unknown";
}

std::string_view to_string(DayCountConvention value) noexcept {
    switch (value) {
        case DayCountConvention::Act360:
            return "ACT/360";
        case DayCountConvention::Act365F:
            return "ACT/365F";
        case DayCountConvention::Act365_25:
            return "ACT/365.25";
        case DayCountConvention::Thirty360Us:
            return "30/360 US";
    }
    return "unknown";
}

std::string_view to_string(BarrierDirection value) noexcept {
    switch (value) {
        case BarrierDirection::Up:
            return "up";
        case BarrierDirection::Down:
            return "down";
    }
    return "unknown";
}

std::string_view to_string(BarrierAction value) noexcept {
    switch (value) {
        case BarrierAction::In:
            return "in";
        case BarrierAction::Out:
            return "out";
    }
    return "unknown";
}

std::string_view to_string(BarrierMonitoring value) noexcept {
    switch (value) {
        case BarrierMonitoring::Continuous:
            return "continuous";
        case BarrierMonitoring::Discrete:
            return "discrete";
    }
    return "unknown";
}

std::string_view to_string(RebateTiming value) noexcept {
    switch (value) {
        case RebateTiming::AtHit:
            return "at_hit";
        case RebateTiming::AtExpiry:
            return "at_expiry";
    }
    return "unknown";
}

std::string_view name(PricingMethod value) noexcept {
    switch (value) {
        case PricingMethod::Analytical:
            return "ANALYTICAL";
        case PricingMethod::Binomial:
            return "BINOMIAL";
        case PricingMethod::PdeFd:
            return "PDE_FD";
        case PricingMethod::MonteCarlo:
            return "MONTE_CARLO";
    }
    return "UNKNOWN";
}

std::string_view name(ExerciseType value) noexcept {
    switch (value) {
        case ExerciseType::European:
            return "EUROPEAN";
        case ExerciseType::American:
            return "AMERICAN";
    }
    return "UNKNOWN";
}

std::string_view name(GreekCalculationMethod value) noexcept {
    switch (value) {
        case GreekCalculationMethod::Analytical:
            return "ANALYTICAL";
        case GreekCalculationMethod::Tree:
            return "TREE";
        case GreekCalculationMethod::Grid:
            return "GRID";
        case GreekCalculationMethod::Pathwise:
            return "PATHWISE";
        case GreekCalculationMethod::LikelihoodRatio:
            return "LIKELIHOOD_RATIO";
        case GreekCalculationMethod::Numerical:
            return "NUMERICAL";
    }
    return "UNKNOWN";
}

}  // namespace dp
