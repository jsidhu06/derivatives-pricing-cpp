#include "dp/enums.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("enum to_string matches the Python reference values", "[enums]") {
    using namespace dp;
    CHECK(to_string(OptionType::Call) == "call");
    CHECK(to_string(OptionType::Put) == "put");
    CHECK(to_string(ExerciseType::European) == "european");
    CHECK(to_string(ExerciseType::American) == "american");
    CHECK(to_string(PricingMethod::Analytical) == "analytical");
    CHECK(to_string(PricingMethod::PdeFd) == "pde_fd");
    CHECK(to_string(PricingMethod::MonteCarlo) == "monte_carlo");
    CHECK(to_string(GreekCalculationMethod::LikelihoodRatio) == "likelihood_ratio");
    CHECK(to_string(ImpliedVolMethod::NewtonRaphson) == "newton_raphson");
    CHECK(to_string(PdeMethod::CrankNicolson) == "crank_nicolson");
    CHECK(to_string(PdeSpaceGrid::LogSpot) == "log_spot");
    CHECK(to_string(PdeEarlyExercise::GaussSeidel) == "gauss_seidel");
    CHECK(to_string(DayCountConvention::Act360) == "ACT/360");
    CHECK(to_string(DayCountConvention::Act365F) == "ACT/365F");
    CHECK(to_string(DayCountConvention::Act365_25) == "ACT/365.25");
    CHECK(to_string(DayCountConvention::Thirty360Us) == "30/360 US");
    CHECK(to_string(BarrierDirection::Up) == "up");
    CHECK(to_string(BarrierAction::Out) == "out");
    CHECK(to_string(BarrierMonitoring::Discrete) == "discrete");
    CHECK(to_string(RebateTiming::AtExpiry) == "at_expiry");
}
