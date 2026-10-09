#include "dp/enums.hpp"
#include "dp/exceptions.hpp"
#include "dp/params.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace dp;
using dp::test::message_contains;

TEST_CASE("default params validate", "[params]") {
    CHECK_NOTHROW(MonteCarloParams{}.validate());
    CHECK_NOTHROW(BinomialParams{}.validate());
    CHECK_NOTHROW(PdeParams{}.validate());
    CHECK(BinomialParams{}.num_steps == 500);
    CHECK(PdeParams{}.method == PdeMethod::CrankNicolson);
    CHECK_FALSE(PdeParams{}.smax_mult.has_value());  // auto domain from sigma*sqrt(T)
    CHECK(MonteCarloParams{}.random_seed == 42ULL);
}

TEST_CASE("MonteCarloParams validation", "[params]") {
    CHECK_THROWS_MATCHES(MonteCarloParams{.basis_degree = 0}.validate(),
                         ValidationError,
                         message_contains("basis_degree must be >= 1"));
    CHECK_THROWS_MATCHES(MonteCarloParams{.ridge_lambda = -1e-3}.validate(),
                         ValidationError,
                         message_contains("ridge_lambda must be >= 0"));
    CHECK_THROWS_MATCHES(MonteCarloParams{.min_itm = 0}.validate(),
                         ValidationError,
                         message_contains("min_itm must be >= 1"));
    CHECK_THROWS_MATCHES(MonteCarloParams{.std_error_warn_ratio = 0.0}.validate(),
                         ValidationError,
                         message_contains("std_error_warn_ratio must be > 0"));
    CHECK_THROWS_MATCHES(MonteCarloParams{.std_error_warn_ratio = -0.1}.validate(),
                         ValidationError,
                         message_contains("std_error_warn_ratio must be > 0"));
    CHECK_NOTHROW(MonteCarloParams{.std_error_warn_ratio = std::nullopt}.validate());
}

TEST_CASE("BinomialParams validation", "[params]") {
    CHECK_THROWS_MATCHES(BinomialParams{.num_steps = 0}.validate(),
                         ValidationError,
                         message_contains("num_steps must be >= 1"));
}

TEST_CASE("PdeParams validation", "[params]") {
    // <= 1 would give a domain that need not contain the spot.
    for (const double smax_mult : {0.0, -1.0, 0.5, 1.0}) {
        CHECK_THROWS_MATCHES(PdeParams{.smax_mult = smax_mult}.validate(),
                             ValidationError,
                             message_contains("smax_mult must be > 1"));
    }
    CHECK_NOTHROW(PdeParams{.smax_mult = 6.0}.validate());
    CHECK_THROWS_MATCHES(PdeParams{.spot_steps = 2}.validate(),
                         ValidationError,
                         message_contains("spot_steps must be >= 3"));
    CHECK_THROWS_MATCHES(PdeParams{.time_steps = 0}.validate(),
                         ValidationError,
                         message_contains("time_steps must be >= 1"));
    CHECK_THROWS_MATCHES(PdeParams{.omega = 1.0}.validate(),
                         ValidationError,
                         message_contains("omega must be in (1.0, 2.0)"));
    CHECK_THROWS_MATCHES(PdeParams{.omega = 2.0}.validate(),
                         ValidationError,
                         message_contains("omega must be in (1.0, 2.0)"));
    CHECK_THROWS_MATCHES(PdeParams{.tol = 0.0}.validate(),
                         ValidationError,
                         message_contains("tol must be positive"));
    CHECK_THROWS_MATCHES(PdeParams{.max_iter = 0}.validate(),
                         ValidationError,
                         message_contains("max_iter must be >= 1"));
    CHECK_THROWS_MATCHES(PdeParams{.rannacher_steps = -1}.validate(),
                         ValidationError,
                         message_contains("rannacher_steps must be >= 0"));
}

TEST_CASE("PdeParams::for_barriers picks scheme from monitoring", "[params]") {
    const auto cont = PdeParams::for_barriers(BarrierMonitoring::Continuous);
    CHECK(cont.method == PdeMethod::CrankNicolson);
    CHECK(cont.space_grid == PdeSpaceGrid::LogSpot);
    CHECK(cont.spot_steps == 1200);
    CHECK(cont.time_steps == 800);
    CHECK(cont.american_solver == PdeEarlyExercise::GaussSeidel);
    CHECK_FALSE(cont.control_variate_european);

    const auto disc = PdeParams::for_barriers(BarrierMonitoring::Discrete);
    CHECK(disc.method == PdeMethod::ExplicitHull);
    CHECK(disc.space_grid == PdeSpaceGrid::LogSpot);
    CHECK(disc.spot_steps == 1200);
    CHECK(disc.time_steps == 3000);
    CHECK(disc.american_solver == PdeEarlyExercise::Intrinsic);
    CHECK_NOTHROW(disc.validate());
}

TEST_CASE("for_barriers presets can be overridden field-by-field", "[params]") {
    // Mirrors the Python `PDEParams.for_barriers(monitoring=..., **overrides)`
    // pattern: preset first, then assign the fields to override.
    auto params = PdeParams::for_barriers(BarrierMonitoring::Discrete);
    params.time_steps = 5000;
    params.tol = 1e-8;
    params.method = PdeMethod::CrankNicolson;  // even the preset-chosen scheme

    CHECK(params.time_steps == 5000);
    CHECK(params.tol == 1e-8);
    CHECK(params.method == PdeMethod::CrankNicolson);
    // Untouched presets survive.
    CHECK(params.spot_steps == 1200);
    CHECK(params.space_grid == PdeSpaceGrid::LogSpot);
    CHECK(params.american_solver == PdeEarlyExercise::Intrinsic);
    CHECK_NOTHROW(params.validate());

    // Invalid overrides are still rejected by validate().
    params.omega = 2.5;
    CHECK_THROWS_MATCHES(
        params.validate(), ValidationError, message_contains("omega must be in (1.0, 2.0)"));
}
