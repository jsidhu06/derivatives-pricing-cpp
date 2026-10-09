#include "dp/params.hpp"

#include "dp/exceptions.hpp"

#include <format>

namespace dp {

void MonteCarloParams::validate() const {
    if (basis_degree < 1) {
        throw ValidationError(std::format("basis_degree must be >= 1, got {}", basis_degree));
    }
    if (ridge_lambda < 0.0) {
        throw ValidationError(std::format("ridge_lambda must be >= 0, got {}", ridge_lambda));
    }
    if (min_itm < 1) {
        throw ValidationError(std::format("min_itm must be >= 1, got {}", min_itm));
    }
    if (std_error_warn_ratio.has_value() && *std_error_warn_ratio <= 0.0) {
        throw ValidationError(std::format("std_error_warn_ratio must be > 0 when set, got {}",
                                          *std_error_warn_ratio));
    }
}

void BinomialParams::validate() const {
    if (num_steps < 1) {
        throw ValidationError(std::format("num_steps must be >= 1, got {}", num_steps));
    }
}

void PdeParams::validate() const {
    if (smax_mult.has_value() && !(*smax_mult > 1.0)) {
        throw ValidationError(
            std::format("smax_mult must be > 1 (or unset for auto), got {}", *smax_mult));
    }
    if (spot_steps < 3) {
        throw ValidationError(std::format("spot_steps must be >= 3, got {}", spot_steps));
    }
    if (time_steps < 1) {
        throw ValidationError(std::format("time_steps must be >= 1, got {}", time_steps));
    }
    if (omega <= 1.0 || omega >= 2.0) {
        throw ValidationError(std::format("omega must be in (1.0, 2.0), got {}", omega));
    }
    if (tol <= 0.0) {
        throw ValidationError(std::format("tol must be positive, got {}", tol));
    }
    if (max_iter < 1) {
        throw ValidationError(std::format("max_iter must be >= 1, got {}", max_iter));
    }
    if (rannacher_steps < 0) {
        throw ValidationError(std::format("rannacher_steps must be >= 0, got {}", rannacher_steps));
    }
}

PdeParams PdeParams::for_barriers(BarrierMonitoring monitoring) {
    PdeParams params;
    params.space_grid = PdeSpaceGrid::LogSpot;
    params.spot_steps = 1200;
    if (monitoring == BarrierMonitoring::Discrete) {
        // Each observation projects V=0 past the barrier, introducing a fresh
        // step discontinuity; cheap explicit steps let time resolution dominate.
        params.method = PdeMethod::ExplicitHull;
        params.time_steps = 3000;
        params.american_solver = PdeEarlyExercise::Intrinsic;
    } else {
        // Single payoff discontinuity at maturity: Rannacher-started
        // Crank-Nicolson gives the best PV / Greek quality.
        params.method = PdeMethod::CrankNicolson;
        params.time_steps = 800;
        params.american_solver = PdeEarlyExercise::GaussSeidel;
    }
    return params;
}

}  // namespace dp
