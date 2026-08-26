#include "dp/params.hpp"

#include "dp/diagnostics.hpp"
#include "dp/exceptions.hpp"

#include <format>

namespace dp {

void MonteCarloParams::validate() const {
    if (deg < 1) {
        throw ValidationError(std::format("deg must be >= 1, got {}", deg));
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
    if (mc_paths.has_value() && asian_tree_averages.has_value()) {
        throw ValidationError("Only one of mc_paths and asian_tree_averages can be set, got both");
    }
    if (mc_paths.has_value() && *mc_paths < 1) {
        throw ValidationError(std::format("mc_paths must be >= 1, got {}", *mc_paths));
    }
    if (asian_tree_averages.has_value()) {
        if (*asian_tree_averages < 1) {
            throw ValidationError(
                std::format("asian_tree_averages must be >= 1, got {}", *asian_tree_averages));
        }
        const double ratio = static_cast<double>(*asian_tree_averages) / num_steps;
        if (ratio < 1.5) {
            warn(
                "asian_tree_averages / num_steps < 1.5; Hull-style Asian valuation may "
                "exhibit upward interpolation bias. A ratio of 1.5–2.0 is recommended.");
        }
        if (ratio > 2.1) {
            warn(
                "asian_tree_averages is large relative to num_steps; memory usage may be "
                "high with limited accuracy gains.");
        }
        const double est_bytes = static_cast<double>(*asian_tree_averages) *
                                 static_cast<double>(num_steps + 1) *
                                 static_cast<double>(num_steps + 1) * 8.0;
        if (est_bytes > 1e9) {
            warn(std::format(
                "Estimated memory for Hull Asian grid is ~{:.2f} GiB; consider reducing "
                "num_steps or asian_tree_averages.",
                est_bytes / (1024.0 * 1024.0 * 1024.0)));
        }
    }
}

void PdeParams::validate() const {
    if (smax_mult <= 0.0) {
        throw ValidationError(std::format("smax_mult must be positive, got {}", smax_mult));
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
