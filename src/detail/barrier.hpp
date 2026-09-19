#pragma once

#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/enums.hpp"

#include "detail/log.hpp"
#include "valuation_state.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <variant>
#include <vector>

/// Internal barrier helpers shared by the facade and the barrier engines.
namespace dp::detail {

/// Monitoring dates for a DISCRETE barrier; nullopt for CONTINUOUS.
inline std::optional<std::vector<DateTime>> barrier_monitoring_dates(const ValuationState& state) {
    const auto& spec = std::get<BarrierSpec>(state.spec);
    if (spec.monitoring() == BarrierMonitoring::Continuous) {
        return std::nullopt;
    }
    if (spec.monitoring_dates().has_value()) {
        return spec.monitoring_dates();
    }
    // N+1 points from pricing_date to maturity; drop t=0 to leave N dates at
    // T/N, 2T/N, ..., T (the academic convention for discrete monitoring).
    auto grid = date_range(state.pricing_date(),
                           state.maturity(),
                           static_cast<std::size_t>(*spec.num_observations()) + 1);
    grid.erase(grid.begin());
    return grid;
}

/// True only if the barrier has been hit AND that hit is observable at the
/// pricing date. Continuous monitoring treats every instant as an
/// observation, so a spot past the barrier at t=0 is an immediate trigger;
/// discrete monitoring only observes at explicit monitoring dates, so the
/// pricing date qualifies only if it appears in that schedule.
inline bool barrier_triggered_at_inception(const ValuationState& state) {
    const auto& spec = std::get<BarrierSpec>(state.spec);
    if (!spec.is_spot_past_barrier(state.underlying.initial_value())) {
        return false;
    }
    if (spec.monitoring() == BarrierMonitoring::Continuous) {
        return true;
    }
    const auto dates = barrier_monitoring_dates(state);
    return dates.has_value() &&
           std::find(dates->begin(), dates->end(), state.pricing_date()) != dates->end();
}

/// Cap `epsilon` so a central spot bump cannot cross the barrier. An
/// unconstrained bump of `s0 ± epsilon` may cross the barrier on one side,
/// putting the bumped option in a different (knocked-out vs alive) regime
/// than the unbumped option; the resulting numerical greek then averages two
/// regimes and is biased — sometimes by an order of magnitude. No-op for
/// non-barrier specs and inception-triggered barriers.
inline double resolve_barrier_spot_bump(const ValuationState& state, double epsilon) {
    constexpr double kBarrierBumpMaxFraction = 0.5;
    const auto* spec = std::get_if<BarrierSpec>(&state.spec);
    if (spec == nullptr) {
        return epsilon;
    }
    const double s0 = state.underlying.initial_value();
    const double barrier = spec->barrier();
    const double gap = spec->direction() == BarrierDirection::Down ? s0 - barrier : barrier - s0;
    if (gap <= 0.0) {
        // Inception-triggered; let the engine handle it.
        return epsilon;
    }
    const double max_bump = gap * kBarrierBumpMaxFraction;
    if (epsilon <= max_bump) {
        return epsilon;
    }
    logger()->warn(
        "Numerical greek bump epsilon={:g} would cross {} barrier H={:g} (spot={:g}); "
        "shrinking to {:g}.",
        epsilon,
        to_string(spec->direction()),
        barrier,
        s0,
        max_bump);
    return max_bump;
}

}  // namespace dp::detail
