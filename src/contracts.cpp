#include "dp/contracts.hpp"

#include "dp/exceptions.hpp"

#include "detail/validate.hpp"

#include <algorithm>
#include <set>
#include <utility>

namespace dp {

namespace {

void validate_contract_size(double contract_size, std::string_view spec_name) {
    detail::require_positive(contract_size, std::string(spec_name) + ".contract_size");
}

void validate_currency(const std::optional<std::string>& currency) {
    if (currency.has_value() && currency->empty()) {
        throw ValidationError("currency must be a non-empty string when provided");
    }
}

}  // namespace

// ── VanillaSpec ─────────────────────────────────────────────────────

VanillaSpec::VanillaSpec(Params params) : p_(std::move(params)) {
    detail::require_non_negative(p_.strike, "VanillaSpec.strike");
    validate_contract_size(p_.contract_size, "VanillaSpec");
    validate_currency(p_.currency);
}

// ── PayoffSpec ──────────────────────────────────────────────────────

PayoffSpec::PayoffSpec(Params params) : p_(std::move(params)) {
    if (!p_.payoff_fn) {
        throw ConfigurationError("payoff_fn must be callable");
    }
    validate_contract_size(p_.contract_size, "PayoffSpec");
    validate_currency(p_.currency);
    if (p_.boundary_model.has_value()) {
        detail::require_finite(p_.boundary_model->left.slope, "boundary_model.left.slope");
        detail::require_finite(p_.boundary_model->left.intercept, "boundary_model.left.intercept");
        detail::require_finite(p_.boundary_model->right.slope, "boundary_model.right.slope");
        detail::require_finite(p_.boundary_model->right.intercept,
                               "boundary_model.right.intercept");
    }
}

// ── BarrierSpec ─────────────────────────────────────────────────────

BarrierSpec::BarrierSpec(Params params) : p_(std::move(params)) {
    detail::require_non_negative(p_.strike, "BarrierSpec.strike");
    detail::require_positive(p_.barrier, "BarrierSpec.barrier");
    detail::require_non_negative(p_.rebate, "BarrierSpec.rebate");
    validate_contract_size(p_.contract_size, "BarrierSpec");
    validate_currency(p_.currency);

    // A barrier hit *activates* a knock-in, so an at-hit rebate is contradictory.
    if (p_.action == BarrierAction::In && p_.rebate > 0.0 &&
        p_.rebate_timing == RebateTiming::AtHit) {
        throw ValidationError(
            "Knock-in rebate must use RebateTiming.AT_EXPIRY. "
            "AT_HIT is only valid for knock-out options.");
    }

    if (p_.monitoring == BarrierMonitoring::Continuous) {
        if (p_.num_observations.has_value() || p_.monitoring_dates.has_value()) {
            throw ValidationError(
                "CONTINUOUS monitoring must not specify num_observations or monitoring_dates.");
        }
        return;
    }

    // DISCRETE: exactly one schedule source.
    if (p_.num_observations.has_value() == p_.monitoring_dates.has_value()) {
        throw ValidationError(
            "DISCRETE monitoring requires exactly one of num_observations or monitoring_dates.");
    }
    if (p_.num_observations.has_value() && *p_.num_observations < 1) {
        throw ValidationError("num_observations must be an integer >= 1");
    }
    if (p_.monitoring_dates.has_value()) {
        const auto& dates = *p_.monitoring_dates;
        if (dates.empty()) {
            throw ValidationError("monitoring_dates must be non-empty when provided.");
        }
        const std::set<DateTime> unique(dates.begin(), dates.end());
        if (unique.size() != dates.size()) {
            throw ValidationError("monitoring_dates must contain unique dates.");
        }
        if (!std::is_sorted(dates.begin(), dates.end())) {
            throw ValidationError("monitoring_dates must be in strictly ascending order.");
        }
        if (dates.back() > p_.maturity) {
            throw ValidationError("monitoring_dates must not extend beyond maturity.");
        }
    }
}

bool BarrierSpec::is_spot_past_barrier(double spot) const noexcept {
    if (p_.direction == BarrierDirection::Up) {
        return spot >= p_.barrier;
    }
    return spot <= p_.barrier;
}

// ── Union helpers ───────────────────────────────────────────────────

std::string_view spec_type_name(const OptionSpec& spec) {
    return std::visit(
        [](const auto& s) -> std::string_view {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, VanillaSpec>) {
                return "VanillaSpec";
            } else if constexpr (std::is_same_v<T, PayoffSpec>) {
                return "PayoffSpec";
            } else {
                return "BarrierSpec";
            }
        },
        spec);
}

DateTime maturity_of(const OptionSpec& spec) {
    return std::visit([](const auto& s) { return s.maturity(); }, spec);
}

ExerciseType exercise_type_of(const OptionSpec& spec) {
    return std::visit([](const auto& s) { return s.exercise_type(); }, spec);
}

const std::optional<std::string>& currency_of(const OptionSpec& spec) {
    return std::visit(
        [](const auto& s) -> const std::optional<std::string>& { return s.currency(); }, spec);
}

double contract_size_of(const OptionSpec& spec) {
    return std::visit([](const auto& s) { return s.contract_size(); }, spec);
}

std::optional<double> strike_of(const OptionSpec& spec) {
    return std::visit(
        [](const auto& s) -> std::optional<double> {
            if constexpr (std::is_same_v<std::decay_t<decltype(s)>, PayoffSpec>) {
                return std::nullopt;
            } else {
                return s.strike();
            }
        },
        spec);
}

std::optional<OptionType> option_type_of(const OptionSpec& spec) {
    return std::visit(
        [](const auto& s) -> std::optional<OptionType> {
            if constexpr (std::is_same_v<std::decay_t<decltype(s)>, PayoffSpec>) {
                return std::nullopt;
            } else {
                return s.option_type();
            }
        },
        spec);
}

std::optional<std::vector<DateTime>> BarrierSpec::monitoring_schedule(DateTime pricing_date) const {
    if (p_.monitoring == BarrierMonitoring::Continuous) {
        return std::nullopt;
    }
    if (p_.monitoring_dates.has_value()) {
        return p_.monitoring_dates;
    }
    // N+1 points from pricing_date to maturity; drop t=0 to leave N dates at
    // T/N, 2T/N, ..., T (the academic convention for discrete monitoring).
    auto grid = date_range(
        pricing_date, p_.maturity, static_cast<std::size_t>(p_.num_observations.value_or(0)) + 1);
    grid.erase(grid.begin());
    return grid;
}

std::vector<DateTime> contract_event_dates(const OptionSpec& spec, DateTime pricing_date) {
    std::set<DateTime> dates{maturity_of(spec)};
    if (const auto* barrier = std::get_if<BarrierSpec>(&spec)) {
        if (const auto schedule = barrier->monitoring_schedule(pricing_date)) {
            dates.insert(schedule->begin(), schedule->end());
        }
    }
    std::vector<DateTime> result;
    for (const DateTime d : dates) {
        if (d >= pricing_date) {
            result.push_back(d);
        }
    }
    return result;
}

}  // namespace dp
