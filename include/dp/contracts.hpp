#pragma once

#include "dp/dates.hpp"
#include "dp/enums.hpp"

#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

/// Contract specifications accepted by `OptionValuation`.
///
/// Each spec is constructed from an aggregate `Params` struct (so call sites
/// can use designated initialisers) and is validated and normalised in its
/// constructor. Specs are immutable.
namespace dp {

// ── Vanilla ─────────────────────────────────────────────────────────

/// Plain call/put with European or American exercise.
class VanillaSpec {
public:
    struct Params {
        OptionType option_type;
        ExerciseType exercise_type;
        double strike;
        DateTime maturity;
        /// Contract currency. When unset the underlying's currency is used.
        std::optional<std::string> currency = std::nullopt;
        /// Contract multiplier. Not applied by `OptionValuation` (per-unit
        /// values); intended for portfolio-level position sizing.
        double contract_size = 100.0;
    };

    explicit VanillaSpec(Params params);

    [[nodiscard]] OptionType option_type() const noexcept { return p_.option_type; }
    [[nodiscard]] ExerciseType exercise_type() const noexcept { return p_.exercise_type; }
    [[nodiscard]] double strike() const noexcept { return p_.strike; }
    [[nodiscard]] DateTime maturity() const noexcept { return p_.maturity; }
    [[nodiscard]] const std::optional<std::string>& currency() const noexcept {
        return p_.currency;
    }
    [[nodiscard]] double contract_size() const noexcept { return p_.contract_size; }

private:
    Params p_;
};

// ── Custom payoff ───────────────────────────────────────────────────

/// Affine boundary model on one wing: `payoff(S) ~ slope * S + intercept`.
/// Used by the PDE engine at the truncated computational boundary.
struct WingBoundary {
    double slope;
    double intercept;
};

/// Left/right affine boundary models for a custom payoff.
struct PayoffBoundaryModel {
    WingBoundary left;
    WingBoundary right;
};

/// Scalar payoff as a function of spot.
using PayoffFunction = std::function<double(double spot)>;

/// Single-contract custom payoff (e.g. capped combinations). American
/// exercise compares intrinsic vs continuation on the full payoff.
class PayoffSpec {
public:
    struct Params {
        ExerciseType exercise_type;
        DateTime maturity;
        PayoffFunction payoff_fn;
        std::optional<std::string> currency = std::nullopt;
        double contract_size = 100.0;
        /// Explicit affine boundary models for PDE boundary conditions. When
        /// unset the PDE engine fits them numerically from `payoff_fn`.
        std::optional<PayoffBoundaryModel> boundary_model = std::nullopt;
    };

    explicit PayoffSpec(Params params);

    [[nodiscard]] ExerciseType exercise_type() const noexcept { return p_.exercise_type; }
    [[nodiscard]] DateTime maturity() const noexcept { return p_.maturity; }
    [[nodiscard]] const std::optional<std::string>& currency() const noexcept {
        return p_.currency;
    }
    [[nodiscard]] double contract_size() const noexcept { return p_.contract_size; }
    [[nodiscard]] const std::optional<PayoffBoundaryModel>& boundary_model() const noexcept {
        return p_.boundary_model;
    }

    /// Evaluate the payoff at `spot`.
    [[nodiscard]] double payoff(double spot) const { return p_.payoff_fn(spot); }

private:
    Params p_;
};

// ── Barrier ─────────────────────────────────────────────────────────

/// Knock-in / knock-out barrier option with continuous or discrete monitoring.
class BarrierSpec {
public:
    struct Params {
        OptionType option_type;
        ExerciseType exercise_type;
        double strike;
        DateTime maturity;
        double barrier;
        BarrierDirection direction;
        BarrierAction action;
        BarrierMonitoring monitoring = BarrierMonitoring::Continuous;
        /// Cash rebate. Knock-out: paid at hit (or expiry, per `rebate_timing`).
        /// Knock-in: paid at expiry if the barrier is never hit.
        double rebate = 0.0;
        RebateTiming rebate_timing = RebateTiming::AtHit;
        std::optional<std::string> currency = std::nullopt;
        double contract_size = 100.0;
        /// DISCRETE monitoring: N equally spaced observations at
        /// `t = T/N, ..., T` (pricing date excluded). Exclusive with
        /// `monitoring_dates`.
        std::optional<int> num_observations = std::nullopt;
        /// DISCRETE monitoring: explicit, strictly ascending observation dates.
        std::optional<std::vector<DateTime>> monitoring_dates = std::nullopt;
    };

    explicit BarrierSpec(Params params);

    [[nodiscard]] OptionType option_type() const noexcept { return p_.option_type; }
    [[nodiscard]] ExerciseType exercise_type() const noexcept { return p_.exercise_type; }
    [[nodiscard]] double strike() const noexcept { return p_.strike; }
    [[nodiscard]] DateTime maturity() const noexcept { return p_.maturity; }
    [[nodiscard]] double barrier() const noexcept { return p_.barrier; }
    [[nodiscard]] BarrierDirection direction() const noexcept { return p_.direction; }
    [[nodiscard]] BarrierAction action() const noexcept { return p_.action; }
    [[nodiscard]] BarrierMonitoring monitoring() const noexcept { return p_.monitoring; }
    [[nodiscard]] double rebate() const noexcept { return p_.rebate; }
    [[nodiscard]] RebateTiming rebate_timing() const noexcept { return p_.rebate_timing; }
    [[nodiscard]] const std::optional<std::string>& currency() const noexcept {
        return p_.currency;
    }
    [[nodiscard]] double contract_size() const noexcept { return p_.contract_size; }
    [[nodiscard]] std::optional<int> num_observations() const noexcept {
        return p_.num_observations;
    }
    [[nodiscard]] const std::optional<std::vector<DateTime>>& monitoring_dates() const noexcept {
        return p_.monitoring_dates;
    }

    /// Is `spot` on the dead/active side of the barrier? UP: `spot >= barrier`;
    /// DOWN: `spot <= barrier`. Not monitoring-aware.
    [[nodiscard]] bool is_spot_past_barrier(double spot) const noexcept;

    /// Discrete monitoring dates seen from `pricing_date`: the explicit
    /// `monitoring_dates`, or `num_observations` dates at T/N, 2T/N, ..., T
    /// (the academic convention). `nullopt` for continuous monitoring.
    [[nodiscard]] std::optional<std::vector<DateTime>> monitoring_schedule(
        DateTime pricing_date) const;

private:
    Params p_;
};

// ── Union ───────────────────────────────────────────────────────────

/// Any contract spec accepted by `OptionValuation`.
using OptionSpec = std::variant<VanillaSpec, PayoffSpec, BarrierSpec>;

/// Human-readable spec class name (`VanillaSpec`, `PayoffSpec`, `BarrierSpec`).
[[nodiscard]] std::string_view spec_type_name(const OptionSpec& spec);
[[nodiscard]] DateTime maturity_of(const OptionSpec& spec);
[[nodiscard]] ExerciseType exercise_type_of(const OptionSpec& spec);
[[nodiscard]] const std::optional<std::string>& currency_of(const OptionSpec& spec);
[[nodiscard]] double contract_size_of(const OptionSpec& spec);
/// Strike, or nullopt for specs without one (`PayoffSpec`).
[[nodiscard]] std::optional<double> strike_of(const OptionSpec& spec);
/// Call/put direction, or nullopt for specs without one (`PayoffSpec`).
[[nodiscard]] std::optional<OptionType> option_type_of(const OptionSpec& spec);

/// Dates a Monte Carlo grid must contain to value `spec` from
/// `pricing_date`: the maturity plus any discrete barrier monitoring dates,
/// sorted, unique and on or after `pricing_date`. `shared_time_grid`
/// applies it to every spec in its `GridEvents`.
[[nodiscard]] std::vector<DateTime> contract_event_dates(const OptionSpec& spec,
                                                         DateTime pricing_date);

}  // namespace dp
