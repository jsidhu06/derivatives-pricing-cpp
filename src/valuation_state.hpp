#pragma once

#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/exceptions.hpp"
#include "dp/params.hpp"
#include "dp/simulation.hpp"
#include "dp/underlying.hpp"

#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <utility>
#include <variant>

/// Internal: the validated inputs of an `OptionValuation`, shared with the
/// engine it owns. Lives on the heap inside `OptionValuation::Impl`, so
/// references to it stay valid when the facade is moved.
namespace dp::detail {

/// Either a deterministic-engine underlying (`UnderlyingData`) or a Monte
/// Carlo path process (GBM, jump diffusion, or square-root diffusion).
/// Forwards the accessors the alternatives share, so engines are written
/// against one interface.
class Underlying {
public:
    // NOLINTBEGIN(google-explicit-constructor): deliberate implicit lift.
    Underlying(UnderlyingData data) : v_(std::move(data)) {}
    Underlying(GbmProcess process) : v_(std::move(process)) {}
    Underlying(JdProcess process) : v_(std::move(process)) {}
    Underlying(SrdProcess process) : v_(std::move(process)) {}
    // NOLINTEND(google-explicit-constructor)

    [[nodiscard]] bool is_process() const noexcept {
        return !std::holds_alternative<UnderlyingData>(v_);
    }
    /// True for the one process whose paths are log-normal (pathwise/LR
    /// Greeks and the BSM control-variate leg are GBM-only).
    [[nodiscard]] bool is_gbm() const noexcept { return std::holds_alternative<GbmProcess>(v_); }

    [[nodiscard]] std::string_view type_name() const {
        return std::visit(
            [](const auto& u) -> std::string_view {
                using T = std::decay_t<decltype(u)>;
                if constexpr (std::is_same_v<T, UnderlyingData>) {
                    return "UnderlyingData";
                } else if constexpr (std::is_same_v<T, GbmProcess>) {
                    return "GbmProcess";
                } else if constexpr (std::is_same_v<T, JdProcess>) {
                    return "JdProcess";
                } else {
                    return "SrdProcess";
                }
            },
            v_);
    }

    [[nodiscard]] const std::variant<UnderlyingData, GbmProcess, JdProcess, SrdProcess>& value()
        const noexcept {
        return v_;
    }

    template <typename F>
    decltype(auto) visit(F&& f) const {
        return std::visit(std::forward<F>(f), v_);
    }

    /// Run the path process; throws `ConfigurationError` for `UnderlyingData`.
    [[nodiscard]] SimulationResult simulate(std::optional<std::uint64_t> seed) const {
        return visit([&](const auto& u) -> SimulationResult {
            if constexpr (requires { u.simulate(seed); }) {
                return u.simulate(seed);
            } else {
                throw ConfigurationError("underlying is not a path process");
            }
        });
    }

    [[nodiscard]] double initial_value() const {
        return visit([](const auto& u) { return u.initial_value(); });
    }
    [[nodiscard]] double volatility() const {
        return visit([](const auto& u) { return u.volatility(); });
    }
    [[nodiscard]] const DividendSchedule& discrete_dividends() const {
        return visit(
            [](const auto& u) -> const DividendSchedule& { return u.discrete_dividends(); });
    }
    [[nodiscard]] const std::optional<DiscountCurve>& dividend_curve() const {
        return visit([](const auto& u) -> const std::optional<DiscountCurve>& {
            return u.dividend_curve();
        });
    }
    [[nodiscard]] DateTime pricing_date() const {
        return visit([](const auto& u) { return u.pricing_date(); });
    }
    [[nodiscard]] const DiscountCurve& discount_curve() const {
        return visit([](const auto& u) -> const DiscountCurve& { return u.discount_curve(); });
    }
    [[nodiscard]] const std::string& currency() const {
        return visit([](const auto& u) -> const std::string& { return u.currency(); });
    }
    [[nodiscard]] DayCountConvention day_count_convention() const {
        return visit([](const auto& u) { return u.day_count_convention(); });
    }

    [[nodiscard]] Underlying with_initial_value(double value) const {
        return visit([&](const auto& u) { return Underlying(u.with_initial_value(value)); });
    }
    [[nodiscard]] Underlying with_volatility(double value) const {
        return visit([&](const auto& u) { return Underlying(u.with_volatility(value)); });
    }
    [[nodiscard]] Underlying with_pricing_date(DateTime date) const {
        return visit([&](const auto& u) { return Underlying(u.with_pricing_date(date)); });
    }
    [[nodiscard]] Underlying with_discount_curve(DiscountCurve curve) const {
        return visit(
            [&](const auto& u) { return Underlying(u.with_discount_curve(std::move(curve))); });
    }

    /// Deterministic-engine view (for the BSM control-variate leg). Lossless
    /// for `UnderlyingData` and GBM processes; other processes would silently
    /// drop their jump / mean-reversion parameters, so they are rejected.
    [[nodiscard]] UnderlyingData as_underlying_data() const {
        if (const auto* data = std::get_if<UnderlyingData>(&v_)) {
            return *data;
        }
        if (const auto* p = std::get_if<GbmProcess>(&v_)) {
            return {p->initial_value(),
                    p->volatility(),
                    p->market_data(),
                    p->discrete_dividends(),
                    p->dividend_curve()};
        }
        throw ConfigurationError(
            std::format("Expected UnderlyingData or GbmProcess, got {}.", type_name()));
    }

private:
    std::variant<UnderlyingData, GbmProcess, JdProcess, SrdProcess> v_;
};

struct ValuationState {
    Underlying underlying;
    OptionSpec spec;
    PricingMethod pricing_method;
    std::optional<ValuationParams> params;
    std::string currency;
    std::optional<OptionType> option_type;

    [[nodiscard]] DateTime maturity() const { return maturity_of(spec); }
    [[nodiscard]] ExerciseType exercise_type() const { return exercise_type_of(spec); }
    [[nodiscard]] std::optional<double> strike() const { return strike_of(spec); }
    [[nodiscard]] DateTime pricing_date() const { return underlying.pricing_date(); }
    [[nodiscard]] const DiscountCurve& discount_curve() const {
        return underlying.discount_curve();
    }
    [[nodiscard]] DayCountConvention day_count_convention() const {
        return underlying.day_count_convention();
    }
    [[nodiscard]] double maturity_year_fraction() const {
        return year_fraction(pricing_date(), maturity(), day_count_convention());
    }
};

}  // namespace dp::detail
