#pragma once

#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"
#include "dp/params.hpp"
#include "dp/underlying.hpp"

#include <optional>
#include <string>

/// Internal: the validated inputs of an `OptionValuation`, shared with the
/// engine it owns. Lives on the heap inside `OptionValuation::Impl`, so
/// references to it stay valid when the facade is moved.
namespace dp::detail {

struct ValuationState {
    UnderlyingData underlying;
    OptionSpec spec;
    PricingMethod pricing_method;
    std::optional<ValuationParams> params;
    std::string currency;
    std::optional<OptionType> option_type;

    [[nodiscard]] DateTime maturity() const { return maturity_of(spec); }
    [[nodiscard]] ExerciseType exercise_type() const { return exercise_type_of(spec); }
    [[nodiscard]] std::optional<double> strike() const { return strike_of(spec); }
    [[nodiscard]] DateTime pricing_date() const noexcept { return underlying.pricing_date(); }
    [[nodiscard]] const DiscountCurve& discount_curve() const noexcept {
        return underlying.discount_curve();
    }
    [[nodiscard]] DayCountConvention day_count_convention() const noexcept {
        return underlying.day_count_convention();
    }
    [[nodiscard]] double maturity_year_fraction() const {
        return year_fraction(pricing_date(), maturity(), day_count_convention());
    }
};

}  // namespace dp::detail
