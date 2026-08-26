#pragma once

#include "dp/exceptions.hpp"

#include <format>
#include <string_view>

/// Internal: interface implemented by every pricing engine.
///
/// Engines are owned by `OptionValuation`, hold a reference to its
/// `ValuationState`, and may cache their own solves (`const` methods with
/// `mutable` state). Native Greeks default to "unsupported"; the facade only
/// calls them after resolving that the engine's method provides them.
namespace dp::engines {

class PricingEngine {
public:
    virtual ~PricingEngine() = default;
    PricingEngine(const PricingEngine&) = delete;
    PricingEngine& operator=(const PricingEngine&) = delete;
    PricingEngine(PricingEngine&&) = delete;
    PricingEngine& operator=(PricingEngine&&) = delete;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual double present_value() const = 0;

    [[nodiscard]] virtual double delta() const { return unsupported("delta"); }
    [[nodiscard]] virtual double gamma() const { return unsupported("gamma"); }
    [[nodiscard]] virtual double vega() const { return unsupported("vega"); }
    [[nodiscard]] virtual double theta() const { return unsupported("theta"); }
    [[nodiscard]] virtual double rho() const { return unsupported("rho"); }

protected:
    PricingEngine() = default;

private:
    [[noreturn]] double unsupported(std::string_view greek) const {
        throw UnsupportedFeatureError(
            std::format("{} does not provide a native {}", name(), greek));
    }
};

}  // namespace dp::engines
