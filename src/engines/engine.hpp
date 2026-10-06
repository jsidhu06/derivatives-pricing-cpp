#pragma once

#include "dp/exceptions.hpp"

#include <format>
#include <optional>
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
    /// Standard error of `present_value()`; `nullopt` for deterministic
    /// engines.
    [[nodiscard]] virtual std::optional<double> std_error() const { return std::nullopt; }

    [[nodiscard]] virtual double delta() const { return unsupported("delta"); }
    [[nodiscard]] virtual double gamma() const { return unsupported("gamma"); }
    [[nodiscard]] virtual double vega() const { return unsupported("vega"); }
    [[nodiscard]] virtual double theta() const { return unsupported("theta"); }
    [[nodiscard]] virtual double rho() const { return unsupported("rho"); }

    // Monte Carlo estimators (pathwise / likelihood-ratio); only the MC
    // engine provides them, and the facade validates eligibility first.
    [[nodiscard]] virtual double delta_pathwise() const { return unsupported("delta_pathwise"); }
    [[nodiscard]] virtual double vega_pathwise() const { return unsupported("vega_pathwise"); }
    [[nodiscard]] virtual double theta_pathwise() const { return unsupported("theta_pathwise"); }
    [[nodiscard]] virtual double rho_pathwise() const { return unsupported("rho_pathwise"); }
    [[nodiscard]] virtual double delta_lr() const { return unsupported("delta_lr"); }
    [[nodiscard]] virtual double vega_lr() const { return unsupported("vega_lr"); }
    [[nodiscard]] virtual double theta_lr() const { return unsupported("theta_lr"); }
    [[nodiscard]] virtual double rho_lr() const { return unsupported("rho_lr"); }
    [[nodiscard]] virtual double gamma_pathwise_fd(std::optional<double> /*epsilon*/) const {
        return unsupported("gamma_pathwise_fd");
    }

protected:
    PricingEngine() = default;

private:
    [[noreturn]] double unsupported(std::string_view greek) const {
        throw UnsupportedFeatureError(
            std::format("{} does not provide a native {}", name(), greek));
    }
};

}  // namespace dp::engines
