#pragma once

#include "dp/dates.hpp"
#include "dp/diagnostics.hpp"
#include "dp/discount_curve.hpp"

#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

/// Shared fixtures and helpers for the test suite.
namespace dp::test {

inline const DateTime kPricingDate = make_datetime(2025, 1, 1);
inline const DateTime kMaturity = make_datetime(2026, 1, 1);
inline constexpr const char* kCurrency = "USD";
inline constexpr double kSpot = 100.0;
inline constexpr double kStrike = 100.0;
inline constexpr double kRate = 0.05;
inline constexpr double kVol = 0.20;

/// Flat curve whose domain ends exactly at `maturity` (mirrors the Python
/// test helper of the same name).
inline DiscountCurve flat_curve(DateTime pricing_date, DateTime maturity, double rate) {
    return DiscountCurve::flat(rate, year_fraction(pricing_date, maturity));
}

/// RAII capture of `dp::warn` messages. Restores the default handler on exit.
///
/// Messages are stored behind a shared_ptr so the handler (which outlives
/// no-one but must not write through a possibly-const `this`) appends to a
/// buffer the capture merely observes; instances may therefore be `const`.
class WarningCapture {
public:
    WarningCapture() : messages_(std::make_shared<std::vector<std::string>>()) {
        set_warning_handler(
            [buffer = messages_](std::string_view message) { buffer->emplace_back(message); });
    }
    ~WarningCapture() { set_warning_handler({}); }

    WarningCapture(const WarningCapture&) = delete;
    WarningCapture& operator=(const WarningCapture&) = delete;
    WarningCapture(WarningCapture&&) = delete;
    WarningCapture& operator=(WarningCapture&&) = delete;

    [[nodiscard]] const std::vector<std::string>& messages() const noexcept { return *messages_; }
    [[nodiscard]] bool empty() const noexcept { return messages_->empty(); }
    [[nodiscard]] bool any_contains(std::string_view needle) const {
        return std::ranges::any_of(*messages_, [needle](const std::string& m) {
            return m.find(needle) != std::string::npos;
        });
    }

private:
    std::shared_ptr<std::vector<std::string>> messages_;
};

/// Matcher for `REQUIRE_THROWS_MATCHES`: exception message contains `needle`.
inline auto message_contains(const std::string& needle) {
    return Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(needle));
}

}  // namespace dp::test
