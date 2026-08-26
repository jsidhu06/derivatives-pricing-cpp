#pragma once

#include "dp/exceptions.hpp"

#include <cmath>
#include <format>
#include <string_view>

/// Internal validation helpers. Not part of the public API.
namespace dp::detail {

/// Throw `ValidationError` unless `value` is finite.
inline void require_finite(double value, std::string_view name) {
    if (!std::isfinite(value)) {
        throw ValidationError(std::format("{} must be finite, got {}", name, value));
    }
}

/// Throw `ValidationError` unless `value` is finite and `> 0`.
inline void require_positive(double value, std::string_view name) {
    require_finite(value, name);
    if (value <= 0.0) {
        throw ValidationError(std::format("{} must be > 0, got {}", name, value));
    }
}

/// Throw `ValidationError` unless `value` is finite and `>= 0`.
inline void require_non_negative(double value, std::string_view name) {
    require_finite(value, name);
    if (value < 0.0) {
        throw ValidationError(std::format("{} must be >= 0, got {}", name, value));
    }
}

}  // namespace dp::detail
