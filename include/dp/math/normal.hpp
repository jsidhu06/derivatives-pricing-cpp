#pragma once

/// Standard normal distribution helpers (thin wrappers over Boost.Math).
namespace dp::math {

/// Standard normal CDF `Φ(x)`. Accurate in both tails.
[[nodiscard]] double normal_cdf(double x) noexcept;

/// Standard normal PDF `φ(x)`.
[[nodiscard]] double normal_pdf(double x) noexcept;

/// Inverse standard normal CDF `Φ⁻¹(p)` for `p` in (0, 1).
/// Returns ±infinity at the boundaries; throws `ValidationError` outside [0, 1].
[[nodiscard]] double normal_quantile(double p);

}  // namespace dp::math
