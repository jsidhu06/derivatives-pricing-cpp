#pragma once

/// Standard normal distribution helpers (thin wrappers over Boost.Math, plus
/// Wichura's AS241 inverse CDF for simulation hot loops).
namespace dp::math {

/// Standard normal CDF `Φ(x)`. Accurate in both tails.
[[nodiscard]] double normal_cdf(double x) noexcept;

/// Standard normal PDF `φ(x)`.
[[nodiscard]] double normal_pdf(double x) noexcept;

/// Inverse standard normal CDF `Φ⁻¹(p)` for `p` in (0, 1).
/// Returns ±infinity at the boundaries; throws `ValidationError` outside [0, 1].
[[nodiscard]] double normal_quantile(double p);

/// Inverse standard normal CDF by Wichura's AS241 (PPND16) rational
/// approximations (Applied Statistics 37(3), 1988) — the algorithm behind
/// R's `qnorm`. Relative accuracy ~1e-16 across the double range, several
/// times faster than `normal_quantile`, which makes it the inverse-CDF
/// transform for Monte Carlo normals.
///
/// Precondition: `p` in the open interval (0, 1). No validation — callers
/// feed it open-interval uniforms.
[[nodiscard]] double normal_quantile_fast(double p) noexcept;

}  // namespace dp::math
