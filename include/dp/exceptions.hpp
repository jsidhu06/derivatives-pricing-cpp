#pragma once

#include <stdexcept>
#include <string>

/// Exception hierarchy for the derivatives_pricing library.
///
/// All library-specific exceptions derive from `DerivativesPricingError`, so
/// callers can catch any library error with a single handler:
///
///     try {
///         auto pv = valuation.present_value();
///     } catch (const dp::DerivativesPricingError& e) {
///         log_error(e.what());
///     }
namespace dp {

/// Base exception for all library errors.
class DerivativesPricingError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// ── Input validation ────────────────────────────────────────────────

/// Invalid input values (out-of-range, non-finite, mutually exclusive inputs, ...).
class ValidationError : public DerivativesPricingError {
public:
    using DerivativesPricingError::DerivativesPricingError;
};

/// A public API was configured inconsistently (e.g. wrong parameter type for
/// the chosen pricing method). Most type errors are compile-time in C++, so
/// this is rarer than in the Python reference implementation.
class ConfigurationError : public DerivativesPricingError {
public:
    using DerivativesPricingError::DerivativesPricingError;
};

// ── Feature support ─────────────────────────────────────────────────

/// Requested feature combination is not (yet) supported.
class UnsupportedFeatureError : public DerivativesPricingError {
public:
    using DerivativesPricingError::DerivativesPricingError;
};

// ── Numerical issues ────────────────────────────────────────────────

/// Base for errors arising from numerical computation.
class NumericalError : public DerivativesPricingError {
public:
    using DerivativesPricingError::DerivativesPricingError;
};

/// Model parameters imply an arbitrage (e.g. risk-neutral probability outside [0, 1]).
class ArbitrageViolationError : public NumericalError {
public:
    using NumericalError::NumericalError;
};

/// An iterative solver failed to converge within the allowed tolerance / iterations.
class ConvergenceError : public NumericalError {
public:
    using NumericalError::NumericalError;
};

/// A numerical scheme's stability conditions are violated.
class StabilityError : public NumericalError {
public:
    using NumericalError::NumericalError;
};

}  // namespace dp
