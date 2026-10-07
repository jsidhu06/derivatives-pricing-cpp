#pragma once

#include "dp/enums.hpp"

#include <optional>
#include <variant>

/// Method-specific valuation parameters.
///
/// These are plain aggregates (use designated initialisers) with a
/// `validate()` member; `OptionValuation` validates the parameters it is
/// given at construction.
namespace dp {

/// Parameters for Monte Carlo valuation.
struct MonteCarloParams {
    /// Seed for reproducibility. Fixed by default so bump-and-revalue Greeks
    /// share common random numbers. `nullopt` uses fresh entropy per call.
    std::optional<unsigned long long> random_seed = 42;
    /// Laguerre polynomial degree for Longstaff-Schwartz regression.
    int deg = 3;
    /// Ridge regularisation for the LSM regression.
    double ridge_lambda = 1e-8;
    /// Minimum in-the-money paths required to run the regression.
    int min_itm = 25;
    /// Warn when the MC standard error exceeds this fraction of |PV|.
    std::optional<double> std_error_warn_ratio = 0.1;
    /// Control-variate adjustment for American options using the European
    /// analytical price and the MC European price from the same simulation.
    /// Rejected with discrete dividends: MC drops the spot on each ex-date,
    /// while the analytical reference uses the escrowed model.
    bool control_variate_european = false;
    /// Augment the LSM basis with barrier-distance features (KO only).
    bool barrier_aware_basis = true;

    void validate() const;
};

/// Parameters for binomial-tree valuation.
struct BinomialParams {
    int num_steps = 500;
    /// Hull-style control variate for American vanilla pricing. Valid with
    /// discrete dividends: the tree and the analytical reference both use the
    /// escrowed-dividend model.
    bool control_variate_european = false;

    void validate() const;
};

/// Parameters for finite-difference PDE valuation.
struct PdeParams {
    /// Spatial domain control. Unset (default) sizes the domain from
    /// `sigma * sqrt(T)`: vanillas use the smallest log-interval containing
    /// the spot and forward with 6 standard deviations of ln(S_T) of margin
    /// (capped at `4 * max(spot, strike)` on the uniform Spot grid);
    /// barriers use the legacy `4 * max(spot, strike)` span, widened on the
    /// LogSpot grid when it reaches fewer than 3 standard deviations; custom
    /// payoffs use the legacy span. A value sets the legacy domain exactly:
    /// `[0, m * max(S, K)]` (Spot) or `[max(S, K) / m, m * max(S, K)]`
    /// (LogSpot), and must be > 1. A boundary within 3 standard deviations
    /// of the spot/forward warns.
    std::optional<double> smax_mult = std::nullopt;
    int spot_steps = 200;
    int time_steps = 200;
    /// SOR relaxation parameter for PSOR, in (1, 2).
    double omega = 1.5;
    double tol = 1e-6;
    int max_iter = 20'000;
    PdeMethod method = PdeMethod::CrankNicolson;
    /// Initial Crank-Nicolson intervals replaced by implicit half-steps.
    int rannacher_steps = 2;
    PdeSpaceGrid space_grid = PdeSpaceGrid::Spot;
    PdeEarlyExercise american_solver = PdeEarlyExercise::GaussSeidel;
    /// Hull-style control variate for American pricing. Rejected with
    /// discrete dividends: the PDE drops the spot on each ex-date, while the
    /// analytical reference uses the escrowed model.
    bool control_variate_european = false;

    void validate() const;

    /// Parameters tuned for barrier pricing: fine log-spot grid, with the
    /// time-marching scheme chosen from the monitoring style (Crank-Nicolson
    /// for continuous, explicit-Hull for discrete).
    ///
    /// To override individual presets, assign to the returned struct — the
    /// C++ analogue of the Python `PDEParams.for_barriers(..., **overrides)`:
    ///
    ///     auto params = PdeParams::for_barriers(BarrierMonitoring::Discrete);
    ///     params.time_steps = 5000;  // override; validated at use
    ///
    /// Note: if you raise `spot_steps` without also raising `time_steps`,
    /// the explicit schemes' CFL stability check may warn — keep roughly
    /// `time_steps >= 2 * spot_steps` for sigma ~ 0.25.
    [[nodiscard]] static PdeParams for_barriers(BarrierMonitoring monitoring);
};

/// Any method-specific parameter set.
using ValuationParams = std::variant<MonteCarloParams, BinomialParams, PdeParams>;

}  // namespace dp
