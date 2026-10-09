#pragma once

#include "dp/contracts.hpp"
#include "dp/enums.hpp"
#include "dp/params.hpp"

#include "valuation_state.hpp"

#include <Eigen/Core>

#include <optional>
#include <string_view>
#include <utility>
#include <vector>

/// Internal: finite-difference numerics shared by the vanilla and barrier
/// PDE engines — tau/space grids, dividend jump conditions, boundary models,
/// operator coefficients, time-step kernels and the grid-Greek stencils.
namespace dp::engines::pde {

using Eigen::ArrayXd;
using Eigen::Index;

inline constexpr double kDaysPerYear = 365.0;

/// `smax_mult` used when `PdeParams::smax_mult` is unset: the fixed span of
/// custom payoffs and barriers (before widening), and the vanilla Spot grid
/// cap.
inline constexpr double kDefaultSmaxMult = 4.0;

[[nodiscard]] inline double resolved_smax_mult(const PdeParams& params) {
    return params.smax_mult.value_or(kDefaultSmaxMult);
}

/// Far-field domain `(smin, smax)` of a PDE grid.
using Domain = std::pair<double, double>;

/// Round to 12 decimal places, ties to even (matches `np.round(x, 12)`),
/// so taus from different arithmetic paths compare equal.
[[nodiscard]] double round12(double x);

/// `np.linspace(start, stop, n + 1)`: `start + i * step` with the endpoint
/// pinned exactly.
[[nodiscard]] ArrayXd linspace(double start, double stop, int intervals);

/// `np.interp(x, xs, ys)` for one query point: linear interpolation with the
/// end values held flat outside the range.
[[nodiscard]] double interp_linear(double x, const ArrayXd& xs, const ArrayXd& ys);

/// Thomas algorithm for a tridiagonal system; diagonals are named as in the
/// reference implementation (lower/diag/upper of lengths n-1, n, n-1).
[[nodiscard]] ArrayXd solve_tridiagonal_thomas(const ArrayXd& lower,
                                               ArrayXd diag,
                                               const ArrayXd& upper,
                                               ArrayXd rhs);

/// Tau grid snapping extra times (dividends) onto the uniform base grid.
[[nodiscard]] ArrayXd build_tau_grid(double time_to_maturity,
                                     int time_steps,
                                     const std::vector<double>& extras);

/// (tau, amount) for dividends with pricing_date <= ex_date <= maturity,
/// amounts merged per rounded tau, sorted ascending.
[[nodiscard]] std::vector<std::pair<double, double>> dividend_tau_schedule(
    const detail::ValuationState& state);

/// Cash dividend jump condition V(S, t^-) = V(S - D, t^+).
void apply_dividend_jump(ArrayXd& values,
                         const ArrayXd& grid,
                         double amount,
                         PdeSpaceGrid space_grid);

// ── Affine wing boundary models (custom payoffs) ────────────────────

[[nodiscard]] WingBoundary fit_affine_boundary_model(const PayoffSpec& payoff,
                                                     std::string_view wing,
                                                     const ArrayXd& spot_samples);

/// Dirichlet boundary values at S = smin (left) and S = smax (right).
[[nodiscard]] std::pair<double, double> boundary_values(
    std::optional<OptionType> option_type,
    std::optional<double> strike,
    double smin,
    double smax,
    double df_fwd,
    double dq_fwd,
    bool early_exercise,
    const PayoffSpec* payoff,
    const std::optional<PayoffBoundaryModel>& model);

// ── Grids ───────────────────────────────────────────────────────────

struct SpaceGridBuild {
    ArrayXd grid;     ///< PDE coordinate (spot, or log-spot z).
    ArrayXd spots;    ///< exp(grid) on log grids; == grid on spot grids.
    double dx = 0.0;  ///< dS or dz.
};

/// Log-spot grid. Explicit schemes target Hull's stability spacing
/// dz = vol * sqrt(3 dt) when it covers the domain. When `anchor_spot` is
/// given the grid is sized so the anchor sits exactly on an interior node
/// (or halfway between two nodes with `anchor_half_step`): CN/IMPLICIT grow
/// dz on the binding half, explicit schemes keep dz fixed by stability and
/// shift the grid in place instead.
[[nodiscard]] SpaceGridBuild build_log_grid(double spot,
                                            double strike,
                                            double time_to_maturity,
                                            double volatility,
                                            double smax_mult,
                                            int spot_steps,
                                            int time_steps,
                                            PdeMethod method,
                                            std::optional<double> anchor_spot = std::nullopt,
                                            bool anchor_half_step = false,
                                            std::optional<Domain> domain = std::nullopt);

// ── Domain sizing ───────────────────────────────────────────────────

/// ln F(0, T) with the continuous yield and cash dividends (escrowed PV).
[[nodiscard]] double log_forward(const detail::ValuationState& state);

/// Auto vanilla domain: the smallest log-interval containing the spot and
/// the forward, each with 6 standard deviations of ln(S_T) of margin. The
/// strike need not lie inside: the forward-intrinsic boundary conditions
/// (`boundary_values`) are exact far from it, and a boundary error 6
/// standard deviations away decays by ~erfc(6 / sqrt(2)) ~ 2e-9 before
/// reaching the spot. Uncapped on the LogSpot grid (wider costs little
/// resolution in log space); capped at `4 * max(S, K)` on the Spot grid.
[[nodiscard]] Domain fitted_vanilla_domain(const detail::ValuationState& state,
                                           double strike,
                                           double log_fwd,
                                           PdeSpaceGrid space_grid);

/// Barrier far-field targets: the fixed span (`[0, smax_mult * max(S, K)]`
/// Spot, `[max(S, K) / smax_mult, smax_mult * max(S, K)]` LogSpot, with
/// `smax_mult` defaulting to 4). With
/// `smax_mult` unset on the LogSpot grid, a side reaching fewer than 3
/// standard deviations from the spot/forward is widened to 6; the domain is
/// never narrowed, so it always contains the strike. A continuously
/// monitored barrier replaces its side in the caller.
[[nodiscard]] Domain barrier_domain(const detail::ValuationState& state,
                                    double strike,
                                    double log_fwd,
                                    const PdeParams& params);

/// Warn when a far-field boundary is within 3 standard deviations of the
/// distribution (the nearer of spot and forward). `nullopt` skips a side (a
/// continuously monitored barrier is an exact boundary).
void warn_if_domain_truncated(std::optional<double> smin,
                              std::optional<double> smax,
                              double spot,
                              double log_fwd,
                              double time_to_maturity,
                              double volatility);

/// Warn when the grid has fewer than 3 nodes per standard deviation of
/// ln(S_T) at the spot.
void warn_if_under_resolved(double spot,
                            double spacing,
                            double time_to_maturity,
                            double volatility,
                            PdeSpaceGrid space_grid);

/// Uniform spot grid, optionally aligning an anchor to a node or midpoint.
[[nodiscard]] SpaceGridBuild build_spot_grid(double smin,
                                             double smax,
                                             int spot_steps,
                                             std::optional<double> anchor_spot = std::nullopt,
                                             bool anchor_half_step = false);

// ── Spatial operator coefficients ───────────────────────────────────

struct OperatorCoeffs {
    ArrayXd gamma;  ///< Sub-diagonal weight.
    ArrayXd beta;   ///< Diagonal weight.
    ArrayXd alpha;  ///< Super-diagonal weight.
};

/// `implicit_discounting` (Hull's explicit scheme) excludes the rV term from
/// beta; it is applied as a 1/(1 + r dt) divisor in the step instead.
[[nodiscard]] OperatorCoeffs spot_operator_coeffs(const ArrayXd& interior_spots,
                                                  double dS,
                                                  double r,
                                                  double q,
                                                  double volatility,
                                                  bool implicit_discounting);

[[nodiscard]] OperatorCoeffs log_operator_coeffs(
    double dz, double r, double q, double volatility, bool implicit_discounting, Index size);

// ── Time steps ──────────────────────────────────────────────────────

struct TimeStep {
    double tau_start;
    double tau_end;
    PdeMethod method;
};

/// Rannacher smoothing: the first `rannacher_steps` CN intervals become two
/// implicit half-steps each (Pooley-Vetzal-Forsyth 2003).
[[nodiscard]] std::vector<TimeStep> build_time_step_schedule(const ArrayXd& tau_grid,
                                                             PdeMethod method,
                                                             int rannacher_steps);

void check_explicit_spot_stability(const ArrayXd& tau_grid,
                                   double volatility,
                                   double smax,
                                   double dS,
                                   double time_to_maturity,
                                   const DiscountCurve& discount_curve,
                                   const std::optional<DiscountCurve>& dividend_curve,
                                   bool implicit_discounting);

// ── Time-step kernels ───────────────────────────────────────────────

/// Forward-Euler step. `r_dt > 0` applies Hull's implicit discounting of rV.
[[nodiscard]] ArrayXd explicit_step(const ArrayXd& v_prev,
                                    const OperatorCoeffs& scaled,
                                    double left,
                                    double right,
                                    const ArrayXd* intrinsic,
                                    double r_dt);

/// Outcome of one projected-SOR solve.
struct PsorResult {
    int iterations;
    bool converged;
};

/// PSOR iteration counts over one PDE solve: logs them at debug level and
/// warns when any time step failed to converge.
class PsorStats {
public:
    void record(const std::optional<PsorResult>& result);
    void report(std::string_view label, int max_iter, double tol) const;

private:
    int steps_ = 0;
    long total_iters_ = 0;
    int max_iters_ = 0;
    int not_converged_ = 0;
};

/// One implicit or Crank-Nicolson step, optionally with early exercise.
/// Returns the PSOR outcome, or nullopt when PSOR was not used.
std::optional<PsorResult> implicit_cn_step(const ArrayXd& v_prev,
                                           ArrayXd& v,
                                           OperatorCoeffs scaled,
                                           double left,
                                           double right,
                                           PdeMethod method,
                                           const ArrayXd* intrinsic,
                                           PdeEarlyExercise american_solver,
                                           double omega,
                                           double tol,
                                           int max_iter);

// ── Parameters and validation ───────────────────────────────────────

[[nodiscard]] const PdeParams& pde_params(const detail::ValuationState& state);

void validate_fd_inputs(const detail::ValuationState& state,
                        const PdeParams& params,
                        double time_to_maturity,
                        bool early_exercise,
                        bool has_custom_payoff);

// ── Grid Greek stencils ─────────────────────────────────────────────

/// Nearest interior grid index to `spot` (searchsorted, clamped).
[[nodiscard]] Index spot_grid_index(const ArrayXd& S, double spot);

/// Parabolic-Lagrange first derivative evaluated exactly at `spot`.
[[nodiscard]] double grid_delta_at_spot(const ArrayXd& S, const ArrayXd& V, Index j, double spot);

[[nodiscard]] double grid_gamma_safe(const ArrayXd& S, const ArrayXd& V, Index j, double spot);

/// Parabolic-Lagrange interpolation of V at exactly `spot`.
[[nodiscard]] double grid_value_at_spot(const ArrayXd& S, const ArrayXd& V, Index j, double spot);

}  // namespace dp::engines::pde
