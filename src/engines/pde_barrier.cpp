#include "engines/pde_barrier.hpp"

#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/diagnostics.hpp"
#include "dp/exceptions.hpp"
#include "dp/params.hpp"

#include "detail/barrier.hpp"
#include "engines/pde_numerics.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <variant>
#include <vector>

namespace dp::engines {

using pde::apply_dividend_jump;
using pde::ArrayXd;
using pde::boundary_values;
using pde::build_log_grid;
using pde::build_spot_grid;
using pde::build_tau_grid;
using pde::build_time_step_schedule;
using pde::check_explicit_spot_stability;
using pde::dividend_tau_schedule;
using pde::explicit_step;
using pde::grid_delta_at_spot;
using pde::grid_gamma_safe;
using pde::implicit_cn_step;
using pde::Index;
using pde::interp_linear;
using pde::kDaysPerYear;
using pde::linspace;
using pde::log_operator_coeffs;
using pde::OperatorCoeffs;
using pde::pde_params;
using pde::round12;
using pde::solve_tridiagonal_thomas;
using pde::SpaceGridBuild;
using pde::spot_grid_index;
using pde::spot_operator_coeffs;
using pde::validate_fd_inputs;

namespace {

using Mask = Eigen::Array<bool, Eigen::Dynamic, 1>;

bool is_explicit(PdeMethod method) {
    return method == PdeMethod::Explicit || method == PdeMethod::ExplicitHull;
}

/// Log-spot grid for a continuous knock-out: the barrier is a boundary node.
/// Explicit-family schemes preserve Hull's dz scale; CN/IMPLICIT honor
/// spot_steps directly.
SpaceGridBuild build_ko_continuous_log_grid(double smin_target,
                                            double smax_target,
                                            BarrierDirection direction,
                                            double volatility,
                                            double time_to_maturity,
                                            int spot_steps,
                                            int time_steps,
                                            PdeMethod method) {
    const bool explicit_scheme = is_explicit(method);
    double zlo = 0.0;
    double zhi = 0.0;
    if (direction == BarrierDirection::Down) {
        zlo = std::log(smin_target);
        zhi = std::log(smax_target);
        if (explicit_scheme) {
            const double dz_hull = volatility * std::sqrt(3.0 * (time_to_maturity / time_steps));
            const double grid_width = spot_steps * dz_hull;
            if ((zhi - zlo) <= grid_width) {
                zhi = zlo + grid_width;
            }
        }
    } else {
        zhi = std::log(smax_target);
        zlo = std::log(smin_target);
        if (explicit_scheme) {
            const double dz_hull = volatility * std::sqrt(3.0 * (time_to_maturity / time_steps));
            const double grid_width = spot_steps * dz_hull;
            if ((zhi - zlo) <= grid_width) {
                zlo = zhi - grid_width;
            }
        }
    }
    const ArrayXd z = linspace(zlo, zhi, spot_steps);
    return {.grid = z, .spots = z.exp(), .dx = (zhi - zlo) / spot_steps};
}

/// One PDE time-step restricted to the contiguous sub-interior
/// `[j_lo, j_hi]` — used by the coupled KI solve's inactive surface.
/// `scaled` holds full-interior coefficients (node i -> index i-1).
/// Only the explicit branch writes the sub-grid boundary nodes into the
/// result (mirroring the reference); implicit/CN callers re-impose them.
ArrayXd subgrid_pde_step(const ArrayXd& v,
                         const ArrayXd& v_prev,
                         Index j_lo,
                         Index j_hi,
                         const OperatorCoeffs& scaled,
                         double left,
                         double right,
                         PdeMethod method,
                         double r_dt) {
    ArrayXd v_new = v;
    const Index m = j_hi - j_lo + 1;
    const ArrayXd a_s = scaled.gamma.segment(j_lo - 1, m);
    const ArrayXd b_s = scaled.beta.segment(j_lo - 1, m);
    const ArrayXd c_s = scaled.alpha.segment(j_lo - 1, m);

    if (is_explicit(method)) {
        v_new.segment(j_lo, m) =
            (-a_s * v_prev.segment(j_lo - 1, m) + (1.0 - b_s) * v_prev.segment(j_lo, m) -
             c_s * v_prev.segment(j_lo + 1, m)) /
            (1.0 + r_dt);
        v_new[j_lo - 1] = left;
        v_new[j_hi + 1] = right;
    } else if (method == PdeMethod::Implicit) {
        const ArrayXd diag = 1.0 + b_s;
        ArrayXd rhs = v_prev.segment(j_lo, m);
        rhs[0] -= a_s[0] * left;
        rhs[m - 1] -= c_s[m - 1] * right;
        v_new.segment(j_lo, m) =
            solve_tridiagonal_thomas(a_s.tail(m - 1), diag, c_s.head(m - 1), rhs);
    } else {
        const ArrayXd a_h = a_s * 0.5;
        const ArrayXd b_h = b_s * 0.5;
        const ArrayXd c_h = c_s * 0.5;
        const ArrayXd diag = 1.0 + b_h;
        ArrayXd rhs = -a_h * v_prev.segment(j_lo - 1, m) + (1.0 - b_h) * v_prev.segment(j_lo, m) -
                      c_h * v_prev.segment(j_lo + 1, m);
        rhs[0] -= a_h[0] * left;
        rhs[m - 1] -= c_h[m - 1] * right;
        v_new.segment(j_lo, m) =
            solve_tridiagonal_thomas(a_h.tail(m - 1), diag, c_h.head(m - 1), rhs);
    }
    return v_new;
}

/// Grid nodes on the knocked side of the barrier (at or beyond it).
Mask knocked_nodes(const ArrayXd& S, double barrier, bool is_up) {
    return is_up ? (S >= barrier).eval() : (S <= barrier).eval();
}

/// Continuous knock-in: on the hit side of the barrier node the option is
/// already active, so the inactive surface takes the active values there.
void couple_hit_side(ArrayXd& v_inact, const ArrayXd& v_act, Index j_barrier, bool is_up) {
    if (!is_up) {
        v_inact.head(j_barrier + 1) = v_act.head(j_barrier + 1);
    } else {
        v_inact.tail(v_act.size() - j_barrier) = v_act.tail(v_act.size() - j_barrier);
    }
}

/// Vanilla payoff on the grid spots.
ArrayXd vanilla_payoff(const ArrayXd& S, double strike, bool is_call) {
    ArrayXd payoff(S.size());
    if (is_call) {
        payoff = (S - strike).cwiseMax(0.0);
    } else {
        payoff = (strike - S).cwiseMax(0.0);
    }
    return payoff;
}

/// Value a knock-out takes where the barrier is hit: the rebate if paid at
/// hit, else the rebate paid at maturity discounted by `df_fwd` (the
/// discount factor from the hit time to maturity). Zero without a rebate.
double knock_out_rebate_value(const BarrierSpec& spec, double df_fwd) {
    if (spec.rebate() == 0.0) {
        return 0.0;
    }
    return spec.rebate_timing() == RebateTiming::AtHit ? spec.rebate() : spec.rebate() * df_fwd;
}

/// Dates on which both barrier solves act besides the regular time steps.
/// Taus are rounded to 12 digits, as on the time grid.
struct EventSchedule {
    /// Cash dividends strictly inside (0, ttm), applied as jumps in the loop.
    std::map<double, double> dividends;
    /// Dividend going ex at maturity: applied to the terminal slice.
    std::optional<double> maturity_dividend;
    /// Dividend going ex on the pricing date: the price is read at spot - D.
    std::optional<double> pricing_dividend;
    std::set<double> monitoring;
    /// Time grid holding every dividend and (discrete) monitoring tau.
    ArrayXd tau_grid;

    [[nodiscard]] bool is_monitoring(double tau) const { return monitoring.contains(round12(tau)); }
    [[nodiscard]] std::optional<double> dividend_at(double tau) const {
        const auto it = dividends.find(round12(tau));
        return it == dividends.end() ? std::nullopt : std::optional<double>(it->second);
    }
};

EventSchedule event_schedule(const detail::ValuationState& state,
                             const std::vector<double>& monitoring_taus,
                             bool continuous,
                             double ttm,
                             int time_steps) {
    EventSchedule events;
    for (const auto& [tau, amount] : dividend_tau_schedule(state)) {
        events.dividends.emplace(tau, amount);
    }
    if (const auto it = events.dividends.find(0.0); it != events.dividends.end()) {
        events.maturity_dividend = it->second;
        events.dividends.erase(it);
    }
    if (const auto it = events.dividends.find(round12(ttm)); it != events.dividends.end()) {
        events.pricing_dividend = it->second;
        events.dividends.erase(it);
    }
    for (const double tau : monitoring_taus) {
        events.monitoring.insert(round12(tau));
    }
    std::vector<double> extra_taus;
    extra_taus.reserve(events.dividends.size());
    for (const auto& [tau, amount] : events.dividends) {
        extra_taus.push_back(tau);
    }
    if (!continuous) {
        extra_taus.insert(extra_taus.end(), monitoring_taus.begin(), monitoring_taus.end());
    }
    events.tau_grid = build_tau_grid(ttm, time_steps, extra_taus);
    return events;
}

/// Per-step operator and discount factors, identical for every barrier solve.
struct StepCoefficients {
    double d_tau = 0.0;
    /// Operator coefficients scaled by -d_tau.
    OperatorCoeffs scaled;
    /// Discount factors from the end of this step to maturity.
    double df_fwd = 1.0;
    double dq_fwd = 1.0;
    /// Explicit-Hull discounting term (0 for the other schemes).
    double r_dt = 0.0;
};

/// Builds `StepCoefficients` for one solve. Non-owning, like the solve's
/// other locals: the curves, volatility and maturity come from the valuation
/// state that owns them (it outlives every solve), and `S` must outlive the
/// operator (it is the solve's own grid).
class StepOperator {
public:
    StepOperator(const ArrayXd& S,
                 double dx,
                 const detail::ValuationState& state,
                 const PdeParams& params)
        : spots_(&S),
          dx_(dx),
          spot_steps_(params.spot_steps),
          volatility_(state.underlying.volatility()),
          ttm_(state.maturity_year_fraction()),
          space_grid_(params.space_grid),
          curve_(&state.discount_curve()),
          q_curve_(detail::curve_or_null(state.underlying.dividend_curve())),
          df_full_(curve_->df(ttm_)),
          dq_full_(q_curve_ != nullptr ? q_curve_->df(ttm_) : 1.0) {}

    [[nodiscard]] StepCoefficients operator()(const pde::TimeStep& step) const {
        const double d_tau = step.tau_end - step.tau_start;
        const double t_prev = ttm_ - step.tau_start;
        const double t_curr = ttm_ - step.tau_end;
        const double r = curve_->forward_rate(t_curr, t_prev);
        const double q = q_curve_ != nullptr ? q_curve_->forward_rate(t_curr, t_prev) : 0.0;
        const bool implicit_discounting = step.method == PdeMethod::ExplicitHull;
        const OperatorCoeffs coeffs =
            space_grid_ == PdeSpaceGrid::Spot
                ? spot_operator_coeffs(spots_->segment(1, spot_steps_ - 1),
                                       dx_,
                                       r,
                                       q,
                                       volatility_,
                                       implicit_discounting)
                : log_operator_coeffs(
                      dx_, r, q, volatility_, implicit_discounting, spot_steps_ - 1);
        return {.d_tau = d_tau,
                .scaled = {.gamma = -d_tau * coeffs.gamma,
                           .beta = -d_tau * coeffs.beta,
                           .alpha = -d_tau * coeffs.alpha},
                .df_fwd = df_full_ / curve_->df(t_curr),
                .dq_fwd = q_curve_ != nullptr ? dq_full_ / q_curve_->df(t_curr) : 1.0,
                .r_dt = implicit_discounting ? r * d_tau : 0.0};
    }

private:
    const ArrayXd* spots_;
    double dx_;
    int spot_steps_;
    double volatility_;
    double ttm_;
    PdeSpaceGrid space_grid_;
    const DiscountCurve* curve_;
    const DiscountCurve* q_curve_;
    double df_full_;
    double dq_full_;
};

/// One time step of the full-grid PDE, explicit or implicit/CN (PSOR when
/// `intrinsic` is set). PSOR iterations are recorded when `psor_stats` is set.
void advance(const ArrayXd& v_prev,
             ArrayXd& v,
             const StepCoefficients& c,
             double left,
             double right,
             PdeMethod method,
             const ArrayXd* intrinsic,
             const PdeParams& params,
             pde::PsorStats* psor_stats) {
    if (is_explicit(method)) {
        v = explicit_step(v_prev, c.scaled, left, right, intrinsic, c.r_dt);
        return;
    }
    const auto result = implicit_cn_step(v_prev,
                                         v,
                                         c.scaled,
                                         left,
                                         right,
                                         method,
                                         intrinsic,
                                         params.american_solver,
                                         params.omega,
                                         params.tol,
                                         params.max_iter);
    if (psor_stats != nullptr) {
        psor_stats->record(result);
    }
}

/// The solved surface read at the spot, less a dividend going ex on the
/// pricing date.
double price_at_spot(double spot,
                     std::optional<double> pricing_dividend,
                     const ArrayXd& S,
                     const ArrayXd& values) {
    return interp_linear(spot - pricing_dividend.value_or(0.0), S, values);
}

/// Space grid for a knock-out. Continuous: truncated at the barrier, which
/// becomes a boundary node. Discrete: the full far-field grid with the
/// barrier anchored half a cell between nodes.
SpaceGridBuild build_knock_out_grid(const detail::ValuationState& state,
                                    const PdeParams& params,
                                    const BarrierSpec& spec,
                                    const pde::Domain& domain) {
    const double spot = state.underlying.initial_value();
    const double strike = spec.strike();
    const double barrier = spec.barrier();
    const bool is_up = spec.direction() == BarrierDirection::Up;
    const bool continuous = spec.monitoring() == BarrierMonitoring::Continuous;
    const double volatility = state.underlying.volatility();
    const double ttm = state.maturity_year_fraction();
    // Far-field targets from pde::barrier_domain. Its Spot-grid lower edge is
    // 0 (the fixed-span floor of the discrete/KI full grids); a continuously
    // monitored up-barrier Spot grid keeps its floor max(S, K) / smax_mult.
    const auto [far_lo, far_hi] = domain;
    const double smax_mult = pde::resolved_smax_mult(params);
    const double up_floor =
        far_lo > 0.0 ? far_lo : std::max(std::max(spot, strike) / smax_mult, 1.0e-8);
    const double smin_target = is_up ? up_floor : barrier;
    const double smax_target = is_up ? barrier : far_hi;
    if (params.space_grid == PdeSpaceGrid::LogSpot) {
        if (continuous) {
            return build_ko_continuous_log_grid(smin_target,
                                                smax_target,
                                                spec.direction(),
                                                volatility,
                                                ttm,
                                                params.spot_steps,
                                                params.time_steps,
                                                params.method);
        }
        return build_log_grid(spot,
                              strike,
                              ttm,
                              volatility,
                              smax_mult,
                              params.spot_steps,
                              params.time_steps,
                              params.method,
                              barrier,
                              /*anchor_half_step=*/true,
                              domain);
    }
    if (continuous) {
        return build_spot_grid(smin_target, smax_target, params.spot_steps);
    }
    return build_spot_grid(0.0, far_hi, params.spot_steps, barrier, /*anchor_half_step=*/true);
}

/// Boundary values for a continuous knock-out, whose grid ends at the
/// barrier: that side carries the rebate value, the far side the usual
/// vanilla asymptote.
std::pair<double, double> continuous_knock_out_boundaries(
    const BarrierSpec& spec, bool early, double smin, double smax, double df_fwd, double dq_fwd) {
    const double strike = spec.strike();
    const bool is_call = spec.option_type() == OptionType::Call;
    const double barrier_value = knock_out_rebate_value(spec, df_fwd);
    if (spec.direction() == BarrierDirection::Down) {
        if (!is_call) {
            return {barrier_value, 0.0};
        }
        const double continuation = smax * dq_fwd - strike * df_fwd;
        const double intrinsic = std::max(smax - strike, 0.0);
        return {barrier_value,
                early ? std::max(continuation, intrinsic) : std::max(continuation, 0.0)};
    }
    if (is_call) {
        return {0.0, barrier_value};
    }
    const double intrinsic = std::max(strike - smin, 0.0);
    const double continuation = strike * df_fwd - smin * dq_fwd;
    return {early ? std::max(continuation, intrinsic) : continuation, barrier_value};
}

}  // namespace

// ── Engine ──────────────────────────────────────────────────────────

PdeBarrierEngine::PdeBarrierEngine(const detail::ValuationState& state) : PdeEngine(state) {
    if (!std::holds_alternative<BarrierSpec>(state_.spec)) {
        throw ConfigurationError("PdeBarrierEngine requires a BarrierSpec");
    }
}

const BarrierSpec& PdeBarrierEngine::spec() const {
    return std::get<BarrierSpec>(state_.spec);
}

PdeEngine::Solution PdeBarrierEngine::compute_solution() const {
    // Public entry points short-circuit inception-triggered barriers before
    // ever reaching the solvers, so no triggered handling is needed here.
    const BarrierSpec& barrier = spec();
    if (barrier.action() == BarrierAction::Out) {
        return compute_barrier_knock_out();
    }
    if (state_.exercise_type() == ExerciseType::American) {
        return compute_barrier_knock_in();
    }
    // European knock-in via native-surface parity:
    // V_KI = V_vanilla + R df_T - V_KO (R = 0 reduces to parity).
    const auto& [ko, van] = european_ki_components();
    const double ttm = state_.maturity_year_fraction();
    const auto& curve = state_.discount_curve();
    const double df_maturity = curve.df(ttm);
    const double rebate = barrier.rebate();
    const double price = van.price + rebate * df_maturity - ko.price;

    const double previous_ttm = ko.last_dtau > 0.0 ? std::max(ttm - ko.last_dtau, 0.0) : ttm;
    const double rebate_prev = rebate * curve.df(previous_ttm);

    // Grid greeks read the KO grid (the best single surface we
    // have); reconstruct the KI surface on it via parity.
    ArrayXd values(ko.spots.size());
    ArrayXd values_prev(ko.spots.size());
    for (Index i = 0; i < ko.spots.size(); ++i) {
        const double s = ko.spots[i];
        values[i] = interp_linear(s, van.spots, van.values) + rebate * df_maturity - ko.values[i];
        values_prev[i] =
            interp_linear(s, van.spots, van.values_prev) + rebate_prev - ko.values_prev[i];
    }
    return Solution{
        .price = price,
        .spots = ko.spots,
        .values = std::move(values),
        .values_prev = std::move(values_prev),
        .last_dtau = ko.last_dtau,
    };
}

double PdeBarrierEngine::present_value() const {
    const BarrierSpec& barrier = spec();
    if (detail::barrier_triggered_at_inception(state_)) {
        if (barrier.action() == BarrierAction::Out) {
            if (barrier.rebate() <= 0.0) {
                return 0.0;
            }
            if (barrier.rebate_timing() == RebateTiming::AtHit) {
                return barrier.rebate();
            }
            return barrier.rebate() * state_.discount_curve().df(state_.maturity_year_fraction());
        }
        return vanilla_equivalent_engine().present_value();
    }
    return PdeEngine::present_value();
}

double PdeBarrierEngine::delta() const {
    if (detail::barrier_triggered_at_inception(state_)) {
        return spec().action() == BarrierAction::Out ? 0.0 : vanilla_equivalent_engine().delta();
    }
    if (is_european_knock_in()) {
        // Native-surface parity: extract on each component grid rather
        // than differentiating the reconstructed KI surface.
        const auto& [ko, van] = european_ki_components();
        const double spot = state_.underlying.initial_value();
        const Index j_van = spot_grid_index(van.spots, spot);
        const Index j_ko = spot_grid_index(ko.spots, spot);
        return grid_delta_at_spot(van.spots, van.values, j_van, spot) -
               grid_delta_at_spot(ko.spots, ko.values, j_ko, spot);
    }
    return PdeEngine::delta();
}

double PdeBarrierEngine::gamma() const {
    if (detail::barrier_triggered_at_inception(state_)) {
        return spec().action() == BarrierAction::Out ? 0.0 : vanilla_equivalent_engine().gamma();
    }
    if (is_european_knock_in()) {
        const auto& [ko, van] = european_ki_components();
        const double spot = state_.underlying.initial_value();
        const Index j_van = spot_grid_index(van.spots, spot);
        const Index j_ko = spot_grid_index(ko.spots, spot);
        return grid_gamma_safe(van.spots, van.values, j_van, spot) -
               grid_gamma_safe(ko.spots, ko.values, j_ko, spot);
    }
    return PdeEngine::gamma();
}

double PdeBarrierEngine::theta() const {
    if (detail::barrier_triggered_at_inception(state_)) {
        if (spec().action() == BarrierAction::Out) {
            return discounted_rebate_theta(barrier_last_dtau());
        }
        return vanilla_equivalent_engine().theta();
    }
    if (is_european_knock_in()) {
        // Parity with the discounted-rebate leg's own theta.
        const auto& [ko, van] = european_ki_components();
        return grid_theta_identity(van) + discounted_rebate_theta(ko.last_dtau) -
               grid_theta_identity(ko);
    }
    return PdeEngine::theta();
}

// ── Barrier machinery ───────────────────────────────────────────────

bool PdeBarrierEngine::is_european_knock_in() const {
    return spec().action() == BarrierAction::In && spec().exercise_type() == ExerciseType::European;
}

std::vector<double> PdeBarrierEngine::barrier_monitoring_taus() const {
    std::vector<double> taus;
    const auto dates = detail::barrier_monitoring_dates(state_);
    if (!dates.has_value()) {
        return taus;  // continuous monitoring
    }
    const double ttm = state_.maturity_year_fraction();
    for (const DateTime date : *dates) {
        if (date >= state_.pricing_date() && date <= state_.maturity()) {
            const double t =
                year_fraction(state_.pricing_date(), date, state_.day_count_convention());
            taus.push_back(round12(ttm - t));
        }
    }
    std::sort(taus.begin(), taus.end());
    return taus;
}

const std::pair<PdeEngine::Solution, PdeEngine::Solution>&
PdeBarrierEngine::european_ki_components() const {
    if (!ki_components_.has_value()) {
        // The vanilla leg reads the barrier spec's strike/type through the
        // shared state accessors, so `compute_vanilla` doubles as the
        // reference `_fd_core` call (European by construction here).
        // The vanilla leg is solved over the barrier's far-field targets
        // (pde::barrier_domain), so its far field is widened whenever the
        // knock-out's is and its span covers the knock-out's nodes (up to one
        // cell of anchoring slack). The grids still differ: the knock-out is
        // truncated at a continuous barrier and anchored on the barrier for
        // discrete monitoring, so vanilla values are interpolated onto the
        // knock-out nodes.
        ki_components_.emplace(compute_barrier_knock_out(), compute_vanilla(far_field_domain()));
    }
    return *ki_components_;
}

double PdeBarrierEngine::barrier_last_dtau() const {
    // The solve's own time grid: continuous monitoring has no monitoring
    // taus, so `continuous = false` is the same grid for both monitoring styles.
    const ArrayXd tau_grid = event_schedule(state_,
                                            barrier_monitoring_taus(),
                                            /*continuous=*/false,
                                            state_.maturity_year_fraction(),
                                            pde_params(state_).time_steps)
                                 .tau_grid;
    if (tau_grid.size() < 2) {
        return 0.0;
    }
    return tau_grid[tau_grid.size() - 1] - tau_grid[tau_grid.size() - 2];
}

double PdeBarrierEngine::discounted_rebate_theta(double last_dtau) const {
    const auto& spec = std::get<BarrierSpec>(state_.spec);
    if (spec.rebate() <= 0.0 || spec.rebate_timing() != RebateTiming::AtExpiry ||
        last_dtau <= 0.0) {
        return 0.0;
    }
    const double ttm = state_.maturity_year_fraction();
    const auto& curve = state_.discount_curve();
    const double current_value = spec.rebate() * curve.df(ttm);
    const double previous_value = spec.rebate() * curve.df(std::max(ttm - last_dtau, 0.0));
    return (previous_value - current_value) / last_dtau / kDaysPerYear;
}

pde::Domain PdeBarrierEngine::far_field_domain() const {
    return pde::barrier_domain(
        state_, spec().strike(), pde::log_forward(state_), pde_params(state_));
}

const PdeEngine& PdeBarrierEngine::vanilla_equivalent_engine() const {
    if (vanilla_engine_ == nullptr) {
        const auto& spec = std::get<BarrierSpec>(state_.spec);
        vanilla_state_.emplace(detail::ValuationState{
            .underlying = state_.underlying,
            .spec = OptionSpec{detail::vanilla_equivalent_spec(spec)},
            .pricing_method = PricingMethod::PdeFd,
            .params = state_.params,
            .currency = state_.currency,
            .option_type = spec.option_type(),
        });
        vanilla_engine_ = std::make_unique<PdeEngine>(*vanilla_state_);
    }
    return *vanilla_engine_;
}

PdeEngine::Solution PdeBarrierEngine::compute_barrier_knock_out() const {
    const PdeParams& params = pde_params(state_);
    const auto& spec = std::get<BarrierSpec>(state_.spec);
    const bool early = state_.exercise_type() == ExerciseType::American;
    const double spot = state_.underlying.initial_value();
    const bool is_up = spec.direction() == BarrierDirection::Up;
    const bool continuous = spec.monitoring() == BarrierMonitoring::Continuous;
    const double volatility = state_.underlying.volatility();
    const double ttm = state_.maturity_year_fraction();
    const auto& curve = state_.discount_curve();
    const auto& q_curve = state_.underlying.dividend_curve();

    validate_fd_inputs(state_, params, ttm, early, /*has_custom_payoff=*/false);

    // ── Grid ────────────────────────────────────────────────────────
    const SpaceGridBuild built = build_knock_out_grid(state_, params, spec, far_field_domain());
    const ArrayXd& grid = built.grid;
    const ArrayXd& S = built.spots;
    pde::warn_if_under_resolved(spot, built.dx, ttm, volatility, params.space_grid);
    // Far-field sides only: a continuously monitored barrier is an exact boundary.
    pde::warn_if_domain_truncated(
        continuous && !is_up ? std::nullopt : std::optional<double>(S[0]),
        continuous && is_up ? std::nullopt : std::optional<double>(S[S.size() - 1]),
        spot,
        pde::log_forward(state_),
        ttm,
        volatility);
    const double smin = S[0];
    const double smax = S[S.size() - 1];
    const Mask knocked = knocked_nodes(S, spec.barrier(), is_up);

    // ── Terminal slice and event schedule ───────────────────────────
    // American intrinsic stays the vanilla payoff at any in-life moment,
    // regardless of the KO-zone reset applied to the terminal slice below.
    const ArrayXd intrinsic_values =
        vanilla_payoff(S, spec.strike(), spec.option_type() == OptionType::Call);
    const ArrayXd* intrinsic = early ? &intrinsic_values : nullptr;
    const EventSchedule events =
        event_schedule(state_, barrier_monitoring_taus(), continuous, ttm, params.time_steps);

    ArrayXd payoff = intrinsic_values;
    // A maturity observation knocks out the KO-zone nodes *at* maturity:
    // they receive the rebate (at-hit and at-expiry coincide at T).
    if (!continuous && events.is_monitoring(0.0)) {
        payoff = knocked.select(spec.rebate(), payoff);
    }
    ArrayXd v = payoff;
    if (events.maturity_dividend.has_value()) {
        apply_dividend_jump(v, grid, *events.maturity_dividend, params.space_grid);
        if (early) {
            v = v.max(payoff);
        }
    }

    if (is_explicit(params.method) && params.space_grid == PdeSpaceGrid::Spot) {
        check_explicit_spot_stability(events.tau_grid,
                                      volatility,
                                      smax,
                                      built.dx,
                                      ttm,
                                      curve,
                                      q_curve,
                                      params.method == PdeMethod::ExplicitHull);
    }

    // ── Time loop ───────────────────────────────────────────────────
    const StepOperator step_operator(S, built.dx, state_, params);
    ArrayXd v_prev = v;
    double last_dtau = 0.0;
    pde::PsorStats psor_stats;

    for (const auto& step :
         build_time_step_schedule(events.tau_grid, params.method, params.rannacher_steps)) {
        const StepCoefficients c = step_operator(step);
        const auto [left, right] = continuous ? continuous_knock_out_boundaries(
                                                    spec, early, smin, smax, c.df_fwd, c.dq_fwd)
                                              : boundary_values(state_.option_type,
                                                                spec.strike(),
                                                                smin,
                                                                smax,
                                                                c.df_fwd,
                                                                c.dq_fwd,
                                                                early,
                                                                nullptr,
                                                                std::nullopt);
        v_prev = v;
        last_dtau = c.d_tau;
        advance(v_prev, v, c, left, right, step.method, intrinsic, params, &psor_stats);

        // Discrete barrier reset at monitoring dates: knocked-out nodes take
        // the rebate value; early exercise is re-imposed on surviving nodes.
        // Imposed on the ex-dividend surface: an observation on an ex-date
        // sees the price after the dividend drop (ex-div first, then
        // monitor), and the dividend jump below carries the knocked-out
        // state back to cum-dividend spots in (H, H + D]. Same order as the
        // knock-in solve.
        if (!continuous && events.is_monitoring(step.tau_end)) {
            const double reset_value = knock_out_rebate_value(spec, c.df_fwd);
            v = knocked.select(reset_value, early ? v.max(intrinsic_values) : v);
        }

        if (const auto dividend = events.dividend_at(step.tau_end)) {
            apply_dividend_jump(v, grid, *dividend, params.space_grid);
            if (early) {
                v = v.max(intrinsic_values);
            }
        }
    }

    psor_stats.report("PDE barrier", params.max_iter, params.tol);
    const double price = price_at_spot(spot, events.pricing_dividend, S, v);
    return Solution{
        .price = price,
        .spots = S,
        .values = std::move(v),
        .values_prev = std::move(v_prev),
        .last_dtau = last_dtau,
    };
}

PdeEngine::Solution PdeBarrierEngine::compute_barrier_knock_in() const {
    const PdeParams& params = pde_params(state_);
    const auto& spec = std::get<BarrierSpec>(state_.spec);
    const bool early = state_.exercise_type() == ExerciseType::American;
    const double spot = state_.underlying.initial_value();
    const double barrier = spec.barrier();
    const bool is_up = spec.direction() == BarrierDirection::Up;
    const bool continuous = spec.monitoring() == BarrierMonitoring::Continuous;
    // KI rebates are AT_EXPIRY by contract: paid at maturity if never hit.
    const double rebate = spec.rebate();
    const double volatility = state_.underlying.volatility();
    const double ttm = state_.maturity_year_fraction();

    validate_fd_inputs(state_, params, ttm, early, /*has_custom_payoff=*/false);

    // ── Grid ────────────────────────────────────────────────────────
    // Full (untruncated) grid, anchored on the barrier: on a node for
    // continuous coupling, half a cell off for discrete monitoring.
    const pde::Domain domain = far_field_domain();
    const SpaceGridBuild built =
        params.space_grid == PdeSpaceGrid::LogSpot
            ? build_log_grid(spot,
                             spec.strike(),
                             ttm,
                             volatility,
                             pde::resolved_smax_mult(params),
                             params.spot_steps,
                             params.time_steps,
                             params.method,
                             barrier,
                             /*anchor_half_step=*/!continuous,
                             domain)
            : build_spot_grid(
                  0.0, domain.second, params.spot_steps, barrier, /*anchor_half_step=*/!continuous);
    const ArrayXd& grid = built.grid;
    const ArrayXd& S = built.spots;
    pde::warn_if_under_resolved(spot, built.dx, ttm, volatility, params.space_grid);
    pde::warn_if_domain_truncated(
        S[0], S[S.size() - 1], spot, pde::log_forward(state_), ttm, volatility);
    const double smin = S[0];
    const double smax = S[S.size() - 1];
    const Mask knocked = knocked_nodes(S, barrier, is_up);

    // ── Terminal slices and event schedule ──────────────────────────
    const ArrayXd payoff = vanilla_payoff(S, spec.strike(), spec.option_type() == OptionType::Call);
    const ArrayXd* intrinsic = early ? &payoff : nullptr;
    const EventSchedule events =
        event_schedule(state_, barrier_monitoring_taus(), continuous, ttm, params.time_steps);

    // Active surface: the barrier has been hit (vanilla behaviour).
    // Inactive surface: not yet hit; terminal value is the maturity rebate.
    ArrayXd v_act = payoff;
    ArrayXd v_inact = ArrayXd::Constant(S.size(), rebate);
    if (events.maturity_dividend.has_value()) {
        apply_dividend_jump(v_act, grid, *events.maturity_dividend, params.space_grid);
        if (early) {
            v_act = v_act.max(payoff);
        }
        // The inactive terminal surface is spot-independent, so the
        // maturity-date jump is a no-op for it.
    }

    // Nearest node to the barrier level, for the sub-grid coupling.
    Index j_barrier = 0;
    (S - barrier).abs().minCoeff(&j_barrier);

    // Terminal coupling: a barrier already triggered at maturity moves the
    // inactive state onto the active payoff.
    if (continuous || events.is_monitoring(0.0)) {
        v_inact = knocked.select(v_act, v_inact);
    }

    // Continuous monitoring solves the inactive surface only on the far side
    // of the barrier, with the active value at the barrier node as the inner
    // Dirichlet boundary (implicit coupling — no operator-splitting error).
    Index j_lo = 1;
    Index j_hi = params.spot_steps - 1;
    if (continuous) {
        if (!is_up) {
            j_lo = j_barrier + 1;
        } else {
            j_hi = j_barrier - 1;
        }
    }

    // ── Time loop ───────────────────────────────────────────────────
    const StepOperator step_operator(S, built.dx, state_, params);
    ArrayXd v_act_prev = v_act;
    ArrayXd v_inact_prev = v_inact;
    double last_dtau = 0.0;
    pde::PsorStats psor_stats;

    for (const auto& step :
         build_time_step_schedule(events.tau_grid, params.method, params.rannacher_steps)) {
        const StepCoefficients c = step_operator(step);
        const auto [left, right] = boundary_values(state_.option_type,
                                                   spec.strike(),
                                                   smin,
                                                   smax,
                                                   c.df_fwd,
                                                   c.dq_fwd,
                                                   early,
                                                   nullptr,
                                                   std::nullopt);
        v_act_prev = v_act;
        v_inact_prev = v_inact;
        last_dtau = c.d_tau;

        // Step A: active surface (vanilla, with early exercise).
        advance(v_act_prev, v_act, c, left, right, step.method, intrinsic, params, &psor_stats);

        // Step B: inactive surface (pure continuation, no exercise).
        const double rebate_bv = rebate * c.df_fwd;
        if (continuous) {
            const double left_inact = !is_up ? v_act[j_barrier] : rebate_bv;
            const double right_inact = !is_up ? rebate_bv : v_act[j_barrier];
            if (j_lo <= j_hi) {
                v_inact = subgrid_pde_step(v_inact,
                                           v_inact_prev,
                                           j_lo,
                                           j_hi,
                                           c.scaled,
                                           left_inact,
                                           right_inact,
                                           step.method,
                                           c.r_dt);
            }
            // Complete the slice: impose the hit-side coupling region and
            // the far-field boundary.
            couple_hit_side(v_inact, v_act, j_barrier, is_up);
            v_inact[!is_up ? v_inact.size() - 1 : 0] = rebate_bv;
        } else {
            // Discrete monitoring: full-grid solve with a one-step
            // look-ahead boundary proxy on the risky side (activation can
            // only occur at the next observation), coupling applied at
            // monitoring dates on the ex-dividend surface (the dividend jump
            // below carries it back to cum-dividend spots).
            const bool is_monitoring_step = events.is_monitoring(step.tau_end);
            double left_inact = rebate_bv;
            double right_inact = rebate_bv;
            if (!is_up) {
                left_inact = is_monitoring_step ? v_act[0] : v_act_prev[0];
            } else {
                right_inact = is_monitoring_step ? v_act[v_act.size() - 1]
                                                 : v_act_prev[v_act_prev.size() - 1];
            }
            advance(v_inact_prev,
                    v_inact,
                    c,
                    left_inact,
                    right_inact,
                    step.method,
                    nullptr,
                    params,
                    nullptr);
            if (is_monitoring_step) {
                v_inact = knocked.select(v_act, v_inact);
            }
        }

        // Discrete dividend jumps on both surfaces.
        if (const auto dividend = events.dividend_at(step.tau_end)) {
            apply_dividend_jump(v_act, grid, *dividend, params.space_grid);
            if (early) {
                v_act = v_act.max(payoff);
            }
            apply_dividend_jump(v_inact, grid, *dividend, params.space_grid);
            if (continuous) {
                couple_hit_side(v_inact, v_act, j_barrier, is_up);
            }
        }
    }

    psor_stats.report("PDE barrier KI", params.max_iter, params.tol);
    // Price from the inactive surface: the option starts inactive.
    const double price = price_at_spot(spot, events.pricing_dividend, S, v_inact);
    return Solution{
        .price = price,
        .spots = S,
        .values = std::move(v_inact),
        .values_prev = std::move(v_inact_prev),
        .last_dtau = last_dtau,
    };
}

}  // namespace dp::engines
