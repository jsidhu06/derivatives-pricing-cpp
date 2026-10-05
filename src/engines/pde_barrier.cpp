#include "engines/pde_barrier.hpp"

#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/diagnostics.hpp"
#include "dp/exceptions.hpp"
#include "dp/params.hpp"

#include "detail/barrier.hpp"
#include "detail/log.hpp"
#include "engines/pde_numerics.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <format>
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

/// Log-spot grid for a continuous knock-out: the barrier is a boundary node.
/// Explicit-family schemes preserve Hull's dz scale; CN/IMPLICIT honor
/// spot_steps directly.
SpaceGridBuild build_ko_continuous_log_grid(double smin_target,
                                            double smax_target,
                                            double ref_price,
                                            double smax_mult,
                                            BarrierDirection direction,
                                            double volatility,
                                            double time_to_maturity,
                                            int spot_steps,
                                            int time_steps,
                                            PdeMethod method) {
    const bool explicit_scheme = method == PdeMethod::Explicit || method == PdeMethod::ExplicitHull;
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
        zlo = std::log(std::max(ref_price / smax_mult, 1.0e-8));
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

    if (method == PdeMethod::Explicit || method == PdeMethod::ExplicitHull) {
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
        ki_components_.emplace(compute_barrier_knock_out(), compute_vanilla());
    }
    return *ki_components_;
}

double PdeBarrierEngine::barrier_last_dtau() const {
    const double ttm = state_.maturity_year_fraction();
    std::vector<double> extra_taus;
    for (const auto& [tau, amount] : dividend_tau_schedule(state_)) {
        if (tau > 1.0e-12 && tau < ttm - 1.0e-12) {
            extra_taus.push_back(tau);
        }
    }
    for (const double tau : barrier_monitoring_taus()) {
        extra_taus.push_back(tau);
    }
    const ArrayXd tau_grid = build_tau_grid(ttm, pde_params(state_).time_steps, extra_taus);
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
    const double strike = spec.strike();
    const double barrier = spec.barrier();
    const bool is_up = spec.direction() == BarrierDirection::Up;
    const bool is_call = spec.option_type() == OptionType::Call;
    const bool continuous = spec.monitoring() == BarrierMonitoring::Continuous;
    const double rebate = spec.rebate();
    const double volatility = state_.underlying.volatility();
    const double ttm = state_.maturity_year_fraction();
    const auto& curve = state_.discount_curve();
    const auto& q_curve = state_.underlying.dividend_curve();

    validate_fd_inputs(state_, params, ttm, early, /*has_custom_payoff=*/false);

    const std::vector<double> monitoring_taus = barrier_monitoring_taus();
    const int m_steps = params.spot_steps;
    const double ref_price = std::max(spot, strike);

    // For continuous KO the grid is truncated at the barrier (the barrier
    // becomes a boundary node); discrete KO uses the full grid with the
    // barrier anchored half a cell between nodes.
    SpaceGridBuild built;
    if (params.space_grid == PdeSpaceGrid::LogSpot) {
        if (continuous) {
            const double smin_target =
                is_up ? std::max(ref_price / params.smax_mult, 1.0e-8) : barrier;
            const double smax_target = is_up ? barrier : params.smax_mult * ref_price;
            built = build_ko_continuous_log_grid(smin_target,
                                                 smax_target,
                                                 ref_price,
                                                 params.smax_mult,
                                                 spec.direction(),
                                                 volatility,
                                                 ttm,
                                                 m_steps,
                                                 params.time_steps,
                                                 params.method);
        } else {
            built = build_log_grid(spot,
                                   strike,
                                   ttm,
                                   volatility,
                                   params.smax_mult,
                                   m_steps,
                                   params.time_steps,
                                   params.method,
                                   barrier,
                                   /*anchor_half_step=*/true);
        }
    } else {
        if (continuous) {
            const double smin_target =
                is_up ? std::max(ref_price / params.smax_mult, 1.0e-8) : barrier;
            const double smax_target = is_up ? barrier : params.smax_mult * ref_price;
            built = build_spot_grid(smin_target, smax_target, m_steps);
        } else {
            built = build_spot_grid(
                0.0, params.smax_mult * ref_price, m_steps, barrier, /*anchor_half_step=*/true);
        }
    }
    const ArrayXd grid = std::move(built.grid);
    const ArrayXd S = std::move(built.spots);
    const double dx = built.dx;
    const double smin = S[0];
    const double smax = S[S.size() - 1];

    const auto hit = [&](Index i) { return is_up ? S[i] >= barrier : S[i] <= barrier; };

    ArrayXd payoff(S.size());
    if (is_call) {
        payoff = (S - strike).cwiseMax(0.0);
    } else {
        payoff = (strike - S).cwiseMax(0.0);
    }
    // American intrinsic stays the vanilla payoff at any in-life moment,
    // regardless of the KO-zone reset applied to the terminal slice below.
    const ArrayXd intrinsic_values = payoff;
    const ArrayXd* intrinsic = early ? &intrinsic_values : nullptr;

    std::set<double> monitoring_tau_set;
    for (const double tau : monitoring_taus) {
        monitoring_tau_set.insert(round12(tau));
    }
    // A maturity observation knocks out the KO-zone nodes *at* maturity:
    // they receive the rebate (at-hit and at-expiry coincide at T).
    if (!continuous && monitoring_tau_set.contains(0.0)) {
        for (Index i = 0; i < S.size(); ++i) {
            if (hit(i)) {
                payoff[i] = rebate;
            }
        }
    }
    ArrayXd v = payoff;

    std::map<double, double> dividend_map;
    for (const auto& [tau, amount] : dividend_tau_schedule(state_)) {
        dividend_map.emplace(tau, amount);
    }
    if (const auto it = dividend_map.find(0.0); it != dividend_map.end()) {
        apply_dividend_jump(v, grid, it->second, params.space_grid);
        if (early) {
            v = v.max(payoff);
        }
        dividend_map.erase(it);
    }
    std::optional<double> pricing_div;
    if (const auto it = dividend_map.find(round12(ttm)); it != dividend_map.end()) {
        pricing_div = it->second;
        dividend_map.erase(it);
    }

    std::vector<double> extra_taus;
    extra_taus.reserve(dividend_map.size());
    for (const auto& [tau, amount] : dividend_map) {
        extra_taus.push_back(tau);
    }
    if (!continuous) {
        extra_taus.insert(extra_taus.end(), monitoring_taus.begin(), monitoring_taus.end());
    }
    const ArrayXd tau_grid = build_tau_grid(ttm, params.time_steps, extra_taus);

    if ((params.method == PdeMethod::Explicit || params.method == PdeMethod::ExplicitHull) &&
        params.space_grid == PdeSpaceGrid::Spot) {
        check_explicit_spot_stability(tau_grid,
                                      volatility,
                                      smax,
                                      dx,
                                      ttm,
                                      curve,
                                      q_curve,
                                      params.method == PdeMethod::ExplicitHull);
    }

    const double df_full = curve.df(ttm);
    const double dq_full = q_curve.has_value() ? q_curve->df(ttm) : 1.0;
    const auto steps = build_time_step_schedule(tau_grid, params.method, params.rannacher_steps);

    ArrayXd v_prev = v;
    double last_dtau = 0.0;
    int psor_steps = 0;
    long psor_total_iters = 0;
    int psor_max_iters = 0;
    int psor_not_converged = 0;

    for (const auto& step : steps) {
        const double d_tau = step.tau_end - step.tau_start;
        const double t_prev = ttm - step.tau_start;
        const double t_curr = ttm - step.tau_end;

        const double r = curve.forward_rate(t_curr, t_prev);
        const double q = q_curve.has_value() ? q_curve->forward_rate(t_curr, t_prev) : 0.0;
        const bool implicit_discounting = step.method == PdeMethod::ExplicitHull;

        const OperatorCoeffs coeffs =
            params.space_grid == PdeSpaceGrid::Spot
                ? spot_operator_coeffs(
                      S.segment(1, m_steps - 1), dx, r, q, volatility, implicit_discounting)
                : log_operator_coeffs(dx, r, q, volatility, implicit_discounting, m_steps - 1);

        const double df_fwd = df_full / curve.df(t_curr);
        const double dq_fwd = q_curve.has_value() ? dq_full / q_curve->df(t_curr) : 1.0;

        double left = 0.0;
        double right = 0.0;
        if (continuous) {
            // The barrier sits on a grid boundary carrying the rebate value.
            double barrier_bv = 0.0;
            if (rebate != 0.0) {
                barrier_bv = spec.rebate_timing() == RebateTiming::AtHit ? rebate : rebate * df_fwd;
            }
            if (!is_up) {
                left = barrier_bv;
                if (!is_call) {
                    right = 0.0;
                } else {
                    const double continuation = smax * dq_fwd - strike * df_fwd;
                    const double intrinsic_bv = std::max(smax - strike, 0.0);
                    right =
                        early ? std::max(continuation, intrinsic_bv) : std::max(continuation, 0.0);
                }
            } else {
                right = barrier_bv;
                if (!is_call) {
                    const double intrinsic_bv = std::max(strike - smin, 0.0);
                    const double continuation = strike * df_fwd - smin * dq_fwd;
                    left = early ? std::max(continuation, intrinsic_bv) : continuation;
                } else {
                    left = 0.0;
                }
            }
        } else {
            std::tie(left, right) = boundary_values(state_.option_type,
                                                    strike,
                                                    smin,
                                                    smax,
                                                    df_fwd,
                                                    dq_fwd,
                                                    early,
                                                    nullptr,
                                                    std::nullopt);
        }

        v_prev = v;
        last_dtau = d_tau;

        const OperatorCoeffs scaled{
            .gamma = -d_tau * coeffs.gamma,
            .beta = -d_tau * coeffs.beta,
            .alpha = -d_tau * coeffs.alpha,
        };

        if (step.method == PdeMethod::Explicit || step.method == PdeMethod::ExplicitHull) {
            v = explicit_step(
                v_prev, scaled, left, right, intrinsic, implicit_discounting ? r * d_tau : 0.0);
        } else {
            const auto psor_iters = implicit_cn_step(v_prev,
                                                     v,
                                                     scaled,
                                                     left,
                                                     right,
                                                     step.method,
                                                     intrinsic,
                                                     params.american_solver,
                                                     params.omega,
                                                     params.tol,
                                                     params.max_iter);
            if (psor_iters.has_value()) {
                psor_steps += 1;
                psor_total_iters += *psor_iters;
                psor_max_iters = std::max(psor_max_iters, *psor_iters);
                if (*psor_iters == params.max_iter) {
                    psor_not_converged += 1;
                }
            }
        }

        if (!dividend_map.empty()) {
            if (const auto it = dividend_map.find(round12(step.tau_end));
                it != dividend_map.end()) {
                apply_dividend_jump(v, grid, it->second, params.space_grid);
                if (early) {
                    v = v.max(intrinsic_values);
                }
            }
        }

        // Discrete barrier reset at monitoring dates: knocked-out nodes take
        // the rebate value; early exercise is re-imposed on surviving nodes.
        if (!continuous && monitoring_tau_set.contains(round12(step.tau_end))) {
            double reset_value = 0.0;
            if (rebate != 0.0) {
                reset_value =
                    spec.rebate_timing() == RebateTiming::AtHit ? rebate : rebate * df_fwd;
            }
            for (Index i = 0; i < S.size(); ++i) {
                if (hit(i)) {
                    v[i] = reset_value;
                } else if (early && intrinsic_values[i] > v[i]) {
                    v[i] = intrinsic_values[i];
                }
            }
        }
    }

    if (psor_steps > 0) {
        detail::logger()->debug(
            "PDE barrier PSOR steps={} avg_iters={:.2f} max_iters={} not_converged={}",
            psor_steps,
            static_cast<double>(psor_total_iters) / psor_steps,
            psor_max_iters,
            psor_not_converged);
    }

    const double interp_spot = spot - pricing_div.value_or(0.0);
    const double price = interp_linear(interp_spot, S, v);
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
    const double strike = spec.strike();
    const double barrier = spec.barrier();
    const bool is_up = spec.direction() == BarrierDirection::Up;
    const bool is_call = spec.option_type() == OptionType::Call;
    const bool continuous = spec.monitoring() == BarrierMonitoring::Continuous;
    // KI rebates are AT_EXPIRY by contract: paid at maturity if never hit.
    const double rebate = spec.rebate();
    const double volatility = state_.underlying.volatility();
    const double ttm = state_.maturity_year_fraction();
    const auto& curve = state_.discount_curve();
    const auto& q_curve = state_.underlying.dividend_curve();

    validate_fd_inputs(state_, params, ttm, early, /*has_custom_payoff=*/false);

    const std::vector<double> monitoring_taus = barrier_monitoring_taus();
    const int m_steps = params.spot_steps;
    const double ref_price = std::max(spot, strike);

    // Full (untruncated) grid, anchored on the barrier: on a node for
    // continuous coupling, half a cell off for discrete monitoring.
    SpaceGridBuild built;
    if (params.space_grid == PdeSpaceGrid::LogSpot) {
        built = build_log_grid(spot,
                               strike,
                               ttm,
                               volatility,
                               params.smax_mult,
                               m_steps,
                               params.time_steps,
                               params.method,
                               barrier,
                               /*anchor_half_step=*/!continuous);
    } else {
        built = build_spot_grid(0.0,
                                params.smax_mult * ref_price,
                                m_steps,
                                barrier,
                                /*anchor_half_step=*/!continuous);
    }
    const ArrayXd grid = std::move(built.grid);
    const ArrayXd S = std::move(built.spots);
    const double dx = built.dx;
    const double smin = S[0];
    const double smax = S[S.size() - 1];

    const auto hit = [&](Index i) { return is_up ? S[i] >= barrier : S[i] <= barrier; };

    ArrayXd payoff(S.size());
    if (is_call) {
        payoff = (S - strike).cwiseMax(0.0);
    } else {
        payoff = (strike - S).cwiseMax(0.0);
    }
    const ArrayXd* intrinsic = early ? &payoff : nullptr;

    // Active surface: the barrier has been hit (vanilla behaviour).
    // Inactive surface: not yet hit; terminal value is the maturity rebate.
    ArrayXd v_act = payoff;
    ArrayXd v_inact = ArrayXd::Constant(S.size(), rebate);

    std::map<double, double> dividend_map;
    for (const auto& [tau, amount] : dividend_tau_schedule(state_)) {
        dividend_map.emplace(tau, amount);
    }
    if (const auto it = dividend_map.find(0.0); it != dividend_map.end()) {
        apply_dividend_jump(v_act, grid, it->second, params.space_grid);
        if (early) {
            v_act = v_act.max(payoff);
        }
        // The inactive terminal surface is spot-independent, so the
        // maturity-date jump is a no-op for it.
        dividend_map.erase(it);
    }
    std::optional<double> pricing_div;
    if (const auto it = dividend_map.find(round12(ttm)); it != dividend_map.end()) {
        pricing_div = it->second;
        dividend_map.erase(it);
    }

    std::set<double> monitoring_tau_set;
    for (const double tau : monitoring_taus) {
        monitoring_tau_set.insert(round12(tau));
    }

    std::vector<double> extra_taus;
    extra_taus.reserve(dividend_map.size());
    for (const auto& [tau, amount] : dividend_map) {
        extra_taus.push_back(tau);
    }
    if (!continuous) {
        extra_taus.insert(extra_taus.end(), monitoring_taus.begin(), monitoring_taus.end());
    }
    const ArrayXd tau_grid = build_tau_grid(ttm, params.time_steps, extra_taus);

    // Nearest node to the barrier level, for the sub-grid coupling.
    Index j_barrier = 0;
    (S - barrier).abs().minCoeff(&j_barrier);

    // Terminal coupling: a barrier already triggered at maturity moves the
    // inactive state onto the active payoff.
    if (continuous || monitoring_tau_set.contains(0.0)) {
        for (Index i = 0; i < S.size(); ++i) {
            if (hit(i)) {
                v_inact[i] = v_act[i];
            }
        }
    }

    // Continuous monitoring solves the inactive surface only on the far side
    // of the barrier, with the active value at the barrier node as the inner
    // Dirichlet boundary (implicit coupling — no operator-splitting error).
    Index j_lo = 1;
    Index j_hi = m_steps - 1;
    if (continuous) {
        if (!is_up) {
            j_lo = j_barrier + 1;
        } else {
            j_hi = j_barrier - 1;
        }
    }

    const double df_full = curve.df(ttm);
    const double dq_full = q_curve.has_value() ? q_curve->df(ttm) : 1.0;
    const auto steps = build_time_step_schedule(tau_grid, params.method, params.rannacher_steps);

    ArrayXd v_act_prev = v_act;
    ArrayXd v_inact_prev = v_inact;
    double last_dtau = 0.0;
    int psor_steps = 0;
    long psor_total_iters = 0;
    int psor_max_iters = 0;
    int psor_not_converged = 0;

    for (const auto& step : steps) {
        const double d_tau = step.tau_end - step.tau_start;
        const double t_prev = ttm - step.tau_start;
        const double t_curr = ttm - step.tau_end;

        const double r = curve.forward_rate(t_curr, t_prev);
        const double q = q_curve.has_value() ? q_curve->forward_rate(t_curr, t_prev) : 0.0;
        const bool implicit_discounting = step.method == PdeMethod::ExplicitHull;

        const OperatorCoeffs coeffs =
            params.space_grid == PdeSpaceGrid::Spot
                ? spot_operator_coeffs(
                      S.segment(1, m_steps - 1), dx, r, q, volatility, implicit_discounting)
                : log_operator_coeffs(dx, r, q, volatility, implicit_discounting, m_steps - 1);

        const double df_fwd = df_full / curve.df(t_curr);
        const double dq_fwd = q_curve.has_value() ? dq_full / q_curve->df(t_curr) : 1.0;

        const auto [left, right] = boundary_values(
            state_.option_type, strike, smin, smax, df_fwd, dq_fwd, early, nullptr, std::nullopt);

        v_act_prev = v_act;
        v_inact_prev = v_inact;
        last_dtau = d_tau;

        const OperatorCoeffs scaled{
            .gamma = -d_tau * coeffs.gamma,
            .beta = -d_tau * coeffs.beta,
            .alpha = -d_tau * coeffs.alpha,
        };
        const double r_dt = implicit_discounting ? r * d_tau : 0.0;

        // Step A: active surface (vanilla, with early exercise).
        if (step.method == PdeMethod::Explicit || step.method == PdeMethod::ExplicitHull) {
            v_act = explicit_step(v_act_prev, scaled, left, right, intrinsic, r_dt);
        } else {
            const auto psor_iters = implicit_cn_step(v_act_prev,
                                                     v_act,
                                                     scaled,
                                                     left,
                                                     right,
                                                     step.method,
                                                     intrinsic,
                                                     params.american_solver,
                                                     params.omega,
                                                     params.tol,
                                                     params.max_iter);
            if (psor_iters.has_value()) {
                psor_steps += 1;
                psor_total_iters += *psor_iters;
                psor_max_iters = std::max(psor_max_iters, *psor_iters);
                if (*psor_iters == params.max_iter) {
                    psor_not_converged += 1;
                }
            }
        }

        // Step B: inactive surface (pure continuation, no exercise).
        const double rebate_bv = rebate * df_fwd;
        if (continuous) {
            const double left_inact = !is_up ? v_act[j_barrier] : rebate_bv;
            const double right_inact = !is_up ? rebate_bv : v_act[j_barrier];
            if (j_lo <= j_hi) {
                v_inact = subgrid_pde_step(v_inact,
                                           v_inact_prev,
                                           j_lo,
                                           j_hi,
                                           scaled,
                                           left_inact,
                                           right_inact,
                                           step.method,
                                           r_dt);
            }
            // Complete the slice: impose the hit-side coupling region and
            // the far-field boundary.
            if (!is_up) {
                v_inact.head(j_barrier + 1) = v_act.head(j_barrier + 1);
                v_inact[v_inact.size() - 1] = rebate_bv;
            } else {
                v_inact.tail(S.size() - j_barrier) = v_act.tail(S.size() - j_barrier);
                v_inact[0] = rebate_bv;
            }
        } else {
            // Discrete monitoring: full-grid solve with a one-step
            // look-ahead boundary proxy on the risky side (activation can
            // only occur at the next observation), coupling applied at
            // monitoring dates.
            const bool is_monitoring_step = monitoring_tau_set.contains(round12(step.tau_end));
            double left_inact = rebate_bv;
            double right_inact = rebate_bv;
            if (!is_up) {
                left_inact = is_monitoring_step ? v_act[0] : v_act_prev[0];
            } else {
                right_inact = is_monitoring_step ? v_act[v_act.size() - 1]
                                                 : v_act_prev[v_act_prev.size() - 1];
            }
            if (step.method == PdeMethod::Explicit || step.method == PdeMethod::ExplicitHull) {
                v_inact =
                    explicit_step(v_inact_prev, scaled, left_inact, right_inact, nullptr, r_dt);
            } else {
                implicit_cn_step(v_inact_prev,
                                 v_inact,
                                 scaled,
                                 left_inact,
                                 right_inact,
                                 step.method,
                                 nullptr,
                                 params.american_solver,
                                 params.omega,
                                 params.tol,
                                 params.max_iter);
            }
            if (is_monitoring_step) {
                for (Index i = 0; i < S.size(); ++i) {
                    if (hit(i)) {
                        v_inact[i] = v_act[i];
                    }
                }
            }
        }

        // Discrete dividend jumps on both surfaces.
        if (!dividend_map.empty()) {
            if (const auto it = dividend_map.find(round12(step.tau_end));
                it != dividend_map.end()) {
                apply_dividend_jump(v_act, grid, it->second, params.space_grid);
                if (early) {
                    v_act = v_act.max(payoff);
                }
                apply_dividend_jump(v_inact, grid, it->second, params.space_grid);
                if (continuous) {
                    if (!is_up) {
                        v_inact.head(j_barrier + 1) = v_act.head(j_barrier + 1);
                    } else {
                        v_inact.tail(S.size() - j_barrier) = v_act.tail(S.size() - j_barrier);
                    }
                }
            }
        }
    }

    if (psor_steps > 0) {
        detail::logger()->debug(
            "PDE barrier KI PSOR steps={} avg_iters={:.2f} max_iters={} not_converged={}",
            psor_steps,
            static_cast<double>(psor_total_iters) / psor_steps,
            psor_max_iters,
            psor_not_converged);
    }

    // Price from the inactive surface: the option starts inactive.
    const double interp_spot = spot - pricing_div.value_or(0.0);
    const double price = interp_linear(interp_spot, S, v_inact);
    return Solution{
        .price = price,
        .spots = S,
        .values = std::move(v_inact),
        .values_prev = std::move(v_inact_prev),
        .last_dtau = last_dtau,
    };
}

}  // namespace dp::engines
