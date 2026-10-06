#include "engines/pde.hpp"

#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/diagnostics.hpp"
#include "dp/exceptions.hpp"
#include "dp/params.hpp"

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
using pde::build_tau_grid;
using pde::build_time_step_schedule;
using pde::check_explicit_spot_stability;
using pde::dividend_tau_schedule;
using pde::explicit_step;
using pde::fit_affine_boundary_model;
using pde::grid_delta_at_spot;
using pde::grid_gamma_safe;
using pde::grid_value_at_spot;
using pde::implicit_cn_step;
using pde::Index;
using pde::interp_linear;
using pde::kDaysPerYear;
using pde::linspace;
using pde::log_operator_coeffs;
using pde::OperatorCoeffs;
using pde::pde_params;
using pde::round12;
using pde::spot_grid_index;
using pde::spot_operator_coeffs;
using pde::validate_fd_inputs;

// ── Engine ──────────────────────────────────────────────────────────

PdeEngine::PdeEngine(const detail::ValuationState& state) : state_(state) {}

const PdeEngine::Solution& PdeEngine::solution() const {
    if (!solution_.has_value()) {
        solution_.emplace(compute_solution());
    }
    return *solution_;
}

PdeEngine::Solution PdeEngine::compute_solution() const {
    // smax_mult unset: vanillas get the domain sized to sigma*sqrt(T); custom
    // payoffs keep the legacy span (their affine boundary models are fitted
    // to the payoff wings). An explicit smax_mult is the legacy domain.
    const PdeParams& params = pde_params(state_);
    std::optional<pde::Domain> domain;
    if (const std::optional<double> strike = state_.strike();
        !params.smax_mult.has_value() && strike.has_value() &&
        !std::holds_alternative<PayoffSpec>(state_.spec)) {
        domain = pde::fitted_vanilla_domain(
            state_, *strike, pde::log_forward(state_), params.space_grid);
    }
    return compute_vanilla(domain);
}

PdeEngine::Solution PdeEngine::compute_vanilla(std::optional<pde::Domain> domain) const {
    const PdeParams& params = pde_params(state_);
    const bool early = state_.exercise_type() == ExerciseType::American;
    const auto* custom = std::get_if<PayoffSpec>(&state_.spec);
    const double spot = state_.underlying.initial_value();
    const std::optional<double> strike = state_.strike();
    const double volatility = state_.underlying.volatility();
    const double ttm = state_.maturity_year_fraction();
    const auto& curve = state_.discount_curve();
    const auto& q_curve = state_.underlying.dividend_curve();

    validate_fd_inputs(state_, params, ttm, early, custom != nullptr);

    const int m_steps = params.spot_steps;
    const double ref_price = strike.has_value() ? std::max(spot, *strike) : spot;

    const double smax_mult = pde::resolved_smax_mult(params);
    ArrayXd grid;
    ArrayXd S;
    double dx = 0.0;
    if (params.space_grid == PdeSpaceGrid::Spot) {
        const auto [lo, hi] = domain.value_or(pde::Domain{0.0, smax_mult * ref_price});
        grid = linspace(lo, hi, m_steps);
        S = grid;
        dx = (hi - lo) / m_steps;
    } else {
        auto built = build_log_grid(spot,
                                    strike.value_or(spot),
                                    ttm,
                                    volatility,
                                    smax_mult,
                                    m_steps,
                                    params.time_steps,
                                    params.method,
                                    std::nullopt,
                                    false,
                                    domain);
        grid = std::move(built.grid);
        S = std::move(built.spots);
        dx = built.dx;
    }
    pde::warn_if_under_resolved(spot, dx, ttm, volatility, params.space_grid);
    pde::warn_if_domain_truncated(
        S[0], S[S.size() - 1], spot, pde::log_forward(state_), ttm, volatility);
    const double smin = S[0];
    const double smax = S[S.size() - 1];

    // Terminal payoff.
    ArrayXd payoff(S.size());
    if (custom != nullptr) {
        for (Index i = 0; i < S.size(); ++i) {
            payoff[i] = custom->payoff(S[i]);
        }
    } else {
        if (!strike.has_value()) {
            throw ConfigurationError("vanilla PDE pricing requires a strike");
        }
        if (state_.option_type == OptionType::Put) {
            payoff = (*strike - S).cwiseMax(0.0);
        } else {
            payoff = (S - *strike).cwiseMax(0.0);
        }
    }

    // Affine wing boundary models for custom payoffs: explicit metadata or a
    // local fit on the actual boundary nodes.
    std::optional<PayoffBoundaryModel> boundary_model;
    if (custom != nullptr) {
        boundary_model = custom->boundary_model();
        if (!boundary_model.has_value()) {
            boundary_model = PayoffBoundaryModel{
                .left = fit_affine_boundary_model(*custom, "left", S.head(4)),
                .right = fit_affine_boundary_model(*custom, "right", S.tail(4)),
            };
        } else if (params.space_grid == PdeSpaceGrid::LogSpot) {
            warn(
                "Explicit PayoffBoundaryModel with LOG_SPOT grid is interpreted as an affine "
                "boundary model on the finite truncated PDE domain, not as a true payoff tail "
                "asymptote. Ensure the supplied boundary model is appropriate at the actual "
                "grid boundaries.");
        }
    }

    ArrayXd v = payoff;
    const ArrayXd* intrinsic = early ? &payoff : nullptr;

    std::map<double, double> dividend_map;
    for (const auto& [tau, amount] : dividend_tau_schedule(state_)) {
        dividend_map.emplace(tau, amount);
    }

    // Maturity-date dividend (tau = 0): immediate jump on the terminal slice.
    if (const auto it = dividend_map.find(0.0); it != dividend_map.end()) {
        apply_dividend_jump(v, grid, it->second, params.space_grid);
        if (early) {
            v = v.max(payoff);
        }
        dividend_map.erase(it);
    }
    // Pricing-date dividend (tau = ttm): applied as a spot shift at the end.
    std::optional<double> pricing_div;
    if (const auto it = dividend_map.find(round12(ttm)); it != dividend_map.end()) {
        pricing_div = it->second;
        dividend_map.erase(it);
    }

    std::vector<double> dividend_taus;
    dividend_taus.reserve(dividend_map.size());
    for (const auto& [tau, amount] : dividend_map) {
        dividend_taus.push_back(tau);
    }
    const ArrayXd tau_grid = build_tau_grid(ttm, params.time_steps, dividend_taus);

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

    pde::PsorStats psor_stats;

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
            state_.option_type, strike, smin, smax, df_fwd, dq_fwd, early, custom, boundary_model);

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
            psor_stats.record(implicit_cn_step(v_prev,
                                               v,
                                               scaled,
                                               left,
                                               right,
                                               step.method,
                                               intrinsic,
                                               params.american_solver,
                                               params.omega,
                                               params.tol,
                                               params.max_iter));
        }

        if (!dividend_map.empty()) {
            if (const auto it = dividend_map.find(round12(step.tau_end));
                it != dividend_map.end()) {
                apply_dividend_jump(v, grid, it->second, params.space_grid);
                if (early) {
                    v = v.max(payoff);
                }
            }
        }
    }

    psor_stats.report("PDE", params.max_iter, params.tol);

    // The input spot is cum-dividend: a pricing-date dividend shifts the
    // interpolation point.
    const double interp_spot = spot - pricing_div.value_or(0.0);
    const double price = interp_linear(interp_spot, S, v);

    return Solution{
        .price = price,
        .spots = std::move(S),
        .values = std::move(v),
        .values_prev = std::move(v_prev),
        .last_dtau = last_dtau,
    };
}

double PdeEngine::present_value() const {
    return solution().price;
}

// ── Grid Greeks ─────────────────────────────────────────────────────

std::optional<std::pair<double, double>> PdeEngine::intrinsic_short_circuit(const ArrayXd& S,
                                                                            const ArrayXd& V,
                                                                            Index j) const {
    // American vanilla/barrier only: in the exercise region the local value
    // function is the intrinsic, so delta = ±1 and gamma = 0 exactly, while
    // the PDE stencils are polluted by PSOR oscillations.
    if (!std::holds_alternative<VanillaSpec>(state_.spec) &&
        !std::holds_alternative<BarrierSpec>(state_.spec)) {
        return std::nullopt;
    }
    if (state_.exercise_type() != ExerciseType::American) {
        return std::nullopt;
    }
    const auto strike = state_.strike();
    if (!strike.has_value()) {
        return std::nullopt;
    }
    const double s_node = S[j];
    double intrinsic = 0.0;
    double sign = 0.0;
    if (state_.option_type == OptionType::Call) {
        intrinsic = s_node - *strike;
        sign = 1.0;
    } else {
        intrinsic = *strike - s_node;
        sign = -1.0;
    }
    if (intrinsic <= 0.0) {
        return std::nullopt;
    }
    if (std::abs(V[j] - intrinsic) > std::max(1e-8, 1e-6 * intrinsic)) {
        return std::nullopt;
    }
    return std::pair{sign, 0.0};
}

double PdeEngine::delta() const {
    const Solution& r = solution();
    const double spot = state_.underlying.initial_value();
    const Index j = spot_grid_index(r.spots, spot);
    if (const auto sc = intrinsic_short_circuit(r.spots, r.values, j); sc.has_value()) {
        return sc->first;
    }
    return grid_delta_at_spot(r.spots, r.values, j, spot);
}

double PdeEngine::gamma() const {
    const Solution& r = solution();
    const double spot = state_.underlying.initial_value();
    const Index j = spot_grid_index(r.spots, spot);
    if (const auto sc = intrinsic_short_circuit(r.spots, r.values, j); sc.has_value()) {
        return sc->second;
    }
    return grid_gamma_safe(r.spots, r.values, j, spot);
}

double PdeEngine::grid_theta_identity(const Solution& result) const {
    // Black-Scholes PDE identity: theta = rV - (r-q) S delta - 0.5 sigma^2 S^2 gamma,
    // per calendar day. Matches the accuracy of the grid delta/gamma.
    if (result.last_dtau <= 0.0) {
        return 0.0;
    }
    const double spot = state_.underlying.initial_value();
    const Index j = spot_grid_index(result.spots, spot);
    const double pv_at_spot = grid_value_at_spot(result.spots, result.values, j, spot);
    const double delta_val = grid_delta_at_spot(result.spots, result.values, j, spot);
    const double gamma_val = grid_gamma_safe(result.spots, result.values, j, spot);

    const double sigma = state_.underlying.volatility();
    const double dt_probe = std::max(result.last_dtau, 1.0e-8);
    const double r = state_.discount_curve().forward_rate(0.0, dt_probe);
    const auto& q_curve = state_.underlying.dividend_curve();
    const double q = q_curve.has_value() ? q_curve->forward_rate(0.0, dt_probe) : 0.0;

    const double theta_annual =
        r * pv_at_spot - (r - q) * spot * delta_val - 0.5 * sigma * sigma * spot * spot * gamma_val;
    return theta_annual / kDaysPerYear;
}

double PdeEngine::theta() const {
    const Solution& res = solution();
    const double spot = state_.underlying.initial_value();
    const Index j = spot_grid_index(res.spots, spot);
    if (intrinsic_short_circuit(res.spots, res.values, j).has_value()) {
        return 0.0;
    }
    return grid_theta_identity(res);
}

}  // namespace dp::engines
