#include "engines/binomial.hpp"

#include "dp/dates.hpp"
#include "dp/dividends.hpp"
#include "dp/exceptions.hpp"
#include "dp/params.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <variant>
#include <vector>

namespace dp::engines {

namespace {

constexpr double kDaysPerYear = 365.0;

/// Same tolerance as `np.allclose` (used by the Python reference to verify
/// the calendar grid produced uniform year-fraction steps).
bool all_close_to_first(const Eigen::ArrayXd& values) {
    const double first = values[0];
    return ((values - first).abs() <= 1e-8 + 1e-5 * std::abs(first)).all();
}

}  // namespace

BinomialEngine::BinomialEngine(const detail::ValuationState& state) : state_(state) {}

int BinomialEngine::num_steps() const {
    // resolve_params guarantees BinomialParams are present for this engine.
    const auto* params =
        state_.params.has_value() ? std::get_if<BinomialParams>(&*state_.params) : nullptr;
    if (params == nullptr) {
        throw ConfigurationError("BinomialEngine requires BinomialParams");
    }
    return params->num_steps;
}

bool BinomialEngine::early_exercise() const {
    return state_.exercise_type() == ExerciseType::American;
}

const BinomialEngine::Setup& BinomialEngine::setup() const {
    if (setup_.has_value()) {
        return *setup_;
    }

    const int n = num_steps();
    const DateTime start = state_.pricing_date();
    const DateTime end = state_.maturity();
    const auto dates = date_range(start, end, static_cast<std::size_t>(n) + 1);

    Eigen::ArrayXd time_grid(n + 1);
    for (int i = 0; i <= n; ++i) {
        time_grid[i] = year_fraction(start, dates[static_cast<std::size_t>(i)],
                                     state_.day_count_convention());
    }
    const Eigen::ArrayXd dt_steps = time_grid.tail(n) - time_grid.head(n);
    if (!all_close_to_first(dt_steps)) {
        throw UnsupportedFeatureError(std::format(
            "Binomial trees (CRR) require equal time steps, but the '{}' day-count "
            "convention produces non-uniform year fractions on a calendar date grid. Use an "
            "actual-day convention (ACT/365F, ACT/360) or switch to PDE / Monte Carlo pricing.",
            to_string(state_.day_count_convention())));
    }
    const double delta_t = dt_steps[0];

    const double sigma = state_.underlying.volatility();
    const double u = std::exp(sigma * std::sqrt(delta_t));
    const double d = 1.0 / u;

    const auto& curve = state_.discount_curve();
    const Eigen::ArrayXd forward_rates = curve.step_forward_rates(time_grid);
    Eigen::ArrayXd dividend_forwards = Eigen::ArrayXd::Zero(n);
    if (const auto& q_curve = state_.underlying.dividend_curve(); q_curve.has_value()) {
        dividend_forwards = q_curve->step_forward_rates(time_grid);
    }

    const Eigen::ArrayXd growth = ((forward_rates - dividend_forwards) * delta_t).exp();
    const auto too_low = static_cast<int>((growth <= d).count());
    const auto too_high = static_cast<int>((growth >= u).count());
    if (too_low > 0 || too_high > 0) {
        std::string parts;
        if (too_low > 0) {
            parts += std::format("exp((r-q)*dt) <= d at {} step(s)", too_low);
        }
        if (too_high > 0) {
            if (!parts.empty()) {
                parts += "; ";
            }
            parts += std::format("exp((r-q)*dt) >= u at {} step(s)", too_high);
        }
        throw ArbitrageViolationError(
            "No-arbitrage condition d < exp((r-q)*dt) < u violated: " + parts);
    }

    // Escrowed dividends: price the tree on spot minus the PV of all future
    // dividends; the PV of dividends not yet gone ex is restored per date.
    double spot = state_.underlying.initial_value();
    const auto& dividends = state_.underlying.discrete_dividends();
    Eigen::ArrayXd pv_remaining;
    if (!dividends.empty()) {
        const double pv_all =
            pv_discrete_dividends(dividends, dates.front(), dates.back(), curve, std::nullopt,
                                  state_.day_count_convention(), /*include_start=*/true);
        spot = std::max(spot - pv_all, 0.0);
        pv_remaining.resize(n + 1);
        for (int t = 0; t <= n; ++t) {
            pv_remaining[t] = pv_discrete_dividends(
                dividends, dates.front(), dates.back(), curve, dates[static_cast<std::size_t>(t)],
                state_.day_count_convention(), /*include_start=*/false);
        }
    }

    // u^(t-2i) via a shared power table so node spots are bit-identical to
    // the reference lattice while costing O(n) pow calls instead of O(n^2).
    Eigen::ArrayXd up_powers(2 * n + 1);
    for (int k = 0; k <= 2 * n; ++k) {
        up_powers[k] = std::pow(u, k - n);
    }

    setup_.emplace(Setup{
        .delta_t = delta_t,
        .discount_factors = (-forward_rates * delta_t).exp(),
        .p = (growth - d) / (u - d),
        .up_powers = std::move(up_powers),
        .pv_remaining = std::move(pv_remaining),
        .escrowed_spot = spot,
    });
    return *setup_;
}

double BinomialEngine::spot_at(const Setup& s, int i, int t) const {
    const int n = num_steps();
    double value = s.escrowed_spot * s.up_powers[t - 2 * i + n];
    if (s.pv_remaining.size() > 0) {
        value += s.pv_remaining[t];
    }
    return value;
}

const BinomialEngine::Solution& BinomialEngine::solution() const {
    if (solution_.has_value()) {
        return *solution_;
    }

    const int n = num_steps();
    const Setup& s = setup();
    const bool early = early_exercise();
    const auto* custom = std::get_if<PayoffSpec>(&state_.spec);
    const bool is_call = state_.option_type == OptionType::Call;
    const double strike = state_.strike().value_or(0.0);

    const auto intrinsic = [&](double spot) {
        if (custom != nullptr) {
            return custom->payoff(spot);
        }
        return is_call ? std::max(spot - strike, 0.0) : std::max(strike - spot, 0.0);
    };

    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    Solution out{.pv = kNaN,
                 .f01 = kNaN,
                 .f11 = kNaN,
                 .s01 = kNaN,
                 .s11 = kNaN,
                 .f02 = kNaN,
                 .f12 = kNaN,
                 .f22 = kNaN,
                 .s02 = kNaN,
                 .s12 = kNaN,
                 .s22 = kNaN};

    // Rolling backward induction: O(n) memory. `values[i]` holds the option
    // value at (down-moves i, step t) for the current t.
    std::vector<double> values(static_cast<std::size_t>(n) + 1);
    for (int i = 0; i <= n; ++i) {
        values[static_cast<std::size_t>(i)] = intrinsic(spot_at(s, i, n));
    }

    // Greeks read option values and spots at steps 1 and 2; grab them as the
    // rollback passes through. For n <= 2 those steps are the terminal layer,
    // which the loop below never revisits, hence the capture before it too.
    const auto capture_nodes = [&](int t) {
        if (t == 2) {
            out.f02 = values[0];
            out.f12 = values[1];
            out.f22 = values[2];
            out.s02 = spot_at(s, 0, 2);
            out.s12 = spot_at(s, 1, 2);
            out.s22 = spot_at(s, 2, 2);
        } else if (t == 1) {
            out.f01 = values[0];
            out.f11 = values[1];
            out.s01 = spot_at(s, 0, 1);
            out.s11 = spot_at(s, 1, 1);
        }
    };
    capture_nodes(n);

    // Hot-loop inputs hoisted out of the per-node work: the payoff sign
    // trick (phi * (S - K) == S - K for calls, K - S for puts, bit-exactly),
    // the dividend add-back per step, and raw pointers the optimiser can
    // reason about.
    const bool has_dividends = s.pv_remaining.size() > 0;
    const double phi = is_call ? 1.0 : -1.0;
    const double escrowed = s.escrowed_spot;
    const double* up_powers = s.up_powers.data();
    double* v = values.data();

    for (int t = n - 1; t >= 0; --t) {
        const double p_up = s.p[t];
        const double p_down = 1.0 - p_up;
        const double df = s.discount_factors[t];
        // Each specialised loop keeps the reference per-node expression and
        // operation order, so results agree bit-for-bit with the lattice
        // formulation; the split only removes per-node branching.
        if (!early) {
            for (int i = 0; i <= t; ++i) {
                v[i] = (p_up * v[i] + p_down * v[i + 1]) * df;
            }
        } else if (custom == nullptr) {
            const double pv_rem = has_dividends ? s.pv_remaining[t] : 0.0;
            const int base = n + t;
            for (int i = 0; i <= t; ++i) {
                const double continuation = (p_up * v[i] + p_down * v[i + 1]) * df;
                const double spot = escrowed * up_powers[base - 2 * i] + pv_rem;
                v[i] = std::max(phi * (spot - strike), std::max(continuation, 0.0));
            }
        } else {
            for (int i = 0; i <= t; ++i) {
                const double continuation = (p_up * v[i] + p_down * v[i + 1]) * df;
                v[i] = std::max(intrinsic(spot_at(s, i, t)), continuation);
            }
        }
        capture_nodes(t);
    }
    out.pv = values[0];

    solution_.emplace(out);
    return *solution_;
}

double BinomialEngine::present_value() const { return solution().pv; }

double BinomialEngine::delta() const {
    const Solution& r = solution();
    return (r.f01 - r.f11) / (r.s01 - r.s11);
}

double BinomialEngine::gamma() const {
    if (num_steps() < 2) {
        throw ValidationError("Tree gamma requires num_steps >= 2.");
    }
    const Solution& r = solution();
    const double delta_up = (r.f02 - r.f12) / (r.s02 - r.s12);
    const double delta_down = (r.f12 - r.f22) / (r.s12 - r.s22);
    const double h = (r.s02 - r.s22) / 2.0;
    return (delta_up - delta_down) / h;
}

double BinomialEngine::theta() const {
    if (num_steps() < 2) {
        throw ValidationError("Tree theta requires num_steps >= 2.");
    }
    const Solution& r = solution();
    const double dt = state_.maturity_year_fraction() / num_steps();
    return (r.f12 - r.pv) / (2.0 * dt) / kDaysPerYear;
}

}  // namespace dp::engines
