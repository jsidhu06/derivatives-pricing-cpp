#include "engines/binomial.hpp"

#include "dp/dates.hpp"
#include "dp/diagnostics.hpp"
#include "dp/dividends.hpp"
#include "dp/exceptions.hpp"
#include "dp/params.hpp"

#include "detail/barrier.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <format>
#include <limits>
#include <set>
#include <span>
#include <variant>
#include <vector>

namespace dp::engines {

namespace {

constexpr double kDaysPerYear = 365.0;

/// True when every step equals the first within numpy-`allclose` tolerances
/// (rtol 1e-5, atol 1e-8): the calendar grid has uniform year-fraction steps.
bool all_close_to_first(const Eigen::ArrayXd& values) {
    const double first = values[0];
    return ((values - first).abs() <= 1e-8 + 1e-5 * std::abs(first)).all();
}

}  // namespace

/// A Solution with every field NaN (step-2 fields stay NaN when n < 2).
BinomialEngine::Solution BinomialEngine::nan_solution() {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    return {.pv = kNaN,
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
}

void BinomialEngine::capture_step_nodes(Solution& out,
                                        const Setup& s,
                                        const std::vector<double>& values,
                                        int t) {
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
}

BinomialEngine::BinomialEngine(const detail::ValuationState& state) : state_(state) {}

int BinomialEngine::base_num_steps() const {
    // resolve_params guarantees BinomialParams are present for this engine.
    const auto* params =
        state_.params.has_value() ? std::get_if<BinomialParams>(&*state_.params) : nullptr;
    if (params == nullptr) {
        throw ConfigurationError("BinomialEngine requires BinomialParams");
    }
    return params->num_steps;
}

int BinomialEngine::num_steps() const {
    return base_num_steps();
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
        time_grid[i] =
            year_fraction(start, dates[static_cast<std::size_t>(i)], state_.day_count_convention());
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
        throw ArbitrageViolationError("No-arbitrage condition d < exp((r-q)*dt) < u violated: " +
                                      parts);
    }

    // Escrowed dividends: price the tree on spot minus the PV of all future
    // dividends; the PV of dividends not yet gone ex is restored per date.
    double spot = state_.underlying.initial_value();
    const auto& dividends = state_.underlying.discrete_dividends();
    Eigen::ArrayXd pv_remaining;
    if (!dividends.empty()) {
        const DiscountCurve* q_curve = detail::curve_or_null(state_.underlying.dividend_curve());
        const double pv_all = pv_discrete_dividends(dividends,
                                                    dates.front(),
                                                    dates.back(),
                                                    curve,
                                                    std::nullopt,
                                                    state_.day_count_convention(),
                                                    /*include_start=*/true,
                                                    q_curve);
        spot = std::max(spot - pv_all, 0.0);
        pv_remaining.resize(n + 1);
        for (int t = 0; t <= n; ++t) {
            pv_remaining[t] = pv_discrete_dividends(dividends,
                                                    dates.front(),
                                                    dates.back(),
                                                    curve,
                                                    dates[static_cast<std::size_t>(t)],
                                                    state_.day_count_convention(),
                                                    /*include_start=*/false,
                                                    q_curve);
        }
    }

    // u^(t-2i) via a shared power table so node spots are bit-identical to
    // the reference lattice while costing O(n) pow calls instead of O(n^2).
    Eigen::ArrayXd up_powers(2 * n + 1);
    for (int k = 0; k <= 2 * n; ++k) {
        up_powers[k] = std::pow(u, k - n);
    }

    setup_.emplace(Setup{
        .num_steps = n,
        .delta_t = delta_t,
        .discount_factors = (-forward_rates * delta_t).exp(),
        .p = (growth - d) / (u - d),
        .up_powers = std::move(up_powers),
        .pv_remaining = std::move(pv_remaining),
        .escrowed_spot = spot,
    });
    return *setup_;
}

double BinomialEngine::spot_at(const Setup& s, int i, int t) {
    double value = s.escrowed_spot * s.up_powers[t - 2 * i + s.num_steps];
    if (s.pv_remaining.size() > 0) {
        value += s.pv_remaining[t];
    }
    return value;
}

const BinomialEngine::Solution& BinomialEngine::solution() const {
    if (!solution_.has_value()) {
        solution_.emplace(compute_solution());
    }
    return *solution_;
}

BinomialEngine::Solution BinomialEngine::compute_solution() const {
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

    Solution out = nan_solution();

    // Rolling backward induction: O(n) memory. `values[i]` holds the option
    // value at (down-moves i, step t) for the current t.
    std::vector<double> values(static_cast<std::size_t>(n) + 1);
    for (int i = 0; i <= n; ++i) {
        values[static_cast<std::size_t>(i)] = intrinsic(spot_at(s, i, n));
    }

    // Greeks read option values and spots at steps 1 and 2; grab them as the
    // rollback passes through. For n <= 2 those steps are the terminal layer,
    // which the loop below never revisits, hence the capture before it too.
    capture_step_nodes(out, s, values, n);

    // Hot-loop inputs hoisted out of the per-node work: the payoff sign
    // trick (phi * (S - K) == S - K for calls, K - S for puts, bit-exactly)
    // and the dividend add-back per step. The spans are bounds-typed views
    // over the buffers; indexing them costs the same as raw pointers in a
    // release build.
    const bool has_dividends = s.pv_remaining.size() > 0;
    const double phi = is_call ? 1.0 : -1.0;
    const double escrowed = s.escrowed_spot;
    const std::span<const double> up_powers{s.up_powers.data(),
                                            static_cast<std::size_t>(s.up_powers.size())};
    const std::span<double> v{values};

    for (int t = n - 1; t >= 0; --t) {
        const double p_up = s.p[t];
        const double p_down = 1.0 - p_up;
        const double df = s.discount_factors[t];
        const auto last = static_cast<std::size_t>(t);
        // Each specialised loop keeps the reference per-node expression and
        // operation order, so results agree bit-for-bit with the lattice
        // formulation; the split only removes per-node branching.
        if (!early) {
            for (std::size_t i = 0; i <= last; ++i) {
                v[i] = (p_up * v[i] + p_down * v[i + 1]) * df;
            }
        } else if (custom == nullptr) {
            const double pv_rem = has_dividends ? s.pv_remaining[t] : 0.0;
            const auto base = static_cast<std::size_t>(n) + static_cast<std::size_t>(t);
            for (std::size_t i = 0; i <= last; ++i) {
                const double continuation = (p_up * v[i] + p_down * v[i + 1]) * df;
                const double spot = escrowed * up_powers[base - 2 * i] + pv_rem;
                v[i] = std::max(phi * (spot - strike), std::max(continuation, 0.0));
            }
        } else {
            for (std::size_t i = 0; i <= last; ++i) {
                const double continuation = (p_up * v[i] + p_down * v[i + 1]) * df;
                v[i] = std::max(intrinsic(spot_at(s, static_cast<int>(i), t)), continuation);
            }
        }
        capture_step_nodes(out, s, values, t);
    }
    out.pv = values[0];
    return out;
}

// ── Public interface ────────────────────────────────────────────────

double BinomialEngine::present_value() const {
    return solution().pv;
}

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

// ── Barrier engine ──────────────────────────────────────────────────

BinomialBarrierEngine::BinomialBarrierEngine(const detail::ValuationState& state)
    : BinomialEngine(state) {
    if (!std::holds_alternative<BarrierSpec>(state_.spec)) {
        throw ConfigurationError("BinomialBarrierEngine requires a BarrierSpec");
    }
}

const BarrierSpec& BinomialBarrierEngine::spec() const {
    return std::get<BarrierSpec>(state_.spec);
}

int BinomialBarrierEngine::num_steps() const {
    if (!barrier_steps_.has_value()) {
        barrier_steps_ = resolve_barrier_num_steps();
    }
    return *barrier_steps_;
}

int BinomialBarrierEngine::resolve_barrier_num_steps() const {
    const BarrierSpec& spec = this->spec();
    const int base_steps = base_num_steps();
    const double spot = state_.underlying.initial_value();
    const double barrier = spec.barrier();
    const double sigma = state_.underlying.volatility();
    const double ttm = state_.maturity_year_fraction();
    if (spot <= 0.0 || barrier <= 0.0 || sigma <= 0.0 || ttm <= 0.0) {
        return base_steps;
    }

    // Inception-triggered specs short-circuit before the barrier solver ever
    // runs, so alignment is moot.
    if (detail::barrier_triggered_at_inception(state_)) {
        return base_steps;
    }

    const double log_distance = std::log(std::max(spot, barrier) / std::min(spot, barrier));
    const double divisor = log_distance * log_distance;
    if (divisor <= 0.0) {
        return base_steps;
    }

    // Continuous: place a CRR layer on H (Boyle-Lau 1994, factor = i).
    // Discrete:   place H midway between two CRR layers (factor = i - 0.5),
    // the binomial analog of Cheuk-Vorst 1996 / Boyle-Tian 1998 — on-layer
    // placement introduces a first-order grid bias in the kill threshold.
    const double shift = spec.monitoring() == BarrierMonitoring::Continuous ? 0.0 : 0.5;
    const int max_steps = std::max(1000, base_steps * 5);
    for (int i = 1; i <= base_steps; ++i) {
        const double factor = i - shift;
        const auto candidate = static_cast<int>((factor * factor * sigma * sigma * ttm) / divisor);
        if (candidate >= base_steps) {
            if (candidate > max_steps) {
                // Alignment unattainable within the memory budget — warn and
                // fall back to unaligned at `max_steps`.
                const double log_distance_pct =
                    100.0 * log_distance / std::max(std::log(spot), 1.0e-12);
                warn(std::format(
                    "Binomial barrier pricing: Boyle-Lau step alignment needs ~{} time steps "
                    "to place a tree layer on the barrier (spot={:g}, barrier={:g}, "
                    "|log(H/S)|={:.2e}, ~{:.3f}% of log-spot), but the step cap is {}. The "
                    "tree will run at {} steps without barrier alignment and the price will "
                    "converge only as O(1/√n) with a large constant. For barriers this close "
                    "to spot, PDE_FD is strongly recommended as the more accurate engine.",
                    candidate,
                    spot,
                    barrier,
                    log_distance,
                    log_distance_pct,
                    max_steps,
                    max_steps));
            }
            return std::min(candidate, max_steps);
        }
    }
    return base_steps;
}

std::vector<char> BinomialBarrierEngine::monitoring_step_mask(int num_steps) const {
    const BarrierSpec& spec = this->spec();
    const auto size = static_cast<std::size_t>(num_steps) + 1;

    // The `.empty()` guards are provably dead (size >= 1) but stop GCC's
    // -Wnull-dereference false positive on the vector data pointer.
    if (spec.monitoring() == BarrierMonitoring::Continuous) {
        std::vector<char> mask(size, 1);
        if (!mask.empty()) {
            mask.front() = 0;
        }
        return mask;
    }

    std::set<int> observation_indices;
    if (const auto& monitoring_dates = spec.monitoring_dates(); monitoring_dates.has_value()) {
        // Map each observation date to the nearest tree date.
        const auto tree_dates = date_range(state_.pricing_date(), state_.maturity(), size);
        for (const DateTime obs : *monitoring_dates) {
            int best = 0;
            auto best_gap = std::chrono::abs(tree_dates[0] - obs);
            for (int t = 1; t <= num_steps; ++t) {
                const auto gap = std::chrono::abs(tree_dates[static_cast<std::size_t>(t)] - obs);
                if (gap < best_gap) {
                    best_gap = gap;
                    best = t;
                }
            }
            observation_indices.insert(best);
        }
    } else if (const std::optional<int> num_observations = spec.num_observations();
               num_observations.has_value()) {
        // N tree indices at num_steps/N, 2*num_steps/N, ..., num_steps —
        // index 0 (the pricing date) excluded per BGK / Boyle-Tian
        // convention. Round half-to-even to match numpy.
        const int observations = *num_observations;
        for (int i = 1; i <= observations; ++i) {
            const double position =
                static_cast<double>(i) * static_cast<double>(num_steps) / observations;
            observation_indices.insert(static_cast<int>(std::nearbyint(position)));
        }
    } else {
        throw ValidationError(
            "Discrete barrier monitoring requires num_observations or monitoring_dates.");
    }

    std::vector<char> mask(size, 0);
    for (const int index : observation_indices) {
        mask[static_cast<std::size_t>(index)] = 1;
    }
    if (!mask.empty()) {
        mask.front() = 0;
    }
    return mask;
}

Eigen::ArrayXd BinomialBarrierEngine::ko_rebate_values(const Setup& s) const {
    const BarrierSpec& spec = this->spec();
    const int n = s.num_steps;
    const double rebate = spec.rebate();
    if (rebate <= 0.0) {
        return Eigen::ArrayXd::Zero(n + 1);
    }
    if (spec.rebate_timing() == RebateTiming::AtHit) {
        return Eigen::ArrayXd::Constant(n + 1, rebate);
    }
    // AT_EXPIRY: discount the rebate from maturity back to each step.
    Eigen::ArrayXd values(n + 1);
    values[n] = rebate;
    double to_maturity = 1.0;
    for (int t = n - 1; t >= 0; --t) {
        to_maturity = to_maturity * s.discount_factors[t];
        values[t] = rebate * to_maturity;
    }
    return values;
}

BinomialEngine::Solution BinomialBarrierEngine::solve_knock_out() const {
    const int n = num_steps();
    const Setup& s = setup();
    const bool early = early_exercise();
    const BarrierSpec& spec = this->spec();
    const bool is_call = spec.option_type() == OptionType::Call;
    const double strike = spec.strike();
    const double barrier = spec.barrier();
    const bool is_up = spec.direction() == BarrierDirection::Up;
    const std::vector<char> mask = monitoring_step_mask(n);
    const Eigen::ArrayXd rebate = ko_rebate_values(s);

    const auto intrinsic = [&](double spot) {
        return is_call ? std::max(spot - strike, 0.0) : std::max(strike - spot, 0.0);
    };
    const auto hit = [&](double spot) { return is_up ? spot >= barrier : spot <= barrier; };

    Solution out = nan_solution();
    std::vector<double> values(static_cast<std::size_t>(n) + 1);
    for (int i = 0; i <= n; ++i) {
        values[static_cast<std::size_t>(i)] = intrinsic(spot_at(s, i, n));
    }
    if (mask[static_cast<std::size_t>(n)] != 0) {
        for (int i = 0; i <= n; ++i) {
            if (hit(spot_at(s, i, n))) {
                values[static_cast<std::size_t>(i)] = rebate[n];
            }
        }
    }
    capture_step_nodes(out, s, values, n);

    for (int t = n - 1; t >= 0; --t) {
        const double p_up = s.p[t];
        const double p_down = 1.0 - p_up;
        const double df = s.discount_factors[t];
        for (int i = 0; i <= t; ++i) {
            const auto k = static_cast<std::size_t>(i);
            const double continuation = df * (p_up * values[k] + p_down * values[k + 1]);
            values[k] = early ? std::max(intrinsic(spot_at(s, i, t)), continuation) : continuation;
        }
        if (mask[static_cast<std::size_t>(t)] != 0) {
            for (int i = 0; i <= t; ++i) {
                if (hit(spot_at(s, i, t))) {
                    values[static_cast<std::size_t>(i)] = rebate[t];
                }
            }
        }
        capture_step_nodes(out, s, values, t);
    }
    out.pv = values[0];
    return out;
}

BinomialEngine::Solution BinomialBarrierEngine::solve_knock_in() const {
    const int n = num_steps();
    const Setup& s = setup();
    const bool early = early_exercise();
    const BarrierSpec& spec = this->spec();
    const bool is_call = spec.option_type() == OptionType::Call;
    const double strike = spec.strike();
    const double barrier = spec.barrier();
    const bool is_up = spec.direction() == BarrierDirection::Up;
    const std::vector<char> mask = monitoring_step_mask(n);
    // KI rebates are AT_EXPIRY by contract: paid at maturity if never hit,
    // so the inactive terminal layer carries the undiscounted rebate.
    const double rebate = spec.rebate();

    const auto intrinsic = [&](double spot) {
        return is_call ? std::max(spot - strike, 0.0) : std::max(strike - spot, 0.0);
    };
    const auto hit = [&](double spot) { return is_up ? spot >= barrier : spot <= barrier; };

    // Two-state recursion: `active` is the vanilla (post-knock-in) value,
    // `inactive` the not-yet-knocked-in value; a hit at a monitoring step
    // moves the inactive state onto the active value.
    Solution out = nan_solution();
    std::vector<double> active(static_cast<std::size_t>(n) + 1);
    std::vector<double> inactive(static_cast<std::size_t>(n) + 1);
    for (int i = 0; i <= n; ++i) {
        const auto k = static_cast<std::size_t>(i);
        active[k] = intrinsic(spot_at(s, i, n));
        inactive[k] = rebate;
        if (mask[static_cast<std::size_t>(n)] != 0 && hit(spot_at(s, i, n))) {
            inactive[k] = active[k];
        }
    }
    capture_step_nodes(out, s, inactive, n);

    for (int t = n - 1; t >= 0; --t) {
        const double p_up = s.p[t];
        const double p_down = 1.0 - p_up;
        const double df = s.discount_factors[t];
        const bool monitored = mask[static_cast<std::size_t>(t)] != 0;
        for (int i = 0; i <= t; ++i) {
            const auto k = static_cast<std::size_t>(i);
            const double continuation = df * (p_up * active[k] + p_down * active[k + 1]);
            active[k] = early ? std::max(intrinsic(spot_at(s, i, t)), continuation) : continuation;
        }
        for (int i = 0; i <= t; ++i) {
            const auto k = static_cast<std::size_t>(i);
            double continuation = df * (p_up * inactive[k] + p_down * inactive[k + 1]);
            if (monitored && hit(spot_at(s, i, t))) {
                continuation = active[k];
            }
            inactive[k] = continuation;
        }
        capture_step_nodes(out, s, inactive, t);
    }
    out.pv = inactive[0];
    return out;
}

BinomialEngine::Solution BinomialBarrierEngine::resolved_knock_out_solution() const {
    const int n = num_steps();
    const Setup& s = setup();
    const Eigen::ArrayXd rebate = ko_rebate_values(s);

    Solution out = nan_solution();
    out.pv = rebate[0];
    if (n >= 1) {
        out.f01 = rebate[1];
        out.f11 = rebate[1];
        out.s01 = spot_at(s, 0, 1);
        out.s11 = spot_at(s, 1, 1);
    }
    if (n >= 2) {
        out.f02 = rebate[2];
        out.f12 = rebate[2];
        out.f22 = rebate[2];
        out.s02 = spot_at(s, 0, 2);
        out.s12 = spot_at(s, 1, 2);
        out.s22 = spot_at(s, 2, 2);
    }
    return out;
}

BinomialEngine::Solution BinomialBarrierEngine::compute_solution() const {
    const bool knock_out = spec().action() == BarrierAction::Out;
    if (detail::barrier_triggered_at_inception(state_)) {
        // Triggered knock-in: the contract is the vanilla.
        return knock_out ? resolved_knock_out_solution() : BinomialEngine::compute_solution();
    }
    return knock_out ? solve_knock_out() : solve_knock_in();
}

bool BinomialBarrierEngine::stencil_straddles_barrier(int step) const {
    // Only continuously monitored, still-alive barriers put an absorbing
    // discontinuity inside the extraction stencil: discrete monitoring only
    // observes at monitoring nodes, and triggered specs collapse to smooth
    // lattices (KO: spot-independent rebate; KI: the vanilla).
    const BarrierSpec& spec = this->spec();
    if (spec.monitoring() != BarrierMonitoring::Continuous) {
        return false;
    }
    if (detail::barrier_triggered_at_inception(state_)) {
        return false;
    }
    const Setup& s = setup();
    for (int i = 0; i <= step; ++i) {
        const double spot = spot_at(s, i, step);
        if (spec.direction() == BarrierDirection::Up ? spot >= spec.barrier()
                                                     : spot <= spec.barrier()) {
            return true;
        }
    }
    return false;
}

double BinomialBarrierEngine::present_value() const {
    const BarrierSpec& spec = this->spec();
    if (spec.action() == BarrierAction::Out && detail::barrier_triggered_at_inception(state_)) {
        // Knocked out immediately: O(1) rebate PV, no tree.
        if (spec.rebate() <= 0.0) {
            return 0.0;
        }
        if (spec.rebate_timing() == RebateTiming::AtHit) {
            return spec.rebate();
        }
        return spec.rebate() * state_.discount_curve().df(state_.maturity_year_fraction());
    }
    return BinomialEngine::present_value();
}

double BinomialBarrierEngine::delta() const {
    if (stencil_straddles_barrier(1)) {
        throw UnsupportedFeatureError(
            "Binomial tree delta is unreliable when the step-1 extraction stencil straddles "
            "the barrier (spot is too close to H, so the up- or down-node lies past the "
            "absorbing boundary). Use PricingMethod.PDE_FD for greeks in this regime.");
    }
    return BinomialEngine::delta();
}

double BinomialBarrierEngine::gamma() const {
    if (num_steps() < 2) {
        throw ValidationError("Tree gamma requires num_steps >= 2.");
    }
    if (stencil_straddles_barrier(2)) {
        throw UnsupportedFeatureError(
            "Binomial tree gamma is unreliable when the step-2 extraction stencil straddles "
            "the barrier (one of the uu/dd nodes lies past the absorbing boundary). Use "
            "PricingMethod.PDE_FD for greeks in this regime.");
    }
    return BinomialEngine::gamma();
}

}  // namespace dp::engines
