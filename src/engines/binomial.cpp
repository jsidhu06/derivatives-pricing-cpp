#include "engines/binomial.hpp"

#include "dp/dates.hpp"
#include "dp/dividends.hpp"
#include "dp/exceptions.hpp"
#include "dp/params.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <variant>

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
    // dividends, then add back at each date the PV of those not yet gone ex.
    double spot = state_.underlying.initial_value();
    const auto& dividends = state_.underlying.discrete_dividends();
    if (!dividends.empty()) {
        const double pv_all = pv_discrete_dividends(dividends,
                                                    dates.front(),
                                                    dates.back(),
                                                    curve,
                                                    std::nullopt,
                                                    state_.day_count_convention(),
                                                    /*include_start=*/true);
        spot = std::max(spot - pv_all, 0.0);
    }

    // Full (n+1)x(n+1) matrix, mirroring the reference implementation; only
    // the upper-triangular region (row <= column) enters the recursion.
    Eigen::MatrixXd lattice(n + 1, n + 1);
    for (int t = 0; t <= n; ++t) {
        for (int i = 0; i <= n; ++i) {
            lattice(i, t) = spot * std::pow(u, t - 2 * i);
        }
    }
    if (!dividends.empty()) {
        for (int t = 0; t <= n; ++t) {
            const double pv_remaining = pv_discrete_dividends(dividends,
                                                              dates.front(),
                                                              dates.back(),
                                                              curve,
                                                              dates[static_cast<std::size_t>(t)],
                                                              state_.day_count_convention(),
                                                              /*include_start=*/false);
            lattice.col(t).array() += pv_remaining;
        }
    }

    setup_.emplace(Setup{
        .delta_t = delta_t,
        .discount_factors = (-forward_rates * delta_t).exp(),
        .p = (growth - d) / (u - d),
        .spot_lattice = std::move(lattice),
    });
    return *setup_;
}

Eigen::MatrixXd BinomialEngine::intrinsic_values(const Eigen::MatrixXd& spots) const {
    if (const auto* payoff = std::get_if<PayoffSpec>(&state_.spec)) {
        Eigen::MatrixXd out(spots.rows(), spots.cols());
        for (Eigen::Index j = 0; j < spots.cols(); ++j) {
            for (Eigen::Index i = 0; i < spots.rows(); ++i) {
                out(i, j) = payoff->payoff(spots(i, j));
            }
        }
        return out;
    }
    const double strike = state_.strike().value_or(0.0);
    if (state_.option_type == OptionType::Call) {
        return (spots.array() - strike).cwiseMax(0.0);
    }
    return (strike - spots.array()).cwiseMax(0.0);
}

const Eigen::MatrixXd& BinomialEngine::option_lattice() const {
    if (option_lattice_.has_value()) {
        return *option_lattice_;
    }

    const int n = num_steps();
    const Setup& s = setup();
    const Eigen::MatrixXd intrinsic = intrinsic_values(s.spot_lattice);
    const bool early = early_exercise();

    Eigen::MatrixXd values = Eigen::MatrixXd::Zero(n + 1, n + 1);
    values.col(n) = intrinsic.col(n);
    for (int t = n - 1; t >= 0; --t) {
        const auto continuation = (s.p[t] * values.col(t + 1).head(t + 1).array() +
                                   (1.0 - s.p[t]) * values.col(t + 1).segment(1, t + 1).array()) *
                                  s.discount_factors[t];
        if (early) {
            values.col(t).head(t + 1) =
                continuation.max(intrinsic.col(t).head(t + 1).array()).matrix();
        } else {
            values.col(t).head(t + 1) = continuation.matrix();
        }
    }

    option_lattice_.emplace(std::move(values));
    return *option_lattice_;
}

double BinomialEngine::present_value() const {
    return option_lattice()(0, 0);
}

double BinomialEngine::delta() const {
    const Eigen::MatrixXd& f = option_lattice();
    const Eigen::MatrixXd& S = setup().spot_lattice;
    return (f(0, 1) - f(1, 1)) / (S(0, 1) - S(1, 1));
}

double BinomialEngine::gamma() const {
    if (num_steps() < 2) {
        throw ValidationError("Tree gamma requires num_steps >= 2.");
    }
    const Eigen::MatrixXd& f = option_lattice();
    const Eigen::MatrixXd& S = setup().spot_lattice;
    const double delta_up = (f(0, 2) - f(1, 2)) / (S(0, 2) - S(1, 2));
    const double delta_down = (f(1, 2) - f(2, 2)) / (S(1, 2) - S(2, 2));
    const double h = (S(0, 2) - S(2, 2)) / 2.0;
    return (delta_up - delta_down) / h;
}

double BinomialEngine::theta() const {
    if (num_steps() < 2) {
        throw ValidationError("Tree theta requires num_steps >= 2.");
    }
    const Eigen::MatrixXd& f = option_lattice();
    const double dt = state_.maturity_year_fraction() / num_steps();
    return (f(1, 2) - f(0, 0)) / (2.0 * dt) / kDaysPerYear;
}

}  // namespace dp::engines
