#include "dp/valuation.hpp"

#include "dp/diagnostics.hpp"
#include "dp/exceptions.hpp"

#include "engines/binomial.hpp"
#include "engines/bsm.hpp"
#include "engines/engine.hpp"
#include "valuation_state.hpp"

#include <cctype>
#include <chrono>
#include <compare>
#include <format>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace dp {

namespace {

/// Default time bump (calendar days) for numerical barrier theta. 1d is too
/// small for MC (CRN noise dominates) and questionable for FD/binomial; 7d
/// dampens noise while skipping few monitoring dates.
constexpr double kBarrierThetaTimeBumpDays = 7.0;
constexpr double kDefaultThetaTimeBumpDays = 1.0;
constexpr double kDefaultVegaBump = 0.01;
constexpr double kDefaultRateBump = 0.01;
constexpr double kPerCentScale = 0.01;

enum class GreekKind { PresentValue, Delta, Gamma, Vega, Theta, Rho };

/// Memoisation key: mirrors the Python cache keyed on accessor + kwargs.
struct CacheKey {
    GreekKind kind;
    std::optional<GreekCalculationMethod> method;
    std::optional<double> bump;

    friend auto operator<=>(const CacheKey&, const CacheKey&) = default;
};

std::string capitalised(std::string_view value) {
    std::string out(value);
    if (!out.empty()) {
        out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
    }
    return out;
}

std::string upper(std::string_view value) {
    std::string out(value);
    for (auto& c : out) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return out;
}

// ── Parameter resolution ────────────────────────────────────────────

std::optional<ValuationParams> resolve_params(PricingMethod method,
                                              std::optional<ValuationParams> params,
                                              const OptionSpec& spec) {
    const auto* barrier = std::get_if<BarrierSpec>(&spec);

    if (!params.has_value()) {
        switch (method) {
            case PricingMethod::MonteCarlo:
                return MonteCarloParams{};
            case PricingMethod::Binomial:
                if (barrier != nullptr) {
                    // Continuous: on-node alignment (Boyle-Lau 1994). Discrete:
                    // half-step alignment; 3000 base steps keeps CRR's O(1/N)
                    // error tight on small-price reverse-barrier cases.
                    const int steps =
                        barrier->monitoring() == BarrierMonitoring::Continuous ? 1000 : 3000;
                    return BinomialParams{.num_steps = steps};
                }
                return BinomialParams{};
            case PricingMethod::PdeFd:
                if (barrier != nullptr) {
                    return PdeParams::for_barriers(barrier->monitoring());
                }
                return PdeParams{};
            case PricingMethod::Bsm:
                return std::nullopt;
        }
        return std::nullopt;
    }

    const auto require = [&](auto tag, std::string_view params_name) {
        using Expected = decltype(tag);
        const auto* typed = std::get_if<Expected>(&*params);
        if (typed == nullptr) {
            throw ConfigurationError(
                std::format("pricing_method={} requires params={}", name(method), params_name));
        }
        typed->validate();
    };

    switch (method) {
        case PricingMethod::MonteCarlo:
            require(MonteCarloParams{}, "MonteCarloParams");
            return params;
        case PricingMethod::Binomial:
            require(BinomialParams{}, "BinomialParams");
            return params;
        case PricingMethod::PdeFd:
            require(PdeParams{}, "PdeParams");
            return params;
        case PricingMethod::Bsm:
            break;
    }
    throw ConfigurationError(
        std::format("pricing_method={} does not accept valuation params", name(method)));
}

// ── Barrier schedule ────────────────────────────────────────────────

/// Monitoring dates for a DISCRETE barrier; nullopt for CONTINUOUS.
std::optional<std::vector<DateTime>> barrier_monitoring_dates(const detail::ValuationState& state) {
    const auto& spec = std::get<BarrierSpec>(state.spec);
    if (spec.monitoring() == BarrierMonitoring::Continuous) {
        return std::nullopt;
    }
    if (spec.monitoring_dates().has_value()) {
        return spec.monitoring_dates();
    }
    // N+1 points from pricing_date to maturity; drop t=0 to leave N dates at
    // T/N, 2T/N, ..., T (the academic convention for discrete monitoring).
    auto grid = date_range(state.pricing_date(),
                           state.maturity(),
                           static_cast<std::size_t>(*spec.num_observations()) + 1);
    grid.erase(grid.begin());
    return grid;
}

// ── Engine dispatch ─────────────────────────────────────────────────

std::unique_ptr<engines::PricingEngine> build_engine(const detail::ValuationState& state) {
    const auto method = state.pricing_method;
    const auto exercise = state.exercise_type();

    if (std::holds_alternative<BarrierSpec>(state.spec)) {
        throw UnsupportedFeatureError(std::format(
            "Barrier options with {} exercise do not yet support {} pricing in this port.",
            name(exercise),
            name(method)));
    }

    switch (method) {
        case PricingMethod::Bsm:
            if (exercise != ExerciseType::European) {
                throw UnsupportedFeatureError(
                    "BSM is only applicable to European option valuation. Select a different "
                    "pricing method for American options such as Binomial, PDE_FD or "
                    "MONTE_CARLO.");
            }
            return std::make_unique<engines::BsmEngine>(state);
        case PricingMethod::Binomial:
            return std::make_unique<engines::BinomialEngine>(state);
        case PricingMethod::PdeFd:
            throw UnsupportedFeatureError(
                std::format("{} pricing is not yet implemented in this port.", name(method)));
        case PricingMethod::MonteCarlo:
            break;
    }
    throw ConfigurationError(
        "Monte Carlo pricing requires underlying to be a PathSimulation instance");
}

// ── Greek method resolution ─────────────────────────────────────────

GreekCalculationMethod auto_select_greek_method(const detail::ValuationState& state,
                                                bool tree_capable,
                                                bool grid_capable) {
    const bool is_barrier = std::holds_alternative<BarrierSpec>(state.spec);
    if (state.pricing_method == PricingMethod::Bsm && !is_barrier) {
        return GreekCalculationMethod::Analytical;
    }
    if (tree_capable && state.pricing_method == PricingMethod::Binomial) {
        return GreekCalculationMethod::Tree;
    }
    if (grid_capable && state.pricing_method == PricingMethod::PdeFd) {
        return GreekCalculationMethod::Grid;
    }
    return GreekCalculationMethod::Numerical;
}

/// Resolve and validate the Greek method. `nullopt` auto-selects the best
/// engine-native method; an explicit choice is checked against the engine and
/// the per-Greek capability flags.
GreekCalculationMethod resolve_greek_method(const detail::ValuationState& state,
                                            std::optional<GreekCalculationMethod> requested,
                                            bool tree_capable = false,
                                            bool grid_capable = false) {
    if (!requested.has_value()) {
        return auto_select_greek_method(state, tree_capable, grid_capable);
    }
    const auto method = *requested;

    // Closed-form barrier Greeks are not exposed by the analytical barrier engine.
    if (std::holds_alternative<BarrierSpec>(state.spec) &&
        method == GreekCalculationMethod::Analytical) {
        throw UnsupportedFeatureError(
            "Barrier options do not support GreekCalculationMethod.ANALYTICAL. Use TREE "
            "(BINOMIAL), GRID (PDE_FD), or NUMERICAL (bump-and-revalue).");
    }

    struct Rule {
        PricingMethod required;
        bool capable;
        std::string_view supported_greeks;
    };
    std::optional<Rule> rule;
    switch (method) {
        case GreekCalculationMethod::Analytical:
            rule = Rule{PricingMethod::Bsm, true, "all BSM greeks"};
            break;
        case GreekCalculationMethod::Tree:
            rule = Rule{PricingMethod::Binomial, tree_capable, "delta, gamma, and theta"};
            break;
        case GreekCalculationMethod::Grid:
            rule = Rule{PricingMethod::PdeFd, grid_capable, "delta, gamma, and theta"};
            break;
        case GreekCalculationMethod::Pathwise:
        case GreekCalculationMethod::LikelihoodRatio:
            if (state.pricing_method != PricingMethod::MonteCarlo) {
                throw ValidationError(
                    std::format("{} greeks are only available for MONTE_CARLO pricing method.",
                                to_string(method)));
            }
            break;
        case GreekCalculationMethod::Numerical:
            break;
    }

    if (rule.has_value()) {
        if (state.pricing_method != rule->required) {
            throw ValidationError(std::format("{} greeks are only available for {} pricing method.",
                                              capitalised(to_string(method)),
                                              name(rule->required)));
        }
        if (!rule->capable) {
            throw ValidationError(
                std::format("{} extraction is not available for this greek. Only {} support "
                            "GreekCalculationMethod.{}.",
                            capitalised(to_string(method)),
                            rule->supported_greeks,
                            name(method)));
        }
    }
    return method;
}

/// A finite-difference bump must be strictly positive, and is only meaningful
/// for methods that bump (NUMERICAL, plus any `extra_allowed`).
void validate_bump(std::optional<double> bump,
                   GreekCalculationMethod resolved,
                   std::string_view bump_name,
                   std::initializer_list<GreekCalculationMethod> extra_allowed = {}) {
    if (!bump.has_value()) {
        return;
    }
    if (!(*bump > 0.0)) {
        throw ValidationError(
            std::format("{} must be strictly positive, got {}.", bump_name, *bump));
    }
    if (resolved == GreekCalculationMethod::Numerical) {
        return;
    }
    for (const auto allowed : extra_allowed) {
        if (resolved == allowed) {
            return;
        }
    }
    throw ValidationError(std::format(
        "{}={} was passed but the greek calculation method resolved to {}, which does not "
        "use a finite-difference bump. Either remove {} (to use the analytical/grid/tree "
        "path) or pass greek_calc_method=GreekCalculationMethod.NUMERICAL to opt in to "
        "bump-and-revalue.",
        bump_name,
        *bump,
        name(resolved),
        bump_name));
}

/// True when the method-specific params carry `control_variate_european`.
bool control_variate_requested(const std::optional<ValuationParams>& params) {
    if (!params.has_value()) {
        return false;
    }
    return std::visit([](const auto& p) { return p.control_variate_european; }, *params);
}

/// Copy of `params` with the control-variate flag cleared (for the European
/// numerical leg, which must not recurse into another adjustment).
ValuationParams without_control_variate(const ValuationParams& params) {
    return std::visit(
        [](auto p) -> ValuationParams {
            p.control_variate_european = false;
            return p;
        },
        params);
}

DateTime add_days(DateTime date, double days) {
    using namespace std::chrono;
    return date + round<microseconds>(duration<double, std::chrono::days::period>(days));
}

}  // namespace

// ── Impl ────────────────────────────────────────────────────────────

struct OptionValuation::Impl {
    explicit Impl(detail::ValuationState s) : state(std::move(s)) {}

    detail::ValuationState state;
    std::unique_ptr<engines::PricingEngine> engine;

    // Output cache. A recursive mutex lets memoised accessors call each other
    // (gamma -> present_value) from the same thread; concurrent callers on
    // one instance serialise so an expensive solve runs once.
    mutable std::recursive_mutex mutex;
    mutable std::map<CacheKey, double> cache;

    template <typename Compute>
    double memoize(const CacheKey& key, const Compute& compute) const {
        const std::lock_guard lock(mutex);
        if (const auto it = cache.find(key); it != cache.end()) {
            return it->second;
        }
        const double value = compute();
        cache.emplace(key, value);
        return value;
    }

    /// Fresh valuation with the same spec/method/params on a bumped underlying.
    [[nodiscard]] OptionValuation rebuild(UnderlyingData bumped) const {
        return {std::move(bumped), state.spec, state.pricing_method, state.params};
    }

    /// European control-variate adjustment for American pricing:
    /// `PV_am + (PV_eu_bsm - PV_eu_numerical)`, both European legs priced on
    /// the same underlying (the numerical leg with the same params, CV off).
    [[nodiscard]] double apply_control_variate(double base_pv) const {
        if (state.exercise_type() != ExerciseType::American) {
            throw ValidationError(
                "control_variate_european is only valid for options with American exercise.");
        }
        if (state.pricing_method != PricingMethod::Binomial &&
            state.pricing_method != PricingMethod::PdeFd &&
            state.pricing_method != PricingMethod::MonteCarlo) {
            throw UnsupportedFeatureError(
                "control_variate_european is only supported for BINOMIAL, PDE_FD, and "
                "MONTE_CARLO pricing.");
        }
        if (std::holds_alternative<PayoffSpec>(state.spec)) {
            throw UnsupportedFeatureError(
                "control_variate_european is not supported for PayoffSpec.");
        }
        const auto* vanilla = std::get_if<VanillaSpec>(&state.spec);
        if (vanilla == nullptr || !state.option_type.has_value()) {
            throw UnsupportedFeatureError(
                "control_variate_european requires a CALL or PUT option type.");
        }

        if (!state.params.has_value()) {
            throw ConfigurationError("control_variate_european requires method params");
        }
        const VanillaSpec euro_spec({
            .option_type = vanilla->option_type(),
            .exercise_type = ExerciseType::European,
            .strike = vanilla->strike(),
            .maturity = vanilla->maturity(),
            .currency = vanilla->currency(),
            .contract_size = vanilla->contract_size(),
        });

        const double euro_numerical = OptionValuation(state.underlying,
                                                      euro_spec,
                                                      state.pricing_method,
                                                      without_control_variate(*state.params))
                                          .present_value();
        const double euro_bsm =
            OptionValuation(state.underlying, euro_spec, PricingMethod::Bsm).present_value();

        return base_pv + (euro_bsm - euro_numerical);
    }

    [[nodiscard]] bool is_barrier() const noexcept {
        return std::holds_alternative<BarrierSpec>(state.spec);
    }
};

// ── Construction ────────────────────────────────────────────────────

OptionValuation::OptionValuation(UnderlyingData underlying,
                                 OptionSpec spec,
                                 PricingMethod pricing_method,
                                 std::optional<ValuationParams> params) {
    const auto& spec_currency = currency_of(spec);
    if (spec_currency.has_value() && *spec_currency != underlying.currency()) {
        throw UnsupportedFeatureError(
            "Cross-currency valuation is not supported. Option currency must match the "
            "underlying market currency.");
    }
    std::string currency = spec_currency.value_or(underlying.currency());

    const auto option_type = option_type_of(spec);
    if (pricing_method == PricingMethod::Bsm && !option_type.has_value()) {
        throw UnsupportedFeatureError(
            "BSM pricing requires a CALL or PUT option type (not available for custom "
            "PayoffSpec).");
    }

    if (maturity_of(spec) <= underlying.pricing_date()) {
        throw ValidationError("Option maturity must be after pricing_date.");
    }

    if (pricing_method == PricingMethod::MonteCarlo) {
        throw ConfigurationError(
            "Monte Carlo pricing requires underlying to be a PathSimulation instance");
    }

    auto resolved_params = resolve_params(pricing_method, params, spec);

    if (pricing_method == PricingMethod::Binomial && std::holds_alternative<BarrierSpec>(spec) &&
        !underlying.discrete_dividends().empty()) {
        throw UnsupportedFeatureError(
            "Binomial pricing of barrier options with discrete dividends is not supported. "
            "The escrowed-dividend tree adjustment used for vanilla options does not "
            "preserve the correct barrier-hit dynamics at ex-dividend dates. Use PDE_FD or "
            "MONTE_CARLO instead.");
    }

    impl_ = std::make_unique<Impl>(detail::ValuationState{
        .underlying = std::move(underlying),
        .spec = std::move(spec),
        .pricing_method = pricing_method,
        .params = resolved_params,
        .currency = std::move(currency),
        .option_type = option_type,
    });

    if (impl_->is_barrier()) {
        const auto dates = barrier_monitoring_dates(impl_->state);
        if (dates.has_value() && dates->front() < impl_->state.pricing_date()) {
            throw ValidationError(
                "Barrier monitoring schedule must not start before pricing_date.");
        }
    }

    impl_->engine = build_engine(impl_->state);
}

OptionValuation::~OptionValuation() = default;
OptionValuation::OptionValuation(OptionValuation&&) noexcept = default;
OptionValuation& OptionValuation::operator=(OptionValuation&&) noexcept = default;

// ── Present value ───────────────────────────────────────────────────

double OptionValuation::present_value() const {
    return impl_->memoize({GreekKind::PresentValue, std::nullopt, std::nullopt}, [&] {
        const double base = impl_->engine->present_value();
        if (control_variate_requested(impl_->state.params)) {
            return impl_->apply_control_variate(base);
        }
        return base;
    });
}

// ── Greeks ──────────────────────────────────────────────────────────

double OptionValuation::delta(GreekOptions options) const {
    return impl_->memoize({GreekKind::Delta, options.method, options.bump}, [&] {
        const auto& state = impl_->state;
        const auto method = resolve_greek_method(state, options.method, true, true);
        validate_bump(options.bump, method, "epsilon");
        if (method != GreekCalculationMethod::Numerical) {
            return impl_->engine->delta();
        }
        const double s0 = state.underlying.initial_value();
        const double eps = options.bump.value_or(s0 / 100.0);
        const auto up = impl_->rebuild(state.underlying.with_initial_value(s0 + eps));
        const auto dn = impl_->rebuild(state.underlying.with_initial_value(s0 - eps));
        return (up.present_value() - dn.present_value()) / (2.0 * eps);
    });
}

double OptionValuation::gamma(GreekOptions options) const {
    return impl_->memoize({GreekKind::Gamma, options.method, options.bump}, [&] {
        const auto& state = impl_->state;
        const auto method = resolve_greek_method(state, options.method, true, true);
        validate_bump(options.bump, method, "epsilon", {GreekCalculationMethod::Pathwise});
        if (method == GreekCalculationMethod::LikelihoodRatio) {
            throw ValidationError(
                "likelihood_ratio is not available for gamma. Use PATHWISE (central-difference "
                "of pathwise delta) or NUMERICAL.");
        }
        if (method != GreekCalculationMethod::Numerical) {
            return impl_->engine->gamma();
        }
        const double s0 = state.underlying.initial_value();
        const double eps = options.bump.value_or(s0 / 100.0);
        const auto up = impl_->rebuild(state.underlying.with_initial_value(s0 + eps));
        const auto dn = impl_->rebuild(state.underlying.with_initial_value(s0 - eps));
        const double centre = present_value();
        return (up.present_value() - 2.0 * centre + dn.present_value()) / (eps * eps);
    });
}

double OptionValuation::vega(GreekOptions options) const {
    return impl_->memoize({GreekKind::Vega, options.method, options.bump}, [&] {
        const auto& state = impl_->state;
        const auto method = resolve_greek_method(state, options.method);
        validate_bump(options.bump, method, "epsilon");
        if (method != GreekCalculationMethod::Numerical) {
            return impl_->engine->vega();
        }
        const double vol = state.underlying.volatility();
        const double eps = options.bump.value_or(kDefaultVegaBump);
        const auto up = impl_->rebuild(state.underlying.with_volatility(vol + eps));
        const auto dn = impl_->rebuild(state.underlying.with_volatility(vol - eps));
        return (up.present_value() - dn.present_value()) / (2.0 * eps) * kPerCentScale;
    });
}

double OptionValuation::theta(GreekOptions options) const {
    return impl_->memoize({GreekKind::Theta, options.method, options.bump}, [&] {
        const auto& state = impl_->state;
        const auto method = resolve_greek_method(state, options.method, true, true);
        validate_bump(options.bump, method, "time_bump_days");
        if (method != GreekCalculationMethod::Numerical) {
            return impl_->engine->theta();
        }
        const double days = options.bump.value_or(impl_->is_barrier() ? kBarrierThetaTimeBumpDays
                                                                      : kDefaultThetaTimeBumpDays);
        const DateTime bumped_date = add_days(state.pricing_date(), days);
        if (bumped_date >= state.maturity()) {
            return 0.0;
        }
        const double value_now = present_value();
        const auto bumped = impl_->rebuild(state.underlying.with_pricing_date(bumped_date));
        return (bumped.present_value() - value_now) / days;
    });
}

double OptionValuation::rho(GreekOptions options) const {
    return impl_->memoize({GreekKind::Rho, options.method, options.bump}, [&] {
        const auto& state = impl_->state;
        const auto method = resolve_greek_method(state, options.method);
        validate_bump(options.bump, method, "rate_bump");
        if (method != GreekCalculationMethod::Numerical) {
            return impl_->engine->rho();
        }
        const double bump = options.bump.value_or(kDefaultRateBump);
        const auto& curve = state.discount_curve();
        const auto up = impl_->rebuild(
            state.underlying.with_discount_curve(curve.bump_parallel_zero_rate(bump / 2.0)));
        const auto dn = impl_->rebuild(
            state.underlying.with_discount_curve(curve.bump_parallel_zero_rate(-bump / 2.0)));
        return (up.present_value() - dn.present_value()) / bump * kPerCentScale;
    });
}

// ── Accessors ───────────────────────────────────────────────────────

const UnderlyingData& OptionValuation::underlying() const noexcept {
    return impl_->state.underlying;
}
const OptionSpec& OptionValuation::spec() const noexcept {
    return impl_->state.spec;
}
PricingMethod OptionValuation::pricing_method() const noexcept {
    return impl_->state.pricing_method;
}
const std::optional<ValuationParams>& OptionValuation::params() const noexcept {
    return impl_->state.params;
}
std::optional<OptionType> OptionValuation::option_type() const noexcept {
    return impl_->state.option_type;
}
DateTime OptionValuation::maturity() const {
    return impl_->state.maturity();
}
std::optional<double> OptionValuation::strike() const {
    return impl_->state.strike();
}
const std::string& OptionValuation::currency() const noexcept {
    return impl_->state.currency;
}
ExerciseType OptionValuation::exercise_type() const {
    return impl_->state.exercise_type();
}
double OptionValuation::contract_size() const {
    return contract_size_of(impl_->state.spec);
}
DateTime OptionValuation::pricing_date() const noexcept {
    return impl_->state.pricing_date();
}
const DiscountCurve& OptionValuation::discount_curve() const noexcept {
    return impl_->state.discount_curve();
}
DayCountConvention OptionValuation::day_count_convention() const noexcept {
    return impl_->state.day_count_convention();
}
double OptionValuation::maturity_year_fraction() const {
    return impl_->state.maturity_year_fraction();
}

std::string OptionValuation::describe() const {
    const auto& state = impl_->state;
    const std::string option =
        state.option_type.has_value() ? ", " + upper(to_string(*state.option_type)) : "";
    const auto strike = state.strike();
    const std::string strike_text = strike.has_value() ? std::format(", K={}", *strike) : "";
    return std::format("OptionValuation({}, {}{}, S={}{}, T={}, pricing_date={}, {})",
                       name(state.pricing_method),
                       name(state.exercise_type()),
                       option,
                       state.underlying.initial_value(),
                       strike_text,
                       to_string(state.maturity()),
                       to_string(state.pricing_date()),
                       spec_type_name(state.spec));
}

}  // namespace dp
