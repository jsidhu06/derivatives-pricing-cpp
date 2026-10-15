#include "dp/valuation.hpp"

#include "dp/diagnostics.hpp"
#include "dp/exceptions.hpp"

#include "detail/barrier.hpp"
#include "engines/barrier_analytical.hpp"
#include "engines/binomial.hpp"
#include "engines/bsm.hpp"
#include "engines/engine.hpp"
#include "engines/mc.hpp"
#include "engines/pde.hpp"
#include "engines/pde_barrier.hpp"
#include "valuation_state.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <compare>
#include <format>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
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

enum class GreekKind { Delta, Gamma, Vega, Theta, Rho };

/// Memoisation key: the greek plus the options that change its result.
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
                if (std::holds_alternative<PayoffSpec>(spec)) {
                    // Custom payoffs keep the spot spatial grid for now.
                    return PdeParams{.space_grid = PdeSpaceGrid::Spot};
                }
                return PdeParams{};
            case PricingMethod::Analytical:
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
        case PricingMethod::Analytical:
            break;
    }
    throw ConfigurationError(
        std::format("pricing_method={} does not accept valuation params", name(method)));
}

// ── Engine dispatch ─────────────────────────────────────────────────

std::unique_ptr<engines::PricingEngine> build_engine(const detail::ValuationState& state) {
    const auto method = state.pricing_method;
    const auto exercise = state.exercise_type();

    if (std::holds_alternative<BarrierSpec>(state.spec)) {
        switch (method) {
            case PricingMethod::Analytical:
                if (exercise != ExerciseType::European) {
                    throw UnsupportedFeatureError(
                        "ANALYTICAL is only applicable to European option valuation. Select a "
                        "different pricing method for American options such as Binomial, "
                        "PDE_FD or MONTE_CARLO.");
                }
                return std::make_unique<engines::AnalyticalBarrierEngine>(state);
            case PricingMethod::Binomial:
                return std::make_unique<engines::BinomialBarrierEngine>(state);
            case PricingMethod::PdeFd:
                return std::make_unique<engines::PdeBarrierEngine>(state);
            case PricingMethod::MonteCarlo:
                return std::make_unique<engines::McBarrierEngine>(state);
        }
        throw ConfigurationError("unknown pricing method");
    }

    switch (method) {
        case PricingMethod::Analytical:
            if (exercise != ExerciseType::European) {
                throw UnsupportedFeatureError(
                    "ANALYTICAL is only applicable to European option valuation. Select a "
                    "different pricing method for American options such as Binomial, PDE_FD "
                    "or MONTE_CARLO.");
            }
            return std::make_unique<engines::BsmEngine>(state);
        case PricingMethod::Binomial:
            return std::make_unique<engines::BinomialEngine>(state);
        case PricingMethod::PdeFd:
            return std::make_unique<engines::PdeEngine>(state);
        case PricingMethod::MonteCarlo:
            return std::make_unique<engines::McEngine>(state);
    }
    throw ConfigurationError("unknown pricing method");
}

// ── Greek method resolution ─────────────────────────────────────────

/// Pathwise / likelihood-ratio estimators exist for European vanillas on a
/// GBM process without discrete dividends.
bool mc_analytic_eligible(const detail::ValuationState& state) {
    const auto* vanilla = std::get_if<VanillaSpec>(&state.spec);
    return state.pricing_method == PricingMethod::MonteCarlo && state.underlying.is_gbm() &&
           vanilla != nullptr && vanilla->exercise_type() == ExerciseType::European &&
           state.underlying.discrete_dividends().empty();
}

void validate_mc_greek_method(const detail::ValuationState& state, GreekCalculationMethod method) {
    if (state.pricing_method != PricingMethod::MonteCarlo) {
        throw ValidationError(std::format(
            "{} greeks are only available for MONTE_CARLO pricing method.", to_string(method)));
    }
    if (!state.underlying.is_gbm()) {
        throw ValidationError("MC greeks are only available for GBMProcess underlying.");
    }
    const auto* vanilla = std::get_if<VanillaSpec>(&state.spec);
    if (vanilla == nullptr || vanilla->exercise_type() != ExerciseType::European) {
        throw ValidationError(std::format(
            "{} greeks are only implemented for vanilla European options (VanillaSpec).",
            to_string(method)));
    }
    if (!state.underlying.discrete_dividends().empty()) {
        throw UnsupportedFeatureError(
            "Pathwise and likelihood-ratio MC Greeks are not supported with discrete "
            "dividends.");
    }
}

GreekCalculationMethod auto_select_greek_method(const detail::ValuationState& state,
                                                bool tree_capable,
                                                bool grid_capable) {
    const bool is_barrier = std::holds_alternative<BarrierSpec>(state.spec);
    if (state.pricing_method == PricingMethod::Analytical && !is_barrier) {
        return GreekCalculationMethod::Analytical;
    }
    if (tree_capable && state.pricing_method == PricingMethod::Binomial) {
        return GreekCalculationMethod::Tree;
    }
    if (grid_capable && state.pricing_method == PricingMethod::PdeFd) {
        return GreekCalculationMethod::Grid;
    }
    if (mc_analytic_eligible(state)) {
        return GreekCalculationMethod::Pathwise;
    }
    return GreekCalculationMethod::Numerical;
}

/// Block NUMERICAL bump-and-revalue greeks on binomial barrier specs: the
/// Boyle-Lau step count depends on spot, volatility, and time-to-maturity,
/// so bumping any of them re-resolves alignment and the central difference
/// compares trees of different topology. Rho is exempt (rates do not enter
/// the alignment formula); inception-triggered barriers collapse to a
/// deterministic state and are exempt too.
void reject_barrier_binomial_numerical(const detail::ValuationState& state,
                                       GreekCalculationMethod method,
                                       bool allow) {
    if (allow || method != GreekCalculationMethod::Numerical) {
        return;
    }
    if (!std::holds_alternative<BarrierSpec>(state.spec) ||
        state.pricing_method != PricingMethod::Binomial) {
        return;
    }
    if (detail::barrier_triggered_at_inception(state)) {
        return;
    }
    throw UnsupportedFeatureError(
        "Binomial NUMERICAL bump-and-revalue greeks are not supported for barrier options "
        "(rho is exempt). Bumping spot, volatility or time causes Boyle-Lau barrier "
        "alignment to pick a different tree step count for each bumped valuation, so the "
        "central difference compares two different tree topologies. Use "
        "GreekCalculationMethod.TREE for delta/gamma/theta, or switch to "
        "PricingMethod.PDE_FD for vega and for NUMERICAL bump-and-revalue on the full "
        "grid.");
}

/// Resolve and validate the Greek method. `nullopt` auto-selects the best
/// engine-native method; an explicit choice is checked against the engine and
/// the per-Greek capability flags.
GreekCalculationMethod resolve_greek_method(const detail::ValuationState& state,
                                            std::optional<GreekCalculationMethod> requested,
                                            bool tree_capable = false,
                                            bool grid_capable = false,
                                            bool allow_barrier_binomial_numerical = false) {
    if (!requested.has_value()) {
        const auto resolved = auto_select_greek_method(state, tree_capable, grid_capable);
        reject_barrier_binomial_numerical(state, resolved, allow_barrier_binomial_numerical);
        return resolved;
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
            rule = Rule{PricingMethod::Analytical, true, "all greeks"};
            break;
        case GreekCalculationMethod::Tree:
            rule = Rule{PricingMethod::Binomial, tree_capable, "delta, gamma, and theta"};
            break;
        case GreekCalculationMethod::Grid:
            rule = Rule{PricingMethod::PdeFd, grid_capable, "delta, gamma, and theta"};
            break;
        case GreekCalculationMethod::Pathwise:
        case GreekCalculationMethod::LikelihoodRatio:
            validate_mc_greek_method(state, method);
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
    reject_barrier_binomial_numerical(state, method, allow_barrier_binomial_numerical);
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

/// Block NUMERICAL bump-and-revalue on barrier greek/engine combinations
/// known to be unreliable: gamma on MC barriers (central-difference noise
/// dominates the signal at practical path counts) and theta on discretely
/// monitored barriers (bumping the pricing date re-resolves the monitoring
/// schedule, so the bumped side is a different contract).
void reject_barrier_numerical(const detail::ValuationState& state,
                              GreekCalculationMethod method,
                              std::string_view greek,
                              std::string_view reason,
                              std::optional<PricingMethod> engine_constraint,
                              std::optional<BarrierMonitoring> monitoring_constraint) {
    if (method != GreekCalculationMethod::Numerical) {
        return;
    }
    const auto* spec = std::get_if<BarrierSpec>(&state.spec);
    if (spec == nullptr) {
        return;
    }
    if (engine_constraint.has_value() && state.pricing_method != *engine_constraint) {
        return;
    }
    if (monitoring_constraint.has_value() && spec->monitoring() != *monitoring_constraint) {
        return;
    }
    throw UnsupportedFeatureError(
        std::format("Numerical {} is not supported for this barrier valuation: {}. Use "
                    "PricingMethod.PDE_FD (GRID) or PricingMethod.BINOMIAL (TREE) for accurate "
                    "barrier {}.",
                    greek,
                    reason,
                    greek));
}

/// Barrier theta via the Black-Scholes PDE identity
/// `theta = r V - (r - q) S delta - sigma^2 S^2 gamma / 2`, per calendar day,
/// with flat-equivalent rates from the curves at maturity. V/delta/gamma
/// route through the facade's own greeks, so they share the memoised solves
/// and the barrier-clamped bumps, and any future engine-native barrier
/// greeks flow in automatically. Only valid in the continuation region: the
/// caller must have excluded inception-triggered barriers.
double barrier_bsm_identity_theta(const OptionValuation& valuation,
                                  const detail::ValuationState& state) {
    const double maturity = state.maturity_year_fraction();
    const double df_r = state.discount_curve().df(maturity);
    const double rate = -std::log(df_r) / maturity;
    const auto& dividend_curve = state.underlying.dividend_curve();
    const double df_q = dividend_curve.has_value() ? dividend_curve->df(maturity) : 1.0;
    const double yield = -std::log(df_q) / maturity;
    const double spot = state.underlying.initial_value();
    const double sigma = state.underlying.volatility();
    const double theta_annual = rate * valuation.present_value() -
                                (rate - yield) * spot * valuation.delta() -
                                0.5 * sigma * sigma * spot * spot * valuation.gamma();
    return theta_annual / 365.0;  // per calendar day
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

    // Output caches. A recursive mutex lets memoised accessors call each
    // other (gamma -> present_value) from the same thread; concurrent callers
    // on one instance serialise so an expensive solve runs once.
    mutable std::recursive_mutex mutex;
    mutable std::optional<double> present_value;
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
    /// A grid engine's copy inherits this valuation's far-field domain, so
    /// both legs of a finite difference are solved on the same domain.
    [[nodiscard]] OptionValuation rebuild(const detail::Underlying& bumped) const {
        return {bumped, state.spec, state.pricing_method, state.params, engine->grid_domain()};
    }

    /// The vanilla valuation an inception-triggered knock-in collapses to.
    [[nodiscard]] OptionValuation vanilla_equivalent() const {
        const VanillaSpec vanilla =
            detail::vanilla_equivalent_spec(std::get<BarrierSpec>(state.spec));
        return state.underlying.visit([&](const auto& u) {
            return OptionValuation(u, vanilla, state.pricing_method, state.params);
        });
    }

    /// Short-circuit value for a NUMERICAL greek on a barrier triggered at
    /// inception, or nullopt when no short-circuit applies. Bumping the
    /// underlying could push the spot across the trigger boundary, so the
    /// central difference would compare two structurally different
    /// contracts. Knock-in: delegate to the vanilla equivalent. Knock-out:
    /// zero, except the AT_EXPIRY-rebate sensitivity `ko_at_expiry` covers
    /// (theta, rho); the discounted rebate is insensitive to every other bump.
    [[nodiscard]] std::optional<double> barrier_inception_triggered_greek(
        GreekKind kind,
        std::optional<double> bump,
        const std::function<double(const BarrierSpec&)>& ko_at_expiry = {}) const {
        const auto* barrier = std::get_if<BarrierSpec>(&state.spec);
        if (barrier == nullptr || !detail::barrier_triggered_at_inception(state)) {
            return std::nullopt;
        }
        if (barrier->action() == BarrierAction::In) {
            const OptionValuation vanilla = vanilla_equivalent();
            const GreekOptions numerical{.method = GreekCalculationMethod::Numerical, .bump = bump};
            switch (kind) {
                case GreekKind::Delta:
                    return vanilla.delta(numerical);
                case GreekKind::Gamma:
                    return vanilla.gamma(numerical);
                case GreekKind::Vega:
                    return vanilla.vega(numerical);
                case GreekKind::Theta:
                    return vanilla.theta(numerical);
                case GreekKind::Rho:
                    return vanilla.rho(numerical);
            }
            throw ConfigurationError("unexpected greek kind in barrier short-circuit");
        }
        if (barrier->rebate() <= 0.0 || barrier->rebate_timing() == RebateTiming::AtHit) {
            return 0.0;
        }
        if (!ko_at_expiry) {
            return 0.0;
        }
        return ko_at_expiry(*barrier);
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
        if (!state.option_type.has_value()) {
            throw UnsupportedFeatureError(
                "control_variate_european requires a CALL or PUT option type.");
        }

        if (!state.params.has_value()) {
            throw ConfigurationError("control_variate_european requires method params");
        }
        if (state.pricing_method != PricingMethod::Binomial) {
            // The correction (european_analytical - european_numerical) is
            // meant to hold only the engine's discretisation error. PDE_FD and
            // MONTE_CARLO drop the spot by each dividend on its ex-date, while
            // ANALYTICAL uses the escrowed model, so with cash dividends it
            // would also hold the model gap and give a price neither model
            // supports.
            const auto& dividends = state.underlying.discrete_dividends();
            const bool in_life = std::ranges::any_of(dividends, [&](const auto& dividend) {
                return dividend.ex_date > state.pricing_date() &&
                       dividend.ex_date <= state.maturity();
            });
            if (in_life) {
                throw UnsupportedFeatureError(std::format(
                    "control_variate_european is not supported for {} with discrete dividends: "
                    "this engine drops the spot by each dividend on its ex-date, while the "
                    "ANALYTICAL European reference uses the escrowed-dividend model, so the "
                    "correction would mix two models. Use BINOMIAL (escrowed, like ANALYTICAL) "
                    "or turn the control variate off.",
                    name(state.pricing_method)));
            }
        }
        // The European leg keeps the contract shape: a barrier control
        // variate uses European barrier legs.
        std::optional<OptionSpec> euro;
        if (const auto* vanilla = std::get_if<VanillaSpec>(&state.spec)) {
            euro = VanillaSpec({
                .option_type = vanilla->option_type(),
                .exercise_type = ExerciseType::European,
                .strike = vanilla->strike(),
                .maturity = vanilla->maturity(),
                .currency = vanilla->currency(),
                .contract_size = vanilla->contract_size(),
            });
        } else {
            const auto& barrier = std::get<BarrierSpec>(state.spec);
            euro = BarrierSpec({.option_type = barrier.option_type(),
                                .exercise_type = ExerciseType::European,
                                .strike = barrier.strike(),
                                .maturity = barrier.maturity(),
                                .barrier = barrier.barrier(),
                                .direction = barrier.direction(),
                                .action = barrier.action(),
                                .monitoring = barrier.monitoring(),
                                .rebate = barrier.rebate(),
                                .rebate_timing = barrier.rebate_timing(),
                                .currency = barrier.currency(),
                                .contract_size = barrier.contract_size(),
                                .num_observations = barrier.num_observations(),
                                .monitoring_dates = barrier.monitoring_dates()});
        }
        const OptionSpec& euro_spec = *euro;

        const double euro_numerical =
            state.underlying
                .visit([&](const auto& u) {
                    return OptionValuation(
                        u, euro_spec, state.pricing_method, without_control_variate(*state.params));
                })
                .present_value();
        const double euro_bsm =
            OptionValuation(
                state.underlying.as_underlying_data(), euro_spec, PricingMethod::Analytical)
                .present_value();

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
    init(detail::Underlying(std::move(underlying)), std::move(spec), pricing_method, params);
}

OptionValuation::OptionValuation(GbmProcess process,
                                 OptionSpec spec,
                                 PricingMethod pricing_method,
                                 std::optional<ValuationParams> params) {
    init(detail::Underlying(std::move(process)), std::move(spec), pricing_method, params);
}

OptionValuation::OptionValuation(JdProcess process,
                                 OptionSpec spec,
                                 PricingMethod pricing_method,
                                 std::optional<ValuationParams> params) {
    init(detail::Underlying(std::move(process)), std::move(spec), pricing_method, params);
}

OptionValuation::OptionValuation(SrdProcess process,
                                 OptionSpec spec,
                                 PricingMethod pricing_method,
                                 std::optional<ValuationParams> params) {
    init(detail::Underlying(std::move(process)), std::move(spec), pricing_method, params);
}

OptionValuation::OptionValuation(detail::Underlying underlying,
                                 OptionSpec spec,
                                 PricingMethod pricing_method,
                                 std::optional<ValuationParams> params,
                                 std::optional<std::pair<double, double>> pde_domain) {
    init(std::move(underlying), std::move(spec), pricing_method, params, pde_domain);
}

void OptionValuation::init(detail::Underlying underlying,
                           OptionSpec spec,
                           PricingMethod pricing_method,
                           std::optional<ValuationParams> params,
                           std::optional<std::pair<double, double>> pde_domain) {
    const auto& spec_currency = currency_of(spec);
    if (spec_currency.has_value() && *spec_currency != underlying.currency()) {
        throw UnsupportedFeatureError(
            "Cross-currency valuation is not supported. Option currency must match the "
            "underlying market currency.");
    }
    std::string currency = spec_currency.value_or(underlying.currency());

    const auto option_type = option_type_of(spec);
    if (pricing_method == PricingMethod::Analytical && !option_type.has_value()) {
        throw UnsupportedFeatureError(
            "ANALYTICAL pricing requires a CALL or PUT option type (not available for custom "
            "PayoffSpec).");
    }

    if (maturity_of(spec) <= underlying.pricing_date()) {
        throw ValidationError("Option maturity must be after pricing_date.");
    }

    if (pricing_method == PricingMethod::MonteCarlo && !underlying.is_process()) {
        throw ConfigurationError(
            "Monte Carlo pricing requires underlying to be a path process instance "
            "(GbmProcess, JdProcess, or SrdProcess)");
    }
    if (pricing_method != PricingMethod::MonteCarlo && underlying.is_process()) {
        throw ConfigurationError(
            std::format("{} pricing requires an UnderlyingData instance, "
                        "got {}.",
                        name(pricing_method),
                        underlying.type_name()));
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
        .pde_domain = pde_domain,
    });

    if (impl_->is_barrier()) {
        const auto dates = detail::barrier_monitoring_dates(impl_->state);
        if (dates.has_value() && dates->front() < impl_->state.pricing_date()) {
            throw ValidationError(
                "Barrier monitoring schedule must not start before pricing_date.");
        }
        // Monte Carlo needs every discrete monitoring date on the simulation
        // grid: inject the schedule into the process's observation dates. A
        // correlated process's grid is fixed by its context instead (checked
        // below).
        if (pricing_method == PricingMethod::MonteCarlo && dates.has_value() &&
            impl_->state.underlying.correlation_context() == nullptr) {
            impl_->state.underlying = impl_->state.underlying.with_observation_dates(*dates);
        }
    }
    if (const auto* context = impl_->state.underlying.correlation_context()) {
        // The context fixes the grid for every asset: the dates this contract
        // needs must already be on it (injecting them would misalign this
        // asset's draws with the others').
        const auto& grid = context->time_grid();
        std::string missing;
        for (const DateTime d :
             contract_event_dates(impl_->state.spec, impl_->state.pricing_date())) {
            if (!std::binary_search(grid.begin(), grid.end(), d)) {
                missing += (missing.empty() ? "" : ", ") + to_string(d);
            }
        }
        if (!missing.empty()) {
            throw ValidationError(std::format(
                "Contract dates [{}] are missing from the CorrelationContext's shared time grid. "
                "Build the grid with shared_time_grid(..., {{.specs = ...}}) listing every "
                "contract priced on the context.",
                missing));
        }
    }

    impl_->engine = build_engine(impl_->state);
}

OptionValuation::~OptionValuation() = default;
OptionValuation::OptionValuation(OptionValuation&&) noexcept = default;
OptionValuation& OptionValuation::operator=(OptionValuation&&) noexcept = default;

// ── Present value ───────────────────────────────────────────────────

double OptionValuation::present_value() const {
    const std::lock_guard lock(impl_->mutex);
    std::optional<double>& cached = impl_->present_value;
    if (cached) {
        return *cached;
    }
    const double base = impl_->engine->present_value();
    const double value =
        control_variate_requested(impl_->state.params) ? impl_->apply_control_variate(base) : base;
    cached = value;
    return value;
}

std::optional<double> OptionValuation::std_error() const {
    if (impl_->state.pricing_method == PricingMethod::MonteCarlo &&
        control_variate_requested(impl_->state.params)) {
        // The adjusted estimator's error depends on the covariance between
        // the American and European legs, which the engine alone cannot see.
        throw UnsupportedFeatureError(
            "std_error is not supported with control_variate_european for MONTE_CARLO "
            "pricing.");
    }
    // The engine memoises its estimate; the lock serialises it like PV.
    const std::lock_guard lock(impl_->mutex);
    return impl_->engine->std_error();
}

// ── Greeks ──────────────────────────────────────────────────────────

double OptionValuation::delta(GreekOptions options) const {
    return impl_->memoize({GreekKind::Delta, options.method, options.bump}, [&] {
        const auto& state = impl_->state;
        const auto method = resolve_greek_method(state, options.method, true, true);
        validate_bump(options.bump, method, "epsilon");
        if (method == GreekCalculationMethod::Pathwise) {
            return impl_->engine->delta_pathwise();
        }
        if (method == GreekCalculationMethod::LikelihoodRatio) {
            return impl_->engine->delta_lr();
        }
        if (method != GreekCalculationMethod::Numerical) {
            return impl_->engine->delta();
        }
        if (const auto triggered =
                impl_->barrier_inception_triggered_greek(GreekKind::Delta, options.bump)) {
            return *triggered;
        }
        const double s0 = state.underlying.initial_value();
        const double eps =
            detail::resolve_barrier_spot_bump(state, options.bump.value_or(s0 / 100.0));
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
        if (method == GreekCalculationMethod::Pathwise) {
            return impl_->engine->gamma_pathwise_fd(options.bump);
        }
        if (method != GreekCalculationMethod::Numerical) {
            return impl_->engine->gamma();
        }
        if (const auto triggered =
                impl_->barrier_inception_triggered_greek(GreekKind::Gamma, options.bump)) {
            return *triggered;
        }
        reject_barrier_numerical(state,
                                 method,
                                 "gamma",
                                 "with practical path counts the central-difference noise "
                                 "dominates the |Γ| signal and sign flips occur near zero",
                                 PricingMethod::MonteCarlo,
                                 std::nullopt);
        const double s0 = state.underlying.initial_value();
        const double eps =
            detail::resolve_barrier_spot_bump(state, options.bump.value_or(s0 / 100.0));
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
        if (method == GreekCalculationMethod::Pathwise) {
            return impl_->engine->vega_pathwise();
        }
        if (method == GreekCalculationMethod::LikelihoodRatio) {
            return impl_->engine->vega_lr();
        }
        if (method != GreekCalculationMethod::Numerical) {
            return impl_->engine->vega();
        }
        if (const auto triggered =
                impl_->barrier_inception_triggered_greek(GreekKind::Vega, options.bump)) {
            return *triggered;
        }
        const double vol = state.underlying.volatility();
        const double eps = options.bump.value_or(kDefaultVegaBump);
        const auto up = impl_->rebuild(state.underlying.with_volatility(vol + eps));
        if (vol <= eps) {
            // No room for the down bump: a volatility cannot be negative, and
            // zero is degenerate. Forward difference from the current value.
            return (up.present_value() - present_value()) / eps * kPerCentScale;
        }
        const auto dn = impl_->rebuild(state.underlying.with_volatility(vol - eps));
        return (up.present_value() - dn.present_value()) / (2.0 * eps) * kPerCentScale;
    });
}

double OptionValuation::theta(GreekOptions options) const {
    return impl_->memoize({GreekKind::Theta, options.method, options.bump}, [&] {
        const auto& state = impl_->state;
        // BSM analytical barrier theta: prefer the PDE identity over a
        // forward time bump (better accuracy). Auto-select only; an explicit
        // NUMERICAL request or a user bump opts back into bump-and-revalue,
        // and inception-triggered barriers short-circuit below.
        if (impl_->is_barrier() && state.pricing_method == PricingMethod::Analytical &&
            !options.method.has_value() && !options.bump.has_value() &&
            !detail::barrier_triggered_at_inception(state)) {
            return barrier_bsm_identity_theta(*this, state);
        }
        const auto method = resolve_greek_method(state, options.method, true, true);
        validate_bump(options.bump, method, "time_bump_days");
        if (method == GreekCalculationMethod::Pathwise) {
            return impl_->engine->theta_pathwise();
        }
        if (method == GreekCalculationMethod::LikelihoodRatio) {
            return impl_->engine->theta_lr();
        }
        if (method != GreekCalculationMethod::Numerical) {
            return impl_->engine->theta();
        }
        // Per-day theta of the AT_EXPIRY rebate cashflow R*df_r(T): the
        // discount factor unwinds at rate r, so theta = r*pv/365.
        const auto ko_theta_at_expiry = [&](const BarrierSpec& barrier) {
            const double maturity = state.maturity_year_fraction();
            const double df_r = state.discount_curve().df(maturity);
            const double pv = barrier.rebate() * df_r;
            const double rate = -std::log(df_r) / maturity;
            return rate * pv / 365.0;
        };
        if (const auto triggered = impl_->barrier_inception_triggered_greek(
                GreekKind::Theta, options.bump, ko_theta_at_expiry)) {
            return *triggered;
        }
        reject_barrier_numerical(state,
                                 method,
                                 "theta",
                                 "bump-and-revalue theta on discretely-monitored barriers is "
                                 "unreliable: bumping the pricing date forces re-resolution of "
                                 "the monitoring schedule (a different contract on the bumped "
                                 "side)",
                                 std::nullopt,
                                 BarrierMonitoring::Discrete);
        const double days = options.bump.value_or(impl_->is_barrier() ? kBarrierThetaTimeBumpDays
                                                                      : kDefaultThetaTimeBumpDays);
        const DateTime bumped_date = add_days(state.pricing_date(), days);
        if (bumped_date >= state.maturity()) {
            return 0.0;
        }
        if (state.underlying.correlation_context() != nullptr) {
            throw UnsupportedFeatureError(
                "Numerical theta moves the pricing date, but a process in a CorrelationContext "
                "simulates on the context's fixed grid. Use GreekCalculationMethod::Pathwise or "
                "LikelihoodRatio instead.");
        }
        const double value_now = present_value();
        const auto bumped = impl_->rebuild(state.underlying.with_pricing_date(bumped_date));
        return (bumped.present_value() - value_now) / days;
    });
}

double OptionValuation::rho(GreekOptions options) const {
    return impl_->memoize({GreekKind::Rho, options.method, options.bump}, [&] {
        const auto& state = impl_->state;
        // Rho is exempt from the barrier-binomial NUMERICAL guard: rate bumps
        // do not change the Boyle-Lau step count, so both trees share one
        // topology and the central difference is valid.
        const auto method = resolve_greek_method(state,
                                                 options.method,
                                                 /*tree_capable=*/false,
                                                 /*grid_capable=*/false,
                                                 /*allow_barrier_binomial_numerical=*/true);
        validate_bump(options.bump, method, "rate_bump");
        if (method == GreekCalculationMethod::Pathwise) {
            return impl_->engine->rho_pathwise();
        }
        if (method == GreekCalculationMethod::LikelihoodRatio) {
            return impl_->engine->rho_lr();
        }
        if (method != GreekCalculationMethod::Numerical) {
            return impl_->engine->rho();
        }
        // Central-difference rho of the AT_EXPIRY rebate cashflow R*df_r(T)
        // via a parallel zero-rate bump, scaled per 1% rate move.
        const auto ko_rho_at_expiry = [&](const BarrierSpec& barrier) {
            const double eps_r = options.bump.value_or(kDefaultRateBump);
            const double maturity = state.maturity_year_fraction();
            const auto& curve = state.discount_curve();
            const double df_up = curve.bump_parallel_zero_rate(eps_r / 2.0).df(maturity);
            const double df_dn = curve.bump_parallel_zero_rate(-eps_r / 2.0).df(maturity);
            return barrier.rebate() * (df_up - df_dn) / eps_r * kPerCentScale;
        };
        if (const auto triggered = impl_->barrier_inception_triggered_greek(
                GreekKind::Rho, options.bump, ko_rho_at_expiry)) {
            return *triggered;
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

const std::variant<UnderlyingData, GbmProcess, JdProcess, SrdProcess>& OptionValuation::underlying()
    const noexcept {
    return impl_->state.underlying.value();
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
DateTime OptionValuation::pricing_date() const {
    return impl_->state.pricing_date();
}
const DiscountCurve& OptionValuation::discount_curve() const {
    return impl_->state.discount_curve();
}
DayCountConvention OptionValuation::day_count_convention() const {
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
