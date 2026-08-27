#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/enums.hpp"
#include "dp/exceptions.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <variant>
#include <vector>

using namespace dp;
using dp::test::kMaturity;
using dp::test::kPricingDate;
using dp::test::message_contains;

namespace {

BarrierSpec::Params barrier_params() {
    return {
        .option_type = OptionType::Call,
        .exercise_type = ExerciseType::European,
        .strike = 100.0,
        .maturity = kMaturity,
        .barrier = 120.0,
        .direction = BarrierDirection::Up,
        .action = BarrierAction::Out,
    };
}

std::vector<DateTime> monthly_dates(int first, int last) {
    std::vector<DateTime> out;
    for (int i = first; i <= last; ++i) {
        out.push_back(kPricingDate + std::chrono::days{30 * i});
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// VanillaSpec
// ---------------------------------------------------------------------------

TEST_CASE("VanillaSpec smoke", "[contracts]") {
    const VanillaSpec spec({.option_type = OptionType::Call,
                            .exercise_type = ExerciseType::European,
                            .strike = 100.0,
                            .maturity = kMaturity,
                            .currency = "USD"});
    CHECK(spec.option_type() == OptionType::Call);
    CHECK(spec.exercise_type() == ExerciseType::European);
    CHECK(spec.strike() == 100.0);
    CHECK(spec.maturity() == kMaturity);
    CHECK(spec.currency() == "USD");
    CHECK(spec.contract_size() == 100.0);
}

TEST_CASE("VanillaSpec validation", "[contracts]") {
    CHECK_THROWS_MATCHES(VanillaSpec({.option_type = OptionType::Call,
                                      .exercise_type = ExerciseType::European,
                                      .strike = -1.0,
                                      .maturity = kMaturity}),
                         ValidationError,
                         message_contains("VanillaSpec.strike must be >= 0"));
    CHECK_THROWS_MATCHES(VanillaSpec({.option_type = OptionType::Call,
                                      .exercise_type = ExerciseType::European,
                                      .strike = dp::test::kNaN,
                                      .maturity = kMaturity}),
                         ValidationError,
                         message_contains("must be finite"));
    CHECK_THROWS_AS(VanillaSpec({.option_type = OptionType::Call,
                                 .exercise_type = ExerciseType::European,
                                 .strike = 100.0,
                                 .maturity = kMaturity,
                                 .currency = ""}),
                    ValidationError);
    CHECK_THROWS_AS(VanillaSpec({.option_type = OptionType::Call,
                                 .exercise_type = ExerciseType::European,
                                 .strike = 100.0,
                                 .maturity = kMaturity,
                                 .contract_size = 0.0}),
                    ValidationError);
    // Zero strike is a valid boundary value.
    CHECK_NOTHROW(VanillaSpec({.option_type = OptionType::Put,
                               .exercise_type = ExerciseType::American,
                               .strike = 0.0,
                               .maturity = kMaturity}));
}

// ---------------------------------------------------------------------------
// PayoffSpec
// ---------------------------------------------------------------------------

TEST_CASE("PayoffSpec smoke", "[contracts]") {
    const PayoffSpec spec({.exercise_type = ExerciseType::American,
                           .maturity = kMaturity,
                           .payoff_fn = [](double s) { return s; },
                           .currency = "USD"});
    CHECK(spec.exercise_type() == ExerciseType::American);
    CHECK(spec.payoff(42.0) == 42.0);
    CHECK_FALSE(spec.boundary_model().has_value());
}

TEST_CASE("PayoffSpec requires a callable and finite boundary model", "[contracts]") {
    CHECK_THROWS_MATCHES(
        PayoffSpec(
            {.exercise_type = ExerciseType::European, .maturity = kMaturity, .payoff_fn = {}}),
        ConfigurationError,
        message_contains("payoff_fn must be callable"));
    CHECK_THROWS_AS(
        PayoffSpec({.exercise_type = ExerciseType::European,
                    .maturity = kMaturity,
                    .payoff_fn = [](double s) { return s; },
                    .boundary_model = PayoffBoundaryModel{{0.0, dp::test::kNaN}, {1.0, 0.0}}}),
        ValidationError);
}

// ---------------------------------------------------------------------------
// BarrierSpec
// ---------------------------------------------------------------------------

TEST_CASE("BarrierSpec valid construction", "[contracts][barrier]") {
    const BarrierSpec spec(barrier_params());
    CHECK(spec.option_type() == OptionType::Call);
    CHECK(spec.direction() == BarrierDirection::Up);
    CHECK(spec.action() == BarrierAction::Out);
    CHECK(spec.monitoring() == BarrierMonitoring::Continuous);
    CHECK(spec.rebate() == 0.0);
    CHECK(spec.strike() == 100.0);
    CHECK(spec.barrier() == 120.0);
    CHECK_FALSE(spec.num_observations().has_value());
    CHECK_FALSE(spec.monitoring_dates().has_value());
}

TEST_CASE("BarrierSpec numeric validation", "[contracts][barrier]") {
    auto p = barrier_params();
    SECTION("negative strike") {
        p.strike = -1.0;
        CHECK_THROWS_MATCHES(
            BarrierSpec(p), ValidationError, message_contains("strike must be >= 0"));
    }
    SECTION("non-finite strike") {
        p.strike = dp::test::kInfinity;
        CHECK_THROWS_MATCHES(
            BarrierSpec(p), ValidationError, message_contains("strike must be finite"));
    }
    SECTION("barrier must be positive") {
        p.barrier = 0.0;
        CHECK_THROWS_MATCHES(
            BarrierSpec(p), ValidationError, message_contains("barrier must be > 0"));
        p.barrier = -10.0;
        CHECK_THROWS_MATCHES(
            BarrierSpec(p), ValidationError, message_contains("barrier must be > 0"));
    }
    SECTION("negative rebate") {
        p.rebate = -1.0;
        CHECK_THROWS_MATCHES(
            BarrierSpec(p), ValidationError, message_contains("rebate must be >= 0"));
    }
}

TEST_CASE("BarrierSpec knock-in rebate timing", "[contracts][barrier]") {
    auto p = barrier_params();
    p.action = BarrierAction::In;
    p.rebate = 5.0;
    p.rebate_timing = RebateTiming::AtHit;
    CHECK_THROWS_MATCHES(BarrierSpec(p), ValidationError, message_contains("Knock-in rebate"));
    p.rebate_timing = RebateTiming::AtExpiry;
    CHECK(BarrierSpec(p).rebate() == 5.0);
}

TEST_CASE("BarrierSpec continuous monitoring rejects schedules", "[contracts][barrier]") {
    auto p = barrier_params();
    SECTION("num_observations") {
        p.num_observations = 50;
        CHECK_THROWS_MATCHES(BarrierSpec(p), ValidationError, message_contains("CONTINUOUS"));
    }
    SECTION("monitoring_dates") {
        p.monitoring_dates = monthly_dates(1, 4);
        CHECK_THROWS_MATCHES(BarrierSpec(p), ValidationError, message_contains("CONTINUOUS"));
    }
}

TEST_CASE("BarrierSpec discrete monitoring schedule", "[contracts][barrier]") {
    auto p = barrier_params();
    p.monitoring = BarrierMonitoring::Discrete;
    SECTION("requires exactly one schedule source") {
        CHECK_THROWS_MATCHES(BarrierSpec(p), ValidationError, message_contains("exactly one"));
        p.num_observations = 50;
        p.monitoring_dates = monthly_dates(1, 4);
        CHECK_THROWS_MATCHES(BarrierSpec(p), ValidationError, message_contains("exactly one"));
    }
    SECTION("num_observations valid, including N=1") {
        p.num_observations = 50;
        CHECK(BarrierSpec(p).num_observations() == 50);
        p.num_observations = 1;
        CHECK(BarrierSpec(p).num_observations() == 1);
    }
    SECTION("num_observations too small") {
        p.num_observations = 0;
        CHECK_THROWS_MATCHES(BarrierSpec(p), ValidationError, message_contains("num_observations"));
    }
    SECTION("monitoring_dates valid") {
        p.monitoring_dates = monthly_dates(1, 9);
        const auto dates = BarrierSpec(p).monitoring_dates();
        REQUIRE(dates.has_value());
        CHECK(dates.value_or(std::vector<DateTime>{}).size() == 9);
    }
    SECTION("monitoring_dates beyond maturity") {
        p.monitoring_dates = std::vector{kMaturity + std::chrono::days{10}};
        CHECK_THROWS_MATCHES(BarrierSpec(p), ValidationError, message_contains("beyond maturity"));
    }
    SECTION("monitoring_dates not ascending") {
        p.monitoring_dates =
            std::vector{kPricingDate + std::chrono::days{60}, kPricingDate + std::chrono::days{30}};
        CHECK_THROWS_MATCHES(BarrierSpec(p), ValidationError, message_contains("ascending"));
    }
    SECTION("monitoring_dates duplicates") {
        const auto d = kPricingDate + std::chrono::days{30};
        p.monitoring_dates = std::vector{d, d};
        CHECK_THROWS_MATCHES(BarrierSpec(p), ValidationError, message_contains("unique"));
    }
    SECTION("monitoring_dates empty") {
        p.monitoring_dates = std::vector<DateTime>{};
        CHECK_THROWS_MATCHES(BarrierSpec(p), ValidationError, message_contains("non-empty"));
    }
}

TEST_CASE("BarrierSpec is_spot_past_barrier", "[contracts][barrier]") {
    auto p = barrier_params();
    const BarrierSpec up(p);
    CHECK(up.is_spot_past_barrier(120.0));
    CHECK(up.is_spot_past_barrier(150.0));
    CHECK_FALSE(up.is_spot_past_barrier(119.99));

    p.direction = BarrierDirection::Down;
    p.barrier = 80.0;
    const BarrierSpec down(p);
    CHECK(down.is_spot_past_barrier(80.0));
    CHECK(down.is_spot_past_barrier(50.0));
    CHECK_FALSE(down.is_spot_past_barrier(80.01));
}

// ---------------------------------------------------------------------------
// OptionSpec variant helpers
// ---------------------------------------------------------------------------

TEST_CASE("OptionSpec accessors dispatch across alternatives", "[contracts]") {
    const OptionSpec vanilla = VanillaSpec({.option_type = OptionType::Put,
                                            .exercise_type = ExerciseType::American,
                                            .strike = 95.0,
                                            .maturity = kMaturity,
                                            .currency = "EUR",
                                            .contract_size = 10.0});
    const OptionSpec payoff = PayoffSpec({.exercise_type = ExerciseType::European,
                                          .maturity = kMaturity,
                                          .payoff_fn = [](double) { return 1.0; }});
    const OptionSpec barrier = BarrierSpec(barrier_params());

    CHECK(spec_type_name(vanilla) == "VanillaSpec");
    CHECK(spec_type_name(payoff) == "PayoffSpec");
    CHECK(spec_type_name(barrier) == "BarrierSpec");

    CHECK(maturity_of(vanilla) == kMaturity);
    CHECK(exercise_type_of(vanilla) == ExerciseType::American);
    CHECK(exercise_type_of(payoff) == ExerciseType::European);
    CHECK(currency_of(vanilla) == "EUR");
    CHECK_FALSE(currency_of(payoff).has_value());
    CHECK(contract_size_of(vanilla) == 10.0);

    CHECK(strike_of(vanilla) == 95.0);
    CHECK_FALSE(strike_of(payoff).has_value());
    CHECK(strike_of(barrier) == 100.0);

    CHECK(option_type_of(vanilla) == OptionType::Put);
    CHECK_FALSE(option_type_of(payoff).has_value());
    CHECK(option_type_of(barrier) == OptionType::Call);
}
