#include "dp/dates.hpp"
#include "dp/enums.hpp"
#include "dp/exceptions.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using dp::DayCountConvention;
using dp::make_datetime;
using dp::year_fraction;

// ---------------------------------------------------------------------------
// Construction / formatting
// ---------------------------------------------------------------------------

TEST_CASE("make_datetime builds midnight instants", "[dates]") {
    const auto d = make_datetime(2025, 1, 1);
    CHECK(d == dp::DateTime{std::chrono::sys_days{std::chrono::January / 1 / 2025}});
    CHECK(dp::to_string(d) == "2025-01-01");
}

TEST_CASE("make_datetime accepts a time of day", "[dates]") {
    const auto d = make_datetime(2025, 6, 15, 13, 45, 30, 250'000);
    CHECK(dp::to_string(d) == "2025-06-15T13:45:30.250000");
    CHECK(dp::to_string(make_datetime(2025, 6, 15, 13, 45, 30)) == "2025-06-15T13:45:30");
}

TEST_CASE("make_datetime rejects invalid calendar fields", "[dates]") {
    CHECK_THROWS_AS(make_datetime(2025, 2, 30), dp::ValidationError);
    CHECK_THROWS_AS(make_datetime(2025, 13, 1), dp::ValidationError);
    CHECK_THROWS_AS(make_datetime(2025, 1, 1, 24), dp::ValidationError);
    CHECK_THROWS_AS(make_datetime(2025, 1, 1, 0, 60), dp::ValidationError);
    CHECK_THROWS_AS(make_datetime(2025, 1, 1, 0, 0, 0, 1'000'000), dp::ValidationError);
    CHECK_NOTHROW(make_datetime(2024, 2, 29));  // leap day
}

TEST_CASE("to_ymd drops the time of day", "[dates]") {
    const auto ymd = dp::to_ymd(make_datetime(2025, 3, 31, 23, 59, 59));
    CHECK(static_cast<int>(ymd.year()) == 2025);
    CHECK(static_cast<unsigned>(ymd.month()) == 3);
    CHECK(static_cast<unsigned>(ymd.day()) == 31);
}

TEST_CASE("DateTime is totally ordered and usable in ordered containers", "[dates]") {
    const auto a = make_datetime(2025, 1, 1);
    const auto b = make_datetime(2025, 1, 1, 0, 0, 0, 1);
    CHECK(a < b);
    CHECK(a != b);
    CHECK(a == make_datetime(2025, 1, 1));
}

// ---------------------------------------------------------------------------
// ACT/365F
// ---------------------------------------------------------------------------

TEST_CASE("ACT/365F year fractions", "[dates][daycount]") {
    SECTION("one year") {
        CHECK_THAT(
            year_fraction(
                make_datetime(2025, 1, 1), make_datetime(2026, 1, 1), DayCountConvention::Act365F),
            WithinRel(1.0, 1e-12));
    }
    SECTION("one day") {
        CHECK_THAT(year_fraction(make_datetime(2025, 6, 15),
                                 make_datetime(2025, 6, 16),
                                 DayCountConvention::Act365F),
                   WithinRel(1.0 / 365.0, 1e-12));
    }
    SECTION("same date returns zero") {
        const auto d = make_datetime(2025, 3, 15);
        CHECK(year_fraction(d, d, DayCountConvention::Act365F) == 0.0);
    }
    SECTION("leap year has 366 actual days") {
        CHECK_THAT(
            year_fraction(
                make_datetime(2024, 1, 1), make_datetime(2025, 1, 1), DayCountConvention::Act365F),
            WithinRel(366.0 / 365.0, 1e-12));
    }
    SECTION("ACT/365F is the default convention") {
        const auto s = make_datetime(2025, 1, 1);
        const auto e = make_datetime(2025, 7, 1);
        CHECK(year_fraction(s, e) == year_fraction(s, e, DayCountConvention::Act365F));
    }
    SECTION("half year") {
        CHECK_THAT(
            year_fraction(
                make_datetime(2025, 1, 1), make_datetime(2025, 7, 2), DayCountConvention::Act365F),
            WithinRel(182.0 / 365.0, 1e-12));
    }
    SECTION("intraday resolution is honoured") {
        // 12 hours = half a day.
        CHECK_THAT(year_fraction(make_datetime(2025, 1, 1),
                                 make_datetime(2025, 1, 1, 12),
                                 DayCountConvention::Act365F),
                   WithinRel(0.5 / 365.0, 1e-12));
    }
    SECTION("negative when end precedes start") {
        CHECK(year_fraction(make_datetime(2025, 1, 2), make_datetime(2025, 1, 1)) < 0.0);
    }
}

// ---------------------------------------------------------------------------
// ACT/360
// ---------------------------------------------------------------------------

TEST_CASE("ACT/360 year fractions", "[dates][daycount]") {
    SECTION("365 actual days") {
        CHECK_THAT(
            year_fraction(
                make_datetime(2025, 1, 1), make_datetime(2026, 1, 1), DayCountConvention::Act360),
            WithinRel(365.0 / 360.0, 1e-12));
    }
    SECTION("one day") {
        CHECK_THAT(
            year_fraction(
                make_datetime(2025, 6, 15), make_datetime(2025, 6, 16), DayCountConvention::Act360),
            WithinRel(1.0 / 360.0, 1e-12));
    }
    SECTION("same date returns zero") {
        const auto d = make_datetime(2025, 3, 15);
        CHECK(year_fraction(d, d, DayCountConvention::Act360) == 0.0);
    }
    SECTION("90 days is exactly a quarter") {
        CHECK_THAT(
            year_fraction(
                make_datetime(2025, 1, 1), make_datetime(2025, 4, 1), DayCountConvention::Act360),
            WithinRel(0.25, 1e-12));
    }
}

// ---------------------------------------------------------------------------
// ACT/365.25
// ---------------------------------------------------------------------------

TEST_CASE("ACT/365.25 year fractions", "[dates][daycount]") {
    SECTION("one year") {
        CHECK_THAT(year_fraction(make_datetime(2025, 1, 1),
                                 make_datetime(2026, 1, 1),
                                 DayCountConvention::Act365_25),
                   WithinRel(365.0 / 365.25, 1e-12));
    }
    SECTION("four years including one leap year is exactly 4.0") {
        CHECK_THAT(year_fraction(make_datetime(2024, 1, 1),
                                 make_datetime(2028, 1, 1),
                                 DayCountConvention::Act365_25),
                   WithinRel(4.0, 1e-10));
    }
    SECTION("same date returns zero") {
        const auto d = make_datetime(2025, 3, 15);
        CHECK(year_fraction(d, d, DayCountConvention::Act365_25) == 0.0);
    }
}

// ---------------------------------------------------------------------------
// 30/360 US
// ---------------------------------------------------------------------------

TEST_CASE("30/360 US year fractions", "[dates][daycount]") {
    const auto thirty360 = [](dp::DateTime s, dp::DateTime e) {
        return year_fraction(s, e, DayCountConvention::Thirty360Us);
    };
    SECTION("Jan 31 -> Feb 28") {
        CHECK_THAT(thirty360(make_datetime(2025, 1, 31), make_datetime(2025, 2, 28)),
                   WithinRel(28.0 / 360.0, 1e-12));
    }
    SECTION("one full year") {
        CHECK_THAT(thirty360(make_datetime(2025, 1, 1), make_datetime(2026, 1, 1)),
                   WithinRel(1.0, 1e-12));
    }
    SECTION("same month") {
        CHECK_THAT(thirty360(make_datetime(2025, 6, 15), make_datetime(2025, 6, 25)),
                   WithinRel(10.0 / 360.0, 1e-12));
    }
    SECTION("same date returns zero") {
        const auto d = make_datetime(2025, 3, 15);
        CHECK(thirty360(d, d) == 0.0);
    }
    SECTION("d2=31 stays 31 when d1 is not 30 or 31") {
        CHECK_THAT(thirty360(make_datetime(2025, 1, 1), make_datetime(2025, 1, 31)),
                   WithinRel(30.0 / 360.0, 1e-12));
    }
    SECTION("31st to 31st: both become 30") {
        CHECK_THAT(thirty360(make_datetime(2025, 1, 31), make_datetime(2025, 3, 31)),
                   WithinRel(60.0 / 360.0, 1e-12));
    }
    SECTION("quarter") {
        CHECK_THAT(thirty360(make_datetime(2025, 1, 1), make_datetime(2025, 4, 1)),
                   WithinRel(0.25, 1e-12));
    }
    SECTION("time of day is ignored") {
        CHECK(thirty360(make_datetime(2025, 1, 1, 9), make_datetime(2025, 4, 1, 17)) ==
              thirty360(make_datetime(2025, 1, 1), make_datetime(2025, 4, 1)));
    }
}

// ---------------------------------------------------------------------------
// Cross-convention ordering
// ---------------------------------------------------------------------------

TEST_CASE("ACT/360 > ACT/365F > ACT/365.25 for a positive period", "[dates][daycount]") {
    const auto s = make_datetime(2025, 1, 1);
    const auto e = make_datetime(2025, 7, 1);
    const double f360 = year_fraction(s, e, DayCountConvention::Act360);
    const double f365 = year_fraction(s, e, DayCountConvention::Act365F);
    const double f365_25 = year_fraction(s, e, DayCountConvention::Act365_25);
    CHECK(f360 > f365);
    CHECK(f365 > f365_25);
}

// ---------------------------------------------------------------------------
// date_range
// ---------------------------------------------------------------------------

TEST_CASE("date_range is inclusive, evenly spaced and ends exactly at end", "[dates]") {
    const auto start = make_datetime(2025, 1, 1);
    const auto end = make_datetime(2025, 1, 11);
    const auto grid = dp::date_range(start, end, 11);

    REQUIRE(grid.size() == 11);
    CHECK(grid.front() == start);
    CHECK(grid.back() == end);
    for (std::size_t i = 0; i < grid.size(); ++i) {
        CHECK(grid[i] == start + std::chrono::days{static_cast<int>(i)});
    }
}

TEST_CASE("date_range rounds interior points to the nearest microsecond", "[dates]") {
    const auto start = make_datetime(2025, 1, 1);
    const auto end = make_datetime(2025, 1, 2);       // 86 400 000 000 us
    const auto grid = dp::date_range(start, end, 7);  // step = 14 400 000 000 us exactly

    REQUIRE(grid.size() == 7);
    CHECK(grid[3] == start + std::chrono::hours{12});

    // Non-integer step: 1 day / 7 intervals. All points strictly increasing,
    // last is exact.
    const auto odd = dp::date_range(start, end, 8);
    for (std::size_t i = 1; i < odd.size(); ++i) {
        CHECK(odd[i] > odd[i - 1]);
    }
    CHECK(odd.back() == end);
}

TEST_CASE("date_range validates its inputs", "[dates]") {
    const auto start = make_datetime(2025, 1, 1);
    const auto end = make_datetime(2025, 1, 2);
    CHECK_THROWS_AS(dp::date_range(start, end, 1), dp::ValidationError);
    CHECK_THROWS_AS(dp::date_range(end, start, 3), dp::ValidationError);
    CHECK_THROWS_AS(dp::date_range(start, start, 3), dp::ValidationError);
}

TEST_CASE("days_between honours sub-day resolution", "[dates]") {
    CHECK_THAT(dp::days_between(make_datetime(2025, 1, 1), make_datetime(2025, 1, 2, 6)),
               WithinAbs(1.25, 1e-12));
}
