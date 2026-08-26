#include "dp/dates.hpp"

#include "dp/exceptions.hpp"

#include <chrono>
#include <cmath>
#include <format>
#include <string>
#include <vector>

namespace dp {

namespace {

using namespace std::chrono;

constexpr unsigned kMicrosPerSecond = 1'000'000;

/// 30/360 (US) day-count fraction between two calendar dates.
double thirty_360_us(year_month_day start, year_month_day end) noexcept {
    const int y1 = static_cast<int>(start.year());
    const int y2 = static_cast<int>(end.year());
    const int m1 = static_cast<int>(static_cast<unsigned>(start.month()));
    const int m2 = static_cast<int>(static_cast<unsigned>(end.month()));
    int d1 = static_cast<int>(static_cast<unsigned>(start.day()));
    int d2 = static_cast<int>(static_cast<unsigned>(end.day()));

    if (d1 == 31) {
        d1 = 30;
    }
    if (d2 == 31 && d1 == 30) {
        d2 = 30;
    }
    return static_cast<double>(360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360.0;
}

}  // namespace

DateTime make_datetime(std::chrono::year_month_day ymd) {
    if (!ymd.ok()) {
        throw ValidationError(std::format("invalid calendar date {}-{}-{}",
                                          static_cast<int>(ymd.year()),
                                          static_cast<unsigned>(ymd.month()),
                                          static_cast<unsigned>(ymd.day())));
    }
    return DateTime{sys_days{ymd}};
}

DateTime make_datetime(int year,
                       unsigned month,
                       unsigned day,
                       unsigned hour,
                       unsigned minute,
                       unsigned second,
                       unsigned microsecond) {
    const year_month_day ymd{
        std::chrono::year{year}, std::chrono::month{month}, std::chrono::day{day}};
    if (hour >= 24 || minute >= 60 || second >= 60 || microsecond >= kMicrosPerSecond) {
        throw ValidationError(std::format(
            "invalid time of day {:02}:{:02}:{:02}.{:06}", hour, minute, second, microsecond));
    }
    return make_datetime(ymd) + hours{hour} + minutes{minute} + seconds{second} +
           microseconds{microsecond};
}

std::chrono::year_month_day to_ymd(DateTime value) noexcept {
    return year_month_day{floor<days>(value)};
}

double days_between(DateTime start, DateTime end) noexcept {
    return duration<double, days::period>(end - start).count();
}

double year_fraction(DateTime start, DateTime end, DayCountConvention convention) {
    switch (convention) {
        case DayCountConvention::Thirty360Us:
            return thirty_360_us(to_ymd(start), to_ymd(end));
        case DayCountConvention::Act360:
            return days_between(start, end) / 360.0;
        case DayCountConvention::Act365_25:
            return days_between(start, end) / 365.25;
        case DayCountConvention::Act365F:
            return days_between(start, end) / 365.0;
    }
    throw ValidationError("Unsupported day_count_convention");
}

std::vector<DateTime> date_range(DateTime start, DateTime end, std::size_t num_points) {
    if (num_points < 2) {
        throw ValidationError(
            std::format("date_range requires at least 2 points, got {}", num_points));
    }
    if (end <= start) {
        throw ValidationError("date_range requires end > start");
    }

    const auto total = (end - start).count();  // microseconds
    const auto intervals = static_cast<std::int64_t>(num_points - 1);
    if (total < intervals) {
        throw ValidationError(
            "date_range: interval too short to hold the requested number of distinct points");
    }

    std::vector<DateTime> out;
    out.reserve(num_points);
    out.push_back(start);
    for (std::int64_t i = 1; i < intervals; ++i) {
        // Round each offset independently (rather than accumulating a step)
        // so rounding error does not drift across the grid.
        const auto offset = std::llround(static_cast<double>(total) * static_cast<double>(i) /
                                         static_cast<double>(intervals));
        out.push_back(start + microseconds{offset});
    }
    out.push_back(end);
    return out;
}

std::string to_string(DateTime value) {
    const auto day_start = floor<days>(value);
    const year_month_day ymd{day_start};
    const hh_mm_ss<microseconds> tod{value - day_start};

    std::string out = std::format("{:04}-{:02}-{:02}",
                                  static_cast<int>(ymd.year()),
                                  static_cast<unsigned>(ymd.month()),
                                  static_cast<unsigned>(ymd.day()));
    if (tod.to_duration() == microseconds::zero()) {
        return out;
    }
    out += std::format(
        "T{:02}:{:02}:{:02}", tod.hours().count(), tod.minutes().count(), tod.seconds().count());
    if (tod.subseconds() != microseconds::zero()) {
        out += std::format(".{:06}", tod.subseconds().count());
    }
    return out;
}

}  // namespace dp
