#pragma once

#include "dp/enums.hpp"

#include <chrono>
#include <string>
#include <vector>

/// Date-time types and day-count arithmetic.
///
/// `DateTime` is a timezone-free (UTC-clock) time point with microsecond
/// resolution. The type itself guarantees "no timezone", so no runtime
/// validation is needed.
namespace dp {

/// Microsecond-resolution, timezone-free time point.
using DateTime = std::chrono::sys_time<std::chrono::microseconds>;

/// Calendar-day duration (24h) used for day-count arithmetic.
using Days = std::chrono::days;

/// Construct a `DateTime` from calendar fields. Throws `ValidationError` if the
/// fields do not form a valid calendar date or time of day.
[[nodiscard]] DateTime make_datetime(int year,
                                     unsigned month,
                                     unsigned day,
                                     unsigned hour = 0,
                                     unsigned minute = 0,
                                     unsigned second = 0,
                                     unsigned microsecond = 0);

/// Construct a `DateTime` at midnight on the given `year_month_day`.
[[nodiscard]] DateTime make_datetime(std::chrono::year_month_day ymd);

/// Calendar date of a `DateTime` (time of day dropped).
[[nodiscard]] std::chrono::year_month_day to_ymd(DateTime value) noexcept;

/// Elapsed time between two instants in (fractional) calendar days.
[[nodiscard]] double days_between(DateTime start, DateTime end) noexcept;

/// Year fraction between two instants under the given day-count convention.
///
/// ACT conventions use the exact elapsed time (including time of day);
/// 30/360 US uses calendar fields only.
[[nodiscard]] double year_fraction(DateTime start,
                                   DateTime end,
                                   DayCountConvention convention = DayCountConvention::Act365F);

/// `num_points` instants evenly spaced on `[start, end]`, inclusive of both
/// endpoints. Interior points are rounded to the nearest microsecond; the
/// last element is exactly `end`. Equivalent to
/// `pandas.date_range(start, end, periods=num_points)`.
///
/// Requires `num_points >= 2` and `end > start`.
[[nodiscard]] std::vector<DateTime> date_range(DateTime start,
                                               DateTime end,
                                               std::size_t num_points);

/// ISO-8601 representation: `YYYY-MM-DD` when the time of day is midnight,
/// otherwise `YYYY-MM-DDTHH:MM:SS[.ffffff]`.
[[nodiscard]] std::string to_string(DateTime value);

}  // namespace dp
