#pragma once

#include "dp/dates.hpp"
#include "dp/discount_curve.hpp"
#include "dp/enums.hpp"

#include <vector>

/// Discrete cash dividends.
namespace dp {

/// A single cash dividend going ex on `ex_date`.
struct Dividend {
    DateTime ex_date;
    double amount;

    friend bool operator==(const Dividend&, const Dividend&) = default;
};

/// Ordered (by ex-date) list of cash dividends.
using DividendSchedule = std::vector<Dividend>;

/// Validate amounts are finite and return a copy sorted by ex-date.
/// Throws `ValidationError` on non-finite amounts.
[[nodiscard]] DividendSchedule normalize_dividends(DividendSchedule dividends);

/// Present value, as of `start_date`, of dividends with ex-dates in the range.
///
/// Discounting uses `discount_curve` anchored at `curve_date`:
/// `PV = sum amount * D(t_ex) / D(t_start)`.
///
/// When `include_start` is true the range is closed on the left
/// (`start <= ex <= end`) — the convention for computing the clean spot at
/// inception. When false it is open on the left (`start < ex <= end`) — the
/// convention for escrowed-dividend add-backs at intermediate tree nodes,
/// where a dividend going ex at exactly `start` has already left the price.
///
/// `start_date` defaults to `curve_date` when it is not a valid time point
/// (pass `std::nullopt`).
[[nodiscard]] double pv_discrete_dividends(
    const DividendSchedule& dividends,
    DateTime curve_date,
    DateTime end_date,
    const DiscountCurve& discount_curve,
    std::optional<DateTime> start_date = std::nullopt,
    DayCountConvention day_count_convention = DayCountConvention::Act365F,
    bool include_start = true);

}  // namespace dp
