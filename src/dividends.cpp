#include "dp/dividends.hpp"

#include "dp/exceptions.hpp"

#include "detail/validate.hpp"

#include <algorithm>

namespace dp {

DividendSchedule normalize_dividends(DividendSchedule dividends) {
    for (const auto& dividend : dividends) {
        detail::require_finite(dividend.amount, "dividend amount");
    }
    std::stable_sort(dividends.begin(), dividends.end(), [](const auto& a, const auto& b) {
        return a.ex_date < b.ex_date;
    });
    return dividends;
}

double pv_discrete_dividends(const DividendSchedule& dividends,
                             DateTime curve_date,
                             DateTime end_date,
                             const DiscountCurve& discount_curve,
                             std::optional<DateTime> start_date,
                             DayCountConvention day_count_convention,
                             bool include_start,
                             const DiscountCurve* dividend_curve) {
    if (dividends.empty()) {
        return 0.0;
    }

    const DateTime start = start_date.value_or(curve_date);
    const double t_start = year_fraction(curve_date, start, day_count_convention);
    const double r_df_start = discount_curve.df(t_start);
    const double q_df_start = dividend_curve == nullptr ? 1.0 : dividend_curve->df(t_start);

    double pv = 0.0;
    for (const auto& [ex_date, amount] : dividends) {
        const bool in_range = include_start ? (start <= ex_date && ex_date <= end_date)
                                            : (start < ex_date && ex_date <= end_date);
        if (!in_range) {
            continue;
        }
        const double t_ex = year_fraction(curve_date, ex_date, day_count_convention);
        // Discount factors from start_date to the ex-date.
        const double df_r = discount_curve.df(t_ex) / r_df_start;
        const double df_q = dividend_curve == nullptr ? 1.0 : dividend_curve->df(t_ex) / q_df_start;
        pv += amount * df_r / df_q;
    }
    return pv;
}

}  // namespace dp
