#include "dp/math/normal.hpp"

#include "dp/exceptions.hpp"

#include <boost/math/distributions/normal.hpp>
#include <boost/math/policies/policy.hpp>

#include <cmath>
#include <format>
#include <limits>

namespace dp::math {

namespace {

// Boost.Math defaults to promoting double -> long double internally and to
// throwing on domain/overflow errors. Pricing code wants plain double speed
// and IEEE semantics (NaN in -> NaN out, quantile(0) -> -inf), so use an
// explicit policy.
using NormalPolicy = boost::math::policies::policy<
    boost::math::policies::promote_double<false>,
    boost::math::policies::domain_error<boost::math::policies::ignore_error>,
    boost::math::policies::overflow_error<boost::math::policies::ignore_error>>;

using StandardNormal = boost::math::normal_distribution<double, NormalPolicy>;

const StandardNormal& standard_normal() noexcept {
    static const StandardNormal dist(0.0, 1.0);
    return dist;
}

}  // namespace

double normal_cdf(double x) noexcept {
    return boost::math::cdf(standard_normal(), x);
}

double normal_pdf(double x) noexcept {
    return boost::math::pdf(standard_normal(), x);
}

double normal_quantile(double p) {
    if (std::isnan(p) || p < 0.0 || p > 1.0) {
        throw ValidationError(std::format("normal_quantile: p must be in [0, 1], got {}", p));
    }
    if (p == 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    if (p == 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    return boost::math::quantile(standard_normal(), p);
}

}  // namespace dp::math
