#include "dp/math/normal.hpp"

#include "dp/exceptions.hpp"

#include <boost/math/distributions/normal.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/tools/rational.hpp>

#include <array>
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

// Wichura, M. J. (1988). Algorithm AS 241: The percentage points of the
// normal distribution. Applied Statistics 37(3), 477-484 — routine PPND16.
// Coefficients copied verbatim from the published Fortran
// (https://lib.stat.cmu.edu/apstat/241), in ascending powers as Boost's
// evaluate_rational expects, with each denominator's constant term 1.
namespace as241 {

// Central region numerator: A0, A1, A2, A3, A4, A5, A6, A7
constexpr std::array<double, 8> kA = {3.3871328727963666080,
                                      1.3314166789178437745e2,
                                      1.9715909503065514427e3,
                                      1.3731693765509461125e4,
                                      4.5921953931549871457e4,
                                      6.7265770927008700853e4,
                                      3.3430575583588128105e4,
                                      2.5090809287301226727e3};

// Central region denominator: 1, B1, B2, B3, B4, B5, B6, B7
constexpr std::array<double, 8> kB = {1.0,
                                      4.2313330701600911252e1,
                                      6.8718700749205790830e2,
                                      5.3941960214247511077e3,
                                      2.1213794301586595867e4,
                                      3.9307895800092710610e4,
                                      2.8729085735721942674e4,
                                      5.2264952788528545610e3};

// Intermediate tail numerator: C0, C1, C2, C3, C4, C5, C6, C7
constexpr std::array<double, 8> kC = {1.42343711074968357734,
                                      4.63033784615654529590,
                                      5.76949722146069140550,
                                      3.64784832476320460504,
                                      1.27045825245236838258,
                                      2.41780725177450611770e-1,
                                      2.27238449892691845833e-2,
                                      7.74545014278341407640e-4};

// Intermediate tail denominator: 1, D1, D2, D3, D4, D5, D6, D7
constexpr std::array<double, 8> kD = {1.0,
                                      2.05319162663775882187,
                                      1.67638483018380384940,
                                      6.89767334985100004550e-1,
                                      1.48103976427480074590e-1,
                                      1.51986665636164571966e-2,
                                      5.47593808499534494600e-4,
                                      1.05075007164441684324e-9};

// Far tail numerator: E0, E1, E2, E3, E4, E5, E6, E7
constexpr std::array<double, 8> kE = {6.65790464350110377720,
                                      5.46378491116411436990,
                                      1.78482653991729133580,
                                      2.96560571828504891230e-1,
                                      2.65321895265761230930e-2,
                                      1.24266094738807843860e-3,
                                      2.71155556874348757815e-5,
                                      2.01033439929228813265e-7};

// Far tail denominator: 1, F1, F2, F3, F4, F5, F6, F7
constexpr std::array<double, 8> kF = {1.0,
                                      5.99832206555887937690e-1,
                                      1.36929880922735805310e-1,
                                      1.48753612908506148525e-2,
                                      7.86869131145613259100e-4,
                                      1.84631831751005468180e-5,
                                      1.42151175831644588870e-7,
                                      2.04426310338993978564e-15};

}  // namespace as241

constexpr double kSplit1 = 0.425;     // SPLIT1: |p - 0.5| bound of the central region
constexpr double kSplit2 = 5.0;       // SPLIT2: r bound between the two tail fits
constexpr double kConst1 = 0.180625;  // CONST1 = SPLIT1^2
constexpr double kConst2 = 1.6;       // CONST2

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

double normal_quantile_fast(double p) noexcept {
    using boost::math::tools::evaluate_rational;
    const double q = p - 0.5;
    if (std::abs(q) <= kSplit1) {
        return q * evaluate_rational(as241::kA, as241::kB, kConst1 - q * q);
    }
    const double r = std::sqrt(-std::log(q < 0.0 ? p : 1.0 - p));
    const double value = r <= kSplit2 ? evaluate_rational(as241::kC, as241::kD, r - kConst2)
                                      : evaluate_rational(as241::kE, as241::kF, r - kSplit2);
    return q < 0.0 ? -value : value;
}

}  // namespace dp::math
