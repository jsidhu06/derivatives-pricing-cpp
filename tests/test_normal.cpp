#include "dp/exceptions.hpp"
#include "dp/math/normal.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using dp::math::normal_cdf;
using dp::math::normal_pdf;
using dp::math::normal_quantile;

TEST_CASE("normal_cdf reference values", "[math][normal]") {
    CHECK_THAT(normal_cdf(0.0), WithinAbs(0.5, 1e-16));
    CHECK_THAT(normal_cdf(1.0), WithinRel(0.841344746068543, 1e-14));
    CHECK_THAT(normal_cdf(1.959963984540054), WithinRel(0.975, 1e-14));
    CHECK_THAT(normal_cdf(-1.0), WithinRel(0.158655253931457, 1e-14));
    CHECK_THAT(normal_cdf(3.0), WithinRel(0.998650101968370, 1e-14));
}

TEST_CASE("normal_cdf is accurate deep in the tails", "[math][normal]") {
    // Values that would be destroyed by 0.5 * (1 + erf(x / sqrt2)).
    CHECK_THAT(normal_cdf(-10.0), WithinRel(7.619853024160527e-24, 1e-12));
    CHECK_THAT(normal_cdf(-30.0), WithinRel(4.906713927148187e-198, 1e-12));
    CHECK(normal_cdf(-40.0) >= 0.0);
    CHECK(normal_cdf(40.0) == 1.0);
}

TEST_CASE("normal_cdf symmetry", "[math][normal]") {
    const double x = GENERATE(0.1, 0.5, 1.3, 2.7, 4.2);
    CHECK_THAT(normal_cdf(x) + normal_cdf(-x), WithinAbs(1.0, 1e-15));
}

TEST_CASE("normal_cdf handles non-finite input without throwing", "[math][normal]") {
    CHECK(normal_cdf(std::numeric_limits<double>::infinity()) == 1.0);
    CHECK(normal_cdf(-std::numeric_limits<double>::infinity()) == 0.0);
    CHECK(std::isnan(normal_cdf(std::numeric_limits<double>::quiet_NaN())));
}

TEST_CASE("normal_pdf reference values", "[math][normal]") {
    CHECK_THAT(normal_pdf(0.0), WithinRel(0.398942280401433, 1e-14));
    CHECK_THAT(normal_pdf(1.0), WithinRel(0.241970724519143, 1e-14));
    CHECK_THAT(normal_pdf(-2.0), WithinRel(normal_pdf(2.0), 1e-16));
}

TEST_CASE("normal_quantile inverts normal_cdf", "[math][normal]") {
    SECTION("central region and lower tail") {
        const double x = GENERATE(-6.0, -2.5, -0.3, 0.0, 0.7, 1.96, 3.0);
        CHECK_THAT(normal_quantile(normal_cdf(x)), WithinAbs(x, 1e-12));
    }
    SECTION("far upper tail is limited by the conditioning of p near 1") {
        // Phi(5) = 1 - 2.9e-7: the ulp of p is ~1e-16, so x is recoverable
        // only to ~1e-16 / 2.9e-7 / phi(5) ~ 2e-10 absolute.
        CHECK_THAT(normal_quantile(normal_cdf(5.0)), WithinAbs(5.0, 1e-9));
    }
}

TEST_CASE("normal_quantile reference values and boundaries", "[math][normal]") {
    CHECK_THAT(normal_quantile(0.5), WithinAbs(0.0, 1e-16));
    CHECK_THAT(normal_quantile(0.975), WithinRel(1.959963984540054, 1e-13));
    CHECK(normal_quantile(0.0) == -std::numeric_limits<double>::infinity());
    CHECK(normal_quantile(1.0) == std::numeric_limits<double>::infinity());
    CHECK_THROWS_AS(normal_quantile(-0.1), dp::ValidationError);
    CHECK_THROWS_AS(normal_quantile(1.1), dp::ValidationError);
    CHECK_THROWS_AS(normal_quantile(std::numeric_limits<double>::quiet_NaN()), dp::ValidationError);
}
