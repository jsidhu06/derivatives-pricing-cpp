#include "dp/discount_curve.hpp"
#include "dp/exceptions.hpp"

#include "test_helpers.hpp"

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using dp::DiscountCurve;
using dp::test::message_contains;

namespace {

Eigen::ArrayXd arr(std::initializer_list<double> values) {
    Eigen::ArrayXd out(static_cast<Eigen::Index>(values.size()));
    Eigen::Index i = 0;
    for (const double v : values) {
        out[i++] = v;
    }
    return out;
}

/// Non-flat curve used by the interpolation tests.
DiscountCurve sample_curve() {
    return DiscountCurve(arr({0.0, 0.5, 1.0, 2.0}), arr({1.0, 0.975, 0.95, 0.90}));
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction / validation
// ---------------------------------------------------------------------------

TEST_CASE("flat curve basics", "[discount_curve]") {
    const auto curve = DiscountCurve::flat(0.05, 1.0);
    const auto rate = curve.flat_rate();
    REQUIRE(rate.has_value());
    CHECK_THAT(rate.value_or(0.0), WithinRel(0.05, 1e-12));
    CHECK_THAT(curve.df(0.0), WithinAbs(1.0, 1e-15));
    CHECK_THAT(curve.df(1.0), WithinRel(std::exp(-0.05), 1e-12));
}

TEST_CASE("flat curve defaults to a 100y domain", "[discount_curve]") {
    const dp::test::WarningCapture capture;
    const auto curve = DiscountCurve::flat(0.05);
    CHECK_THAT(curve.df(1.0), WithinRel(std::exp(-0.05), 1e-12));
    CHECK_THAT(curve.df(50.0), WithinRel(std::exp(-0.05 * 50.0), 1e-12));
    CHECK(capture.empty());
}

TEST_CASE("flat curve with zero rate", "[discount_curve]") {
    const auto curve = DiscountCurve::flat(0.0, 1.0);
    CHECK_THAT(curve.df(0.5), WithinAbs(1.0, 1e-15));
}

TEST_CASE("flat curve rejects non-positive end_time", "[discount_curve]") {
    CHECK_THROWS_MATCHES(DiscountCurve::flat(0.05, -1.0),
                         dp::ValidationError,
                         message_contains("end_time must be positive"));
    CHECK_THROWS_MATCHES(DiscountCurve::flat(0.05, 0.0),
                         dp::ValidationError,
                         message_contains("end_time must be positive"));
}

TEST_CASE("constructor validation", "[discount_curve]") {
    SECTION("non-increasing times") {
        CHECK_THROWS_MATCHES(DiscountCurve(arr({0.0, 0.5, 0.5}), arr({1.0, 0.98, 0.96})),
                             dp::ValidationError,
                             message_contains("strictly increasing"));
    }
    SECTION("negative discount factor") {
        CHECK_THROWS_MATCHES(DiscountCurve(arr({0.0, 1.0}), arr({1.0, -0.5})),
                             dp::ValidationError,
                             message_contains("discount factors must be positive"));
    }
    SECTION("mismatched lengths") {
        CHECK_THROWS_MATCHES(DiscountCurve(arr({0.0, 1.0}), arr({1.0})),
                             dp::ValidationError,
                             message_contains("same length"));
    }
    SECTION("empty arrays") {
        CHECK_THROWS_MATCHES(
            DiscountCurve(arr({}), arr({})), dp::ValidationError, message_contains("at least 2"));
    }
    SECTION("single point") {
        CHECK_THROWS_MATCHES(DiscountCurve(arr({0.5}), arr({0.975})),
                             dp::ValidationError,
                             message_contains("at least 2"));
    }
    SECTION("two points accepted") {
        const DiscountCurve curve(arr({0.5, 1.0}), arr({0.975, 0.95}));
        CHECK(curve.times().size() == 2);
    }
    SECTION("non-finite inputs rejected") {
        CHECK_THROWS_AS(DiscountCurve(arr({0.0, 1.0}), arr({1.0, NAN})), dp::ValidationError);
        CHECK_THROWS_AS(DiscountCurve(arr({0.0, INFINITY}), arr({1.0, 0.9})), dp::ValidationError);
    }
}

TEST_CASE("discount factors > 1 are accepted with a warning", "[discount_curve]") {
    const dp::test::WarningCapture capture;
    const DiscountCurve curve(arr({0.0, 1.0}), arr({1.0, 1.5}));
    CHECK(capture.any_contains("Discount factors > 1 detected"));
    CHECK_THAT(curve.df(1.0), WithinRel(1.5, 1e-12));
}

TEST_CASE("flat_rate detects flat and non-flat curves", "[discount_curve]") {
    const auto t = arr({0.0, 1.0});
    const double rate = 0.03;
    const DiscountCurve flat(t, (-rate * t).exp());
    const auto detected = flat.flat_rate();
    REQUIRE(detected.has_value());
    CHECK_THAT(detected.value_or(0.0), WithinRel(rate, 1e-10));

    const DiscountCurve bumpy(arr({0.0, 0.5, 1.0}), arr({1.0, 0.98, 0.90}));
    CHECK_FALSE(bumpy.flat_rate().has_value());
}

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

TEST_CASE("from_forwards builds piecewise-constant forward curve", "[discount_curve]") {
    const auto times = arr({0.0, 0.5, 1.0, 2.0});
    const auto fwds = arr({0.03, 0.05, 0.04});
    const auto curve = DiscountCurve::from_forwards(times, fwds);

    CHECK_THAT(curve.df(0.5), WithinRel(std::exp(-0.03 * 0.5), 1e-12));
    CHECK_THAT(curve.df(1.0), WithinRel(std::exp(-0.03 * 0.5 - 0.05 * 0.5), 1e-12));
    CHECK_THAT(curve.df(2.0), WithinRel(std::exp(-0.03 * 0.5 - 0.05 * 0.5 - 0.04), 1e-12));

    const auto recovered = curve.step_forward_rates(times);
    REQUIRE(recovered.size() == 3);
    for (Eigen::Index i = 0; i < 3; ++i) {
        CHECK_THAT(recovered[i], WithinRel(fwds[i], 1e-10));
    }
}

TEST_CASE("from_forwards validation", "[discount_curve]") {
    CHECK_THROWS_MATCHES(DiscountCurve::from_forwards(arr({0.0}), arr({})),
                         dp::ValidationError,
                         message_contains("at least [0, T]"));
    CHECK_THROWS_MATCHES(DiscountCurve::from_forwards(arr({0.0, 1.0}), arr({0.1, 0.1})),
                         dp::ValidationError,
                         message_contains("len(times) - 1"));
    CHECK_THROWS_MATCHES(DiscountCurve::from_forwards(arr({0.5, 1.0}), arr({0.1})),
                         dp::ValidationError,
                         message_contains("start at 0.0"));
}

TEST_CASE("from_zero_rates", "[discount_curve]") {
    const auto curve =
        DiscountCurve::from_zero_rates(arr({0.5, 1.0, 2.0}), arr({0.02, 0.03, 0.04}));
    CHECK_THAT(curve.df(0.5), WithinRel(std::exp(-0.02 * 0.5), 1e-12));
    CHECK_THAT(curve.df(1.0), WithinRel(std::exp(-0.03), 1e-12));
    CHECK_THAT(curve.df(2.0), WithinRel(std::exp(-0.08), 1e-12));
    CHECK_THROWS_MATCHES(DiscountCurve::from_zero_rates(arr({0.5, 1.0}), arr({0.02})),
                         dp::ValidationError,
                         message_contains("same length"));
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

TEST_CASE("df at exact node", "[discount_curve]") {
    CHECK_THAT(sample_curve().df(0.5), WithinRel(0.975, 1e-12));
}

TEST_CASE("df interpolates log-linearly", "[discount_curve]") {
    const double expected = std::exp(0.5 * (std::log(1.0) + std::log(0.975)));
    CHECK_THAT(sample_curve().df(0.25), WithinRel(expected, 1e-10));
}

TEST_CASE("vectorised df matches scalar df", "[discount_curve]") {
    const auto curve = sample_curve();
    const auto t = arr({0.0, 0.25, 0.5, 1.0, 1.7});
    const auto result = curve.df(t);
    REQUIRE(result.size() == t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        CHECK_THAT(result[i], WithinRel(curve.df(t[i]), 1e-15));
    }
}

TEST_CASE("extrapolation is flat in log-DF and warns", "[discount_curve]") {
    const auto curve = sample_curve();
    const double t_far = GENERATE(3.0, 10.0, 20.0, 100.0);

    const dp::test::WarningCapture capture;
    const double df_far = curve.df(t_far);
    CHECK_THAT(df_far, WithinRel(curve.df(2.0), 1e-12));
    CHECK(capture.any_contains("Extrapolating discount curve"));
}

TEST_CASE("vectorised extrapolation warns once per call", "[discount_curve]") {
    const auto curve = sample_curve();
    const dp::test::WarningCapture capture;
    const auto result = curve.df(arr({3.0, 4.0, 5.0}));
    CHECK(result.size() == 3);
    CHECK(capture.messages().size() == 1);
}

TEST_CASE("no warning inside the knot range", "[discount_curve]") {
    const dp::test::WarningCapture capture;
    const auto curve = sample_curve();
    (void)curve.df(0.0);
    (void)curve.df(1.3);
    (void)curve.df(2.0);
    (void)curve.df(arr({0.0, 0.1, 2.0}));
    CHECK(capture.empty());
}

TEST_CASE("continuity around a node", "[discount_curve]") {
    const auto curve = sample_curve();
    const double eps = GENERATE(1e-12, 1e-10, 1e-8, 1e-6);
    const double node = curve.df(0.5);
    CHECK_THAT(curve.df(0.5 - eps), WithinRel(node, 1e-5));
    CHECK_THAT(curve.df(0.5 + eps), WithinRel(node, 1e-5));
}

TEST_CASE("df(0) is one even when the curve starts after zero", "[discount_curve]") {
    const DiscountCurve curve(arr({0.25, 0.5, 1.0}), arr({0.99, 0.975, 0.95}));
    const dp::test::WarningCapture capture;
    CHECK_THAT(curve.df(0.0), WithinAbs(1.0, 1e-15));
    const auto vec = curve.df(arr({0.0, 0.25, 0.75}));
    CHECK_THAT(vec[0], WithinAbs(1.0, 1e-15));
    CHECK_THAT(vec[1], WithinRel(0.99, 1e-12));
    CHECK(capture.empty());  // t=0 is not "outside"
}

// ---------------------------------------------------------------------------
// Forward rates
// ---------------------------------------------------------------------------

TEST_CASE("flat curve forward rate equals flat rate", "[discount_curve]") {
    const auto curve = DiscountCurve::flat(0.05, 2.0);
    CHECK_THAT(curve.forward_rate(0.25, 0.75), WithinRel(0.05, 1e-10));
}

TEST_CASE("forward_rate rejects t1 <= t0", "[discount_curve]") {
    const auto curve = DiscountCurve::flat(0.05, 1.0);
    CHECK_THROWS_MATCHES(
        curve.forward_rate(0.5, 0.5), dp::ValidationError, message_contains("Need t1 > t0"));
}

TEST_CASE("forward rate is consistent with discount factors", "[discount_curve]") {
    const auto curve = sample_curve();
    const double t0 = 0.25;
    const double t1 = 1.5;
    const double fwd = curve.forward_rate(t0, t1);
    const double reconstructed = curve.df(t0) * std::exp(-fwd * (t1 - t0));
    CHECK_THAT(reconstructed, WithinRel(curve.df(t1), 1e-10));
}

TEST_CASE("step_forward_rates on a flat curve", "[discount_curve]") {
    const auto curve = DiscountCurve::flat(0.05, 2.0);
    const auto fwds = curve.step_forward_rates(arr({0.0, 0.5, 1.0, 1.5, 2.0}));
    REQUIRE(fwds.size() == 4);
    for (const double fwd : fwds) {
        CHECK_THAT(fwd, WithinRel(0.05, 1e-10));
    }
}

TEST_CASE("step_forward_rates rejects non-increasing grid", "[discount_curve]") {
    const auto curve = DiscountCurve::flat(0.05, 1.0);
    CHECK_THROWS_MATCHES(curve.step_forward_rates(arr({0.0, 0.5, 0.5})),
                         dp::ValidationError,
                         message_contains("strictly increasing"));
}

TEST_CASE("step_forward_rates on a non-flat curve", "[discount_curve]") {
    const auto times = arr({0.0, 0.5, 1.0, 2.0});
    const auto dfs = arr({1.0,
                          std::exp(-0.03 * 0.5),
                          std::exp(-0.03 * 0.5 - 0.05 * 0.5),
                          std::exp(-0.03 * 0.5 - 0.05 * 0.5 - 0.04 * 1.0)});
    const DiscountCurve curve(times, dfs);
    const auto fwds = curve.step_forward_rates(times);
    REQUIRE(fwds.size() == 3);
    CHECK_THAT(fwds[0], WithinRel(0.03, 1e-10));
    CHECK_THAT(fwds[1], WithinRel(0.05, 1e-10));
    CHECK_THAT(fwds[2], WithinRel(0.04, 1e-10));
}

// ---------------------------------------------------------------------------
// Bumping
// ---------------------------------------------------------------------------

TEST_CASE("bump_parallel_zero_rate shifts every zero rate", "[discount_curve]") {
    const auto curve = sample_curve();
    const double bump = 0.01;
    const auto bumped = curve.bump_parallel_zero_rate(bump);
    for (const double t : {0.5, 1.0, 1.7, 2.0}) {
        CHECK_THAT(bumped.df(t), WithinRel(curve.df(t) * std::exp(-bump * t), 1e-12));
    }
    // Original untouched.
    CHECK_THAT(curve.df(1.0), WithinRel(0.95, 1e-12));
}
