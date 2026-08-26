#include "dp/diagnostics.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

TEST_CASE("custom warning handler receives warnings", "[diagnostics]") {
    const dp::test::WarningCapture capture;
    dp::warn("hello");
    dp::warn("world");
    REQUIRE(capture.messages().size() == 2);
    CHECK(capture.messages()[0] == "hello");
    CHECK(capture.messages()[1] == "world");
}

TEST_CASE("resetting the handler restores the default without throwing", "[diagnostics]") {
    {
        const dp::test::WarningCapture capture;
        dp::warn("captured");
        CHECK(capture.any_contains("captured"));
    }
    // Default handler logs via spdlog; must not throw.
    dp::set_log_level(dp::LogLevel::Off);
    CHECK_NOTHROW(dp::warn("not captured"));
    dp::set_log_level(dp::LogLevel::Warn);
}

TEST_CASE("a handler may replace itself while being invoked", "[diagnostics]") {
    std::string seen;
    dp::set_warning_handler([&seen](std::string_view m) {
        seen = std::string(m);
        dp::set_warning_handler({});  // must not deadlock
    });
    dp::warn("once");
    CHECK(seen == "once");
    dp::set_warning_handler({});
}
