#include "dp/diagnostics.hpp"

#include "detail/log.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>

#include <memory>
#include <mutex>
#include <string>

namespace dp {

namespace detail {

std::shared_ptr<spdlog::logger> logger() {
    // Function-local static: thread-safe initialisation, created on first use.
    static const std::shared_ptr<spdlog::logger> instance = [] {
        if (auto existing = spdlog::get("dp")) {
            return existing;
        }
        auto created = spdlog::stderr_color_mt("dp");
        created->set_level(spdlog::level::warn);
        return created;
    }();
    return instance;
}

}  // namespace detail

namespace {

std::mutex& handler_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::shared_ptr<const WarningHandler>& handler_slot() {
    static std::shared_ptr<const WarningHandler> slot;
    return slot;
}

void default_warning_handler(std::string_view message) {
    detail::logger()->warn("{}", message);
}

}  // namespace

void set_warning_handler(WarningHandler handler) {
    auto next = handler ? std::make_shared<const WarningHandler>(std::move(handler)) : nullptr;
    const std::lock_guard lock(handler_mutex());
    handler_slot() = std::move(next);
}

void warn(std::string_view message) {
    std::shared_ptr<const WarningHandler> current;
    {
        const std::lock_guard lock(handler_mutex());
        current = handler_slot();
    }
    // Invoke outside the lock so a handler may itself call set_warning_handler.
    if (current) {
        (*current)(message);
    } else {
        default_warning_handler(message);
    }
}

void set_log_level(LogLevel level) {
    auto to_spdlog = [](LogLevel value) {
        switch (value) {
            case LogLevel::Trace:
                return spdlog::level::trace;
            case LogLevel::Debug:
                return spdlog::level::debug;
            case LogLevel::Info:
                return spdlog::level::info;
            case LogLevel::Warn:
                return spdlog::level::warn;
            case LogLevel::Error:
                return spdlog::level::err;
            case LogLevel::Off:
                return spdlog::level::off;
        }
        return spdlog::level::warn;
    };
    detail::logger()->set_level(to_spdlog(level));
}

}  // namespace dp
