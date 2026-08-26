#pragma once

#include <spdlog/spdlog.h>

#include <memory>

/// Internal access to the library logger. Not part of the public API.
namespace dp::detail {

/// The shared `"dp"` logger, created on first use.
std::shared_ptr<spdlog::logger> logger();

}  // namespace dp::detail
