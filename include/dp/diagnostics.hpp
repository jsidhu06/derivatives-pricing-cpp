#pragma once

#include <functional>
#include <string_view>

/// Library diagnostics: non-fatal warnings and log verbosity.
///
/// The Python reference implementation used `warnings.warn` for recoverable
/// conditions (extrapolating a curve, negative rates, ...). Here those are
/// routed through a process-wide warning handler so applications can forward
/// them to their own logging / monitoring. The default handler writes to the
/// library's spdlog logger at WARN level.
namespace dp {

/// Callback invoked with a warning message. Must be thread-safe.
using WarningHandler = std::function<void(std::string_view message)>;

/// Install a process-wide warning handler. Passing an empty function restores
/// the default (spdlog) handler.
void set_warning_handler(WarningHandler handler);

/// Emit a warning through the current handler.
void warn(std::string_view message);

/// Verbosity of the library's internal logger.
enum class LogLevel { Trace, Debug, Info, Warn, Error, Off };

/// Set the minimum level emitted by the library's internal logger.
void set_log_level(LogLevel level);

}  // namespace dp
