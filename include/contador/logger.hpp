#pragma once

#include <mutex>
#include <sstream>
#include <string>
#include <string_view>

namespace contador {

enum class LogLevel {
    Info = 0,
    Warn = 1,
    Error = 2,
};

/**
 * Logger mínimo thread-safe para edge/SSH.
 * Formato: 2026-08-03 14:05:01.123 [INFO ] [Component] mensagem
 */
class Logger {
public:
    static Logger& instance();

    void setMinLevel(LogLevel level);
    LogLevel minLevel() const;

    void log(LogLevel level, std::string_view component, std::string_view message);

private:
    Logger() = default;
    static const char* levelName(LogLevel level);
    static std::string timestamp();

    mutable std::mutex mutex_;
    LogLevel min_level_{LogLevel::Info};
};

namespace detail {

class LogLine {
public:
    LogLine(LogLevel level, std::string_view component)
        : level_(level), component_(component) {}

    ~LogLine() {
        Logger::instance().log(level_, component_, stream_.str());
    }

    template <typename T>
    LogLine& operator<<(const T& value) {
        stream_ << value;
        return *this;
    }

private:
    LogLevel level_;
    std::string component_;
    std::ostringstream stream_;
};

}  // namespace detail

inline detail::LogLine LogInfo(std::string_view component) {
    return {LogLevel::Info, component};
}
inline detail::LogLine LogWarn(std::string_view component) {
    return {LogLevel::Warn, component};
}
inline detail::LogLine LogError(std::string_view component) {
    return {LogLevel::Error, component};
}

}  // namespace contador
