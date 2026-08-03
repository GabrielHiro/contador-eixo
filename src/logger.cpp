#include "contador/logger.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>

namespace contador {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::setMinLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    min_level_ = level;
}

LogLevel Logger::minLevel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return min_level_;
}

const char* Logger::levelName(LogLevel level) {
    switch (level) {
        case LogLevel::Info:
            return "INFO ";
        case LogLevel::Warn:
            return "WARN ";
        case LogLevel::Error:
            return "ERROR";
    }
    return "?????";
}

std::string Logger::timestamp() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()) %
                    1000;
    const std::time_t t = clock::to_time_t(now);
    std::tm tm_buf{};
    localtime_r(&t, &tm_buf);

    char buf[64];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
                  tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                  tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
                  static_cast<int>(ms.count()));
    return buf;
}

void Logger::log(LogLevel level, std::string_view component, std::string_view message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (static_cast<int>(level) < static_cast<int>(min_level_)) {
        return;
    }

    std::ostream& out = (level == LogLevel::Error) ? std::cerr : std::cout;
    out << timestamp() << " [" << levelName(level) << "] [" << component << "] " << message
        << '\n';
    out.flush();
}

}  // namespace contador
