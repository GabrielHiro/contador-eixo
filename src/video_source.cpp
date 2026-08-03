#include "contador/video_source.hpp"
#include "contador/logger.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cctype>
#include <thread>

namespace contador {
namespace {

bool startsWithIgnoreCase(const std::string& s, const char* prefix) {
    const size_t n = std::char_traits<char>::length(prefix);
    if (s.size() < n) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (std::tolower(static_cast<unsigned char>(s[i])) !=
            std::tolower(static_cast<unsigned char>(prefix[i]))) {
            return false;
        }
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// CvVideoSource
// ---------------------------------------------------------------------------

bool CvVideoSource::detectLiveUri(const std::string& uri) {
    if (uri.size() == 1 && std::isdigit(static_cast<unsigned char>(uri[0]))) {
        return true;
    }
    return startsWithIgnoreCase(uri, "rtsp://") || startsWithIgnoreCase(uri, "rtsps://") ||
           startsWithIgnoreCase(uri, "http://") || startsWithIgnoreCase(uri, "https://") ||
           startsWithIgnoreCase(uri, "rtmp://") || startsWithIgnoreCase(uri, "rtp://");
}

CvVideoSource::CvVideoSource(std::string uri, VideoSourceOptions options)
    : uri_(std::move(uri)), options_(options), live_(detectLiveUri(uri_)) {}

void CvVideoSource::requestStop() {
    stop_.store(true, std::memory_order_relaxed);
}

void CvVideoSource::bindRunningFlag(std::atomic<bool>* running) {
    running_flag_ = running;
}

bool CvVideoSource::shouldStop() const {
    if (stop_.load(std::memory_order_relaxed)) {
        return true;
    }
    if (running_flag_ != nullptr && !running_flag_->load(std::memory_order_relaxed)) {
        return true;
    }
    return false;
}

void CvVideoSource::hardRelease() {
    if (cap_.isOpened()) {
        cap_.release();
    }
    // Garante destruição completa do handle FFmpeg/GStreamer
    cap_ = cv::VideoCapture();
}

bool CvVideoSource::waitReconnectDelay() {
    const auto total = options_.reconnect_delay;
    constexpr auto slice = std::chrono::milliseconds(100);
    auto waited = std::chrono::milliseconds(0);
    while (waited < total) {
        if (shouldStop()) {
            return false;
        }
        std::this_thread::sleep_for(slice);
        waited += slice;
    }
    return !shouldStop();
}

bool CvVideoSource::tryOpenOnce() {
    hardRelease();

    bool ok = false;
    if (uri_.size() == 1 && std::isdigit(static_cast<unsigned char>(uri_[0]))) {
        ok = cap_.open(uri_[0] - '0');
    } else if (live_) {
        auto applyTimeouts = [this]() {
            if (options_.open_timeout_ms > 0) {
                cap_.set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, options_.open_timeout_ms);
            }
            if (options_.read_timeout_ms > 0) {
                cap_.set(cv::CAP_PROP_READ_TIMEOUT_MSEC, options_.read_timeout_ms);
            }
            cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
        };

        // Somente backends de stream — evita CAP_IMAGES spit errors em RTSP
        applyTimeouts();
        ok = cap_.open(uri_, cv::CAP_FFMPEG);
        if (!ok || !cap_.isOpened()) {
            hardRelease();
            applyTimeouts();
            ok = cap_.open(uri_, cv::CAP_GSTREAMER);
        }
    } else {
        ok = cap_.open(uri_);
    }

    if (!ok || !cap_.isOpened()) {
        return false;
    }

    if (live_) {
        if (options_.open_timeout_ms > 0) {
            cap_.set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, options_.open_timeout_ms);
        }
        if (options_.read_timeout_ms > 0) {
            cap_.set(cv::CAP_PROP_READ_TIMEOUT_MSEC, options_.read_timeout_ms);
        }
        cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }

    cap_.set(cv::CAP_PROP_FRAME_WIDTH, options_.preferred_width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, options_.preferred_height);

    fps_ = cap_.get(cv::CAP_PROP_FPS);
    if (fps_ <= 1.0) {
        fps_ = 25.0;
    }

    const int w = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    const int h = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    LogInfo("VideoSource") << "Aberto: " << uri_ << " " << w << "x" << h << " @ " << fps_
                           << " FPS"
                           << (live_ ? " [live/reconnect]" : " [file]");
    return true;
}

bool CvVideoSource::open() {
    stop_.store(false, std::memory_order_relaxed);

    if (!live_) {
        if (!tryOpenOnce()) {
            LogError("VideoSource") << "Falha ao abrir arquivo: " << uri_;
            return false;
        }
        return true;
    }

    // Live: tenta até sucesso ou requestStop()
    uint64_t attempt = 0;
    while (!shouldStop()) {
        ++attempt;
        if (tryOpenOnce()) {
            return true;
        }
        LogWarn("VideoSource") << "Falha ao abrir (tentativa " << attempt
                               << ") — nova tentativa em "
                               << options_.reconnect_delay.count() << " ms: " << uri_;
        if (!waitReconnectDelay()) {
            break;
        }
    }
    return false;
}

bool CvVideoSource::read(cv::Mat& frame) {
    frame.release();

    if (!live_) {
        if (!cap_.isOpened()) {
            return false;
        }
        if (!cap_.read(frame) || frame.empty()) {
            LogInfo("VideoSource") << "EOF / fim do arquivo: " << uri_;
            return false;
        }
        return true;
    }

    // Live: nunca desiste até requestStop()
    while (!shouldStop()) {
        if (!cap_.isOpened()) {
            LogWarn("VideoSource") << "Capture fechado — reconectando em "
                                   << options_.reconnect_delay.count() << " ms";
            if (!waitReconnectDelay()) {
                break;
            }
            if (tryOpenOnce()) {
                reconnect_count_.fetch_add(1, std::memory_order_relaxed);
                LogInfo("VideoSource")
                    << "Reconectado (#" << reconnect_count_.load() << "): " << uri_;
            } else {
                LogWarn("VideoSource") << "Reconexão falhou: " << uri_;
            }
            continue;
        }

        const bool ok = cap_.read(frame);
        if (ok && !frame.empty()) {
            return true;
        }

        // Timeout FFmpeg, socket drop, frame vazio → watchdog
        frame.release();
        LogWarn("VideoSource") << "Sem frame (timeout/perda) — liberando e reconectando: "
                               << uri_;
        hardRelease();

        if (!waitReconnectDelay()) {
            break;
        }

        if (tryOpenOnce()) {
            reconnect_count_.fetch_add(1, std::memory_order_relaxed);
            LogInfo("VideoSource")
                << "Reconectado (#" << reconnect_count_.load() << "): " << uri_;
        } else {
            LogWarn("VideoSource") << "Reconexão falhou: " << uri_;
        }
    }

    frame.release();
    return false;
}

void CvVideoSource::release() {
    stop_.store(true, std::memory_order_relaxed);
    hardRelease();
}

bool CvVideoSource::isOpened() const {
    return cap_.isOpened();
}

double CvVideoSource::fps() const {
    return fps_;
}

// ---------------------------------------------------------------------------
// SyntheticVideoSource
// ---------------------------------------------------------------------------

SyntheticVideoSource::SyntheticVideoSource(int width, int height, double fps)
    : width_(width), height_(height), fps_(fps) {}

void SyntheticVideoSource::requestStop() {
    stop_.store(true, std::memory_order_relaxed);
}

void SyntheticVideoSource::bindRunningFlag(std::atomic<bool>* running) {
    running_flag_ = running;
}

bool SyntheticVideoSource::open() {
    stop_.store(false, std::memory_order_relaxed);
    opened_ = true;
    frame_index_ = 0;
    LogInfo("VideoSource") << "Modo sintético " << width_ << "x" << height_ << " @ " << fps_
                           << " FPS";
    return true;
}

bool SyntheticVideoSource::read(cv::Mat& frame) {
    if (!opened_ || stop_.load(std::memory_order_relaxed) ||
        (running_flag_ != nullptr && !running_flag_->load(std::memory_order_relaxed))) {
        return false;
    }

    frame = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(40, 40, 40));

    const int road_top = height_ * 2 / 5;
    const int road_bottom = height_ * 4 / 5;
    cv::rectangle(frame, cv::Point(0, road_top), cv::Point(width_, road_bottom),
                  cv::Scalar(70, 70, 70), cv::FILLED);

    for (int x = 0; x < width_; x += 60) {
        cv::line(frame, cv::Point(x, (road_top + road_bottom) / 2),
                 cv::Point(x + 30, (road_top + road_bottom) / 2),
                 cv::Scalar(200, 200, 200), 2);
    }

    const int cycle = width_ + 400;
    const int x = (frame_index_ * 6) % cycle - 200;
    const int y = (road_top + road_bottom) / 2;
    const int axle_gap = 120;

    cv::rectangle(frame, cv::Point(x - 40, y - 50), cv::Point(x + axle_gap + 40, y + 20),
                  cv::Scalar(30, 90, 180), cv::FILLED);

    cv::circle(frame, cv::Point(x, y + 25), 28, cv::Scalar(20, 20, 20), cv::FILLED);
    cv::circle(frame, cv::Point(x, y + 25), 28, cv::Scalar(0, 220, 255), 2);
    cv::circle(frame, cv::Point(x + axle_gap, y + 25), 28, cv::Scalar(20, 20, 20),
               cv::FILLED);
    cv::circle(frame, cv::Point(x + axle_gap, y + 25), 28, cv::Scalar(0, 220, 255), 2);

    cv::putText(frame, "SYNTHETIC FEED - Contador de Eixos", cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(220, 220, 220), 2);

    ++frame_index_;
    return true;
}

void SyntheticVideoSource::release() {
    stop_.store(true, std::memory_order_relaxed);
    opened_ = false;
}

bool SyntheticVideoSource::isOpened() const {
    return opened_;
}

double SyntheticVideoSource::fps() const {
    return fps_;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<IVideoSource> createVideoSource(const std::string& uri,
                                                VideoSourceOptions options) {
    if (uri.empty() || uri == "synthetic" || uri == "mock") {
        return std::make_unique<SyntheticVideoSource>();
    }
    return std::make_unique<CvVideoSource>(uri, options);
}

}  // namespace contador
