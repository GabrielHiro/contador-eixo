#include "contador/video_source.hpp"
#include "contador/utils.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <utility>

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
    : uri_(std::move(uri)), options_(std::move(options)), live_(detectLiveUri(uri_)) {}

CvVideoSource::~CvVideoSource() {
    release();
}

void CvVideoSource::requestStop() {
    stop_.store(true, std::memory_order_relaxed);
    frame_cv_.notify_all();
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
    // Destrói o handle FFmpeg/GStreamer por completo (evita leak em reconnect)
    cap_ = cv::VideoCapture();
    connected_.store(false, std::memory_order_relaxed);
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
                           << (live_ ? " [live/grabber]" : " [file]");
    connected_.store(true, std::memory_order_relaxed);
    return true;
}

void CvVideoSource::startGrabber() {
    if (grabber_.joinable()) {
        return;
    }
    grabber_ = std::thread(&CvVideoSource::grabberLoop, this);
}

void CvVideoSource::stopGrabber() {
    requestStop();
    if (grabber_.joinable()) {
        grabber_.join();
    }
}

void CvVideoSource::grabberLoop() {
    int empty_streak = 0;
    const int max_empty = std::max(1, options_.max_empty_frames);

    LogInfo("VideoSource") << "Grabber iniciado (reconnect="
                           << options_.reconnect_delay.count() << " ms, max_empty="
                           << max_empty << ")";

    while (!shouldStop()) {
        if (!cap_.isOpened()) {
            connected_.store(false, std::memory_order_relaxed);
            if (!tryOpenOnce()) {
                LogWarn("VideoSource") << "Falha ao abrir — retry em "
                                       << options_.reconnect_delay.count() << " ms: " << uri_;
                if (!waitReconnectDelay()) {
                    break;
                }
                continue;
            }
            if (reconnect_count_.load() > 0) {
                LogInfo("VideoSource")
                    << "Reconectado (#" << reconnect_count_.load() << "): " << uri_;
            }
            empty_streak = 0;
        }

        cv::Mat frame;
        const bool ok = cap_.read(frame);
        if (!ok || frame.empty()) {
            ++empty_streak;
            if (empty_streak >= max_empty) {
                LogWarn("VideoSource")
                    << "Watchdog: " << empty_streak
                    << " frames vazios/timeout — liberando e reconectando: " << uri_;
                hardRelease();
                reconnect_count_.fetch_add(1, std::memory_order_relaxed);
                empty_streak = 0;
                if (!waitReconnectDelay()) {
                    break;
                }
            }
            continue;
        }

        empty_streak = 0;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_frame_ = std::move(frame);
            ++frame_seq_;
        }
        frame_cv_.notify_all();
    }

    hardRelease();
    LogInfo("VideoSource") << "Grabber encerrado";
}

bool CvVideoSource::open() {
    stop_.store(false, std::memory_order_relaxed);
    last_consumed_seq_ = 0;
    frame_seq_ = 0;

    if (!live_) {
        if (!tryOpenOnce()) {
            LogError("VideoSource") << "Falha ao abrir arquivo: " << uri_;
            return false;
        }
        return true;
    }

    // Live: grabber faz open + reconnect em background.
    // open() só retorna true após o primeiro frame (ou shutdown).
    startGrabber();

    while (!shouldStop()) {
        {
            std::unique_lock<std::mutex> lock(frame_mutex_);
            if (frame_cv_.wait_for(lock, std::chrono::milliseconds(200), [&] {
                    return frame_seq_ > 0 || shouldStop();
                })) {
                if (frame_seq_ > 0) {
                    LogInfo("VideoSource") << "Primeiro frame recebido — pipeline pode seguir";
                    return true;
                }
            }
        }
    }
    stopGrabber();
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

    // Live: espera próximo frame do grabber. Durante reconnect o main NÃO cai —
    // apenas aguarda (interruptível por shouldStop).
    while (!shouldStop()) {
        std::unique_lock<std::mutex> lock(frame_mutex_);
        const bool woke = frame_cv_.wait_for(lock, std::chrono::milliseconds(200), [&] {
            return frame_seq_ > last_consumed_seq_ || shouldStop();
        });
        if (!woke) {
            continue;
        }
        if (shouldStop()) {
            break;
        }
        if (frame_seq_ > last_consumed_seq_) {
            latest_frame_.copyTo(frame);
            last_consumed_seq_ = frame_seq_;
            return !frame.empty();
        }
    }

    frame.release();
    return false;
}

void CvVideoSource::release() {
    stopGrabber();
    hardRelease();
    std::lock_guard<std::mutex> lock(frame_mutex_);
    latest_frame_.release();
}

bool CvVideoSource::isOpened() const {
    if (live_) {
        return connected_.load(std::memory_order_relaxed) || grabber_.joinable();
    }
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
