#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace contador {

/**
 * Abstração da fonte de vídeo (RTSP, arquivo local ou frame sintético).
 */
class IVideoSource {
public:
    virtual ~IVideoSource() = default;
    virtual bool open() = 0;
    virtual bool read(cv::Mat& frame) = 0;
    virtual void release() = 0;
    virtual bool isOpened() const = 0;
    virtual double fps() const = 0;

    /** Interrompe loops de reconexão / grabber (SIGTERM / Ctrl+C). */
    virtual void requestStop() {}

    /**
     * Observa flag externo de "ainda rodando" (ex.: g_running do main).
     * Quando ficar false, open()/read()/grabber saem sem travar o shutdown.
     */
    virtual void bindRunningFlag(std::atomic<bool>* running) { (void)running; }
};

struct VideoSourceOptions {
    /** Intervalo entre tentativas de reconexão (live/RTSP). */
    std::chrono::milliseconds reconnect_delay{5000};
    /** Timeout de abertura do backend (FFmpeg). 0 = default do OpenCV. */
    int open_timeout_ms{8000};
    /** Timeout de leitura de frame (FFmpeg). 0 = default do OpenCV. */
    int read_timeout_ms{5000};
    /** Falhas/vazios consecutivos antes de forçar reconnect (watchdog). */
    int max_empty_frames{3};
    int preferred_width{1280};
    int preferred_height{720};
};

/**
 * Lê câmera RTSP / HTTP ou arquivo MP4 via OpenCV VideoCapture.
 *
 * Fontes live (rtsp://, http://, rtmp://, índice de câmera):
 *   - thread de grab dedicada (reconexão NÃO bloqueia o pipeline principal)
 *   - watchdog: timeout FFmpeg + N frames vazios consecutivos
 *   - reconexão indefinida a cada reconnect_delay
 *   - hardRelease() a cada ciclo (evita leak de FFmpeg)
 *
 * Arquivos locais (MP4 etc.):
 *   - leitura síncrona; EOF → read() retorna false
 */
class CvVideoSource : public IVideoSource {
public:
    explicit CvVideoSource(std::string uri, VideoSourceOptions options = {});
    ~CvVideoSource() override;

    CvVideoSource(const CvVideoSource&) = delete;
    CvVideoSource& operator=(const CvVideoSource&) = delete;

    bool open() override;
    bool read(cv::Mat& frame) override;
    void release() override;
    bool isOpened() const override;
    double fps() const override;
    void requestStop() override;
    void bindRunningFlag(std::atomic<bool>* running) override;

    bool isLive() const { return live_; }
    bool isConnected() const { return connected_.load(std::memory_order_relaxed); }
    uint64_t reconnectCount() const { return reconnect_count_.load(); }

private:
    bool tryOpenOnce();
    void hardRelease();
    bool waitReconnectDelay();
    bool shouldStop() const;
    void startGrabber();
    void stopGrabber();
    void grabberLoop();

    static bool detectLiveUri(const std::string& uri);

    std::string uri_;
    VideoSourceOptions options_;
    bool live_{false};
    cv::VideoCapture cap_;
    double fps_{25.0};

    std::atomic<bool> stop_{false};
    std::atomic<bool>* running_flag_{nullptr};
    std::atomic<bool> connected_{false};
    std::atomic<uint64_t> reconnect_count_{0};

    // Buffer compartilhado com o pipeline (grabber → read)
    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    cv::Mat latest_frame_;
    uint64_t frame_seq_{0};
    uint64_t last_consumed_seq_{0};

    std::thread grabber_;
};

/**
 * Gera frames sintéticos com um "eixo" em movimento (útil sem câmera).
 */
class SyntheticVideoSource : public IVideoSource {
public:
    SyntheticVideoSource(int width = 1280, int height = 720, double fps = 30.0);

    bool open() override;
    bool read(cv::Mat& frame) override;
    void release() override;
    bool isOpened() const override;
    double fps() const override;
    void requestStop() override;
    void bindRunningFlag(std::atomic<bool>* running) override;

private:
    int width_;
    int height_;
    double fps_;
    bool opened_{false};
    int frame_index_{0};
    std::atomic<bool> stop_{false};
    std::atomic<bool>* running_flag_{nullptr};
};

std::unique_ptr<IVideoSource> createVideoSource(const std::string& uri,
                                                VideoSourceOptions options = {});

}  // namespace contador
