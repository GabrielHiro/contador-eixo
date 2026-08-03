#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>

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

    /** Interrompe loops de reconexão (ex.: SIGTERM / Ctrl+C). */
    virtual void requestStop() {}

    /**
     * Observa um flag externo de "ainda rodando" (ex.: g_running do main).
     * Quando o flag ficar false, read()/open() saem do loop de reconexão.
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
    int preferred_width{1280};
    int preferred_height{720};
};

/**
 * Lê câmera RTSP / HTTP ou arquivo MP4 via OpenCV VideoCapture.
 *
 * Fontes live (rtsp://, http://, rtmp://, índice de câmera):
 *   - watchdog de falha/timeout de frame
 *   - reconexão indefinida a cada reconnect_delay
 *   - release() explícito a cada ciclo (evita leak de FFmpeg)
 *
 * Arquivos locais (MP4 etc.):
 *   - sem reconexão; EOF → read() retorna false
 */
class CvVideoSource : public IVideoSource {
public:
    explicit CvVideoSource(std::string uri, VideoSourceOptions options = {});

    bool open() override;
    bool read(cv::Mat& frame) override;
    void release() override;
    bool isOpened() const override;
    double fps() const override;
    void requestStop() override;
    void bindRunningFlag(std::atomic<bool>* running) override;

    bool isLive() const { return live_; }
    uint64_t reconnectCount() const { return reconnect_count_.load(); }

private:
    bool tryOpenOnce();
    void hardRelease();
    bool waitReconnectDelay();
    bool shouldStop() const;

    static bool detectLiveUri(const std::string& uri);

    std::string uri_;
    VideoSourceOptions options_;
    bool live_{false};
    cv::VideoCapture cap_;
    double fps_{25.0};
    std::atomic<bool> stop_{false};
    std::atomic<bool>* running_flag_{nullptr};
    std::atomic<uint64_t> reconnect_count_{0};
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
