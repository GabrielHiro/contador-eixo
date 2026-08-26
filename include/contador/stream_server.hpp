#pragma once

#include "contador/config_store.hpp"
#include "contador/pipeline_controller.hpp"

#include <opencv2/core.hpp>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace contador {

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

/**
 * Servidor HTTP/MJPEG minimalista (POSIX sockets).
 * Endpoints:
 *   GET  /            — painel: estado do pipeline, contagem atual, link p/ stream
 *   GET  /config      — formulário de configuração (fonte, modelo, linha, thresholds...)
 *   POST /config      — aplica hot-reload via PipelineController e persiste em disco
 *   POST /api/count   — processa um vídeo de um veículo e retorna a contagem em JSON
 *   GET  /api/status  — status do pipeline em JSON
 *   GET  /stream      — multipart/x-mixed-replace (MJPEG)
 *   GET  /health      — "ok"
 */
class StreamServer {
public:
    StreamServer();
    ~StreamServer();

    StreamServer(const StreamServer&) = delete;
    StreamServer& operator=(const StreamServer&) = delete;

    bool start(uint16_t port = 8080, const std::string& bind_address = "0.0.0.0");
    void stop();
    bool isRunning() const { return running_.load(); }

    /** Publica o frame mais recente (thread-safe). Codifica como JPEG. */
    void publish(const cv::Mat& frame, int jpeg_quality = 80);

    uint16_t port() const { return port_; }

    /**
     * Liga o servidor a um PipelineController (telas de status/config) e ao
     * caminho do arquivo de configuração persistida (config/settings.json).
     */
    void attachController(PipelineController* controller, std::string config_path);

private:
    void acceptLoop();
    void handleClient(int client_fd);
    void writeStream(int client_fd);
    void handleConfigPost(int client_fd, const std::string& body);
    void handleCountPost(int client_fd, const std::string& body);

    std::string renderDashboard() const;
    std::string renderConfigForm(const PipelineConfig& cfg, const std::string& message) const;
    std::string renderStatusJson() const;

    static bool readRequest(int fd, HttpRequest& out);
    static bool sendAll(int fd, const void* data, size_t len);
    static std::string buildHttpResponse(int status,
                                         const std::string& content_type,
                                         const std::string& body,
                                         bool connection_close = true);

    uint16_t port_{8080};
    std::string bind_address_{"0.0.0.0"};
    int server_fd_{-1};
    std::atomic<bool> running_{false};
    std::thread accept_thread_;

    mutable std::mutex frame_mutex_;
    std::vector<uchar> jpeg_buffer_;
    std::atomic<uint64_t> frame_seq_{0};

    PipelineController* controller_{nullptr};
    std::string config_path_;
    std::mutex job_mutex_;
};

}  // namespace contador
