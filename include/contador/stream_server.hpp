#pragma once

#include <opencv2/core.hpp>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace contador {

/**
 * Servidor HTTP/MJPEG minimalista (POSIX sockets).
 * Endpoints:
 *   GET /          — página HTML com <img> apontando para /stream
 *   GET /stream    — multipart/x-mixed-replace (MJPEG)
 *   GET /health    — "ok"
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

private:
    void acceptLoop();
    void handleClient(int client_fd);
    void writeStream(int client_fd);
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
};

}  // namespace contador
