#include "contador/stream_server.hpp"
#include "contador/logger.hpp"

#include <opencv2/imgcodecs.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <sstream>

namespace contador {
namespace {

constexpr const char* kHtmlPage =
    "<!DOCTYPE html>\n"
    "<html lang=\"pt-BR\">\n"
    "<head>\n"
    "  <meta charset=\"utf-8\"/>\n"
    "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n"
    "  <title>Contador de Eixos</title>\n"
    "  <style>\n"
    "    * { box-sizing: border-box; margin: 0; padding: 0; }\n"
    "    body { background: #111; color: #eee; font-family: sans-serif;\n"
    "           min-height: 100vh; display: flex; flex-direction: column;\n"
    "           align-items: center; justify-content: center; gap: 1rem; }\n"
    "    h1 { font-weight: 500; letter-spacing: 0.04em; }\n"
    "    img { max-width: 96vw; max-height: 80vh; border: 1px solid #333; }\n"
    "    .meta { color: #888; font-size: 0.9rem; }\n"
    "  </style>\n"
    "</head>\n"
    "<body>\n"
    "  <h1>Contador de Eixos</h1>\n"
    "  <img src=\"/stream\" alt=\"MJPEG stream\"/>\n"
    "  <p class=\"meta\">stream &middot; /stream</p>\n"
    "</body>\n"
    "</html>\n";

}  // namespace

StreamServer::StreamServer() = default;

StreamServer::~StreamServer() {
    stop();
}

bool StreamServer::start(uint16_t port, const std::string& bind_address) {
    if (running_.load()) {
        return true;
    }

    port_ = port;
    bind_address_ = bind_address;

    server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        LogError("StreamServer") << "socket() falhou";
        return false;
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    if (inet_pton(AF_INET, bind_address_.c_str(), &addr.sin_addr) <= 0) {
        addr.sin_addr.s_addr = INADDR_ANY;
    }

    if (bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        LogError("StreamServer") << "bind() falhou na porta " << port_;
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (listen(server_fd_, 8) < 0) {
        LogError("StreamServer") << "listen() falhou";
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    running_.store(true);
    accept_thread_ = std::thread(&StreamServer::acceptLoop, this);

    LogInfo("StreamServer") << "Escutando em http://" << bind_address_ << ":" << port_
                            << "/stream";
    return true;
}

void StreamServer::stop() {
    if (!running_.exchange(false)) {
        return;
    }

    if (server_fd_ >= 0) {
        ::shutdown(server_fd_, SHUT_RDWR);
        ::close(server_fd_);
        server_fd_ = -1;
    }

    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
}

void StreamServer::publish(const cv::Mat& frame, int jpeg_quality) {
    if (frame.empty()) {
        return;
    }

    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
    std::vector<uchar> buffer;
    if (!cv::imencode(".jpg", frame, buffer, params)) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        jpeg_buffer_.swap(buffer);
    }
    frame_seq_.fetch_add(1);
}

void StreamServer::acceptLoop() {
    while (running_.load()) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        const int client_fd =
            ::accept(server_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (!running_.load()) {
                break;
            }
            continue;
        }

        // Cada cliente em thread destacada (carga baixa no edge: poucos viewers).
        std::thread(&StreamServer::handleClient, this, client_fd).detach();
    }
}

void StreamServer::handleClient(int client_fd) {
    char req[2048];
    const ssize_t n = ::recv(client_fd, req, sizeof(req) - 1, 0);
    if (n <= 0) {
        ::close(client_fd);
        return;
    }
    req[n] = '\0';

    std::string request(req);
    std::string path = "/";
    {
        const auto method_end = request.find(' ');
        if (method_end != std::string::npos) {
            const auto path_end = request.find(' ', method_end + 1);
            if (path_end != std::string::npos) {
                path = request.substr(method_end + 1, path_end - method_end - 1);
            }
        }
        const auto q = path.find('?');
        if (q != std::string::npos) {
            path = path.substr(0, q);
        }
    }

    if (path == "/health") {
        const std::string resp = buildHttpResponse(200, "text/plain", "ok\n");
        sendAll(client_fd, resp.data(), resp.size());
        ::close(client_fd);
        return;
    }

    if (path == "/" || path == "/index.html") {
        const std::string resp = buildHttpResponse(200, "text/html; charset=utf-8", kHtmlPage);
        sendAll(client_fd, resp.data(), resp.size());
        ::close(client_fd);
        return;
    }

    if (path == "/stream") {
        writeStream(client_fd);
        ::close(client_fd);
        return;
    }

    const std::string resp = buildHttpResponse(404, "text/plain", "not found\n");
    sendAll(client_fd, resp.data(), resp.size());
    ::close(client_fd);
}

void StreamServer::writeStream(int client_fd) {
    static constexpr char kBoundary[] = "frame";

    std::ostringstream header;
    header << "HTTP/1.1 200 OK\r\n"
           << "Connection: close\r\n"
           << "Cache-Control: no-cache, no-store, must-revalidate\r\n"
           << "Pragma: no-cache\r\n"
           << "Content-Type: multipart/x-mixed-replace; boundary=" << kBoundary << "\r\n"
           << "\r\n";
    const std::string header_str = header.str();
    if (!sendAll(client_fd, header_str.data(), header_str.size())) {
        return;
    }

    uint64_t last_seq = 0;
    while (running_.load()) {
        std::vector<uchar> jpeg;
        uint64_t seq = frame_seq_.load();

        // Espera novo frame (evita reenviar o mesmo JPEG em loop apertado).
        if (seq == last_seq) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            if (jpeg_buffer_.empty()) {
                continue;
            }
            jpeg = jpeg_buffer_;
            seq = frame_seq_.load();
        }
        last_seq = seq;

        std::ostringstream part;
        part << "--" << kBoundary << "\r\n"
             << "Content-Type: image/jpeg\r\n"
             << "Content-Length: " << jpeg.size() << "\r\n"
             << "\r\n";
        const std::string part_header = part.str();

        if (!sendAll(client_fd, part_header.data(), part_header.size())) {
            break;
        }
        if (!sendAll(client_fd, jpeg.data(), jpeg.size())) {
            break;
        }
        if (!sendAll(client_fd, "\r\n", 2)) {
            break;
        }
    }
}

bool StreamServer::sendAll(int fd, const void* data, size_t len) {
    const char* ptr = static_cast<const char*>(data);
    size_t sent = 0;
    while (sent < len) {
        const ssize_t n = ::send(fd, ptr + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

std::string StreamServer::buildHttpResponse(int status,
                                            const std::string& content_type,
                                            const std::string& body,
                                            bool connection_close) {
    const char* status_text = (status == 200) ? "OK" : (status == 404) ? "Not Found" : "Error";
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << ' ' << status_text << "\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n";
    if (connection_close) {
        oss << "Connection: close\r\n";
    }
    oss << "\r\n" << body;
    return oss.str();
}

}  // namespace contador
