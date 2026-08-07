#include "contador/stream_server.hpp"
#include "contador/logger.hpp"

#include <opencv2/imgcodecs.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace contador {
namespace {

constexpr const char* kBaseStyle =
    "<style>\n"
    "  * { box-sizing: border-box; margin: 0; padding: 0; }\n"
    "  body { background: #111; color: #eee; font-family: sans-serif;\n"
    "         min-height: 100vh; display: flex; flex-direction: column;\n"
    "         align-items: center; padding: 2rem 1rem; gap: 1rem; }\n"
    "  h1 { font-weight: 500; letter-spacing: 0.04em; }\n"
    "  .wrap { width: 100%; max-width: 640px; display: flex; flex-direction: column; gap: 1rem; }\n"
    "  nav { text-align: center; color: #9ad; }\n"
    "  nav a { color: #7cf; text-decoration: none; }\n"
    "  nav a:hover { text-decoration: underline; }\n"
    "  .card { background: #1b1b1b; border: 1px solid #333; border-radius: 8px; padding: 1.2rem 1.4rem; }\n"
    "  .card p { margin: 0.35rem 0; }\n"
    "  .label { color: #999; }\n"
    "  .count { color: #4f4; font-size: 1.3rem; }\n"
    "  .error, .notice { padding: 0.6rem 0.9rem; border-radius: 6px; margin-bottom: 0.5rem; }\n"
    "  .error { background: #3a1414; color: #f88; border: 1px solid #722; }\n"
    "  .notice { background: #123a1c; color: #8f8; border: 1px solid #274; }\n"
    "  .badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.85rem; }\n"
    "  .badge-running { background: #123a1c; color: #8f8; }\n"
    "  .badge-finished { background: #1a2a3a; color: #8cf; }\n"
    "  .badge-starting { background: #2a2a1a; color: #fe8; }\n"
    "  .badge-error { background: #3a1414; color: #f88; }\n"
    "  .badge-stopped { background: #2a2a2a; color: #aaa; }\n"
    "  .badge-idle { background: #2a2a2a; color: #aaa; }\n"
    "  .preview { max-width: 100%; border: 1px solid #333; border-radius: 8px; }\n"
    "  .meta { color: #777; font-size: 0.85rem; }\n"
    "  form.card { display: flex; flex-direction: column; gap: 0.7rem; }\n"
    "  form label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.9rem; color: #bbb; }\n"
    "  form input { background: #0f0f0f; border: 1px solid #333; border-radius: 6px; color: #eee;\n"
    "               padding: 0.5rem 0.6rem; font-size: 0.95rem; }\n"
    "  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; }\n"
    "  button { background: #245; color: #fff; border: none; border-radius: 6px; padding: 0.6rem 1rem;\n"
    "            font-size: 1rem; cursor: pointer; }\n"
    "  button:hover { background: #357; }\n"
    "</style>\n";

std::string htmlEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (const char c : s) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            default: out += c;
        }
    }
    return out;
}

std::string urlDecode(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '%' && i + 2 < in.size()) {
            const std::string hex = in.substr(i + 1, 2);
            char* end = nullptr;
            const long value = std::strtol(hex.c_str(), &end, 16);
            if (end == hex.c_str() + 2) {
                out.push_back(static_cast<char>(value));
                i += 2;
                continue;
            }
        }
        out.push_back(in[i] == '+' ? ' ' : in[i]);
    }
    return out;
}

std::unordered_map<std::string, std::string> parseFormBody(const std::string& body) {
    std::unordered_map<std::string, std::string> fields;
    size_t start = 0;
    while (start <= body.size()) {
        const size_t amp = body.find('&', start);
        const std::string pair =
            body.substr(start, amp == std::string::npos ? std::string::npos : amp - start);
        const auto eq = pair.find('=');
        if (eq != std::string::npos) {
            fields[urlDecode(pair.substr(0, eq))] = urlDecode(pair.substr(eq + 1));
        } else if (!pair.empty()) {
            fields[urlDecode(pair)] = "";
        }
        if (amp == std::string::npos) {
            break;
        }
        start = amp + 1;
    }
    return fields;
}

std::string textField(const char* name, const char* label, const std::string& value) {
    std::ostringstream out;
    out << "<label>" << label << "<input type=\"text\" name=\"" << name << "\" value=\""
        << htmlEscape(value) << "\"></label>\n";
    return out.str();
}

template <typename T>
std::string numField(const char* name, const char* label, T value, const char* step = "any") {
    std::ostringstream out;
    out << "<label>" << label << "<input type=\"number\" step=\"" << step << "\" name=\"" << name
        << "\" value=\"" << value << "\"></label>\n";
    return out.str();
}

}  // namespace

StreamServer::StreamServer() = default;

StreamServer::~StreamServer() {
    stop();
}

void StreamServer::attachController(PipelineController* controller, std::string config_path) {
    controller_ = controller;
    config_path_ = std::move(config_path);
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
                            << "/  (painel) /stream (MJPEG) /config (configuração)";
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

bool StreamServer::readRequest(int fd, HttpRequest& out) {
    std::string data;
    char buf[4096];
    size_t header_end = std::string::npos;

    while (header_end == std::string::npos && data.size() < 16384) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            return false;
        }
        data.append(buf, static_cast<size_t>(n));
        header_end = data.find("\r\n\r\n");
    }
    if (header_end == std::string::npos) {
        return false;
    }

    const std::string header_block = data.substr(0, header_end);
    std::istringstream header_stream(header_block);
    std::string request_line;
    std::getline(header_stream, request_line);
    if (!request_line.empty() && request_line.back() == '\r') {
        request_line.pop_back();
    }

    {
        std::istringstream line_stream(request_line);
        line_stream >> out.method >> out.path;
    }
    const auto q = out.path.find('?');
    if (q != std::string::npos) {
        out.path = out.path.substr(0, q);
    }

    size_t content_length = 0;
    std::string header_line;
    while (std::getline(header_stream, header_line)) {
        if (!header_line.empty() && header_line.back() == '\r') {
            header_line.pop_back();
        }
        const auto pos = header_line.find(':');
        if (pos == std::string::npos) {
            continue;
        }
        std::string name = header_line.substr(0, pos);
        std::string value = header_line.substr(pos + 1);
        const size_t v0 = value.find_first_not_of(' ');
        value = (v0 == std::string::npos) ? std::string() : value.substr(v0);
        for (auto& c : name) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (name == "content-length") {
            try {
                content_length = static_cast<size_t>(std::stoul(value));
            } catch (...) {
                content_length = 0;
            }
        }
    }

    content_length = std::min(content_length, static_cast<size_t>(1 << 20));  // cap 1MB
    std::string body = data.substr(header_end + 4);
    while (body.size() < content_length) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        body.append(buf, static_cast<size_t>(n));
    }
    out.body = std::move(body);
    return true;
}

void StreamServer::handleClient(int client_fd) {
    HttpRequest req;
    if (!readRequest(client_fd, req)) {
        ::close(client_fd);
        return;
    }

    if (req.path == "/health") {
        const std::string resp = buildHttpResponse(200, "text/plain", "ok\n");
        sendAll(client_fd, resp.data(), resp.size());
        ::close(client_fd);
        return;
    }

    if (req.path == "/" || req.path == "/index.html") {
        const std::string resp =
            buildHttpResponse(200, "text/html; charset=utf-8", renderDashboard());
        sendAll(client_fd, resp.data(), resp.size());
        ::close(client_fd);
        return;
    }

    if (req.path == "/config") {
        if (req.method == "POST") {
            handleConfigPost(client_fd, req.body);
            return;
        }
        const PipelineConfig cfg = controller_ != nullptr ? controller_->config() : PipelineConfig{};
        const std::string resp =
            buildHttpResponse(200, "text/html; charset=utf-8", renderConfigForm(cfg, ""));
        sendAll(client_fd, resp.data(), resp.size());
        ::close(client_fd);
        return;
    }

    if (req.path == "/api/status") {
        const std::string resp = buildHttpResponse(200, "application/json", renderStatusJson());
        sendAll(client_fd, resp.data(), resp.size());
        ::close(client_fd);
        return;
    }

    if (req.path == "/stream") {
        writeStream(client_fd);
        ::close(client_fd);
        return;
    }

    const std::string resp = buildHttpResponse(404, "text/plain", "not found\n");
    sendAll(client_fd, resp.data(), resp.size());
    ::close(client_fd);
}

void StreamServer::handleConfigPost(int client_fd, const std::string& body) {
    const auto fields = parseFormBody(body);
    PipelineConfig cfg = controller_ != nullptr ? controller_->config() : PipelineConfig{};

    auto getStr = [&](const char* key, std::string& target) {
        const auto it = fields.find(key);
        if (it != fields.end()) {
            target = it->second;
        }
    };
    bool ok = true;
    auto getFloat = [&](const char* key, float& target) {
        const auto it = fields.find(key);
        if (it == fields.end()) {
            return;
        }
        try {
            target = std::stof(it->second);
        } catch (...) {
            ok = false;
        }
    };
    auto getInt = [&](const char* key, int& target) {
        const auto it = fields.find(key);
        if (it == fields.end()) {
            return;
        }
        try {
            target = std::stoi(it->second);
        } catch (...) {
            ok = false;
        }
    };

    getStr("source", cfg.source);
    getStr("model_path", cfg.model_path);
    getFloat("conf", cfg.conf);
    getFloat("nms", cfg.nms);
    getInt("imgsz", cfg.imgsz);
    getInt("threads", cfg.threads);
    getFloat("line_x1", cfg.line.p1.x);
    getFloat("line_y1", cfg.line.p1.y);
    getFloat("line_x2", cfg.line.p2.x);
    getFloat("line_y2", cfg.line.p2.y);
    getInt("reconnect_ms", cfg.reconnect_ms);
    getInt("read_timeout_ms", cfg.read_timeout_ms);

    if (!ok || cfg.source.empty() || cfg.model_path.empty()) {
        const std::string resp = buildHttpResponse(
            400, "text/html; charset=utf-8",
            renderConfigForm(cfg, "Valores inválidos — nada foi alterado."));
        sendAll(client_fd, resp.data(), resp.size());
        ::close(client_fd);
        return;
    }

    if (controller_ != nullptr) {
        controller_->reconfigure(cfg);
        LogInfo("StreamServer") << "Config aplicada via /config (hot-reload): fonte=" << cfg.source;
    }
    if (!config_path_.empty()) {
        saveConfig(config_path_, cfg);
    }

    std::ostringstream resp;
    resp << "HTTP/1.1 303 See Other\r\n"
         << "Location: /config?ok=1\r\n"
         << "Content-Length: 0\r\n"
         << "Connection: close\r\n\r\n";
    const std::string resp_str = resp.str();
    sendAll(client_fd, resp_str.data(), resp_str.size());
    ::close(client_fd);
}

std::string StreamServer::renderDashboard() const {
    std::ostringstream html;
    html << "<!DOCTYPE html><html lang=\"pt-BR\"><head>"
         << "<meta charset=\"utf-8\"/><meta name=\"viewport\" "
            "content=\"width=device-width, initial-scale=1\"/>"
         << "<title>Contador de Eixos</title>"
         << "<meta http-equiv=\"refresh\" content=\"3\">" << kBaseStyle << "</head><body>"
         << "<div class=\"wrap\">"
         << "<h1>Contador de Eixos</h1>"
         << "<nav><a href=\"/\">Painel</a> &middot; <a href=\"/config\">Configuração</a> &middot; "
            "<a href=\"/stream\">Stream</a></nav>";

    if (controller_ == nullptr) {
        html << "<div class=\"card\"><p>Controller não conectado.</p></div>";
    } else {
        const auto st = controller_->status();
        const auto cfg = controller_->config();
        html << "<div class=\"card\">"
             << "<p><span class=\"label\">Estado:</span> <span class=\"badge badge-"
             << toString(st.state) << "\">" << toString(st.state) << "</span></p>"
             << "<p><span class=\"label\">Fonte:</span> " << htmlEscape(cfg.source)
             << (st.is_live ? " (live)" : " (arquivo)") << "</p>"
             << "<p><span class=\"label\">Modelo:</span> " << htmlEscape(cfg.model_path) << "</p>"
             << "<p><span class=\"label\">Eixos/objetos contados:</span> <span class=\"count\">"
             << st.total_count << "</span></p>"
             << "<p><span class=\"label\">Frames processados:</span> " << st.frames_processed
             << "</p>"
             << "<p><span class=\"label\">Duração da sessão:</span> " << std::fixed
             << std::setprecision(1) << st.elapsed_sec << "s</p>";
        if (!st.error_message.empty()) {
            html << "<p class=\"error\">" << htmlEscape(st.error_message) << "</p>";
        }
        html << "</div>"
             << "<img class=\"preview\" src=\"/stream\" alt=\"MJPEG stream\"/>";
    }

    html << "<p class=\"meta\">API: <code>/api/status</code> &middot; painel atualiza a cada 3s"
            "</p></div></body></html>";
    return html.str();
}

std::string StreamServer::renderConfigForm(const PipelineConfig& cfg,
                                           const std::string& message) const {
    std::ostringstream html;
    html << "<!DOCTYPE html><html lang=\"pt-BR\"><head>"
         << "<meta charset=\"utf-8\"/><meta name=\"viewport\" "
            "content=\"width=device-width, initial-scale=1\"/>"
         << "<title>Configuração — Contador de Eixos</title>" << kBaseStyle << "</head><body>"
         << "<div class=\"wrap\">"
         << "<h1>Configuração</h1>"
         << "<nav><a href=\"/\">Painel</a> &middot; <a href=\"/config\">Configuração</a> &middot; "
            "<a href=\"/stream\">Stream</a></nav>";

    if (!message.empty()) {
        html << "<p class=\"error\">" << htmlEscape(message) << "</p>";
    }

    html << "<form method=\"POST\" action=\"/config\" class=\"card\">"
         << textField("source", "Fonte (RTSP / caminho de vídeo / 'synthetic')", cfg.source)
         << textField("model_path", "Modelo (.onnx)", cfg.model_path)
         << "<div class=\"row2\">" << numField("conf", "Confiança mínima", cfg.conf, "0.01")
         << numField("nms", "IoU NMS", cfg.nms, "0.01") << "</div>"
         << "<div class=\"row2\">" << numField("imgsz", "Tamanho letterbox", cfg.imgsz, "1")
         << numField("threads", "Threads ORT", cfg.threads, "1") << "</div>"
         << "<p class=\"label\">Linha virtual de contagem (x1,y1 → x2,y2)</p>"
         << "<div class=\"row2\">" << numField("line_x1", "x1", cfg.line.p1.x, "1")
         << numField("line_y1", "y1", cfg.line.p1.y, "1") << "</div>"
         << "<div class=\"row2\">" << numField("line_x2", "x2", cfg.line.p2.x, "1")
         << numField("line_y2", "y2", cfg.line.p2.y, "1") << "</div>"
         << "<div class=\"row2\">"
         << numField("reconnect_ms", "Reconexão RTSP (ms)", cfg.reconnect_ms, "1")
         << numField("read_timeout_ms", "Timeout leitura (ms)", cfg.read_timeout_ms, "1")
         << "</div>"
         << "<button type=\"submit\">Aplicar</button>"
         << "</form>"
         << "<p class=\"meta\">Alterações são aplicadas imediatamente (hot-reload, sem reiniciar o "
            "processo) e salvas em disco. Fontes tipo arquivo de vídeo processam até o fim e "
            "mostram o relatório no painel — sem encerrar o servidor.</p>"
         << "</div></body></html>";
    return html.str();
}

std::string StreamServer::renderStatusJson() const {
    if (controller_ == nullptr) {
        return "{\"error\":\"controller não conectado\"}";
    }
    const auto st = controller_->status();
    const auto cfg = controller_->config();

    std::ostringstream json;
    json << "{"
         << "\"state\":\"" << toString(st.state) << "\","
         << "\"source\":\"" << jsonEscape(cfg.source) << "\","
         << "\"model_path\":\"" << jsonEscape(cfg.model_path) << "\","
         << "\"is_live\":" << (st.is_live ? "true" : "false") << ","
         << "\"total_count\":" << st.total_count << ","
         << "\"frames_processed\":" << st.frames_processed << ","
         << "\"elapsed_sec\":" << st.elapsed_sec << ","
         << "\"error_message\":\"" << jsonEscape(st.error_message) << "\""
         << "}";
    return json.str();
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
    const char* status_text = (status == 200)   ? "OK"
                              : (status == 303) ? "See Other"
                              : (status == 400) ? "Bad Request"
                              : (status == 404) ? "Not Found"
                                                 : "Error";
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
