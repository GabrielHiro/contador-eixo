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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace contador {
namespace {

constexpr const char* kBaseStyle =
    "<style>\n"
    "  * { box-sizing: border-box; margin: 0; padding: 0; }\n"
    "  body { background: linear-gradient(180deg, #0c1118 0%, #111826 100%); color: #eef3fb;\n"
    "         font-family: sans-serif; min-height: 100vh; padding: 1.2rem; }\n"
    "  h1, h2 { font-weight: 600; letter-spacing: 0.02em; }\n"
    "  h1 { font-size: 1.6rem; }\n"
    "  h2 { font-size: 1.1rem; margin-bottom: 0.4rem; }\n"
    "  .wrap { width: 100%; max-width: 1180px; margin: 0 auto; display: flex; flex-direction: column; gap: 1rem; }\n"
    "  .nav { display: flex; flex-wrap: wrap; gap: 0.5rem; }\n"
    "  .nav a { color: #d8e7ff; text-decoration: none; background: #172030; border: 1px solid #2b3b53;\n"
    "           border-radius: 999px; padding: 0.45rem 0.8rem; }\n"
    "  .nav a:hover { background: #213149; }\n"
    "  .layout { display: flex; flex-direction: column; gap: 1rem; }\n"
    "  .card { background: rgba(13, 19, 29, 0.92); border: 1px solid #273244; border-radius: 14px; padding: 1.1rem 1.2rem; box-shadow: 0 12px 32px rgba(0,0,0,0.18); }\n"
    "  .card p { margin: 0.35rem 0; }\n"
    "  .label { color: #999; }\n"
    "  .count { color: #7dff9e; font-size: 1.3rem; font-weight: 700; }\n"
    "  .error, .notice { padding: 0.6rem 0.9rem; border-radius: 6px; margin-bottom: 0.5rem; }\n"
    "  .error { background: #3a1414; color: #ffb2b2; border: 1px solid #7c2f2f; }\n"
    "  .notice { background: #123a1c; color: #a8ffbf; border: 1px solid #245d36; }\n"
    "  .badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.85rem; }\n"
    "  .badge-running { background: #123a1c; color: #8f8; }\n"
    "  .badge-finished { background: #1a2a3a; color: #8cf; }\n"
    "  .badge-starting { background: #2a2a1a; color: #fe8; }\n"
    "  .badge-error { background: #3a1414; color: #f88; }\n"
    "  .badge-stopped { background: #2a2a2a; color: #aaa; }\n"
    "  .badge-idle { background: #2a2a2a; color: #aaa; }\n"
    "  .preview { width: 100%; max-width: 100%; border: 1px solid #304057; border-radius: 12px; background: #0b0f15; }\n"
    "  .preview-stage { position: relative; width: 100%; min-height: 68vh; border: 1px solid #304057; border-radius: 16px; overflow: hidden; background: #0b0f15; }\n"
    "  .preview-stage img { display: block; width: 100%; height: 100%; object-fit: contain; }\n"
    "  .preview-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; touch-action: none; }\n"
    "  .preview-hud { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin-top: 0.5rem; }\n"
    "  .chip { display: inline-flex; align-items: center; gap: 0.35rem; border-radius: 999px; padding: 0.3rem 0.6rem; background: #152233; border: 1px solid #2d415c; color: #d8e7ff; font-size: 0.86rem; }\n"
    "  .editor-card { display: flex; flex-direction: column; gap: 0.85rem; }\n"
    "  .editor-toolbar { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }\n"
    "  .editor-toolbar .ghost { padding: 0.5rem 0.8rem; }\n"
    "  .editor-toolbar select { min-width: 220px; }\n"
    "  .meta { color: #777; font-size: 0.85rem; }\n"
    "  form.card { display: flex; flex-direction: column; gap: 0.7rem; }\n"
    "  form label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.9rem; color: #cad5e4; }\n"
    "  form input { background: #0f0f0f; border: 1px solid #333; border-radius: 6px; color: #eee;\n"
    "               padding: 0.5rem 0.6rem; font-size: 0.95rem; }\n"
    "  form select { background: #0f0f0f; border: 1px solid #333; border-radius: 6px; color: #eee;\n"
    "                padding: 0.5rem 0.6rem; font-size: 0.95rem; }\n"
    "  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; }\n"
    "  .row3 { display: grid; grid-template-columns: 1.2fr 1fr 1fr; gap: 0.7rem; }\n"
    "  .hint { color: #8ea0b8; font-size: 0.86rem; }\n"
    "  button { background: #245; color: #fff; border: none; border-radius: 6px; padding: 0.6rem 1rem;\n"
    "            font-size: 1rem; cursor: pointer; }\n"
    "  button:hover { background: #357; }\n"
    "  .ghost { background: #1a2533; border: 1px solid #304057; }\n"
    "  .ghost:hover { background: #233246; }\n"
    "  .actions { display: flex; flex-wrap: wrap; gap: 0.5rem; }\n"
    "  .split { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }\n"
    "  @media (max-width: 900px) { .layout, .split { grid-template-columns: 1fr; } .row2, .row3 { grid-template-columns: 1fr; } }\n"
    "</style>\n";

constexpr const char* kJsHelpers =
    "<script>\n"
    "let lineDrawState = 0;\n"
    "function el(id) { return document.getElementById(id); }\n"
    "function setField(id, value) { const node = el(id); if (node && value !== '') { node.value = value; redrawPreview(); } }\n"
    "function normalizeNumericValue(value) { let text = String(value ?? '').trim().replace(/\\s+/g, '');\n"
    "  if (!text) return ''; if (text.includes(',') && text.includes('.')) {\n"
    "    if (text.lastIndexOf(',') > text.lastIndexOf('.')) { text = text.replace(/\\./g, '').replace(/,/g, '.'); }\n"
    "    else { text = text.replace(/,/g, ''); } } else { text = text.replace(/,/g, '.'); } return text; }\n"
    "function normalizeNumericField(node) { if (!node) return; const next = normalizeNumericValue(node.value); if (next !== '') node.value = next; }\n"
    "function normalizeForm(form) { if (!form) return; form.querySelectorAll('[data-num-kind]').forEach((node) => normalizeNumericField(node)); }\n"
    "function syncPreset(selectId, fieldId) { const sel = el(selectId); const field = el(fieldId); if (!sel || !field) return;\n"
    "  if (sel.value !== '__custom__') { field.value = sel.value; redrawPreview(); } }\n"
    "function numValue(id) { const node = el(id); return node ? Number.parseFloat(normalizeNumericValue(node.value || '0')) || 0 : 0; }\n"
    "function syncCanvasSize() { const img = el('preview-image'); const canvas = el('line-canvas'); if (!img || !canvas) return;\n"
    "  const rect = img.getBoundingClientRect();\n"
    "  canvas.width = Math.max(1, Math.round(rect.width)); canvas.height = Math.max(1, Math.round(rect.height));\n"
    "  canvas.style.width = rect.width + 'px'; canvas.style.height = rect.height + 'px'; }\n"
    "function redrawPreview() { const img = el('preview-image'); const canvas = el('line-canvas'); if (!img || !canvas) return;\n"
    "  syncCanvasSize(); const ctx = canvas.getContext('2d'); if (!ctx) return; ctx.clearRect(0, 0, canvas.width, canvas.height);\n"
    "  const iw = img.naturalWidth || canvas.width; const ih = img.naturalHeight || canvas.height; if (!iw || !ih) return;\n"
    "  const sx = canvas.width / iw; const sy = canvas.height / ih;\n"
    "  const x1 = numValue('line_x1') * sx, y1 = numValue('line_y1') * sy, x2 = numValue('line_x2') * sx, y2 = numValue('line_y2') * sy;\n"
    "  ctx.save(); ctx.lineWidth = 3; ctx.strokeStyle = '#7dff9e'; ctx.shadowColor = 'rgba(0,0,0,0.35)'; ctx.shadowBlur = 8;\n"
    "  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();\n"
    "  const drawHandle = (x, y, color) => { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#0b0f15'; ctx.stroke(); };\n"
    "  drawHandle(x1, y1, '#ffd166'); drawHandle(x2, y2, '#00d4ff'); ctx.restore();\n"
    "  const label = el('preview-label'); if (label) { label.textContent = lineDrawState === 1 ? 'Clique no segundo ponto da linha' : 'Clique em dois pontos para desenhar'; }\n"
    "}\n"
    "function installPreview() { const img = el('preview-image'); const canvas = el('line-canvas'); if (!img || !canvas) return;\n"
    "  const onResize = () => redrawPreview(); window.addEventListener('resize', onResize); img.addEventListener('load', redrawPreview);\n"
    "  canvas.addEventListener('click', (ev) => { const rect = canvas.getBoundingClientRect(); const iw = img.naturalWidth || rect.width; const ih = img.naturalHeight || rect.height;\n"
    "    const x = Math.max(0, Math.min(iw, ((ev.clientX - rect.left) / rect.width) * iw));\n"
    "    const y = Math.max(0, Math.min(ih, ((ev.clientY - rect.top) / rect.height) * ih));\n"
    "    if (lineDrawState === 0) { setField('line_x1', x.toFixed(2)); setField('line_y1', y.toFixed(2)); setField('line_x2', x.toFixed(2)); setField('line_y2', y.toFixed(2)); lineDrawState = 1; }\n"
    "    else { setField('line_x2', x.toFixed(2)); setField('line_y2', y.toFixed(2)); lineDrawState = 0; } redrawPreview(); }); redrawPreview(); }\n"
    "function resetLineToPreset() { lineDrawState = 0; redrawPreview(); }\n"
    "function zoomPreview(scale) { const stage = el('preview-stage'); if (!stage) return; stage.style.minHeight = Math.max(420, Math.round(window.innerHeight * scale)) + 'px'; redrawPreview(); }\n"
    "document.addEventListener('DOMContentLoaded', installPreview);\n"
    "</script>\n";

const std::string kAttachedVideo = "D2_S20260807075348_E20260807080000.mp4";

struct PresetOption {
    std::string label;
    std::string value;
};

std::vector<PresetOption> videoPresets() {
    std::vector<PresetOption> items = {
        {"Synthetic", "synthetic"},
        {"Vídeo exemplo 181327--vv.mp4", "181327--vv.mp4"},
        {"Vídeo exemplo 181349--vv.mp4", "181349--vv.mp4"},
    };
    if (std::filesystem::exists(kAttachedVideo)) {
        items.push_back({"Vídeo anexado para teste", kAttachedVideo});
    }
    return items;
}

std::vector<PresetOption> vehicleModelPresets() {
    std::vector<PresetOption> items = {
        {"Modelo principal treinado localmente", "models/vehicles.onnx"},
        {"Detector pré-treinado de veículos", "models/vehicles.onnx"},
    };
    return items;
}

std::vector<PresetOption> axleModelPresets() {
    std::vector<PresetOption> items = {
        {"Modelo de eixos padrão", "models/axles.onnx"},
        {"Modelo de eixos pré-treinado", "models/axles.onnx"},
    };
    return items;
}

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

std::string jsonStringField(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) return {};
    const size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) return {};
    const size_t quote = json.find('\"', colon + 1);
    if (quote == std::string::npos) return {};
    std::string value;
    for (size_t i = quote + 1; i < json.size(); ++i) {
        if (json[i] == '\"') return value;
        if (json[i] == '\\' && i + 1 < json.size()) ++i;
        value += json[i];
    }
    return {};
}

std::string textField(const char* name, const char* label, const std::string& value) {
    std::ostringstream out;
    out << "<label>" << label << "<input id=\"" << name << "\" type=\"text\" name=\"" << name << "\" value=\""
        << htmlEscape(value) << "\"></label>\n";
    return out.str();
}

std::string presetSelect(const char* select_id,
                         const char* field_id,
                         const char* label,
                         const std::string& current,
                         const std::vector<PresetOption>& options,
                         const char* hint) {
    std::ostringstream out;
    out << "<label>" << label << "<select id=\"" << select_id << "\" onchange=\"syncPreset('"
        << select_id << "', '" << field_id << "')\">";
    out << "<option value=\"__custom__\">Personalizado</option>";
    for (const auto& opt : options) {
        out << "<option value=\"" << htmlEscape(opt.value) << "\"";
        if (opt.value == current) {
            out << " selected";
        }
        out << ">" << htmlEscape(opt.label) << "</option>";
    }
    out << "</select>";
    if (hint != nullptr && *hint != '\0') {
        out << "<span class=\"hint\">" << htmlEscape(hint) << "</span>";
    }
    out << "</label>\n";
    return out.str();
}

template <typename T>
std::string numField(const char* name, const char* label, T value, const char* step = "any") {
    std::ostringstream out;
    const bool is_integer = std::string(step) == "1";
    out << "<label>" << label << "<input id=\"" << name
        << "\" type=\"text\" name=\"" << name << "\" inputmode=\""
        << (is_integer ? "numeric" : "decimal") << "\" data-num-kind=\""
        << (is_integer ? "int" : "float") << "\" value=\"" << value << "\"></label>\n";
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

    if (req.path == "/api/count") {
        if (req.method != "POST") {
            const std::string resp = buildHttpResponse(405, "text/plain", "use POST\n");
            sendAll(client_fd, resp.data(), resp.size());
            ::close(client_fd);
            return;
        }
        handleCountPost(client_fd, req.body);
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

void StreamServer::handleCountPost(int client_fd, const std::string& body) {
    std::lock_guard<std::mutex> job_lock(job_mutex_);
    PipelineConfig cfg = controller_ != nullptr ? controller_->config() : PipelineConfig{};
    const std::string video = jsonStringField(body, "video");
    if (!video.empty()) cfg.source = video;
    if (video.empty() || !loadConfigJson(body, cfg) || controller_ == nullptr) {
        const std::string resp = buildHttpResponse(
            400, "application/json", "{\"error\":\"JSON deve conter video e configuracao valida\"}\n");
        sendAll(client_fd, resp.data(), resp.size());
        ::close(client_fd);
        return;
    }
    cfg.source = video;
    cfg.axle_enabled = true;
    cfg.single_vehicle_mode = true;

    controller_->reconfigure(cfg);
    bool started = false;
    for (int i = 0; i < 600; ++i) {
        const auto st = controller_->status();
        if (controller_->config().source == video &&
            (st.state == PipelineState::Starting || st.state == PipelineState::Running ||
             st.state == PipelineState::Error)) {
            started = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!started) {
        const std::string resp = buildHttpResponse(
            504, "application/json", "{\"error\":\"tempo excedido ao iniciar o processamento\"}\n");
        sendAll(client_fd, resp.data(), resp.size());
        ::close(client_fd);
        return;
    }
    while (running_.load()) {
        const auto st = controller_->status();
        if (st.state == PipelineState::Finished || st.state == PipelineState::Error) {
            std::ostringstream json;
            json << "{\"video\":\"" << jsonEscape(st.source) << "\",\"vehicles\":"
                 << st.vehicle_count << ",\"axles\":" << st.axle_count
                 << ",\"frames\":" << st.frames_processed << ",\"state\":\""
                 << toString(st.state) << "\"";
            if (!cfg.output_path.empty()) {
                json << ",\"output_video\":\"" << jsonEscape(cfg.output_path) << "\"";
            }
            if (!st.error_message.empty()) {
                json << ",\"error\":\"" << jsonEscape(st.error_message) << "\"";
            }
            json << "}\n";
            const std::string resp = buildHttpResponse(
                st.state == PipelineState::Error ? 422 : 200, "application/json", json.str());
            sendAll(client_fd, resp.data(), resp.size());
            ::close(client_fd);
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ::close(client_fd);
}

void StreamServer::handleConfigPost(int client_fd, const std::string& body) {
    const auto fields = parseFormBody(body);
    PipelineConfig cfg = controller_ != nullptr ? controller_->config() : PipelineConfig{};

    auto trimCopy = [](std::string value) {
        const auto first = value.find_first_not_of(" \t\r\n");
        const auto last = value.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return std::string{};
        }
        return value.substr(first, last - first + 1);
    };

    auto normalizeNumeric = [&](std::string value) {
        value = trimCopy(std::move(value));
        value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
                          return std::isspace(c) != 0 || c == '_';
                      }),
                    value.end());
        if (value.empty()) {
            return value;
        }
        const auto comma = value.find_last_of(',');
        const auto dot = value.find_last_of('.');
        if (comma != std::string::npos && dot != std::string::npos) {
            if (comma > dot) {
                value.erase(std::remove(value.begin(), value.end(), '.'), value.end());
                std::replace(value.begin(), value.end(), ',', '.');
            } else {
                value.erase(std::remove(value.begin(), value.end(), ','), value.end());
            }
        } else {
            std::replace(value.begin(), value.end(), ',', '.');
        }
        return value;
    };

    auto parseFloat = [&](const char* key, float& target) {
        const auto it = fields.find(key);
        if (it == fields.end()) {
            return true;
        }
        try {
            target = std::stof(normalizeNumeric(it->second));
            return true;
        } catch (...) {
            return false;
        }
    };

    auto parseInt = [&](const char* key, int& target) {
        const auto it = fields.find(key);
        if (it == fields.end()) {
            return true;
        }
        try {
            const double parsed = std::stod(normalizeNumeric(it->second));
            target = static_cast<int>(std::lround(parsed));
            return true;
        } catch (...) {
            return false;
        }
    };

    auto getStr = [&](const char* key, std::string& target) {
        const auto it = fields.find(key);
        if (it != fields.end()) {
            target = it->second;
        }
    };
    bool ok = true;
    ok &= parseFloat("conf", cfg.conf);
    ok &= parseFloat("nms", cfg.nms);
    ok &= parseInt("imgsz", cfg.imgsz);
    ok &= parseInt("threads", cfg.threads);
    ok &= parseFloat("line_x1", cfg.line.p1.x);
    ok &= parseFloat("line_y1", cfg.line.p1.y);
    ok &= parseFloat("line_x2", cfg.line.p2.x);
    ok &= parseFloat("line_y2", cfg.line.p2.y);
    ok &= parseInt("reconnect_ms", cfg.reconnect_ms);
    ok &= parseInt("read_timeout_ms", cfg.read_timeout_ms);

    {
        const auto it = fields.find("axle_enabled");
        if (it != fields.end()) {
            const std::string& v = it->second;
            cfg.axle_enabled = (v == "1" || v == "true" || v == "on" || v == "yes");
        } else {
            // checkbox HTML: ausente no POST = desmarcado
            cfg.axle_enabled = false;
        }
    }
    getStr("source", cfg.source);
    getStr("model_path", cfg.model_path);
    getStr("axle_model_path", cfg.axle_model_path);
    ok &= parseFloat("axle_conf", cfg.axle_conf);
    ok &= parseFloat("axle_nms", cfg.axle_nms);
    ok &= parseInt("axle_imgsz", cfg.axle_imgsz);
    ok &= parseFloat("axle_crop_margin", cfg.axle_crop_margin);

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
            << "<meta http-equiv=\"refresh\" content=\"3\">" << kBaseStyle << kJsHelpers
            << "</head><body>"
         << "<div class=\"wrap\">"
         << "<h1>Contador de Eixos</h1>"
            << "<div class=\"nav\"><a href=\"/\">Painel</a><a href=\"/config\">Configuração</a>"
                "<a href=\"/stream\">Stream ao vivo</a><a href=\"/api/status\">Status JSON</a>"
                "<a href=\"/health\">Health</a></div>";

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
             << "<p><span class=\"label\">Modelo (veículos):</span> " << htmlEscape(cfg.model_path)
             << "</p>"
             << "<p><span class=\"label\">Modelo (eixos):</span> "
             << (cfg.axle_enabled ? htmlEscape(cfg.axle_model_path) : std::string("(desativado)"))
             << "</p>"
                 << "<p class=\"hint\">Use /config para trocar vídeo/modelos sem reiniciar. O vídeo "
                     "anexado fica disponível como preset.</p>"
             << "<p><span class=\"label\">Veículos contados:</span> <span class=\"count\">"
             << st.vehicle_count << "</span></p>"
             << "<p><span class=\"label\">Eixos totais:</span> <span class=\"count\">"
             << st.axle_count << "</span></p>"
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
            << "<title>Configuração — Contador de Eixos</title>" << kBaseStyle << kJsHelpers
            << "</head><body>"
         << "<div class=\"wrap\">"
         << "<h1>Configuração</h1>"
            << "<div class=\"nav\"><a href=\"/\">Painel</a><a href=\"/config\">Configuração</a>"
                "<a href=\"/stream\">Stream ao vivo</a><a href=\"/api/status\">Status JSON</a>"
                "<a href=\"/health\">Health</a></div>";

    if (!message.empty()) {
          html << "<p class=\"notice\">" << htmlEscape(message) << "</p>";
    }

     html << "<div class=\"layout\">"
            << "<div class=\"card editor-card\">"
            << "<h2>Visualização ao vivo</h2>"
            << "<div class=\"editor-toolbar\">"
            << "<span class=\"chip\" id=\"preview-label\">Clique em dois pontos para desenhar</span>"
            << "<button type=\"button\" class=\"ghost\" onclick=\"resetLineToPreset()\">Redesenhar linha</button>"
            << "<button type=\"button\" class=\"ghost\" onclick=\"zoomPreview(0.62)\">Ampliar</button>"
            << "<button type=\"button\" class=\"ghost\" onclick=\"zoomPreview(0.78)\">Muito amplo</button>"
            << "<button type=\"button\" class=\"ghost\" onclick=\"zoomPreview(0.90)\">Tela quase cheia</button>"
            << "</div>"
            << "<div class=\"preview-stage\" id=\"preview-stage\">"
            << "<img id=\"preview-image\" src=\"/stream\" alt=\"Stream MJPEG\"/>"
            << "<canvas id=\"line-canvas\"></canvas>"
            << "</div>"
            << "<p class=\"hint\">Clique no preview em dois pontos para posicionar a linha virtual. O "
                "overlay é desenhado sobre o stream em tempo real. Se quiser, use os botões de zoom para "
                "deixar a imagem maior antes de marcar o trigger.</p>"
            << "</div>"
            << "<form method=\"POST\" action=\"/config\" class=\"card\" id=\"config-form\" onsubmit=\"normalizeForm(this)\">"
            << "<h2>Menu rápido</h2>"
            << "<div class=\"row3\">"
            << presetSelect("source_preset", "source", "Fonte rápida", cfg.source, videoPresets(),
                                 "Inclui o vídeo anexado quando ele existe no workspace.")
            << presetSelect("model_preset", "model_path", "Modelo de veículos", cfg.model_path,
                                 vehicleModelPresets(), "Use os modelos ONNX pré-treinados como base.")
            << presetSelect("axle_model_preset", "axle_model_path", "Modelo de eixos",
                                 cfg.axle_model_path, axleModelPresets(),
                                 "O estágio 2 continua ativo no cruzamento da linha.")
            << "</div>"
            << "<div class=\"actions\">"
             << "<button type=\"button\" class=\"ghost\" onclick=\"setField('source', '" << kAttachedVideo
            << "')\">Usar vídeo anexado</button>"
            << "<button type=\"button\" class=\"ghost\" onclick=\"setField('source', 'synthetic')\">Synthetic</button>"
            << "<button type=\"button\" class=\"ghost\" onclick=\"setField('model_path', 'models/vehicles.onnx')\">Modelo veículos</button>"
            << "<button type=\"button\" class=\"ghost\" onclick=\"setField('axle_model_path', 'models/axles.onnx')\">Modelo eixos</button>"
            << "</div>"
            << "<p class=\"hint\">Escolha um preset ou edite os campos abaixo para um caminho customizado.</p>"
            << textField("source", "Fonte (RTSP / caminho de vídeo / 'synthetic')", cfg.source)
            << textField("model_path", "Modelo de veículos (.onnx)", cfg.model_path)
            << "<div class=\"row2\">" << numField("conf", "Confiança mínima", cfg.conf, "0.01")
            << numField("nms", "IoU NMS", cfg.nms, "0.01") << "</div>"
            << "<div class=\"row2\">" << numField("imgsz", "Tamanho letterbox", cfg.imgsz, "1")
            << numField("threads", "Threads ORT", cfg.threads, "1") << "</div>"
            << "<p class=\"label\">Linha virtual de contagem (x1,y1 → x2,y2)</p>"
            << "<div class=\"row2\">" << numField("line_x1", "x1", cfg.line.p1.x, "0.01")
            << numField("line_y1", "y1", cfg.line.p1.y, "0.01") << "</div>"
            << "<div class=\"row2\">" << numField("line_x2", "x2", cfg.line.p2.x, "0.01")
            << numField("line_y2", "y2", cfg.line.p2.y, "0.01") << "</div>"
            << "<div class=\"row2\">"
            << numField("reconnect_ms", "Reconexão RTSP (ms)", cfg.reconnect_ms, "1")
            << numField("read_timeout_ms", "Timeout leitura (ms)", cfg.read_timeout_ms, "1")
            << "</div>"
            << "<hr/><h2>Estágio 2 — eixos por veículo</h2>"
            << "<label class=\"label\"><input type=\"checkbox\" name=\"axle_enabled\" value=\"1\""
            << (cfg.axle_enabled ? " checked" : "")
            << "/> Habilitar detector de eixos no cruzamento</label>"
            << textField("axle_model_path", "Modelo de eixos (.onnx)", cfg.axle_model_path)
            << "<div class=\"row2\">" << numField("axle_conf", "Conf. eixos", cfg.axle_conf, "0.01")
            << numField("axle_nms", "NMS eixos", cfg.axle_nms, "0.01") << "</div>"
            << "<div class=\"row2\">" << numField("axle_imgsz", "Letterbox eixos", cfg.axle_imgsz, "1")
            << numField("axle_crop_margin", "Margem do recorte (0–1)", cfg.axle_crop_margin, "0.01")
            << "</div>"
            << "<button type=\"submit\">Aplicar e recarregar</button>"
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
         << "\"axle_model_path\":\"" << jsonEscape(cfg.axle_model_path) << "\","
         << "\"axle_enabled\":" << (cfg.axle_enabled ? "true" : "false") << ","
         << "\"is_live\":" << (st.is_live ? "true" : "false") << ","
         << "\"total_count\":" << st.total_count << ","
         << "\"vehicle_count\":" << st.vehicle_count << ","
         << "\"axle_count\":" << st.axle_count << ","
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
