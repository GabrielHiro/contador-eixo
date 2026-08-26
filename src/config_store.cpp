#include "contador/config_store.hpp"
#include "contador/logger.hpp"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace contador {
namespace {

std::string readWholeFile(const std::string& path, bool& ok) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        ok = false;
        return {};
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    ok = true;
    return buffer.str();
}

void ensureParentDir(const std::string& path) {
    const auto slash = path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return;
    }
    const std::string dir = path.substr(0, slash);
    if (dir.empty()) {
        return;
    }
    // mkdir raso — config/ já deve existir no repositório, mas não custa garantir.
    ::mkdir(dir.c_str(), 0755);
}

/** Busca "key": <valor> no JSON plano e devolve o valor cru (sem aspas para strings). */
bool findRawValue(const std::string& text, const std::string& key, std::string& raw) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos = text.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return false;
    }
    ++pos;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    if (pos >= text.size()) {
        return false;
    }

    if (text[pos] == '"') {
        size_t end = pos + 1;
        std::string value;
        while (end < text.size() && text[end] != '"') {
            if (text[end] == '\\' && end + 1 < text.size()) {
                value += text[end + 1];
                end += 2;
                continue;
            }
            value += text[end];
            ++end;
        }
        raw = value;
        return true;
    }

    size_t end = pos;
    while (end < text.size() && text[end] != ',' && text[end] != '}' && text[end] != '\n' &&
           text[end] != '\r') {
        ++end;
    }
    raw = text.substr(pos, end - pos);
    while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.back()))) {
        raw.pop_back();
    }
    return true;
}

void loadString(const std::string& text, const std::string& key, std::string& target) {
    std::string raw;
    if (findRawValue(text, key, raw)) {
        target = raw;
    }
}

void loadFloat(const std::string& text, const std::string& key, float& target) {
    std::string raw;
    if (findRawValue(text, key, raw)) {
        try {
            target = std::stof(raw);
        } catch (...) {
            LogWarn("ConfigStore") << "Valor inválido para " << key << ": " << raw;
        }
    }
}

void loadInt(const std::string& text, const std::string& key, int& target) {
    std::string raw;
    if (findRawValue(text, key, raw)) {
        try {
            target = std::stoi(raw);
        } catch (...) {
            LogWarn("ConfigStore") << "Valor inválido para " << key << ": " << raw;
        }
    }
}

void loadBool(const std::string& text, const std::string& key, bool& target) {
    std::string raw;
    if (findRawValue(text, key, raw)) {
        for (char& c : raw) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (raw == "true" || raw == "1" || raw == "yes") {
            target = true;
        } else if (raw == "false" || raw == "0" || raw == "no") {
            target = false;
        } else {
            LogWarn("ConfigStore") << "Valor inválido para " << key << ": " << raw;
        }
    }
}

}  // namespace

std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (const char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += c;
        }
    }
    return out;
}

bool loadConfigJson(const std::string& text, PipelineConfig& cfg) {
    loadString(text, "source", cfg.source);
    loadString(text, "model_path", cfg.model_path);
    loadFloat(text, "conf", cfg.conf);
    loadFloat(text, "nms", cfg.nms);
    loadInt(text, "imgsz", cfg.imgsz);
    loadInt(text, "threads", cfg.threads);
    loadFloat(text, "line_x1", cfg.line.p1.x);
    loadFloat(text, "line_y1", cfg.line.p1.y);
    loadFloat(text, "line_x2", cfg.line.p2.x);
    loadFloat(text, "line_y2", cfg.line.p2.y);
    loadInt(text, "reconnect_ms", cfg.reconnect_ms);
    loadInt(text, "read_timeout_ms", cfg.read_timeout_ms);

    int port_tmp = cfg.port;
    loadInt(text, "port", port_tmp);
    cfg.port = static_cast<uint16_t>(port_tmp);

    loadBool(text, "axle_enabled", cfg.axle_enabled);
    loadString(text, "axle_model_path", cfg.axle_model_path);
    loadFloat(text, "axle_conf", cfg.axle_conf);
    loadFloat(text, "axle_nms", cfg.axle_nms);
    loadInt(text, "axle_imgsz", cfg.axle_imgsz);
    loadFloat(text, "axle_crop_margin", cfg.axle_crop_margin);

    return true;
}

bool loadConfig(const std::string& path, PipelineConfig& cfg) {
    bool ok = false;
    const std::string text = readWholeFile(path, ok);
    return ok && loadConfigJson(text, cfg);
}

bool saveConfig(const std::string& path, const PipelineConfig& cfg) {
    ensureParentDir(path);

    std::ostringstream json;
    json << "{\n"
         << "  \"source\": \"" << jsonEscape(cfg.source) << "\",\n"
         << "  \"model_path\": \"" << jsonEscape(cfg.model_path) << "\",\n"
         << "  \"conf\": " << cfg.conf << ",\n"
         << "  \"nms\": " << cfg.nms << ",\n"
         << "  \"imgsz\": " << cfg.imgsz << ",\n"
         << "  \"threads\": " << cfg.threads << ",\n"
         << "  \"line_x1\": " << cfg.line.p1.x << ",\n"
         << "  \"line_y1\": " << cfg.line.p1.y << ",\n"
         << "  \"line_x2\": " << cfg.line.p2.x << ",\n"
         << "  \"line_y2\": " << cfg.line.p2.y << ",\n"
         << "  \"reconnect_ms\": " << cfg.reconnect_ms << ",\n"
         << "  \"read_timeout_ms\": " << cfg.read_timeout_ms << ",\n"
         << "  \"port\": " << cfg.port << ",\n"
         << "  \"axle_enabled\": " << (cfg.axle_enabled ? "true" : "false") << ",\n"
         << "  \"axle_model_path\": \"" << jsonEscape(cfg.axle_model_path) << "\",\n"
         << "  \"axle_conf\": " << cfg.axle_conf << ",\n"
         << "  \"axle_nms\": " << cfg.axle_nms << ",\n"
         << "  \"axle_imgsz\": " << cfg.axle_imgsz << ",\n"
         << "  \"axle_crop_margin\": " << cfg.axle_crop_margin << "\n"
         << "}\n";

    std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        LogError("ConfigStore") << "Não foi possível gravar " << path;
        return false;
    }
    file << json.str();
    return file.good();
}

}  // namespace contador
