#include "contador/config_store.hpp"
#include "contador/pipeline_controller.hpp"
#include "contador/stream_server.hpp"
#include "contador/utils.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

namespace {

std::atomic<bool> g_running{true};

void onSignal(int) {
    g_running.store(false, std::memory_order_relaxed);
}

bool parseLine(const std::string& s, contador::CountLine& line) {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    if (std::sscanf(s.c_str(), "%f,%f,%f,%f", &x1, &y1, &x2, &y2) != 4) {
        return false;
    }
    line = {{x1, y1}, {x2, y2}};
    return true;
}

void printUsage(const char* argv0) {
    std::cout
        << "Uso: " << argv0 << " [opções]\n"
        << "  --source <uri>         RTSP/MP4/índice, ou 'synthetic' (padrão)\n"
        << "  --port <n>             Porta MJPEG/painel web (padrão: 8080)\n"
        << "  --model <path>         Modelo YOLO de veículos .onnx (padrão: models/vehicles.onnx)\n"
        << "  --conf <f>             Confiança mínima (padrão: 0.45)\n"
        << "  --nms <f>              IoU NMS (padrão: 0.45)\n"
        << "  --imgsz <n>            Letterbox YOLO (padrão: 640)\n"
        << "  --threads <n>          Threads IntraOp ORT (padrão: 2)\n"
        << "  --line x1,y1,x2,y2     Linha virtual diagonal\n"
        << "  --reconnect-ms <n>     Intervalo de reconexão RTSP (padrão: 5000)\n"
        << "  --read-timeout-ms <n>  Timeout de frame FFmpeg (padrão: 5000)\n"
        << "  --axle-model <path>    Modelo YOLO de eixos .onnx (padrão: models/axles.onnx)\n"
        << "  --axle-conf <f>        Confiança do estágio 2 (padrão: 0.35)\n"
        << "  --axle-nms <f>         NMS do estágio 2 (padrão: 0.45)\n"
        << "  --axle-imgsz <n>       Letterbox do estágio 2 (padrão: 224)\n"
        << "  --no-axle              Desabilita o detector de eixos (só conta veículos)\n"
        << "  --config <path>        Arquivo de config persistida (padrão: config/settings.json)\n"
        << "                         Carregado como base; flags explícitas acima sobrescrevem.\n"
        << "  --once                 Encerra automaticamente ao concluir uma fonte não-live\n"
        << "                         (arquivo de vídeo) e imprime o relatório final.\n"
        << "  --help\n"
        << "\n"
        << "Telas web (com o processo rodando): /  (painel) · /config (configuração, hot-reload)\n"
        << "                                     /stream (MJPEG) · /api/status (JSON)\n"
        << "API de job: POST /api/count com {\"video\":\"arquivo.mp4\", ...config...}\n";
}

}  // namespace

int main(int argc, char** argv) {
    // --config é lido num pré-scan pois precisa ser resolvido antes das demais flags.
    std::string config_path = "config/settings.json";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    contador::PipelineConfig cfg;  // defaults
    if (contador::loadConfig(config_path, cfg)) {
        contador::LogInfo("main") << "Config carregada de " << config_path;
    }

    bool once_mode = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                contador::LogError("main") << "Falta valor para " << name;
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--config") {
            need("--config");  // valor já resolvido no pré-scan; apenas consome o argumento
        } else if (arg == "--once") {
            once_mode = true;
        } else if (arg == "--source") {
            cfg.source = need("--source");
        } else if (arg == "--port") {
            cfg.port = static_cast<uint16_t>(std::stoi(need("--port")));
        } else if (arg == "--model") {
            cfg.model_path = need("--model");
        } else if (arg == "--conf") {
            cfg.conf = std::stof(need("--conf"));
        } else if (arg == "--nms") {
            cfg.nms = std::stof(need("--nms"));
        } else if (arg == "--imgsz") {
            cfg.imgsz = std::stoi(need("--imgsz"));
        } else if (arg == "--threads") {
            cfg.threads = std::stoi(need("--threads"));
        } else if (arg == "--reconnect-ms") {
            cfg.reconnect_ms = std::stoi(need("--reconnect-ms"));
        } else if (arg == "--read-timeout-ms") {
            cfg.read_timeout_ms = std::stoi(need("--read-timeout-ms"));
        } else if (arg == "--line") {
            contador::CountLine parsed;
            if (!parseLine(need("--line"), parsed)) {
                contador::LogError("main") << "Formato inválido para --line (use x1,y1,x2,y2)";
                return 1;
            }
            cfg.line = parsed;
        } else if (arg == "--axle-model") {
            cfg.axle_model_path = need("--axle-model");
            cfg.axle_enabled = true;
        } else if (arg == "--axle-conf") {
            cfg.axle_conf = std::stof(need("--axle-conf"));
        } else if (arg == "--axle-nms") {
            cfg.axle_nms = std::stof(need("--axle-nms"));
        } else if (arg == "--axle-imgsz") {
            cfg.axle_imgsz = std::stoi(need("--axle-imgsz"));
        } else if (arg == "--no-axle") {
            cfg.axle_enabled = false;
        } else {
            contador::LogError("main") << "Argumento desconhecido: " << arg;
            printUsage(argv[0]);
            return 1;
        }
    }

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    contador::LogInfo("main") << "Iniciando contador_eixo";

    contador::PipelineController controller;
    controller.bindRunningFlag(&g_running);

    contador::StreamServer server;
    controller.setFrameSink([&server](const cv::Mat& frame) { server.publish(frame); });
    server.attachController(&controller, config_path);

    controller.start(cfg);

    if (!server.start(cfg.port)) {
        contador::LogError("main") << "Falha ao iniciar StreamServer na porta " << cfg.port;
        controller.stop();
        return 1;
    }

    contador::LogInfo("main") << "Painel ativo — http://0.0.0.0:" << cfg.port
                              << "/  (Config: /config · Stream: /stream · Status: /api/status)";

    if (once_mode) {
        contador::LogInfo("main") << "--once: aguardando conclusão da fonte não-live...";
        while (g_running.load(std::memory_order_relaxed) && !controller.isTerminalForOnce()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        const auto st = controller.status();
        contador::LogInfo("main") << "Relatório --once: estado=" << contador::toString(st.state)
                                  << " | veiculos=" << st.vehicle_count
                                  << " | eixos=" << st.axle_count
                                  << " | frames=" << st.frames_processed << " | duração="
                                  << st.elapsed_sec << "s";
        if (!st.error_message.empty()) {
            contador::LogWarn("main") << "Mensagem: " << st.error_message;
        }
    } else {
        while (g_running.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    controller.stop();
    server.stop();
    contador::LogInfo("main") << "Encerrado.";
    return 0;
}
