#include "contador/detector.hpp"
#include "contador/logger.hpp"
#include "contador/stream_server.hpp"
#include "contador/tracker_counter.hpp"
#include "contador/video_source.hpp"

#include <opencv2/imgproc.hpp>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

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
        << "  --port <n>             Porta MJPEG (padrão: 8080)\n"
        << "  --model <path>         Modelo YOLO .onnx\n"
        << "  --conf <f>             Confiança mínima (padrão: 0.45)\n"
        << "  --nms <f>              IoU NMS (padrão: 0.45)\n"
        << "  --imgsz <n>            Letterbox YOLO (padrão: 640)\n"
        << "  --threads <n>          Threads IntraOp ORT (padrão: 2)\n"
        << "  --line x1,y1,x2,y2     Linha virtual diagonal\n"
        << "  --reconnect-ms <n>     Intervalo de reconexão RTSP (padrão: 5000)\n"
        << "  --read-timeout-ms <n>  Timeout de frame FFmpeg (padrão: 5000)\n"
        << "  --help\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string source_uri = "synthetic";
    uint16_t port = 8080;
    std::string model_path = "models/wheels.onnx";
    float conf = 0.45f;
    float nms = 0.45f;
    int imgsz = 640;
    int threads = 2;
    contador::VideoSourceOptions vs_opts;

    // Linha diagonal para 1280x720 — ajuste no poste via --line
    contador::CountLine count_line{{180.f, 160.f}, {1100.f, 620.f}};

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
        } else if (arg == "--source") {
            source_uri = need("--source");
        } else if (arg == "--port") {
            port = static_cast<uint16_t>(std::stoi(need("--port")));
        } else if (arg == "--model") {
            model_path = need("--model");
        } else if (arg == "--conf") {
            conf = std::stof(need("--conf"));
        } else if (arg == "--nms") {
            nms = std::stof(need("--nms"));
        } else if (arg == "--imgsz") {
            imgsz = std::stoi(need("--imgsz"));
        } else if (arg == "--threads") {
            threads = std::stoi(need("--threads"));
        } else if (arg == "--reconnect-ms") {
            vs_opts.reconnect_delay =
                std::chrono::milliseconds(std::stoi(need("--reconnect-ms")));
        } else if (arg == "--read-timeout-ms") {
            vs_opts.read_timeout_ms = std::stoi(need("--read-timeout-ms"));
            vs_opts.open_timeout_ms = std::max(vs_opts.open_timeout_ms, vs_opts.read_timeout_ms);
        } else if (arg == "--line") {
            contador::CountLine parsed;
            if (!parseLine(need("--line"), parsed)) {
                contador::LogError("main") << "Formato inválido para --line (use x1,y1,x2,y2)";
                return 1;
            }
            count_line = parsed;
        } else {
            contador::LogError("main") << "Argumento desconhecido: " << arg;
            printUsage(argv[0]);
            return 1;
        }
    }

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    contador::LogInfo("main") << "Iniciando contador_eixo";

    // 1) VideoSource — RTSP com reconexão automática; arquivo sem retry
    auto source = contador::createVideoSource(source_uri, vs_opts);
    source->bindRunningFlag(&g_running);
    if (!source->open()) {
        contador::LogError("main") << "Não foi possível abrir a fonte de vídeo (shutdown?)";
        return 1;
    }

    // 2+3) Detector ONNX
    contador::Detector detector;
    if (!detector.load(model_path, conf, nms, imgsz, threads)) {
        contador::LogError("main") << "Falha ao carregar modelo: " << model_path;
        return 1;
    }

    // 4) Tracker
    contador::TrackerCounter tracker(count_line);
    contador::LogInfo("Tracker") << "Linha virtual: (" << count_line.p1.x << ","
                                 << count_line.p1.y << ") → (" << count_line.p2.x << ","
                                 << count_line.p2.y << ")";

    // 5) StreamServer
    contador::StreamServer server;
    if (!server.start(port)) {
        contador::LogError("main") << "Falha ao iniciar StreamServer na porta " << port;
        return 1;
    }

    contador::LogInfo("main") << "Pipeline ativo — http://0.0.0.0:" << port
                              << "/  (SIGTERM/Ctrl+C para sair)";

    const auto frame_period =
        std::chrono::duration<double>(1.0 / std::max(1.0, source->fps()));

    while (g_running.load(std::memory_order_relaxed)) {
        const auto t0 = std::chrono::steady_clock::now();

        cv::Mat frame;
        if (!source->read(frame)) {
            // Live: só chega aqui após requestStop / g_running=false
            // File/synthetic: EOF ou stop
            if (g_running.load(std::memory_order_relaxed)) {
                contador::LogInfo("main") << "Fonte esgotada — encerrando loop";
            } else {
                contador::LogInfo("main") << "Shutdown solicitado durante leitura";
            }
            break;
        }

        const auto detections = detector.detect(frame);
        tracker.update(detections);
        tracker.drawOverlay(frame);

        for (const auto& d : detections) {
            cv::rectangle(frame, d.box, cv::Scalar(255, 180, 0), 1);
            cv::putText(frame,
                        d.label + " " + std::to_string(d.confidence).substr(0, 4),
                        cv::Point(d.box.x, std::max(15, d.box.y - 4)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 180, 0), 1);
        }

        server.publish(frame);

        const auto elapsed = std::chrono::steady_clock::now() - t0;
        if (elapsed < frame_period) {
            std::this_thread::sleep_for(frame_period - elapsed);
        }
    }

    source->requestStop();
    server.stop();
    source->release();
    contador::LogInfo("main") << "Encerrado. Total de eixos: " << tracker.totalCount();
    return 0;
}
