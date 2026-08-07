#pragma once

#include "contador/config_store.hpp"
#include "contador/detector.hpp"
#include "contador/tracker_counter.hpp"
#include "contador/video_source.hpp"

#include <opencv2/core.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace contador {

enum class PipelineState {
    Idle,      ///< Ainda não iniciado.
    Starting,  ///< Abrindo fonte/modelo.
    Running,   ///< Processando frames normalmente.
    Finished,  ///< Fonte não-live chegou ao EOF; aguardando reconfiguração.
    Error,     ///< Falha ao abrir fonte/modelo; aguardando reconfiguração.
    Stopped,   ///< Encerrado definitivamente (shutdown do processo).
};

const char* toString(PipelineState state);

struct PipelineStatus {
    PipelineState state{PipelineState::Idle};
    std::string source;
    std::string model_path;
    bool is_live{false};
    int total_count{0};       ///< Alias de vehicle_count (compatibilidade).
    int vehicle_count{0};     ///< Veículos que cruzaram a linha.
    int axle_count{0};        ///< Soma de eixos detectados nos crossings.
    uint64_t frames_processed{0};
    double elapsed_sec{0.0};
    std::string error_message;
};

/**
 * Dono do ciclo de vida do pipeline (VideoSource → Detector → TrackerCounter),
 * executado em thread própria. Permite reconfiguração em tempo real
 * (hot-reload) via reconfigure() sem derrubar o processo/StreamServer.
 *
 * Quando a fonte é um arquivo de vídeo e chega ao EOF, o controller entra em
 * estado `Finished` (com o relatório final) em vez de finalizar tudo — o
 * último frame publicado permanece disponível em /stream até uma nova
 * reconfiguração (via tela de configuração ou próxima chamada de start).
 */
class PipelineController {
public:
    using FrameSink = std::function<void(const cv::Mat&)>;

    PipelineController();
    ~PipelineController();

    PipelineController(const PipelineController&) = delete;
    PipelineController& operator=(const PipelineController&) = delete;

    /** Callback invocado a cada frame processado (ex.: StreamServer::publish). */
    void setFrameSink(FrameSink sink);

    /** Observa flag externo (ex.: g_running do main) para sair rápido de fontes live no shutdown. */
    void bindRunningFlag(std::atomic<bool>* flag) { app_running_flag_ = flag; }

    /** Inicia a thread de trabalho com a config inicial. */
    void start(const PipelineConfig& cfg);

    /** Solicita hot-reload: aplicado assim que o worker notar (interrompe a fonte atual). */
    void reconfigure(const PipelineConfig& cfg);

    /** Encerra a thread de trabalho definitivamente (shutdown do processo). */
    void stop();

    PipelineConfig config() const;
    PipelineStatus status() const;

    /** True quando o pipeline chegou a um estado terminal útil para `--once`. */
    bool isTerminalForOnce() const;

private:
    enum class LoopExit { Stop, Pending, Eof, LiveInterrupted };

    void workerLoop();
    bool openPipeline(const PipelineConfig& cfg, std::string& error);
    void closePipeline();
    /** Bloqueia até uma reconfiguração (aplica em current_cfg_) ou stop. */
    void waitForReconfigureOrStop();

    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool>* app_running_flag_{nullptr};

    mutable std::mutex mutex_;
    PipelineConfig current_cfg_;
    PipelineConfig pending_cfg_;
    bool has_pending_{false};

    PipelineStatus status_;
    FrameSink frame_sink_;

    std::unique_ptr<IVideoSource> source_;
    std::unique_ptr<Detector> detector_;
    std::unique_ptr<Detector> axle_detector_;
    std::unique_ptr<TrackerCounter> tracker_;
};

}  // namespace contador
