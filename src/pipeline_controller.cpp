#include "contador/pipeline_controller.hpp"
#include "contador/logger.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

namespace contador {

const char* toString(PipelineState state) {
    switch (state) {
        case PipelineState::Idle:
            return "idle";
        case PipelineState::Starting:
            return "starting";
        case PipelineState::Running:
            return "running";
        case PipelineState::Finished:
            return "finished";
        case PipelineState::Error:
            return "error";
        case PipelineState::Stopped:
            return "stopped";
    }
    return "unknown";
}

PipelineController::PipelineController() = default;

PipelineController::~PipelineController() {
    stop();
}

void PipelineController::setFrameSink(FrameSink sink) {
    std::lock_guard<std::mutex> lock(mutex_);
    frame_sink_ = std::move(sink);
}

void PipelineController::start(const PipelineConfig& cfg) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_cfg_ = cfg;
        pending_cfg_ = PipelineConfig{};
        has_pending_ = false;
        status_ = PipelineStatus{};
        status_.source = cfg.source;
        status_.model_path = cfg.model_path;
    }
    stop_requested_.store(false, std::memory_order_relaxed);
    running_.store(true, std::memory_order_relaxed);
    worker_ = std::thread(&PipelineController::workerLoop, this);
}

void PipelineController::reconfigure(const PipelineConfig& cfg) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_cfg_ = cfg;
    has_pending_ = true;
    if (source_) {
        source_->requestStop();
    }
}

void PipelineController::stop() {
    if (!running_.exchange(false, std::memory_order_relaxed)) {
        return;
    }
    stop_requested_.store(true, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (source_) {
            source_->requestStop();
        }
    }
    if (worker_.joinable()) {
        worker_.join();
    }
}

PipelineConfig PipelineController::config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_cfg_;
}

PipelineStatus PipelineController::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

bool PipelineController::isTerminalForOnce() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_.state == PipelineState::Finished || status_.state == PipelineState::Error ||
           status_.state == PipelineState::Stopped;
}

bool PipelineController::openPipeline(const PipelineConfig& cfg, std::string& error) {
    VideoSourceOptions vs_opts;
    vs_opts.reconnect_delay = std::chrono::milliseconds(cfg.reconnect_ms);
    vs_opts.read_timeout_ms = cfg.read_timeout_ms;
    vs_opts.open_timeout_ms = std::max(vs_opts.open_timeout_ms, cfg.read_timeout_ms);

    auto new_source = createVideoSource(cfg.source, vs_opts);
    new_source->bindRunningFlag(app_running_flag_);
    if (!new_source->open()) {
        error = "Não foi possível abrir a fonte de vídeo: " + cfg.source;
        return false;
    }

    auto new_detector = std::make_unique<Detector>();
    if (!new_detector->load(cfg.model_path, cfg.conf, cfg.nms, cfg.imgsz, cfg.threads)) {
        error = "Falha ao carregar modelo: " + cfg.model_path;
        new_source->release();
        return false;
    }

    std::unique_ptr<Detector> new_axle;
    if (cfg.axle_enabled && !cfg.axle_model_path.empty()) {
        new_axle = std::make_unique<Detector>();
        if (!new_axle->load(cfg.axle_model_path, cfg.axle_conf, cfg.axle_nms, cfg.axle_imgsz,
                            cfg.threads)) {
            LogWarn("PipelineController")
                << "Falha ao carregar modelo de eixos: " << cfg.axle_model_path
                << " — seguindo só com contagem de veículos";
            new_axle.reset();
        } else {
            LogInfo("PipelineController")
                << "Estágio 2 (eixos) ativo: " << cfg.axle_model_path;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        source_ = std::move(new_source);
        detector_ = std::move(new_detector);
        axle_detector_ = std::move(new_axle);
        tracker_ = std::make_unique<TrackerCounter>(cfg.line);
    }
    return true;
}

void PipelineController::closePipeline() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (source_) {
        source_->requestStop();
        source_->release();
        source_.reset();
    }
    detector_.reset();
    axle_detector_.reset();
    tracker_.reset();
}

void PipelineController::waitForReconfigureOrStop() {
    while (!stop_requested_.load(std::memory_order_relaxed)) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (has_pending_) {
                current_cfg_ = pending_cfg_;
                has_pending_ = false;
                return;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
}

void PipelineController::workerLoop() {
    while (!stop_requested_.load(std::memory_order_relaxed)) {
        PipelineConfig cfg;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cfg = current_cfg_;
            status_.state = PipelineState::Starting;
            status_.source = cfg.source;
            status_.model_path = cfg.model_path;
            status_.error_message.clear();
            status_.total_count = 0;
            status_.vehicle_count = 0;
            status_.axle_count = 0;
            status_.frames_processed = 0;
            status_.elapsed_sec = 0.0;
        }

        std::string error;
        if (!openPipeline(cfg, error)) {
            LogError("PipelineController") << error;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                status_.state = PipelineState::Error;
                status_.error_message = error;
            }
            waitForReconfigureOrStop();
            continue;
        }

        bool is_live = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_live = source_->isLive();
            status_.state = PipelineState::Running;
            status_.is_live = is_live;
        }
        LogInfo("PipelineController") << "Pipeline rodando: fonte=" << cfg.source
                                      << " modelo=" << cfg.model_path
                                      << (is_live ? " [live]" : " [arquivo]");

        const auto started_at = std::chrono::steady_clock::now();
        double source_fps = 25.0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (source_) {
                source_fps = source_->fps();
            }
        }
        const auto frame_period = std::chrono::duration<double>(1.0 / std::max(1.0, source_fps));

        LoopExit exit_reason = LoopExit::LiveInterrupted;
        while (true) {
            if (stop_requested_.load(std::memory_order_relaxed)) {
                exit_reason = LoopExit::Stop;
                break;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (has_pending_) {
                    exit_reason = LoopExit::Pending;
                    break;
                }
            }

            const auto t0 = std::chrono::steady_clock::now();
            cv::Mat frame;
            const bool read_ok = source_->read(frame);
            if (!read_ok) {
                exit_reason = is_live ? LoopExit::LiveInterrupted : LoopExit::Eof;
                break;
            }

            const auto detections = detector_->detect(frame);
            std::vector<CrossingEvent> crossings;
            tracker_->update(detections, &crossings);

            // Estágio 2: para cada veículo que acabou de cruzar a linha, recorta
            // a bbox (com margem) e roda o detector de eixos uma vez.
            for (const auto& ev : crossings) {
                int n_axles = 0;
                if (axle_detector_ && axle_detector_->isLoaded()) {
                    const float margin = std::max(0.f, cfg.axle_crop_margin);
                    const int pad_x = static_cast<int>(ev.box.width * margin);
                    const int pad_y = static_cast<int>(ev.box.height * margin);
                    cv::Rect crop = ev.box;
                    crop.x = std::max(0, crop.x - pad_x);
                    crop.y = std::max(0, crop.y - pad_y);
                    crop.width = std::min(frame.cols - crop.x, crop.width + 2 * pad_x);
                    crop.height = std::min(frame.rows - crop.y, crop.height + 2 * pad_y);

                    if (crop.width >= 8 && crop.height >= 8) {
                        const cv::Mat roi = frame(crop);
                        const auto axle_dets = axle_detector_->detect(roi);
                        n_axles = static_cast<int>(axle_dets.size());
                        for (const auto& ad : axle_dets) {
                            cv::Rect mapped = ad.box;
                            mapped.x += crop.x;
                            mapped.y += crop.y;
                            cv::rectangle(frame, mapped, cv::Scalar(0, 255, 255), 2);
                        }
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    status_.axle_count += n_axles;
                }
                LogInfo("PipelineController")
                    << "veiculo #" << ev.track_id << " cruzou linha -> " << n_axles << " eixos";
            }

            int axle_overlay = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                axle_overlay = status_.axle_count;
            }
            tracker_->drawOverlay(frame, axle_detector_ ? axle_overlay : -1);
            for (const auto& d : detections) {
                cv::rectangle(frame, d.box, cv::Scalar(255, 180, 0), 1);
                cv::putText(frame,
                            d.label + " " + std::to_string(d.confidence).substr(0, 4),
                            cv::Point(d.box.x, std::max(15, d.box.y - 4)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 180, 0), 1);
            }

            FrameSink sink;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                sink = frame_sink_;
                status_.vehicle_count = tracker_->totalCount();
                status_.total_count = status_.vehicle_count;
                status_.frames_processed += 1;
                status_.elapsed_sec =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at)
                        .count();
            }
            if (sink) {
                sink(frame);
            }

            const auto elapsed = std::chrono::steady_clock::now() - t0;
            if (elapsed < frame_period) {
                std::this_thread::sleep_for(frame_period - elapsed);
            }
        }

        int final_count = 0;
        int final_axles = 0;
        uint64_t final_frames = 0;
        double final_elapsed = 0.0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            final_count = status_.vehicle_count;
            final_axles = status_.axle_count;
            final_frames = status_.frames_processed;
            final_elapsed = status_.elapsed_sec;
        }
        closePipeline();

        if (exit_reason == LoopExit::Stop) {
            std::lock_guard<std::mutex> lock(mutex_);
            status_.state = PipelineState::Stopped;
            continue;  // condição do while externo já é falsa — sai no topo
        }

        if (exit_reason == LoopExit::Pending) {
            std::lock_guard<std::mutex> lock(mutex_);
            current_cfg_ = pending_cfg_;
            has_pending_ = false;
            LogInfo("PipelineController") << "Reconfigurado — reiniciando pipeline.";
            continue;
        }

        if (exit_reason == LoopExit::Eof) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                status_.state = PipelineState::Finished;
            }
            LogInfo("PipelineController")
                << "Fim do arquivo — veiculos: " << final_count << " | eixos: " << final_axles
                << " | frames: " << final_frames << " | duração: " << final_elapsed << "s";
        } else {  // LiveInterrupted sem stop/pending — condição rara; trata como erro transitório
            std::lock_guard<std::mutex> lock(mutex_);
            status_.state = PipelineState::Error;
            status_.error_message = "Fonte encerrada inesperadamente";
        }

        waitForReconfigureOrStop();
    }
}

}  // namespace contador
