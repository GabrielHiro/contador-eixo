#include "contador/pipeline_controller.hpp"
#include "contador/logger.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <thread>
#include <vector>
#include <sstream>

namespace contador {
namespace {

std::vector<std::string> splitNames(const std::string& value) {
    std::vector<std::string> names;
    std::stringstream stream(value);
    std::string name;
    while (std::getline(stream, name, ',')) {
        if (!name.empty()) names.push_back(name);
    }
    return names;
}

std::vector<int> splitIds(const std::string& value) {
    std::vector<int> ids;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        try {
            if (!item.empty()) ids.push_back(std::stoi(item));
        } catch (...) {
            // Ignora IDs inválidos para manter o serviço operando.
        }
    }
    return ids;
}

bool hasVehicleTypes(const PipelineConfig& cfg) {
    return splitNames(cfg.vehicle_class_names).size() >= 3;
}

int groupedAxles(const std::vector<Detection>& wheels,
                 const cv::Rect& vehicle_box,
                 float grouping_distance) {
    if (wheels.empty()) return 0;
    const bool longitudinal_x = vehicle_box.width >= vehicle_box.height;
    std::vector<float> positions;
    positions.reserve(wheels.size());
    for (const auto& wheel : wheels) {
        positions.push_back(longitudinal_x ? wheel.centroid.x : wheel.centroid.y);
    }
    std::sort(positions.begin(), positions.end());
    const float span = static_cast<float>(longitudinal_x ? vehicle_box.width : vehicle_box.height);
    const float threshold = std::max(4.f, span * std::max(0.01f, grouping_distance));
    int groups = 1;
    for (size_t i = 1; i < positions.size(); ++i) {
        if (positions[i] - positions[i - 1] > threshold) ++groups;
    }
    return std::clamp(groups, 2, 6);
}

}  // namespace

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
    new_detector->setClassNames(splitNames(cfg.vehicle_class_names));
    if (!cfg.vehicle_class_filter.empty()) {
        new_detector->setClassFilter(splitIds(cfg.vehicle_class_filter));
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
            new_axle->setClassFilter({cfg.wheel_class_id});
            new_axle->setClassNames({"wheel"});
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
        cv::VideoWriter output_writer;
        bool output_writer_warned = false;
        bool accepted_single_vehicle = false;
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

            auto detections = detector_->detect(frame);
            if (cfg.single_vehicle_mode && !detections.empty()) {
                const auto best = std::max_element(
                    detections.begin(), detections.end(),
                    [](const Detection& left, const Detection& right) {
                        return left.confidence < right.confidence;
                    });
                detections = {*best};
            }
            std::vector<CrossingEvent> crossings;
            tracker_->update(detections, &crossings);
            if (cfg.single_vehicle_mode && !accepted_single_vehicle && crossings.empty() &&
                !detections.empty() && detections.front().box.area() >= 10000) {
                const auto& detection = detections.front();
                tracker_->ensureSingleVehicleCounted();
                crossings.push_back(CrossingEvent{0, detection.box, detection.class_id,
                                                  detection.label});
            }

            // Estágio 2: para cada veículo que acabou de cruzar a linha, recorta
            // caminhões e agrupa rodas pela posição longitudinal em eixos.
            for (const auto& ev : crossings) {
                if (cfg.single_vehicle_mode && accepted_single_vehicle) continue;
                accepted_single_vehicle = true;
                int n_axles = 0;
                const bool typed_vehicle = hasVehicleTypes(cfg);
                if (ev.class_id == cfg.truck_class_id && cfg.truck_axles_override > 0) {
                    n_axles = std::clamp(cfg.truck_axles_override, 2, 10);
                }
                const bool direct_axle_rule = typed_vehicle &&
                                              (ev.class_id == cfg.light_vehicle_class_id ||
                                               ev.class_id == cfg.motorcycle_class_id);
                if (n_axles > 0) {
                    // Número físico informado pelo radar para esta configuração de câmera.
                } else if (direct_axle_rule) {
                    n_axles = ev.class_id == cfg.motorcycle_class_id
                                  ? std::clamp(cfg.motorcycle_axles, 2, 6)
                                  : std::clamp(cfg.light_vehicle_axles, 2, 6);
                } else if (axle_detector_ && axle_detector_->isLoaded()) {
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
                        n_axles = groupedAxles(axle_dets, ev.box, cfg.axle_group_distance);
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
                    << "veiculo #" << ev.track_id << " [" << ev.label << "] cruzou linha -> "
                    << n_axles << " eixos (rodas agrupadas)";
            }

            int axle_overlay = 0;
            if (cfg.single_vehicle_mode && accepted_single_vehicle) {
                tracker_->ensureSingleVehicleCounted();
            }
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

            if (!cfg.output_path.empty() && !output_writer.isOpened()) {
                std::filesystem::path output_file(cfg.output_path);
                std::error_code fs_error;
                std::filesystem::create_directories(output_file.parent_path(), fs_error);
                output_writer.open(cfg.output_path,
                                   cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                   std::max(1.0, source_fps), frame.size(), true);
                if (!output_writer.isOpened() && !output_writer_warned) {
                    LogWarn("PipelineController") << "Não foi possível gravar vídeo processado: "
                                                    << cfg.output_path;
                    output_writer_warned = true;
                }
            }
            if (output_writer.isOpened()) {
                output_writer.write(frame);
            }

            FrameSink sink;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                sink = frame_sink_;
                status_.vehicle_count = cfg.single_vehicle_mode
                                             ? (accepted_single_vehicle ? 1 : 0)
                                             : tracker_->totalCount();
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

        if (output_writer.isOpened()) {
            output_writer.release();
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
