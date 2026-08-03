#pragma once

#include "contador/types.hpp"

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace contador {

/**
 * Detector YOLO genérico via ONNX Runtime (C++ API).
 *
 * Suporta saídas comuns:
 *  - YOLOv8/v9/v11: [1, 4+nc, N]
 *  - YOLOv5:        [1, N, 5+nc] (com objectness) ou [1, N, 4+nc]
 *
 * Letterbox interno (via PreProcessing) → inferência → NMS → boxes no frame original.
 */
class Detector {
public:
    Detector();
    ~Detector();

    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
    Detector(Detector&&) noexcept;
    Detector& operator=(Detector&&) noexcept;

    /**
     * Carrega modelo .onnx e prepara a sessão ORT.
     * @param model_path       caminho do .onnx
     * @param conf_threshold   score mínimo (após objectness×class, se aplicável)
     * @param nms_threshold    IoU do NMS
     * @param input_size       lado do letterbox quadrado (ex.: 640)
     * @param num_threads      threads IntraOp (edge: 2–4 tipicamente)
     */
    bool load(const std::string& model_path,
              float conf_threshold = 0.45f,
              float nms_threshold = 0.45f,
              int input_size = 640,
              int num_threads = 2);

    bool isLoaded() const;

    /** Classes filtradas (vazio = todas). Útil para modelo multi-classe focando em "wheel". */
    void setClassFilter(std::vector<int> class_ids);
    void setClassNames(std::vector<std::string> names);

    void setConfThreshold(float t);
    void setNmsThreshold(float t);

    /**
     * Roda inferência no frame nativo (ex.: 1280×720).
     * Retorna detecções já mapeadas para o sistema de coordenadas do frame original.
     */
    std::vector<Detection> detect(const cv::Mat& frame);

    int inputSize() const;
    const std::string& modelPath() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace contador
