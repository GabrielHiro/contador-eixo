#pragma once

#include <opencv2/core.hpp>

namespace contador {

/** Metadados do letterbox — necessários para remapear boxes ao frame original. */
struct LetterboxMeta {
    float scale{1.f};
    float pad_x{0.f};
    float pad_y{0.f};
    int input_w{640};
    int input_h{640};
    int orig_w{0};
    int orig_h{0};
};

/**
 * Pré-processamento leve para edge: apenas letterbox/resize para o tensor ONNX.
 * Homografia descartada de propósito (economia de CPU/RAM; YOLO opera na perspectiva natural).
 */
class PreProcessing {
public:
    PreProcessing() = default;
    explicit PreProcessing(int input_size);
    PreProcessing(int input_w, int input_h);

    void setInputSize(int input_w, int input_h);
    int inputWidth() const { return input_w_; }
    int inputHeight() const { return input_h_; }

    /**
     * Redimensiona mantendo aspect ratio e preenche com pad (114,114,114) até input_w×input_h.
     * @param frame  BGR nativo (ex.: 1280×720)
     * @param meta   preenchido com scale/pad para remapear detecções
     * @return imagem letterboxed BGR pronta para normalização NCHW
     */
    cv::Mat letterbox(const cv::Mat& frame, LetterboxMeta& meta) const;

    /** Converte box no espaço letterbox → coordenadas do frame original. */
    static cv::Rect2f mapBoxToOriginal(const cv::Rect2f& box, const LetterboxMeta& meta);

    /** Converte ponto no espaço letterbox → frame original. */
    static cv::Point2f mapPointToOriginal(cv::Point2f pt, const LetterboxMeta& meta);

private:
    int input_w_{640};
    int input_h_{640};
};

}  // namespace contador
