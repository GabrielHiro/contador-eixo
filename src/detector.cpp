#include "contador/detector.hpp"
#include "contador/logger.hpp"
#include "contador/preprocessing.hpp"

#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace contador {
namespace {

enum class YoloLayout {
    kChannelsFirst,  // [1, 4+nc, N]  — YOLOv8+
    kChannelsLast,   // [1, N, 4+nc] ou [1, N, 5+nc] — YOLOv5
};

struct ParsedOutput {
    YoloLayout layout{YoloLayout::kChannelsFirst};
    int64_t num_preds{0};
    int64_t num_attrs{0};  // 4 + nc  ou  5 + nc
    bool has_objectness{false};
};

ParsedOutput inferLayout(const std::vector<int64_t>& shape) {
    // Esperado rank 3: [batch, A, B]
    ParsedOutput p;
    if (shape.size() != 3) {
        throw std::runtime_error("Saída ONNX inesperada: rank != 3");
    }
    const int64_t a = shape[1];
    const int64_t b = shape[2];

    // Se a dimensão do meio é pequena (< 64), é típico [1, 4+nc, N]
    if (a < b && a < 64) {
        p.layout = YoloLayout::kChannelsFirst;
        p.num_attrs = a;
        p.num_preds = b;
        p.has_objectness = false;
    } else {
        p.layout = YoloLayout::kChannelsLast;
        p.num_preds = a;
        p.num_attrs = b;
        // YOLOv5 clássico: cx,cy,w,h,obj,classes...
        p.has_objectness = (b >= 6);
    }
    return p;
}

float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

}  // namespace

struct Detector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "contador_eixo"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::string model_path;
    std::string input_name;
    std::vector<std::string> output_names_storage;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    PreProcessing preprocess{640};
    float conf_threshold{0.45f};
    float nms_threshold{0.45f};
    int input_size{640};
    int num_threads{2};

    std::vector<int> class_filter;
    std::vector<std::string> class_names;

    // Buffer reutilizável (evita alloc por frame na borda)
    std::vector<float> input_tensor_values;
    std::vector<Ort::Float16_t> input_tensor_fp16;
    std::vector<float> output_scratch;  // conversão FP16→FP32 da saída
    ONNXTensorElementDataType input_dtype{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
    LetterboxMeta last_meta{};

    bool load(const std::string& path,
              float conf,
              float nms,
              int size,
              int threads) {
        model_path = path;
        conf_threshold = conf;
        nms_threshold = nms;
        input_size = size;
        num_threads = std::max(1, threads);
        preprocess.setInputSize(input_size, input_size);

        session_options.SetIntraOpNumThreads(num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        try {
            session = std::make_unique<Ort::Session>(env, path.c_str(), session_options);
        } catch (const Ort::Exception& e) {
            LogError("Detector") << "Falha ao carregar ONNX: " << e.what();
            session.reset();
            return false;
        }

        // Input
        {
            auto name = session->GetInputNameAllocated(0, allocator);
            input_name = name.get();
            input_names = {input_name.c_str()};

            // TypeInfo precisa permanecer vivo enquanto usamos TensorTypeAndShapeInfo
            Ort::TypeInfo type_info = session->GetInputTypeInfo(0);
            auto info = type_info.GetTensorTypeAndShapeInfo();
            input_dtype = info.GetElementType();
            auto shape = info.GetShape();
            // shape tipicamente [1, 3, H, W] — se H/W estáticos, alinhar input_size
            if (shape.size() == 4 && shape[2] > 0 && shape[3] > 0) {
                if (shape[2] != shape[3]) {
                    LogWarn("Detector") << "Input não-quadrado " << shape[3] << "x"
                                        << shape[2];
                }
                input_size = static_cast<int>(shape[3]);
                preprocess.setInputSize(static_cast<int>(shape[3]), static_cast<int>(shape[2]));
            }

            if (input_dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                input_dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                LogError("Detector") << "Tipo de input não suportado (use FP32 ou FP16), got="
                                     << static_cast<int>(input_dtype);
                session.reset();
                return false;
            }
        }

        // Outputs
        output_names_storage.clear();
        output_names.clear();
        const size_t n_out = session->GetOutputCount();
        for (size_t i = 0; i < n_out; ++i) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            output_names_storage.emplace_back(name.get());
        }
        for (auto& s : output_names_storage) {
            output_names.push_back(s.c_str());
        }

        const size_t n_elem =
            static_cast<size_t>(3 * preprocess.inputHeight() * preprocess.inputWidth());
        input_tensor_values.resize(n_elem);
        if (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            input_tensor_fp16.resize(n_elem);
        }

        const char* dtype_str =
            (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? "FP16" : "FP32";
        LogInfo("Detector") << "Modelo: " << path << " input=\"" << input_name << "\" "
                            << preprocess.inputWidth() << "x" << preprocess.inputHeight()
                            << " " << dtype_str << " threads=" << num_threads
                            << " conf=" << conf_threshold << " nms=" << nms_threshold;
        return true;
    }

    void fillNCHW(const cv::Mat& letterboxed_bgr) {
        // BGR → RGB, /255, NCHW
        const int h = letterboxed_bgr.rows;
        const int w = letterboxed_bgr.cols;
        const size_t plane = static_cast<size_t>(h * w);
        float* r_plane = input_tensor_values.data();
        float* g_plane = r_plane + plane;
        float* b_plane = g_plane + plane;

        for (int y = 0; y < h; ++y) {
            const cv::Vec3b* row = letterboxed_bgr.ptr<cv::Vec3b>(y);
            for (int x = 0; x < w; ++x) {
                const size_t i = static_cast<size_t>(y * w + x);
                // OpenCV BGR → YOLO RGB
                r_plane[i] = row[x][2] * (1.f / 255.f);
                g_plane[i] = row[x][1] * (1.f / 255.f);
                b_plane[i] = row[x][0] * (1.f / 255.f);
            }
        }
    }

    std::vector<Detection> parseAndNms(const float* data,
                                       const ParsedOutput& layout,
                                       const LetterboxMeta& meta) {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;
        boxes.reserve(static_cast<size_t>(layout.num_preds / 4));
        scores.reserve(boxes.capacity());
        class_ids.reserve(boxes.capacity());

        const int attrs = static_cast<int>(layout.num_attrs);
        const int preds = static_cast<int>(layout.num_preds);
        const int start_cls = layout.has_objectness ? 5 : 4;
        const int num_classes = attrs - start_cls;

        auto passesFilter = [&](int cid) {
            if (class_filter.empty()) {
                return true;
            }
            return std::find(class_filter.begin(), class_filter.end(), cid) != class_filter.end();
        };

        for (int i = 0; i < preds; ++i) {
            float cx = 0, cy = 0, w = 0, h = 0, obj = 1.f;
            int best_cls = 0;
            float best_cls_score = 0.f;

            if (layout.layout == YoloLayout::kChannelsFirst) {
                // data[attr * N + i]
                cx = data[0 * preds + i];
                cy = data[1 * preds + i];
                w = data[2 * preds + i];
                h = data[3 * preds + i];
                for (int c = 0; c < num_classes; ++c) {
                    const float s = data[(4 + c) * preds + i];
                    if (s > best_cls_score) {
                        best_cls_score = s;
                        best_cls = c;
                    }
                }
            } else {
                const float* row = data + static_cast<size_t>(i) * attrs;
                cx = row[0];
                cy = row[1];
                w = row[2];
                h = row[3];
                if (layout.has_objectness) {
                    obj = row[4];
                    // alguns exports já aplicam sigmoid; se score > 1, assume linear
                    if (obj < 0.f || obj > 1.f) {
                        obj = sigmoid(obj);
                    }
                }
                for (int c = 0; c < num_classes; ++c) {
                    float s = row[start_cls + c];
                    if (s < 0.f || s > 1.f) {
                        s = sigmoid(s);
                    }
                    if (s > best_cls_score) {
                        best_cls_score = s;
                        best_cls = c;
                    }
                }
            }

            // YOLOv8: class scores já em [0,1] (sigmoid no grafo). Se >1, aplica sigmoid.
            if (layout.layout == YoloLayout::kChannelsFirst) {
                if (best_cls_score < 0.f || best_cls_score > 1.f) {
                    best_cls_score = sigmoid(best_cls_score);
                }
            }

            const float conf = obj * best_cls_score;
            if (conf < conf_threshold || !passesFilter(best_cls)) {
                continue;
            }

            const float x1 = cx - w * 0.5f;
            const float y1 = cy - h * 0.5f;
            cv::Rect2f box_lb(x1, y1, w, h);
            cv::Rect2f box_orig = PreProcessing::mapBoxToOriginal(box_lb, meta);

            if (box_orig.width < 1.f || box_orig.height < 1.f) {
                continue;
            }

            boxes.emplace_back(cv::Rect(static_cast<int>(std::round(box_orig.x)),
                                        static_cast<int>(std::round(box_orig.y)),
                                        static_cast<int>(std::round(box_orig.width)),
                                        static_cast<int>(std::round(box_orig.height))));
            scores.push_back(conf);
            class_ids.push_back(best_cls);
        }

        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, keep);

        std::vector<Detection> out;
        out.reserve(keep.size());
        for (int idx : keep) {
            Detection d;
            d.box = boxes[static_cast<size_t>(idx)] &
                    cv::Rect(0, 0, meta.orig_w, meta.orig_h);
            d.centroid = {d.box.x + d.box.width * 0.5f, d.box.y + d.box.height * 0.5f};
            d.confidence = scores[static_cast<size_t>(idx)];
            d.class_id = class_ids[static_cast<size_t>(idx)];
            if (d.class_id >= 0 &&
                static_cast<size_t>(d.class_id) < class_names.size()) {
                d.label = class_names[static_cast<size_t>(d.class_id)];
            } else {
                d.label = "cls" + std::to_string(d.class_id);
            }
            out.push_back(std::move(d));
        }
        return out;
    }

    std::vector<Detection> detect(const cv::Mat& frame) {
        if (!session || frame.empty()) {
            return {};
        }

        LetterboxMeta meta;
        cv::Mat lb = preprocess.letterbox(frame, meta);
        last_meta = meta;
        fillNCHW(lb);

        const std::array<int64_t, 4> input_shape{
            1, 3, preprocess.inputHeight(), preprocess.inputWidth()};

        Ort::MemoryInfo mem =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor{nullptr};
        if (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            for (size_t i = 0; i < input_tensor_values.size(); ++i) {
                input_tensor_fp16[i] = Ort::Float16_t(input_tensor_values[i]);
            }
            input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
                mem, input_tensor_fp16.data(), input_tensor_fp16.size(), input_shape.data(),
                input_shape.size());
        } else {
            input_tensor = Ort::Value::CreateTensor<float>(
                mem, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(),
                input_shape.size());
        }

        std::vector<Ort::Value> outputs;
        try {
            outputs = session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor,
                                   1, output_names.data(), output_names.size());
        } catch (const Ort::Exception& e) {
            LogError("Detector") << "Run falhou: " << e.what();
            return {};
        }

        if (outputs.empty()) {
            return {};
        }

        // Usa a maior saída rank-3 como tensor de predições
        size_t best_i = 0;
        size_t best_elems = 0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto info = outputs[i].GetTensorTypeAndShapeInfo();
            const auto sh = info.GetShape();
            const size_t n = info.GetElementCount();
            if (sh.size() >= 2 && n > best_elems) {
                best_elems = n;
                best_i = i;
            }
        }

        auto out_info = outputs[best_i].GetTensorTypeAndShapeInfo();
        auto shape = out_info.GetShape();
        // Alguns exports vêm [N, attrs] (rank 2) — promove para [1, N, attrs]
        if (shape.size() == 2) {
            shape = {1, shape[0], shape[1]};
        }

        ParsedOutput layout;
        try {
            layout = inferLayout(shape);
        } catch (const std::exception& e) {
            LogError("Detector") << e.what();
            return {};
        }

        const float* data = nullptr;
        const auto out_dtype = out_info.GetElementType();
        if (out_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const auto* fp16 = outputs[best_i].GetTensorData<Ort::Float16_t>();
            const size_t n = out_info.GetElementCount();
            output_scratch.resize(n);
            for (size_t i = 0; i < n; ++i) {
                output_scratch[i] = static_cast<float>(fp16[i]);
            }
            data = output_scratch.data();
        } else if (out_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            data = outputs[best_i].GetTensorData<float>();
        } else {
            LogError("Detector") << "Tipo de saída não suportado";
            return {};
        }

        return parseAndNms(data, layout, meta);
    }
};

Detector::Detector() : impl_(std::make_unique<Impl>()) {}
Detector::~Detector() = default;
Detector::Detector(Detector&&) noexcept = default;
Detector& Detector::operator=(Detector&&) noexcept = default;

bool Detector::load(const std::string& model_path,
                    float conf_threshold,
                    float nms_threshold,
                    int input_size,
                    int num_threads) {
    return impl_->load(model_path, conf_threshold, nms_threshold, input_size, num_threads);
}

bool Detector::isLoaded() const {
    return impl_ && impl_->session != nullptr;
}

void Detector::setClassFilter(std::vector<int> class_ids) {
    impl_->class_filter = std::move(class_ids);
}

void Detector::setClassNames(std::vector<std::string> names) {
    impl_->class_names = std::move(names);
}

void Detector::setConfThreshold(float t) {
    impl_->conf_threshold = t;
}

void Detector::setNmsThreshold(float t) {
    impl_->nms_threshold = t;
}

std::vector<Detection> Detector::detect(const cv::Mat& frame) {
    return impl_->detect(frame);
}

int Detector::inputSize() const {
    return impl_->input_size;
}

const std::string& Detector::modelPath() const {
    return impl_->model_path;
}

}  // namespace contador
