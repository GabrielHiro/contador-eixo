#include "contador/preprocessing.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace contador {

PreProcessing::PreProcessing(int input_size) : input_w_(input_size), input_h_(input_size) {}

PreProcessing::PreProcessing(int input_w, int input_h) : input_w_(input_w), input_h_(input_h) {}

void PreProcessing::setInputSize(int input_w, int input_h) {
    input_w_ = input_w;
    input_h_ = input_h;
}

cv::Mat PreProcessing::letterbox(const cv::Mat& frame, LetterboxMeta& meta) const {
    meta = {};
    meta.input_w = input_w_;
    meta.input_h = input_h_;

    if (frame.empty()) {
        return {};
    }

    meta.orig_w = frame.cols;
    meta.orig_h = frame.rows;

    const float r = std::min(static_cast<float>(input_w_) / static_cast<float>(frame.cols),
                             static_cast<float>(input_h_) / static_cast<float>(frame.rows));
    meta.scale = r;

    const int new_w = static_cast<int>(std::round(frame.cols * r));
    const int new_h = static_cast<int>(std::round(frame.rows * r));

    meta.pad_x = (input_w_ - new_w) * 0.5f;
    meta.pad_y = (input_h_ - new_h) * 0.5f;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat out(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    const int top = static_cast<int>(std::round(meta.pad_y - 0.1f));
    const int left = static_cast<int>(std::round(meta.pad_x - 0.1f));
    resized.copyTo(out(cv::Rect(left, top, new_w, new_h)));
    return out;
}

cv::Rect2f PreProcessing::mapBoxToOriginal(const cv::Rect2f& box, const LetterboxMeta& meta) {
    const float x1 = (box.x - meta.pad_x) / meta.scale;
    const float y1 = (box.y - meta.pad_y) / meta.scale;
    const float x2 = (box.x + box.width - meta.pad_x) / meta.scale;
    const float y2 = (box.y + box.height - meta.pad_y) / meta.scale;

    const float xi1 = std::clamp(x1, 0.f, static_cast<float>(meta.orig_w));
    const float yi1 = std::clamp(y1, 0.f, static_cast<float>(meta.orig_h));
    const float xi2 = std::clamp(x2, 0.f, static_cast<float>(meta.orig_w));
    const float yi2 = std::clamp(y2, 0.f, static_cast<float>(meta.orig_h));

    return cv::Rect2f(xi1, yi1, std::max(0.f, xi2 - xi1), std::max(0.f, yi2 - yi1));
}

cv::Point2f PreProcessing::mapPointToOriginal(cv::Point2f pt, const LetterboxMeta& meta) {
    return {(pt.x - meta.pad_x) / meta.scale, (pt.y - meta.pad_y) / meta.scale};
}

}  // namespace contador
