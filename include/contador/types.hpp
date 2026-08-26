#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace contador {

struct Detection {
    cv::Rect box;
    cv::Point2f centroid;
    float confidence{0.f};
    int class_id{0};
    std::string label{"wheel"};
};

struct Track {
    int id{-1};
    cv::Point2f centroid;
    cv::Rect box;
    int age{0};
    int missed{0};
    bool counted{false};
    int class_id{0};
    std::string label{"vehicle"};
};

struct CountLine {
    cv::Point2f p1;
    cv::Point2f p2;
};

}  // namespace contador
