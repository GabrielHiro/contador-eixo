#pragma once

#include "contador/types.hpp"

#include <opencv2/core.hpp>
#include <unordered_map>
#include <vector>

namespace contador {

/**
 * Rastreamento por centróide + contagem por interseção com linha virtual.
 * Evita contagem dupla marcando cada track como `counted` após cruzar a linha.
 */
class TrackerCounter {
public:
    explicit TrackerCounter(CountLine line, float max_match_distance = 80.f);

    void setCountLine(CountLine line);
    const CountLine& countLine() const { return line_; }

    /**
     * Atualiza tracks com as detecções do frame atual.
     * @return número de novos eixos contados neste frame.
     */
    int update(const std::vector<Detection>& detections);

    int totalCount() const { return total_count_; }
    const std::vector<Track>& tracks() const { return tracks_; }

    void reset();

    /** Desenha linha virtual, tracks e contador no frame. */
    void drawOverlay(cv::Mat& frame) const;

private:
    bool crossedLine(cv::Point2f prev, cv::Point2f curr) const;
    static float distance(cv::Point2f a, cv::Point2f b);

    CountLine line_;
    float max_match_distance_;
    int next_id_{1};
    int total_count_{0};
    std::vector<Track> tracks_;
    std::unordered_map<int, cv::Point2f> prev_centroids_;
};

}  // namespace contador
