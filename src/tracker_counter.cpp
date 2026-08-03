#include "contador/tracker_counter.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace contador {

TrackerCounter::TrackerCounter(CountLine line, float max_match_distance)
    : line_(line), max_match_distance_(max_match_distance) {}

void TrackerCounter::setCountLine(CountLine line) {
    line_ = line;
}

float TrackerCounter::distance(cv::Point2f a, cv::Point2f b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

bool TrackerCounter::crossedLine(cv::Point2f prev, cv::Point2f curr) const {
    // Interseção de segmentos: trajetória (prev→curr) × linha (p1→p2)
    auto cross = [](cv::Point2f o, cv::Point2f a, cv::Point2f b) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    };

    const float d1 = cross(line_.p1, line_.p2, prev);
    const float d2 = cross(line_.p1, line_.p2, curr);
    const float d3 = cross(prev, curr, line_.p1);
    const float d4 = cross(prev, curr, line_.p2);

    return (d1 * d2 < 0.f) && (d3 * d4 < 0.f);
}

int TrackerCounter::update(const std::vector<Detection>& detections) {
    constexpr int kMaxMissed = 15;
    int newly_counted = 0;

    std::vector<bool> det_matched(detections.size(), false);
    std::vector<bool> track_matched(tracks_.size(), false);

    // Matching guloso por distância de centróide
    struct Candidate {
        float dist;
        size_t ti;
        size_t di;
    };
    std::vector<Candidate> candidates;
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        for (size_t di = 0; di < detections.size(); ++di) {
            const float dist = distance(tracks_[ti].centroid, detections[di].centroid);
            if (dist <= max_match_distance_) {
                candidates.push_back({dist, ti, di});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });

    for (const auto& c : candidates) {
        if (track_matched[c.ti] || det_matched[c.di]) {
            continue;
        }
        track_matched[c.ti] = true;
        det_matched[c.di] = true;

        Track& t = tracks_[c.ti];
        const cv::Point2f prev = t.centroid;
        t.centroid = detections[c.di].centroid;
        t.box = detections[c.di].box;
        t.age += 1;
        t.missed = 0;

        if (!t.counted && crossedLine(prev, t.centroid)) {
            t.counted = true;
            ++total_count_;
            ++newly_counted;
        }
        prev_centroids_[t.id] = t.centroid;
    }

    // Tracks não casados: incrementa missed
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        if (!track_matched[ti]) {
            tracks_[ti].missed += 1;
        }
    }

    // Remove tracks perdidos
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [kMaxMissed](const Track& t) {
                                     return t.missed > kMaxMissed;
                                 }),
                  tracks_.end());

    // Novas detecções → novos tracks
    for (size_t di = 0; di < detections.size(); ++di) {
        if (det_matched[di]) {
            continue;
        }
        Track t;
        t.id = next_id_++;
        t.centroid = detections[di].centroid;
        t.box = detections[di].box;
        t.age = 1;
        t.missed = 0;
        t.counted = false;
        tracks_.push_back(t);
        prev_centroids_[t.id] = t.centroid;
    }

    return newly_counted;
}

void TrackerCounter::reset() {
    tracks_.clear();
    prev_centroids_.clear();
    total_count_ = 0;
    next_id_ = 1;
}

void TrackerCounter::drawOverlay(cv::Mat& frame) const {
    if (frame.empty()) {
        return;
    }

    // Linha virtual de contagem
    cv::line(frame,
             cv::Point(static_cast<int>(line_.p1.x), static_cast<int>(line_.p1.y)),
             cv::Point(static_cast<int>(line_.p2.x), static_cast<int>(line_.p2.y)),
             cv::Scalar(0, 0, 255), 3);

    for (const auto& t : tracks_) {
        const cv::Scalar color = t.counted ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 220, 255);
        cv::rectangle(frame, t.box, color, 2);
        cv::circle(frame, t.centroid, 4, color, cv::FILLED);

        std::ostringstream ss;
        ss << "ID " << t.id;
        cv::putText(frame, ss.str(),
                    cv::Point(t.box.x, std::max(15, t.box.y - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    std::ostringstream count_ss;
    count_ss << "EIXOS: " << total_count_;
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(280, 70), cv::Scalar(0, 0, 0),
                  cv::FILLED);
    cv::putText(frame, count_ss.str(), cv::Point(20, 55), cv::FONT_HERSHEY_SIMPLEX, 1.2,
                cv::Scalar(0, 255, 0), 2);
}

}  // namespace contador
