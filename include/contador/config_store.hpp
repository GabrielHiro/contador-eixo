#pragma once

#include "contador/types.hpp"

#include <cstdint>
#include <string>

namespace contador {

/**
 * Configuração completa do pipeline — usada tanto pelas flags de CLI quanto
 * pelas telas web de configuração (hot-reload) e persistida em JSON simples.
 */
struct PipelineConfig {
    std::string source{"synthetic"};
    std::string output_path{};
    std::string model_path{"models/vehicles.onnx"};
    float conf{0.45f};
    float nms{0.45f};
    int imgsz{640};
    int threads{2};
    CountLine line{{180.f, 160.f}, {1100.f, 620.f}};
    int reconnect_ms{5000};
    int read_timeout_ms{5000};
    uint16_t port{8080};

    /** Estágio 2: detector de eixos no recorte do veículo ao cruzar a linha. */
    bool axle_enabled{true};
    std::string axle_model_path{"models/axles.onnx"};
    float axle_conf{0.35f};
    float axle_nms{0.45f};
    int axle_imgsz{224};
    float axle_crop_margin{0.15f};

    // YOLO26s customizado: light_vehicle, motorcycle, truck por padrão.
    std::string vehicle_class_names{"vehicle"};
    std::string vehicle_class_filter{};
    int light_vehicle_class_id{2};
    int motorcycle_class_id{3};
    int truck_class_id{7};
    int truck_axles_override{7};
    int light_vehicle_axles{2};
    int motorcycle_axles{2};
    int wheel_class_id{0};
    float axle_group_distance{0.12f};
    bool single_vehicle_mode{false};
};

/**
 * Carrega config de um JSON simples e plano (sem arrays/objetos aninhados).
 * Campos ausentes no arquivo mantêm o valor já presente em `cfg` (permite
 * usar `cfg` pré-populado com defaults antes de chamar).
 * @return true se o arquivo existia e foi lido (mesmo que parcialmente).
 */
bool loadConfig(const std::string& path, PipelineConfig& cfg);

/** Carrega a mesma configuração diretamente de um JSON plano em memória. */
bool loadConfigJson(const std::string& json, PipelineConfig& cfg);

/** Persiste `cfg` em JSON simples e legível. @return true em caso de sucesso. */
bool saveConfig(const std::string& path, const PipelineConfig& cfg);

/** Escapa uma string para uso segura como valor JSON (entre aspas). */
std::string jsonEscape(const std::string& s);

}  // namespace contador
