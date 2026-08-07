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
    std::string model_path{"models/wheels.onnx"};
    float conf{0.45f};
    float nms{0.45f};
    int imgsz{640};
    int threads{2};
    CountLine line{{180.f, 160.f}, {1100.f, 620.f}};
    int reconnect_ms{5000};
    int read_timeout_ms{5000};
    uint16_t port{8080};
};

/**
 * Carrega config de um JSON simples e plano (sem arrays/objetos aninhados).
 * Campos ausentes no arquivo mantêm o valor já presente em `cfg` (permite
 * usar `cfg` pré-populado com defaults antes de chamar).
 * @return true se o arquivo existia e foi lido (mesmo que parcialmente).
 */
bool loadConfig(const std::string& path, PipelineConfig& cfg);

/** Persiste `cfg` em JSON simples e legível. @return true em caso de sucesso. */
bool saveConfig(const std::string& path, const PipelineConfig& cfg);

/** Escapa uma string para uso segura como valor JSON (entre aspas). */
std::string jsonEscape(const std::string& s);

}  // namespace contador
