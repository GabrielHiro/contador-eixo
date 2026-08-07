cmake_minimum_required(VERSION 3.16)
project(contador_eixo VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)
message(STATUS "OpenCV ${OpenCV_VERSION} — ${OpenCV_INCLUDE_DIRS}")

# ---------------------------------------------------------------------------
# ONNX Runtime (obrigatório para o Detector)
# Procura em -DONNXRUNTIME_ROOT=... ou third_party/onnxruntime
# ---------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH
    "Prefixo de instalação do ONNX Runtime")

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        ${ONNXRUNTIME_ROOT}/include
        ${ONNXRUNTIME_ROOT}/include/onnxruntime
        /usr/include/onnxruntime
        /usr/local/include/onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS
        ${ONNXRUNTIME_ROOT}/lib
        /usr/lib
        /usr/local/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime não encontrado.\n"
        "  Baixe o pacote C++ em https://github.com/microsoft/onnxruntime/releases\n"
        "  Extraia em third_party/onnxruntime (include/ + lib/) ou passe\n"
        "  -DONNXRUNTIME_ROOT=/caminho/para/onnxruntime\n"
        "  x86_64: onnxruntime-linux-x64-*.tgz\n"
        "  aarch64 (Armbian): onnxruntime-linux-aarch64-*.tgz")
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

# ---------------------------------------------------------------------------
# Biblioteca do domínio
# ---------------------------------------------------------------------------
add_library(contador_core
    src/logger.cpp
    src/video_source.cpp
    src/preprocessing.cpp
    src/detector.cpp
    src/tracker_counter.cpp
    src/config_store.cpp
    src/pipeline_controller.cpp
    src/stream_server.cpp
)

target_include_directories(contador_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(contador_core
    PUBLIC
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIBRARY}
        pthread
)

target_compile_definitions(contador_core PUBLIC CONTADOR_HAS_ONNX=1)

# RPATH para achar libonnxruntime.so sem LD_LIBRARY_PATH no device
get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
set_target_properties(contador_core PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)

# ---------------------------------------------------------------------------
# Executável
# ---------------------------------------------------------------------------
add_executable(contador_eixo src/main.cpp)
target_link_libraries(contador_eixo PRIVATE contador_core)
set_target_properties(contador_eixo PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)

install(TARGETS contador_eixo RUNTIME DESTINATION bin)
install(DIRECTORY models/ DESTINATION share/contador_eixo/models)
install(DIRECTORY config/ DESTINATION share/contador_eixo/config)
install(FILES contador-eixo.service DESTINATION lib/systemd/system)

message(STATUS "---- contador_eixo ${PROJECT_VERSION} ----")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
