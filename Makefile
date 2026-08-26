# Contador de Eixos — Makefile
# Uso: make help

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Paths / toolchains
# ---------------------------------------------------------------------------
BUILD_DIR      ?= build
BUILD_TYPE     ?= Release
ONNXRUNTIME_ROOT ?= $(CURDIR)/third_party/onnxruntime
CMAKE_FLAGS    ?= -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DONNXRUNTIME_ROOT=$(ONNXRUNTIME_ROOT)

BIN            := $(BUILD_DIR)/contador_eixo
JOBS           ?= $(shell nproc 2>/dev/null || echo 2)

# Runtime defaults (dev)
SOURCE         ?= synthetic
MODEL          ?= models/vehicles.onnx
AXLE_MODEL     ?= models/axles.onnx
PORT           ?= 8080
LINE           ?= 180,160,1100,620
THREADS        ?= 2
CONF           ?= 0.45
RECONNECT_MS   ?= 5000

# YOLO-World validator
VENV           ?= .venv
VIDEO          ?= video_teste.mp4
WORLD_MODEL    ?= yolov8s-world.pt
WORLD_CONF     ?= 0.15
SAVE_DATASET   ?= 0

# Treino a partir de vídeo (auto-rotulagem local, sem Roboflow)
VIDEOS         ?= 181327--vv.mp4 181349--vv.mp4
WORLD_SKIP     ?= 4
VAL_RATIO      ?= 0.15
EPOCHS         ?= 100

# Treino de eixos / veículos (2 estágios)
KAGGLE_WHEELS_DIR ?= datasets/external/vehicle-wheel-detection
AXLE_IMGSZ     ?= 224
AXLE_EPOCHS    ?= 50
VEHICLE_EPOCHS ?= 40
BASE_WEIGHTS   ?= yolov8n.pt
PRETRAINED_CACHE ?= models/pretrained_cache

INSTALL_PREFIX ?= /opt/contador-eixo
SERVICE_FILE   ?= contador-eixo.service

.PHONY: help ort configure build run count run-ui clean distclean \
        venv-world validate train train-export dataset-from-video train-from-video \
	fetch-axle-datasets train-axles train-vehicles \
	download-pretrained download-vehicles download-axles \
	train-axles-auto quick-setup \
        install-service uninstall-service status logs \
        tree

help: ## Mostra esta ajuda
	@echo "Contador de Eixos — alvos disponíveis:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Exemplos:"
	@echo "  make ort && make build && make run"
	@echo "  make run SOURCE='rtsp://user:pass@ip:554/stream1' MODEL=models/vehicles.onnx"
	@echo "  make validate VIDEO=video_teste.mp4 WORLD_CONF=0.12 SAVE_DATASET=1"
	@echo "  make train-vehicles   # requer ROBOFLOW_API_KEY (opcional com --skip-roboflow)"
	@echo "  make train-axles      # Zenodo + Roboflow (+ Kaggle se KAGGLE_WHEELS_DIR existir)"
	@echo "  make download-vehicles / download-axles / download-pretrained"
	@echo "  make quick-setup      # assistente interativo para testar com um vídeo local"

# ---------------------------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------------------------
ort: ## Baixa ONNX Runtime C++ em third_party/onnxruntime
	@chmod +x scripts/fetch_onnxruntime.sh
	./scripts/fetch_onnxruntime.sh
	@test -f "$(ONNXRUNTIME_ROOT)/include/onnxruntime_cxx_api.h" || \
		(echo "ERRO: onnxruntime_cxx_api.h não encontrado em $(ONNXRUNTIME_ROOT)"; exit 1)
	@echo "OK: $(ONNXRUNTIME_ROOT)"

# ---------------------------------------------------------------------------
# Build C++
# ---------------------------------------------------------------------------
configure: ## Configura CMake (Release)
	@test -f "$(ONNXRUNTIME_ROOT)/include/onnxruntime_cxx_api.h" || \
		(echo "ONNX Runtime ausente. Rode: make ort"; exit 1)
	cmake -S . -B $(BUILD_DIR) $(CMAKE_FLAGS)

build: configure ## Compila contador_eixo
	cmake --build $(BUILD_DIR) -j$(JOBS)
	@echo "Binário: $(BIN)"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
run: build ## Roda o pipeline (SOURCE/MODEL/PORT configuráveis)
	@test -f "$(BIN)" || (echo "Compile primeiro: make build"; exit 1)
	$(BIN) \
		--source "$(SOURCE)" \
		--model "$(MODEL)" \
		--axle-model "$(AXLE_MODEL)" \
		--line "$(LINE)" \
		--port $(PORT) \
		--threads $(THREADS) \
		--conf $(CONF) \
		--reconnect-ms $(RECONNECT_MS)

count: build ## Conta eixos em um arquivo de vídeo e sai (--once) — ex: make count VIDEO=181327--vv.mp4
	@test -f "$(BIN)" || (echo "Compile primeiro: make build"; exit 1)
	@test -f "$(VIDEO)" || (echo "Vídeo não encontrado: $(VIDEO)"; exit 1)
	$(BIN) \
		--source "$(VIDEO)" \
		--model "$(MODEL)" \
		--axle-model "$(AXLE_MODEL)" \
		--line "$(LINE)" \
		--port $(PORT) \
		--threads $(THREADS) \
		--conf $(CONF) \
		--once

run-ui: ## Inicia a interface web cliente da API na porta 8090
	python3 interface_service.py

# ---------------------------------------------------------------------------
# Python — YOLO-World
# ---------------------------------------------------------------------------
venv-world: ## Cria .venv e instala requirements-yoloworld.txt
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements-yoloworld.txt
	@echo "Ative com: source $(VENV)/bin/activate"

validate: ## Validador Zero-Shot (VIDEO=... WORLD_CONF=... SAVE_DATASET=0|1)
	@test -x "$(VENV)/bin/python" || (echo "Rode: make venv-world"; exit 1)
	@test -f "$(VIDEO)" || (echo "Vídeo não encontrado: $(VIDEO)"; exit 1)
	$(VENV)/bin/python validador_yoloworld.py "$(VIDEO)" \
		--model "$(WORLD_MODEL)" \
		--conf $(WORLD_CONF) \
		$(if $(filter 1 true TRUE yes YES,$(SAVE_DATASET)),--save-dataset,)

train: ## Treina YOLOv8n via Roboflow e exporta models/vehicles.onnx (legado: treinar_modelo.py)
	@test -x "$(VENV)/bin/python" || (echo "Ative o venv e instale deps (ultralytics, roboflow)"; exit 1)
	$(VENV)/bin/python treinar_modelo.py --base-weights "$(BASE_WEIGHTS)"

train-export: ## Só reexporta ONNX a partir do best.pt já treinado
	@test -x "$(VENV)/bin/python" || (echo "Ative o venv primeiro"; exit 1)
	$(VENV)/bin/python treinar_modelo.py --skip-train

dataset-from-video: ## Auto-rotula VIDEOS com YOLO-World e gera datasets/wheels_video (sem treinar)
	@test -x "$(VENV)/bin/python" || (echo "Rode: make venv-world"; exit 1)
	$(VENV)/bin/python treinar_do_video.py \
		--videos $(VIDEOS) \
		--conf $(WORLD_CONF) \
		--skip $(WORLD_SKIP) \
		--val-ratio $(VAL_RATIO) \
		--class-name vehicle \
		--only-dataset

train-from-video: ## Auto-rotula VIDEOS (YOLO-World) + treina YOLOv8n + exporta models/vehicles.onnx
	@test -x "$(VENV)/bin/python" || (echo "Rode: make venv-world"; exit 1)
	$(VENV)/bin/python treinar_do_video.py \
		--videos $(VIDEOS) \
		--conf $(WORLD_CONF) \
		--skip $(WORLD_SKIP) \
		--val-ratio $(VAL_RATIO) \
		--epochs $(EPOCHS) \
		--class-name vehicle \
		--prompts car vehicle \
		--base-weights "$(BASE_WEIGHTS)"

fetch-axle-datasets: ## Baixa/converte Zenodo + Roboflow eixos (+ Kaggle se existir) sem treinar
	@test -x "$(VENV)/bin/python" || (echo "Rode: make venv-world"; exit 1)
	$(VENV)/bin/python treinar_eixos.py \
		--only-dataset \
		--kaggle-dir "$(KAGGLE_WHEELS_DIR)" \
		$(if $(wildcard $(KAGGLE_WHEELS_DIR)/.),,--skip-kaggle)

train-axles: ## Monta dataset de eixos + treina YOLOv8n + exporta models/axles.onnx
	@test -x "$(VENV)/bin/python" || (echo "Rode: make venv-world"; exit 1)
	$(VENV)/bin/python treinar_eixos.py \
		--epochs $(AXLE_EPOCHS) \
		--imgsz $(AXLE_IMGSZ) \
		--base-weights "$(BASE_WEIGHTS)" \
		--kaggle-dir "$(KAGGLE_WHEELS_DIR)" \
		$(if $(wildcard $(KAGGLE_WHEELS_DIR)/.),,--skip-kaggle)

train-axles-auto: ## Fluxo automático de eixos com base pré-treinado em cache quando existir
	@test -x "$(VENV)/bin/python" || (echo "Rode: make venv-world"; exit 1)
	$(VENV)/bin/python treinar_eixos.py \
		--auto \
		--epochs $(AXLE_EPOCHS) \
		--imgsz $(AXLE_IMGSZ) \
		--kaggle-dir "$(KAGGLE_WHEELS_DIR)" \
		$(if $(wildcard $(KAGGLE_WHEELS_DIR)/.),,--skip-kaggle)

quick-setup: ## Assistente interativo para testar com vídeo local
	@test -x "$(VENV)/bin/python" || (echo "Rode: make venv-world"; exit 1)
	$(VENV)/bin/python quick_setup.py

train-vehicles: ## Merge Roboflow vehicles + dataset local + treina → models/vehicles.onnx
	@test -x "$(VENV)/bin/python" || (echo "Rode: make venv-world"; exit 1)
	$(VENV)/bin/python treinar_veiculos.py --epochs $(VEHICLE_EPOCHS) --base-weights "$(BASE_WEIGHTS)"

download-pretrained: ## Menu interativo de modelos pré-treinados / cache local
	@test -x "$(VENV)/bin/python" || (echo "Rode: make venv-world"; exit 1)
	$(VENV)/bin/python download_pretrained_models.py --cache-root "$(PRETRAINED_CACHE)"

download-vehicles: ## Copia o ONNX embarcado de veículos → models/vehicles.onnx
	@test -x "$(VENV)/bin/python" || (echo "Rode: make venv-world"; exit 1)
	$(VENV)/bin/python download_pretrained_models.py \
		--copy-vehicle-bundled \
		--cache-root "$(PRETRAINED_CACHE)" \
		--output models/vehicles.onnx \
		--device cpu

download-axles: ## Copia o ONNX embarcado de eixos → models/axles.onnx
	@test -x "$(VENV)/bin/python" || (echo "Rode: make venv-world"; exit 1)
	$(VENV)/bin/python download_pretrained_models.py \
		--copy-axle-bundled \
		--cache-root "$(PRETRAINED_CACHE)" \
		--output models/axles.onnx \
		--device cpu

# ---------------------------------------------------------------------------
# Systemd (Armbian)
# ---------------------------------------------------------------------------
install-service: ## Instala contador-eixo.service (sudo)
	sudo cp $(SERVICE_FILE) /etc/systemd/system/$(SERVICE_FILE)
	sudo systemctl daemon-reload
	sudo systemctl enable $(SERVICE_FILE)
	@echo "Edite RTSP/modelo em /etc/systemd/system/$(SERVICE_FILE)"
	@echo "Depois: sudo systemctl start contador-eixo"

uninstall-service: ## Remove o serviço systemd (sudo)
	-sudo systemctl disable --now contador-eixo
	-sudo rm -f /etc/systemd/system/$(SERVICE_FILE)
	sudo systemctl daemon-reload

status: ## Status do serviço
	systemctl status contador-eixo --no-pager || true

logs: ## Segue logs do journald
	journalctl -u contador-eixo -f

# ---------------------------------------------------------------------------
# Limpeza / util
# ---------------------------------------------------------------------------
clean: ## Remove diretório build/
	rm -rf $(BUILD_DIR)

distclean: clean ## clean + remove ORT baixado e venv
	rm -rf third_party/onnxruntime $(VENV) dataset_yoloworld \
		datasets/wheels_video datasets/axles_merged datasets/axles_staging \
		datasets/vehicles_merged datasets/vehicles_staging \
		runs/wheels_video runs/axles runs/vehicles

tree: ## Lista arquivos do projeto (sem build/third_party)
	@find . -type f \
		! -path './build/*' ! -path './third_party/*' ! -path './.venv/*' \
		! -path './.git/*' ! -path './dataset_yoloworld/*' \
		! -name '*.onnx' ! -name '*.pt' | sort
