# Contador de Eixos (Edge)

Sistema de contagem de eixos de veículos na borda (Linux/Armbian), com câmera em ângulo lateral/diagonal.

Pipeline C++17: **VideoSource (RTSP + reconexão) → letterbox → YOLO ONNX → tracker/counter → MJPEG**.

Validação pré-treino em Python com **YOLO-World** (zero-shot) antes de gerar o modelo leve para a placa.

## Arquitetura

```
RTSP / MP4 / synthetic
        │
        ▼
  VideoSource          ← watchdog + reconexão a cada 5s (live)
        │
        ▼
  Detector (ONNX)      ← letterbox 640, FP16/FP32, NMS
        │                boxes no espaço nativo 1280×720
        ▼
  TrackerCounter       ← centróide × linha virtual diagonal
        │
        ▼
  StreamServer MJPEG   ← http://<IP>:8080/stream
```

| Módulo | Responsabilidade |
|--------|------------------|
| `VideoSource` | RTSP/MP4/sintético; timeouts FFmpeg; reconnect sem leak |
| `PreProcessing` | Letterbox (sem homografia — economia na borda) |
| `Detector` | YOLO genérico via ONNX Runtime |
| `TrackerCounter` | Rastreamento + anti-contagem dupla |
| `StreamServer` | HTTP/MJPEG embutido |
| `Logger` | INFO / WARN / ERROR para SSH/`journalctl` |

## Requisitos

### Runtime C++ (borda)

- CMake ≥ 3.16, g++ com C++17
- OpenCV 4.x (`core`, `imgproc`, `imgcodecs`, `videoio`, `dnn`)
- ONNX Runtime C++ (baixado em `third_party/onnxruntime`)

### Validação Python (dev)

- Python 3.10+
- Ver `requirements-yoloworld.txt`

## Build rápido

```bash
# 1) ONNX Runtime (x86_64 ou aarch64)
make ort

# 2) Compilar
make build

# 3) Rodar feed sintético + modelo de teste (se houver)
make run
# ou:
./build/contador_eixo --source synthetic --model models/yolov5n.onnx --port 8080
```

Abrir no navegador: `http://<IP>:8080/` ou `/stream`.

### Flags principais

| Flag | Descrição |
|------|-----------|
| `--source` | `rtsp://...`, MP4, índice de câmera, ou `synthetic` |
| `--model` | Caminho do `.onnx` (YOLO) |
| `--line x1,y1,x2,y2` | Linha virtual diagonal no espaço da imagem |
| `--conf` / `--nms` | Thresholds de detecção |
| `--threads` | Threads IntraOp do ORT (edge: 2–4) |
| `--reconnect-ms` | Intervalo de reconexão RTSP (padrão: 5000) |
| `--read-timeout-ms` | Timeout de frame FFmpeg (padrão: 5000) |
| `--port` | Porta MJPEG (padrão: 8080) |

Exemplo RTSP:

```bash
./build/contador_eixo \
  --source 'rtsp://user:pass@192.168.1.10:554/stream1' \
  --model models/wheels.onnx \
  --line 180,160,1100,620 \
  --threads 2 --port 8080
```

## Validação Zero-Shot (YOLO-World)

Antes de treinar o modelo de rodas para ONNX, valide o ângulo da câmera:

```bash
make venv-world          # cria .venv e instala deps
make validate VIDEO=video_teste.mp4 CONF=0.12

# Auto-labeling (dataset YOLO → dataset_yoloworld/)
make validate VIDEO=video_teste.mp4 CONF=0.12 SAVE_DATASET=1
```

Ou manualmente:

```bash
source .venv/bin/activate
pip install -r requirements-yoloworld.txt
python validador_yoloworld.py video_teste.mp4 --conf 0.15
```

Teclas: `q` sair · `p` pausar/despausar.

## Deploy Armbian (systemd)

1. Compile na placa (ou cross-compile) e instale em `/opt/contador-eixo`:

```text
/opt/contador-eixo/
  bin/contador_eixo
  lib/libonnxruntime.so*
  models/wheels.onnx
```

2. Edite `contador-eixo.service` (RTSP, linha, modelo).

3. Ative o serviço:

```bash
sudo cp contador-eixo.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now contador-eixo
journalctl -u contador-eixo -f
```

Atalhos Make:

```bash
make install-service   # copia o unit (requer sudo)
make status
make logs
```

## Estrutura do repositório

```text
contador-eixo/
├── CMakeLists.txt
├── Makefile
├── contador-eixo.service
├── validador_yoloworld.py
├── requirements-yoloworld.txt
├── include/contador/     # headers
├── src/                  # implementação C++
├── models/               # .onnx (wheels.onnx em produção)
├── config/               # configs futuras
├── scripts/
│   └── fetch_onnxruntime.sh
└── third_party/onnxruntime/   # gerado por `make ort`
```

## Makefile — alvos

| Alvo | Função |
|------|--------|
| `make` / `make build` | Configura CMake (Release) e compila |
| `make ort` | Baixa ONNX Runtime para `third_party/` |
| `make run` | Executa com defaults de desenvolvimento |
| `make validate` | Roda o validador YOLO-World |
| `make clean` | Remove `build/` |
| `make install-service` | Instala unit systemd |
| `make help` | Lista alvos |

## Notas de projeto

- **Sem homografia:** inferência no frame original; o YOLO deve ser treinado na perspectiva defasada.
- **Resolução nativa:** 1280×720; letterbox interno 640×640 mantendo aspect ratio.
- **Linha virtual:** calibrada visualmente (`--line`); o centroid do bbox incrementa o contador ao cruzar.
- Coloque o modelo de produção em `models/wheels.onnx` após treino/export ONNX.
