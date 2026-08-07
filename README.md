# Contador de Eixos (Edge)

Sistema de contagem de eixos de veículos (ou de veículos, dependendo do ângulo de câmera — veja
[Treino a partir de vídeo](#treino-a-partir-de-vídeo-100-local)) na borda (Linux/Armbian).

Pipeline C++17: **PipelineController (VideoSource → letterbox → YOLO ONNX → tracker/counter) → StreamServer (MJPEG + telas web)**.

O `PipelineController` roda em thread própria e suporta **hot-reload**: a tela `/config` altera
fonte, modelo, linha virtual e thresholds em tempo real, sem reiniciar o processo. Uma fonte tipo
arquivo de vídeo processa até o fim e mantém o relatório final disponível (estado `finished`) em vez
de derrubar o servidor — permitindo contar eixos/veículos a partir de um vídeo além do streaming ao vivo.

Validação pré-treino em Python com **YOLO-World** (zero-shot) antes de gerar o modelo leve para a placa.

## Arquitetura

```
RTSP / MP4 / synthetic
        │
        ▼
  VideoSource            ← watchdog + reconexão a cada 5s (live) | EOF em arquivo
        │
        ▼
  Detector (ONNX)        ← letterbox 640, FP16/FP32, NMS
        │                  boxes no espaço nativo do frame
        ▼
  TrackerCounter          ← centróide × linha virtual diagonal
        │
        ▼
  PipelineController      ← thread própria; hot-reload; EOF ⇒ estado "finished" (não mata o processo)
        │
        ▼
  StreamServer (HTTP)     ← http://<IP>:8080/  (painel) /config (hot-reload) /stream (MJPEG) /api/status
```

| Módulo | Responsabilidade |
|--------|------------------|
| `VideoSource` | RTSP/MP4/sintético; timeouts FFmpeg; reconnect sem leak; `isLive()` distingue EOF de interrupção |
| `PreProcessing` | Letterbox (sem homografia — economia na borda) |
| `Detector` | YOLO genérico via ONNX Runtime |
| `TrackerCounter` | Rastreamento + anti-contagem dupla |
| `PipelineController` | Ciclo de vida do pipeline em thread própria; hot-reload; relatório de EOF |
| `ConfigStore` | Persistência de `PipelineConfig` em `config/settings.json` (JSON simples) |
| `StreamServer` | HTTP embutido: painel `/`, config `/config`, MJPEG `/stream`, status `/api/status` |
| `Logger` | INFO / WARN / ERROR para SSH/`journalctl` |

## Requisitos

### Runtime C++ (borda)

- CMake ≥ 3.16, g++ com C++17
- OpenCV 4.x (`core`, `imgproc`, `imgcodecs`, `videoio`, `dnn`)
- ONNX Runtime C++ (baixado em `third_party/onnxruntime`)
- Linux (usa sockets POSIX diretamente); não compila nativamente no Windows

### Validação/treino Python (dev)

- Python 3.10+
- Ver `requirements-yoloworld.txt` (ultralytics, opencv-python, torch)

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

Abrir no navegador:

| URL | Conteúdo |
|-----|----------|
| `http://<IP>:8080/` | Painel: estado do pipeline, fonte, contagem atual, preview |
| `http://<IP>:8080/config` | Formulário de configuração — aplica em tempo real (hot-reload) |
| `http://<IP>:8080/stream` | MJPEG puro (`<img src="/stream">`) |
| `http://<IP>:8080/api/status` | Status em JSON (scripts/monitoramento) |
| `http://<IP>:8080/health` | `ok` (healthcheck) |

### Flags principais

| Flag | Descrição |
|------|-----------|
| `--source` | `rtsp://...`, caminho de MP4, índice de câmera, ou `synthetic` |
| `--model` | Caminho do `.onnx` (YOLO) |
| `--line x1,y1,x2,y2` | Linha virtual diagonal no espaço da imagem |
| `--conf` / `--nms` | Thresholds de detecção |
| `--imgsz` | Lado do letterbox (padrão: 640) — use o mesmo valor do treino |
| `--threads` | Threads IntraOp do ORT (edge: 2–4) |
| `--reconnect-ms` | Intervalo de reconexão RTSP (padrão: 5000) |
| `--read-timeout-ms` | Timeout de frame FFmpeg (padrão: 5000) |
| `--port` | Porta HTTP (painel + MJPEG, padrão: 8080) |
| `--config <path>` | Config persistida a carregar como base (padrão: `config/settings.json`); flags explícitas acima sobrescrevem o que estiver no arquivo |
| `--once` | Ao processar uma fonte não-live (arquivo de vídeo) até o fim, imprime o relatório e encerra o processo automaticamente — útil para contagem em lote via CLI/scripts |

Exemplo RTSP (streaming contínuo, ajustável depois via `/config`):

```bash
./build/contador_eixo \
  --source 'rtsp://user:pass@192.168.1.10:554/stream1' \
  --model models/wheels.onnx \
  --line 180,160,1100,620 \
  --threads 2 --port 8080
```

Exemplo de **contagem em um arquivo de vídeo** (sem RTSP), com relatório e saída automática:

```bash
./build/contador_eixo --source video.mp4 --model models/wheels.onnx --once
# ou:
make count VIDEO=181349--vv.mp4 MODEL=models/wheels.onnx
```

Sem `--once`, o mesmo comando processa o vídeo até o fim, mostra "EIXOS: N" sobreposto no último
frame publicado em `/stream`, e o painel `/` mostra estado `finished` com a contagem total — o
servidor continua no ar (você pode então trocar a fonte por outro vídeo/RTSP via `/config`).

## Telas de configuração (hot-reload)

A tela `/config` expõe todos os campos de `PipelineConfig` (fonte, modelo, thresholds, linha
virtual, timeouts de reconexão). Ao submeter:

1. O `PipelineController` é reconfigurado imediatamente (a fonte/detector atuais são encerrados e
   reabertos com os novos valores) — **sem reiniciar o processo/binário**.
2. Os valores são persistidos em `config/settings.json`, carregados automaticamente no próximo
   boot (via `--config`, ou o padrão já usado se a flag não for passada).

Isso cobre tanto o caso de câmera ao vivo (RTSP) quanto o de arquivo de vídeo: basta apontar
`source` para o caminho do `.mp4` desejado e aplicar — o pipeline processa o arquivo, publica os
frames anotados em `/stream`, e ao chegar ao fim mostra o relatório final no painel.

## Treino a partir de vídeo (100% local)

Quando não há dataset rotulado, o fluxo é: **auto-rotulagem zero-shot (YOLO-World) → dataset YOLO
local → treino YOLOv8n → export ONNX**. Todo o código compartilhado de treino/export/rotulagem vive
em `mlops_common.py`, usado tanto por `treinar_modelo.py` (dataset Roboflow) quanto por
`treinar_do_video.py` (dataset gerado localmente a partir de vídeos brutos).

```bash
make venv-world   # cria .venv e instala ultralytics/opencv-python/torch

# Gera só o dataset rotulado (sem treinar) — útil para revisar antes de treinar:
make dataset-from-video VIDEOS="181327--vv.mp4 181349--vv.mp4"

# Rotula + treina YOLOv8n + exporta models/wheels.onnx:
make train-from-video VIDEOS="181327--vv.mp4 181349--vv.mp4" EPOCHS=100
```

Por padrão, `treinar_do_video.py` usa prompts de **roda/eixo** (`car wheel`, `truck wheel`,
`vehicle tire`, `axle`), pensados para a câmera lateral/diagonal rente ao solo descrita neste
projeto. **Revise sempre o dataset gerado antes de treinar** — se a câmera for aérea/frontal (rodas
pouco visíveis), os prompts de roda vão gerar poucas ou nenhuma detecção. Nesse caso, use prompts
genéricos de veículo, que tendem a funcionar bem em qualquer ângulo:

```bash
.venv/bin/python treinar_do_video.py \
  --videos 181327--vv.mp4 181349--vv.mp4 \
  --prompts car vehicle --class-name vehicle \
  --conf 0.2 --skip 0 --epochs 25 --imgsz 416
```

> Os dois vídeos de exemplo deste repositório (`181327--vv.mp4`, `181349--vv.mp4`) são de uma câmera
> aérea noturna (infravermelho) de um cruzamento/estacionamento — as rodas praticamente não aparecem.
> Por isso o modelo de exemplo em `models/wheels.onnx` foi treinado para **contar veículos** (classe
> `vehicle`) com esses vídeos, não eixos. Para contagem de eixo de verdade, use vídeos com a câmera
> em ângulo lateral/diagonal (ver `--line` e a seção de Notas de projeto abaixo) e mantenha os
> prompts padrão de roda/eixo.

Principais flags de `treinar_do_video.py`:

| Flag | Descrição |
|------|-----------|
| `--videos` | Lista de vídeos de entrada (padrão: os dois vídeos de exemplo do repo) |
| `--prompts` | Prompts zero-shot YOLO-World (padrão: roda/eixo) |
| `--class-name` | Nome da classe única gravada em `data.yaml` (padrão: `wheel`) |
| `--conf` | Limiar de confiança do YOLO-World na rotulagem (padrão: 0.12) |
| `--skip` | Amostra 1 a cada N+1 frames (padrão: 4 — vídeo é redundante) |
| `--val-ratio` | Fração final (cronológica) de cada vídeo reservada para validação (padrão: 0.15) |
| `--only-dataset` | Só gera o dataset rotulado; não treina |
| `--skip-dataset` | Reusa dataset já gerado; só treina/exporta |
| `--epochs` / `--batch` / `--imgsz` | Hiperparâmetros de treino YOLOv8n |

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
  config/settings.json    ← opcional; criado/atualizado pela tela /config
```

2. Edite `contador-eixo.service` (RTSP, linha, modelo) — esses valores servem como base inicial;
   depois do primeiro boot, ajustes feitos em `/config` são persistidos em `config/settings.json`
   e prevalecem sobre as flags do `ExecStart` nos próximos boots.

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
├── mlops_common.py          # treino/export YOLOv8→ONNX + rotulagem zero-shot (compartilhado)
├── treinar_modelo.py        # treino a partir de dataset Roboflow
├── treinar_do_video.py      # treino 100% local a partir de vídeos brutos (auto-labeling)
├── validador_yoloworld.py
├── requirements-yoloworld.txt
├── include/contador/        # headers
├── src/                      # implementação C++
├── models/                   # .onnx (wheels.onnx em produção)
├── config/                   # settings.json persistido pela tela /config
├── datasets/wheels_video/    # gerado por treinar_do_video.py (dataset local)
├── scripts/
│   └── fetch_onnxruntime.sh
└── third_party/onnxruntime/   # gerado por `make ort`
```

## Makefile — alvos

| Alvo | Função |
|------|--------|
| `make` / `make build` | Configura CMake (Release) e compila |
| `make ort` | Baixa ONNX Runtime para `third_party/` |
| `make run` | Executa com defaults de desenvolvimento (painel web + MJPEG) |
| `make count` | Conta eixos/objetos em `VIDEO=...` via CLI (`--once`) e sai |
| `make validate` | Roda o validador YOLO-World |
| `make venv-world` | Cria `.venv` e instala deps de treino/validação |
| `make dataset-from-video` | Auto-rotula `VIDEOS=...` (YOLO-World) sem treinar |
| `make train-from-video` | Auto-rotula + treina YOLOv8n + exporta `models/wheels.onnx` |
| `make train` / `make train-export` | Treino a partir de dataset Roboflow |
| `make clean` / `make distclean` | Remove `build/` (+ ORT/venv/datasets gerados) |
| `make install-service` | Instala unit systemd |
| `make help` | Lista alvos |

## Notas de projeto

- **Sem homografia:** inferência no frame original; o YOLO deve ser treinado na perspectiva defasada.
- **Resolução nativa:** depende da câmera; letterbox interno (`--imgsz`, padrão 640) mantendo aspect ratio.
- **Linha virtual:** calibrada visualmente (`--line` ou tela `/config`); o centroide do bbox incrementa o contador ao cruzar.
- **Hot-reload:** troca de fonte/modelo/linha via `/config` não reinicia o processo; fonte tipo arquivo que chega ao EOF vira estado `finished` (relatório disponível) em vez de encerrar o servidor.
- Coloque o modelo de produção em `models/wheels.onnx` após treino/export ONNX.
