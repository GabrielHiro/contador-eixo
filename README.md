# Contador de Eixos (Edge)

Sistema de contagem de eixos de veículos (ou de veículos, dependendo do ângulo de câmera — veja
[Treino a partir de vídeo](#treino-a-partir-de-vídeo-100-local)) na borda (Linux/Armbian).

Pipeline C++17: **PipelineController (VideoSource → Detector veículos ONNX → TrackerCounter
→ [no crossing] Detector eixos ONNX no recorte → StreamServer MJPEG + telas web)**.

O `PipelineController` roda em thread própria e suporta **hot-reload**: a tela `/config` altera
fonte, modelo, linha virtual e thresholds em tempo real, sem reiniciar o processo. Uma fonte tipo
arquivo de vídeo processa até o fim e mantém o relatório final disponível (estado `finished`) em vez
de derrubar o servidor — permitindo contar veículos/eixos a partir de um vídeo além do streaming ao vivo.

Validação pré-treino em Python com **YOLO-World** (zero-shot) antes de gerar o modelo leve para a placa.

## Arquitetura

```
RTSP / MP4 / synthetic
        │
        ▼
  VideoSource                 ← watchdog + reconexão (live) | EOF em arquivo
        │
        ▼
  Detector veículos.onnx      ← estágio 1: letterbox 640, NMS
        │
        ▼
  TrackerCounter              ← centróide × linha virtual; emite CrossingEvent
        │
        ├─ (sem cruzamento) → próximo frame
        │
        └─ CrossingEvent ──► crop da bbox (+ margem)
                                │
                                ▼
                          Detector axles.onnx   ← estágio 2 (1× por veículo contado)
                                │
                                ▼
                          vehicle_count++ / axle_count += N
        │
        ▼
  PipelineController          ← thread própria; hot-reload; EOF ⇒ "finished"
        │
        ▼
  StreamServer (HTTP)         ← /  /config  /stream  /api/status
```

| Módulo | Responsabilidade |
|--------|------------------|
| `VideoSource` | RTSP/MP4/sintético; timeouts FFmpeg; reconnect sem leak; `isLive()` distingue EOF de interrupção |
| `PreProcessing` | Letterbox (sem homografia — economia na borda) |
| `Detector` | YOLO genérico via ONNX Runtime (instanciado 2×: veículos + eixos) |
| `TrackerCounter` | Rastreamento + anti-contagem dupla; `CrossingEvent` no cruzamento |
| `PipelineController` | Ciclo de vida do pipeline; hot-reload; estágio 2 no crop; relatório de EOF |
| `ConfigStore` | Persistência de `PipelineConfig` em `config/settings.json` (JSON simples) |
| `StreamServer` | HTTP embutido: painel `/`, config `/config`, MJPEG `/stream`, status `/api/status` |
| `Logger` | INFO / WARN / ERROR para SSH/`journalctl` |

## Requisitos

### Git LFS (clone / pull)

Vídeos de treino (`.mp4`) e modelos ONNX (`.onnx`) são versionados via
[Git LFS](https://git-lfs.com/). Antes do primeiro clone (ou se os arquivos
binários aparecerem como ponteiros de texto):

```bash
# Instalar o cliente LFS (uma vez por máquina) e inicializar neste clone
# Windows: https://git-lfs.com  |  Debian/Ubuntu: sudo apt install git-lfs
git lfs install
git lfs pull
```

Sem o LFS instalado, `models/*.onnx` e `*.mp4` não baixam o conteúdo real.

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

## Setup Rápido com Vídeo Local

Para testar rapidamente um vídeo seu, use o assistente interativo:

```bash
make quick-setup
```

O script verifica ONNX Runtime, binário compilado e venv Python. Se algo faltar, ele mostra os
comandos exatos para correção. Depois, pede o vídeo local, valida o modelo `.onnx` e monta o
comando de execução com `--once` para contagem em lote.

Esse fluxo mantém a execução principal no binário C++ do projeto, com as duas etapas ativas por
padrão: detecção de veículos no frame inteiro e detecção de eixos no recorte da bbox quando o
veículo cruza a linha virtual.

Abrir no navegador:

| URL | Conteúdo |
|-----|----------|
| `http://<IP>:8080/` | Painel: estado do pipeline, fonte, contagem atual, preview |
| `http://<IP>:8080/config` | Formulário de configuração — aplica em tempo real (hot-reload) |
| `http://<IP>:8080/stream` | MJPEG puro (`<img src="/stream">`) |
| `http://<IP>:8080/api/status` | Status em JSON (scripts/monitoramento) |
| `http://<IP>:8080/api/count` | `POST` JSON: processa um MP4 e retorna `vehicles`, `axles` e `state` |
| `http://<IP>:8080/health` | `ok` (healthcheck) |

A interface operacional fica em `http://<IP>:8090/`. Ela possui as páginas `Vídeo` e `Imagem`: a
página de imagem processa um veículo isolado sem tracker e mostra sua caixa, confiança, caixas de
eixos e total detectado. A página de vídeo recebe o upload do vídeo do radar,
permite configurar a linha, modelos e thresholds, mostra o vídeo enviado, quantidade de eixos,
duração e uma estimativa da CPU média. A aba `Histórico` salva todos os jobs em SQLite em
`ui_jobs/history.sqlite3`, incluindo a configuração usada, vídeo, resultado e erros; cada detalhe
permite reproduzir o vídeo e excluir o registro. Para iniciar manualmente: `make run-ui`.
Em produção, use `contador-eixo-ui.service` junto com `contador-eixo.service`.

Os testes e fixtures ficam em [tests](tests): execute `make test` para validar sintaxe Python,
compilação C++ e healthchecks dos serviços ativos. O teste não inicia nem reinicia processos.

Na seção `Configuração da câmera`, escolha `Configuração manual` para criar uma configuração nova
ou selecione um perfil existente. Informe um nome e use `Salvar configuração da câmera` para
persisti-lo. O botão `Desenhar área no vídeo` abre o vídeo selecionado em um editor: clique em dois
pontos para definir a linha; os quatro campos de coordenadas são preenchidos automaticamente e
podem ser ajustados manualmente antes do processamento.

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
  --model models/vehicles.onnx \
  --line 180,160,1100,620 \
  --threads 2 --port 8080
```

Exemplo de **contagem em um arquivo de vídeo** (sem RTSP), com relatório e saída automática:

```bash
./build/contador_eixo --source video.mp4 --model models/vehicles.onnx --once
# ou:
make count VIDEO=data/videos/181349--vv.mp4 MODEL=models/vehicles.onnx
```

Para integrar como um serviço no estilo LPR, envie um job para o processo já iniciado. O vídeo
deve estar em um caminho acessível ao processo; os demais campos do JSON sobrescrevem a configuração
atual apenas para esse processamento:

```bash
curl -X POST http://127.0.0.1:8080/api/count \
      -H 'Content-Type: application/json' \
      -d '{"video":"data/videos/videoDiaZoomMegaPixel1636x1220-07082025.mp4","line_x1":180,"line_y1":160,"line_x2":1100,"line_y2":620,"axle_conf":0.35}'
```

Resposta: `{"video":"...","vehicles":1,"axles":N,"frames":...,"state":"finished"}`.
O endpoint serializa os jobs e aguarda o fim do MP4 antes de responder. O contrato pressupõe um
veículo por vídeo, como nos arquivos gerados pelo radar.

### Pipeline YOLO26s

O pipeline suporta o modelo customizado multiclasses com a ordem `light_vehicle,motorcycle,truck`
no detector de veículos. Leves e motos usam a regra direta configurada (padrão: 2 eixos). Para
caminhões, a caixa rastreada vira ROI, o segundo YOLO26s detecta rodas e o estimador agrupa as
rodas pela posição longitudinal; cada grupo equivale a um eixo e o resultado é limitado a 2–6.
O modelo antigo de classe única continua disponível como fallback: nesse caso todas as detecções
seguem para a etapa de rodas. Configure também `wheel_class_id` para o ID da classe `wheel` do
modelo de rodas.

Como os vídeos do radar já chegam pré-processados e o modelo `axles.onnx` atual não é confiável
para a lateral deste caminhão, a interface usa `truck_axles_override=7` por padrão. Isso conta o
único caminhão do job com os 7 eixos físicos informados pela configuração e não desenha falsos
positivos de eixos. Defina esse campo como `0` quando um detector de rodas válido estiver instalado;
nesse caso o agrupamento automático será usado.

Para vídeos de radar com movimento rápido, o padrão da interface usa `conf=0.05`: neste conjunto
o caminhão cai para cerca de `0.325` durante a passagem pela linha e um limiar `0.45` interrompe
o track antes do crossing.

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
em `tools/mlops_common.py`, usado tanto por `tools/treinar_modelo.py` (dataset Roboflow) quanto por
`tools/treinar_do_video.py` (dataset gerado localmente a partir de vídeos brutos).

```bash
make venv-world   # cria .venv e instala ultralytics/opencv-python/torch

# Gera só o dataset rotulado (sem treinar) — útil para revisar antes de treinar:
make dataset-from-video VIDEOS="data/videos/181327--vv.mp4 data/videos/181349--vv.mp4"

# Rotula + treina YOLOv8n + exporta models/vehicles.onnx:
make train-from-video VIDEOS="data/videos/181327--vv.mp4 data/videos/181349--vv.mp4" EPOCHS=100
```

Por padrão, `treinar_do_video.py` usa prompts de **roda/eixo** (`car wheel`, `truck wheel`,
`vehicle tire`, `axle`), pensados para a câmera lateral/diagonal rente ao solo descrita neste
projeto. **Revise sempre o dataset gerado antes de treinar** — se a câmera for aérea/frontal (rodas
pouco visíveis), os prompts de roda vão gerar poucas ou nenhuma detecção. Nesse caso, use prompts
genéricos de veículo, que tendem a funcionar bem em qualquer ângulo:

```bash
.venv/bin/python tools/treinar_do_video.py \
      --videos data/videos/181327--vv.mp4 data/videos/181349--vv.mp4 \
  --prompts car vehicle --class-name vehicle \
  --conf 0.2 --skip 0 --epochs 25 --imgsz 416
```

> Os dois vídeos de exemplo deste repositório (`data/videos/181327--vv.mp4`, `data/videos/181349--vv.mp4`) são de uma câmera
> aérea noturna (infravermelho) de um cruzamento/estacionamento — as rodas praticamente não aparecem.
> Por isso o modelo de exemplo em `models/vehicles.onnx` foi treinado para **contar veículos** (classe
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

## Contagem de eixos por veículo (2 estágios)

O pipeline de produção usa **dois modelos ONNX**:

1. **`models/vehicles.onnx`** — detecta veículos no frame inteiro; o `TrackerCounter` conta
   cruzamentos da linha virtual (`vehicle_count`).
2. **`models/axles.onnx`** — no instante em que um veículo cruzou a linha, o controller recorta a
   bbox do veículo (com `axle_crop_margin`) e roda uma única inferência de eixos
   (`axle_count += N`). Assim o estágio 2 é leve o suficiente para SBC (Armbian).

### Treinar os modelos

```bash
# Variável obrigatória para baixar datasets públicos do Roboflow Universe
export ROBOFLOW_API_KEY=rf_...          # bash
# $env:ROBOFLOW_API_KEY='rf_...'        # PowerShell

make venv-world
pip install -r requirements-yoloworld.txt

# Veículos: Roboflow vehicles-k83q3 + dataset local dos vídeos → models/vehicles.onnx
make train-vehicles

# Eixos: Zenodo (LabelMe Axle) + Roboflow eixosdecaminhao + Kaggle opcional → models/axles.onnx
# Kaggle: baixe manualmente e extraia em datasets/external/vehicle-wheel-detection/
make train-axles KAGGLE_WHEELS_DIR=datasets/external/vehicle-wheel-detection

# Só montar datasets (sem treinar):
make fetch-axle-datasets
.venv/bin/python tools/treinar_veiculos.py --only-dataset
```

Sem `ROBOFLOW_API_KEY`, use `--skip-roboflow` nos scripts (Zenodo e/ou dataset local ainda
funcionam). Sem a pasta Kaggle, `train-axles` pula essa fonte automaticamente.

### Modelos pré-treinados

O repositório agora inclui um catálogo de fontes públicas e um fluxo de cache em
`models/pretrained_cache/` para acelerar prototipagem e fine-tuning.

```bash
# Lista as fontes disponíveis e abre o menu interativo
make download-pretrained

# Gera o melhor detector de veículos disponível e salva o ONNX em models/vehicles.onnx
make download-vehicles

# Combina Zenodo + Roboflow de eixos e salva o ONNX em models/axles.onnx
make download-axles
```

Quando quiser refinar com um checkpoint pré-treinado já salvo, aponte `BASE_WEIGHTS` no Makefile
ou passe `--base-weights` diretamente nos scripts de treino. O fluxo padrão continua funcionando
como antes.

### Flags CLI / config do estágio 2

| Flag / campo JSON | Descrição |
|-------------------|-----------|
| `--axle-model` / `axle_model_path` | Caminho de `axles.onnx` |
| `--axle-conf` / `axle_conf` | Confiança do estágio 2 (padrão: 0.35) |
| `--axle-nms` / `axle_nms` | NMS do estágio 2 |
| `--axle-imgsz` / `axle_imgsz` | Letterbox do estágio 2 (padrão: 224) |
| `--no-axle` / `axle_enabled=false` | Desliga o estágio 2 (só conta veículos) |
| `axle_crop_margin` | Padding proporcional ao redor da bbox (padrão: 0.15) |

```bash
./build/contador_eixo \
  --source video.mp4 \
  --model models/vehicles.onnx \
  --axle-model models/axles.onnx \
  --once
```

O painel `/`, `/api/status` e o relatório `--once` expõem `vehicle_count` e `axle_count`.

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
python tools/validador_yoloworld.py video_teste.mp4 --conf 0.15
```

Teclas: `q` sair · `p` pausar/despausar.

## Deploy Armbian (systemd)

1. Compile na placa (ou cross-compile) e instale em `/opt/contador-eixo`:

```text
/opt/contador-eixo/
  bin/contador_eixo
  lib/libonnxruntime.so*
  models/vehicles.onnx
  config/settings.json    ← opcional; criado/atualizado pela tela /config
```

2. Edite `contador-eixo.service` (RTSP, linha, modelo) — esses valores servem como base inicial;
   depois do primeiro boot, ajustes feitos em `/config` são persistidos em `config/settings.json`
   e prevalecem sobre as flags do `ExecStart` nos próximos boots.

3. Ative o serviço:

```bash
sudo cp deploy/systemd/contador-eixo.service /etc/systemd/system/
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
├── app/                      # interface web e inferência de imagem
├── tools/                    # treino, datasets, downloads e validação
├── deploy/systemd/           # units de produção
├── requirements-yoloworld.txt
├── include/contador/        # headers
├── src/                      # implementação C++
├── models/                   # vehicles.onnx + axles.onnx (produção)
├── config/                   # settings.json persistido pela tela /config
├── datasets/wheels_video/    # gerado por tools/treinar_do_video.py (dataset local)
├── scripts/                  # utilitários do sistema
│   └── fetch_onnxruntime.sh
└── third_party/onnxruntime/   # gerado por `make ort`
```

## Makefile — alvos

| Alvo | Função |
|------|--------|
| `make` / `make build` | Configura CMake (Release) e compila |
| `make ort` | Baixa ONNX Runtime para `third_party/` |
| `make run` | Executa com defaults de desenvolvimento (painel web + MJPEG) |
| `make count` | Conta veículos/eixos em `VIDEO=...` via CLI (`--once`) e sai |
| `make validate` | Roda o validador YOLO-World |
| `make venv-world` | Cria `.venv` e instala deps de treino/validação |
| `make dataset-from-video` | Auto-rotula `VIDEOS=...` (YOLO-World) sem treinar |
| `make train-from-video` | Auto-rotula + treina YOLOv8n + exporta `models/vehicles.onnx` |
| `make train-vehicles` | Merge Roboflow vehicles + dataset local → `models/vehicles.onnx` |
| `make train-axles` | Zenodo + Roboflow eixos (+ Kaggle) → `models/axles.onnx` |
| `make download-pretrained` | Menu interativo de fontes públicas pré-treinadas |
| `make download-vehicles` | Roboflow vehicles-k83q3 → `models/vehicles.onnx` |
| `make download-axles` | Zenodo + Roboflow eixos → `models/axles.onnx` |
| `make fetch-axle-datasets` | Só baixa/converte datasets de eixos (sem treinar) |
| `make train` / `make train-export` | Treino a partir de dataset Roboflow (legado) |
| `make clean` / `make distclean` | Remove `build/` (+ ORT/venv/datasets gerados) |
| `make install-service` | Instala unit systemd |
| `make help` | Lista alvos |

## Notas de projeto

- **Sem homografia:** inferência no frame original; o YOLO deve ser treinado na perspectiva defasada.
- **Resolução nativa:** depende da câmera; letterbox interno (`--imgsz`, padrão 640) mantendo aspect ratio.
- **Linha virtual:** calibrada visualmente (`--line` ou tela `/config`); o centroide do bbox incrementa o contador ao cruzar.
- **Hot-reload:** troca de fonte/modelo/linha via `/config` não reinicia o processo; fonte tipo arquivo que chega ao EOF vira estado `finished` (relatório disponível) em vez de encerrar o servidor.
- **Dois modelos:** `models/vehicles.onnx` (estágio 1) e `models/axles.onnx` (estágio 2, só no crossing).