# Changelog

Todas as mudanças notáveis deste projeto são documentadas neste arquivo.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adota [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Added

- Git LFS para vídeos (`.mp4`/`.mov`) e modelos (`.onnx`, `.pt`, `.pth`,
  `.weights`), com nota de setup no README.
- Licença MIT (`LICENSE`).
- Guia de contribuição (`CONTRIBUTING.md`).
- Ampliação do `.gitignore` (caches de pytest/ruff, egg-info, dist, logs,
  artefatos de SO).
- Pipeline de **2 estágios**: detector de veículos + detector de eixos no
  recorte ao cruzar a linha (`CrossingEvent`, `axles.onnx`, campos `axle_*`
  na config/CLI/UI).
- Scripts MLOps: `datasets_axles.py`, `treinar_eixos.py`, `treinar_veiculos.py`;
  helper `download_roboflow_dataset()`; alvos Make `train-axles` /
  `train-vehicles` / `fetch-axle-datasets`.
- Modelo padrão renomeado: `models/vehicles.onnx` (antes `wheels.onnx`).

### Notes

- A migração para Git LFS **não reescreve o histórico**: os blobs binários
  já commitados antes desta fase permanecem nos commits antigos. A partir do
  commit de LFS, o repositório versiona apenas ponteiros LFS. Uma limpeza
  futura deliberada (`git lfs migrate import` + force-push) pode ser feita
  se o tamanho do histórico se tornar um problema.
- Compilação/teste C++ nesta máquina de desenvolvimento Windows depende do
  CI (GitHub Actions, Fase 2): não há WSL com distro nem Docker instalados
  localmente. Testes Python podem rodar no `.venv`.
- Treino de eixos com Zenodo baixa ~979 MB; Roboflow exige `ROBOFLOW_API_KEY`;
  Kaggle é opcional (pasta local).
## [0.1.0] - 2026-08-06

Baseline do projeto antes da higiene de repositório (Fase 0).

### Added

- Pipeline C++17 na borda: `VideoSource` → `Detector` (YOLO ONNX) →
  `TrackerCounter` → `StreamServer` (HTTP/MJPEG).
- `PipelineController` com worker thread, hot-reload de configuração e
  tratamento de EOF em arquivos de vídeo (estado `finished` sem derrubar o
  servidor).
- `ConfigStore` / `config/settings.json` para persistir fonte, modelo, linha
  virtual, thresholds e porta.
- Telas web `/`, `/config`, `/stream`, `/api/status`, `/health`.
- Flags CLI `--config` e `--once` (contagem em lote a partir de vídeo).
- Pipeline Python de MLOps: `validador_yoloworld.py`, `mlops_common.py`,
  `treinar_do_video.py`, `treinar_modelo.py` (YOLOv8n → ONNX).
- Makefile (`ort`, `build`, `run`, `count`, `validate`, `train-from-video`,
  `install-service`) e unit systemd `contador-eixo.service`.

[Unreleased]: https://github.com/GabrielHiro/contador-eixo/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/GabrielHiro/contador-eixo/releases/tag/v0.1.0
