#!/usr/bin/env python3
"""
treinar_modelo.py — fluxo MLOps: Roboflow → YOLOv8n → ONNX (borda C++).

Gera models/wheels.onnx compatível com o Detector ONNX Runtime do contador_eixo:
  - simplify=True
  - dynamic=False   (shape fixo; evita surpresas no Ort::Session em Armbian)

Uso (com venv ativado):
  python treinar_modelo.py
  python treinar_modelo.py --epochs 100 --device 0
  python treinar_modelo.py --skip-download   # reusa dataset já baixado
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlops_common as mlc

# =============================================================================
# CONFIGURAÇÃO — preencha com os dados do seu projeto Roboflow
# =============================================================================

ROBOFLOW_API_KEY = "xNHlAEjLqtDRSa2BZLQn"
ROBOFLOW_WORKSPACE = "class-oyl7p"
ROBOFLOW_PROJECT = "wheels-detection-vuaey"
ROBOFLOW_VERSION = 1

# Onde o dataset YOLOv8 será extraído (data.yaml fica em DATASET_DIR/data.yaml)
DATASET_DIR = Path("datasets/wheels_roboflow")

# Treino
BASE_WEIGHTS = "yolov8n.pt"               # nano — leve para edge
EPOCHS = 100
IMGSZ = 640
BATCH = 16                                # reduza (8/4) se faltar VRAM
PROJECT_RUNS = Path("runs/wheels")        # ultralytics project/
RUN_NAME = "train"

# Export ONNX → pasta models/ consumida pelo binário C++
ONNX_OUTPUT = Path("models/wheels.onnx")

# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina YOLOv8n (wheels) e exporta ONNX")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--device", default="", help="cpu | 0 | 0,1 | ''=auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Não baixa do Roboflow; usa DATASET_DIR existente",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Só exporta a partir do best.pt já treinado",
    )
    p.add_argument(
        "--weights",
        default="",
        help="Checkpoint .pt para export (padrão: runs/.../weights/best.pt)",
    )
    return p.parse_args()


def assert_roboflow_config() -> None:
    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY.startswith("SUA_"):
        raise SystemExit(
            "[ERRO] Preencha ROBOFLOW_API_KEY no topo de treinar_modelo.py"
        )
    if ROBOFLOW_WORKSPACE.startswith("seu-") or ROBOFLOW_PROJECT.startswith("seu-"):
        raise SystemExit(
            "[ERRO] Preencha ROBOFLOW_WORKSPACE e ROBOFLOW_PROJECT no topo do script"
        )


def download_dataset() -> Path:
    """Baixa a versão do projeto no formato YOLOv8 (ultralytics)."""
    from roboflow import Roboflow

    assert_roboflow_config()

    print("[INFO] Conectando ao Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)

    print(
        f"[INFO] Download: workspace={ROBOFLOW_WORKSPACE} "
        f"project={ROBOFLOW_PROJECT} version={ROBOFLOW_VERSION} → {DATASET_DIR}"
    )
    dataset = version.download(
        model_format="yolov8",
        location=str(DATASET_DIR),
        overwrite=True,
    )

    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.is_file():
        # Algumas versões do SDK já apontam location para a pasta com data.yaml
        alt = Path(dataset.location)
        if (alt / "data.yaml").is_file():
            data_yaml = alt / "data.yaml"
        else:
            raise FileNotFoundError(f"data.yaml não encontrado em {dataset.location}")

    print(f"[INFO] Dataset pronto: {data_yaml.resolve()}")
    return data_yaml


def resolve_data_yaml(skip_download: bool) -> Path:
    if skip_download:
        data_yaml = DATASET_DIR / "data.yaml"
        if not data_yaml.is_file():
            raise FileNotFoundError(
                f"data.yaml ausente em {data_yaml}. Remova --skip-download ou baixe antes."
            )
        print(f"[INFO] Reutilizando dataset: {data_yaml.resolve()}")
        return data_yaml
    return download_dataset()


def train(data_yaml: Path, args: argparse.Namespace) -> Path:
    """Treina YOLOv8n e retorna o caminho de weights/best.pt."""
    return mlc.train_yolo(
        data_yaml,
        base_weights=BASE_WEIGHTS,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=PROJECT_RUNS,
        name=RUN_NAME,
    )


def export_onnx(weights: Path, imgsz: int) -> Path:
    """Exporta para ONNX estático + simplificado (ver mlops_common.export_onnx)."""
    return mlc.export_onnx(weights, imgsz, ONNX_OUTPUT)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    # Garante paths relativos à raiz do repo
    import os

    os.chdir(root)

    try:
        if args.skip_train:
            weights = Path(args.weights) if args.weights else (
                PROJECT_RUNS / RUN_NAME / "weights" / "best.pt"
            )
            if not weights.is_file():
                print(f"[ERRO] Checkpoint não encontrado: {weights}", file=sys.stderr)
                return 1
            export_onnx(weights, args.imgsz)
            return 0

        data_yaml = resolve_data_yaml(args.skip_download)
        best_pt = train(data_yaml, args)
        export_onnx(best_pt, args.imgsz)
    except Exception as exc:  # noqa: BLE001 — script de ops: falha visível
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 1

    print("[INFO] Pipeline concluído com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
