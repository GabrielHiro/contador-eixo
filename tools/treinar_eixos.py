#!/usr/bin/env python3
"""
treinar_eixos.py — monta dataset de eixos (Zenodo + Roboflow + Kaggle opcional),
treina YOLOv8n e exporta models/axles.onnx.

Fontes:
  - Zenodo Truck Image Dataset (LabelMe, DOI 10.5281/zenodo.17128113) — automático
  - Roboflow class-h27po/eixosdecaminhao — requer ROBOFLOW_API_KEY
  - Kaggle dataclusterlabs/vehicle-wheel-detection — pasta local (--kaggle-dir)

Uso:
  $env:ROBOFLOW_API_KEY='rf_...'
  python treinar_eixos.py
  python treinar_eixos.py --skip-roboflow --skip-kaggle   # só Zenodo
  python treinar_eixos.py --only-dataset
  python treinar_eixos.py --skip-dataset --epochs 40
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import datasets_axles as dax
import mlops_common as mlc

STAGING = Path("datasets/axles_staging")
MERGED = Path("datasets/axles_merged")
ZENODO_DIR = Path("datasets/external/zenodo_trucks")
ROBOFLOW_DIR = Path("datasets/external/eixosdecaminhao")
DEFAULT_KAGGLE = Path("datasets/external/vehicle-wheel-detection")
PROJECT_RUNS = Path("runs/axles")
ONNX_OUTPUT = Path("models/axles.onnx")

RF_WORKSPACE = "class-h27po"
RF_PROJECT = "eixosdecaminhao"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina detector de eixos -> models/axles.onnx")
    p.add_argument("--kaggle-dir", type=Path, default=DEFAULT_KAGGLE)
    p.add_argument("--roboflow-workspace", default=RF_WORKSPACE)
    p.add_argument("--roboflow-project", default=RF_PROJECT)
    p.add_argument("--roboflow-version", type=int, default=None)
    p.add_argument("--skip-zenodo", action="store_true")
    p.add_argument("--skip-roboflow", action="store_true")
    p.add_argument("--skip-kaggle", action="store_true")
    p.add_argument("--force-zenodo", action="store_true", help="Rebaixar/reextrair Zenodo")
    p.add_argument(
        "--auto",
        action="store_true",
        help="Modo automático: melhor esforço com todas as fontes e uso de pré-treinado em cache se existir",
    )
    p.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Usa um peso base pré-treinado em cache quando disponível",
    )
    p.add_argument("--only-dataset", action="store_true")
    p.add_argument("--skip-dataset", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=224, help="Crops de eixo são menores (padrão: 224)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="")
    p.add_argument("--base-weights", default="models/yolov8n.pt", help="Checkpoint base do YOLO")
    p.add_argument(
        "--pretrained-cache",
        type=Path,
        default=Path("models/pretrained_cache"),
        help="Diretório do cache de modelos pré-treinados",
    )
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--onnx-output", type=Path, default=ONNX_OUTPUT)
    return p.parse_args()


def resolve_base_weights(args: argparse.Namespace) -> str:
    if args.auto:
        args.use_pretrained = True

    if not args.use_pretrained:
        return args.base_weights

    cached = args.pretrained_cache / "weights__axles-composite" / "best.pt"
    if cached.is_file():
        print(f"[INFO] Peso pré-treinado em cache: {cached}")
        return str(cached)

    print(f"[INFO] Cache de pré-treinado ausente em {cached}; usando {args.base_weights}")
    return args.base_weights


def build_dataset(args: argparse.Namespace) -> Path:
    staging_images = STAGING / "images"
    staging_labels = STAGING / "labels"
    if STAGING.exists():
        shutil.rmtree(STAGING)
    staging_images.mkdir(parents=True)
    staging_labels.mkdir(parents=True)

    total = 0

    if not args.skip_zenodo:
        try:
            dax.fetch_zenodo(ZENODO_DIR, force=args.force_zenodo)
            total += dax.convert_zenodo_labelme(ZENODO_DIR, staging_images, staging_labels)
        except SystemExit as exc:
            print(f"[WARN] Zenodo pulado: {exc}")
        except Exception as exc:
            print(f"[WARN] Zenodo falhou: {exc}")

    if not args.skip_roboflow:
        try:
            loc = mlc.download_roboflow_dataset(
                args.roboflow_workspace,
                args.roboflow_project,
                ROBOFLOW_DIR,
                version=args.roboflow_version,
                model_format="yolov8",
            )
            total += dax.convert_roboflow_yolo(loc, staging_images, staging_labels, prefix="rf_eixos")
        except SystemExit as exc:
            print(f"[WARN] Roboflow eixos pulado: {exc}")
        except Exception as exc:
            print(f"[WARN] Roboflow eixos falhou: {exc}")

    if not args.skip_kaggle:
        if args.kaggle_dir.is_dir():
            try:
                total += dax.convert_kaggle_wheels(
                    args.kaggle_dir, staging_images, staging_labels, prefix="kaggle"
                )
            except SystemExit as exc:
                print(f"[WARN] Kaggle pulado: {exc}")
            except Exception as exc:
                print(f"[WARN] Kaggle falhou: {exc}")
        else:
            print(
                f"[WARN] Pasta Kaggle ausente ({args.kaggle_dir}) — "
                "baixе manualmente e passe --kaggle-dir, ou use --skip-kaggle"
            )

    if total == 0:
        raise SystemExit(
            "Nenhuma imagem convertida. Verifique Zenodo/Roboflow/Kaggle "
            "(ROBOFLOW_API_KEY, --kaggle-dir, rede para Zenodo)."
        )

    return dax.merge_and_split(
        staging_images,
        staging_labels,
        MERGED,
        class_name="axle",
        val_ratio=args.val_ratio,
    )


def main() -> int:
    args = parse_args()
    base_weights = resolve_base_weights(args)

    if args.skip_dataset:
        yaml_path = MERGED / "data.yaml"
        if not yaml_path.is_file():
            print(f"[ERRO] Dataset não encontrado: {yaml_path}", file=sys.stderr)
            return 1
        print(f"[INFO] Reusando dataset: {yaml_path}")
    else:
        yaml_path = build_dataset(args)

    if args.only_dataset:
        print(f"[INFO] --only-dataset: pronto em {yaml_path}")
        return 0

    best = mlc.train_yolo(
        yaml_path,
        base_weights=base_weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=PROJECT_RUNS,
        name="train",
    )
    mlc.export_onnx(best, args.imgsz, args.onnx_output)
    print(f"[OK] Modelo de eixos: {args.onnx_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
