#!/usr/bin/env python3
"""
treinar_veiculos.py — reforça o detector de veículos com Roboflow vehicles-k83q3
+ dataset local dos vídeos (datasets/wheels_video), treina YOLOv8n e exporta
models/vehicles.onnx.

Classes do Roboflow (car/truck/bus/motorcycle) são remapeadas para classe única
`vehicle`, alinhada ao dataset local e ao TrackerCounter de uma classe.

Uso:
  $env:ROBOFLOW_API_KEY='rf_...'
  python treinar_veiculos.py
  python treinar_veiculos.py --skip-roboflow   # só dataset local dos vídeos
  python treinar_veiculos.py --only-dataset
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import datasets_axles as dax
import mlops_common as mlc

LOCAL_VIDEO_DS = Path("datasets/wheels_video")
ROBOFLOW_DIR = Path("datasets/external/vehicles_k83q3")
STAGING = Path("datasets/vehicles_staging")
MERGED = Path("datasets/vehicles_merged")
PROJECT_RUNS = Path("runs/vehicles")
ONNX_OUTPUT = Path("models/vehicles.onnx")

RF_WORKSPACE = "exjobb-dq06p"
RF_PROJECT = "vehicles-k83q3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina detector de veículos -> models/vehicles.onnx")
    p.add_argument("--local-dataset", type=Path, default=LOCAL_VIDEO_DS)
    p.add_argument("--roboflow-workspace", default=RF_WORKSPACE)
    p.add_argument("--roboflow-project", default=RF_PROJECT)
    p.add_argument("--roboflow-version", type=int, default=None)
    p.add_argument("--skip-roboflow", action="store_true")
    p.add_argument("--skip-local", action="store_true")
    p.add_argument("--only-dataset", action="store_true")
    p.add_argument("--skip-dataset", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="")
    p.add_argument("--base-weights", default="models/yolov8n.pt", help="Checkpoint base do YOLO")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--onnx-output", type=Path, default=ONNX_OUTPUT)
    return p.parse_args()


def _ingest_local_yolo(src: Path, out_images: Path, out_labels: Path, prefix: str) -> int:
    """Copia dataset YOLO local (já classe vehicle) para o staging."""
    return dax.convert_roboflow_yolo(src, out_images, out_labels, prefix=prefix)


def build_dataset(args: argparse.Namespace) -> Path:
    staging_images = STAGING / "images"
    staging_labels = STAGING / "labels"
    if STAGING.exists():
        shutil.rmtree(STAGING)
    staging_images.mkdir(parents=True)
    staging_labels.mkdir(parents=True)

    total = 0

    if not args.skip_local:
        if args.local_dataset.is_dir() and (args.local_dataset / "data.yaml").is_file():
            total += _ingest_local_yolo(
                args.local_dataset, staging_images, staging_labels, prefix="video"
            )
        else:
            print(
                f"[WARN] Dataset local ausente ({args.local_dataset}). "
                "Rode antes: python treinar_do_video.py --only-dataset --class-name vehicle"
            )

    if not args.skip_roboflow:
        try:
            loc = mlc.download_roboflow_dataset(
                args.roboflow_workspace,
                args.roboflow_project,
                ROBOFLOW_DIR,
                version=args.roboflow_version,
                model_format="yolov8",
            )
            # Remapeia car/truck/bus/motorcycle -> 0 (vehicle)
            total += dax.convert_roboflow_yolo(
                loc, staging_images, staging_labels, prefix="rf_veh"
            )
        except SystemExit as exc:
            print(f"[WARN] Roboflow veículos pulado: {exc}")
        except Exception as exc:
            print(f"[WARN] Roboflow veículos falhou: {exc}")

    if total == 0:
        raise SystemExit(
            "Nenhuma imagem para veículos. Precisa do dataset local e/ou ROBOFLOW_API_KEY."
        )

    return dax.merge_and_split(
        staging_images,
        staging_labels,
        MERGED,
        class_name="vehicle",
        val_ratio=args.val_ratio,
    )


def main() -> int:
    args = parse_args()

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
        base_weights=args.base_weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=PROJECT_RUNS,
        name="train",
    )
    mlc.export_onnx(best, args.imgsz, args.onnx_output)
    print(f"[OK] Modelo de veículos: {args.onnx_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
