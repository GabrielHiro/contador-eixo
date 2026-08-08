#!/usr/bin/env python3
"""
treinar_do_video.py — MLOps 100% local: vídeos brutos → modelo pronto para a borda.

Pipeline:
  1) Auto-rotulagem zero-shot (YOLO-World) de cada vídeo, amostrando frames.
  2) Split cronológico por vídeo (últimos --val-ratio % de cada vídeo → val),
     reduzindo vazamento entre frames quase-idênticos (vídeo é muito redundante).
  3) Treino YOLOv8n (ultralytics) no dataset gerado.
  4) Export ONNX estático/simplificado → models/vehicles.onnx (consumido pelo
     Detector ONNX Runtime do contador_eixo).

Uso (com venv ativado):
  python treinar_do_video.py
  python treinar_do_video.py --videos 181327--vv.mp4 181349--vv.mp4 --conf 0.12
  python treinar_do_video.py --only-dataset          # só gera o dataset, não treina
  python treinar_do_video.py --skip-dataset          # reusa dataset já gerado, só treina
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

import mlops_common as mlc

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

DEFAULT_VIDEOS = ["181327--vv.mp4", "181349--vv.mp4"]

WORLD_MODEL = "yolov8s-world.pt"   # checkpoint YOLO-World p/ auto-rotulagem
BASE_WEIGHTS = "yolov8n.pt"        # nano — leve para edge

DATASET_DIR = Path("datasets/wheels_video")
PROJECT_RUNS = Path("runs/wheels_video")
RUN_NAME = "train"
ONNX_OUTPUT = Path("models/vehicles.onnx")

# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-rotula vídeos com YOLO-World, treina YOLOv8n (wheels) e exporta ONNX",
    )
    p.add_argument(
        "--videos",
        nargs="+",
        default=DEFAULT_VIDEOS,
        help=f"Vídeos de treino (padrão: {' '.join(DEFAULT_VIDEOS)})",
    )
    p.add_argument("--model", default=WORLD_MODEL, help=f"Checkpoint YOLO-World (padrão: {WORLD_MODEL})")
    p.add_argument("--conf", type=float, default=0.12, help="Limiar de confiança YOLO-World (padrão: 0.12)")
    p.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Prompts zero-shot (padrão: prompts de roda/eixo em mlops_common.WHEEL_CLASS_PROMPTS). "
        "Ex.: --prompts car vehicle  (útil quando a câmera não mostra rodas, só o veículo)",
    )
    p.add_argument(
        "--class-name",
        default="wheel",
        help="Nome da classe única gravada em data.yaml (padrão: wheel)",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--base-weights", default=BASE_WEIGHTS, help="Checkpoint base do YOLO")
    p.add_argument(
        "--skip",
        type=int,
        default=4,
        help="Processa 1 a cada N+1 frames (padrão: 4 — vídeo é muito redundante)",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fração final (cronológica) de cada vídeo reservada para validação (padrão: 0.15)",
    )
    p.add_argument("--device", default="", help="Device ultralytics (ex: cpu, 0). Vazio = auto")
    p.add_argument(
        "--only-dataset",
        action="store_true",
        help="Só gera o dataset rotulado; não treina nem exporta",
    )
    p.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Não rotula de novo; reusa DATASET_DIR já existente",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def ensure_dirs(root: Path) -> dict[str, tuple[Path, Path]]:
    dirs: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "val"):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        dirs[split] = (images, labels)
    return dirs


def write_data_yaml(root: Path, class_name: str) -> Path:
    yaml_path = root / "data.yaml"
    # Caminho absoluto (POSIX-style) evita ambiguidade do ultralytics resolvendo
    # "path: ." a partir do cwd do processo em vez da pasta do próprio data.yaml.
    yaml_path.write_text(
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {class_name}\n",
        encoding="utf-8",
    )
    return yaml_path


def label_video(
    video_path: Path,
    model,
    dirs: dict[str, tuple[Path, Path]],
    args: argparse.Namespace,
    video_tag: str,
) -> dict[str, int]:
    """Roda YOLO-World nos frames amostrados de um vídeo e grava dataset YOLO."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Split cronológico: os últimos val_ratio% dos frames do vídeo vão para val.
    val_start = int(total * (1.0 - args.val_ratio)) if total > 0 else None

    predict_kwargs = {"conf": args.conf, "imgsz": args.imgsz, "verbose": False}
    if args.device:
        predict_kwargs["device"] = args.device

    saved = {"train": 0, "val": 0}
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.skip > 0 and (frame_idx % (args.skip + 1) != 0):
            frame_idx += 1
            continue

        results = model.predict(frame, **predict_kwargs)
        result = results[0]
        n_det = 0 if result.boxes is None else len(result.boxes)

        if n_det > 0:
            split = "val" if (val_start is not None and frame_idx >= val_start) else "train"
            images_dir, labels_dir = dirs[split]
            stem = f"{video_tag}_{frame_idx:06d}"
            h, w = frame.shape[:2]
            cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame)
            (labels_dir / f"{stem}.txt").write_text(
                mlc.boxes_to_yolo_txt(result, w, h), encoding="utf-8"
            )
            saved[split] += 1

        frame_idx += 1

    cap.release()
    print(
        f"[INFO] {video_path.name}: {frame_idx} frames lidos, "
        f"{saved['train']} train / {saved['val']} val rotulados (>=1 detecção)"
    )
    return saved


def build_dataset(args: argparse.Namespace) -> Path:
    from ultralytics import YOLOWorld

    prompts = args.prompts if args.prompts else mlc.WHEEL_CLASS_PROMPTS

    print(f"[INFO] Carregando YOLO-World: {args.model}")
    model = YOLOWorld(args.model)
    model.set_classes(prompts)
    print(f"[INFO] Prompts: {prompts}  (classe única gravada como '{args.class_name}')")
    print(f"[INFO] conf={args.conf}  imgsz={args.imgsz}  skip={args.skip}  val_ratio={args.val_ratio}")

    dirs = ensure_dirs(DATASET_DIR)
    total_saved = {"train": 0, "val": 0}

    for video_name in args.videos:
        video_path = Path(video_name)
        if not video_path.is_file():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path.resolve()}")
        tag = video_path.stem.replace(" ", "_")
        saved = label_video(video_path, model, dirs, args, tag)
        for k in total_saved:
            total_saved[k] += saved[k]

    data_yaml = write_data_yaml(DATASET_DIR, args.class_name)
    print(
        f"[INFO] Dataset pronto: {total_saved['train']} train / {total_saved['val']} val "
        f"→ {data_yaml.resolve()}"
    )

    if total_saved["train"] == 0:
        raise RuntimeError(
            "Nenhum frame com detecção (0 rótulos). Ajuste --conf/--model ou "
            "revise o ângulo da câmera nos vídeos de entrada."
        )
    return data_yaml


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    import os

    os.chdir(root)

    try:
        if args.skip_dataset:
            data_yaml = DATASET_DIR / "data.yaml"
            if not data_yaml.is_file():
                print(
                    f"[ERRO] data.yaml ausente em {data_yaml}. Remova --skip-dataset ou gere antes.",
                    file=sys.stderr,
                )
                return 1
            print(f"[INFO] Reusando dataset: {data_yaml.resolve()}")
        else:
            data_yaml = build_dataset(args)

        if args.only_dataset:
            print("[INFO] --only-dataset: pipeline encerrado sem treinar.")
            return 0

        best_pt = mlc.train_yolo(
            data_yaml,
            base_weights=args.base_weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=PROJECT_RUNS,
            name=RUN_NAME,
        )
        mlc.export_onnx(best_pt, args.imgsz, ONNX_OUTPUT)
    except Exception as exc:  # noqa: BLE001 — script de ops: falha visível
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 1

    print("[INFO] Pipeline concluído com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
