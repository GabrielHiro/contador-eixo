#!/usr/bin/env python3
"""
Validador Zero-Shot de rodas/eixos com YOLO-World (ultralytics).

Objetivo: antes de treinar um modelo leve para a borda (C++/ONNX), verificar se
as rodas são detectáveis no ângulo defasado da câmera usando apenas prompts de texto.

Uso:
  python validador_yoloworld.py video_teste.mp4
    python validador_yoloworld.py video_teste.mp4 --conf 0.12 --model models/yolov8s-world.pt
  python validador_yoloworld.py video_teste.mp4 --save-dataset

Teclas:
  q — sair
  p — pausar / despausar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLOWorld

import mlops_common as mlc

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# Prompts zero-shot focados em rodas/eixos na perspectiva lateral/diagonal.
CLASS_PROMPTS = mlc.WHEEL_CLASS_PROMPTS

# Bônus: prepare auto-labeling no formato YOLO (classe cx cy w h normalizados).
# Pode ser ligado via CLI (--save-dataset) ou alterando o default abaixo.
SAVE_DATASET = False
DATASET_DIR = Path("dataset_yoloworld")

DEFAULT_MODEL = "models/yolov8s-world.pt"
DEFAULT_CONF = 0.15
WINDOW_NAME = "YOLO-World — Contador de Eixos (validação)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validação zero-shot de rodas/eixos com YOLO-World",
    )
    parser.add_argument(
        "video",
        nargs="?",
        default="video_teste.mp4",
        help="Caminho do vídeo local (padrão: video_teste.mp4)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Checkpoint YOLO-World (padrão: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"Threshold de confiança (padrão: {DEFAULT_CONF})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamanho de inferência (padrão: 640)",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Device ultralytics (ex: cpu, 0). Vazio = auto",
    )
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        default=SAVE_DATASET,
        help="Salva frames + labels YOLO em dataset_yoloworld/",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Processa 1 a cada N+1 frames (0 = todos). Útil em vídeos longos.",
    )
    return parser.parse_args()


def ensure_dataset_dirs(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    # data.yaml mínimo para treino futuro (1 classe unificada "wheel")
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(
            "path: .\n"
            "train: images\n"
            "val: images\n"
            "names:\n"
            "  0: wheel\n",
            encoding="utf-8",
        )
    return images, labels


def save_auto_label(
    frame_bgr,
    result,
    frame_idx: int,
    images_dir: Path,
    labels_dir: Path,
) -> None:
    stem = f"frame_{frame_idx:06d}"
    img_path = images_dir / f"{stem}.jpg"
    lbl_path = labels_dir / f"{stem}.txt"

    h, w = frame_bgr.shape[:2]
    cv2.imwrite(str(img_path), frame_bgr)
    lbl_path.write_text(mlc.boxes_to_yolo_txt(result, w, h), encoding="utf-8")


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)

    if not video_path.is_file():
        print(f"[ERRO] Vídeo não encontrado: {video_path.resolve()}", file=sys.stderr)
        print("Coloque video_teste.mp4 na pasta atual ou passe o caminho como argumento.")
        return 1

    print(f"[INFO] Carregando modelo {args.model} ...")
    model = YOLOWorld(args.model)
    model.set_classes(CLASS_PROMPTS)
    print(f"[INFO] Classes/prompts: {CLASS_PROMPTS}")
    print(f"[INFO] conf={args.conf}  imgsz={args.imgsz}")

    images_dir = labels_dir = None
    if args.save_dataset:
        images_dir, labels_dir = ensure_dataset_dirs(DATASET_DIR)
        print(f"[INFO] Auto-labeling ATIVO → {DATASET_DIR.resolve()}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo: {video_path}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] Vídeo: {video_path} | ~{fps:.1f} FPS | frames={total}")
    print("[INFO] Teclas: [q] sair  |  [p] pausar/despausar")

    paused = False
    frame_idx = 0
    saved_count = 0
    delay_ms = max(1, int(1000 / fps))
    last_display = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    predict_kwargs = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] Fim do vídeo.")
                break

            run_infer = args.skip <= 0 or (frame_idx % (args.skip + 1) == 0)

            if run_infer:
                results = model.predict(frame, **predict_kwargs)
                annotated = results[0].plot()

                if args.save_dataset and images_dir is not None and labels_dir is not None:
                    # Só grava se houver pelo menos uma detecção (evita lixo no dataset)
                    n_det = 0 if results[0].boxes is None else len(results[0].boxes)
                    if n_det > 0:
                        save_auto_label(frame, results[0], frame_idx, images_dir, labels_dir)
                        saved_count += 1
            else:
                annotated = frame.copy()

            status = f"frame {frame_idx}/{total}  conf>={args.conf}"
            if args.save_dataset:
                status += f"  saved={saved_count}"
            if paused:
                status += "  [PAUSED]"
            cv2.putText(
                annotated,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (40, 220, 40),
                2,
                cv2.LINE_AA,
            )
            last_display = annotated
            cv2.imshow(WINDOW_NAME, annotated)
            frame_idx += 1
            wait = delay_ms
        else:
            if last_display is not None:
                hud = last_display.copy()
                cv2.putText(
                    hud,
                    "PAUSED — pressione 'p' para continuar",
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, hud)
            wait = 50

        key = cv2.waitKey(wait) & 0xFF
        if key == ord("q"):
            print("[INFO] Encerrado pelo usuário (q).")
            break
        if key == ord("p"):
            paused = not paused
            print("[INFO] Pausado." if paused else "[INFO] Retomado.")

    cap.release()
    cv2.destroyAllWindows()

    if args.save_dataset:
        print(f"[INFO] Dataset: {saved_count} amostras em {DATASET_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
