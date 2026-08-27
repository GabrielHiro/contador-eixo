#!/usr/bin/env python3
"""
mlops_common.py — funções compartilhadas de treino/export YOLOv8 -> ONNX
e de rotulagem zero-shot (YOLO-World) usadas por:

  - treinar_modelo.py     (dataset baixado do Roboflow)
  - treinar_do_video.py   (dataset gerado localmente a partir de vídeos brutos)
  - validador_yoloworld.py (validação visual + auto-labeling opcional)

Mantém as mesmas convenções de export usadas no binário C++ (Detector ONNX Runtime):
  - simplify=True
  - dynamic=False   (shape fixo; evita surpresas no Ort::Session em Armbian)
  - opset=12        (amplo suporte no ORT 1.x da borda)
"""

from __future__ import annotations

import time
import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Rotulagem zero-shot (YOLO-World)
# ---------------------------------------------------------------------------

# Prompts zero-shot focados em rodas/eixos na perspectiva lateral/diagonal.
WHEEL_CLASS_PROMPTS = [
    "car wheel",
    "truck wheel",
    "vehicle tire",
    "axle",
]


def boxes_to_yolo_txt(result, img_w: int, img_h: int) -> str:
    """
    Converte detecções ultralytics -> linhas YOLO.
    Todas as classes de prompt são mapeadas para class_id=0 (wheel),
    alinhado ao Detector C++ de uma classe.
    """
    lines: list[str] = []
    if result.boxes is None or len(result.boxes) == 0:
        return ""

    xyxy = result.boxes.xyxy.cpu().numpy()
    for box in xyxy:
        x1, y1, x2, y2 = map(float, box)
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw < 1.0 or bh < 1.0:
            continue
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        nw = bw / img_w
        nh = bh / img_h
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# Download de datasets públicos (Roboflow Universe)
# ---------------------------------------------------------------------------


def download_roboflow_dataset(
    workspace: str,
    project: str,
    dest: Path,
    *,
    version: int | None = None,
    model_format: str = "yolov8",
    api_key_env: str = "ROBOFLOW_API_KEY",
    overwrite: bool = True,
) -> Path:
    """
    Baixa um dataset público do Roboflow Universe via SDK.

    A chave é lida de ``os.environ[api_key_env]`` (padrão: ROBOFLOW_API_KEY).
    Nunca hardcodeie a chave no repositório.
    """
    import roboflow

    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"Defina a variável de ambiente {api_key_env} com sua chave do Roboflow.\n"
            f"  Windows PowerShell: $env:{api_key_env}='rf_...'\n"
            f"  bash:               export {api_key_env}=rf_..."
        )

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Roboflow: {workspace}/{project}" + (f" v{version}" if version else " (última versão)"))
    rf = roboflow.Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)

    if version is not None:
        ver = proj.version(version)
    else:
        versions = list(proj.versions())
        if not versions:
            raise SystemExit(f"Projeto {workspace}/{project} sem versões publicadas")
        # versions() retorna objetos Version; o mais recente costuma ser o de maior número.
        latest = max(versions, key=lambda v: int(getattr(v, "version", 0) or 0))
        ver = proj.version(int(latest.version))

    ds = ver.download(model_format, location=str(dest), overwrite=overwrite)
    location = Path(getattr(ds, "location", dest))
    print(f"[INFO] Dataset Roboflow em {location.resolve()}")
    return location


# ---------------------------------------------------------------------------
# Treino / export YOLOv8n -> ONNX
# ---------------------------------------------------------------------------


def train_yolo(
    data_yaml: Path,
    *,
    base_weights: str = "models/yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 4,
    device: str = "",
    project: Path = Path("runs/wheels"),
    name: str = "train",
) -> Path:
    """Treina YOLOv8n e retorna o caminho de weights/best.pt."""
    from ultralytics import YOLO

    print(f"[INFO] Carregando pesos base: {base_weights}")
    model = YOLO(base_weights)

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
    }
    if device != "":
        train_kwargs["device"] = device

    print(
        f"[INFO] Treino: epochs={epochs} imgsz={imgsz} "
        f"batch={batch} device={device or 'auto'}"
    )
    model.train(**train_kwargs)

    # Não confiar em project/name para montar o caminho: dependendo da versão do
    # ultralytics e do runs_dir configurado em ~/.config/Ultralytics/settings.json,
    # o save_dir efetivo pode ficar aninhado diferente do esperado (ex.: prefixado
    # com o nome da task). model.trainer.save_dir é a fonte da verdade.
    save_dir = Path(model.trainer.save_dir) if getattr(model, "trainer", None) else Path(project) / name
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        raise FileNotFoundError(f"best.pt não encontrado em {best_pt}")

    print(f"[INFO] Melhor checkpoint: {best_pt.resolve()}")
    return best_pt


def export_onnx(weights: Path, imgsz: int, onnx_output: Path) -> Path:
    """
    Exporta para ONNX estático + simplificado.
    dynamic=False  -> input fixo [1,3,imgsz,imgsz] (amigável ao Detector C++).
    simplify=True  -> onnxsim; grafo mais enxuto na borda.
    """
    from ultralytics import YOLO

    print(f"[INFO] Export ONNX a partir de {weights}")
    model = YOLO(str(weights))

    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        dynamic=False,
        opset=12,
    )

    src = Path(exported)
    if not src.is_file():
        # Fallback típico: mesmo stem do .pt
        candidate = weights.with_suffix(".onnx")
        if candidate.is_file():
            src = candidate
        else:
            raise FileNotFoundError(f"Export ONNX não gerou arquivo: {exported}")

    onnx_output = Path(onnx_output)
    onnx_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, onnx_output)
    print(f"[INFO] ONNX copiado -> {onnx_output.resolve()} ({onnx_output.stat().st_size} bytes)")
    print(
        "[INFO] Flags: simplify=True, dynamic=False, imgsz="
        f"{imgsz} — pronto para ./build/contador_eixo --model {onnx_output}"
    )
    return onnx_output


def merge_pretrained_with_local_dataset(
    pretrained_weights: str,
    local_data_yaml: str,
    output_dir: Path,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 4,
    device: str = "",
) -> Path:
    """
    Faz fine-tuning de um checkpoint pré-treinado no dataset local e retorna best.pt.

    O nome segue o contrato pedido no prompt, mas a implementação reaproveita o
    fluxo já existente de treino YOLOv8.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_pt = train_yolo(
        Path(local_data_yaml),
        base_weights=pretrained_weights,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=output_dir,
        name="fine_tune",
    )
    return best_pt


def compare_models(model_paths: list[str], test_video: str) -> dict:
    """
    Compara múltiplos modelos ONNX/YOLO no mesmo vídeo de teste.

    Retorna métricas simples e estáveis para seleção rápida:
    - fps
    - detections_per_frame
    - mean_confidence
    """
    import cv2
    from ultralytics import YOLO

    video_path = Path(test_video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Vídeo de teste não encontrado: {video_path}")

    results: dict[str, dict[str, float]] = {}
    max_frames = 120

    for model_path_str in model_paths:
        model_path = Path(model_path_str)
        if not model_path.is_file():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        model = YOLO(str(model_path))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

        frame_count = 0
        total_detections = 0
        confidence_sum = 0.0
        detections_with_conf = 0
        started = time.perf_counter()

        while frame_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            prediction = model.predict(frame, verbose=False)
            result = prediction[0]
            boxes = getattr(result, "boxes", None)
            n_boxes = 0 if boxes is None else len(boxes)
            total_detections += n_boxes
            if boxes is not None and n_boxes > 0 and getattr(boxes, "conf", None) is not None:
                confidence_sum += float(boxes.conf.mean().item())
                detections_with_conf += 1

            frame_count += 1

        elapsed = max(time.perf_counter() - started, 1e-9)
        cap.release()

        results[str(model_path)] = {
            "fps": frame_count / elapsed,
            "detections_per_frame": (total_detections / frame_count) if frame_count else 0.0,
            "mean_confidence": (confidence_sum / detections_with_conf) if detections_with_conf else 0.0,
            "frames": float(frame_count),
        }

    return results
