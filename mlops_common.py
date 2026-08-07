#!/usr/bin/env python3
"""
mlops_common.py — funções compartilhadas de treino/export YOLOv8 → ONNX
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
    Converte detecções ultralytics → linhas YOLO.
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
# Treino / export YOLOv8n → ONNX
# ---------------------------------------------------------------------------


def train_yolo(
    data_yaml: Path,
    *,
    base_weights: str = "yolov8n.pt",
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
    dynamic=False  → input fixo [1,3,imgsz,imgsz] (amigável ao Detector C++).
    simplify=True  → onnxsim; grafo mais enxuto na borda.
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
    print(f"[INFO] ONNX copiado → {onnx_output.resolve()} ({onnx_output.stat().st_size} bytes)")
    print(
        "[INFO] Flags: simplify=True, dynamic=False, imgsz="
        f"{imgsz} — pronto para ./build/contador_eixo --model {onnx_output}"
    )
    return onnx_output
