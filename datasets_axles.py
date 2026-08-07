#!/usr/bin/env python3
"""
datasets_axles.py — conversores e merge para o dataset de eixos (classe única `axle`).

Fontes suportadas:
  - Zenodo Truck Image Dataset (LabelMe JSON, DOI 10.5281/zenodo.17128113)
  - Roboflow YOLO export (ex.: eixosdecaminhao) — todas as classes -> 0 (axle)
  - Kaggle vehicle-wheel-detection (pasta local) — só a classe `wheel` -> 0 (axle)

Uso típico (via treinar_eixos.py):
  from datasets_axles import fetch_zenodo, convert_zenodo_labelme, merge_and_split
"""

from __future__ import annotations

import json
import random
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ZENODO_RECORD = "17128113"
ZENODO_ZIP_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/Trucks.zip/content"
)
ZENODO_ZIP_NAME = "Trucks.zip"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _copy_pair(img_src: Path, label_lines: list[str], out_images: Path, out_labels: Path, stem: str) -> None:
    ext = img_src.suffix.lower() or ".jpg"
    dst_img = out_images / f"{stem}{ext}"
    dst_lbl = out_labels / f"{stem}.txt"
    shutil.copy2(img_src, dst_img)
    dst_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")


def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> str | None:
    if img_w <= 0 or img_h <= 0:
        return None
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    bw = x2 - x1
    bh = y2 - y1
    if bw < 1.0 or bh < 1.0:
        return None
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    nw = bw / img_w
    nh = bh / img_h
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def _shape_to_xyxy(shape: dict) -> tuple[float, float, float, float] | None:
    """Converte shape LabelMe (rectangle ou polygon) para bbox xyxy."""
    pts = shape.get("points") or []
    if len(pts) < 2:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Zenodo
# ---------------------------------------------------------------------------


def fetch_zenodo(dest_dir: Path, *, force: bool = False) -> Path:
    """
    Baixa Trucks.zip (~979 MB) do Zenodo e extrai em dest_dir.
    Retorna o diretório raiz extraído (ou dest_dir se já existir conteúdo).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / ZENODO_ZIP_NAME

    # Já convertido/extraído?
    jsons = list(dest_dir.rglob("*.json"))
    jpgs = list(dest_dir.rglob("*.jpg")) + list(dest_dir.rglob("*.JPG"))
    if not force and jsons and jpgs:
        print(f"[INFO] Zenodo já presente em {dest_dir} ({len(jpgs)} imagens, {len(jsons)} JSONs)")
        return dest_dir

    if force or not zip_path.is_file():
        print(f"[INFO] Baixando Zenodo Trucks.zip (~979 MB) -> {zip_path}")
        print(f"[INFO] URL: {ZENODO_ZIP_URL}")

        def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size <= 0:
                return
            done = block_num * block_size
            pct = min(100.0, 100.0 * done / total_size)
            if block_num % 200 == 0 or done >= total_size:
                print(f"\r[INFO] Download Zenodo: {pct:5.1f}% ({done // (1024 * 1024)} / {total_size // (1024 * 1024)} MB)", end="", flush=True)

        try:
            urllib.request.urlretrieve(ZENODO_ZIP_URL, zip_path, reporthook=_reporthook)
            print()
        except Exception as exc:
            raise SystemExit(f"Falha ao baixar Zenodo: {exc}") from exc
    else:
        print(f"[INFO] Reusando zip local: {zip_path}")

    print(f"[INFO] Extraindo {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return dest_dir


def convert_zenodo_labelme(
    src_dir: Path,
    out_images: Path,
    out_labels: Path,
    *,
    prefix: str = "zenodo",
    axle_labels: frozenset[str] | None = None,
) -> int:
    """
    Lê JSONs LabelMe do Zenodo; filtra shapes com label Axle (case-insensitive)
    e grava pares imagem/label YOLO com classe 0 (axle).
    """
    if axle_labels is None:
        axle_labels = frozenset({"axle"})

    src_dir = Path(src_dir)
    _ensure_dirs(out_images, out_labels)
    count = 0

    for json_path in sorted(src_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] JSON inválido {json_path}: {exc}")
            continue

        img_name = data.get("imagePath") or (json_path.stem + ".jpg")
        # LabelMe às vezes aponta path relativo; procurar ao lado do JSON e sob src_dir.
        candidates = [
            json_path.parent / img_name,
            json_path.parent / Path(img_name).name,
        ]
        stem_guess = Path(img_name).stem
        for ext in (".jpg", ".JPG", ".jpeg", ".png"):
            candidates.append(json_path.parent / f"{stem_guess}{ext}")
        img_path = next((c for c in candidates if c.is_file()), None)
        if img_path is None:
            # Busca global pelo nome do arquivo
            matches = list(src_dir.rglob(Path(img_name).name))
            img_path = matches[0] if matches else None
        if img_path is None or not img_path.is_file():
            print(f"[WARN] Imagem não encontrada para {json_path.name}")
            continue

        img_w = int(data.get("imageWidth") or 0)
        img_h = int(data.get("imageHeight") or 0)
        if img_w <= 0 or img_h <= 0:
            try:
                import cv2

                im = cv2.imread(str(img_path))
                if im is None:
                    continue
                img_h, img_w = im.shape[:2]
            except Exception:
                continue

        lines: list[str] = []
        for shape in data.get("shapes") or []:
            label = str(shape.get("label") or "").strip().lower()
            if label not in axle_labels:
                continue
            xyxy = _shape_to_xyxy(shape)
            if xyxy is None:
                continue
            line = _xyxy_to_yolo(*xyxy, img_w, img_h)
            if line:
                lines.append(line)

        if not lines:
            continue

        stem = f"{prefix}_{count:05d}_{json_path.stem}"
        _copy_pair(img_path, lines, out_images, out_labels, stem)
        count += 1

    print(f"[INFO] Zenodo -> {count} imagens com ao menos 1 Axle")
    return count


# ---------------------------------------------------------------------------
# Roboflow YOLO
# ---------------------------------------------------------------------------


def convert_roboflow_yolo(
    src_dir: Path,
    out_images: Path,
    out_labels: Path,
    *,
    prefix: str = "rf",
) -> int:
    """
    Ingere export YOLOv8 do Roboflow (train/valid/test com images/ + labels/).
    Remapeia qualquer class_id para 0 (axle).
    """
    src_dir = Path(src_dir)
    _ensure_dirs(out_images, out_labels)
    count = 0

    image_files: list[Path] = []
    for split in ("train", "valid", "val", "test"):
        # Layout Roboflow: train/images
        img_dir = src_dir / split / "images"
        if img_dir.is_dir():
            image_files.extend(sorted(img_dir.glob("*")))
        # Layout Ultralytics/local: images/train
        img_dir2 = src_dir / "images" / split
        if img_dir2.is_dir():
            image_files.extend(sorted(img_dir2.glob("*")))
    # Layout flat: images/*.jpg
    if not image_files and (src_dir / "images").is_dir():
        image_files = [
            p
            for p in sorted((src_dir / "images").rglob("*"))
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    if not image_files:
        image_files = [
            p
            for p in sorted(src_dir.rglob("*"))
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            and "label" not in {part.lower() for part in p.parts}
        ]

    for img_path in image_files:
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        # Label irmao tipico:
        #   .../images/x.jpg -> .../labels/x.txt
        #   .../images/train/x.jpg -> .../labels/train/x.txt  (Ultralytics)
        #   .../train/images/x.jpg -> .../train/labels/x.txt  (Roboflow)
        label_candidates = [
            img_path.parent.parent / "labels" / img_path.parent.name / f"{img_path.stem}.txt",
            img_path.parent.parent / "labels" / f"{img_path.stem}.txt",
            img_path.parent / f"{img_path.stem}.txt",
            src_dir / "labels" / img_path.parent.name / f"{img_path.stem}.txt",
            src_dir / "labels" / f"{img_path.stem}.txt",
        ]
        label_path = next((c for c in label_candidates if c.is_file()), None)
        lines: list[str] = []
        if label_path is not None:
            for raw in label_path.read_text(encoding="utf-8").splitlines():
                parts = raw.strip().split()
                if len(parts) < 5:
                    continue
                # remapeia class_id -> 0
                lines.append(f"0 {' '.join(parts[1:5])}")

        if not lines:
            continue

        stem = f"{prefix}_{count:05d}_{img_path.stem}"
        _copy_pair(img_path, lines, out_images, out_labels, stem)
        count += 1

    print(f"[INFO] Roboflow YOLO -> {count} imagens")
    return count


# ---------------------------------------------------------------------------
# Kaggle (defensivo: YOLO / COCO / VOC)
# ---------------------------------------------------------------------------


def _convert_kaggle_yolo(src_dir: Path, out_images: Path, out_labels: Path, prefix: str) -> int:
    """YOLO com data.yaml — filtra apenas classes cujo nome contém 'wheel'."""
    yaml_files = list(src_dir.rglob("data.yaml")) + list(src_dir.rglob("data.yml"))
    class_filter: set[int] | None = None
    if yaml_files:
        text = yaml_files[0].read_text(encoding="utf-8")
        # parse rudimentar de names
        names: list[str] = []
        in_names = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("names:"):
                rest = stripped[len("names:") :].strip()
                if rest.startswith("["):
                    names = [n.strip().strip("'\"") for n in rest.strip("[]").split(",") if n.strip()]
                    in_names = False
                else:
                    in_names = True
                continue
            if in_names:
                if ":" in stripped and stripped[0].isdigit():
                    names.append(stripped.split(":", 1)[1].strip().strip("'\""))
                elif stripped.startswith("-"):
                    names.append(stripped.lstrip("-").strip().strip("'\""))
                elif not stripped or not stripped[0].isspace() and not stripped[0].isdigit() and not stripped.startswith("-"):
                    in_names = False
        wheel_ids = {i for i, n in enumerate(names) if "wheel" in n.lower()}
        if wheel_ids:
            class_filter = wheel_ids
            print(f"[INFO] Kaggle YOLO: filtrando class_ids={sorted(wheel_ids)} ({[names[i] for i in sorted(wheel_ids)]})")

    count = 0
    for img_path in sorted(src_dir.rglob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if "label" in {p.lower() for p in img_path.parts}:
            continue
        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if not label_path.is_file():
            label_path = img_path.with_suffix(".txt")
        if not label_path.is_file():
            alt = img_path.parent / "labels" / f"{img_path.stem}.txt"
            label_path = alt if alt.is_file() else label_path
        if not label_path.is_file():
            continue

        lines: list[str] = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            if class_filter is not None and cid not in class_filter:
                continue
            lines.append(f"0 {' '.join(parts[1:5])}")

        if not lines:
            continue
        stem = f"{prefix}_{count:05d}_{img_path.stem}"
        _copy_pair(img_path, lines, out_images, out_labels, stem)
        count += 1
    return count


def _convert_kaggle_coco(src_dir: Path, out_images: Path, out_labels: Path, prefix: str) -> int:
    ann_files = list(src_dir.rglob("*annotations*.json")) + list(src_dir.rglob("instances_*.json"))
    if not ann_files:
        return 0

    count = 0
    for ann_path in ann_files:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        wheel_cat_ids = {cid for cid, name in cats.items() if "wheel" in name.lower()}
        if not wheel_cat_ids:
            # se só houver 1 categoria, assume wheel
            if len(cats) == 1:
                wheel_cat_ids = set(cats.keys())
            else:
                print(f"[WARN] COCO {ann_path}: nenhuma categoria 'wheel' em {list(cats.values())}")
                continue

        images = {im["id"]: im for im in data.get("images", [])}
        by_image: dict[int, list[str]] = {}
        for ann in data.get("annotations", []):
            if ann.get("category_id") not in wheel_cat_ids:
                continue
            bbox = ann.get("bbox")  # xywh
            if not bbox or len(bbox) < 4:
                continue
            im = images.get(ann["image_id"])
            if not im:
                continue
            x, y, w, h = map(float, bbox[:4])
            line = _xyxy_to_yolo(x, y, x + w, y + h, int(im["width"]), int(im["height"]))
            if line:
                by_image.setdefault(ann["image_id"], []).append(line)

        for img_id, lines in by_image.items():
            im = images[img_id]
            file_name = im["file_name"]
            candidates = list(src_dir.rglob(Path(file_name).name))
            if not candidates:
                continue
            stem = f"{prefix}_{count:05d}_{Path(file_name).stem}"
            _copy_pair(candidates[0], lines, out_images, out_labels, stem)
            count += 1
    return count


def _convert_kaggle_voc(src_dir: Path, out_images: Path, out_labels: Path, prefix: str) -> int:
    xmls = list(src_dir.rglob("*.xml"))
    if not xmls:
        return 0
    count = 0
    for xml_path in xmls:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        size = root.find("size")
        if size is None:
            continue
        img_w = int(size.findtext("width", "0"))
        img_h = int(size.findtext("height", "0"))
        lines: list[str] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").lower()
            if "wheel" not in name:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            x1 = float(bnd.findtext("xmin", "0"))
            y1 = float(bnd.findtext("ymin", "0"))
            x2 = float(bnd.findtext("xmax", "0"))
            y2 = float(bnd.findtext("ymax", "0"))
            line = _xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            if line:
                lines.append(line)
        if not lines:
            continue
        img_name = root.findtext("filename") or (xml_path.stem + ".jpg")
        candidates = list(src_dir.rglob(Path(img_name).name))
        if not candidates:
            continue
        stem = f"{prefix}_{count:05d}_{xml_path.stem}"
        _copy_pair(candidates[0], lines, out_images, out_labels, stem)
        count += 1
    return count


def convert_kaggle_wheels(
    src_dir: Path,
    out_images: Path,
    out_labels: Path,
    *,
    prefix: str = "kaggle",
) -> int:
    """
    Ingestão defensiva do dataset Kaggle vehicle-wheel-detection.
    Tenta YOLO -> COCO -> Pascal VOC; aborta com mensagem clara se nenhum funcionar.
    """
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise SystemExit(
            f"Pasta Kaggle não encontrada: {src_dir}\n"
            "Baixe manualmente em https://www.kaggle.com/datasets/dataclusterlabs/vehicle-wheel-detection\n"
            "e extraia em datasets/external/vehicle-wheel-detection/"
        )

    _ensure_dirs(out_images, out_labels)
    print(f"[INFO] Ingerindo Kaggle de {src_dir}…")

    count = _convert_kaggle_yolo(src_dir, out_images, out_labels, prefix)
    if count:
        print(f"[INFO] Kaggle (YOLO) -> {count} imagens")
        return count

    count = _convert_kaggle_coco(src_dir, out_images, out_labels, prefix)
    if count:
        print(f"[INFO] Kaggle (COCO) -> {count} imagens")
        return count

    count = _convert_kaggle_voc(src_dir, out_images, out_labels, prefix)
    if count:
        print(f"[INFO] Kaggle (VOC) -> {count} imagens")
        return count

    raise SystemExit(
        f"Não foi possível detectar formato YOLO/COCO/VOC em {src_dir}.\n"
        "Informe a estrutura real da pasta (liste algumas subpastas) para ajustar o parser."
    )


# ---------------------------------------------------------------------------
# Merge + split + data.yaml
# ---------------------------------------------------------------------------


def merge_and_split(
    staging_images: Path,
    staging_labels: Path,
    out_root: Path,
    *,
    class_name: str = "axle",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """
    Junta imagens/labels de staging em out_root/{images,labels}/{train,val}
    e gera data.yaml com path absoluto.
    """
    staging_images = Path(staging_images)
    staging_labels = Path(staging_labels)
    out_root = Path(out_root)

    pairs: list[tuple[Path, Path]] = []
    for img in sorted(staging_images.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        lbl = staging_labels / f"{img.stem}.txt"
        if lbl.is_file():
            pairs.append((img, lbl))

    if not pairs:
        raise SystemExit("Nenhum par imagem/label no staging — nada para fazer merge")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * val_ratio))) if len(pairs) > 1 else 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:] or pairs  # garante train não vazio

    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        img_out = out_root / "images" / split
        lbl_out = out_root / "labels" / split
        if img_out.exists():
            shutil.rmtree(img_out)
        if lbl_out.exists():
            shutil.rmtree(lbl_out)
        _ensure_dirs(img_out, lbl_out)
        for img, lbl in split_pairs:
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(lbl, lbl_out / lbl.name)

    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(
        f"path: {out_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {class_name}\n",
        encoding="utf-8",
    )
    print(
        f"[INFO] Merge: train={len(train_pairs)} val={len(val_pairs)} -> {yaml_path.resolve()}"
    )
    return yaml_path
