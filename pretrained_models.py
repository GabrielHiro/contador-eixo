#!/usr/bin/env python3
"""
pretrained_models.py — catálogo e utilitários para modelos/datasets pré-treinados.

Este módulo concentra o que o prompt pede em um ponto só:
- catálogo de fontes públicas conhecidas;
- cache local com hash SHA256;
- validação ONNX;
- conversão de checkpoint .pt para ONNX;
- pipeline prático para baixar fonte pública e gerar um modelo ONNX pronto.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import datasets_axles as dax
import mlops_common as mlc


class ModelCache:
    """Cache local simples com hash SHA256 por artefato."""

    def __init__(self, cache_root: Path = Path("models/pretrained_cache")) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def entry_dir(self, model_type: str, model_name: str) -> Path:
        safe_name = f"{model_type}__{model_name}".replace("/", "_")
        path = self.cache_root / safe_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def sha256(self, path: Path) -> str:
        path = Path(path)
        digest = hashlib.sha256()
        if path.is_file():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        for child in sorted(path.rglob("*")):
            if child.is_file():
                digest.update(child.relative_to(path).as_posix().encode("utf-8"))
                with child.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        return digest.hexdigest()

    def store(self, source: Path, model_type: str, model_name: str) -> Path:
        source = Path(source)
        target_dir = self.entry_dir(model_type, model_name)
        target = target_dir / source.name

        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)

        digest = self.sha256(target)
        metadata = {
            "source": source.resolve().as_posix(),
            "cached": target.resolve().as_posix(),
            "sha256": digest,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (target_dir / f"{source.name}.sha256.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target

    def validate(self, path: Path) -> bool:
        path = Path(path)
        meta_path = path.parent / f"{path.name}.sha256.json"
        if not meta_path.is_file():
            return path.exists()

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = metadata.get("sha256")
        return bool(expected) and expected == self.sha256(path)


class PreTrainedModelManager:
    """Gerencia fontes públicas e export ONNX para o projeto contador-eixo."""

    ROBOFLOW_MODELS: dict[str, dict[str, Any]] = {
        "vehicles-k83q3": {
            "family": "vehicle",
            "model_type": "roboflow",
            "workspace": "exjobb-dq06p",
            "project": "vehicles-k83q3",
            "version": None,
            "source_url": "https://universe.roboflow.com/exjobb-dq06p/vehicles-k83q3",
            "description": "Dataset Roboflow de veículos para o estágio 1",
        },
        "eixosdecaminhao": {
            "family": "axle",
            "model_type": "roboflow",
            "workspace": "class-h27po",
            "project": "eixosdecaminhao",
            "version": None,
            "source_url": "https://universe.roboflow.com/class-h27po/eixosdecaminhao",
            "description": "Dataset Roboflow de eixos para o estágio 2",
        },
    }

    ZENODO_MODELS: dict[str, dict[str, Any]] = {
        "labelme-axle": {
            "family": "axle",
            "model_type": "zenodo",
            "record": "17128113",
            "source_url": "https://zenodo.org/api/records/17128113/files/Trucks.zip/content",
            "description": "Zenodo LabelMe Axle / Trucks.zip",
        },
    }

    COMPOSITE_MODELS: dict[str, dict[str, Any]] = {
        "axles-composite": {
            "family": "axle",
            "model_type": "composite",
            "description": "Zenodo LabelMe Axle + Roboflow eixosdecaminhao",
        }
    }

    def __init__(self, cache_root: Path = Path("models/pretrained_cache")) -> None:
        self.cache = ModelCache(cache_root)

    def list_available(self) -> list[dict[str, Any]]:
        available: list[dict[str, Any]] = []
        for name, meta in self.ROBOFLOW_MODELS.items():
            available.append({"name": name, **meta})
        for name, meta in self.ZENODO_MODELS.items():
            available.append({"name": name, **meta})
        for name, meta in self.COMPOSITE_MODELS.items():
            available.append({"name": name, **meta})
        return available

    def _resolve_source(self, model_type: str, model_name: str) -> dict[str, Any]:
        catalog = {
            "roboflow": self.ROBOFLOW_MODELS,
            "zenodo": self.ZENODO_MODELS,
            "composite": self.COMPOSITE_MODELS,
        }
        if model_type not in catalog:
            raise ValueError(f"model_type inválido: {model_type}")
        try:
            return catalog[model_type][model_name]
        except KeyError as exc:
            raise KeyError(f"Modelo desconhecido: {model_type}/{model_name}") from exc

    def _best_weights(self, model_name: str, use_pretrained: bool) -> str:
        if use_pretrained:
            candidate = self.cache.entry_dir("weights", model_name) / "best.pt"
            if candidate.is_file():
                return str(candidate)
        return "yolov8n.pt"

    def _download_roboflow(self, model_name: str, output_dir: Path) -> Path:
        meta = self._resolve_source("roboflow", model_name)
        return mlc.download_roboflow_dataset(
            meta["workspace"],
            meta["project"],
            output_dir,
            version=meta.get("version"),
            model_format="yolov8",
        )

    def _build_axles_dataset(self, model_name: str, output_dir: Path) -> Path:
        meta = self._resolve_source("composite", model_name)
        staging = output_dir / "staging"
        merged = output_dir / "merged"
        if staging.exists():
            shutil.rmtree(staging)
        if merged.exists():
            shutil.rmtree(merged)

        staging_images = staging / "images"
        staging_labels = staging / "labels"
        staging_images.mkdir(parents=True, exist_ok=True)
        staging_labels.mkdir(parents=True, exist_ok=True)

        zenodo_root = output_dir / "zenodo"
        roboflow_root = output_dir / "roboflow"
        dax.fetch_zenodo(zenodo_root, force=False)
        dax.convert_zenodo_labelme(zenodo_root, staging_images, staging_labels, prefix="zenodo")

        roboflow_dir = self._download_roboflow("eixosdecaminhao", roboflow_root)
        dax.convert_roboflow_yolo(roboflow_dir, staging_images, staging_labels, prefix="rf_eixos")

        return dax.merge_and_split(
            staging_images,
            staging_labels,
            merged,
            class_name="axle",
            val_ratio=0.15,
        )

    def validate_onnx(self, onnx_path: str) -> dict[str, Any]:
        path = Path(onnx_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX não encontrado: {path}")

        info: dict[str, Any] = {
            "path": path.resolve().as_posix(),
            "size_bytes": path.stat().st_size,
        }

        try:
            import onnx  # type: ignore

            model = onnx.load(str(path))
            info["opset"] = [op.version for op in model.opset_import]
            info["inputs"] = [inp.name for inp in model.graph.input]
        except Exception as exc:  # noqa: BLE001 — validação defensiva
            info["onnx_parse_error"] = str(exc)

        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            info["session_inputs"] = [item.name for item in session.get_inputs()]
            info["session_outputs"] = [item.name for item in session.get_outputs()]
        except Exception as exc:  # noqa: BLE001 — validação defensiva
            raise RuntimeError(f"Falha ao validar ONNX {path}: {exc}") from exc

        return info

    def convert_pt_to_onnx(self, pt_path: str, imgsz: int, output_path: str) -> Path:
        return mlc.export_onnx(Path(pt_path), imgsz, Path(output_path))

    def download(
        self,
        model_type: str,
        model_name: str,
        output_path: str,
        *,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        workers: int = 4,
        device: str = "",
        use_pretrained: bool = False,
    ) -> Path:
        """Baixa a fonte pública e, se necessário, exporta um ONNX pronto."""
        output = Path(output_path)
        source_root = self.cache.entry_dir(model_type, model_name) / "source"

        if model_type == "roboflow":
            dataset = self._download_roboflow(model_name, source_root)
            if output.suffix.lower() != ".onnx":
                return self.cache.store(dataset, model_type, model_name)

            best_pt = mlc.train_yolo(
                Path(dataset) / "data.yaml",
                base_weights=self._best_weights(model_name, use_pretrained),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                device=device,
                project=self.cache.entry_dir(model_type, model_name) / "runs",
                name="train",
            )
            pt_cache = self.cache.store(best_pt, "weights", model_name)
            onnx_path = mlc.export_onnx(best_pt, imgsz, output)
            self.validate_onnx(str(onnx_path))
            return onnx_path

        if model_type == "zenodo":
            dataset_root = dax.fetch_zenodo(source_root, force=False)
            staging_images = source_root / "staging" / "images"
            staging_labels = source_root / "staging" / "labels"
            if staging_images.exists():
                shutil.rmtree(staging_images.parent)
            staging_images.mkdir(parents=True, exist_ok=True)
            staging_labels.mkdir(parents=True, exist_ok=True)
            dax.convert_zenodo_labelme(dataset_root, staging_images, staging_labels)
            merged = dax.merge_and_split(
                staging_images,
                staging_labels,
                source_root / "merged",
                class_name="axle",
                val_ratio=0.15,
            )
            if output.suffix.lower() != ".onnx":
                return self.cache.store(merged, model_type, model_name)

            best_pt = mlc.train_yolo(
                merged,
                base_weights=self._best_weights(model_name, use_pretrained),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                device=device,
                project=self.cache.entry_dir(model_type, model_name) / "runs",
                name="train",
            )
            self.cache.store(best_pt, "weights", model_name)
            onnx_path = mlc.export_onnx(best_pt, imgsz, output)
            self.validate_onnx(str(onnx_path))
            return onnx_path

        if model_type == "composite":
            merged = self._build_axles_dataset(model_name, source_root)
            if output.suffix.lower() != ".onnx":
                return self.cache.store(merged, model_type, model_name)

            best_pt = mlc.train_yolo(
                merged,
                base_weights=self._best_weights(model_name, use_pretrained),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                device=device,
                project=self.cache.entry_dir(model_type, model_name) / "runs",
                name="train",
            )
            self.cache.store(best_pt, "weights", model_name)
            onnx_path = mlc.export_onnx(best_pt, imgsz, output)
            self.validate_onnx(str(onnx_path))
            return onnx_path

        raise ValueError(f"Tipo de modelo não suportado: {model_type}")


def _format_catalog_row(item: dict[str, Any]) -> str:
    return (
        f"- {item['type']}/{item['name']}: {item.get('description', '')} "
        f"[{item.get('source_url', 'sem URL')}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baixa/valida/exporta modelos pré-treinados")
    parser.add_argument("--list", action="store_true", help="Lista fontes disponíveis")
    parser.add_argument("--download-vehicle-pretrained", action="store_true")
    parser.add_argument("--download-axle-pretrained", action="store_true")
    parser.add_argument("--model-type", choices=("roboflow", "zenodo", "composite"))
    parser.add_argument("--model-name")
    parser.add_argument("--output", type=Path, default=Path("models/pretrained_cache/output.onnx"))
    parser.add_argument("--cache-root", type=Path, default=Path("models/pretrained_cache"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--use-pretrained", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manager = PreTrainedModelManager(args.cache_root)

    if args.list:
        for item in manager.list_available():
            print(_format_catalog_row({"type": item["model_type"], **item}))
        return 0

    model_type = args.model_type
    model_name = args.model_name

    if args.download_vehicle_pretrained:
        model_type = "roboflow"
        model_name = "vehicles-k83q3"
    elif args.download_axle_pretrained:
        model_type = "composite"
        model_name = "axles-composite"

    if not model_type or not model_name:
        print("Fontes disponíveis:")
        for index, item in enumerate(manager.list_available(), start=1):
            print(f"  {index}. {item['model_type']}/{item['name']} — {item.get('description', '')}")
        choice = input("Escolha um número: ").strip()
        try:
            selected = manager.list_available()[int(choice) - 1]
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Escolha inválida: {choice}") from exc
        model_type = selected["model_type"]
        model_name = selected["name"]

    result = manager.download(
        model_type,
        model_name,
        str(args.output),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        use_pretrained=args.use_pretrained,
    )
    print(f"[OK] Artefato pronto: {result.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())