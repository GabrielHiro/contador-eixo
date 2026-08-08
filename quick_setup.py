#!/usr/bin/env python3
"""
quick_setup.py — setup rápido interativo para testar o contador-eixo com um vídeo local.

Objetivo:
- checar dependências essenciais;
- orientar comandos de correção quando algo faltar;
- pedir o vídeo local;
- montar e, opcionalmente, executar o teste com o binário já compilado.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BUILD_BIN = REPO_ROOT / "build" / "contador_eixo"
ORT_DIR = REPO_ROOT / "third_party" / "onnxruntime"
VENV_DIR = REPO_ROOT / ".venv"
DEFAULT_MODEL = REPO_ROOT / "models" / "vehicles.onnx"
DEFAULT_AXLE_MODEL = REPO_ROOT / "models" / "axles.onnx"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix_command: str


def _fmt_bool(ok: bool) -> str:
    return "OK" if ok else "FALTA"


def check_dependencies() -> list[CheckResult]:
    results: list[CheckResult] = []

    ort_ok = (ORT_DIR / "include" / "onnxruntime_cxx_api.h").is_file()
    results.append(
        CheckResult(
            name="ONNX Runtime",
            ok=ort_ok,
            detail=str(ORT_DIR) if ort_ok else f"não encontrado em {ORT_DIR}",
            fix_command="make ort",
        )
    )

    build_ok = BUILD_BIN.is_file()
    results.append(
        CheckResult(
            name="Binário compilado",
            ok=build_ok,
            detail=str(BUILD_BIN) if build_ok else f"não encontrado em {BUILD_BIN}",
            fix_command="make build",
        )
    )

    venv_ok = (VENV_DIR / "bin" / "python").is_file()
    results.append(
        CheckResult(
            name="Python venv",
            ok=venv_ok,
            detail=str(VENV_DIR) if venv_ok else f"não encontrado em {VENV_DIR}",
            fix_command="make venv-world",
        )
    )

    return results


def print_checks(results: list[CheckResult]) -> None:
    print("\n[1/3] Verificação do ambiente")
    for item in results:
        print(f"- {item.name}: {_fmt_bool(item.ok)} ({item.detail})")
        if not item.ok:
            print(f"  Corrija com: {item.fix_command}")


def prompt_text(message: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{message}{suffix}: ").strip()
    return value or (default or "")


def resolve_video_path(raw_value: str) -> Path:
    candidate = Path(raw_value).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    if raw_value in {"synthetic", "0", "1"}:
        return Path(raw_value)

    raise FileNotFoundError(f"Vídeo não encontrado: {raw_value}")


def build_test_command(
    video_path: Path,
    model_path: Path,
    axle_model_path: Path,
    once: bool = True,
) -> list[str]:
    command = [
        str(BUILD_BIN),
        "--source",
        str(video_path),
        "--model",
        str(model_path),
        "--axle-model",
        str(axle_model_path),
        "--port",
        "8080",
    ]
    if once:
        command.append("--once")
    return command


def maybe_run_command(command: list[str]) -> int:
    printable = " ".join(shlex.quote(part) for part in command)
    print("\n[3/3] Comando sugerido")
    print(printable)

    answer = input("Executar agora? [S/n]: ").strip().lower()
    if answer in {"n", "nao", "não"}:
        return 0

    return subprocess.call(command, cwd=REPO_ROOT)


def main() -> int:
    os.chdir(REPO_ROOT)
    print("Setup rápido do contador-eixo")

    results = check_dependencies()
    print_checks(results)

    if not all(item.ok for item in results):
        print("\nCorrija os itens pendentes e rode o script novamente.")
        return 1

    print("\n[2/3] Arquivo de vídeo")
    video_raw = prompt_text("Informe o caminho do vídeo local", default="video.mp4")
    try:
        video_path = resolve_video_path(video_raw)
    except FileNotFoundError as exc:
        print(f"[ERRO] {exc}")
        return 1

    model_raw = prompt_text("Informe o modelo ONNX de veículos", default=str(DEFAULT_MODEL))
    model_path = Path(model_raw).expanduser()
    if not model_path.is_file():
        print(f"[ERRO] Modelo não encontrado: {model_path}")
        return 1

    axle_raw = prompt_text("Informe o modelo ONNX de eixos", default=str(DEFAULT_AXLE_MODEL))
    axle_path = Path(axle_raw).expanduser()
    if not axle_path.is_file():
        print(f"[ERRO] Modelo de eixos não encontrado: {axle_path}")
        return 1

    command = build_test_command(video_path, model_path, axle_path)
    return maybe_run_command(command)


if __name__ == "__main__":
    raise SystemExit(main())