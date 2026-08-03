#!/usr/bin/env bash
# Baixa ONNX Runtime C++ para third_party/onnxruntime
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VER="${ORT_VERSION:-1.28.0}"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)  PKG="onnxruntime-linux-x64-${VER}" ;;
  aarch64) PKG="onnxruntime-linux-aarch64-${VER}" ;;
  *) echo "Arch não suportada: $ARCH"; exit 1 ;;
esac
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VER}/${PKG}.tgz"
DEST="${ROOT}/third_party/onnxruntime"
mkdir -p "${ROOT}/third_party"
TMP="$(mktemp -d)"
echo "Baixando ${URL}"
curl -fsSL -o "${TMP}/ort.tgz" "$URL"
tar -xzf "${TMP}/ort.tgz" -C "$TMP"
rm -rf "$DEST"
mv "${TMP}/${PKG}" "$DEST"
rm -rf "$TMP"
echo "OK → ${DEST}"
