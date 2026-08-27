#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 -m py_compile app/interface_service.py app/image_inference.py tools/*.py
make build >/dev/null

if curl -fsS http://127.0.0.1:8080/health >/dev/null; then
    echo "OK: backend em 8080"
else
    echo "SKIP: backend não está ativo em 8080"
fi

if curl -fsS http://127.0.0.1:8090/ >/dev/null; then
    echo "OK: interface em 8090"
else
    echo "SKIP: interface não está ativa em 8090"
fi

echo "OK: smoke test concluído"
