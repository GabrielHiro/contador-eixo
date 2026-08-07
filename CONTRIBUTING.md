# Contribuindo

Obrigado por contribuir com o **contador-eixo**. Este guia resume o fluxo
esperado para mudanças no código C++ (borda) e no pipeline Python (MLOps).

## Setup

```bash
# Cliente Git LFS (vídeos e modelos .onnx)
git lfs install
git lfs pull

# Runtime C++ (Linux)
make ort && make build

# Ambiente Python (validação / treino)
make venv-world
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
```

Requisitos detalhados estão no [README.md](README.md).

## Estilo de código

### C++

- C++17, namespace `contador`.
- Preferir RAII; o `Detector` já usa PImpl — mantenha esse padrão quando o
  tipo carregar dependências pesadas (ONNX Runtime, sockets etc.).
- Comentários, logs e mensagens de erro em **português**, alinhados ao resto
  do repositório.
- Não reescreva módulos estáveis sem necessidade; evolua incrementalmente e
  preserve a compatibilidade das flags de CLI existentes.

### Python

- Scripts de MLOps na raiz (`mlops_common.py`, `treinar_*.py`,
  `validador_yoloworld.py`).
- Funções compartilhadas de treino/export ficam em `mlops_common.py`.

## Commits

- Commits **atômicos** e bem descritos, em **português**.
- Uma mudança lógica por commit (ex.: LFS, LICENSE, um módulo de teste).
- Não faça squash de fases inteiras num único commit sem necessidade.

## Testes

Quando o alvo existir (Fase 1+):

```bash
make test          # C++ (ctest) + Python (pytest)
```

Antes de abrir um PR, rode a suíte relevante ao que você alterou. Em
Windows sem WSL/Docker, testes C++ dependem do CI (GitHub Actions); testes
Python (`pytest`) podem rodar localmente no venv.

## Checklist de PR

- [ ] Escopo claro; mudanças fora do escopo ficam para outro PR
- [ ] `make build` (ou CI) passa para mudanças C++
- [ ] Testes novos/atualizados quando a lógica for testável
- [ ] README / CHANGELOG atualizados se houver flag, rota ou comportamento novo
- [ ] Sem credenciais (RTSP, API keys) commitadas em texto puro
- [ ] Binários grandes (vídeos, `.onnx`) via Git LFS, não como blob normal

## Licença

Ao contribuir, você concorda que o código entra sob a licença [MIT](LICENSE)
do projeto.
