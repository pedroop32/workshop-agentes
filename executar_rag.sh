#!/usr/bin/env bash
set -euo pipefail
RAG_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$RAG_PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
    echo "Crie o ambiente com: python3 -m venv .venv"
    exit 1
fi
if ! .venv/bin/python -c 'import groq, pypdf, fastembed, rank_bm25, numpy' >/dev/null 2>&1; then
    echo "Instale as dependências: .venv/bin/python -m pip install -r requirements-rag.txt"
    exit 1
fi
exec .venv/bin/python -m workshop_agentes.rag "$@"
