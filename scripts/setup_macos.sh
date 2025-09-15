#!/usr/bin/env bash
set -euo pipefail

# --- config ---
PYVER=${PYVER:-"3.11"}
VENV_DIR=${VENV_DIR:-".venv"}

echo "▶ Checking Homebrew..."
if ! command -v brew >/dev/null 2>&1; then
  echo "❌ Homebrew not found. Install from https://brew.sh first."; exit 1
fi

echo "▶ System deps (for OCR PDFs & embeddings)…"
brew list --versions tesseract >/dev/null 2>&1 || brew install tesseract
brew list --versions poppler   >/dev/null 2>&1 || brew install poppler
# optional local LLM
if ! command -v ollama >/dev/null 2>&1; then
  echo "ℹ️  (optional) 'brew install ollama' for local models"; fi

echo "▶ Python ${PYVER} & venv…"
if ! command -v python${PYVER} >/dev/null 2>&1; then
  echo "❌ Python ${PYVER} not found. Install via 'brew install python@${PYVER}'"; exit 1
fi
python${PYVER} -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel

echo "▶ Installing Python packages…"
# core project deps (your pyproject already lists most of these)
pip install -e ".[dev]"
# PDF/OCR extras (kept out of pyproject to avoid forcing system deps on everyone)
pip install pdfplumber pdf2image pytesseract Pillow
# Vector DB + embeddings (already in your pyproject, but ensure installed)
pip install chromadb sentence-transformers
# Local LLM bridge (optional)
pip install langchain-ollama

echo "▶ Creating default .env (if missing)…"
if [ ! -f .env ]; then
  cat > .env <<'ENV'
# App config
SETLIST_INDEX_DIR=.chroma
CATALOG_PATH=src/setlistgraph/data/catalog.csv

# Optional: Ollama host (local default)
# OLLAMA_HOST=http://127.0.0.1:11434

# Optional: OpenAI if you use cloud models later
# OPENAI_API_KEY=
ENV
fi

echo "✅ Done. Next:
1) source ${VENV_DIR}/bin/activate
2) (optional) ollama pull llama3.2 && ollama serve
3) python -m pytest -q
4) Build Chroma: python scripts/build_chroma_from_csv.py src/setlistgraph/data/song_catalog.sample.csv
"
