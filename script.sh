#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
RUN_SERVER=false
SKIP_INSTALL=false

usage() {
  cat <<'EOF'
Usage: ./script.sh [options]

Setup script to run after cloning this repository.

Options:
  --run           Start the FastAPI server after setup.
  --skip-install  Skip pip dependency installation.
  -h, --help      Show this help.

Environment overrides:
  PYTHON_BIN=/path/to/python3  Choose Python executable.
  VENV_DIR=/path/to/.venv      Choose virtualenv directory.

Examples:
  ./script.sh
  ./script.sh --run
  PYTHON_BIN=python3.11 ./script.sh
EOF
}

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

warn() {
  printf '\n[WARN] %s\n' "$*" >&2
}

die() {
  printf '\n[ERROR] %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_SERVER=true
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

cd "$ROOT_DIR"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    die "Python is not installed or not in PATH."
  fi
fi

log "Using Python: $("$PYTHON_BIN" --version)"

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  log "Virtual environment already exists: $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

log "Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

if [[ "$SKIP_INSTALL" == false ]]; then
  [[ -f requirements.txt ]] || die "requirements.txt not found."
  log "Installing Python dependencies from requirements.txt"
  pip install -r requirements.txt
else
  warn "Skipping dependency installation."
fi

if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    log "Creating .env from .env.example"
    cp .env.example .env
    warn "Review .env before running production models, especially API keys, Neo4j, GPU, LLM, STT, and TTS settings."
  else
    warn ".env.example not found; create .env manually before running the server."
  fi
else
  log ".env already exists; leaving it unchanged."
fi

log "Creating runtime directories"
mkdir -p artifacts/audio artifacts/vectorstore/medical_faiss_v4

log "Checking application imports"
python - <<'PY'
from app.core.config import settings
from app.ai_core.core.config import get_settings

ai_settings = get_settings()
print(f"App: {settings.APP_NAME}")
print(f"Database: {settings.DATABASE_URL}")
print(f"AI model: {'mock' if ai_settings.use_mock_llm else ai_settings.model_name}")
print(f"Retriever: {'mock' if ai_settings.use_mock_retriever else 'hybrid'}")
PY

cat <<EOF

Setup complete.

Activate the environment:
  source "$VENV_DIR/bin/activate"

Run the server:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Open API docs:
  http://localhost:8000/docs
EOF

if [[ "$RUN_SERVER" == true ]]; then
  log "Starting FastAPI server"
  exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
fi
