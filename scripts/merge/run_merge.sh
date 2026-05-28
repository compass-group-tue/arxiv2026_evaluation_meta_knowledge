#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Auto-export vars from .env when present.
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

# --- Virtual environment ---
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"
if [[ "${VENV_PATH}" != /* ]]; then
  VENV_PATH="${ROOT_DIR}/${VENV_PATH}"
fi
if [[ -d "${VENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
else
  echo "Warning: venv not found at ${VENV_PATH}; continuing without activation."
fi

# --- Hugging Face ---
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_TRUST_REMOTE_CODE=1
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "[merge] Starting merge for MODEL_KEYS=${MODEL_KEYS}"

cd "${ROOT_DIR}"
python scripts/merge/merge_lora_models.py ${MODEL_KEYS}
