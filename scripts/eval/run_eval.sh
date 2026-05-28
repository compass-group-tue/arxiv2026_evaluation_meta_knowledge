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

has_arg() {
  local needle="$1"
  shift
  for arg in "$@"; do
    if [[ "${arg}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

extract_tensor_parallel_size() {
  local config_path="$1"
  awk '
    /^[[:space:]]*#/ { next }
    /^[^[:space:]]/ { in_inference = ($1 == "inference:") }
    in_inference && /^[[:space:]]+tensor_parallel_size:[[:space:]]*/ {
      value = $2
      sub(/#.*/, "", value)
      print value
      exit
    }
  ' "${config_path}"
}

write_config_with_tensor_parallel_size() {
  local src_config="$1"
  local dst_config="$2"
  local tp_size="$3"
  awk -v tp="${tp_size}" '
    BEGIN { in_inference = 0; replaced = 0 }
    /^[[:space:]]*#/ { print; next }
    /^[^[:space:]]/ { in_inference = ($1 == "inference:") }
    in_inference && /^[[:space:]]+tensor_parallel_size:[[:space:]]*/ && !replaced {
      indent = ""
      if (match($0, /^[[:space:]]+/)) {
        indent = substr($0, RSTART, RLENGTH)
      }
      print indent "tensor_parallel_size: " tp
      replaced = 1
      next
    }
    { print }
    END {
      if (!replaced) exit 2
    }
  ' "${src_config}" > "${dst_config}"
}

detect_assigned_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
    echo "${#visible_gpus[@]}"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_count
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')"
    if [[ "${gpu_count}" =~ ^[0-9]+$ ]] && [[ "${gpu_count}" -gt 0 ]]; then
      echo "${gpu_count}"
      return 0
    fi
  fi

  return 1
}

clean_optional_env() {
  local val="${1:-}"
  if [[ "${val}" == "UNDEFINED" || "${val}" == "undefined" ]]; then
    echo ""
  else
    echo "${val}"
  fi
}

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

# --- Run evaluation ---
CONFIG_PATH="$(clean_optional_env "${CONFIG_PATH:-configs/eval/default.yaml}")"
echo "[eval] Starting mode=eval config=${CONFIG_PATH}"

config_tp_size="$(extract_tensor_parallel_size "${CONFIG_PATH}" || true)"

if visible_gpu_count="$(detect_assigned_gpu_count)"; then
  if [[ -z "${config_tp_size}" || "${visible_gpu_count}" -ne "${config_tp_size}" ]]; then
    tmp_config="$(mktemp "/tmp/eval_config.XXXXXX.yaml")"
    write_config_with_tensor_parallel_size "${CONFIG_PATH}" "${tmp_config}" "${visible_gpu_count}"
    echo "[eval] Overriding tensor_parallel_size -> ${visible_gpu_count}"
    CONFIG_PATH="${tmp_config}"
    trap '[[ -n "${tmp_config:-}" && -f "${tmp_config}" ]] && rm -f "${tmp_config}"' EXIT
  fi
elif [[ -z "${config_tp_size}" ]]; then
  echo "Warning: could not determine tensor_parallel_size from config or visible GPUs."
fi

cli_args=("$@")
if ! has_arg "--config" "${cli_args[@]}"; then
  cli_args=(--config "${CONFIG_PATH}" "${cli_args[@]}")
fi
if ! has_arg "--mode" "${cli_args[@]}"; then
  cli_args=(--mode eval "${cli_args[@]}")
fi

python -m src.cli "${cli_args[@]}"
