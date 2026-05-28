#!/usr/bin/env bash
# build_jobs.sh — build concrete eval configs and print runnable eval commands.
#
# Usage:
#   bash build_jobs.sh
#   JOBS_CONFIG=configs/eval/submit/jobs.yaml bash build_jobs.sh
#
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

JOBS_CONFIG="${JOBS_CONFIG:-configs/eval/submit/jobs.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/submit}"
COMMANDS_FILE="${OUTPUT_DIR}/eval_commands.sh"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

python scripts/eval/build_eval_jobs.py \
  --jobs-config "${JOBS_CONFIG}" \
  --output-dir "${OUTPUT_DIR}"

echo
echo "Resolved job configs were written to ${OUTPUT_DIR}/generated_configs/."
echo "Runnable eval commands were written to ${COMMANDS_FILE}."
echo
echo "These commands are not executed automatically."
echo "TODO: wire them into your own local or cluster execution pipeline to produce .eval files."
echo "The notebooks and judge utilities should work once those eval outputs exist."
echo
echo "Commands:"
cat "${COMMANDS_FILE}"
