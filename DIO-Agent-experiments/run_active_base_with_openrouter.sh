#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv310}"

usage() {
  cat <<'EOF'
Usage:
  ./run_active_base_with_openrouter.sh [--provider openrouter|dashscope|anthropic] single --task <TaskName> [run_active_base.py args...]
  ./run_active_base_with_openrouter.sh [--provider openrouter|dashscope|anthropic] all [--start N] [--end M] [run_active_base.py args shared by every task]

Examples:
  ./run_active_base_with_openrouter.sh --provider openrouter single \
    --task Abs_Current \
    --max-iterations 50 \
    --primary-model deepseek/deepseek-v3.2 \
    --output-subdir active_base_oracle_abs_current

  ./run_active_base_with_openrouter.sh --provider openrouter all \
    --start 113 --end 115 \
    --max-iterations 50 \
    --primary-model deepseek/deepseek-v3.2 \
    --output-subdir active_base_oracle_all115

Required:
  - --provider openrouter: OPENROUTER_API_KEY must be set.
  - --provider dashscope: DASHSCOPE_API_KEY must be set.
  - --provider anthropic: ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY must be set.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

PROVIDER="${LLM_PROVIDER:-openrouter}"
FILTERED_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider)
      if [[ $# -lt 2 ]]; then
        echo "Error: --provider requires a value."
        exit 1
      fi
      PROVIDER="$2"
      shift 2
      ;;
    --provider=*)
      PROVIDER="${1#*=}"
      shift
      ;;
    *)
      FILTERED_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${FILTERED_ARGS[@]}"

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE="$1"
shift || true

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" || "${MODE}" == "help" ]]; then
  usage
  exit 0
fi

if [[ "${MODE}" != "single" && "${MODE}" != "all" ]]; then
  echo "Error: supported modes are 'single' and 'all'."
  usage
  exit 1
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Error: venv not found at ${VENV_PATH}"
  exit 1
fi

case "${PROVIDER}" in
  openrouter)
    LLM_API_BASE="https://openrouter.ai/api/v1"
    LLM_API_KEY_ENV="OPENROUTER_API_KEY"
    DEFAULT_PRIMARY_MODEL="deepseek/deepseek-v3.2"
    ;;
  dashscope)
    LLM_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_API_KEY_ENV="DASHSCOPE_API_KEY"
    DEFAULT_PRIMARY_MODEL="qwen3.6-plus"
    ;;
  anthropic)
    if [[ -n "${ANTHROPIC_BASE_URL:-}" ]]; then
      ANTHROPIC_BASE_TRIMMED="${ANTHROPIC_BASE_URL%/}"
      if [[ "${ANTHROPIC_BASE_TRIMMED}" == */chat/completions ]]; then
        LLM_API_BASE="${ANTHROPIC_BASE_TRIMMED%/chat/completions}"
      elif [[ "${ANTHROPIC_BASE_TRIMMED}" == */v1 ]]; then
        LLM_API_BASE="${ANTHROPIC_BASE_TRIMMED}"
      else
        LLM_API_BASE="${ANTHROPIC_BASE_TRIMMED}/v1"
      fi
      LLM_API_KEY_ENV="ANTHROPIC_AUTH_TOKEN"
    else
      LLM_API_BASE="https://api.anthropic.com/v1"
      if [[ -n "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
        LLM_API_KEY_ENV="ANTHROPIC_AUTH_TOKEN"
      else
        LLM_API_KEY_ENV="ANTHROPIC_API_KEY"
      fi
    fi
    DEFAULT_PRIMARY_MODEL="claude-sonnet-4-6"
    ;;
  *)
    echo "Error: unsupported provider '${PROVIDER}'. Use openrouter, dashscope, or anthropic."
    exit 1
    ;;
esac

if [[ -z "${!LLM_API_KEY_ENV:-}" ]]; then
  echo "Error: ${LLM_API_KEY_ENV} is not set for provider '${PROVIDER}'."
  exit 1
fi

has_flag() {
  local flag="$1"
  shift || true
  for arg in "$@"; do
    if [[ "${arg}" == "${flag}" || "${arg}" == "${flag}="* ]]; then
      return 0
    fi
  done
  return 1
}

source "${VENV_PATH}/bin/activate"
cd "${PROJECT_DIR}"

EXTRA_ARGS=()
TASK_START=1
TASK_END=115
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start)
      if [[ $# -lt 2 ]]; then
        echo "Error: --start requires a value."
        exit 1
      fi
      TASK_START="$2"
      shift 2
      ;;
    --start=*)
      TASK_START="${1#*=}"
      shift
      ;;
    --end)
      if [[ $# -lt 2 ]]; then
        echo "Error: --end requires a value."
        exit 1
      fi
      TASK_END="$2"
      shift 2
      ;;
    --end=*)
      TASK_END="${1#*=}"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! has_flag "--api-base" "${EXTRA_ARGS[@]}"; then
  EXTRA_ARGS+=(--api-base "${LLM_API_BASE}")
fi
if ! has_flag "--api-key-env" "${EXTRA_ARGS[@]}"; then
  EXTRA_ARGS+=(--api-key-env "${LLM_API_KEY_ENV}")
fi
if ! has_flag "--primary-model" "${EXTRA_ARGS[@]}"; then
  EXTRA_ARGS+=(--primary-model "${ACTIVE_BASE_PRIMARY_MODEL:-${DEFAULT_PRIMARY_MODEL}}")
fi

if [[ "${MODE}" == "single" ]]; then
  python run_active_base.py "${EXTRA_ARGS[@]}"
  exit $?
fi

if has_flag "--task" "${EXTRA_ARGS[@]}"; then
  echo "Error: all mode does not accept --task; use --start/--end to select a task range."
  exit 1
fi

if ! [[ "${TASK_START}" =~ ^[0-9]+$ && "${TASK_END}" =~ ^[0-9]+$ ]]; then
  echo "Error: --start and --end must be positive integers."
  exit 1
fi
if [[ "${TASK_START}" -lt 1 || "${TASK_END}" -lt "${TASK_START}" ]]; then
  echo "Error: invalid task range --start ${TASK_START} --end ${TASK_END}."
  exit 1
fi

TASK_FILE="$(mktemp)"
TASK_START="${TASK_START}" TASK_END="${TASK_END}" python - <<'PY' > "${TASK_FILE}"
import os
from benchmark_dio_agent import _build_task_list
tasks = _build_task_list()[:115]
start = max(1, int(os.environ["TASK_START"]))
end = min(len(tasks), int(os.environ["TASK_END"]))
if start > end:
    raise SystemExit(f"Invalid range after clamping: start={start}, end={end}, total={len(tasks)}")
for task in tasks[start - 1:end]:
    print(task)
PY

TOTAL="$(wc -l < "${TASK_FILE}" | tr -d ' ')"
if [[ "${TOTAL}" -eq 0 ]]; then
  echo "Error: no tasks discovered."
  rm -f "${TASK_FILE}"
  exit 1
fi

OUTPUT_SUBDIR="active_base_oracle_0514"
arg_count="${#EXTRA_ARGS[@]}"
idx=0
while [[ "${idx}" -lt "${arg_count}" ]]; do
  arg="${EXTRA_ARGS[$idx]}"
  if [[ "${arg}" == "--output-subdir" && $((idx + 1)) -lt "${arg_count}" ]]; then
    OUTPUT_SUBDIR="${EXTRA_ARGS[$((idx + 1))]}"
  elif [[ "${arg}" == --output-subdir=* ]]; then
    OUTPUT_SUBDIR="${arg#*=}"
  fi
  idx=$((idx + 1))
done

RESULT_DIR="active_base_batch_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULT_DIR}"
RESULT_JSONL="${RESULT_DIR}/results.jsonl"
RESULT_CSV="${RESULT_DIR}/results.csv"
printf 'task,exit_code,elapsed_sec,stop_reason,final_visible_example_count,accuracy,correct,total,combined_score,summary_path\n' > "${RESULT_CSV}"

echo "Running active base on ${TOTAL} tasks (task range ${TASK_START}..${TASK_END}, clamped to first 115 benchmark tasks)."
echo "Batch result log: ${PROJECT_DIR}/${RESULT_JSONL}"
echo "Batch CSV log: ${PROJECT_DIR}/${RESULT_CSV}"

FAILURES=0
human_idx=0
while IFS= read -r task; do
  human_idx=$((human_idx + 1))
  echo
  echo "[${human_idx}/${TOTAL}] ${task}"
  start_ts="$(date +%s)"
  set +e
  python run_active_base.py --task "${task}" "${EXTRA_ARGS[@]}"
  status=$?
  set -e
  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  if [[ "${status}" -ne 0 ]]; then
    FAILURES=$((FAILURES + 1))
  fi
  TASK_NAME="${task}" STATUS_CODE="${status}" ELAPSED_SEC="${elapsed}" OUTPUT_SUBDIR="${OUTPUT_SUBDIR}" RESULT_JSONL="${RESULT_JSONL}" RESULT_CSV="${RESULT_CSV}" python - <<'PY'
import os
import json
import csv
from pathlib import Path

task = os.environ["TASK_NAME"]
status = int(os.environ["STATUS_CODE"])
elapsed = int(os.environ["ELAPSED_SEC"])
output_subdir = os.environ["OUTPUT_SUBDIR"]
summary_path = Path("tasks") / task / output_subdir / "active_base_summary.json"
payload = {"task": task, "exit_code": status, "elapsed_sec": elapsed, "summary_path": str(summary_path)}
metrics = {}
if summary_path.exists():
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["stop_reason"] = summary.get("stop_reason")
        payload["final_full_evaluation"] = summary.get("final_full_evaluation")
        payload["final_visible_example_count"] = summary.get("final_visible_example_count")
        final_eval = summary.get("final_full_evaluation")
        if isinstance(final_eval, dict) and isinstance(final_eval.get("metrics"), dict):
            metrics = final_eval["metrics"]
    except Exception as exc:
        payload["summary_read_error"] = f"{type(exc).__name__}: {exc}"
with open(os.environ["RESULT_JSONL"], "a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
with open(os.environ["RESULT_CSV"], "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        task,
        status,
        elapsed,
        payload.get("stop_reason", ""),
        payload.get("final_visible_example_count", ""),
        metrics.get("accuracy", ""),
        metrics.get("correct", ""),
        metrics.get("total", ""),
        metrics.get("combined_score", ""),
        str(summary_path),
    ])
PY
done < "${TASK_FILE}"
rm -f "${TASK_FILE}"

echo
echo "All-mode complete: ${TOTAL} tasks, ${FAILURES} nonzero exits."
echo "Batch result log: ${PROJECT_DIR}/${RESULT_JSONL}"
echo "Batch CSV log: ${PROJECT_DIR}/${RESULT_CSV}"
exit 0
