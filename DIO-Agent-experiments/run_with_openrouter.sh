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
  ./run_with_openrouter.sh [--provider openrouter|dashscope|anthropic] benchmark [benchmark_simple.py args...]
  ./run_with_openrouter.sh multimodal-base [multimodal/run_multimodal_base.py args...]
  ./run_with_openrouter.sh multimodal-base-all [multimodal/run_multimodal_base.py args...]

Examples:
  ./run_with_openrouter.sh --provider openrouter benchmark --start 1 --end 115 --iterations 20 --parallel 4 --primary-model deepseek/deepseek-v3.2 --output-subdir dio_agent_base_openrouter_deepseek_v32
  ./run_with_openrouter.sh --provider anthropic benchmark --start 1 --end 62 --iterations 20 --parallel 4 --primary-model claude-sonnet-4-6 --output-subdir dio_agent_base_anthropic_sonnet46_0502


  ./run_with_openrouter.sh benchmark --start 1 --end 62 --iterations 20 --parallel 3 --verbose-live --output-subdir dio_agent_base_final_0424
  ./run_with_openrouter.sh benchmark --start 1 --end 102 --include-error-feedback
  ./run_with_openrouter.sh --provider openrouter benchmark --start 1 --end 62 --iterations 20 --parallel 4 --include-error-feedback --with_dio_agent --primary-model deepseek/deepseek-v3.2 --output-subdir dio_agent_base_ds0503_with_dio_agent
./run_with_openrouter.sh multimodal-base-all \
  --iterations 10 \
  --parse-model qwen/qwen3.5-flash-02-23 \
  --llm-timeout-sec 180 \
  --solve-timeout-sec 240 \
  --pass-threshold 0.9 \
  --evaluator-timeout-sec 1800 \
  --output-subdir dio_agent_multimodal_base_10task_batch


Required:
  - `--provider openrouter`: `OPENROUTER_API_KEY` must be set.
  - `--provider dashscope`: `DASHSCOPE_API_KEY` must be set.
  - `--provider anthropic`: `ANTHROPIC_AUTH_TOKEN` must be set.
    `ANTHROPIC_BASE_URL` is used when present; otherwise fallback is `https://api.anthropic.com/v1`.
Notes:
  - base mode is plain by default; `--include-error-feedback` and `--with_dio_agent` are optional ablation toggles.
  - Active multimodal benchmark now contains 10 tasks (`multimodal_task1` ~ `multimodal_task10`).
  - multimodal-base-all runs multiple multimodal tasks sequentially with one command.
  - If --task-list is omitted, multimodal-base-all auto-discovers multimodal_task*/dataset_index.json under multimodal/data.
  - benchmark mode supports --output-subdir; default is BASE_OUTPUT_SUBDIR.
  - benchmark mode supports --primary-model/--secondary-model; defaults from BASE_PRIMARY_MODEL/BASE_SECONDARY_MODEL.
  - benchmark mode also receives provider-specific `--api-base` and `--api-key-env` automatically.
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
      LLM_API_KEY_ENV="ANTHROPIC_API_KEY"
    fi
    DEFAULT_PRIMARY_MODEL="claude-sonnet-4-6"
    ;;
  *)
    echo "Error: unsupported provider '${PROVIDER}'. Use 'openrouter', 'dashscope', or 'anthropic'."
    exit 1
    ;;
esac

if [[ -z "${!LLM_API_KEY_ENV:-}" ]]; then
  echo "Error: ${LLM_API_KEY_ENV} is not set for provider '${PROVIDER}'."
  exit 1
fi

# Activate your preferred venv and run from experiments directory.
source "${VENV_PATH}/bin/activate"
cd "${PROJECT_DIR}"

has_flag() {
  local flag="$1"
  shift || true
  for arg in "$@"; do
    if [[ "${arg}" == "${flag}" ]]; then
      return 0
    fi
  done
  return 1
}

# Check whether current args contain a specific flag.
has_any_flag() {
  local needle="$1"
  shift || true
  for arg in "$@"; do
    if [[ "${arg}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

discover_multimodal_task_list() {
  local data_root="${1:-multimodal/data}"
  local fallback="${2:-multimodal_task1,multimodal_task2,multimodal_task3,multimodal_task4,multimodal_task5,multimodal_task6,multimodal_task7,multimodal_task8,multimodal_task9,multimodal_task10}"
  local discovered_lines=()
  local dataset_index=""
  local task_name=""
  local task_num=""
  local joined=""
  local line=""

  if [[ -d "${data_root}" ]]; then
    while IFS= read -r dataset_index; do
      task_name="$(basename "$(dirname "${dataset_index}")")"
      if [[ "${task_name}" =~ ^multimodal_task([0-9]+)$ ]]; then
        task_num="${BASH_REMATCH[1]}"
        discovered_lines+=("${task_num}:${task_name}")
      fi
    done < <(find "${data_root}" -mindepth 2 -maxdepth 2 -type f -name "dataset_index.json")
  fi

  if [[ ${#discovered_lines[@]} -eq 0 ]]; then
    echo "${fallback}"
    return 0
  fi

  while IFS= read -r line; do
    task_name="${line#*:}"
    if [[ -n "${joined}" ]]; then
      joined+=","
    fi
    joined+="${task_name}"
  done < <(printf '%s\n' "${discovered_lines[@]}" | sort -t ':' -k1,1n)

  echo "${joined}"
}

# Wrapper-level defaults for multimodal-base mode.
MM_BASE_PARSE_MODEL="${MM_BASE_PARSE_MODEL:-qwen/qwen3.5-flash-02-23}"
MM_BASE_LLM_TIMEOUT_SEC="${MM_BASE_LLM_TIMEOUT_SEC:-180}"
MM_BASE_SOLVE_TIMEOUT_SEC="${MM_BASE_SOLVE_TIMEOUT_SEC:-240}"
MM_BASE_PASS_THRESHOLD="${MM_BASE_PASS_THRESHOLD:-0.9}"
MM_BASE_TASK_LIST="${MM_BASE_TASK_LIST:-}"
BASE_OUTPUT_SUBDIR="${BASE_OUTPUT_SUBDIR:-dio_agent_base_default}"
BASE_PRIMARY_MODEL="${BASE_PRIMARY_MODEL:-${DEFAULT_PRIMARY_MODEL}}"
BASE_SECONDARY_MODEL="${BASE_SECONDARY_MODEL:-}"

if [[ "${MODE}" == "benchmark" || "${MODE}" == "verbose" ]]; then
  if ! has_flag "--api-base" "$@"; then
    set -- "$@" --api-base "${LLM_API_BASE}"
  fi
  if ! has_flag "--api-key-env" "$@"; then
    set -- "$@" --api-key-env "${LLM_API_KEY_ENV}"
  fi
  if ! has_flag "--output-subdir" "$@"; then
    set -- "$@" --output-subdir "${BASE_OUTPUT_SUBDIR}"
  fi
  if ! has_flag "--primary-model" "$@"; then
    set -- "$@" --primary-model "${BASE_PRIMARY_MODEL}"
  fi
  if [[ -n "${BASE_SECONDARY_MODEL}" ]] && ! has_flag "--secondary-model" "$@"; then
    set -- "$@" --secondary-model "${BASE_SECONDARY_MODEL}"
  fi
fi

if [[ "${MODE}" == "multimodal-base" || "${MODE}" == "multimodal-cubes-base" || "${MODE}" == "multimodal-base-all" ]]; then
  if [[ "${PROVIDER}" != "openrouter" ]]; then
    echo "Error: multimodal base modes currently only support provider=openrouter."
    exit 1
  fi
  # Keep base mode aligned with "no feedback / no penalty" requirement.
  # if has_any_flag "--include-error-feedback" "$@"; then
  #   echo "Error: multimodal-base does not support --include-error-feedback (DIO-Agent only)."
  #   exit 1
  # fi
  if has_any_flag "--score-with-penalties" "$@"; then
    echo "Error: multimodal-base does not support --score-with-penalties (DIO-Agent-only behavior)."
    exit 1
  fi

  if ! has_flag "--parse-model" "$@"; then
    set -- "$@" --parse-model "${MM_BASE_PARSE_MODEL}"
  fi
  if ! has_flag "--llm-timeout-sec" "$@"; then
    set -- "$@" --llm-timeout-sec "${MM_BASE_LLM_TIMEOUT_SEC}"
  fi
  if ! has_flag "--solve-timeout-sec" "$@"; then
    set -- "$@" --solve-timeout-sec "${MM_BASE_SOLVE_TIMEOUT_SEC}"
  fi
  if ! has_flag "--pass-threshold" "$@"; then
    set -- "$@" --pass-threshold "${MM_BASE_PASS_THRESHOLD}"
  fi

fi

run_multimodal_base_for_task() {
  local task_name="$1"
  shift || true
  local cmd_args=("$@")

  if ! has_flag "--task-name" "${cmd_args[@]}"; then
    cmd_args+=(--task-name "${task_name}")
  fi
  if ! has_flag "--dataset-index" "${cmd_args[@]}"; then
    cmd_args+=(--dataset-index "multimodal/data/${task_name}/dataset_index.json")
  fi

  echo "==> Running multimodal base task: ${task_name}"
  python multimodal/run_multimodal_base.py "${cmd_args[@]}"
}

run_multimodal_base_all() {
  local default_task_list="${MM_BASE_TASK_LIST}"
  if [[ -z "${default_task_list}" ]]; then
    default_task_list="$(discover_multimodal_task_list "multimodal/data")"
  fi
  local task_list_csv="${default_task_list}"
  local passthrough=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --task-list)
        if [[ $# -lt 2 ]]; then
          echo "Error: --task-list requires a value."
          exit 1
        fi
        task_list_csv="$2"
        shift 2
        ;;
      --task-name|--dataset-index)
        echo "Error: multimodal-base-all does not accept $1. Use --task-list instead."
        exit 1
        ;;
      *)
        passthrough+=("$1")
        shift
        ;;
    esac
  done

  local raw_tasks=()
  IFS=',' read -r -a raw_tasks <<< "${task_list_csv}"
  if [[ ${#raw_tasks[@]} -eq 0 ]]; then
    echo "Error: empty task list for multimodal-base-all."
    exit 1
  fi

  local launched=0
  for raw_task in "${raw_tasks[@]}"; do
    local task_name="${raw_task//[[:space:]]/}"
    if [[ -z "${task_name}" ]]; then
      continue
    fi
    launched=$((launched + 1))
    run_multimodal_base_for_task "${task_name}" "${passthrough[@]}"
  done

  if [[ ${launched} -eq 0 ]]; then
    echo "Error: no valid task names resolved from --task-list."
    exit 1
  fi
}

case "${MODE}" in
  benchmark)
    python benchmark_simple.py "$@"
    ;;
  multimodal-base)
    run_multimodal_base_for_task "multimodal_task1" "$@"
    ;;
  multimodal-cubes-base)
    echo "Error: multimodal-cubes-base was retired when the multimodal benchmark was reindexed from 15 tasks to 10."
    echo "Use a named task from multimodal_task1..multimodal_task10 instead."
    exit 1
    ;;
  multimodal-base-all)
    run_multimodal_base_all "$@"
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo
    usage
    exit 1
    ;;
esac
