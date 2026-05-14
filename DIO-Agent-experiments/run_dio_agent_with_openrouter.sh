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
  ./run_dio_agent_with_openrouter.sh [--provider openrouter|dashscope|anthropic] benchmark [benchmark_dio_agent.py args...]
  ./run_dio_agent_with_openrouter.sh [--provider openrouter|dashscope|anthropic] single --task <TaskName> [run_dio_agent.py args...]
  ./run_dio_agent_with_openrouter.sh multimodal-dio-agent [multimodal/run_multimodal_dio_agent.py args...]
  ./run_dio_agent_with_openrouter.sh multimodal-dio-agent-all [multimodal/run_multimodal_dio_agent.py args...]

Examples:
  ./run_dio_agent_with_openrouter.sh --provider dashscope benchmark --start 1 --end 115 --parallel 4 --timeout 7200 --stage-iterations 3,3,6,8 --primary-model kimi-k2.6 --include-error-feedback --interstage-init-mode best_only --final-selection-mode stage4_best --output-subdir dio_agent_kimi0501
  ./run_dio_agent_with_openrouter.sh --provider openrouter benchmark --start 1 --end 115 --parallel 4 --timeout 7200 --stage-iterations 3,3,6,8 --primary-model anthropic/claude-sonnet-4.6 --output-subdir dio_agent_anthropic_sonnet46_0502
  
  DIO_AGENT_INCLUDE_ERROR_FEEDBACK=0 ./run_dio_agent_with_openrouter.sh --provider openrouter benchmark --start 1 --end 62 --parallel 4 --timeout 7200 --stage-iterations 3,3,6,8 --primary-model deepseek/deepseek-v3.2 --interstage-init-mode best_only --final-selection-mode stage4_best --output-subdir dio_agent_ds0501_no_error_feedback
  ./run_dio_agent_with_openrouter.sh --provider openrouter benchmark --start 1 --end 115 --parallel 4 --timeout 7200 --stage-iterations 3,3,6,8 --primary-model deepseek/deepseek-v3.2 --no_dio_agent --interstage-init-mode best_only --final-selection-mode stage4_best --output-subdir dio_agent_ds0503_no_dio_agent

  ./run_dio_agent_with_openrouter.sh --provider dashscope single --task Abs_Current --stage-iterations 1,1,1,1 --primary-model qwen3.6-plus --output-subdir dio_agent_dashscope_qwen36plus_smoke
  ./run_dio_agent_with_openrouter.sh benchmark --start 63 --end 102 --parallel 4 --timeout 7200 --stage-iterations 3,3,6,8 --include-error-feedback --interstage-init-mode one_random_island --final-selection-mode stage4_best --output-subdir dio_agent_final_0424
  ./run_dio_agent_with_openrouter.sh benchmark --start 63 --end 122 --parallel 4 --timeout 7200 --stage-iterations 3,3,6,8 --include-error-feedback --interstage-init-mode best_only --final-selection-mode all_stage_candidates_training_reselect --output-subdir dio_agent_final_0424


  ./run_dio_agent_with_openrouter.sh single --task Extra_CountChars --stage-iterations 3,3,6,8 
  ./run_dio_agent_with_openrouter.sh single --task Extra_Dec2Roman --stage-iterations 3,3,6,8
  ./run_dio_agent_with_openrouter.sh single --task Extra_IsPalindrome --stage-iterations 3,3,6,8
  ./run_dio_agent_with_openrouter.sh single --task Extra_Dec2Bin --stage-iterations 3,3,6,8
  ./run_dio_agent_with_openrouter.sh single --task Extra_Dec2Bin --interstage-init-mode one_random_island
  ./run_dio_agent_with_openrouter.sh single --task Extra_Dec2Bin --final-selection-mode all_stage_candidates_training_reselect
  ./run_dio_agent_with_openrouter.sh benchmark --start 1 --end 20 --interstage-init-mode best_only
  
    ./run_dio_agent_with_openrouter.sh multimodal-dio-agent-all \
    --stage-iterations 3,3,6,8 \
    --primary-model deepseek/deepseek-v3.2 \
    --include-error-feedback \
    --no_dio_agent \
    --num-islands 3 \
    --parse-model qwen/qwen3.5-flash-02-23 \
    --score-with-penalties \
    --output-subdir dio_agent_0507-no-dio_agent
  
./run_dio_agent_with_openrouter.sh multimodal-dio-agent-all --stage-iterations 3,3,6,8 --include-error-feedback --parse-model qwen/qwen3.5-flash-02-23 --llm-timeout-sec 180 --solve-timeout-sec 240 --evaluator-timeout-sec 1800 --pass-threshold 0.9 --score-with-penalties --output-subdir dio_agent_multimodal_0504_harness



  ./run_dio_agent_with_openrouter.sh multimodal-dio-agent \
  --task-name multimodal_task3 \
  --dataset-index multimodal/data/multimodal_task3/dataset_index.json \
  --stage-iterations 3,3,6,8 \
  --include-error-feedback \
  --primary-model deepseek/deepseek-v3.2 \
  --num-islands 5 \
  --parse-model qwen/qwen3.5-flash-02-23 \
  --llm-timeout-sec 180 \
  --solve-timeout-sec 240 \
  --evaluator-timeout-sec 1800 \
  --pass-threshold 0.9 \
  --score-with-penalties \
  --output-subdir dio_agent_multimodal_task3


Required:
  - `--provider openrouter`: `OPENROUTER_API_KEY` must be set.
  - `--provider dashscope`: `DASHSCOPE_API_KEY` must be set.
  - `--provider anthropic`: `ANTHROPIC_AUTH_TOKEN` must be set.
    `ANTHROPIC_BASE_URL` is used when present; otherwise fallback is `https://api.anthropic.com/v1`.
Notes:
  - If --interstage-init-mode is omitted, defaults to DIO_AGENT_INTERSTAGE_INIT_MODE (or best_only).
  - If --final-selection-mode is omitted, defaults to DIO_AGENT_FINAL_SELECTION_MODE (or stage4_best).
  - DIO-Agent modes default to include error feedback unless explicitly provided.
  - benchmark/single modes include the DIO-Agent guidance by default; pass `--no_dio_agent` for the w/o dio_agent description ablation.
  - multimodal-dio-agent modes also support passthrough `--primary-model`, `--secondary-model`, `--code-model`, `--num-islands`, and `--no_dio_agent`.
  - Active multimodal benchmark now contains 10 tasks (`multimodal_task1` ~ `multimodal_task10`).
  - multimodal-dio-agent-all runs multiple multimodal tasks sequentially with one command.
  - If --task-list is omitted, multimodal-dio-agent-all auto-discovers multimodal_task*/dataset_index.json under multimodal/data.
  - benchmark/single modes support --output-subdir; default is DIO_AGENT_OUTPUT_SUBDIR (or dio_agent_final_default).
  - benchmark/single modes support --primary-model/--secondary-model; defaults from DIO_AGENT_PRIMARY_MODEL/DIO_AGENT_SECONDARY_MODEL.
  - benchmark/single modes also receive provider-specific `--api-base` and `--api-key-env` automatically.
  - Multimodal DIO-Agent score penalties are optional (MM_DIO_AGENT_SCORE_WITH_PENALTIES=1 to enable).
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

# Optional wrapper-level defaults for DIO-Agent evolution.
DIO_AGENT_INTERSTAGE_INIT_MODE="${DIO_AGENT_INTERSTAGE_INIT_MODE:-best_only}"
DIO_AGENT_FINAL_SELECTION_MODE="${DIO_AGENT_FINAL_SELECTION_MODE:-stage4_best}"
DIO_AGENT_INCLUDE_ERROR_FEEDBACK="${DIO_AGENT_INCLUDE_ERROR_FEEDBACK:-1}"
DIO_AGENT_OUTPUT_SUBDIR="${DIO_AGENT_OUTPUT_SUBDIR:-dio_agent_final_default}"
DIO_AGENT_PRIMARY_MODEL="${DIO_AGENT_PRIMARY_MODEL:-${DEFAULT_PRIMARY_MODEL}}"
DIO_AGENT_SECONDARY_MODEL="${DIO_AGENT_SECONDARY_MODEL:-}"
MM_DIO_AGENT_PARSE_MODEL="${MM_DIO_AGENT_PARSE_MODEL:-qwen/qwen3.5-flash-02-23}"
MM_DIO_AGENT_LLM_TIMEOUT_SEC="${MM_DIO_AGENT_LLM_TIMEOUT_SEC:-180}"
MM_DIO_AGENT_SOLVE_TIMEOUT_SEC="${MM_DIO_AGENT_SOLVE_TIMEOUT_SEC:-240}"
MM_DIO_AGENT_PASS_THRESHOLD="${MM_DIO_AGENT_PASS_THRESHOLD:-0.9}"
MM_DIO_AGENT_INCLUDE_ERROR_FEEDBACK="${MM_DIO_AGENT_INCLUDE_ERROR_FEEDBACK:-1}"
MM_DIO_AGENT_SCORE_WITH_PENALTIES="${MM_DIO_AGENT_SCORE_WITH_PENALTIES:-0}"
MM_DIO_AGENT_TASK_LIST="${MM_DIO_AGENT_TASK_LIST:-}"

# For benchmark/single modes, if user didn't pass DIO-Agent flags explicitly,
# append wrapper defaults.
if [[ "${MODE}" == "benchmark" || "${MODE}" == "single" ]]; then
  if ! has_flag "--api-base" "$@"; then
    set -- "$@" --api-base "${LLM_API_BASE}"
  fi
  if ! has_flag "--api-key-env" "$@"; then
    set -- "$@" --api-key-env "${LLM_API_KEY_ENV}"
  fi

  PASS_INTERSTAGE_MODE=1
  for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--interstage-init-mode" ]]; then
      PASS_INTERSTAGE_MODE=0
      break
    fi
  done
  if [[ ${PASS_INTERSTAGE_MODE} -eq 1 ]]; then
    set -- "$@" --interstage-init-mode "${DIO_AGENT_INTERSTAGE_INIT_MODE}"
  fi

  PASS_FINAL_SELECTION_MODE=1
  for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--final-selection-mode" ]]; then
      PASS_FINAL_SELECTION_MODE=0
      break
    fi
  done
  if [[ ${PASS_FINAL_SELECTION_MODE} -eq 1 ]]; then
    set -- "$@" --final-selection-mode "${DIO_AGENT_FINAL_SELECTION_MODE}"
  fi

  PASS_ERROR_FEEDBACK=1
  for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--include-error-feedback" ]]; then
      PASS_ERROR_FEEDBACK=0
      break
    fi
  done
  if [[ "${DIO_AGENT_INCLUDE_ERROR_FEEDBACK}" == "1" && ${PASS_ERROR_FEEDBACK} -eq 1 ]]; then
    set -- "$@" --include-error-feedback
  fi

  PASS_OUTPUT_SUBDIR=1
  for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--output-subdir" ]]; then
      PASS_OUTPUT_SUBDIR=0
      break
    fi
  done
  if [[ ${PASS_OUTPUT_SUBDIR} -eq 1 ]]; then
    set -- "$@" --output-subdir "${DIO_AGENT_OUTPUT_SUBDIR}"
  fi

  PASS_PRIMARY_MODEL=1
  for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--primary-model" ]]; then
      PASS_PRIMARY_MODEL=0
      break
    fi
  done
  if [[ ${PASS_PRIMARY_MODEL} -eq 1 ]]; then
    set -- "$@" --primary-model "${DIO_AGENT_PRIMARY_MODEL}"
  fi

  PASS_SECONDARY_MODEL=1
  for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--secondary-model" ]]; then
      PASS_SECONDARY_MODEL=0
      break
    fi
  done
  if [[ -n "${DIO_AGENT_SECONDARY_MODEL}" && ${PASS_SECONDARY_MODEL} -eq 1 ]]; then
    set -- "$@" --secondary-model "${DIO_AGENT_SECONDARY_MODEL}"
  fi
fi

if [[ "${MODE}" == "multimodal-dio-agent" || "${MODE}" == "multimodal-cubes-dio-agent" || "${MODE}" == "multimodal-dio-agent-all" ]]; then
  if [[ "${PROVIDER}" != "openrouter" ]]; then
    echo "Error: multimodal DIO-Agent modes currently only support provider=openrouter."
    exit 1
  fi
  if ! has_flag "--parse-model" "$@"; then
    set -- "$@" --parse-model "${MM_DIO_AGENT_PARSE_MODEL}"
  fi
  if ! has_flag "--llm-timeout-sec" "$@"; then
    set -- "$@" --llm-timeout-sec "${MM_DIO_AGENT_LLM_TIMEOUT_SEC}"
  fi
  if ! has_flag "--solve-timeout-sec" "$@"; then
    set -- "$@" --solve-timeout-sec "${MM_DIO_AGENT_SOLVE_TIMEOUT_SEC}"
  fi
  if ! has_flag "--pass-threshold" "$@"; then
    set -- "$@" --pass-threshold "${MM_DIO_AGENT_PASS_THRESHOLD}"
  fi
  if [[ "${MM_DIO_AGENT_INCLUDE_ERROR_FEEDBACK}" == "1" ]] && ! has_flag "--include-error-feedback" "$@"; then
    set -- "$@" --include-error-feedback
  fi
  if [[ "${MM_DIO_AGENT_SCORE_WITH_PENALTIES}" == "1" ]] && ! has_flag "--score-with-penalties" "$@"; then
    set -- "$@" --score-with-penalties
  fi

fi

run_multimodal_dio_agent_for_task() {
  local task_name="$1"
  shift || true
  local cmd_args=("$@")

  if ! has_flag "--task-name" "${cmd_args[@]}"; then
    cmd_args+=(--task-name "${task_name}")
  fi
  if ! has_flag "--dataset-index" "${cmd_args[@]}"; then
    cmd_args+=(--dataset-index "multimodal/data/${task_name}/dataset_index.json")
  fi

  echo "==> Running multimodal DIO-Agent task: ${task_name}"
  python multimodal/run_multimodal_dio_agent.py "${cmd_args[@]}"
}

run_multimodal_dio_agent_all() {
  local default_task_list="${MM_DIO_AGENT_TASK_LIST}"
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
        echo "Error: multimodal-dio-agent-all does not accept $1. Use --task-list instead."
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
    echo "Error: empty task list for multimodal-dio-agent-all."
    exit 1
  fi

  local launched=0
  for raw_task in "${raw_tasks[@]}"; do
    local task_name="${raw_task//[[:space:]]/}"
    if [[ -z "${task_name}" ]]; then
      continue
    fi
    launched=$((launched + 1))
    run_multimodal_dio_agent_for_task "${task_name}" "${passthrough[@]}"
  done

  if [[ ${launched} -eq 0 ]]; then
    echo "Error: no valid task names resolved from --task-list."
    exit 1
  fi
}

case "${MODE}" in
  benchmark)
    python benchmark_dio_agent.py "$@"
    ;;
  single)
    python run_dio_agent.py "$@"
    ;;
  multimodal-dio-agent)
    run_multimodal_dio_agent_for_task "multimodal_task1" "$@"
    ;;
  multimodal-cubes-dio-agent)
    echo "Error: multimodal-cubes-dio-agent was retired when the multimodal benchmark was reindexed from 15 tasks to 10."
    echo "Use a named task from multimodal_task1..multimodal_task10 instead."
    exit 1
    ;;
  multimodal-dio-agent-all)
    run_multimodal_dio_agent_all "$@"
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo
    usage
    exit 1
    ;;
esac
