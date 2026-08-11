# IO2Code

IO2Code is a benchmark for synthesizing programs from input-output examples. This repository contains the IO2Code task definitions, a local DIO-Agent evolution engine, and experiment runners for baseline, curriculum, active-oracle, and multimodal settings.

DIO-Agent combines two ideas:

- Curriculum evolution: expose examples from easy to hard and evolve programs stage by stage.
- Evolutionary search: maintain DIO-Agent populations, score candidates with task evaluators, and carry the best programs across stages.

Active-oracle runners add one more loop: an LLM proposes new inputs, the hidden task oracle returns outputs, and DIO-Agent continues evolving on the expanded visible example set.

## Repository Layout

```text
IO2Code/
  io2code_tasks/                 # IO2Code benchmark task definitions and oracle functions
  DIO-Agent/                     # Local DIO-Agent engine used by the runners
  DIO-Agent-experiments/         # Experiment entry points, task configs, and generated outputs
    tasks/                       # Per-task config.yaml, evaluator.py, initial_program.py
    multimodal/                  # Multimodal task helpers and runners
```

Main entry points:

- `DIO-Agent-experiments/run_dio_agent.py`: single-task stage-wise DIO-Agent run.
- `DIO-Agent-experiments/benchmark_dio_agent.py`: batch stage-wise DIO-Agent benchmark.
- `DIO-Agent-experiments/benchmark_simple.py`: baseline DIO-Agent evolution without stage-wise curriculum.
- `DIO-Agent-experiments/run_active_dio_agent.py`: active-oracle DIO-Agent.
- `DIO-Agent-experiments/run_active_base.py`: active-oracle baseline.
- `DIO-Agent-experiments/run_dio_agent_with_openrouter.sh`: provider-aware wrapper for standard and multimodal runs.
- `DIO-Agent-experiments/run_active_dio_agent_with_openrouter.sh`: provider-aware wrapper for active DIO-Agent.
- `DIO-Agent-experiments/run_active_base_with_openrouter.sh`: provider-aware wrapper for active baseline.

## Environment Setup

Use Python 3.10+.

```bash
git clone <repo-url> IO2Code
cd IO2Code
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pyyaml requests numpy
```

The runners use the local `DIO-Agent/` source tree through `PYTHONPATH`, so no package install step is required.

Set one API key for the provider you use:

```bash
export OPENROUTER_API_KEY="..."    # OpenRouter
export DASHSCOPE_API_KEY="..."     # DashScope
export ANTHROPIC_API_KEY="..."     # Anthropic-compatible endpoint
```

## Quick Checks

Verify imports and task discovery:

```bash
cd DIO-Agent-experiments
python - <<'PY'
from adapter import get_all_task_names
print(len(get_all_task_names(include_extra=True)))
print(get_all_task_names(include_extra=True)[:5])
PY
```

Run a single DIO-Agent smoke test:

```bash
./run_dio_agent_with_openrouter.sh --provider openrouter single \
  --task Abs_Current \
  --stage-iterations 1,1,1,1 \
  --primary-model deepseek/deepseek-v3.2 \
  --output-subdir dio_agent_smoke_abs_current
```

Run a small batch:

```bash
./run_dio_agent_with_openrouter.sh --provider openrouter benchmark \
  --start 1 --end 5 \
  --parallel 2 \
  --timeout 7200 \
  --stage-iterations 3,3,6,8 \
  --include-error-feedback \
  --primary-model deepseek/deepseek-v3.2 \
  --output-subdir dio_agent_batch_smoke
```

## Active-Oracle Runs

Active DIO-Agent starts with a small LLM-generated IO batch, evolves for one iteration, and asks for a larger batch only after the current best program passes all visible examples.

```bash
./run_active_dio_agent_with_openrouter.sh --provider openrouter single \
  --task Abs_Current \
  --max-iterations 50 \
  --initial-batch-size 2 \
  --batch-size-step 2 \
  --primary-model deepseek/deepseek-v3.2 \
  --output-subdir active_dio_agent_abs_current
```

Active baseline uses the same oracle-example loop without DIO-Agent curriculum guidance:

```bash
./run_active_base_with_openrouter.sh --provider openrouter single \
  --task Abs_Current \
  --max-iterations 50 \
  --primary-model deepseek/deepseek-v3.2 \
  --output-subdir active_base_abs_current
```

For all-mode active runs, select a task range:

```bash
./run_active_dio_agent_with_openrouter.sh --provider openrouter all \
  --start 1 --end 20 \
  --max-iterations 50 \
  --primary-model deepseek/deepseek-v3.2 \
  --output-subdir active_dio_agent_1_20
```

## Multimodal Runs

Multimodal DIO-Agent modes are exposed through the standard wrapper:

```bash
./run_dio_agent_with_openrouter.sh --provider openrouter multimodal-dio-agent-all \
  --stage-iterations 3,3,6,8 \
  --include-error-feedback \
  --primary-model deepseek/deepseek-v3.2 \
  --parse-model qwen/qwen3.5-flash-02-23 \
  --output-subdir dio_agent_multimodal_batch
```

## Outputs

Per-task results are written under:

```text
DIO-Agent-experiments/tasks/<TASK>/<output-subdir>/
```

Important files include:

- `run_summary.json`: summary for stage-wise DIO-Agent.
- `active_dio_agent_summary.json`: summary for active DIO-Agent.
- `active_base_summary.json`: summary for active baseline.
- `final_best_program.py`: final selected program when available.
- `example_query_logs/`: active-oracle LLM query prompts, responses, and execution metadata.

Batch wrappers additionally write JSONL and CSV summaries under `dio_agent_batch_results/`, `active_dio_agent_batch_results/`, or `active_base_batch_results/`.

## Citation
```bibtex
@article{dong2026IO2Code,
      title={From I/O to Code with Discovery Agent}, 
      author={Yihong Dong and Jiaru Qian and Haoran Zhang and Peixu Wang and Binhua Li and Zhi Jin and Yongbin Li and Ge Li and Xiaokang Yang and Xue Jiang},
      journal={arXiv preprint arXiv:2605.15334},
      year={2026}
}
```
