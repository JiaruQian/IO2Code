"""
Active oracle-example baseline runner.

This is the non-DIO-Agent counterpart of run_active_dio_agent.py. It keeps the same
black-box oracle example-generation loop, but uses the baseline DIO-Agent
prompt/evaluator semantics:
  - no DIO-Agent curriculum context
  - no stage-wise curriculum scoring
  - default prompt context sizes from each task config
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DIO_AGENT_ROOT = (SCRIPT_DIR.parent / "DIO-Agent").resolve()
if str(LOCAL_DIO_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIO_AGENT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adapter import TaskAdapter  # noqa: E402
from run_active_dio_agent import (  # noqa: E402
    _generate_oracle_examples,
    _latest_checkpoint,
)
from run_dio_agent import (  # noqa: E402
    _build_seed_checkpoint_for_next_stage,
    _dump_yaml,
    _evaluate_cases,
    _evaluate_with_full_evaluator,
    _load_function_name,
    _load_yaml,
    _run_subprocess,
)
from training_evaluator_utils import _render_training_evaluator  # noqa: E402


OUTPUT_DIR = "active_base_oracle"


def _make_active_base_system_prompt(
    base_system_prompt: str,
    visible_cases: Sequence[tuple[Any, Any]],
    batch_size_next: int,
) -> str:
    lines = []
    for idx, (inp, out) in enumerate(visible_cases, start=1):
        lines.append(f"Example {idx}:\n  Input:  {repr(inp)}\n  Output: {repr(out)}")
    examples_text = "\n".join(lines)

    import re

    pattern = re.compile(
        r"(\*\*Training Examples \(you can see these\):\*\*\s*)(.*?)(\s*\*\*IMPORTANT NOTES:\*\*)",
        re.DOTALL,
    )
    replacement = (
        r"\1"
        f"\n{examples_text}\n\n"
        f"- Active oracle examples currently visible: {len(visible_cases)}\n"
        f"- If all current examples pass, the runner will query {batch_size_next} more hidden-oracle examples.\n"
        r"\3"
    )
    prompt = pattern.sub(replacement, base_system_prompt, count=1)
    return f"""{prompt.strip()}

## ACTIVE_ORACLE_CONTEXT
- You only see the I/O examples above; the hidden oracle implementation is not shown.
- Pass every currently visible example while preserving behavior that already works.
- Do NOT hardcode full concrete inputs or build lookup tables from examples.
- Infer a rule that should generalize to unseen final tests.
"""


def _configure_iteration(
    *,
    base_config: dict,
    base_prompt: str,
    visible_cases: Sequence[tuple[Any, Any]],
    next_batch_size: int,
    api_base: str | None,
    api_key_env: str | None,
    primary_model: str | None,
    secondary_model: str | None,
) -> dict:
    config = copy.deepcopy(base_config)
    config["max_iterations"] = 1
    config["checkpoint_interval"] = 1
    config["early_stopping_patience"] = None

    llm_cfg = config.setdefault("llm", {})
    if api_base:
        llm_cfg["api_base"] = api_base
    if api_key_env:
        llm_cfg["api_key"] = f"${{{api_key_env}}}"
    if primary_model:
        llm_cfg["primary_model"] = primary_model
        llm_cfg.pop("models", None)
    if secondary_model:
        llm_cfg["secondary_model"] = secondary_model

    prompt_cfg = config.setdefault("prompt", {})
    prompt_cfg["system_message"] = _make_active_base_system_prompt(
        base_prompt, visible_cases, next_batch_size
    )
    return config


def _evaluate_program_cases(
    program_path: Path,
    function_name: str,
    cases: Sequence[tuple[Any, Any]],
) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(f"active_base_candidate_{uuid.uuid4().hex}", str(program_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, function_name):
        return {"accuracy": 0.0, "correct": 0, "total": len(cases), "errors": [{"error": "missing function"}]}
    acc, correct, total, errors = _evaluate_cases(getattr(module, function_name), cases)
    return {"accuracy": acc, "correct": correct, "total": total, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run active oracle-example baseline")
    parser.add_argument("--task", required=True, help="Task name under DIO-Agent-experiments/tasks")
    parser.add_argument("--max-iterations", type=int, default=50, help="Total DIO-Agent iterations")
    parser.add_argument("--initial-batch-size", type=int, default=2, help="First oracle example batch size")
    parser.add_argument("--batch-size-step", type=int, default=2, help="Increase added batch size by this amount")
    parser.add_argument("--output-subdir", default=OUTPUT_DIR, help="Output folder under tasks/<TASK>/")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base")
    parser.add_argument("--api-key-env", default=None, help="Environment variable containing API key")
    parser.add_argument("--primary-model", default=None, help="Primary model for evolution and example generation")
    parser.add_argument("--secondary-model", default=None, help="Optional secondary evolution model")
    parser.add_argument("--example-model", default=None, help="Override model used to propose new inputs")
    parser.add_argument("--example-temperature", type=float, default=0.8)
    parser.add_argument("--example-timeout", type=int, default=60)
    parser.add_argument("--example-max-attempts", type=int, default=3)
    parser.add_argument("--promotion-early-stop", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Prepare first files without running DIO-Agent")
    args = parser.parse_args()

    task_dir = SCRIPT_DIR / "tasks" / args.task
    base_config_path = task_dir / "config.yaml"
    base_evaluator_path = task_dir / "evaluator.py"
    base_initial_program = task_dir / "initial_program.py"
    if not base_config_path.exists() or not base_evaluator_path.exists() or not base_initial_program.exists():
        raise FileNotFoundError("Task must include config.yaml, evaluator.py, and initial_program.py")

    adapter = TaskAdapter(args.task)
    base_config = _load_yaml(base_config_path)
    base_prompt = base_config.get("prompt", {}).get("system_message", "")
    function_name = _load_function_name(base_initial_program)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    example_model = args.example_model or args.primary_model

    active_root = task_dir / args.output_subdir
    if active_root.exists() and not args.dry_run:
        shutil.rmtree(active_root)
    active_root.mkdir(parents=True, exist_ok=True)
    query_log_dir = active_root / "example_query_logs"
    query_log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{LOCAL_DIO_AGENT_ROOT}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(LOCAL_DIO_AGENT_ROOT)
    )

    seen_inputs: set[str] = set()
    visible_cases: list[tuple[Any, Any]] = []
    examples_log: list[dict[str, Any]] = []
    stop_reason: str | None = None
    stop_details: dict[str, Any] = {}

    try:
        visible_cases, first_log = _generate_oracle_examples(
            adapter=adapter,
            visible_cases=[],
            batch_size=args.initial_batch_size,
            seen=seen_inputs,
            api_base=args.api_base,
            api_key=api_key,
            model=example_model,
            timeout=args.example_timeout,
            temperature=args.example_temperature,
            max_attempts=args.example_max_attempts,
            query_log_dir=query_log_dir,
            event_name="initial",
        )
        examples_log.append(
            {
                "event": "initial_examples",
                "batch_size": args.initial_batch_size,
                "examples": visible_cases,
                "generation": first_log,
            }
        )
    except Exception as exc:
        stop_reason = "initial_example_generation_failed"
        stop_details = {"error": f"{type(exc).__name__}: {exc}"}

    next_batch_size = args.initial_batch_size + args.batch_size_step
    current_checkpoint: Path | None = None
    current_program = base_initial_program
    best_program = base_initial_program
    iteration_summaries: list[dict[str, Any]] = []
    consecutive_promotions = 0

    for iteration in range(1, args.max_iterations + 1):
        if stop_reason is not None:
            break

        iteration_dir = active_root / f"iteration_{iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        evaluator_path = iteration_dir / "evaluator_active_base.py"
        evaluator_path.write_text(
            _render_training_evaluator(
                function_name=function_name,
                train_cases=list(visible_cases),
                include_error_feedback=True,
            ),
            encoding="utf-8",
        )
        config = _configure_iteration(
            base_config=base_config,
            base_prompt=base_prompt,
            visible_cases=visible_cases,
            next_batch_size=next_batch_size,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            primary_model=args.primary_model,
            secondary_model=args.secondary_model,
        )
        config_path = iteration_dir / "config_active_base.yaml"
        _dump_yaml(config_path, config)

        if args.dry_run:
            break

        output_dir = iteration_dir / "dio_agent_output"
        cmd = [
            sys.executable,
            "-m",
            "dio_agent.cli",
            str(current_program),
            str(evaluator_path),
            "--config",
            str(config_path),
            "--iterations",
            "1",
            "--output",
            str(output_dir),
        ]
        if args.api_base:
            cmd.extend(["--api-base", args.api_base])
        if args.api_key_env:
            cmd.extend(["--api-key-env", args.api_key_env])
        if args.primary_model:
            cmd.extend(["--primary-model", args.primary_model])
        if args.secondary_model:
            cmd.extend(["--secondary-model", args.secondary_model])
        if current_checkpoint:
            cmd.extend(["--checkpoint", str(current_checkpoint)])

        started = time.time()
        _run_subprocess(cmd, cwd=SCRIPT_DIR, env=env)
        elapsed = time.time() - started

        candidate_best = output_dir / "best" / "best_program.py"
        if candidate_best.exists():
            best_program = candidate_best
            current_program = candidate_best
        current_checkpoint = _latest_checkpoint(output_dir)

        active_eval = _evaluate_program_cases(best_program, function_name, visible_cases)
        promoted = bool(active_eval["total"] > 0 and active_eval["correct"] == active_eval["total"])
        item: dict[str, Any] = {
            "iteration": iteration,
            "visible_examples": len(visible_cases),
            "next_batch_size": next_batch_size,
            "best_program": str(best_program),
            "checkpoint": str(current_checkpoint) if current_checkpoint else None,
            "active_eval": active_eval,
            "elapsed_sec": elapsed,
            "promoted": promoted,
        }

        if promoted:
            consecutive_promotions += 1
            if consecutive_promotions >= args.promotion_early_stop:
                stop_reason = "promotion_early_stop"
                stop_details = {
                    "iteration": iteration,
                    "consecutive_promotions": consecutive_promotions,
                    "threshold": args.promotion_early_stop,
                }
                item["stop_reason"] = stop_reason
                item["stop_details"] = stop_details
                iteration_summaries.append(item)
                break
            try:
                new_examples, gen_log = _generate_oracle_examples(
                    adapter=adapter,
                    visible_cases=visible_cases,
                    batch_size=next_batch_size,
                    seen=seen_inputs,
                    api_base=args.api_base,
                    api_key=api_key,
                    model=example_model,
                    timeout=args.example_timeout,
                    temperature=args.example_temperature,
                    max_attempts=args.example_max_attempts,
                    query_log_dir=query_log_dir,
                    event_name=f"promotion_after_iteration_{iteration:03d}",
                )
            except Exception as exc:
                stop_reason = "promotion_example_generation_failed"
                stop_details = {
                    "iteration": iteration,
                    "requested_batch_size": next_batch_size,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                item["stop_reason"] = stop_reason
                item["stop_details"] = stop_details
                iteration_summaries.append(item)
                break
            visible_cases.extend(new_examples)
            examples_log.append(
                {
                    "event": "promotion_examples",
                    "after_iteration": iteration,
                    "batch_size": next_batch_size,
                    "examples": new_examples,
                    "generation": gen_log,
                }
            )
            seed_dir = iteration_dir / "seed_checkpoint_after_promotion"
            best_code = best_program.read_text(encoding="utf-8")
            num_islands = max(1, int(config.get("database", {}).get("num_islands", 3)))
            current_checkpoint = _build_seed_checkpoint_for_next_stage(
                checkpoint_dir=seed_dir,
                num_islands=num_islands,
                best_code=best_code,
                random_program_payload=None,
                random_island_id=0,
            )
            item["added_examples"] = len(new_examples)
            item["post_promotion_checkpoint"] = str(current_checkpoint)
            next_batch_size += args.batch_size_step
        else:
            consecutive_promotions = 0

        iteration_summaries.append(item)

        summary_path = active_root / "active_base_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task": args.task,
                    "max_iterations": args.max_iterations,
                    "iterations_completed": iteration,
                    "visible_examples": visible_cases,
                    "examples_log": examples_log,
                    "iterations": iteration_summaries,
                    "current_best_program": str(best_program),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    final_best_copy = active_root / "final_best_program.py"
    if best_program.exists() and not args.dry_run:
        shutil.copyfile(best_program, final_best_copy)
        final_eval = _evaluate_with_full_evaluator(base_evaluator_path, final_best_copy)
    else:
        final_eval = None

    run_summary = {
        "task": args.task,
        "method": "active_base",
        "dry_run": args.dry_run,
        "max_iterations": args.max_iterations,
        "initial_batch_size": args.initial_batch_size,
        "batch_size_step": args.batch_size_step,
        "example_max_attempts": args.example_max_attempts,
        "promotion_early_stop": args.promotion_early_stop,
        "stop_reason": stop_reason or ("dry_run" if args.dry_run else "max_iterations_reached"),
        "stop_details": stop_details,
        "final_visible_example_count": len(visible_cases),
        "visible_examples": visible_cases,
        "examples_log": examples_log,
        "iterations": iteration_summaries,
        "final_best_program": str(final_best_copy if final_best_copy.exists() else best_program),
        "final_full_evaluation": final_eval,
    }
    summary_path = active_root / "active_base_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(f"\nActive base summary saved to: {summary_path}")
    if final_eval:
        metrics = final_eval.get("metrics", {})
        print("Final held-out evaluator metrics:")
        for key, value in metrics.items():
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
