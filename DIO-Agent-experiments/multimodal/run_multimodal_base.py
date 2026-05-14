from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from training_evaluator_utils import evaluate_program_with_evaluator
from multimodal.create_multimodal_task import (
    DEFAULT_DATASET_INDEX,
    DEFAULT_PRIMARY_MODEL,
    build_system_message_for_train_items,
    prepare_items_with_targets,
    render_evaluator_for_cases,
    setup_multimodal_task,
)


LOCAL_DIO_AGENT_ROOT = (EXPERIMENT_ROOT.parent / "DIO-Agent").resolve()
REQUIRED_PARSE_MODEL = "qwen/qwen3.5-flash-02-23"
REQUIRED_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _build_env(
    parse_model: str,
    llm_timeout_sec: int,
    solve_timeout_sec: int,
    code_model: str,
) -> dict:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    mm_path = str(EXPERIMENT_ROOT)
    oe_path = str(LOCAL_DIO_AGENT_ROOT)
    merged = [oe_path, mm_path]
    if existing_pythonpath:
        merged.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(merged)
    env["MM_PARSE_MODEL"] = parse_model
    env["CODE_MODEL"] = code_model
    env["MM_LLM_TIMEOUT_SEC"] = str(llm_timeout_sec)
    env["MM_SOLVE_TIMEOUT_SEC"] = str(solve_timeout_sec)
    env["OPENROUTER_BASE_URL"] = REQUIRED_OPENROUTER_BASE
    return env


def _evaluate(task_dir: Path, best_program: Path, env_for_mm: dict) -> dict:
    backup = {}
    mm_keys = [
        "MM_PARSE_MODEL",
        "MM_LLM_TIMEOUT_SEC",
        "MM_SOLVE_TIMEOUT_SEC",
        "MM_PASS_THRESHOLD",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "CODE_MODEL",
    ]
    try:
        for key in mm_keys:
            backup[key] = os.environ.get(key)
            if key in env_for_mm:
                os.environ[key] = env_for_mm[key]
        return evaluate_program_with_evaluator(task_dir / "evaluator.py", best_program)
    finally:
        for key, val in backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal task in DIO-Agent base mode")
    parser.add_argument("--task-name", default="multimodal_task1")
    parser.add_argument("--dataset-index", default=str(DEFAULT_DATASET_INDEX))
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--evaluator-timeout-sec", type=int, default=1800)
    parser.add_argument("--pass-threshold", type=float, default=0.9)
    parser.add_argument("--parse-model", default="qwen/qwen3.5-flash-02-23")
    parser.add_argument("--llm-timeout-sec", type=int, default=180)
    parser.add_argument("--solve-timeout-sec", type=int, default=240)
    parser.add_argument(
        "--include-error-feedback",
        action="store_true",
        help="Include training error artifacts for prompt feedback in base evolution.",
    )
    parser.add_argument(
        "--with_dio_agent",
        action="store_true",
        help="Append the multimodal DIO-Agent guidance to the base prompt.",
    )
    parser.add_argument("--num-islands", type=int, default=3)
    parser.add_argument("--primary-model", default=DEFAULT_PRIMARY_MODEL)
    parser.add_argument("--secondary-model", default=None)
    parser.add_argument(
        "--code-model",
        default=None,
        help="Runtime CODE_MODEL env passed to evolved multimodal programs. Defaults to --primary-model.",
    )
    parser.add_argument("--output-subdir", default="")
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args()

    if args.parse_model != REQUIRED_PARSE_MODEL:
        raise ValueError(
            f"Invalid parse model: {args.parse_model}. "
            f"Required model is {REQUIRED_PARSE_MODEL}."
        )
    if args.num_islands < 1:
        raise ValueError("--num-islands must be >= 1")

    dataset_index_path = Path(args.dataset_index).resolve()
    dataset = json.loads(dataset_index_path.read_text(encoding="utf-8"))
    train_items = prepare_items_with_targets(
        dataset.get("train", []),
        dataset_index_path=dataset_index_path,
        split_name="train",
    )
    if not train_items:
        raise ValueError("Dataset index contains empty train split.")

    task_dir = setup_multimodal_task(
        task_name=args.task_name,
        dataset_index_path=dataset_index_path,
        evaluator_timeout_sec=args.evaluator_timeout_sec,
    )
    if args.output_subdir.strip():
        output_subdir = args.output_subdir.strip()
    else:
        output_subdir = f"dio_agent_multimodal_base-{datetime.now().strftime('%m%d-%H%M%S')}"

    output_dir = task_dir / output_subdir
    if output_dir.exists():
        if not args.overwrite_output:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --output-subdir with a new name or pass --overwrite-output."
            )
        import shutil
        shutil.rmtree(output_dir)

    env = _build_env(
        parse_model=args.parse_model,
        llm_timeout_sec=args.llm_timeout_sec,
        solve_timeout_sec=args.solve_timeout_sec,
        code_model=args.code_model or args.primary_model,
    )
    env["MM_PASS_THRESHOLD"] = str(max(0.0, min(1.0, args.pass_threshold)))
    runtime_config_path = task_dir / f".config_multimodal_base_runtime_{os.getpid()}.yaml"
    runtime_evaluator_path = task_dir / f".evaluator_train_runtime_{os.getpid()}.py"

    base_config = yaml.safe_load((task_dir / "config.yaml").read_text(encoding="utf-8"))
    database_cfg = base_config.setdefault("database", {})
    database_cfg["num_islands"] = int(args.num_islands)
    prompt_cfg = base_config.setdefault("prompt", {})
    if args.with_dio_agent:
        prompt_cfg["system_message"] = build_system_message_for_train_items(
            train_items,
            include_dio_agent_description=True,
        )
    runtime_config_path.write_text(
        yaml.safe_dump(base_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    runtime_evaluator_path.write_text(
        render_evaluator_for_cases(
            train_items,
            include_error_feedback=args.include_error_feedback,
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "-m",
        "dio_agent.cli",
        str(task_dir / "initial_program.py"),
        str(runtime_evaluator_path),
        "--config",
        str(runtime_config_path),
        "--iterations",
        str(args.iterations),
        "--output",
        str(output_dir),
    ]
    if args.primary_model:
        cmd.extend(["--primary-model", str(args.primary_model)])
    if args.secondary_model:
        cmd.extend(["--secondary-model", str(args.secondary_model)])
    try:
        subprocess.run(cmd, cwd=str(EXPERIMENT_ROOT), check=True, env=env)
    finally:
        for path in (runtime_config_path, runtime_evaluator_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    best_program = output_dir / "best" / "best_program.py"
    if not best_program.exists():
        raise FileNotFoundError(f"Best program not found: {best_program}")
    holdout = _evaluate(task_dir, best_program, env)

    summary = {
        "task_name": args.task_name,
        "iterations": args.iterations,
        "evaluator_timeout_sec": args.evaluator_timeout_sec,
        "pass_threshold": float(max(0.0, min(1.0, args.pass_threshold))),
        "parse_model": args.parse_model,
        "primary_model": args.primary_model,
        "secondary_model": args.secondary_model,
        "code_model": args.code_model or args.primary_model,
        "llm_timeout_sec": args.llm_timeout_sec,
        "solve_timeout_sec": args.solve_timeout_sec,
        "include_error_feedback": bool(args.include_error_feedback),
        "with_dio_agent": bool(args.with_dio_agent),
        "num_islands": int(args.num_islands),
        "output_dir": str(output_dir),
        "best_program": str(best_program),
        "holdout_evaluation": holdout,
    }
    summary_path = output_dir / "multimodal_base_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Base multimodal summary saved to: {summary_path}")
    print(f"Holdout metrics: {holdout.get('metrics', {})}")


if __name__ == "__main__":
    main()
