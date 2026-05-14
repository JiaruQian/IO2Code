from __future__ import annotations

import argparse
import copy
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


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _stage_boundaries(total: int, stage_fractions: list[float]) -> list[int]:
    out = []
    prev = 0
    for idx, frac in enumerate(stage_fractions):
        value = int(round(total * frac))
        value = min(total, max(prev + 1, value)) if idx < len(stage_fractions) - 1 else total
        out.append(value)
        prev = value
    return out


def _build_env(
    parse_model: str,
    llm_timeout_sec: int,
    solve_timeout_sec: int,
    code_model: str,
) -> dict:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    merged = [str(LOCAL_DIO_AGENT_ROOT), str(EXPERIMENT_ROOT)]
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
    parser = argparse.ArgumentParser(description="Run multimodal task in stage-wise DIO-Agent mode")
    parser.add_argument("--task-name", default="multimodal_task1")
    parser.add_argument("--dataset-index", default=str(DEFAULT_DATASET_INDEX))
    parser.add_argument("--stage-fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--stage-iterations", default="4,4,6,8")
    parser.add_argument("--evaluator-timeout-sec", type=int, default=1800)
    parser.add_argument("--pass-threshold", type=float, default=0.9)
    parser.add_argument("--parse-model", default="qwen/qwen3.5-flash-02-23")
    parser.add_argument("--llm-timeout-sec", type=int, default=180)
    parser.add_argument("--solve-timeout-sec", type=int, default=240)
    parser.add_argument("--num-islands", type=int, default=3)
    parser.add_argument("--primary-model", default=DEFAULT_PRIMARY_MODEL)
    parser.add_argument("--secondary-model", default=None)
    parser.add_argument(
        "--code-model",
        default=None,
        help="Runtime CODE_MODEL env passed to evolved multimodal programs. Defaults to --primary-model.",
    )
    parser.add_argument(
        "--include-error-feedback",
        action="store_true",
        help="Include curriculum error artifacts for prompt feedback in DIO-Agent evolution.",
    )
    parser.add_argument(
        "--no_dio_agent",
        action="store_true",
        help="Disable the multimodal DIO-Agent guidance while keeping stage-wise curriculum evolution.",
    )
    parser.add_argument(
        "--score-with-penalties",
        action="store_true",
        help=(
            "Use DIO-Agent stage combined_score with penalties: "
            "max(0, acc_curriculum - 0.1*complexity_penalty - 0.1*similarity_penalty). "
            "Disabled by default."
        ),
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

    fractions = _parse_csv_floats(args.stage_fractions)
    stage_iters = _parse_csv_ints(args.stage_iterations)
    if len(fractions) != len(stage_iters):
        raise ValueError("stage-fractions and stage-iterations length mismatch")

    dataset = json.loads(Path(args.dataset_index).resolve().read_text(encoding="utf-8"))
    train_items = prepare_items_with_targets(
        dataset.get("train", []),
        dataset_index_path=Path(args.dataset_index).resolve(),
        split_name="train",
    )
    if not train_items:
        raise ValueError("Dataset index contains empty train split.")

    task_dir = setup_multimodal_task(
        task_name=args.task_name,
        dataset_index_path=Path(args.dataset_index).resolve(),
        evaluator_timeout_sec=args.evaluator_timeout_sec,
    )
    if args.output_subdir.strip():
        output_subdir = args.output_subdir.strip()
    else:
        output_subdir = f"dio_agent_multimodal-{datetime.now().strftime('%m%d-%H%M%S')}"

    output_root = task_dir / output_subdir
    if output_root.exists():
        if not args.overwrite_output:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                "Use --output-subdir with a new name or pass --overwrite-output."
            )
        import shutil
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = yaml.safe_load((task_dir / "config.yaml").read_text(encoding="utf-8"))
    boundaries = _stage_boundaries(len(train_items), fractions)
    env = _build_env(
        parse_model=args.parse_model,
        llm_timeout_sec=args.llm_timeout_sec,
        solve_timeout_sec=args.solve_timeout_sec,
        code_model=args.code_model or args.primary_model,
    )
    env["MM_PASS_THRESHOLD"] = str(max(0.0, min(1.0, args.pass_threshold)))

    previous_best = task_dir / "initial_program.py"
    previous_boundary = 0
    stage_logs = []

    for stage_idx, (boundary, iterations) in enumerate(zip(boundaries, stage_iters), start=1):
        stage_dir = output_root / f"stage_{stage_idx}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        visible_cases = train_items[:boundary]
        new_cases = train_items[previous_boundary:boundary]

        evaluator_stage = stage_dir / "evaluator_stage.py"
        evaluator_stage.write_text(
            render_evaluator_for_cases(
                visible_cases,
                include_error_feedback=args.include_error_feedback,
                score_with_penalties=args.score_with_penalties,
            ),
            encoding="utf-8",
        )

        stage_config = copy.deepcopy(base_config)
        stage_config["max_iterations"] = iterations
        database_cfg = stage_config.setdefault("database", {})
        database_cfg["num_islands"] = int(args.num_islands)
        prompt_cfg = stage_config.setdefault("prompt", {})
        prompt_cfg["system_message"] = build_system_message_for_train_items(
            visible_cases,
            include_dio_agent_description=not args.no_dio_agent,
        )
        stage_config_path = stage_dir / "config_stage.yaml"
        stage_config_path.write_text(
            yaml.safe_dump(stage_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        stage_output = stage_dir / "dio_agent_output"
        cmd = [
            sys.executable,
            "-m",
            "dio_agent.cli",
            str(previous_best),
            str(evaluator_stage),
            "--config",
            str(stage_config_path),
            "--iterations",
            str(iterations),
            "--output",
            str(stage_output),
        ]
        if args.primary_model:
            cmd.extend(["--primary-model", str(args.primary_model)])
        if args.secondary_model:
            cmd.extend(["--secondary-model", str(args.secondary_model)])
        subprocess.run(cmd, cwd=str(EXPERIMENT_ROOT), check=True, env=env)

        stage_best = stage_output / "best" / "best_program.py"
        if not stage_best.exists():
            raise FileNotFoundError(f"Stage {stage_idx} best program not found: {stage_best}")
        previous_best = stage_best
        stage_logs.append(
            {
                "stage": stage_idx,
                "iterations": iterations,
                "visible_cases": len(visible_cases),
                "new_cases": len(new_cases),
                "best_program": str(stage_best),
            }
        )
        previous_boundary = boundary

    final_eval = _evaluate(task_dir, previous_best, env)
    summary = {
        "task_name": args.task_name,
        "evaluator_timeout_sec": args.evaluator_timeout_sec,
        "pass_threshold": float(max(0.0, min(1.0, args.pass_threshold))),
        "parse_model": args.parse_model,
        "primary_model": args.primary_model,
        "secondary_model": args.secondary_model,
        "code_model": args.code_model or args.primary_model,
        "llm_timeout_sec": args.llm_timeout_sec,
        "solve_timeout_sec": args.solve_timeout_sec,
        "num_islands": int(args.num_islands),
        "include_error_feedback": bool(args.include_error_feedback),
        "include_dio_agent_description": bool(not args.no_dio_agent),
        "score_with_penalties": bool(args.score_with_penalties),
        "stage_fractions": fractions,
        "stage_iterations": stage_iters,
        "stages": stage_logs,
        "final_best_program": str(previous_best),
        "final_holdout_evaluation": final_eval,
    }
    summary_path = output_root / "multimodal_dio_agent_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DIO-Agent multimodal summary saved to: {summary_path}")
    print(f"Final holdout metrics: {final_eval.get('metrics', {})}")


if __name__ == "__main__":
    main()
