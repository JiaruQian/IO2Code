from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_INDEX = SCRIPT_DIR / "data" / "multimodal_task1" / "dataset_index.json"
DEFAULT_PRIMARY_MODEL = "deepseek/deepseek-v3.2"
TOOL_LIST = [
    "env MM_PARSE_MODEL (a multimodal LLM)",
    "env OPENROUTER_API_KEY (to fetch models from OpenRouter)",
    "python",
    "numpy",
    "scipy",
    "PIL",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "scikit-image",
    "opencv-python",
]

# TOOL_LIST = [
#     "env MM_PARSE_MODEL (a multimodal LLM)",
#     "env CODE_MODEL (a code LLM)",
#     "env OPENROUTER_API_KEY (to fetch models from OpenRouter)",
# ]

# TOOL_LIST = [
#     "env MM_PARSE_MODEL (a multimodal LLM) and env OPENROUTER_API_KEY (to fetch models from OpenRouter)",
#     "python",
#     "numpy",
#     "scipy",
#     "PIL",
#     "matplotlib",
#     "seaborn",
#     "scikit-learn",
#     "scikit-image",
#     "opencv-python",
# ]

# TOOL_LIST = [
#     "env MM_PARSE_MODEL (a multimodal LLM) and env OPENROUTER_API_KEY (to fetch models from OpenRouter)",
#     "python",
# ]

# TOOL_LIST = [
#     "python",
#     "numpy",
#     "scipy",
#     "PIL",
#     "matplotlib",
#     "seaborn",
#     "scikit-learn",
#     "scikit-image",
#     "opencv-python",
# ]

# 只给model key python会如何发展
# 前两个合起来

def _render_initial_program() -> str:
    return '''def solve(input_data):
    """
    Return result for the given input_data.
    """
    pass
'''


def _count_2x2_squares(occupied_cells: list[list[int]]) -> int:
    cells = {(int(x), int(y)) for x, y in occupied_cells}
    total = 0
    for x, y in cells:
        if (x + 1, y) in cells and (x, y + 1) in cells and (x + 1, y + 1) in cells:
            total += 1
    return total


def _to_experiment_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(EXPERIMENT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_dataset_file_path(
    dataset_root: Path,
    split_name: str | None,
    kind: str,
    raw_path: str | None,
) -> str:
    if not raw_path:
        return ""

    raw_candidate = Path(raw_path)
    if raw_candidate.exists():
        return _to_experiment_relative_path(raw_candidate)

    if split_name:
        subdir = "images" if kind == "image_png" else "prompts"
        candidate = dataset_root / split_name / subdir / raw_candidate.name
        if candidate.exists():
            return _to_experiment_relative_path(candidate)

    return str(raw_candidate)


def _augment_items_with_targets(
    items: list[dict[str, Any]],
    dataset_index_path: Path | None = None,
    split_name: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    dataset_root = dataset_index_path.parent if dataset_index_path is not None else None
    for item in items:
        copied = dict(item)
        files = item.get("files")
        if dataset_root is not None and isinstance(files, dict):
            copied_files = dict(files)
            for kind in ("image_png", "prompt_txt"):
                copied_files[kind] = _resolve_dataset_file_path(
                    dataset_root=dataset_root,
                    split_name=split_name,
                    kind=kind,
                    raw_path=copied_files.get(kind),
                )
            copied["files"] = copied_files
        explicit_target = item.get("target_n")
        if explicit_target is not None:
            copied["target_n"] = int(explicit_target)
        else:
            # Legacy fallback for archived unlabelled cell datasets.
            copied["target_n"] = _count_2x2_squares(item.get("occupied_cells", []))
        out.append(copied)
    return out


def prepare_items_with_targets(
    items: list[dict[str, Any]],
    dataset_index_path: Path | None = None,
    split_name: str | None = None,
) -> list[dict[str, Any]]:
    return _augment_items_with_targets(
        items,
        dataset_index_path=dataset_index_path,
        split_name=split_name,
    )


def _format_tool_list(tool_list: list[str] | None) -> str:
    if not tool_list:
        return "[]"
    normalized = [str(item).strip() for item in tool_list if str(item).strip()]
    return json.dumps(normalized, ensure_ascii=False)


def _build_dio_agent_context() -> str:
    return """
## DIO_AGENT_CONTEXT
- Goal: pass all visible training examples while preserving already-correct behavior.
- Apply minimal transformations first (incremental style), avoid over-engineering.

### Transformation priority
1) nil -> constant
2) constant -> scalar
3) statement -> statements
4) unconditional -> if
5) scalar -> array
6) if -> while
7) expression -> function

### Anti-hardcode rules
- Do NOT hardcode by image path, filename, or example id.
- Do NOT build lookup tables keyed by seen examples.
- Infer a general visual rule that extrapolates to unseen images.
""".strip()


def _build_system_message(
    train_items: list[dict[str, Any]],
    tool_list: list[str] | None = None,
    include_dio_agent_description: bool = False,
) -> str:
    rendered_tool_list = _format_tool_list(tool_list if tool_list is not None else TOOL_LIST)
    lines: list[str] = [
        "You are solving a multimodal IO2Code task: infer the rule from training (input, output) examples and implement solve(input_data) to return the integer output for unseen images.",
        f"You can use tools in this list: {rendered_tool_list}.",
        "",
        "**Training Examples (you can see these):**",
        "",
    ]

    # lines: list[str] = [
    #     "You are solving a multimodal IO2Code task: infer the rule from training (input, output) examples and implement solve(input_data) to return the integer output for unseen images.",
    #     f"You should use all the tools in this list: {rendered_tool_list}.",
    #     "",
    #     "**Training Examples (you can see these):**",
    #     "",
    # ]
    
    
    
 
    # lines: list[str] = [
    #     "You are an expert programmer specializing in program synthesis from examples.",
    #     "Your task is to implement a function that works correctly on the given input-output examples.",
    #     "You need to infer the pattern from these examples and write code that generalizes to all test cases.",
    #     "**Training Examples (you can see these):**",
    #     "",
    # ]
    for idx, item in enumerate(train_items, start=1):
        input_payload = {"image_path": item["files"]["image_png"]}
        output_value = int(item["target_n"])
        lines.append(f"Example {idx}:")
        lines.append(f"  Input:  {repr(input_payload)}")
        lines.append(f"  Output: {output_value}")

    lines.extend(
        [
            "",
            "**IMPORTANT NOTES:**",
            "- These are TRAINING examples only.",
            "- Your program will be evaluated on unseen test images.",
            "- Infer the general pattern, do not hardcode by image path or id.",
            "",
            "**Requirements:**",
            "- Implement solve(input_data) only.",
            "- Input format: dict with key image_path.",
            "- Output format: integer.",
        ]
    )
    if include_dio_agent_description:
        lines.extend(["", _build_dio_agent_context()])
    return "\n".join(lines)


def build_system_message_for_train_items(
    train_items: list[dict[str, Any]],
    tool_list: list[str] | None = None,
    include_dio_agent_description: bool = False,
) -> str:
    return _build_system_message(
        train_items,
        tool_list=tool_list,
        include_dio_agent_description=include_dio_agent_description,
    )


def render_evaluator_for_cases(
    cases: list[dict[str, Any]],
    include_error_feedback: bool = False,
    score_with_penalties: bool = False,
) -> str:
    serialized = []
    for item in cases:
        serialized.append(
            {
                "id": item["id"],
                "input_data": {"image_path": item["files"]["image_png"]},
                "expected_output": int(item["target_n"]),
            }
        )
    return f'''"""
Auto-generated multimodal evaluator (image -> integer).
"""

import ast
import os
import importlib.util
import traceback
from dio_agent.evaluation_result import EvaluationResult

TEST_CASES = {repr(serialized)}


def _complexity_penalty(program_source):
    # Keep aligned with run_dio_agent.py complexity heuristic.
    try:
        tree = ast.parse(program_source)
        branch_nodes = 0
        loop_nodes = 0
        func_nodes = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp, ast.Match)):
                branch_nodes += 1
            elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                loop_nodes += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_nodes += 1
        size_term = max(0.0, min(1.0, (len(program_source) - 400) / 1600))
        structure_term = min(1.0, 0.03 * branch_nodes + 0.04 * loop_nodes + 0.02 * max(0, func_nodes - 1))
        return min(1.0, size_term + structure_term)
    except Exception:
        return 1.0


def _collect_example_literals(value):
    literals = set()
    try:
        rendered = repr(value)
        if len(rendered) <= 120:
            literals.add(rendered)
    except Exception:
        pass

    if isinstance(value, (list, tuple, set)):
        for item in value:
            literals.update(_collect_example_literals(item))
    elif isinstance(value, dict):
        for key, val in value.items():
            literals.update(_collect_example_literals(key))
            literals.update(_collect_example_literals(val))
    return literals


def _hardcode_similarity_penalty(program_source):
    # Weak hardcode detector: literal overlap with visible curriculum cases.
    try:
        tree = ast.parse(program_source)
        code_literals = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                value = node.value
                if isinstance(value, (bool, type(None))):
                    continue
                if isinstance(value, (int, float)) and abs(value) <= 2:
                    continue
                rendered = repr(value)
                if len(rendered) <= 120:
                    code_literals.add(rendered)

        test_literals = set()
        for case in TEST_CASES:
            test_literals.update(_collect_example_literals(case.get("input_data")))
            test_literals.update(_collect_example_literals(case.get("expected_output")))

        if not code_literals or not test_literals:
            return 0.0
        overlap = len(code_literals & test_literals)
        return min(1.0, overlap / max(4, len(code_literals)))
    except Exception:
        return 1.0


def evaluate(program_path: str) -> dict:
    try:
        pass_threshold = float(os.environ.get("MM_PASS_THRESHOLD", "0.9"))
    except ValueError:
        pass_threshold = 0.9
    pass_threshold = max(0.0, min(1.0, pass_threshold))
    try:
        spec = importlib.util.spec_from_file_location("candidate_program", program_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    except Exception as exc:
        return EvaluationResult(
            metrics={{
                "accuracy": 0.0,
                "acc_curriculum": 0.0,
                "complexity_penalty": 1.0,
                "similarity_penalty": 1.0,
                "combined_score": 0.0,
                "correct": 0,
                "tests_passed": 0,
                "total": int(len(TEST_CASES)),
                "pass_threshold": float(pass_threshold),
                "error": f"program_load_failed: {{exc}}",
            }},
            artifacts={{"error_type": type(exc).__name__, "traceback": traceback.format_exc()}},
        )

    if not hasattr(module, "solve") or not callable(getattr(module, "solve")):
        return EvaluationResult(
            metrics={{
                "accuracy": 0.0,
                "acc_curriculum": 0.0,
                "complexity_penalty": 1.0,
                "similarity_penalty": 1.0,
                "combined_score": 0.0,
                "correct": 0,
                "tests_passed": 0,
                "total": int(len(TEST_CASES)),
                "pass_threshold": float(pass_threshold),
                "error": "missing callable solve(input_data)",
            }},
            artifacts={{"error": "MissingSolve"}},
        )

    details = []
    curriculum_errors = []
    success_count = 0
    for i, case in enumerate(TEST_CASES):
        expected = case["expected_output"]
        try:
            actual = module.solve(case["input_data"])
            if isinstance(actual, bool):
                is_correct = False
            elif isinstance(actual, int):
                is_correct = (actual == expected)
            elif isinstance(actual, float):
                is_correct = (abs(actual - float(expected)) <= 1e-6)
            else:
                is_correct = False
            if is_correct:
                success_count += 1
            details.append(
                {{
                    "id": case["id"],
                    "test_case": i,
                    "input": repr(case["input_data"]),
                    "expected": repr(expected),
                    "actual": repr(actual),
                    "correct": bool(is_correct),
                }}
            )
            if (not is_correct) and {str(include_error_feedback)} and len(curriculum_errors) < 3:
                curriculum_errors.append(
                    {{
                        "test_case": i,
                        "input": repr(case["input_data"]),
                        "expected": repr(expected),
                        "actual": repr(actual),
                    }}
                )
        except Exception as exc:
            details.append(
                {{
                    "id": case["id"],
                    "test_case": i,
                    "input": repr(case["input_data"]),
                    "expected": repr(expected),
                    "error": str(exc),
                    "correct": False,
                }}
            )
            if {str(include_error_feedback)} and len(curriculum_errors) < 3:
                curriculum_errors.append(
                    {{
                        "test_case": i,
                        "input": repr(case["input_data"]),
                        "expected": repr(expected),
                        "error": str(exc),
                    }}
                )

    accuracy = float(success_count / len(TEST_CASES)) if TEST_CASES else 0.0
    try:
        with open(program_path, "r", encoding="utf-8") as f:
            program_source = f.read()
    except Exception:
        program_source = ""
    complexity_penalty = _complexity_penalty(program_source)
    similarity_penalty = _hardcode_similarity_penalty(program_source)
    if {str(score_with_penalties)}:
        combined_score = accuracy - 0.1 * complexity_penalty - 0.1 * similarity_penalty
    else:
        combined_score = accuracy
    combined_score = max(0.0, combined_score)
    artifacts = {{
        "pass_threshold": float(pass_threshold),
        "case_details": details,
        "curriculum_total": int(len(TEST_CASES)),
        "curriculum_correct": int(success_count),
    }}
    if {str(include_error_feedback)} and curriculum_errors:
        artifacts["curriculum_errors"] = curriculum_errors
    return EvaluationResult(
        metrics={{
            "accuracy": accuracy,
            "acc_curriculum": accuracy,
            "complexity_penalty": float(complexity_penalty),
            "similarity_penalty": float(similarity_penalty),
            "combined_score": combined_score,
            "correct": int(success_count),
            "tests_passed": int(success_count),
            "total": int(len(TEST_CASES)),
            "pass_threshold": float(pass_threshold),
        }},
        artifacts=artifacts,
    )
'''


def setup_multimodal_task(
    task_name: str,
    dataset_index_path: Path,
    evaluator_timeout_sec: int = 1800,
) -> Path:
    payload = json.loads(dataset_index_path.read_text(encoding="utf-8"))
    train_items = _augment_items_with_targets(
        payload.get("train", []),
        dataset_index_path=dataset_index_path,
        split_name="train",
    )
    test_items = _augment_items_with_targets(
        payload.get("test", []),
        dataset_index_path=dataset_index_path,
        split_name="test",
    )
    if not train_items or not test_items:
        raise ValueError("Dataset index must contain non-empty train/test splits.")

    task_dir = EXPERIMENT_ROOT / "tasks" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "checkpoint_interval": 5,
        "convergence_threshold": 0.001,
        "database": {
            "archive_size": 30,
            "db_path": None,
            "elite_selection_ratio": 0.2,
            "exploitation_ratio": 0.7,
            "in_memory": True,
            "log_prompts": True,
            "num_islands": 3,
            "population_size": 100,
            "similarity_threshold": 0.95,
        },
        "diff_based_evolution": True,
        "early_stopping_metric": "combined_score",
        "early_stopping_patience": 5,
        "evaluator": {
            "cascade_evaluation": False,
            "cascade_thresholds": [0.5, 0.8],
            "parallel_evaluations": 1,
            "timeout": int(evaluator_timeout_sec),
        },
        "llm": {
            "api_base": "https://openrouter.ai/api/v1",
            "api_key": "${OPENROUTER_API_KEY}",
            "evaluator_models": [{"name": "deepseek/deepseek-v3.2", "weight": 1.0}],
            "max_tokens": 16384,
            "models": [{"name": "deepseek/deepseek-v3.2", "weight": 1.0}],
            "retries": 3,
            "temperature": 0.7,
            "timeout": 60,
            "top_p": 0.95,
        },
        "log_level": "INFO",
        "max_code_length": 14000,
        "max_iterations": 12,
        "process_parallel": {"num_workers": 1},
        "evolution_trace": {
            "enabled": True,
            "format": "jsonl",
            "include_code": True,
            "include_prompts": True,
            "output_path": None,
            "buffer_size": 1,
            "compress": False,
        },
        "prompt": {
            "include_artifacts": True,
            "max_artifact_bytes": 20480,
            "system_message": _build_system_message(train_items),
            "num_top_programs": 2,
            "num_diverse_programs": 0,
            "num_inspirations": 1,
            "max_previous_attempts": 0,
            "use_template_stochasticity": True,
        },
    }

    (task_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (task_dir / "initial_program.py").write_text(_render_initial_program(), encoding="utf-8")
    (task_dir / "evaluator.py").write_text(render_evaluator_for_cases(test_items), encoding="utf-8")
    return task_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create multimodal DIO-Agent task files")
    parser.add_argument("--task-name", default="multimodal_task1")
    parser.add_argument("--dataset-index", default=str(DEFAULT_DATASET_INDEX))
    parser.add_argument("--evaluator-timeout-sec", type=int, default=1800)
    args = parser.parse_args()

    task_dir = setup_multimodal_task(
        task_name=args.task_name,
        dataset_index_path=Path(args.dataset_index).resolve(),
        evaluator_timeout_sec=args.evaluator_timeout_sec,
    )
    print(f"Multimodal task created at: {task_dir}")


if __name__ == "__main__":
    main()
