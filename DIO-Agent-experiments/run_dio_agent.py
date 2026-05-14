"""
Stage-wise IO2Code Curriculum DIO-Agent runner.

This runner is additive and does not modify baseline task files in-place.
It creates stage-specific configs/evaluators under:
  tasks/<TASK>/<output-subdir>/stage_*/
"""

import argparse
import ast
import copy
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DIO_AGENT_ROOT = (SCRIPT_DIR.parent / "DIO-Agent").resolve()
if str(LOCAL_DIO_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIO_AGENT_ROOT))


CURRICULUM_SCORE_WEIGHT = 1.0
COMPLEXITY_PENALTY_WEIGHT = 0.1
SIMILARITY_PENALTY_WEIGHT = 0.1
OUTPUT_DIR = "dio_agent_final_default"

def _parse_training_examples_from_system_message(system_message: str) -> List[Tuple[Any, Any]]:
    """
    Parse training examples from prompt.system_message.

    Expected pattern in config messages:
      Example N:
        Input:  <python-literal>
        Output: <python-literal>
    """
    if not system_message:
        return []

    # Parse line-based examples:
    # Example N:
    #   Input:  [...]
    #   Output: [...]
    example_pattern = re.compile(
        r"Example\s+\d+\s*:\s*\n\s*Input:\s*(?P<input>[^\n]*)\n\s*Output:\s*(?P<output>[^\n]*)",
        re.MULTILINE,
    )

    examples: List[Tuple[Any, Any]] = []
    for match in example_pattern.finditer(system_message):
        input_raw = match.group("input").strip()
        output_raw = match.group("output").strip()
        def _safe_parse(raw: str) -> Any:
            if raw == "":
                return ""
            try:
                return ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                # Backward compatibility for old configs where strings are not quoted.
                return raw

        input_value = _safe_parse(input_raw)
        output_value = _safe_parse(output_raw)
        examples.append((input_value, output_value))
    return examples


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _load_test_cases_from_evaluator(evaluator_path: Path) -> List[Tuple[Any, Any]]:
    source = evaluator_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                if len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                if isinstance(target, ast.Name) and target.id == "test_cases":
                    literal = ast.literal_eval(stmt.value)
                    if not isinstance(literal, list):
                        raise ValueError("test_cases in evaluator.py is not a list")
                    return literal

    raise ValueError(f"Could not locate test_cases in {evaluator_path}")


def _load_function_name(initial_program_path: Path) -> str:
    source = initial_program_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    raise ValueError(f"Could not find function definition in {initial_program_path}")


def _calc_complexity(inp: Any, out: Any) -> float:
    # Keep aligned with io2code_tasks/io2code_tasks.py heuristic.
    score = 0.0
    if inp is None:
        score += 0.0
    elif isinstance(inp, (int, float)):
        score += 0.2 + min(abs(inp) / 100.0, 1.0)
    elif isinstance(inp, str):
        score += 0.3 + len(inp) * 0.1
    elif isinstance(inp, (list, tuple)):
        score += 0.5 + len(inp) * 0.2

    if isinstance(out, (list, tuple)):
        score += 0.3 + len(out) * 0.1

    return score


def _value_length(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (bool, int, float)):
        return 1.0
    if isinstance(value, str):
        return float(len(value))
    if isinstance(value, (list, tuple, set)):
        return float(len(value))
    if isinstance(value, dict):
        return float(len(value))
    return 1.0


def _calc_length_priority(inp: Any, out: Any) -> float:
    # Prefer the simple "shorter data first" order used in incremental curricula.
    return _value_length(inp) + 0.5 * _value_length(out)


def _build_curriculum_boundaries(total: int, stage_fractions: Sequence[float]) -> List[int]:
    if not stage_fractions:
        raise ValueError("stage_fractions cannot be empty")
    if stage_fractions[-1] < 1.0:
        raise ValueError("Final stage fraction must be >= 1.0")

    raw = [max(1, math.ceil(total * frac)) for frac in stage_fractions]
    boundaries: List[int] = []
    prev = 0
    for idx, value in enumerate(raw):
        value = min(total, value)
        if idx < len(raw) - 1:
            value = max(prev + 1, value)
        else:
            value = total
        boundaries.append(value)
        prev = value
    return boundaries


def _sample_replay_cases(
    previous_cases: Sequence[Tuple[Any, Any]],
    replay_ratio: float,
    rng: random.Random,
) -> List[Tuple[Any, Any]]:
    if not previous_cases:
        return []
    count = max(1, math.ceil(len(previous_cases) * replay_ratio))
    count = min(count, len(previous_cases))
    return rng.sample(list(previous_cases), count)


def _make_dio_agent_context(stage_index: int, total_stages: int, new_cases: Sequence[Tuple[Any, Any]]) -> str:
    preview_lines = []
    for idx, (inp, out) in enumerate(new_cases[:4], start=1):
        preview_lines.append(f"  - New test {idx}: input={repr(inp)} -> output={repr(out)}")

    if len(new_cases) > 4:
        preview_lines.append(f"  - ... and {len(new_cases) - 4} more newly added tests")

    preview = "\n".join(preview_lines) if preview_lines else "  - No newly added tests in this stage."
    return f"""
## DIO_AGENT_CONTEXT (Stage {stage_index}/{total_stages})
- Goal: pass all curriculum tests while preserving earlier behavior.
- Apply minimal transformations first (incremental style), avoid over-engineering.

### Newly added tests for this stage
{preview}

### Transformation priority
1) nil -> constant
2) constant -> scalar
3) statement -> statements
4) unconditional -> if
5) scalar -> array
6) if -> while
7) expression -> function

### Anti-hardcode rules
- Do NOT match full concrete inputs (e.g. exact list equality checks).
- Do NOT build dictionary lookups from seen examples.
- Infer a general rule that extrapolates to unseen inputs.
""".strip()


def _make_stage_system_prompt(
    base_system_prompt: str,
    stage_index: int,
    total_stages: int,
    new_cases: Sequence[Tuple[Any, Any]],
    visible_cases: Sequence[Tuple[Any, Any]],
    include_dio_agent_description: bool = True,
) -> str:
    # Replace full training examples block with stage-visible subset when possible.
    stage_examples_lines = []
    for idx, (inp, out) in enumerate(visible_cases, start=1):
        stage_examples_lines.append(
            f"Example {idx}:\n"
            f"  Input:  {repr(inp)}\n"
            f"  Output: {repr(out)}"
        )
    stage_examples_text = "\n".join(stage_examples_lines)

    replaced_message = base_system_prompt
    if stage_examples_text:
        training_block_pattern = re.compile(
            r"(\*\*Training Examples \(you can see these\):\*\*\s*)(.*?)(\s*\*\*IMPORTANT NOTES:\*\*)",
            re.DOTALL,
        )
        replacement_block = (
            r"\1"
            f"\n{stage_examples_text}\n"
            f"\n- Visible curriculum examples in this stage: {len(visible_cases)}"
            r"\3"
        )
        replaced_message = training_block_pattern.sub(replacement_block, base_system_prompt, count=1)

    if not include_dio_agent_description:
        return f"{replaced_message.strip()}\n"

    suffix = _make_dio_agent_context(stage_index, total_stages, new_cases)
    return f"{replaced_message.strip()}\n\n{suffix}\n"


def _render_stage_evaluator(
    function_name: str,
    curriculum_cases: Sequence[Tuple[Any, Any]],
    replay_cases: Sequence[Tuple[Any, Any]],
    include_error_feedback: bool = False,
) -> str:
    curriculum_literal = repr(list(curriculum_cases))
    replay_literal = repr(list(replay_cases))
    error_artifact_block = ""
    if include_error_feedback:
        error_artifact_block = """
        artifacts["curriculum_errors"] = curr_errors
        artifacts["replay_errors"] = replay_errors"""

    return f'''"""
Auto-generated DIO-Agent evaluator for a curriculum stage.
"""

import ast
import importlib.util
import traceback
from dio_agent.evaluation_result import EvaluationResult


CURRICULUM_CASES = {curriculum_literal}
REPLAY_CASES = {replay_literal}


def _compare_outputs(actual, expected):
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(_compare_outputs(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, int) and isinstance(actual, int):
            return actual == expected
        return abs(actual - expected) < 1e-6
    return actual == expected


def _evaluate_cases(func, cases):
    if not cases:
        return 1.0, 0, 0, []
    correct = 0
    errors = []
    for i, (input_data, expected_output) in enumerate(cases):
        try:
            actual_output = func(input_data)
            if _compare_outputs(actual_output, expected_output):
                correct += 1
            elif len(errors) < 3:
                errors.append({{
                    "test_case": i,
                    "input": repr(input_data),
                    "expected": repr(expected_output),
                    "actual": repr(actual_output),
                }})
        except Exception as e:
            if len(errors) < 3:
                errors.append({{
                    "test_case": i,
                    "input": repr(input_data),
                    "error": str(e),
                }})
    total = len(cases)
    return correct / total if total else 1.0, correct, total, errors


def _complexity_penalty(program_source):
    # Lightweight structural penalty in [0, 1].
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
        for k, v in value.items():
            literals.update(_collect_example_literals(k))
            literals.update(_collect_example_literals(v))
    return literals


def _hardcode_similarity_penalty(program_source):
    # Measure literal overlap with visible curriculum tests as a weak hardcode signal.
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
        for inp, out in CURRICULUM_CASES:
            test_literals.update(_collect_example_literals(inp))
            test_literals.update(_collect_example_literals(out))

        if not code_literals or not test_literals:
            return 0.0

        overlap = len(code_literals & test_literals)
        # Scale by number of literals in code to avoid penalizing tiny incidental overlap.
        return min(1.0, overlap / max(4, len(code_literals)))
    except Exception:
        return 1.0


def evaluate(program_path: str) -> dict:
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "{function_name}"):
            return EvaluationResult(
                metrics={{
                    "accuracy": 0.0,
                    "acc_curriculum": 0.0,
                    "acc_replay": 0.0,
                    "replay_error_rate": 1.0,
                    "complexity_penalty": 1.0,
                    "similarity_penalty": 1.0,
                    "correct": 0,
                    "total": len(CURRICULUM_CASES),
                    "curriculum_total": len(CURRICULUM_CASES),
                    "curriculum_correct": 0,
                    "combined_score": 0.0,
                    "error": "missing function: {function_name}",
                }},
                artifacts={{"error": "MissingFunction"}},
            )

        func = getattr(program, "{function_name}")
        acc_curr, correct_curr, total_curr, curr_errors = _evaluate_cases(func, CURRICULUM_CASES)
        acc_replay, correct_replay, total_replay, replay_errors = _evaluate_cases(func, REPLAY_CASES)
        replay_error_rate = 1.0 - acc_replay

        with open(program_path, "r", encoding="utf-8") as f:
            program_source = f.read()
        complexity_penalty = _complexity_penalty(program_source)
        similarity_penalty = _hardcode_similarity_penalty(program_source)

        # Stage score uses visible curriculum tests with complexity/similarity regularization.
        combined_score = (
            {CURRICULUM_SCORE_WEIGHT} * acc_curr
            - {COMPLEXITY_PENALTY_WEIGHT} * complexity_penalty
            - {SIMILARITY_PENALTY_WEIGHT} * similarity_penalty
        )
        combined_score = max(0.0, combined_score)

        artifacts = {{
            "curriculum_total": total_curr,
            "curriculum_correct": correct_curr,
            "replay_total": total_replay,
            "replay_correct": correct_replay,
        }}
{error_artifact_block}

        return EvaluationResult(
            metrics={{
                "accuracy": float(acc_curr),
                "acc_curriculum": float(acc_curr),
                "acc_replay": float(acc_replay),
                "replay_error_rate": float(replay_error_rate),
                "complexity_penalty": float(complexity_penalty),
                "similarity_penalty": float(similarity_penalty),
                "correct": int(correct_curr),
                "total": int(total_curr),
                "curriculum_correct": int(correct_curr),
                "curriculum_total": int(total_curr),
                "combined_score": float(combined_score),
            }},
            artifacts=artifacts,
        )
    except Exception as e:
        return EvaluationResult(
            metrics={{
                "accuracy": 0.0,
                "acc_curriculum": 0.0,
                "acc_replay": 0.0,
                "replay_error_rate": 1.0,
                "complexity_penalty": 1.0,
                "similarity_penalty": 1.0,
                "correct": 0,
                "total": len(CURRICULUM_CASES),
                "curriculum_total": len(CURRICULUM_CASES),
                "curriculum_correct": 0,
                "combined_score": 0.0,
                "error": str(e),
            }},
            artifacts={{
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }},
        )
'''


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _run_subprocess(cmd: List[str], cwd: Path, env: dict) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit_code={result.returncode}: {' '.join(cmd)}")


def _read_iterations_executed(output_dir: Path) -> int:
    """
    Read actual executed iterations from evolution_trace.jsonl.
    Falls back to 0 when trace is unavailable.
    """
    trace_path = output_dir / "evolution_trace.jsonl"
    if not trace_path.exists():
        return 0

    max_iteration = 0
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = item.get("iteration", 0) if isinstance(item, dict) else 0
                if isinstance(value, int) and value > max_iteration:
                    max_iteration = value
    except OSError:
        return 0

    return max_iteration


def _collect_generated_stage_programs(stage_output_dir: Path) -> List[dict]:
    """
    Collect generated program payloads from latest checkpoint programs/*.json.
    Only keeps programs with non-empty code and iteration_found > 0 (exclude initial seeds).
    """
    checkpoints_root = stage_output_dir / "checkpoints"
    if not checkpoints_root.exists():
        return []

    checkpoint_dirs = []
    for child in checkpoints_root.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint_"):
            continue
        suffix = child.name.split("checkpoint_", 1)[-1]
        try:
            idx = int(suffix)
        except ValueError:
            continue
        checkpoint_dirs.append((idx, child))

    if not checkpoint_dirs:
        return []

    _, latest_checkpoint = max(checkpoint_dirs, key=lambda x: x[0])
    programs_dir = latest_checkpoint / "programs"
    if not programs_dir.exists():
        return []

    candidates: List[dict] = []
    for program_json in programs_dir.glob("*.json"):
        try:
            with open(program_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        code = payload.get("code")
        if not isinstance(code, str) or not code.strip():
            continue
        iteration_found = payload.get("iteration_found")
        if not isinstance(iteration_found, int) or iteration_found <= 0:
            continue
        candidates.append(payload)

    return candidates


def _build_seed_checkpoint_for_next_stage(
    checkpoint_dir: Path,
    num_islands: int,
    best_code: str,
    random_program_payload: dict | None,
    random_island_id: int,
) -> Path:
    """
    Create a synthetic checkpoint to seed the next stage:
    - one random island (if payload provided),
    - remaining islands seeded by current stage best.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    programs_dir = checkpoint_dir / "programs"
    programs_dir.mkdir(parents=True, exist_ok=True)

    islands: List[List[str]] = []
    all_program_ids: List[str] = []
    for island_id in range(num_islands):
        use_random_seed = random_program_payload is not None and island_id == random_island_id
        if use_random_seed:
            seed_code = random_program_payload.get("code", best_code)
            seed_parent = random_program_payload.get("parent_id")
            seed_generation = random_program_payload.get("generation", 0)
        else:
            seed_code = best_code
            seed_parent = None
            seed_generation = 0

        program_id = str(uuid.uuid4())
        program_payload = {
            "id": program_id,
            "code": seed_code,
            "changes_description": "dio-agent inter-stage seed",
            "language": "python",
            "parent_id": seed_parent,
            "generation": int(seed_generation) if isinstance(seed_generation, int) else 0,
            "timestamp": 0.0,
            "iteration_found": 0,
            # IMPORTANT: never carry previous-stage metrics into next stage.
            # Assign a very low placeholder score so real current-stage evaluations
            # can replace these seeds as soon as any child is evaluated.
            "metrics": {"combined_score": -1e9},
            "complexity": 0.0,
            "diversity": 0.0,
            "metadata": {"island": island_id},
            "prompts": None,
            "artifacts_json": None,
            "artifact_dir": None,
            "embedding": None,
        }
        with open(programs_dir / f"{program_id}.json", "w", encoding="utf-8") as f:
            json.dump(program_payload, f, ensure_ascii=False)

        islands.append([program_id])
        all_program_ids.append(program_id)

    metadata = {
        "island_feature_maps": [{} for _ in range(num_islands)],
        "islands": islands,
        "archive": all_program_ids,
        # Let DIOAgent recalculate best program from current-stage evaluations.
        "best_program_id": None,
        "island_best_programs": [None for _ in range(num_islands)],
        "last_iteration": 0,
        "current_island": 0,
        "island_generations": [0 for _ in range(num_islands)],
        "last_migration_generation": 0,
        "feature_stats": {},
    }
    with open(checkpoint_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    return checkpoint_dir


def _normalize_evaluation_result(raw_result: Any) -> dict:
    """
    Normalize evaluator outputs to a plain dictionary payload.

    Evaluators may return:
      - EvaluationResult(metrics=..., artifacts=...)
      - {"metric": value, ...}
      - {"metrics": {...}, "artifacts": {...}}
    """
    if raw_result is None:
        return {"metrics": {}, "artifacts": {}}

    metrics = getattr(raw_result, "metrics", None)
    artifacts = getattr(raw_result, "artifacts", None)
    if isinstance(metrics, dict):
        return {"metrics": metrics, "artifacts": artifacts if isinstance(artifacts, dict) else {}}

    if isinstance(raw_result, dict):
        if "metrics" in raw_result and isinstance(raw_result["metrics"], dict):
            return {
                "metrics": raw_result["metrics"],
                "artifacts": raw_result.get("artifacts", {})
                if isinstance(raw_result.get("artifacts", {}), dict)
                else {},
            }
        return {"metrics": raw_result, "artifacts": {}}

    raise TypeError(f"Unsupported evaluator return type: {type(raw_result).__name__}")


def _evaluate_with_full_evaluator(full_evaluator_path: Path, best_program_path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("full_evaluator", str(full_evaluator_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    raw_result = module.evaluate(str(best_program_path))
    return _normalize_evaluation_result(raw_result)


def _compare_outputs(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(_compare_outputs(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, int) and isinstance(actual, int):
            return actual == expected
        return abs(actual - expected) < 1e-6
    return actual == expected


def _evaluate_cases(func: Any, cases: Sequence[Tuple[Any, Any]]) -> Tuple[float, int, int, List[dict]]:
    if not cases:
        return 1.0, 0, 0, []
    correct = 0
    errors: List[dict] = []
    for i, (input_data, expected_output) in enumerate(cases):
        try:
            actual_output = func(input_data)
            if _compare_outputs(actual_output, expected_output):
                correct += 1
            elif len(errors) < 3:
                errors.append(
                    {
                        "test_case": i,
                        "input": repr(input_data),
                        "expected": repr(expected_output),
                        "actual": repr(actual_output),
                    }
                )
        except Exception as e:
            if len(errors) < 3:
                errors.append({"test_case": i, "input": repr(input_data), "error": str(e)})
    total = len(cases)
    return correct / total if total else 1.0, correct, total, errors


def _complexity_penalty(program_source: str) -> float:
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


def _collect_example_literals(value: Any) -> set[str]:
    literals: set[str] = set()
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


def _hardcode_similarity_penalty(program_source: str, cases: Sequence[Tuple[Any, Any]]) -> float:
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
        for inp, out in cases:
            test_literals.update(_collect_example_literals(inp))
            test_literals.update(_collect_example_literals(out))

        if not code_literals or not test_literals:
            return 0.0

        overlap = len(code_literals & test_literals)
        return min(1.0, overlap / max(4, len(code_literals)))
    except Exception:
        return 1.0


def _evaluate_program_against_training_examples(
    program_path: Path,
    function_name: str,
    training_cases: Sequence[Tuple[Any, Any]],
) -> dict:
    try:
        spec = importlib.util.spec_from_file_location(f"candidate_program_{uuid.uuid4().hex}", str(program_path))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        if not hasattr(module, function_name):
            return {
                "metrics": {
                    "accuracy": 0.0,
                    "acc_curriculum": 0.0,
                    "complexity_penalty": 1.0,
                    "similarity_penalty": 1.0,
                    "combined_score": 0.0,
                    "error": f"missing function: {function_name}",
                },
                "artifacts": {"error": "MissingFunction"},
            }

        func = getattr(module, function_name)
        acc_curr, correct_curr, total_curr, errors = _evaluate_cases(func, training_cases)
        source = program_path.read_text(encoding="utf-8")
        complexity_penalty = _complexity_penalty(source)
        similarity_penalty = _hardcode_similarity_penalty(source, training_cases)
        combined_score = (
            CURRICULUM_SCORE_WEIGHT * acc_curr
            - COMPLEXITY_PENALTY_WEIGHT * complexity_penalty
            - SIMILARITY_PENALTY_WEIGHT * similarity_penalty
        )
        combined_score = max(0.0, combined_score)
        return {
            "metrics": {
                "accuracy": float(acc_curr),
                "acc_curriculum": float(acc_curr),
                "correct": int(correct_curr),
                "total": int(total_curr),
                "complexity_penalty": float(complexity_penalty),
                "similarity_penalty": float(similarity_penalty),
                "combined_score": float(combined_score),
            },
            "artifacts": {"training_errors": errors},
        }
    except Exception as e:
        return {
            "metrics": {
                "accuracy": 0.0,
                "acc_curriculum": 0.0,
                "complexity_penalty": 1.0,
                "similarity_penalty": 1.0,
                "combined_score": 0.0,
                "error": str(e),
            },
            "artifacts": {"error_type": type(e).__name__, "error_message": str(e)},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-wise DIO-Agent for one task")
    parser.add_argument("--task", required=True, help="Task name under DIO-Agent-experiments/tasks")
    parser.add_argument("--stage-fractions", default="0.2,0.4,0.7,1.0", help="Curriculum fractions per stage")
    parser.add_argument("--stage-iterations", default="8,8,10,12", help="Iteration budget per stage")
    parser.add_argument("--replay-ratio", type=float, default=0.25, help="Replay sample ratio from prior stage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for replay sampling")
    parser.add_argument(
        "--interstage-init-mode",
        choices=["best_only", "one_random_island"],
        default="best_only",
        help=(
            "How to initialize islands between stages. "
            "'best_only': all islands start from stage best (baseline). "
            "'one_random_island': exactly one island starts from a random generated program, others use stage best."
        ),
    )
    parser.add_argument(
        "--curriculum-source",
        choices=["training_examples", "evaluator_test_cases"],
        default="training_examples",
        help="Source for stage curriculum cases (default: training examples from system message)",
    )
    parser.add_argument(
        "--curriculum-order",
        choices=["original", "complexity", "length"],
        default="length",
        help="Order of curriculum cases before stage partition",
    )
    parser.add_argument(
        "--include-error-feedback",
        action="store_true",
        help="Include failed case details in evaluator artifacts for prompt feedback",
    )
    parser.add_argument(
        "--no_dio_agent",
        action="store_true",
        help="Disable the DIO-Agent guidance while keeping stage-wise curriculum evolution",
    )
    parser.add_argument(
        "--final-selection-mode",
        choices=["stage4_best", "all_stage_candidates_training_reselect"],
        default="stage4_best",
        help=(
            "Final program selection strategy. "
            "'stage4_best': keep baseline behavior (use last stage best only). "
            "'all_stage_candidates_training_reselect': re-evaluate all stage best candidates "
            "on full training examples and pick highest combined_score."
        ),
    )
    parser.add_argument(
        "--output-subdir",
        default=OUTPUT_DIR,
        help="Output folder name under tasks/<TASK>/ (default: dio_agent_final_default)",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override the LLM API base for dio_agent.cli",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable name containing the LLM API key",
    )
    parser.add_argument(
        "--primary-model",
        default=None,
        help="Override the primary LLM model name for dio_agent.cli",
    )
    parser.add_argument(
        "--secondary-model",
        default=None,
        help="Optional secondary LLM model name for dio_agent.cli",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only prepare stage files; do not execute")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    task_dir = script_dir / "tasks" / args.task
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    base_config_path = task_dir / "config.yaml"
    base_evaluator_path = task_dir / "evaluator.py"
    base_initial_program = task_dir / "initial_program.py"
    if not base_config_path.exists() or not base_evaluator_path.exists() or not base_initial_program.exists():
        raise FileNotFoundError("Task must include config.yaml, evaluator.py, and initial_program.py")

    stage_fractions = _parse_csv_floats(args.stage_fractions)
    stage_iterations = _parse_csv_ints(args.stage_iterations)
    if len(stage_fractions) != len(stage_iterations):
        raise ValueError("stage-fractions and stage-iterations must have the same length")

    base_config = _load_yaml(base_config_path)
    base_prompt = base_config.get("prompt", {}).get("system_message", "")

    training_examples = _parse_training_examples_from_system_message(base_prompt)
    if args.curriculum_source == "training_examples":
        if not training_examples:
            raise ValueError(
                "No training examples parsed from prompt.system_message. "
                "Use --curriculum-source evaluator_test_cases or fix config system message format."
            )
        curriculum_cases = training_examples
    else:
        curriculum_cases = _load_test_cases_from_evaluator(base_evaluator_path)

    # Full evaluator test cases are always used for stage fitness scoring.
    if args.curriculum_order == "complexity":
        sorted_cases = sorted(curriculum_cases, key=lambda t: _calc_complexity(t[0], t[1]))
    elif args.curriculum_order == "length":
        sorted_cases = sorted(curriculum_cases, key=lambda t: _calc_length_priority(t[0], t[1]))
    else:
        sorted_cases = list(curriculum_cases)

    function_name = _load_function_name(base_initial_program)
    boundaries = _build_curriculum_boundaries(len(sorted_cases), stage_fractions)
    total_stages = len(boundaries)
    rng = random.Random(args.seed)

    local_dio_agent_root = (script_dir.parent / "DIO-Agent").resolve()
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{local_dio_agent_root}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(local_dio_agent_root)

    output_subdir = args.output_subdir.strip() if isinstance(args.output_subdir, str) else OUTPUT_DIR
    if not output_subdir:
        output_subdir = OUTPUT_DIR

    run_root = task_dir / output_subdir
    run_root.mkdir(parents=True, exist_ok=True)

    previous_boundary = 0
    previous_best_program = base_initial_program
    previous_seed_checkpoint: Path | None = None
    stage_best_programs: List[Path] = []
    stage_summaries = []
    task_token_usage = {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    total_iterations_executed = 0

    for stage_index, (boundary, stage_iters) in enumerate(zip(boundaries, stage_iterations), start=1):
        stage_dir = run_root / f"stage_{stage_index}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        curriculum_cases = sorted_cases[:boundary]
        new_cases = sorted_cases[previous_boundary:boundary]
        replay_cases = _sample_replay_cases(sorted_cases[:previous_boundary], args.replay_ratio, rng)

        stage_evaluator_path = stage_dir / "evaluator_stage.py"
        stage_evaluator_path.write_text(
            _render_stage_evaluator(
                function_name=function_name,
                curriculum_cases=curriculum_cases,
                replay_cases=replay_cases,
                include_error_feedback=args.include_error_feedback,
            ),
            encoding="utf-8",
        )

        stage_config = copy.deepcopy(base_config)
        stage_config["max_iterations"] = stage_iters
        if args.interstage_init_mode == "one_random_island":
            # Ensure we can sample from all generated programs at stage end.
            stage_config["checkpoint_interval"] = 1

        llm_cfg = stage_config.setdefault("llm", {})
        if args.api_base:
            llm_cfg["api_base"] = args.api_base
        if args.api_key_env:
            llm_cfg["api_key"] = f"${{{args.api_key_env}}}"

        prompt_cfg = stage_config.setdefault("prompt", {})
        prompt_cfg["num_top_programs"] = 2
        prompt_cfg["num_diverse_programs"] = 0
        prompt_cfg["num_inspirations"] = 1
        prompt_cfg["max_previous_attempts"] = 0
        prompt_cfg["system_message"] = _make_stage_system_prompt(
            base_system_prompt=base_prompt,
            stage_index=stage_index,
            total_stages=total_stages,
            new_cases=new_cases,
            visible_cases=sorted_cases[:boundary],
            include_dio_agent_description=not args.no_dio_agent,
        )

        stage_config_path = stage_dir / "config_stage.yaml"
        _dump_yaml(stage_config_path, stage_config)

        output_dir = stage_dir / "dio_agent_output"
        summary_item = {
            "stage": stage_index,
            "iterations": stage_iters,
            "curriculum_size": len(curriculum_cases),
            "new_cases": len(new_cases),
            "replay_cases": len(replay_cases),
            "config": str(stage_config_path),
            "evaluator": str(stage_evaluator_path),
            "initial_program": str(previous_best_program),
            "initialization_mode": args.interstage_init_mode,
            "output_dir": str(output_dir),
            "token_usage": {
                "llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "iterations_executed": 0,
            "token_usage_per_iteration": {
                "avg_prompt_tokens": 0.0,
                "avg_completion_tokens": 0.0,
                "avg_total_tokens": 0.0,
            },
        }

        if not args.dry_run:
            cmd = [
                sys.executable,
                "-m",
                "dio_agent.cli",
                str(previous_best_program),
                str(stage_evaluator_path),
                "--config",
                str(stage_config_path),
                "--iterations",
                str(stage_iters),
                "--output",
                str(output_dir),
            ]
            if args.api_base:
                cmd.extend(["--api-base", str(args.api_base)])
            if args.api_key_env:
                cmd.extend(["--api-key-env", str(args.api_key_env)])
            if args.primary_model:
                cmd.extend(["--primary-model", str(args.primary_model)])
            if args.secondary_model:
                cmd.extend(["--secondary-model", str(args.secondary_model)])
            if previous_seed_checkpoint is not None:
                cmd.extend(["--checkpoint", str(previous_seed_checkpoint)])
                summary_item["seed_checkpoint"] = str(previous_seed_checkpoint)
            else:
                summary_item["seed_checkpoint"] = None
            _run_subprocess(cmd, cwd=script_dir, env=env)

            stage_best = output_dir / "best" / "best_program.py"
            stage_best_info = output_dir / "best" / "best_program_info.json"
            if not stage_best.exists():
                raise FileNotFoundError(f"Stage {stage_index} best program not found: {stage_best}")
            previous_best_program = stage_best
            stage_best_programs.append(stage_best)

            if stage_best_info.exists():
                with open(stage_best_info, "r", encoding="utf-8") as f:
                    summary_item["best_info"] = json.load(f)
                stage_token_usage = summary_item["best_info"].get("token_usage", {})
                if isinstance(stage_token_usage, dict):
                    for key in ("llm_calls", "prompt_tokens", "completion_tokens", "total_tokens"):
                        value = stage_token_usage.get(key, 0)
                        if isinstance(value, (int, float)):
                            summary_item["token_usage"][key] = int(value)
                            task_token_usage[key] += int(value)

            stage_iters_executed = _read_iterations_executed(output_dir)
            summary_item["iterations_executed"] = stage_iters_executed
            total_iterations_executed += stage_iters_executed
            if stage_iters_executed > 0:
                summary_item["token_usage_per_iteration"] = {
                    "avg_prompt_tokens": summary_item["token_usage"]["prompt_tokens"] / stage_iters_executed,
                    "avg_completion_tokens": summary_item["token_usage"]["completion_tokens"] / stage_iters_executed,
                    "avg_total_tokens": summary_item["token_usage"]["total_tokens"] / stage_iters_executed,
                }

            # Prepare next-stage mixed seeds (ablation: one_random_island).
            previous_seed_checkpoint = None
            if args.interstage_init_mode == "one_random_island" and stage_index < total_stages:
                best_code = stage_best.read_text(encoding="utf-8")

                generated_candidates = _collect_generated_stage_programs(output_dir)
                random_candidate = rng.choice(generated_candidates) if generated_candidates else None

                num_islands = int(stage_config.get("database", {}).get("num_islands", 3))
                num_islands = max(1, num_islands)
                random_island_id = rng.randrange(num_islands)
                next_seed_ckpt_dir = stage_dir / "seed_checkpoint_for_next_stage"
                previous_seed_checkpoint = _build_seed_checkpoint_for_next_stage(
                    checkpoint_dir=next_seed_ckpt_dir,
                    num_islands=num_islands,
                    best_code=best_code,
                    random_program_payload=random_candidate,
                    random_island_id=random_island_id,
                )
                summary_item["next_stage_seed"] = {
                    "mode": "one_random_island",
                    "num_islands": num_islands,
                    "random_island_id": random_island_id,
                    "random_candidate_selected": random_candidate is not None,
                    "seed_checkpoint_path": str(previous_seed_checkpoint),
                }
            else:
                summary_item["next_stage_seed"] = {"mode": "best_only"}

        stage_summaries.append(summary_item)
        previous_boundary = boundary

    final_eval = None
    final_best_program = previous_best_program
    final_selection = {"mode": args.final_selection_mode, "selected_from": "stage4_best"}
    if not args.dry_run:
        if args.final_selection_mode == "all_stage_candidates_training_reselect":
            training_cases_for_reselect = training_examples if training_examples else sorted_cases
            candidate_paths: List[Path] = []
            seen = set()
            for candidate in stage_best_programs:
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidate_paths.append(candidate)

            scored_candidates = []
            for candidate in candidate_paths:
                result = _evaluate_program_against_training_examples(
                    program_path=candidate,
                    function_name=function_name,
                    training_cases=training_cases_for_reselect,
                )
                metrics = result.get("metrics", {})
                combined = metrics.get("combined_score", 0.0) if isinstance(metrics, dict) else 0.0
                accuracy = metrics.get("accuracy", 0.0) if isinstance(metrics, dict) else 0.0
                scored_candidates.append(
                    {
                        "program_path": str(candidate),
                        "training_eval": result,
                        "combined_score": float(combined) if isinstance(combined, (int, float)) else 0.0,
                        "accuracy": float(accuracy) if isinstance(accuracy, (int, float)) else 0.0,
                    }
                )

            if scored_candidates:
                scored_candidates.sort(
                    key=lambda item: (
                        item["combined_score"],
                        item["accuracy"],
                    ),
                    reverse=True,
                )
                final_best_program = Path(scored_candidates[0]["program_path"])
                selected_copy = run_root / "final_selected_program.py"
                shutil.copyfile(final_best_program, selected_copy)
                final_best_program = selected_copy
                final_selection = {
                    "mode": args.final_selection_mode,
                    "selected_from": "all_stage_candidates",
                    "selection_cases": "training_examples" if training_examples else "curriculum_cases_fallback",
                    "selection_case_count": len(training_cases_for_reselect),
                    "selected_candidate_original_path": scored_candidates[0]["program_path"],
                    "ranked_candidates": scored_candidates,
                }

        final_eval = _evaluate_with_full_evaluator(base_evaluator_path, final_best_program)

    run_summary = {
        "task": args.task,
        "dry_run": args.dry_run,
        "seed": args.seed,
        "stage_fractions": stage_fractions,
        "stage_iterations": stage_iterations,
        "replay_ratio": args.replay_ratio,
        "curriculum_source": args.curriculum_source,
        "curriculum_order": args.curriculum_order,
        "final_selection_mode": args.final_selection_mode,
        "total_test_cases": len(sorted_cases),
        "stages": stage_summaries,
        "token_usage": task_token_usage,
        "iterations_executed": total_iterations_executed,
        "token_usage_per_iteration": {
            "avg_prompt_tokens": (
                task_token_usage["prompt_tokens"] / total_iterations_executed if total_iterations_executed > 0 else 0.0
            ),
            "avg_completion_tokens": (
                task_token_usage["completion_tokens"] / total_iterations_executed if total_iterations_executed > 0 else 0.0
            ),
            "avg_total_tokens": (
                task_token_usage["total_tokens"] / total_iterations_executed if total_iterations_executed > 0 else 0.0
            ),
        },
        "final_best_program": str(final_best_program),
        "final_selection": final_selection,
        "final_full_evaluation": final_eval,
    }

    summary_path = run_root / "dio_agent_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(f"\nDIO-Agent run summary saved to: {summary_path}")
    if final_eval:
        metrics = final_eval.get("metrics", {})
        print("Final full-evaluator metrics:")
        for k, v in metrics.items():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
