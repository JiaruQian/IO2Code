"""
Utilities for training-only evaluators in baseline DIOAgent runs.
"""

import ast
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

SIMILARITY_PENALTY_WEIGHT = 0.0


def parse_training_examples_from_system_message(system_message: str) -> List[Tuple[Any, Any]]:
    if not system_message:
        return []

    pattern = re.compile(
        r"Example\s+\d+\s*:\s*\n\s*Input:\s*(?P<input>[^\n]*)\n\s*Output:\s*(?P<output>[^\n]*)",
        re.MULTILINE,
    )
    examples: List[Tuple[Any, Any]] = []
    for match in pattern.finditer(system_message):
        input_raw = match.group("input").strip()
        output_raw = match.group("output").strip()
        def _safe_parse(raw: str) -> Any:
            # Empty line means empty string in prompt examples.
            if raw == "":
                return ""
            try:
                return ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                # Backward compatibility for old configs where strings are not quoted.
                return raw

        inp = _safe_parse(input_raw)
        out = _safe_parse(output_raw)
        examples.append((inp, out))
    return examples


def build_base_dio_agent_context() -> str:
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
- Do NOT match full concrete inputs (e.g. exact list equality checks).
- Do NOT build dictionary lookups from seen examples.
- Infer a general rule that extrapolates to unseen inputs.
""".strip()


def append_base_dio_agent_description(system_message: str) -> str:
    if "## DIO_AGENT_CONTEXT" in system_message:
        return system_message if system_message.endswith("\n") else f"{system_message}\n"
    return f"{system_message.rstrip()}\n\n{build_base_dio_agent_context()}\n"


def prepare_runtime_config(
    task_dir: Path,
    output_name: str = ".config_train_runtime.yaml",
    with_dio_agent: bool = False,
) -> Path:
    config_path = task_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Task missing config file under {task_dir}")

    if not with_dio_agent:
        return config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    prompt_cfg = config.setdefault("prompt", {})
    prompt_cfg["system_message"] = append_base_dio_agent_description(prompt_cfg.get("system_message", ""))

    output_path = task_dir / output_name
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return output_path


def load_function_name(initial_program_path: Path) -> str:
    source = initial_program_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    raise ValueError(f"Could not find function definition in {initial_program_path}")


def _render_training_evaluator(
    function_name: str,
    train_cases: List[Tuple[Any, Any]],
    include_error_feedback: bool = False,
) -> str:
    error_collection_block = ""
    error_artifact_block = ""
    if include_error_feedback:
        error_collection_block = """
                elif len(errors) < 3:
                    errors.append({
                        "test_case": i,
                        "input": _format_value(input_data),
                        "expected": _format_value(expected_output),
                        "actual": _format_value(actual_output),
                    })"""
        error_artifact_block = """
        if errors:
            artifacts["errors"] = errors
            artifacts["curriculum_errors"] = errors"""

    return f'''"""
Auto-generated evaluator for training-set-only evolution.
"""

import importlib.util
import traceback
import time
from dio_agent.evaluation_result import EvaluationResult


train_cases = {repr(list(train_cases))}


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


def _format_value(value, max_len=50):
    s = str(value)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


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
    # Weak hardcode detector: overlap between program literals and training-example literals.
    try:
        import ast

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

        train_literals = set()
        for inp, out in train_cases:
            train_literals.update(_collect_example_literals(inp))
            train_literals.update(_collect_example_literals(out))

        if not code_literals or not train_literals:
            return 0.0

        overlap = len(code_literals & train_literals)
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
                    "correct": 0,
                    "total": len(train_cases),
                    "similarity_penalty": 1.0,
                    "combined_score": 0.0,
                    "error": "missing function: {function_name}",
                }},
                artifacts={{"error_type": "MissingFunction"}},
            )

        func = getattr(program, "{function_name}")
        correct = 0
        total = len(train_cases)
        errors = []

        start_time = time.time()
        for i, (input_data, expected_output) in enumerate(train_cases):
            try:
                actual_output = func(input_data)
                if _compare_outputs(actual_output, expected_output):
                    correct += 1
{error_collection_block}
            except Exception as e:
                if {str(include_error_feedback)} and len(errors) < 3:
                    errors.append({{
                        "test_case": i,
                        "input": _format_value(input_data),
                        "error": str(e),
                    }})
        eval_time = time.time() - start_time
        accuracy = correct / total if total > 0 else 0.0
        with open(program_path, "r", encoding="utf-8") as f:
            program_source = f.read()
        similarity_penalty = _hardcode_similarity_penalty(program_source)
        combined_score = accuracy - {SIMILARITY_PENALTY_WEIGHT} * similarity_penalty
        artifacts = {{
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "eval_time": eval_time,
            "similarity_penalty": similarity_penalty,
            "curriculum_total": total,
            "curriculum_correct": correct,
        }}
{error_artifact_block}

        return EvaluationResult(
            metrics={{
                "accuracy": float(accuracy),
                "correct": correct,
                "total": total,
                "eval_time": eval_time,
                "similarity_penalty": float(similarity_penalty),
                "combined_score": float(combined_score),
            }},
            artifacts=artifacts,
        )
    except Exception as e:
        return EvaluationResult(
            metrics={{
                "accuracy": 0.0,
                "correct": 0,
                "total": len(train_cases),
                "similarity_penalty": 1.0,
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


def create_training_evaluator(
    task_dir: Path,
    output_name: str = "evaluator_train_runtime.py",
    include_error_feedback: bool = False,
    config_path: Path | None = None,
) -> Path:
    if config_path is None:
        config_path = task_dir / "config.yaml"
    initial_program_path = task_dir / "initial_program.py"
    if not config_path.exists() or not initial_program_path.exists():
        raise FileNotFoundError(f"Task missing config/initial program under {task_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    system_message = config.get("prompt", {}).get("system_message", "")
    train_examples = parse_training_examples_from_system_message(system_message)
    if not train_examples:
        raise ValueError(f"No training examples parsed from {config_path}")

    function_name = load_function_name(initial_program_path)
    output_path = task_dir / output_name
    output_path.write_text(
        _render_training_evaluator(
            function_name=function_name,
            train_cases=train_examples,
            include_error_feedback=include_error_feedback,
        ),
        encoding="utf-8",
    )
    return output_path


def normalize_evaluator_result(raw_result: Any) -> Dict[str, Any]:
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


def evaluate_program_with_evaluator(evaluator_path: Path, program_path: Path) -> Dict[str, Any]:
    # Ensure evaluator imports can resolve local dio_agent source tree.
    local_dio_agent_root = (Path(__file__).resolve().parent.parent / "DIO-Agent").resolve()
    added_path = False
    if str(local_dio_agent_root) not in sys.path:
        sys.path.insert(0, str(local_dio_agent_root))
        added_path = True

    spec = importlib.util.spec_from_file_location("eval_module", str(evaluator_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
        raw_result = module.evaluate(str(program_path))
        return normalize_evaluator_result(raw_result)
    finally:
        if added_path:
            try:
                sys.path.remove(str(local_dio_agent_root))
            except ValueError:
                pass
