"""
Auto-generated multimodal evaluator (image -> integer).
"""

import ast
import os
import importlib.util
import traceback
from dio_agent.evaluation_result import EvaluationResult

TEST_CASES = [{'id': 'image_009', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_009.png'}, 'expected_output': 8}, {'id': 'image_010', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_010.png'}, 'expected_output': 5}, {'id': 'image_011', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_011.png'}, 'expected_output': 7}, {'id': 'image_012', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_012.png'}, 'expected_output': 6}, {'id': 'image_013', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_013.png'}, 'expected_output': 3}, {'id': 'image_014', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_014.png'}, 'expected_output': 6}, {'id': 'image_015', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_015.png'}, 'expected_output': 6}, {'id': 'image_016', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_016.png'}, 'expected_output': 7}, {'id': 'image_017', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_017.png'}, 'expected_output': 5}, {'id': 'image_018', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_018.png'}, 'expected_output': 8}, {'id': 'image_019', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_019.png'}, 'expected_output': 7}, {'id': 'image_020', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_020.png'}, 'expected_output': 5}, {'id': 'image_021', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_021.png'}, 'expected_output': 6}, {'id': 'image_022', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_022.png'}, 'expected_output': 8}, {'id': 'image_023', 'input_data': {'image_path': 'multimodal/data/multimodal_task7/test/images/image_023.png'}, 'expected_output': 6}]


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
            metrics={
                "accuracy": 0.0,
                "acc_curriculum": 0.0,
                "complexity_penalty": 1.0,
                "similarity_penalty": 1.0,
                "combined_score": 0.0,
                "correct": 0,
                "tests_passed": 0,
                "total": int(len(TEST_CASES)),
                "pass_threshold": float(pass_threshold),
                "error": f"program_load_failed: {exc}",
            },
            artifacts={"error_type": type(exc).__name__, "traceback": traceback.format_exc()},
        )

    if not hasattr(module, "solve") or not callable(getattr(module, "solve")):
        return EvaluationResult(
            metrics={
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
            },
            artifacts={"error": "MissingSolve"},
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
                {
                    "id": case["id"],
                    "test_case": i,
                    "input": repr(case["input_data"]),
                    "expected": repr(expected),
                    "actual": repr(actual),
                    "correct": bool(is_correct),
                }
            )
            if (not is_correct) and False and len(curriculum_errors) < 3:
                curriculum_errors.append(
                    {
                        "test_case": i,
                        "input": repr(case["input_data"]),
                        "expected": repr(expected),
                        "actual": repr(actual),
                    }
                )
        except Exception as exc:
            details.append(
                {
                    "id": case["id"],
                    "test_case": i,
                    "input": repr(case["input_data"]),
                    "expected": repr(expected),
                    "error": str(exc),
                    "correct": False,
                }
            )
            if False and len(curriculum_errors) < 3:
                curriculum_errors.append(
                    {
                        "test_case": i,
                        "input": repr(case["input_data"]),
                        "expected": repr(expected),
                        "error": str(exc),
                    }
                )

    accuracy = float(success_count / len(TEST_CASES)) if TEST_CASES else 0.0
    try:
        with open(program_path, "r", encoding="utf-8") as f:
            program_source = f.read()
    except Exception:
        program_source = ""
    complexity_penalty = _complexity_penalty(program_source)
    similarity_penalty = _hardcode_similarity_penalty(program_source)
    if False:
        combined_score = accuracy - 0.1 * complexity_penalty - 0.1 * similarity_penalty
    else:
        combined_score = accuracy
    combined_score = max(0.0, combined_score)
    artifacts = {
        "pass_threshold": float(pass_threshold),
        "case_details": details,
        "curriculum_total": int(len(TEST_CASES)),
        "curriculum_correct": int(success_count),
    }
    if False and curriculum_errors:
        artifacts["curriculum_errors"] = curriculum_errors
    return EvaluationResult(
        metrics={
            "accuracy": accuracy,
            "acc_curriculum": accuracy,
            "complexity_penalty": float(complexity_penalty),
            "similarity_penalty": float(similarity_penalty),
            "combined_score": combined_score,
            "correct": int(success_count),
            "tests_passed": int(success_count),
            "total": int(len(TEST_CASES)),
            "pass_threshold": float(pass_threshold),
        },
        artifacts=artifacts,
    )
