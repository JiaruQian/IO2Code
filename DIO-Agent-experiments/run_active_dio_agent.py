"""
Active oracle-example DIO-Agent runner.

The model sees only generated I/O examples. Ground-truth task code is hidden
behind this runner's oracle: the LLM proposes new inputs, then the runner calls
the task implementation to obtain outputs.
"""

from __future__ import annotations

import argparse
import builtins
import concurrent.futures
import copy
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DIO_AGENT_ROOT = (SCRIPT_DIR.parent / "DIO-Agent").resolve()
if str(LOCAL_DIO_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIO_AGENT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adapter import TaskAdapter  # noqa: E402
from run_dio_agent import (  # noqa: E402
    _build_seed_checkpoint_for_next_stage,
    _compare_outputs,
    _dump_yaml,
    _evaluate_cases,
    _evaluate_with_full_evaluator,
    _load_function_name,
    _load_yaml,
    _render_stage_evaluator,
    _run_subprocess,
)


OUTPUT_DIR = "active_dio_agent_oracle"


def _example_key(inp: Any) -> str:
    return repr(inp)


def _extract_python_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _call_openai_compatible(
    *,
    api_base: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    timeout: int,
    temperature: float,
) -> str:
    url = api_base.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"

    def _post() -> requests.Response:
        return requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": 4096,
            },
            timeout=timeout,
        )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_post)
    try:
        response = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"Example-generation LLM request exceeded {timeout}s wall-clock timeout") from exc
    finally:
        if future.done():
            executor.shutdown(wait=False, cancel_futures=True)
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def _oracle_output(adapter: TaskAdapter, inp: Any) -> Any:
    if adapter.is_extra:
        return adapter.task_info["func"](inp)
    return adapter.func(inp)


def _oracle_input_constraints(adapter: TaskAdapter) -> str:
    if adapter.is_extra:
        task_type = adapter.task_info.get("type", "unknown")
        value_range = adapter.task_info.get("range")
        range_text = f", value range: {value_range}" if value_range is not None else ""
        return f"Task-specific input type: {task_type}{range_text}. Generate valid inputs for this type."

    dtype = getattr(adapter, "dtype", None)
    bounds = getattr(adapter, "bounds", None)
    if dtype == "2d_int":
        lo, hi = bounds
        return (
            "Valid input is a list containing exactly two integer lists: [xs, ys]. "
            "Both inner lists should have the same length. "
        )
    if dtype == "2d_bit":
        return (
            "Valid input is a list containing exactly two bit lists: [xs, ys]. "
            "Both inner lists should have the same length."
        )
    if dtype == "int":
        lo, hi = bounds
        return f"Valid input is a list of integers."
        # return f"Valid input is a list of integers, preferably length 3 to 10, with each element in [{lo}, {hi}]."
    if dtype == "bit":
        return "Valid input is a list of bits."
        # return "Valid input is a list of bits, preferably length 3 to 10, with each element 0 or 1."
    if dtype == "float":
        lo, hi = bounds
        return f"Valid input is a list of floats."
        # return f"Valid input is a list of floats, preferably length 3 to 10, with each element in [{lo}, {hi}]."
    return "Generate valid inputs matching the stated input type. Avoid malformed or empty structures unless clearly valid."


def _execute_llm_query_code(
    code: str,
    *,
    adapter: TaskAdapter,
    batch_size: int,
    seen: set[str],
) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
    namespace: dict[str, Any] = {"__builtins__": builtins.__dict__}
    exec(compile(code, "<llm_example_query>", "exec"), namespace, namespace)
    generator = namespace.get("generate_examples")
    if not callable(generator):
        raise ValueError("LLM query code must define generate_examples(oracle, n, seen_inputs)")

    oracle_calls: list[tuple[Any, Any]] = []

    def oracle(inp: Any) -> Any:
        out = _oracle_output(adapter, inp)
        oracle_calls.append((inp, out))
        return out

    raw_examples = generator(oracle, batch_size, set(seen))
    if raw_examples is None:
        raw_examples = oracle_calls
    if not isinstance(raw_examples, list):
        raise ValueError("generate_examples must return a list")
    if len(raw_examples) != batch_size:
        raise ValueError(
            f"generate_examples returned {len(raw_examples)} examples, but this round requires exactly {batch_size}"
        )
    if len(oracle_calls) != batch_size:
        raise ValueError(
            f"generate_examples called oracle {len(oracle_calls)} times, but this round allows exactly {batch_size}"
        )

    accepted: list[tuple[Any, Any]] = []
    rejected: list[dict[str, str]] = []
    duplicates_ignored: list[dict[str, str]] = []
    for item in raw_examples:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            rejected.append({"item": repr(item), "error": "expected (input, output)"})
            continue
        inp, claimed_out = item
        key = _example_key(inp)
        if key in seen:
            duplicates_ignored.append({"input": repr(inp), "reason": "duplicate input"})
            continue
        actual_out = _oracle_output(adapter, inp)
        if not _compare_outputs(claimed_out, actual_out):
            rejected.append(
                {
                    "input": repr(inp),
                    "claimed_output": repr(claimed_out),
                    "oracle_output": repr(actual_out),
                    "error": "output did not match oracle",
                }
            )
            continue
        seen.add(key)
        accepted.append((inp, actual_out))
        if len(accepted) >= batch_size:
            break

    return accepted, {
        "accepted": len(accepted),
        "rejected": rejected,
        "duplicates_ignored": duplicates_ignored,
        "oracle_calls": len(oracle_calls),
    }


def _write_query_attempt_log(
    *,
    query_log_dir: Path,
    event_name: str,
    attempt: int,
    raw_response: str | None = None,
    query_code: str | None = None,
    metadata: dict[str, Any],
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> str:
    attempt_dir = query_log_dir / event_name / f"attempt_{attempt:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    if raw_response is not None:
        (attempt_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
    if query_code is not None:
        (attempt_dir / "query_code.py").write_text(query_code, encoding="utf-8")
    if system_prompt is not None:
        (attempt_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    if user_prompt is not None:
        (attempt_dir / "user_prompt.txt").write_text(user_prompt, encoding="utf-8")
    metadata_path = attempt_dir / "metadata.json"
    metadata_payload = {"event": event_name, "attempt": attempt, **metadata}
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(attempt_dir)


def _generate_examples_with_llm_code(
    *,
    adapter: TaskAdapter,
    visible_cases: Sequence[tuple[Any, Any]],
    batch_size: int,
    seen: set[str],
    api_base: str | None,
    api_key: str | None,
    model: str | None,
    timeout: int,
    temperature: float,
    query_log_dir: Path,
    event_name: str,
    attempt: int,
) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
    if not api_base or not api_key or not model:
        raise ValueError("api_base, api_key, and model are required because every example must be LLM-constructed")

    examples_preview = "\n".join(
        f"{idx}. input={repr(inp)} -> output={repr(out)}"
        for idx, (inp, out) in enumerate(visible_cases[-12:], start=1)
    )
    system = (
        "You design black-box oracle queries for a program-by-example task. "
        "You cannot see the hidden implementation, but your generated Python function "
        "may call oracle(input_value) to obtain outputs. Return Python code only. "
        "This is an active curriculum: if the candidate program passes the current visible examples, "
        "you will later get another chance to generate a larger batch of new examples."
    )
#     user = f"""
# Task name: {adapter.task_name}
# Task description: {adapter.get_task_description()}
# Input type: {adapter.get_input_type()}
# Output type: {adapter.get_output_type()}
# This round's exact example budget: {batch_size}
    user = f"""
Input type: {adapter.get_input_type()}
Valid input constraints:
{_oracle_input_constraints(adapter)}
This round's exact example budget: {batch_size}

Curriculum rule:
- This runner starts with 2 examples.
- Once the evolving program passes all currently visible examples, you will be asked for a larger new batch: 4, then 6, then 8, then 10, and so on.
- Therefore, do not spend future budget now. Use exactly this round's budget and make these {batch_size} examples maximally informative.

Known I/O examples (latest 12 shown):
{examples_preview if examples_preview else "(none)"}

Write Python code defining exactly:

    def generate_examples(oracle, n, seen_inputs):
    ...

Requirements:
- Return a list of exactly n new (input, output) pairs. In this call, n == {batch_size}.
- Do not return fewer than n examples.
- Do not return more than n examples.
- Call oracle exactly n times total.
- Every input must be chosen by your code.
- Obtain each output by calling oracle(input_value); do not guess outputs.
- You may import Python modules if useful.
- seen_inputs is a set of repr(input) strings from all previously accepted examples.
- Avoid exact duplicates by checking repr(input_value) in seen_inputs.
- Prefer diverse, informative inputs that help infer the hidden rule.
- Return only the Python code block.
""".strip()

    raw = None
    code = None
    _write_query_attempt_log(
        query_log_dir=query_log_dir,
        event_name=event_name,
        attempt=attempt,
        system_prompt=system,
        user_prompt=user,
        metadata={
            "status": "started",
            "requested": batch_size,
            "model": model,
            "timeout_sec": timeout,
        },
    )
    print(
        f"[EXAMPLE-GEN] {adapter.task_name} {event_name} attempt {attempt}: "
        f"requesting {batch_size} examples (timeout={timeout}s)",
        flush=True,
    )
    try:
        raw = _call_openai_compatible(
            api_base=api_base,
            api_key=api_key,
            model=model,
            system=system,
            user=user,
            timeout=timeout,
            temperature=temperature,
        )
        code = _extract_python_code(raw)
        print(
            f"[EXAMPLE-GEN] {adapter.task_name} {event_name} attempt {attempt}: "
            f"LLM response received ({len(raw)} chars)",
            flush=True,
        )
        examples, exec_log = _execute_llm_query_code(
            code,
            adapter=adapter,
            batch_size=batch_size,
            seen=seen,
        )
        log_path = _write_query_attempt_log(
            query_log_dir=query_log_dir,
            event_name=event_name,
            attempt=attempt,
            raw_response=raw,
            query_code=code,
            system_prompt=system,
            user_prompt=user,
            metadata={
                "status": "success",
                "requested": batch_size,
                "accepted_examples": len(examples),
                **exec_log,
            },
        )
        return examples, {
            "source": "llm_query_code",
            "raw_response": raw,
            "query_code": code,
            "query_log_path": log_path,
            **exec_log,
        }
    except Exception as exc:
        log_path = _write_query_attempt_log(
            query_log_dir=query_log_dir,
            event_name=event_name,
            attempt=attempt,
            raw_response=raw,
            query_code=code,
            system_prompt=system,
            user_prompt=user,
            metadata={
                "status": "failed",
                "requested": batch_size,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        exc.args = (*exc.args, f"query_log_path={log_path}")
        raise


def _generate_oracle_examples(
    *,
    adapter: TaskAdapter,
    visible_cases: Sequence[tuple[Any, Any]],
    batch_size: int,
    seen: set[str],
    api_base: str | None,
    api_key: str | None,
    model: str | None,
    timeout: int,
    temperature: float,
    max_attempts: int,
    query_log_dir: Path,
    event_name: str,
) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        seen_snapshot = set(seen)
        try:
            examples, generation_log = _generate_examples_with_llm_code(
                adapter=adapter,
                visible_cases=visible_cases,
                batch_size=batch_size,
                seen=seen,
                api_base=api_base,
                api_key=api_key,
                model=model,
                timeout=timeout,
                temperature=temperature,
                query_log_dir=query_log_dir,
                event_name=event_name,
                attempt=attempt,
            )
            generation_log["attempt"] = attempt
            generation_log["requested"] = batch_size
            attempts.append(generation_log)
            if examples:
                return examples, {"requested": batch_size, "attempts": attempts}
        except Exception as exc:
            seen.clear()
            seen.update(seen_snapshot)
            attempts.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})

    raise RuntimeError(
        f"LLM failed to construct {batch_size} valid oracle examples after {max_attempts} attempts: "
        f"{json.dumps(attempts, ensure_ascii=False)[:2000]}"
    )


def _make_active_system_prompt(
    base_system_prompt: str,
    visible_cases: Sequence[tuple[Any, Any]],
    batch_size_next: int,
) -> str:
    lines = []
    for idx, (inp, out) in enumerate(visible_cases, start=1):
        lines.append(f"Example {idx}:\n  Input:  {repr(inp)}\n  Output: {repr(out)}")
    examples_text = "\n".join(lines)

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

## ACTIVE_DIO_AGENT_ORACLE_CONTEXT
- You only see the I/O examples above; the hidden oracle implementation is not shown.
- Pass every currently visible example while preserving earlier behavior.
- Prefer small, behavior-preserving transformations before adding broader control flow.
- Do NOT hardcode full concrete inputs or build lookup tables from examples.
- Infer a rule that should generalize to unseen final tests.
"""


def _latest_checkpoint(output_dir: Path) -> Path | None:
    root = output_dir / "checkpoints"
    if not root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint_"):
            continue
        try:
            candidates.append((int(child.name.split("_")[-1]), child))
        except ValueError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _evaluate_program_cases(program_path: Path, function_name: str, cases: Sequence[tuple[Any, Any]]) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(f"active_candidate_{uuid.uuid4().hex}", str(program_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, function_name):
        return {"accuracy": 0.0, "correct": 0, "total": len(cases), "errors": [{"error": "missing function"}]}
    acc, correct, total, errors = _evaluate_cases(getattr(module, function_name), cases)
    return {"accuracy": acc, "correct": correct, "total": total, "errors": errors}


def _configure_stage(
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
    prompt_cfg["num_top_programs"] = 2
    prompt_cfg["num_diverse_programs"] = 0
    prompt_cfg["num_inspirations"] = 1
    prompt_cfg["max_previous_attempts"] = 0
    prompt_cfg["system_message"] = _make_active_system_prompt(
        base_prompt, visible_cases, next_batch_size
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run active oracle-example DIO-Agent")
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
    parser.add_argument(
        "--example-max-attempts",
        type=int,
        default=3,
        help="Total attempts for example generation (default 3 = initial try + 2 retries)",
    )
    parser.add_argument(
        "--promotion-early-stop",
        type=int,
        default=5,
        help="Stop after this many consecutive successful promotions (default: 5)",
    )
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
        evaluator_path = iteration_dir / "evaluator_active_dio_agent.py"
        evaluator_path.write_text(
            _render_stage_evaluator(
                function_name=function_name,
                curriculum_cases=visible_cases,
                replay_cases=[],
                include_error_feedback=True,
            ),
            encoding="utf-8",
        )
        config = _configure_stage(
            base_config=base_config,
            base_prompt=base_prompt,
            visible_cases=visible_cases,
            next_batch_size=next_batch_size,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            primary_model=args.primary_model,
            secondary_model=args.secondary_model,
        )
        config_path = iteration_dir / "config_active_dio_agent.yaml"
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

        summary_path = active_root / "active_dio_agent_summary.json"
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
    summary_path = active_root / "active_dio_agent_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(f"\nActive DIO-Agent summary saved to: {summary_path}")
    if final_eval:
        metrics = final_eval.get("metrics", {})
        print("Final held-out evaluator metrics:")
        for key, value in metrics.items():
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
