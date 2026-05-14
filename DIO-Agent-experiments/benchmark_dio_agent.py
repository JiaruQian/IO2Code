"""
Batch runner for stage-wise DIO-Agent experiments.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

from adapter import get_all_task_names


script_dir = Path(__file__).parent
os.chdir(script_dir)
local_dio_agent_root = (script_dir.parent / "DIO-Agent").resolve()
OUTPUT_DIR = "dio_agent_final_default"

TASKS = [
    "Abs_Current", "Abs_Diff", "Add_Mod_3", "Add_Mod_4", "Add_Mod_5", "Add_Mod_6", "Add_Mod_7", "Add_Mod_8",
    "Alternating_Last3", "Alternating_Last4", "Balanced_Parenthesis",
    "Base_3_Addition", "Base_4_Addition", "Base_5_Addition", "Base_6_Addition", "Base_7_Addition", "Binary_Addition",
    "Bit_Dot_Prod_Mod2", "Bit_Palindrome", "Bit_Shift_Right",
    "Bitwise_And", "Bitwise_Not", "Bitwise_Or", "Bitwise_Xor",
    "Current_Number", "Diff_Abs_Values", "Diff_Last2", "Dithering",
    "Div_3", "Div_5", "Div_7",
    "Evens_Counter", "Evens_Detector",
    "Majority_0_1", "Majority_0_2", "Majority_0_3",
    "Max_Seen", "Min_Seen",
    "Newton_Freebody", "Newton_Gravity", "Newton_Magnetic", "Newton_Spring",
    "Parity_All", "Parity_Bits_Mod2", "Parity_Last2", "Parity_Last3", "Parity_Last4", "Parity_Zeros",
    "Perfect_Square_Detector",
    "Prev1", "Prev2", "Prev3", "Prev4", "Prev5",
    "Previous_Equals_Current",
    "Sum_All", "Sum_Last2", "Sum_Last3", "Sum_Last4", "Sum_Last5", "Sum_Last6", "Sum_Last7",
]


def _build_task_list():
    tasks = list(TASKS)
    all_tasks = get_all_task_names(include_extra=True)
    extra_tasks = sorted(
        [
            name
            for name in all_tasks
            if (name.startswith("Extra_") or name.startswith("LeetCode_") or name.startswith("Mutated_")) and name not in tasks
        ]
    )
    tasks.extend(extra_tasks)
    return tasks


def _build_subprocess_env() -> dict:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    local_path = str(local_dio_agent_root)
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{local_path}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = local_path
    return env


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass


def _extract_final_metrics(task_name: str, output_subdir: str) -> dict:
    summary_path = script_dir / "tasks" / task_name / output_subdir / "dio_agent_summary.json"
    if not summary_path.exists():
        return {}
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return {}
    metrics = summary.get("final_full_evaluation", {}).get("metrics", {})
    token_usage = summary.get("token_usage", {})
    token_usage_per_iteration = summary.get("token_usage_per_iteration", {})
    iterations_executed = summary.get("iterations_executed", 0)
    prompt_tokens = token_usage.get("prompt_tokens", 0) if isinstance(token_usage, dict) else 0
    completion_tokens = token_usage.get("completion_tokens", 0) if isinstance(token_usage, dict) else 0
    total_tokens = token_usage.get("total_tokens", 0) if isinstance(token_usage, dict) else 0
    avg_prompt_tokens = (
        token_usage_per_iteration.get("avg_prompt_tokens", 0.0)
        if isinstance(token_usage_per_iteration, dict)
        else 0.0
    )
    avg_completion_tokens = (
        token_usage_per_iteration.get("avg_completion_tokens", 0.0)
        if isinstance(token_usage_per_iteration, dict)
        else 0.0
    )
    avg_total_tokens = (
        token_usage_per_iteration.get("avg_total_tokens", 0.0)
        if isinstance(token_usage_per_iteration, dict)
        else 0.0
    )
    if isinstance(metrics, dict):
        return {
            "metrics": metrics,
            "prompt_tokens": int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else 0,
            "completion_tokens": int(completion_tokens) if isinstance(completion_tokens, (int, float)) else 0,
            "total_tokens": int(total_tokens) if isinstance(total_tokens, (int, float)) else 0,
            "iterations_executed": int(iterations_executed) if isinstance(iterations_executed, (int, float)) else 0,
            "avg_prompt_tokens_per_iteration": float(avg_prompt_tokens)
            if isinstance(avg_prompt_tokens, (int, float))
            else 0.0,
            "avg_completion_tokens_per_iteration": float(avg_completion_tokens)
            if isinstance(avg_completion_tokens, (int, float))
            else 0.0,
            "avg_total_tokens_per_iteration": float(avg_total_tokens)
            if isinstance(avg_total_tokens, (int, float))
            else 0.0,
        }
    return {
        "metrics": {},
        "prompt_tokens": int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else 0,
        "completion_tokens": int(completion_tokens) if isinstance(completion_tokens, (int, float)) else 0,
        "total_tokens": int(total_tokens) if isinstance(total_tokens, (int, float)) else 0,
        "iterations_executed": int(iterations_executed) if isinstance(iterations_executed, (int, float)) else 0,
        "avg_prompt_tokens_per_iteration": float(avg_prompt_tokens)
        if isinstance(avg_prompt_tokens, (int, float))
        else 0.0,
        "avg_completion_tokens_per_iteration": float(avg_completion_tokens)
        if isinstance(avg_completion_tokens, (int, float))
        else 0.0,
        "avg_total_tokens_per_iteration": float(avg_total_tokens)
        if isinstance(avg_total_tokens, (int, float))
        else 0.0,
    }


def run_single_task(
    task_name: str,
    timeout: int,
    stage_fractions: str,
    stage_iterations: str,
    replay_ratio: float,
    seed: int,
    interstage_init_mode: str,
    final_selection_mode: str,
    verbose_live: bool,
    include_error_feedback: bool,
    no_dio_agent: bool,
    output_subdir: str,
    api_base: str | None,
    api_key_env: str | None,
    primary_model: str | None,
    secondary_model: str | None,
) -> dict:
    python_cmd = sys.executable
    cmd = [
        str(python_cmd),
        "run_dio_agent.py",
        "--task",
        task_name,
        "--stage-fractions",
        stage_fractions,
        "--stage-iterations",
        stage_iterations,
        "--replay-ratio",
        str(replay_ratio),
        "--seed",
        str(seed),
        "--interstage-init-mode",
        interstage_init_mode,
        "--final-selection-mode",
        final_selection_mode,
        "--output-subdir",
        output_subdir,
    ]
    if api_base:
        cmd.extend(["--api-base", str(api_base)])
    if api_key_env:
        cmd.extend(["--api-key-env", str(api_key_env)])
    if primary_model:
        cmd.extend(["--primary-model", str(primary_model)])
    if secondary_model:
        cmd.extend(["--secondary-model", str(secondary_model)])
    if include_error_feedback:
        cmd.append("--include-error-feedback")
    if no_dio_agent:
        cmd.append("--no_dio_agent")

    start = time.time()
    try:
        if verbose_live:
            line_queue: Queue[str] = Queue()

            def _reader_thread(pipe):
                for line in iter(pipe.readline, ""):
                    line_queue.put(line)
                pipe.close()

            process = subprocess.Popen(
                cmd,
                cwd=str(script_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=_build_subprocess_env(),
                start_new_session=(os.name != "nt"),
            )
            output_lines = []
            assert process.stdout is not None
            reader = threading.Thread(target=_reader_thread, args=(process.stdout,), daemon=True)
            reader.start()

            timed_out = False
            deadline = start + timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    timed_out = True
                    _terminate_process(process)
                    break
                try:
                    line = line_queue.get(timeout=min(0.2, remaining))
                    output_lines.append(line)
                    print(f"[{task_name}] {line}", end="", flush=True)
                except Empty:
                    if process.poll() is not None:
                        break

            while True:
                try:
                    line = line_queue.get_nowait()
                    output_lines.append(line)
                    print(f"[{task_name}] {line}", end="", flush=True)
                except Empty:
                    break

            if timed_out:
                elapsed = time.time() - start
                return {
                    "accuracy": 0.0,
                    "success": False,
                    "time": elapsed,
                    "error": "timeout",
                    "stdout": "".join(output_lines),
                }

            class _Result:
                def __init__(self, returncode, stdout):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = ""

            result = _Result(process.returncode, "".join(output_lines))
        else:
            result = subprocess.run(
                cmd,
                cwd=str(script_dir),
                timeout=timeout,
                capture_output=True,
                text=True,
                env=_build_subprocess_env(),
            )

        elapsed = time.time() - start
        extracted = _extract_final_metrics(task_name, output_subdir)
        metrics = extracted.get("metrics", {}) if isinstance(extracted, dict) else {}
        prompt_tokens = extracted.get("prompt_tokens", 0) if isinstance(extracted, dict) else 0
        completion_tokens = extracted.get("completion_tokens", 0) if isinstance(extracted, dict) else 0
        total_tokens = extracted.get("total_tokens", 0) if isinstance(extracted, dict) else 0
        iterations_executed = extracted.get("iterations_executed", 0) if isinstance(extracted, dict) else 0
        avg_prompt_tokens_per_iteration = (
            extracted.get("avg_prompt_tokens_per_iteration", 0.0) if isinstance(extracted, dict) else 0.0
        )
        avg_completion_tokens_per_iteration = (
            extracted.get("avg_completion_tokens_per_iteration", 0.0) if isinstance(extracted, dict) else 0.0
        )
        avg_total_tokens_per_iteration = (
            extracted.get("avg_total_tokens_per_iteration", 0.0) if isinstance(extracted, dict) else 0.0
        )
        accuracy = float(metrics.get("accuracy", 0.0)) if isinstance(metrics.get("accuracy", 0.0), (int, float)) else 0.0
        combined_score = metrics.get("combined_score", "")
        success = accuracy >= 1.0

        if result.returncode == 0 and metrics:
            return {
                "accuracy": accuracy,
                "combined_score": combined_score,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(total_tokens),
                "iterations_executed": int(iterations_executed),
                "avg_prompt_tokens_per_iteration": float(avg_prompt_tokens_per_iteration),
                "avg_completion_tokens_per_iteration": float(avg_completion_tokens_per_iteration),
                "avg_total_tokens_per_iteration": float(avg_total_tokens_per_iteration),
                "success": success,
                "time": elapsed,
                "error": "",
            }

        error_msg = f"exit_code={result.returncode}" if result.returncode != 0 else "missing_dio_agent_summary"
        stderr_preview = (result.stderr or "").strip()[:300]
        if stderr_preview:
            error_msg = f"{error_msg}|{stderr_preview}"
        return {
            "accuracy": accuracy,
            "combined_score": combined_score,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
            "iterations_executed": int(iterations_executed),
            "avg_prompt_tokens_per_iteration": float(avg_prompt_tokens_per_iteration),
            "avg_completion_tokens_per_iteration": float(avg_completion_tokens_per_iteration),
            "avg_total_tokens_per_iteration": float(avg_total_tokens_per_iteration),
            "success": success,
            "time": elapsed,
            "error": error_msg,
            "stdout": getattr(result, "stdout", ""),
            "stderr": getattr(result, "stderr", ""),
        }
    except subprocess.TimeoutExpired:
        return {
            "accuracy": 0.0,
            "combined_score": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "iterations_executed": 0,
            "avg_prompt_tokens_per_iteration": 0.0,
            "avg_completion_tokens_per_iteration": 0.0,
            "avg_total_tokens_per_iteration": 0.0,
            "success": False,
            "time": timeout,
            "error": "timeout",
        }
    except Exception as e:
        return {
            "accuracy": 0.0,
            "combined_score": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "iterations_executed": 0,
            "avg_prompt_tokens_per_iteration": 0.0,
            "avg_completion_tokens_per_iteration": 0.0,
            "avg_total_tokens_per_iteration": 0.0,
            "success": False,
            "time": 0,
            "error": str(e),
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Batch DIO-Agent benchmark runner")
    parser.add_argument("--start", type=int, default=1, help="1-based task start index")
    parser.add_argument("--end", type=int, default=102, help="1-based task end index (inclusive)")
    parser.add_argument("--timeout", type=int, default=7200, help="Per-task timeout in seconds")
    parser.add_argument("--parallel", type=int, default=1, help="Task-level parallelism")
    parser.add_argument("--stage-fractions", default="0.2,0.4,0.7,1.0")
    parser.add_argument("--stage-iterations", default="8,8,10,12")
    parser.add_argument("--replay-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--interstage-init-mode",
        choices=["best_only", "one_random_island"],
        default="best_only",
        help="Stage间岛屿初始化方式（传给 run_dio_agent.py）",
    )
    parser.add_argument(
        "--final-selection-mode",
        choices=["stage4_best", "all_stage_candidates_training_reselect"],
        default="stage4_best",
        help="最终程序选择方式（传给 run_dio_agent.py）",
    )
    parser.add_argument("--verbose-live", action="store_true")
    parser.add_argument(
        "--include-error-feedback",
        action="store_true",
        help="将阶段评估器中的失败样例详情反馈到后续迭代 prompt（默认关闭）",
    )
    parser.add_argument(
        "--no_dio_agent",
        action="store_true",
        help="关闭 DIO-Agent prompt guidance，仅保留 stage-wise curriculum 进化",
    )
    parser.add_argument(
        "--output-subdir",
        default=OUTPUT_DIR,
        help="Output folder name under each task (default: dio_agent_final_default)",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override the LLM API base for run_dio_agent.py/dio_agent.cli",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable name containing the LLM API key",
    )
    parser.add_argument(
        "--primary-model",
        default=None,
        help="Override the primary LLM model name for run_dio_agent.py/dio_agent.cli",
    )
    parser.add_argument(
        "--secondary-model",
        default=None,
        help="Optional secondary LLM model name for run_dio_agent.py/dio_agent.cli",
    )
    args = parser.parse_args()

    all_tasks = _build_task_list()
    start_idx = max(1, args.start)
    end_idx = min(len(all_tasks), args.end)
    if start_idx > end_idx:
        raise ValueError(f"Invalid range: start={args.start}, end={args.end}")

    tasks_to_run = all_tasks[start_idx - 1 : end_idx]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"dio_agent_results_{timestamp}.csv"
    json_file = f"dio_agent_results_{timestamp}.json"
    csv_lock = threading.Lock()

    print("=" * 60, flush=True)
    print("DIO-Agent Benchmark", flush=True)
    print(f"Tasks: {start_idx} to {end_idx} ({len(tasks_to_run)} tasks)", flush=True)
    print(f"Timeout: {args.timeout}s, parallel: {args.parallel}", flush=True)
    print(f"Stage fractions: {args.stage_fractions}", flush=True)
    print(f"Stage iterations: {args.stage_iterations}", flush=True)
    print(f"Replay ratio: {args.replay_ratio}, seed: {args.seed}", flush=True)
    print(f"Interstage init mode: {args.interstage_init_mode}", flush=True)
    print(f"Final selection mode: {args.final_selection_mode}", flush=True)
    print(f"Verbose live logs: {args.verbose_live}", flush=True)
    print(f"Include error feedback in prompt: {args.include_error_feedback}", flush=True)
    print(f"Include DIO-Agent guidance in prompt: {not args.no_dio_agent}", flush=True)
    print(f"Output subdir: {args.output_subdir}", flush=True)
    print(f"Results: {csv_file}", flush=True)
    print("=" * 60, flush=True)

    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(
            "task,accuracy,combined_score,prompt_tokens,completion_tokens,total_tokens,iterations_executed,avg_prompt_tokens_per_iteration,avg_completion_tokens_per_iteration,avg_total_tokens_per_iteration,success,time,error\n"
        )

    results = []
    success_count = 0
    completed_count = 0
    prompt_tokens_all_tasks = 0
    completion_tokens_all_tasks = 0
    total_tokens_all_tasks = 0
    iterations_executed_all_tasks = 0
    total = len(tasks_to_run)

    def _record(task: str, result: dict) -> None:
        nonlocal success_count, completed_count, prompt_tokens_all_tasks, completion_tokens_all_tasks, total_tokens_all_tasks, iterations_executed_all_tasks
        completed_count += 1
        task_prompt_tokens = int(result.get("prompt_tokens", 0) or 0)
        task_completion_tokens = int(result.get("completion_tokens", 0) or 0)
        task_tokens = int(result.get("total_tokens", 0) or 0)
        task_iterations_executed = int(result.get("iterations_executed", 0) or 0)
        task_avg_prompt_per_iter = float(result.get("avg_prompt_tokens_per_iteration", 0.0) or 0.0)
        task_avg_completion_per_iter = float(result.get("avg_completion_tokens_per_iteration", 0.0) or 0.0)
        task_avg_total_per_iter = float(result.get("avg_total_tokens_per_iteration", 0.0) or 0.0)
        prompt_tokens_all_tasks += task_prompt_tokens
        completion_tokens_all_tasks += task_completion_tokens
        total_tokens_all_tasks += task_tokens
        iterations_executed_all_tasks += task_iterations_executed
        if result["success"]:
            success_count += 1
            print(
                f"  [{completed_count}/{total}] ✅ {task}: acc={result['accuracy']:.2f} score={result.get('combined_score', '')} p={task_prompt_tokens} c={task_completion_tokens} total={task_tokens} "
                f"iters={task_iterations_executed} avg_p/i={task_avg_prompt_per_iter:.1f} avg_c/i={task_avg_completion_per_iter:.1f} time={result['time']:.0f}s",
                flush=True,
            )
        else:
            print(
                f"  [{completed_count}/{total}] ❌ {task}: acc={result['accuracy']:.2f} p={task_prompt_tokens} c={task_completion_tokens} total={task_tokens} "
                f"iters={task_iterations_executed} avg_p/i={task_avg_prompt_per_iter:.1f} avg_c/i={task_avg_completion_per_iter:.1f} error={result.get('error', '')} time={result['time']:.0f}s",
                flush=True,
            )
            if result.get("stderr"):
                print(f"    stderr: {result['stderr'][:500]}", flush=True)

        with csv_lock:
            with open(csv_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{task},{result.get('accuracy', 0.0)},{result.get('combined_score', '')},"
                    f"{task_prompt_tokens},{task_completion_tokens},{task_tokens},"
                    f"{task_iterations_executed},{task_avg_prompt_per_iter},{task_avg_completion_per_iter},{task_avg_total_per_iter},"
                    f"{result.get('success', False)},{result.get('time', 0):.0f},{result.get('error', '')}\n"
                )
        results.append({"task": task, **result})

    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            future_to_task = {}
            for task in tasks_to_run:
                future = executor.submit(
                    run_single_task,
                    task,
                    args.timeout,
                    args.stage_fractions,
                    args.stage_iterations,
                    args.replay_ratio,
                    args.seed,
                    args.interstage_init_mode,
                    args.final_selection_mode,
                    args.verbose_live,
                    args.include_error_feedback,
                    args.no_dio_agent,
                    args.output_subdir,
                    args.api_base,
                    args.api_key_env,
                    args.primary_model,
                    args.secondary_model,
                )
                future_to_task[future] = task

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                result = future.result()
                _record(task, result)
    else:
        for task in tasks_to_run:
            result = run_single_task(
                task,
                args.timeout,
                args.stage_fractions,
                args.stage_iterations,
                args.replay_ratio,
                args.seed,
                args.interstage_init_mode,
                args.final_selection_mode,
                args.verbose_live,
                args.include_error_feedback,
                args.no_dio_agent,
                args.output_subdir,
                args.api_base,
                args.api_key_env,
                args.primary_model,
                args.secondary_model,
            )
            _record(task, result)

    success_rate = 100.0 * success_count / total if total else 0.0
    print("\n" + "=" * 60, flush=True)
    print(f"✅ Success: {success_count}/{total} ({success_rate:.1f}%)", flush=True)
    print(f"🔢 Prompt tokens (all tasks): {prompt_tokens_all_tasks}", flush=True)
    print(f"🔢 Completion tokens (all tasks): {completion_tokens_all_tasks}", flush=True)
    print(f"🔢 Total tokens (all tasks): {total_tokens_all_tasks}", flush=True)
    avg_prompt_per_iter_all = (
        prompt_tokens_all_tasks / iterations_executed_all_tasks if iterations_executed_all_tasks > 0 else 0.0
    )
    avg_completion_per_iter_all = (
        completion_tokens_all_tasks / iterations_executed_all_tasks if iterations_executed_all_tasks > 0 else 0.0
    )
    avg_total_per_iter_all = (
        total_tokens_all_tasks / iterations_executed_all_tasks if iterations_executed_all_tasks > 0 else 0.0
    )
    print(f"🔁 Iterations executed (all tasks): {iterations_executed_all_tasks}", flush=True)
    print(f"🔢 Avg prompt tokens / iteration: {avg_prompt_per_iter_all:.2f}", flush=True)
    print(f"🔢 Avg completion tokens / iteration: {avg_completion_per_iter_all:.2f}", flush=True)
    print(f"🔢 Avg total tokens / iteration: {avg_total_per_iter_all:.2f}", flush=True)
    print(f"Results saved to: {csv_file}", flush=True)

    total_label = "TOTAL_62_TASKS" if total == 62 else "TOTAL_TASKS_IN_RUN"
    with open(csv_file, "a", encoding="utf-8") as f:
        f.write(
            f"{total_label},,,{prompt_tokens_all_tasks},{completion_tokens_all_tasks},{total_tokens_all_tasks},"
            f"{iterations_executed_all_tasks},{avg_prompt_per_iter_all},{avg_completion_per_iter_all},{avg_total_per_iter_all},,,\n"
        )

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed JSON saved to: {json_file}", flush=True)


if __name__ == "__main__":
    main()
