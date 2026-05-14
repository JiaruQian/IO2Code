"""
简化版批量 IO2Code 测试脚本（支持任务间并行）
"""

import os
import sys
import json
import time
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from queue import Empty, Queue
from training_evaluator_utils import (
    create_training_evaluator,
    evaluate_program_with_evaluator,
    prepare_runtime_config,
)
from adapter import get_all_task_names

# 添加路径
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
local_dio_agent_root = (script_dir.parent / "DIO-Agent").resolve()
OUTPUT_DIR_NAME = "dio_agent_base_default"

def _build_subprocess_env():
    """
    Ensure subprocesses prefer the local DIO-Agent source tree over installed packages.
    """
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    local_path = str(local_dio_agent_root)
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{local_path}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = local_path
    return env


def _terminate_process(process: subprocess.Popen):
    """
    Terminate a subprocess and its descendants (best effort), then force kill if needed.
    """
    if process.poll() is not None:
        return

    try:
        if os.name != "nt":
            # start_new_session=True makes child the leader of a new process group.
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

# 62个任务列表
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
    "Sum_All", "Sum_Last2", "Sum_Last3", "Sum_Last4", "Sum_Last5", "Sum_Last6", "Sum_Last7"
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


def _read_iterations_executed(output_dir: Path) -> int:
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

def run_single_task(
    task_name,
    iterations=5,
    timeout=180,
    verbose_live=False,
    include_error_feedback=False,
    with_dio_agent=False,
    output_subdir=OUTPUT_DIR_NAME,
    api_base=None,
    api_key_env=None,
    primary_model=None,
    secondary_model=None,
):
    """运行单个任务"""
    # 使用当前 Python 环境（可以是激活的虚拟环境或系统 Python）
    python_cmd = sys.executable
    
    task_dir = (script_dir / "tasks" / task_name).resolve()
    train_evaluator_path = task_dir / "evaluator_train_runtime.py"
    holdout_evaluator_path = task_dir / "evaluator.py"
    runtime_config_path = task_dir / f".config_train_runtime_{os.getpid()}.yaml"
    
    # 清理旧输出
    output_dir = task_dir / output_subdir
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    cmd = [
        str(python_cmd),
        "-m", "dio_agent.cli",
        str(task_dir / "initial_program.py"),
        str(train_evaluator_path),
        "--iterations", str(iterations),
        "--output", str(output_dir),
    ]
    if api_base:
        cmd.extend(["--api-base", str(api_base)])
    if api_key_env:
        cmd.extend(["--api-key-env", str(api_key_env)])
    if primary_model:
        cmd.extend(["--primary-model", str(primary_model)])
    if secondary_model:
        cmd.extend(["--secondary-model", str(secondary_model)])
    
    start = time.time()
    try:
        config_path_for_run = prepare_runtime_config(
            task_dir,
            output_name=runtime_config_path.name,
            with_dio_agent=with_dio_agent,
        )
        cmd.extend(["--config", str(config_path_for_run)])

        # Evolution must use training examples only (no holdout leakage during search).
        create_training_evaluator(
            task_dir,
            output_name="evaluator_train_runtime.py",
            include_error_feedback=include_error_feedback,
            config_path=config_path_for_run,
        )

        if verbose_live:
            line_queue: Queue[str] = Queue()

            def _reader_thread(pipe):
                for line in iter(pipe.readline, ""):
                    line_queue.put(line)
                pipe.close()

            # Stream child logs in real-time so long-running tasks are observable.
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
            reader = threading.Thread(
                target=_reader_thread, args=(process.stdout,), daemon=True
            )
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

            # Drain remaining buffered lines after process exits/is terminated.
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
                    "train_accuracy": 0.0,
                    "test_accuracy": 0.0,
                    "success": False,
                    "gen": -1,
                    "time": elapsed,
                    "error": "timeout",
                    "stdout": "".join(output_lines),
                    "stderr": "",
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
        
        # 读取结果
        best_info = task_dir / output_subdir / "best" / "best_program_info.json"
        best_program = task_dir / output_subdir / "best" / "best_program.py"
        if best_info.exists():
            with open(best_info) as f:
                info = json.load(f)
            train_accuracy = info.get("metrics", {}).get("accuracy", 0.0)
            token_usage = info.get("token_usage", {})
            prompt_tokens = (
                int(token_usage.get("prompt_tokens", 0))
                if isinstance(token_usage.get("prompt_tokens", 0), (int, float))
                else 0
            )
            completion_tokens = (
                int(token_usage.get("completion_tokens", 0))
                if isinstance(token_usage.get("completion_tokens", 0), (int, float))
                else 0
            )
            total_tokens = (
                int(token_usage.get("total_tokens", 0))
                if isinstance(token_usage.get("total_tokens", 0), (int, float))
                else 0
            )
            iterations_executed = _read_iterations_executed(output_dir)
            avg_prompt_tokens_per_iteration = (
                prompt_tokens / iterations_executed if iterations_executed > 0 else 0.0
            )
            avg_completion_tokens_per_iteration = (
                completion_tokens / iterations_executed if iterations_executed > 0 else 0.0
            )
            avg_total_tokens_per_iteration = (
                total_tokens / iterations_executed if iterations_executed > 0 else 0.0
            )
            test_accuracy = train_accuracy
            if best_program.exists():
                holdout_eval = evaluate_program_with_evaluator(holdout_evaluator_path, best_program)
                test_accuracy = holdout_eval.get("metrics", {}).get("accuracy", train_accuracy)
            gen = info.get("generation", -1)
            return {
                "accuracy": test_accuracy,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "iterations_executed": iterations_executed,
                "avg_prompt_tokens_per_iteration": avg_prompt_tokens_per_iteration,
                "avg_completion_tokens_per_iteration": avg_completion_tokens_per_iteration,
                "avg_total_tokens_per_iteration": avg_total_tokens_per_iteration,
                "success": test_accuracy >= 1.0,
                "gen": gen,
                "time": elapsed,
            }
        
        # 没有输出文件，返回更详细的错误信息
        error_msg = "no_output"
        if result.returncode != 0:
            error_msg = f"exit_code={result.returncode}"
        if result.stderr:
            # 只取前200个字符的stderr
            stderr_preview = result.stderr.strip()[:200]
            error_msg = f"{error_msg}|{stderr_preview}"
        return {
            "accuracy": 0.0,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "iterations_executed": 0,
            "avg_prompt_tokens_per_iteration": 0.0,
            "avg_completion_tokens_per_iteration": 0.0,
            "avg_total_tokens_per_iteration": 0.0,
            "success": False,
            "gen": -1,
            "time": elapsed,
            "error": error_msg,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "accuracy": 0.0,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "iterations_executed": 0,
            "avg_prompt_tokens_per_iteration": 0.0,
            "avg_completion_tokens_per_iteration": 0.0,
            "avg_total_tokens_per_iteration": 0.0,
            "success": False,
            "gen": -1,
            "time": timeout,
            "error": "timeout",
        }
    except Exception as e:
        return {
            "accuracy": 0.0,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "iterations_executed": 0,
            "avg_prompt_tokens_per_iteration": 0.0,
            "avg_completion_tokens_per_iteration": 0.0,
            "avg_total_tokens_per_iteration": 0.0,
            "success": False,
            "gen": -1,
            "time": 0,
            "error": str(e),
        }
    finally:
        if train_evaluator_path.exists():
            try:
                train_evaluator_path.unlink()
            except OSError:
                pass
        if runtime_config_path.exists():
            try:
                runtime_config_path.unlink()
            except OSError:
                pass

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=102)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=6000)
    parser.add_argument("--parallel", type=int, default=2,
                        help="同时运行的任务数量（任务间并行）")
    parser.add_argument(
        "--verbose-live",
        action="store_true",
        help="实时打印每个任务的完整日志（并行时输出会交错）",
    )
    parser.add_argument(
        "--include-error-feedback",
        action="store_true",
        help="将训练评估器中的失败样例详情反馈到后续迭代 prompt（默认关闭）",
    )
    parser.add_argument(
        "--with_dio_agent",
        action="store_true",
        help="在 base prompt 中附加 incremental refinement 描述，但不启用 stage-wise curriculum",
    )
    parser.add_argument(
        "--output-subdir",
        default=OUTPUT_DIR_NAME,
        help="Output folder name under each task",
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
    args = parser.parse_args()

    all_tasks = _build_task_list()
    start_idx = max(1, args.start)
    end_idx = min(len(all_tasks), args.end)
    if start_idx > end_idx:
        raise ValueError(f"Invalid range: start={args.start}, end={args.end}")
    tasks_to_run = all_tasks[start_idx - 1:end_idx]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"results_{timestamp}.csv"
    csv_lock = threading.Lock()
    
    print(f"=" * 60, flush=True)
    print("DIO-Agent IO2Code Benchmark", flush=True)
    print(f"Tasks: {args.start} to {args.end} ({len(tasks_to_run)} tasks)", flush=True)
    print(f"Iterations: {args.iterations}, Timeout: {args.timeout}s", flush=True)
    print(f"Task parallelism: {args.parallel}", flush=True)
    print(f"Verbose live logs: {args.verbose_live}", flush=True)
    print(f"Include error feedback in prompt: {args.include_error_feedback}", flush=True)
    print(f"Include DIO-Agent guidance in prompt: {args.with_dio_agent}", flush=True)
    print(f"Output subdir: {args.output_subdir}", flush=True)
    print(f"Results: {csv_file}", flush=True)
    print(f"=" * 60, flush=True)
    
    with open(csv_file, "w") as f:
        f.write(
            "task,accuracy,train_accuracy,test_accuracy,prompt_tokens,completion_tokens,total_tokens,iterations_executed,avg_prompt_tokens_per_iteration,avg_completion_tokens_per_iteration,avg_total_tokens_per_iteration,success,gen,time,error\n"
        )
    
    results = []
    success_count = 0
    completed_count = 0
    prompt_tokens_all_tasks = 0
    completion_tokens_all_tasks = 0
    total_tokens_all_tasks = 0
    iterations_executed_all_tasks = 0
    total = len(tasks_to_run)
    
    if args.parallel > 1:
        # 并行执行多个任务
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            future_to_task = {}
            for i, task in enumerate(tasks_to_run, start=args.start):
                future = executor.submit(
                    run_single_task,
                    task,
                    args.iterations,
                    args.timeout,
                    args.verbose_live,
                    args.include_error_feedback,
                    args.with_dio_agent,
                    args.output_subdir,
                    args.api_base,
                    args.api_key_env,
                    args.primary_model,
                    args.secondary_model,
                )
                future_to_task[future] = (i, task)
            
            for future in as_completed(future_to_task):
                i, task = future_to_task[future]
                result = future.result()
                completed_count += 1
                task_prompt_tokens = int(result.get("prompt_tokens", 0) or 0)
                task_completion_tokens = int(result.get("completion_tokens", 0) or 0)
                task_total_tokens = int(result.get("total_tokens", 0) or 0)
                task_iterations_executed = int(result.get("iterations_executed", 0) or 0)
                task_avg_prompt_per_iter = float(result.get("avg_prompt_tokens_per_iteration", 0.0) or 0.0)
                task_avg_completion_per_iter = float(result.get("avg_completion_tokens_per_iteration", 0.0) or 0.0)
                task_avg_total_per_iter = float(result.get("avg_total_tokens_per_iteration", 0.0) or 0.0)
                prompt_tokens_all_tasks += task_prompt_tokens
                completion_tokens_all_tasks += task_completion_tokens
                total_tokens_all_tasks += task_total_tokens
                iterations_executed_all_tasks += task_iterations_executed
                
                if result["success"]:
                    success_count += 1
                    print(
                        f"  [{completed_count}/{total}] ✅ {task}: "
                        f"train={result.get('train_accuracy', 0.0):.2f} "
                        f"test={result.get('test_accuracy', result.get('accuracy', 0.0)):.2f} "
                        f"p={task_prompt_tokens} c={task_completion_tokens} total={task_total_tokens} "
                        f"iters={task_iterations_executed} avg_p/i={task_avg_prompt_per_iter:.1f} avg_c/i={task_avg_completion_per_iter:.1f} "
                        f"gen={result['gen']} time={result['time']:.0f}s",
                        flush=True,
                    )
                else:
                    error_str = result.get('error', 'none')
                    print(
                        f"  [{completed_count}/{total}] ❌ {task}: "
                        f"train={result.get('train_accuracy', 0.0):.2f} "
                        f"test={result.get('test_accuracy', result.get('accuracy', 0.0)):.2f} "
                        f"p={task_prompt_tokens} c={task_completion_tokens} total={task_total_tokens} "
                        f"iters={task_iterations_executed} avg_p/i={task_avg_prompt_per_iter:.1f} avg_c/i={task_avg_completion_per_iter:.1f} "
                        f"error={error_str} time={result['time']:.0f}s",
                        flush=True,
                    )
                    # 如果有stderr，打印出来用于调试
                    if result.get('stderr'):
                        print(f"    stderr: {result['stderr'][:500]}", flush=True)
                
                with csv_lock:
                    with open(csv_file, "a") as f:
                        f.write(
                            f"{task},{result.get('accuracy', 0.0)},{result.get('train_accuracy', 0.0)},"
                            f"{result.get('test_accuracy', 0.0)},"
                            f"{task_prompt_tokens},{task_completion_tokens},{task_total_tokens},"
                            f"{task_iterations_executed},{task_avg_prompt_per_iter},{task_avg_completion_per_iter},{task_avg_total_per_iter},"
                            f"{result['success']},{result['gen']},"
                            f"{result['time']:.0f},{result.get('error', '')}\n"
                        )
                
                results.append({"task": task, **result})
    else:
        # 串行执行（原始行为）
        for i, task in enumerate(tasks_to_run, start=args.start):
            print(f"\n[{i}/{args.end}] {task}...", flush=True)
            result = run_single_task(
                task,
                args.iterations,
                args.timeout,
                args.verbose_live,
                args.include_error_feedback,
                args.with_dio_agent,
                args.output_subdir,
                args.api_base,
                args.api_key_env,
                args.primary_model,
                args.secondary_model,
            )
            completed_count += 1
            task_prompt_tokens = int(result.get("prompt_tokens", 0) or 0)
            task_completion_tokens = int(result.get("completion_tokens", 0) or 0)
            task_total_tokens = int(result.get("total_tokens", 0) or 0)
            task_iterations_executed = int(result.get("iterations_executed", 0) or 0)
            task_avg_prompt_per_iter = float(result.get("avg_prompt_tokens_per_iteration", 0.0) or 0.0)
            task_avg_completion_per_iter = float(result.get("avg_completion_tokens_per_iteration", 0.0) or 0.0)
            task_avg_total_per_iter = float(result.get("avg_total_tokens_per_iteration", 0.0) or 0.0)
            prompt_tokens_all_tasks += task_prompt_tokens
            completion_tokens_all_tasks += task_completion_tokens
            total_tokens_all_tasks += task_total_tokens
            iterations_executed_all_tasks += task_iterations_executed
            
            if result["success"]:
                success_count += 1
                print(
                    f"  ✅ train={result.get('train_accuracy', 0.0):.2f} "
                    f"test={result.get('test_accuracy', result.get('accuracy', 0.0)):.2f} "
                    f"p={task_prompt_tokens} c={task_completion_tokens} total={task_total_tokens} "
                    f"iters={task_iterations_executed} avg_p/i={task_avg_prompt_per_iter:.1f} avg_c/i={task_avg_completion_per_iter:.1f} "
                    f"gen={result['gen']} time={result['time']:.0f}s",
                    flush=True,
                )
            else:
                error_str = result.get('error', 'none')
                print(
                    f"  ❌ train={result.get('train_accuracy', 0.0):.2f} "
                    f"test={result.get('test_accuracy', result.get('accuracy', 0.0)):.2f} "
                    f"p={task_prompt_tokens} c={task_completion_tokens} total={task_total_tokens} "
                    f"iters={task_iterations_executed} avg_p/i={task_avg_prompt_per_iter:.1f} avg_c/i={task_avg_completion_per_iter:.1f} "
                    f"error={error_str}",
                    flush=True,
                )
                # 如果有stderr，打印出来用于调试
                if result.get('stderr'):
                    print(f"    stderr: {result['stderr'][:500]}", flush=True)
            
            with open(csv_file, "a") as f:
                f.write(
                    f"{task},{result.get('accuracy', 0.0)},{result.get('train_accuracy', 0.0)},"
                    f"{result.get('test_accuracy', 0.0)},"
                    f"{task_prompt_tokens},{task_completion_tokens},{task_total_tokens},"
                    f"{task_iterations_executed},{task_avg_prompt_per_iter},{task_avg_completion_per_iter},{task_avg_total_per_iter},"
                    f"{result['success']},{result['gen']},"
                    f"{result['time']:.0f},{result.get('error', '')}\n"
                )
            
            results.append({"task": task, **result})
    
    print(f"\n" + "=" * 60, flush=True)
    print(f"✅ Success: {success_count}/{len(tasks_to_run)} ({100*success_count/len(tasks_to_run):.1f}%)", flush=True)
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
    total_label = "TOTAL_62_TASKS" if len(tasks_to_run) == 62 else "TOTAL_TASKS_IN_RUN"
    with open(csv_file, "a") as f:
        f.write(
            f"{total_label},,,,{prompt_tokens_all_tasks},{completion_tokens_all_tasks},{total_tokens_all_tasks},"
            f"{iterations_executed_all_tasks},{avg_prompt_per_iter_all},{avg_completion_per_iter_all},{avg_total_per_iter_all},,,,\n"
        )
    
    # 保存JSON结果
    with open(f"results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
