"""
Evaluator 模板 - 用于评估 IO2Code 任务的程序
此文件将被自动生成，包含任务特定的测试样例

重要说明：
- 此文件包含 15 个测试样例（测试集）
- 这些样例使用随机种子 999 生成，与训练集（种子 42）完全独立
- LLM 在生成代码时看不到这些测试样例，用于评估真实泛化能力
"""

import importlib.util
import traceback
import time
from typing import List, Tuple, Any
from dio_agent.evaluation_result import EvaluationResult


def evaluate(program_path: str) -> dict:
    """
    评估程序在测试样例上的表现（独立测试集，未用于训练）
    
    Args:
        program_path: 程序文件路径
    
    Returns:
        包含评分指标的字典
    """
    # 测试集样例（15 个，随机种子 999）
    # 这些样例 LLM 未见过，用于评估泛化能力
    test_cases = [
        ([18, 18, 17, 15], [18, 36, 53, 50]),
        ([4, 20, 10, 20, 20, 3, 20, 6, 4, 8], [4, 24, 34, 50, 50, 43, 43, 29, 30, 18]),
        ([8, 12, 7], [8, 20, 27]),
        ([2, 8, 0, 6, 20, 16, 6, 5, 10], [2, 10, 10, 14, 26, 42, 42, 27, 21]),
        ([2, 17, 19, 14, 11], [2, 19, 38, 50, 44]),
        ([20, 15, 14, 0, 1], [20, 35, 49, 29, 15]),
        ([3, 20, 7, 12, 8, 16], [3, 23, 30, 39, 27, 36]),
        ([1, 5, 14, 3, 13, 19, 17, 5], [1, 6, 20, 22, 30, 35, 49, 41]),
        ([7, 3, 1, 6, 0, 16], [7, 10, 11, 10, 7, 22]),
        ([4, 7, 15, 18, 10, 14, 3], [4, 11, 26, 40, 43, 42, 27]),
        ([13, 16, 2], [13, 29, 31]),
        ([18, 3, 1, 8, 12, 16, 9, 15, 1], [18, 21, 22, 12, 21, 36, 37, 40, 25]),
        ([1, 1, 14, 3, 20, 3, 8, 2, 3, 17], [1, 2, 16, 18, 37, 26, 31, 13, 13, 22]),
        ([10, 4, 13, 12, 5, 14, 17, 1, 16, 9], [10, 14, 27, 29, 30, 31, 36, 32, 34, 26]),
        ([9, 14, 1, 4], [9, 23, 24, 19]),
    ]
    
    try:
        # 加载程序
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # 检查必需的函数
        if not hasattr(program, "process_sequence"):
            error_artifacts = {
                "error_type": "MissingFunction",
                "error_message": f"Program is missing required 'process_sequence' function",
            }
            return EvaluationResult(
                metrics={
                    "accuracy": 0.0,
                    "correct": 0,
                    "total": len(test_cases),
                    "combined_score": 0.0,
                    "error": f"Missing process_sequence function",
                },
                artifacts=error_artifacts
            )
        
        func = getattr(program, "process_sequence")
        
        # 在测试样例上运行
        correct = 0
        total = len(test_cases)
        errors = []
        
        start_time = time.time()
        
        for i, (input_data, expected_output) in enumerate(test_cases):
            try:
                # 运行函数
                actual_output = func(input_data)
                
                # 比较输出
                if _compare_outputs(actual_output, expected_output):
                    correct += 1
                else:
                    if len(errors) < 3:  # 只保留前3个错误
                        errors.append({
                            "test_case": i,
                            "input": _format_value(input_data),
                            "expected": _format_value(expected_output),
                            "actual": _format_value(actual_output)
                        })
            except Exception as e:
                if len(errors) < 3:
                    errors.append({
                        "test_case": i,
                        "input": _format_value(input_data),
                        "error": str(e)
                    })
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        # 计算指标
        accuracy = correct / total if total > 0 else 0.0
        
        # 构建 artifacts
        artifacts = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "eval_time": eval_time,
        }
        
        if errors:
            artifacts["errors"] = errors
        
        # 返回评估结果
        return EvaluationResult(
            metrics={
                "accuracy": float(accuracy),
                "correct": correct,
                "total": total,
                "eval_time": eval_time,
                "combined_score": float(accuracy),  # DIOAgent 使用此分数进行优化
            },
            artifacts=artifacts
        )
    
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        
        error_artifacts = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        
        return EvaluationResult(
            metrics={
                "accuracy": 0.0,
                "correct": 0,
                "total": len(test_cases) if 'test_cases' in locals() else 0,
                "combined_score": 0.0,
                "error": str(e),
            },
            artifacts=error_artifacts
        )


def _compare_outputs(actual, expected) -> bool:
    """比较实际输出和期望输出"""
    # 处理列表
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(_compare_outputs(a, e) for a, e in zip(actual, expected))
    
    # 处理数值（允许小误差）
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, int) and isinstance(actual, int):
            return actual == expected
        # 浮点数比较
        return abs(actual - expected) < 1e-6
    
    # 其他类型直接比较
    return actual == expected


def _format_value(value, max_len=50) -> str:
    """格式化值用于显示"""
    s = str(value)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s
