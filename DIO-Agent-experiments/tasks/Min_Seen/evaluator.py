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
        ([36, 36, 34, 31], [36, 36, 34, 31]),
        ([8, 41, 20, 41, 50, 41, 6, 41, 12, 9], [8, 8, 8, 8, 8, 8, 6, 6, 6, 6]),
        ([36, 3, 44, 49, 16, 25, 50], [36, 3, 3, 3, 3, 3, 3]),
        ([40, 25, 4, 17, 1, 12], [40, 25, 4, 4, 1, 1]),
        ([11, 43, 21, 10, 5, 35], [11, 11, 11, 10, 5, 5]),
        ([44, 47, 23, 8, 42, 41, 31, 29, 0, 47], [44, 44, 23, 8, 8, 8, 8, 8, 0, 0]),
        ([15, 7, 46], [15, 7, 7]),
        ([25, 50, 17, 32, 22, 2], [25, 25, 17, 17, 17, 2]),
        ([29, 6, 27, 39, 35], [29, 6, 6, 6, 6]),
        ([13, 14, 6, 50, 3], [13, 13, 6, 6, 3]),
        ([1, 33, 40, 18, 9, 15], [1, 1, 1, 1, 1, 1]),
        ([37, 21, 29, 6, 2, 26, 33, 5, 49, 24], [37, 21, 21, 6, 2, 2, 2, 2, 2, 2]),
        ([44, 3, 17, 24], [44, 3, 3, 3]),
        ([49, 31, 3, 39, 49, 29, 2], [49, 31, 3, 3, 3, 3, 2]),
        ([29, 50, 6], [29, 29, 6]),
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
