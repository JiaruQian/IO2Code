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
        ([0.8964540278352238, 0.5677871374320627, 0.5337014607381507, 0.4833852211540276], [1, 0, 1, 0]),
        ([0.6437106936760943, 0.6425809178139255, 0.819230460356274, 0.6455459817825003, 0.8375671296417935, 0.14915934402185516, 0.9747131881932831, 0.05314555542247057], [1, 0, 1, 1, 1, 0, 1, 0]),
        ([0.3980529522933324, 0.7837084828729131, 0.9492958577479171, 0.8336329507179389, 0.4023249378417618, 0.2697971911989604, 0.9615323930991638], [0, 1, 1, 1, 0, 1, 1]),
        ([0.18630302601016202, 0.6750390181745115, 0.16491604567607931, 0.8850249952559692, 0.5517693153288534, 0.9236788719534554], [0, 1, 0, 1, 0, 1]),
        ([0.8948161951922053, 0.9373591672216535, 0.3681422258146553, 0.6611518685035855, 0.49966156829111663, 0.014687506473181733, 0.055068129621015705, 0.9901287340475854, 0.7224844607910017, 0.6541545557311937], [1, 1, 0, 1, 0, 0, 0, 1, 1, 1]),
        ([0.79514461677259, 0.27408366538661844, 0.5015040141086913, 0.04226761245420285, 0.7730661419355433, 0.4536205963497906, 0.10906429906462989, 0.9136468962743243, 0.5532664939256077], [1, 0, 1, 0, 0, 1, 0, 1, 0]),
        ([0.209506029096985, 0.10207683440751536, 0.06060992335969062, 0.01635555563853197, 0.6267560890450867], [0, 0, 0, 0, 1]),
        ([0.24578832439505371, 0.5904288317168812, 0.8593866412858137, 0.8930410005114212, 0.9944881707786234], [0, 1, 1, 1, 1]),
        ([0.4097527006542627, 0.08356302805061433, 0.37726644850031177], [0, 0, 1]),
        ([0.85703420581388, 0.9822997651922313, 0.057338113173413086, 0.3799208963214824], [1, 1, 0, 0]),
        ([0.872402181996006, 0.4931645024649767, 0.880547446149999, 0.7735919723051992, 0.4677383977550522, 0.05168059065157471, 0.8411618231786084], [1, 0, 1, 1, 0, 1, 0]),
        ([0.6557487778832497, 0.2513734611513776, 0.07848019817622676, 0.815662822630094], [1, 0, 0, 1]),
        ([0.5623051411660299, 0.31323820458388973, 0.4342099935368844, 0.8261534676295808], [1, 0, 0, 1]),
        ([0.45240788562466394, 0.03951400147543249, 0.9751232512314327, 0.6152157301423237, 0.10651631480050738], [0, 0, 1, 1, 0]),
        ([0.9546645913434837, 0.048733211999058, 0.14044371140721956, 0.4901461271982086, 0.2779225928149436, 0.77957331092422, 0.7719180377216102], [1, 0, 0, 1, 0, 1, 0]),
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
