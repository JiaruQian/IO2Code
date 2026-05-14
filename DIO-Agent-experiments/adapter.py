"""IO2Code task adapter for DIO-Agent experiments."""

import sys
import os
import random
import string

# Add the benchmark task definitions to the import path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "io2code_tasks"))

from io2code_tasks import TASK_REGISTRY, EXTRA_TASK_REGISTRY


class TaskAdapter:
    """Adapt IO2Code tasks to DIO-Agent experiment files."""
    
    def __init__(self, task_name):
        self.task_name = task_name
        self.is_extra = task_name in EXTRA_TASK_REGISTRY
        
        if self.is_extra:
            self.task_info = EXTRA_TASK_REGISTRY[task_name]
        elif task_name in TASK_REGISTRY:
            self.func, self.dtype, self.bounds = TASK_REGISTRY[task_name]
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def get_task_description(self):
        """获取任务的自然语言描述"""
        if self.is_extra:
            return self.task_info.get("description", "")
        
        # 为序列任务生成描述
        descriptions = {
            "Prev1": "输出序列中每个位置前一个元素的值，第一个位置输出 0",
            "Prev2": "输出序列中每个位置前两个元素的值，前两个位置输出 0",
            "Prev3": "输出序列中每个位置前三个元素的值，前三个位置输出 0",
            "Prev4": "输出序列中每个位置前四个元素的值，前四个位置输出 0",
            "Prev5": "输出序列中每个位置前五个元素的值，前五个位置输出 0",
            "Sum_All": "输出序列的累加和，每个位置输出从开始到当前位置所有元素的和",
            "Sum_Last2": "输出序列中每个位置与前一个位置的和",
            "Sum_Last3": "输出序列中每个位置最近3个元素的和",
            "Sum_Last4": "输出序列中每个位置最近4个元素的和",
            "Sum_Last5": "输出序列中每个位置最近5个元素的和",
            "Sum_Last6": "输出序列中每个位置最近6个元素的和",
            "Sum_Last7": "输出序列中每个位置最近7个元素的和",
            "Max_Seen": "输出序列的累计最大值，每个位置输出从开始到当前位置的最大值",
            "Min_Seen": "输出序列的累计最小值，每个位置输出从开始到当前位置的最小值",
            "Diff_Last2": "输出序列中每个元素与前一个元素的差，第一个位置输出当前值",
            "Abs_Current": "输出序列中每个元素的绝对值",
            "Abs_Diff": "输出序列中相邻元素差的绝对值",
            "Diff_Abs_Values": "输出序列中相邻元素绝对值的差",
            "Current_Number": "输出与输入完全相同的序列",
            "Add_Mod_3": "输出序列的累加和对3取模",
            "Add_Mod_4": "输出序列的累加和对4取模",
            "Add_Mod_5": "输出序列的累加和对5取模",
            "Add_Mod_6": "输出序列的累加和对6取模",
            "Add_Mod_7": "输出序列的累加和对7取模",
            "Add_Mod_8": "输出序列的累加和对8取模",
            "Evens_Counter": "输出序列中到当前位置为止偶数的个数",
            "Evens_Detector": "输出序列中每个位置是否为偶数（1表示偶数，0表示奇数）",
            "Parity_All": "输出序列累计奇偶性（所有元素的异或）",
            "Parity_Last2": "输出最近2个元素的奇偶性",
            "Parity_Last3": "输出最近3个元素的奇偶性",
            "Parity_Last4": "输出最近4个元素的奇偶性",
            "Parity_Zeros": "输出序列中0的奇偶性",
            "Bitwise_And": "输出序列的累计按位与",
            "Bitwise_Or": "输出序列的累计按位或",
            "Bitwise_Xor": "输出序列的累计按位异或",
            "Bitwise_Not": "输出序列中每个位的按位取反",
            "Bit_Shift_Right": "输出序列右移一位",
            "Div_3": "输出累加和整除3的结果",
            "Div_5": "输出累加和整除5的结果",
            "Div_7": "输出累加和整除7的结果",
        }
        
        return descriptions.get(self.task_name, f"实现 {self.task_name} 任务")
    
    def get_input_output_examples(self, num_examples=5):
        """生成输入输出样例"""
        examples = []
        
        if self.is_extra:
            # Extra tasks need train/test split independence.
            # Use local generators that respect current RNG state.
            examples = self._generate_extra_examples(num_examples)
        else:
            # 序列任务的样例生成
            for _ in range(num_examples):
                length = random.randint(3, 10)
                
                if self.dtype == "bit":
                    xs = [random.randint(0, 1) for _ in range(length)]
                elif self.dtype == "int":
                    lo, hi = self.bounds
                    xs = [random.randint(lo, hi) for _ in range(length)]
                elif self.dtype == "float":
                    lo, hi = self.bounds
                    xs = [random.uniform(lo, hi) for _ in range(length)]
                elif self.dtype == "2d_bit":
                    # 2D 输入 (两个序列)
                    xs = [[random.randint(0, 1) for _ in range(length)],
                          [random.randint(0, 1) for _ in range(length)]]
                elif self.dtype == "2d_int":
                    lo, hi = self.bounds
                    xs = [[random.randint(lo, hi-1) for _ in range(length)],
                          [random.randint(lo, hi-1) for _ in range(length)]]
                else:
                    xs = [random.randint(0, 10) for _ in range(length)]
                
                ys = self.func(xs)
                examples.append((xs, ys))
        
        return examples

    def _generate_extra_examples(self, num_examples: int):
        task_func = self.task_info["func"]
        task_type = self.task_info["type"]
        value_range = self.task_info.get("range")
        examples = []
        seen_inputs = set()

        def add_example(inp):
            key = repr(inp)
            if key in seen_inputs:
                return False
            try:
                out = task_func(inp)
            except Exception:
                return False
            seen_inputs.add(key)
            examples.append((inp, out))
            return True

        def fill_with(generator, max_attempts=3000):
            attempts = 0
            while len(examples) < num_examples and attempts < max_attempts:
                attempts += 1
                add_example(generator())

        if task_type == "single_int":
            lo, hi = value_range
            if self.task_name == "Extra_Factorial":
                # Keep train/test split feasible (8 + 15) in DIO-Agent tasks.
                # Keep the original benchmark definition stable.
                hi = max(hi, 20)
            for x in [lo, lo + 1, (lo + hi) // 2, hi - 1, hi, 0, 1, -1]:
                if lo <= x <= hi:
                    add_example(x)
            fill_with(lambda: random.randint(lo, hi))

        elif task_type == "pair":
            lo, hi = value_range
            for pair in [
                (lo, lo),
                (hi, hi),
                (lo, hi),
                (hi, lo),
                ((lo + hi) // 2, max(lo, (lo + hi) // 3)),
                (1, 1),
            ]:
                a, b = pair
                if lo <= a <= hi and lo <= b <= hi:
                    add_example((a, b))
            fill_with(lambda: (random.randint(lo, hi), random.randint(lo, hi)))

        elif task_type == "string":
            base_samples = [
                "",
                "a",
                "aa",
                "aba",
                "abba",
                "abc",
                "racecar",
                "level",
                "python",
                "noon",
                "abcba",
                "abca",
            ]
            for item in base_samples:
                add_example(item)

            def gen_string():
                n = random.randint(1, 10)
                letters = string.ascii_lowercase
                return "".join(random.choice(letters) for _ in range(n))

            fill_with(gen_string)

        elif task_type == "int_list":
            lo, hi = value_range
            canned = [
                [],
                [lo],
                [hi],
                [0] if lo <= 0 <= hi else [lo],
                [1] if lo <= 1 <= hi else [hi],
                [lo, hi],
                [hi, lo],
                [lo, lo, hi],
                [hi, hi, lo],
            ]
            for item in canned:
                add_example(item)

            def gen_int_list():
                length = random.randint(0, 8)
                return [random.randint(lo, hi) for _ in range(length)]

            fill_with(gen_int_list)

        elif task_type == "bit_seq":
            canned = [
                [0, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [1, 0],
                [0, 0, 1, 0, 1, 1],
                [0, 1, 1, 0],
                [0, 0, 0, 1, 1, 1],
                [1, 1, 0, 0],
            ]
            for item in canned:
                add_example(item)
            fill_with(lambda: [random.randint(0, 1) for _ in range(random.randint(1, 10))])

        elif task_type in {"rpn_seq", "rpn_seq2", "rpn_seq3"}:
            op_tokens = {"rpn_seq": [10], "rpn_seq2": [10, 11], "rpn_seq3": [10, 11, 12]}[task_type]

            def gen_rpn():
                stack_depth = 0
                length = random.randint(3, 10)
                seq = []
                for _ in range(length):
                    can_op = stack_depth >= 2
                    if can_op and random.random() < 0.35:
                        seq.append(random.choice(op_tokens))
                        stack_depth -= 1
                    else:
                        seq.append(random.randint(0, 9))
                        stack_depth += 1
                while stack_depth >= 2 and random.random() < 0.6:
                    seq.append(random.choice(op_tokens))
                    stack_depth -= 1
                return seq

            canned = {
                "rpn_seq": [[3, 4, 10], [1, 2, 10, 3, 10], [1, 2, 3, 10, 10]],
                "rpn_seq2": [[3, 4, 10], [5, 2, 11], [1, 2, 10, 3, 11]],
                "rpn_seq3": [[3, 4, 10], [3, 4, 12], [2, 3, 12, 4, 10]],
            }
            for item in canned[task_type]:
                add_example(item)
            fill_with(gen_rpn)

        elif task_type == "bit_pairs":
            canned = [
                [[0, 0]],
                [[1, 0]],
                [[1, 1]],
                [[1, 1], [0, 0]],
                [[1, 1], [1, 1]],
                [[0, 1], [1, 0], [1, 1]],
            ]
            for item in canned:
                add_example(item)
            fill_with(lambda: [[random.randint(0, 1), random.randint(0, 1)] for _ in range(random.randint(1, 8))])

        elif task_type == "bit_list":
            canned = [
                [0],
                [1],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
                [0, 0, 0, 1, 1],
                [1, 0, 1, 0, 1],
                [0, 0, 1, 1, 0, 0],
            ]
            for item in canned:
                add_example(item)
            fill_with(lambda: [random.randint(0, 1) for _ in range(random.randint(1, 12))])

        elif task_type == "point_pair":
            lo, hi = value_range
            canned = [
                [[0, 0], [0, 0]],
                [[0, 0], [1, 1]],
                [[-1, -1], [2, 3]],
                [[lo, hi], [hi, lo]],
                [[-5, 7], [8, -6]],
            ]
            for item in canned:
                add_example(item)

            def gen_point_pair():
                return [
                    [random.randint(lo, hi), random.randint(lo, hi)],
                    [random.randint(lo, hi), random.randint(lo, hi)],
                ]

            fill_with(gen_point_pair)

        elif task_type == "points_triplet":
            lo, hi = value_range
            canned = [
                [[0, 0], [1, 1], [2, 2]],
                [[0, 0], [1, 2], [2, 1]],
                [[-1, -1], [0, 0], [1, 0]],
                [[lo, lo], [0, 0], [hi, hi]],
                [[2, 3], [4, 7], [5, 8]],
            ]
            for item in canned:
                add_example(item)

            def gen_points_triplet():
                return [
                    [random.randint(lo, hi), random.randint(lo, hi)]
                    for _ in range(3)
                ]

            fill_with(gen_points_triplet)

        elif task_type == "points_list":
            lo, hi = value_range
            canned = [
                [[0, 0], [1, 1], [2, 2], [3, 3]],
                [[0, 0], [1, 2], [2, 4], [3, 6]],
                [[0, 0], [1, 1], [2, 3]],
                [[-2, -1], [0, 0], [2, 1], [4, 2]],
            ]
            for item in canned:
                add_example(item)

            def gen_points_list():
                length = random.randint(2, 7)
                return [
                    [random.randint(lo, hi), random.randint(lo, hi)]
                    for _ in range(length)
                ]

            fill_with(gen_points_list)

        elif task_type == "points_path":
            lo, hi = value_range
            canned = [
                [[0, 0]],
                [[0, 0], [1, 1]],
                [[1, 1], [3, 4], [-1, 0]],
                [[-2, -2], [-2, 2], [2, 2], [2, -2]],
            ]
            for item in canned:
                add_example(item)

            def gen_points_path():
                length = random.randint(1, 7)
                return [
                    [random.randint(lo, hi), random.randint(lo, hi)]
                    for _ in range(length)
                ]

            fill_with(gen_points_path)

        elif task_type == "coin_change":
            lo, hi = value_range
            canned = [
                (0, [1, 2, 5]),
                (1, [1]),
                (5, [1, 2, 5]),
                (10, [2, 5]),
                (12, [1, 3, 4]),
            ]
            for item in canned:
                add_example(item)

            coin_sets = [
                [1],
                [1, 2],
                [1, 2, 5],
                [2, 3, 7],
                [1, 3, 4, 10],
            ]

            def gen_coin_change():
                amount = random.randint(max(0, lo), max(1, hi))
                coins = random.choice(coin_sets)
                return (amount, coins)

            fill_with(gen_coin_change)

        # Fallback: duplicate-safe random integer list tasks.
        if len(examples) < num_examples:
            fill_with(lambda: [random.randint(0, 10) for _ in range(random.randint(1, 6))], max_attempts=1000)

        return examples[:num_examples]
    
    def get_function_signature(self):
        """获取函数签名"""
        if self.is_extra:
            sig = self.task_info.get("signature", "process")
            return sig
        
        # 序列任务统一使用 process_sequence
        return "process_sequence"
    
    def get_input_type(self):
        """获取输入类型描述"""
        if self.is_extra:
            task_type = self.task_info.get("type", "")
            mapping = {
                "single_int": "int",
                "pair": "tuple/list with 2 numbers",
                "string": "string",
                "int_list": "list[int]",
                "bit_seq": "list[int]",
                "rpn_seq": "list[int]",
                "rpn_seq2": "list[int]",
                "rpn_seq3": "list[int]",
                "bit_pairs": "list[list[int]]",
                "bit_list": "list[int]",
                "point_pair": "list[list[int]]",
                "points_triplet": "list[list[int]]",
                "points_list": "list[list[int]]",
                "points_path": "list[list[int]]",
                "coin_change": "tuple[int, list[int]]",
            }
            return mapping.get(task_type, "varies")
        
        if self.dtype == "2d_bit" or self.dtype == "2d_int":
            return "list of two lists"
        return "list"
    
    def get_output_type(self):
        """获取输出类型描述"""
        if self.is_extra:
            task_type = self.task_info.get("type", "")
            mapping = {
                "single_int": "number or string",
                "pair": "number",
                "string": "number or bool",
                "int_list": "number/list/bool",
                "bit_seq": "list[int]",
                "rpn_seq": "number",
                "rpn_seq2": "number",
                "rpn_seq3": "number",
                "bit_pairs": "list[int]",
                "bit_list": "number",
                "point_pair": "number",
                "points_triplet": "bool",
                "points_list": "bool",
                "points_path": "number",
                "coin_change": "number",
            }
            return mapping.get(task_type, "varies")
        
        return "list"


def get_all_task_names(include_extra=False):
    """获取所有任务名称"""
    tasks = sorted(TASK_REGISTRY.keys())
    if include_extra:
        tasks.extend(sorted(EXTRA_TASK_REGISTRY.keys()))
    return tasks
