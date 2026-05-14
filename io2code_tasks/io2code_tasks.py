"""
基于任务定义生成测试数据（无需 data.pt 文件）
"""

import os
import sys
import json
import time
import random
from datetime import datetime
from typing import List, Tuple, Any, Callable, Optional
from dataclasses import dataclass


# ============ 任务真值函数定义 ============


def _build_synthesizer():
    from dio_agent_synthesizer import DIOAgentSynthesizer

    return DIOAgentSynthesizer()

def identity(xs):
    """Current_Number: 输出等于输入"""
    return list(xs)

def prev1(xs):
    """Prev1: 输出前一个元素，首位为0"""
    if not xs: return []
    return [0] + list(xs[:-1])

def prev2(xs):
    """Prev2: 输出前2个位置的元素"""
    if not xs: return []
    result = []
    for i in range(len(xs)):
        result.append(xs[i-2] if i >= 2 else 0)
    return result

def prev3(xs):
    """Prev3: 输出前3个位置的元素"""
    if not xs: return []
    result = []
    for i in range(len(xs)):
        result.append(xs[i-3] if i >= 3 else 0)
    return result

def prev4(xs):
    """Prev4: 输出前4个位置的元素"""
    if not xs: return []
    result = []
    for i in range(len(xs)):
        result.append(xs[i-4] if i >= 4 else 0)
    return result

def prev5(xs):
    """Prev5: 输出前5个位置的元素"""
    if not xs: return []
    result = []
    for i in range(len(xs)):
        result.append(xs[i-5] if i >= 5 else 0)
    return result

def cumsum(xs):
    """Sum_All: 累加求和"""
    result = []
    s = 0
    for x in xs:
        s += x
        result.append(s)
    return result

def sum_last2(xs):
    """Sum_Last2: 最近2个元素的和"""
    result = []
    for i in range(len(xs)):
        if i == 0:
            result.append(xs[i])
        else:
            result.append(xs[i] + xs[i-1])
    return result

def sum_last3(xs):
    """Sum_Last3: 最近3个元素的和"""
    result = []
    for i in range(len(xs)):
        s = sum(xs[max(0, i-2):i+1])
        result.append(s)
    return result

def sum_last4(xs):
    """Sum_Last4: 最近4个元素的和"""
    result = []
    for i in range(len(xs)):
        s = sum(xs[max(0, i-3):i+1])
        result.append(s)
    return result

def sum_last5(xs):
    """Sum_Last5: 最近5个元素的和"""
    result = []
    for i in range(len(xs)):
        s = sum(xs[max(0, i-4):i+1])
        result.append(s)
    return result

def sum_last6(xs):
    """Sum_Last6: 最近6个元素的和"""
    result = []
    for i in range(len(xs)):
        s = sum(xs[max(0, i-5):i+1])
        result.append(s)
    return result

def sum_last7(xs):
    """Sum_Last7: 最近7个元素的和"""
    result = []
    for i in range(len(xs)):
        s = sum(xs[max(0, i-6):i+1])
        result.append(s)
    return result

def diff_last2(xs):
    """Diff_Last2: 当前减前一个"""
    result = []
    for i in range(len(xs)):
        if i == 0:
            result.append(xs[i])
        else:
            result.append(xs[i] - xs[i-1])
    return result

def cummax(xs):
    """Max_Seen: 累计最大值"""
    if not xs: return []
    result = []
    m = xs[0]
    for x in xs:
        if x > m: m = x
        result.append(m)
    return result

def cummin(xs):
    """Min_Seen: 累计最小值"""
    if not xs: return []
    result = []
    m = xs[0]
    for x in xs:
        if x < m: m = x
        result.append(m)
    return result

def abs_current(xs):
    """Abs_Current: 绝对值"""
    return [abs(x) for x in xs]

def abs_diff(xs):
    """Abs_Diff: 相邻元素差的绝对值"""
    result = []
    for i in range(len(xs)):
        if i == 0:
            result.append(abs(xs[i]))
        else:
            result.append(abs(xs[i] - xs[i-1]))
    return result

def diff_abs(xs):
    """Diff_Abs_Values: 绝对值的差"""
    result = []
    for i in range(len(xs)):
        if i == 0:
            result.append(abs(xs[i]))
        else:
            result.append(abs(xs[i]) - abs(xs[i-1]))
    return result

def cumsum_mod3(xs):
    """Add_Mod_3: 累加取模3"""
    result = []
    s = 0
    for x in xs:
        s = (s + x) % 3
        result.append(s)
    return result

def cumsum_mod4(xs):
    """Add_Mod_4: 累加取模4"""
    result = []
    s = 0
    for x in xs:
        s = (s + x) % 4
        result.append(s)
    return result

def cumsum_mod5(xs):
    """Add_Mod_5: 累加取模5"""
    result = []
    s = 0
    for x in xs:
        s = (s + x) % 5
        result.append(s)
    return result

def cumsum_mod6(xs):
    """Add_Mod_6: 累加取模6"""
    result = []
    s = 0
    for x in xs:
        s = (s + x) % 6
        result.append(s)
    return result

def cumsum_mod7(xs):
    """Add_Mod_7: 累加取模7"""
    result = []
    s = 0
    for x in xs:
        s = (s + x) % 7
        result.append(s)
    return result

def cumsum_mod8(xs):
    """Add_Mod_8: 累加取模8"""
    result = []
    s = 0
    for x in xs:
        s = (s + x) % 8
        result.append(s)
    return result


# ============ Base-X Addition (两个序列相加，带进位) ============

def base_addition(xs, base):
    """Base-X 加法: 输入 [[a0,b0], [a1,b1], ...]，输出各位和（带进位）"""
    result = []
    carry = 0
    for pair in xs:
        a, b = pair[0], pair[1]
        s = a + b + carry
        result.append(s % base)
        carry = s // base
    return result

def binary_addition_2d(xs):
    """Binary_Addition: 二进制加法 (2D输入)"""
    return base_addition(xs, 2)

def base_3_addition(xs):
    """Base_3_Addition: 三进制加法 (2D输入)"""
    return base_addition(xs, 3)

def base_4_addition(xs):
    """Base_4_Addition: 四进制加法 (2D输入)"""
    return base_addition(xs, 4)

def base_5_addition(xs):
    """Base_5_Addition: 五进制加法 (2D输入)"""
    return base_addition(xs, 5)

def base_6_addition(xs):
    """Base_6_Addition: 六进制加法 (2D输入)"""
    return base_addition(xs, 6)

def base_7_addition(xs):
    """Base_7_Addition: 七进制加法 (2D输入)"""
    return base_addition(xs, 7)


def parity_all(xs):
    """Parity_All: 累计奇偶性 (1的个数 mod 2)"""
    result = []
    p = 0
    for x in xs:
        p = (p + x) % 2
        result.append(p)
    return result

def parity_bits_mod2(xs):
    """Parity_Bits_Mod2: 到目前为止看到的位数 mod 2
    位置 0: 1 bit -> 1
    位置 1: 2 bits -> 0
    位置 2: 3 bits -> 1
    ...
    """
    result = []
    for i in range(len(xs)):
        result.append((i + 1) % 2)
    return result

def parity_last2(xs):
    """Parity_Last2: 最近2个元素的奇偶性"""
    result = []
    for i in range(len(xs)):
        if i == 0:
            result.append(xs[i] % 2)
        else:
            result.append((xs[i] + xs[i-1]) % 2)
    return result

def parity_last3(xs):
    """Parity_Last3: 最近3个元素的奇偶性"""
    result = []
    for i in range(len(xs)):
        s = sum(xs[max(0, i-2):i+1])
        result.append(s % 2)
    return result

def parity_last4(xs):
    """Parity_Last4: 最近4个元素的奇偶性"""
    result = []
    for i in range(len(xs)):
        s = sum(xs[max(0, i-3):i+1])
        result.append(s % 2)
    return result

def parity_zeros(xs):
    """Parity_Zeros: 0的个数的奇偶性"""
    result = []
    count = 0
    for x in xs:
        if x == 0:
            count += 1
        result.append(count % 2)
    return result

def evens_counter(xs):
    """Evens_Counter: 偶数计数"""
    result = []
    count = 0
    for x in xs:
        if x % 2 == 0:
            count += 1
        result.append(count)
    return result

def evens_detector(xs):
    """Evens_Detector: 检测是否为偶数"""
    return [1 if x % 2 == 0 else 0 for x in xs]

def div3(xs):
    """Div_3: 累加和是否能被3整除"""
    result = []
    s = 0
    for x in xs:
        s += x
        result.append(1 if s % 3 == 0 else 0)
    return result

def div5(xs):
    """Div_5: 累加和是否能被5整除"""
    result = []
    s = 0
    for x in xs:
        s += x
        result.append(1 if s % 5 == 0 else 0)
    return result

def div7(xs):
    """Div_7: 累加和是否能被7整除"""
    result = []
    s = 0
    for x in xs:
        s += x
        result.append(1 if s % 7 == 0 else 0)
    return result

def alternating_last3(xs):
    """Alternating_Last3: 最近3个元素是否交替"""
    result = []
    for i in range(len(xs)):
        if i < 2:
            result.append(0)
        else:
            a, b, c = xs[i-2], xs[i-1], xs[i]
            result.append(1 if (a != b and b != c and a != c) else 0)
    return result

def alternating_last4(xs):
    """Alternating_Last4: 最近4个元素模式"""
    result = []
    for i in range(len(xs)):
        if i < 3:
            result.append(0)
        else:
            vals = xs[i-3:i+1]
            alt = all(vals[j] != vals[j+1] for j in range(3))
            result.append(1 if alt else 0)
    return result

def balanced_parenthesis(xs):
    """Balanced_Parenthesis: 括号平衡检测 (0=开, 1=闭)"""
    result = []
    depth = 0
    failed = False
    for x in xs:
        if failed:
            result.append(0)
            continue
        if x == 0:
            depth += 1
        else:
            depth -= 1
        if depth < 0:
            failed = True
            result.append(0)
        else:
            result.append(1 if depth == 0 else 0)
    return result

def majority_0_1(xs):
    """Majority_0_1: 0是否占多数 (margin=1)"""
    result = []
    zeros = 0
    ones = 0
    for x in xs:
        if x == 0:
            zeros += 1
        else:
            ones += 1
        result.append(1 if zeros > ones else 0)
    return result

def majority_0_2(xs):
    """Majority_0_2: 0是否占多数 (margin=2)"""
    result = []
    zeros = 0
    ones = 0
    for x in xs:
        if x == 0:
            zeros += 1
        else:
            ones += 1
        result.append(1 if zeros > ones + 1 else 0)
    return result

def majority_0_3(xs):
    """Majority_0_3: 0是否占多数 (margin=3)"""
    result = []
    zeros = 0
    ones = 0
    for x in xs:
        if x == 0:
            zeros += 1
        else:
            ones += 1
        result.append(1 if zeros > ones + 2 else 0)
    return result

def bitwise_not(xs):
    """Bitwise_Not: 按位取反 (0<->1)"""
    return [1 - x for x in xs]

def bitwise_and(xs):
    """Bitwise_And: 与前一个元素按位与"""
    result = []
    for i in range(len(xs)):
        if i == 0:
            result.append(xs[i])
        else:
            result.append(xs[i] & xs[i-1])
    return result

def bitwise_or(xs):
    """Bitwise_Or: 与前一个元素按位或"""
    result = []
    for i in range(len(xs)):
        if i == 0:
            result.append(xs[i])
        else:
            result.append(xs[i] | xs[i-1])
    return result

def bitwise_xor(xs):
    """Bitwise_Xor: 与前一个元素按位异或"""
    result = []
    for i in range(len(xs)):
        if i == 0:
            result.append(xs[i])
        else:
            result.append(xs[i] ^ xs[i-1])
    return result

def bit_shift_right(xs):
    """Bit_Shift_Right: 右移（同prev1）"""
    return prev1(xs)

def bit_palindrome(xs):
    """Bit_Palindrome: 检测到目前为止是否回文"""
    result = []
    for i in range(len(xs)):
        seq = xs[:i+1]
        result.append(1 if seq == seq[::-1] else 0)
    return result

def bit_dot_prod_mod2(xs):
    """Bit_Dot_Prod_Mod2: 两个bit序列的累计点积mod2
    输入: [[a0,b0], [a1,b1], ...] 表示两个bit序列
    输出: [a0*b0 mod 2, (a0*b0+a1*b1) mod 2, ...]
    """
    result = []
    s = 0
    for pair in xs:
        a, b = pair[0], pair[1]
        s = (s + a * b) % 2
        result.append(s)
    return result

def perfect_square_detector(xs):
    """Perfect_Square_Detector: 检测是否为完全平方数"""
    result = []
    for x in xs:
        root = int(x ** 0.5)
        result.append(1 if root * root == x else 0)
    return result

def prev_equals_current(xs):
    """Previous_Equals_Current: 当前是否等于前一个"""
    result = []
    for i in range(len(xs)):
        if i == 0:
            result.append(0)
        else:
            result.append(1 if xs[i] == xs[i-1] else 0)
    return result

def dithering(xs):
    """Dithering: Floyd-Steinberg dithering style"""
    result = []
    error = 0
    for x in xs:
        val = x + error
        if val >= 0.5:
            result.append(1)
            error = val - 1
        else:
            result.append(0)
            error = val
    return result

def newton_freebody(xs):
    """Newton_Freebody: 自由落体 v = v + a"""
    result = []
    v = 0
    for a in xs:
        v = v + a
        result.append(v)
    return result

def newton_gravity(xs):
    return newton_freebody(xs)

def newton_magnetic(xs):
    return newton_freebody(xs)

def newton_spring(xs):
    return newton_freebody(xs)


# ============ EXTRA 任务真值函数 (非序列任务) ============

def extra_gcd(inputs):
    """GCD: 最大公约数 - 输入 (a, b)，输出 gcd(a, b)"""
    a, b = inputs
    while b:
        a, b = b, a % b
    return a

def extra_dec2bin(n):
    """Dec2Bin: 十进制转二进制 - 输入整数，输出二进制字符串"""
    if n == 0:
        return "0"
    result = ""
    while n > 0:
        result = str(n % 2) + result
        n //= 2
    return result

def extra_is_palindrome(s):
    """Palindrome: 判断字符串是否回文"""
    return s == s[::-1]

def extra_factorial(n):
    """Factorial: 阶乘"""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def extra_fibonacci(n):
    """Fibonacci: 第n个斐波那契数"""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def extra_polish_eval(tokens):
    """Polish: 逆波兰表达式求值 - 10 表示 +"""
    stack = []
    for token in tokens:
        if token == 10:  # + operator
            if len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(a + b)
        else:
            stack.append(token)
    return stack[-1] if stack else 0

def extra_max_subarray(xs):
    """MaxSubarray: 最大子数组和 (Kadane算法)"""
    if not xs:
        return 0
    max_sum = current = xs[0]
    for x in xs[1:]:
        current = max(x, current + x)
        max_sum = max(max_sum, current)
    return max_sum

def extra_binary_add(ab):
    """BinaryAdd: 二进制加法 - 输入 [[a1,b1], [a2,b2]...]，输出和的各位"""
    result = []
    carry = 0
    for a, b in ab:
        s = a + b + carry
        result.append(s % 2)
        carry = s // 2
    return result

def extra_list_sum(xs):
    """ListSum: 简单列表求和"""
    return sum(xs)

def extra_list_product(xs):
    """ListProduct: 列表乘积"""
    if not xs:
        return 1
    result = 1
    for x in xs:
        result *= x
    return result

def extra_count_chars(s):
    """CountChars: 统计字符串长度"""
    return len(s)

def extra_reverse_list(xs):
    """ReverseList: 反转列表"""
    return xs[::-1]

def extra_find_max(xs):
    """FindMax: 找最大值"""
    if not xs:
        return None
    return max(xs)

def extra_find_min(xs):
    """FindMin: 找最小值"""
    if not xs:
        return None
    return min(xs)

def extra_is_sorted(xs):
    """IsSorted: 判断是否升序"""
    for i in range(1, len(xs)):
        if xs[i] < xs[i-1]:
            return False
    return True

def extra_count_zeros(xs):
    """CountZeros: 统计0的个数"""
    return sum(1 for x in xs if x == 0)

def extra_all_positive(xs):
    """AllPositive: 判断是否全为正数"""
    return all(x > 0 for x in xs)


# ============ 遗漏的重要 Extra 任务 ============

def extra_balanced_parentheses(xs):
    """BalancedParentheses: 检查括号是否平衡
    输入: 0 表示 '(', 1 表示 ')'
    输出: 每个位置返回 1 如果到该位置括号平衡，否则 0
    """
    balance = 0
    result = []
    valid = True
    for x in xs:
        if x == 0:  # (
            balance += 1
        else:  # )
            balance -= 1
        if balance < 0:
            valid = False
        result.append(1 if (balance == 0 and valid) else 0)
    return result

def extra_dec2roman(n):
    """Dec2Roman: 十进制转罗马数字"""
    if n <= 0:
        return ""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    result = ""
    for i in range(len(val)):
        while n >= val[i]:
            result += syms[i]
            n -= val[i]
    return result

def extra_polish_rpn(tokens):
    """PolishRPN: 逆波兰表达式求值 (栈顶序列)
    10 表示 + 运算符
    返回每一步栈顶的值
    """
    stack = []
    tops = []
    for token in tokens:
        if token == 10:  # + operator
            if len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(a + b)
        else:
            stack.append(token)
        tops.append(stack[-1] if stack else 0)
    return tops

def extra_binary_addition(ab_pairs):
    """BinaryAddition: 二进制加法
    输入: [[a1,b1], [a2,b2], ...] 从低位到高位
    输出: 每位的和 (带进位)
    """
    result = []
    carry = 0
    for a, b in ab_pairs:
        s = a + b + carry
        result.append(s % 2)
        carry = s // 2
    if carry:
        result.append(carry)
    return result

def extra_kth_root(inputs):
    """KthRoot: 求 n 的 k 次根 (取整)
    输入: (n, k) 元组
    """
    n, k = inputs
    if n <= 0:
        return 0
    return round(n ** (1/k))

def extra_multiply(inputs):
    """Multiply: 两数相乘
    输入: (a, b) 元组
    """
    a, b = inputs
    return a * b

def extra_power(inputs):
    """Power: 求幂 a^b
    输入: (a, b) 元组
    """
    a, b = inputs
    return a ** b

def extra_modulo(inputs):
    """Modulo: 求余数 a % b
    输入: (a, b) 元组
    """
    a, b = inputs
    if b == 0:
        return 0
    return a % b

def extra_integer_division(inputs):
    """IntegerDivision: 整数除法 a // b
    输入: (a, b) 元组
    """
    a, b = inputs
    if b == 0:
        return 0
    return a // b

def extra_abs_value(x):
    """AbsValue: 绝对值"""
    return abs(x)

def extra_sign(x):
    """Sign: 符号函数 (-1, 0, 1)"""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0

def extra_is_prime(n):
    """IsPrime: 判断是否为素数"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def extra_sum_digits(n):
    """SumDigits: 数字各位之和"""
    if n < 0:
        n = -n
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total

def extra_count_digits(n):
    """CountDigits: 数字位数"""
    if n == 0:
        return 1
    if n < 0:
        n = -n
    count = 0
    while n > 0:
        count += 1
        n //= 10
    return count

def extra_reverse_number(n):
    """ReverseNumber: 翻转数字"""
    negative = n < 0
    if negative:
        n = -n
    result = 0
    while n > 0:
        result = result * 10 + n % 10
        n //= 10
    return -result if negative else result


def extra_polish_rpn2(tokens):
    """PolishRPN2: 逆波兰表达式求值 (带 +/-)
    10 表示 + 运算符, 11 表示 - 运算符
    返回每一步栈顶的值
    """
    stack = []
    tops = []
    for token in tokens:
        if token == 10:  # + operator
            if len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(a + b)
        elif token == 11:  # - operator
            if len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(a - b)
        else:
            stack.append(token)
        tops.append(stack[-1] if stack else 0)
    return tops


def extra_polish_rpn3(tokens):
    """PolishRPN3: 逆波兰表达式求值 (带 +/-/*)
    10 表示 +, 11 表示 -, 12 表示 *
    返回每一步栈顶的值
    """
    stack = []
    tops = []
    for token in tokens:
        if token == 10:  # + operator
            if len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(a + b)
        elif token == 11:  # - operator
            if len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(a - b)
        elif token == 12:  # * operator
            if len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(a * b)
        else:
            stack.append(token)
        tops.append(stack[-1] if stack else 0)
    return tops


def extra_sparse_parity(bits):
    """SparseParity: 稀疏奇偶校验 - 前3位的奇偶性
    输入: 长度为 n 的二进制列表
    输出: 前3位的异或结果 (奇偶性)
    """
    if len(bits) < 3:
        return sum(bits) % 2
    return (bits[0] + bits[1] + bits[2]) % 2


def extra_sparse_parity_k(inputs):
    """SparseParityK: 前k位的奇偶性
    输入: (bits, k) 元组
    """
    bits, k = inputs
    return sum(bits[:k]) % 2


def extra_hamming_weight(bits):
    """HammingWeight: 汉明权重 (1的个数)"""
    return sum(bits)


def extra_leading_zeros(bits):
    """LeadingZeros: 前导零个数"""
    count = 0
    for b in bits:
        if b == 0:
            count += 1
        else:
            break
    return count


def extra_trailing_zeros(bits):
    """TrailingZeros: 尾随零个数"""
    count = 0
    for b in reversed(bits):
        if b == 0:
            count += 1
        else:
            break
    return count


def extra_kth_root_multidigit(inputs):
    """KthRootMultidigit: 求 n 的 k 次根，返回各位数字列表
    输入: (n, k) 元组
    输出: 结果的各位数字列表
    """
    n, k = inputs
    if n <= 0:
        return [0]
    root = round(n ** (1/k))
    if root == 0:
        return [0]
    digits = []
    while root > 0:
        digits.append(root % 10)
        root //= 10
    return digits[::-1]  # 返回正序


def extra_house_robber(nums):
    """HouseRobber: 经典动态规划问题
    不能抢劫相邻的房子，求能抢到的最大金额
    输入: 每个房子的金额列表
    输出: 能抢到的最大金额
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums[0], nums[1])
    
    # dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = curr
    return prev1


def extra_climb_stairs(n):
    """ClimbStairs: 爬楼梯问题
    每次可以爬 1 或 2 个台阶，求到达第 n 阶的方法数
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


def extra_coin_change_count(inputs):
    """CoinChangeCount: 硬币找零问题 - 方法数
    输入: (amount, coins) 元组，coins 是硬币面值列表
    输出: 凑成 amount 的方法数
    """
    amount, coins = inputs
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    return dp[amount]


def extra_longest_increasing_subseq_len(nums):
    """LongestIncreasingSubseqLen: 最长递增子序列长度"""
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


# ============ Geometry / Graphical LeetCode tasks ============

def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _dist2(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy


def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def lc_geo_manhattan_distance(points):
    if not points or len(points) < 2:
        return 0
    (x1, y1), (x2, y2) = points[0], points[1]
    return abs(x1 - x2) + abs(y1 - y2)


def lc_geo_chebyshev_distance(points):
    if not points or len(points) < 2:
        return 0
    (x1, y1), (x2, y2) = points[0], points[1]
    return max(abs(x1 - x2), abs(y1 - y2))


def lc_geo_euclidean_distance_squared(points):
    if not points or len(points) < 2:
        return 0
    return _dist2(points[0], points[1])


def lc_geo_rectangle_area_by_diagonal(points):
    if not points or len(points) < 2:
        return 0
    (x1, y1), (x2, y2) = points[0], points[1]
    return abs(x2 - x1) * abs(y2 - y1)


def lc_geo_is_axis_aligned(points):
    if not points or len(points) < 2:
        return True
    (x1, y1), (x2, y2) = points[0], points[1]
    return x1 == x2 or y1 == y2


def lc_geo_midpoint_integer_check(points):
    if not points or len(points) < 2:
        return True
    (x1, y1), (x2, y2) = points[0], points[1]
    return ((x1 + x2) % 2 == 0) and ((y1 + y2) % 2 == 0)


def lc_geo_valid_boomerang(points):
    if not points or len(points) != 3:
        return False
    return _cross(points[0], points[1], points[2]) != 0


def lc_geo_triangle_area2(points):
    if not points or len(points) != 3:
        return 0
    return abs(_cross(points[0], points[1], points[2]))


def lc_geo_orientation(points):
    if not points or len(points) != 3:
        return 0
    c = _cross(points[0], points[1], points[2])
    if c > 0:
        return 1
    if c < 0:
        return -1
    return 0


def lc_geo_check_straight_line(points):
    if not points or len(points) <= 2:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    dx = x1 - x0
    dy = y1 - y0
    for x, y in points[2:]:
        if (x - x0) * dy != (y - y0) * dx:
            return False
    return True


def lc_geo_bounding_box_area(points):
    if not points:
        return 0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def lc_geo_bounding_box_perimeter(points):
    if not points:
        return 0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return 2 * ((max(xs) - min(xs)) + (max(ys) - min(ys)))


def lc_geo_valid_square(points):
    if not points or len(points) != 4:
        return False
    dists = []
    for i in range(4):
        for j in range(i + 1, 4):
            d = _dist2(points[i], points[j])
            if d == 0:
                return False
            dists.append(d)
    dists.sort()
    return (
        dists[0] == dists[1] == dists[2] == dists[3]
        and dists[4] == dists[5]
        and dists[4] > dists[0]
    )


def lc_geo_largest_triangle_area2(points):
    if not points or len(points) < 3:
        return 0
    best = 0
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                best = max(best, abs(_cross(points[i], points[j], points[k])))
    return best


def lc_geo_number_of_boomerangs(points):
    if not points:
        return 0
    total = 0
    for i in range(len(points)):
        counter = {}
        for j in range(len(points)):
            if i == j:
                continue
            d = _dist2(points[i], points[j])
            counter[d] = counter.get(d, 0) + 1
        for cnt in counter.values():
            total += cnt * (cnt - 1)
    return total


def lc_geo_max_points_on_line(points):
    if not points:
        return 0
    n = len(points)
    ans = 1
    for i in range(n):
        slopes = {}
        dup = 0
        local_best = 0
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                dup += 1
                continue
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            if dx < 0:
                dx, dy = -dx, -dy
            elif dx == 0:
                dy = 1
            elif dy == 0:
                dx = 1
            key = (dx, dy)
            slopes[key] = slopes.get(key, 0) + 1
            local_best = max(local_best, slopes[key])
        ans = max(ans, local_best + dup + 1)
    return ans


def lc_geo_has_duplicate_points(points):
    if not points:
        return False
    seen = set()
    for p in points:
        key = (p[0], p[1])
        if key in seen:
            return True
        seen.add(key)
    return False


def lc_geo_min_time_visit_points(points):
    if not points or len(points) <= 1:
        return 0
    total = 0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        total += max(abs(x1 - x0), abs(y1 - y0))
    return total


def lc_geo_path_manhattan_length(points):
    if not points or len(points) <= 1:
        return 0
    total = 0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        total += abs(x1 - x0) + abs(y1 - y0)
    return total


def lc_geo_path_returns_origin(points):
    if not points:
        return True
    return points[0][0] == points[-1][0] and points[0][1] == points[-1][1]


# manually mutated tasks
def mutated_perfect_square_detector_add_5(xs):
    """Perfect_Square_Detector_Add_5: 检测是否为完全平方数加上5的值"""
    result = []
    for x in xs:
        root = int((x-5) ** 0.5)
        result.append(1 if root * root == x-5 else 0)
    return result


def mutated_extra_is_prime_add_3(n):
    """IsPrime_Add_3: 判断是否为素数加上3的值"""
    if n-3 < 2:
        return False
    if n-3 == 2:
        return True
    if (n-3) % 2 == 0:
        return False
    for i in range(3, int((n-3)**0.5) + 1, 2):
        if (n-3) % i == 0:
            return False
    return True


def mutated_extra_reverse_number_add_5(n):
    """ReverseNumber_Add_5: 翻转数字加上5之后的值"""
    n = n + 5
    negative = n < 0
    if negative:
        n = -n
    result = 0
    while n > 0:
        result = result * 10 + n % 10
        n //= 10
    return -result if negative else result


def mutated_always_true(xs):
    """Always_True: 无论输入列表如何,总是返回True"""
    return True

def mutated_extra_sum_digits_odd(n):
    """SumDigits_Odd: 数字各奇数位之和"""
    if n < 0:
        n = -n
    total = 0
    for i in range(len(str(n))):
        if i % 2 == 1:
            total += int(str(n)[i])
    return total

def mutated_extra_list_product_odd(xs):
    """ListProduct_Odd: result初始化为1,按顺序交替加、乘列表元素,返回结果"""
    if not xs:
        return 1
    result = 1
    for i, x in enumerate(xs):
        if i % 2 == 1:
            result *= x
        else:
            result += x
    return result

def mutated_extra_fibonacci_add_1(n):
    """Fibonacci_Add_1: 第n个斐波那契数,每次累加时都会多加上1"""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b+1
    return b

def mutated_extra_all_positive_add_3(xs):
    """AllPositive_Add_3: 判断是否全为大于-3的数"""
    return all(x+3 > 0 for x in xs)


def mutated_extra_coin_change_count_mul_2(inputs):
    """CoinChangeCount: 硬币找零问题的代码变异,每次动态规划累加时都会多乘以2,而不是原来的加法
    正确代码：
    输入: (amount, coins) 元组，coins 是硬币面值列表
    输出: 凑成 amount 的方法数
    人为变异代码：
    原dp[i]+=dp[i - coin]
    变异dp[i]+=dp[i - coin] * 2
    """
    amount, coins = inputs
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += (dp[i - coin] * 2)
    return dp[amount]


def mutated_lc_geo_min_time_visit_points_mul(points):
    """MinTimeVisitPoints_Mul_2: 计算从起点到终点的最短时间,但时间的累加人为变异成累乘"""
    if not points or len(points) <= 1:
        return 0
    total = 1
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        total *= max(abs(x1 - x0), abs(y1 - y0))
    return total
# ============ 任务注册表 ============

TASK_REGISTRY = {
    "Abs_Current": (abs_current, "int", (-20, 20)),
    "Abs_Diff": (abs_diff, "int", (-10, 10)),
    "Add_Mod_3": (cumsum_mod3, "int", (0, 3)),
    "Add_Mod_4": (cumsum_mod4, "int", (0, 4)),
    "Add_Mod_5": (cumsum_mod5, "int", (0, 5)),
    "Add_Mod_6": (cumsum_mod6, "int", (0, 6)),
    "Add_Mod_7": (cumsum_mod7, "int", (0, 7)),
    "Add_Mod_8": (cumsum_mod8, "int", (0, 8)),
    "Alternating_Last3": (alternating_last3, "int", (0, 3)),
    "Alternating_Last4": (alternating_last4, "bit", (0, 2)),
    "Balanced_Parenthesis": (balanced_parenthesis, "bit", (0, 2)),
    "Base_3_Addition": (base_3_addition, "2d_int", (0, 3)),  # 2D输入
    "Base_4_Addition": (base_4_addition, "2d_int", (0, 4)),
    "Base_5_Addition": (base_5_addition, "2d_int", (0, 5)),
    "Base_6_Addition": (base_6_addition, "2d_int", (0, 6)),
    "Base_7_Addition": (base_7_addition, "2d_int", (0, 7)),
    "Binary_Addition": (binary_addition_2d, "2d_bit", (0, 2)),  # 2D输入
    "Bit_Dot_Prod_Mod2": (bit_dot_prod_mod2, "2d_bit", (0, 2)),  # 2D输入
    "Bit_Palindrome": (bit_palindrome, "bit", (0, 2)),
    "Bit_Shift_Right": (bit_shift_right, "bit", (0, 2)),
    "Bitwise_And": (bitwise_and, "bit", (0, 2)),
    "Bitwise_Not": (bitwise_not, "bit", (0, 2)),
    "Bitwise_Or": (bitwise_or, "bit", (0, 2)),
    "Bitwise_Xor": (bitwise_xor, "bit", (0, 2)),
    "Current_Number": (identity, "int", (0, 100)),
    "Diff_Abs_Values": (diff_abs, "int", (-10, 10)),
    "Diff_Last2": (diff_last2, "int", (-10, 10)),
    "Dithering": (dithering, "float", (0, 1)),
    "Div_3": (div3, "int", (0, 10)),
    "Div_5": (div5, "int", (0, 10)),
    "Div_7": (div7, "int", (0, 10)),
    "Evens_Counter": (evens_counter, "int", (0, 10)),
    "Evens_Detector": (evens_detector, "int", (0, 10)),
    "Majority_0_1": (majority_0_1, "bit", (0, 2)),
    "Majority_0_2": (majority_0_2, "bit", (0, 2)),
    "Majority_0_3": (majority_0_3, "bit", (0, 2)),
    "Max_Seen": (cummax, "int", (0, 50)),
    "Min_Seen": (cummin, "int", (0, 50)),
    "Newton_Freebody": (newton_freebody, "int", (-5, 5)),
    "Newton_Gravity": (newton_gravity, "int", (-5, 5)),
    "Newton_Magnetic": (newton_magnetic, "int", (-5, 5)),
    "Newton_Spring": (newton_spring, "int", (-5, 5)),
    "Parity_All": (parity_all, "bit", (0, 2)),
    "Parity_Bits_Mod2": (parity_bits_mod2, "bit", (0, 2)),  # 位数 mod 2，与输入无关
    "Parity_Last2": (parity_last2, "bit", (0, 2)),
    "Parity_Last3": (parity_last3, "bit", (0, 2)),
    "Parity_Last4": (parity_last4, "bit", (0, 2)),
    "Parity_Zeros": (parity_zeros, "bit", (0, 2)),
    "Perfect_Square_Detector": (perfect_square_detector, "int", (0, 20)),
    "Prev1": (prev1, "int", (0, 100)),
    "Prev2": (prev2, "int", (0, 100)),
    "Prev3": (prev3, "int", (0, 100)),
    "Prev4": (prev4, "int", (0, 100)),
    "Prev5": (prev5, "int", (0, 100)),
    "Previous_Equals_Current": (prev_equals_current, "int", (0, 10)),
    "Sum_All": (cumsum, "int", (0, 20)),
    "Sum_Last2": (sum_last2, "int", (0, 20)),
    "Sum_Last3": (sum_last3, "int", (0, 20)),
    "Sum_Last4": (sum_last4, "int", (0, 20)),
    "Sum_Last5": (sum_last5, "int", (0, 20)),
    "Sum_Last6": (sum_last6, "int", (0, 20)),
    "Sum_Last7": (sum_last7, "int", (0, 20)),
}

# ============ Extra 任务注册表 (非序列任务) ============
EXTRA_TASK_REGISTRY = {
    # 数学算法
    "Extra_GCD": {
        "func": extra_gcd,
        "type": "pair",  # 输入是 (a, b) 元组
        "range": (1, 100),
        "description": "最大公约数"
    },
    "Extra_Factorial": {
        "func": extra_factorial,
        "type": "single_int",
        "range": (0, 12),
        "description": "阶乘"
    },
    "Extra_Fibonacci": {
        "func": extra_fibonacci,
        "type": "single_int",
        "range": (0, 20),
        "description": "斐波那契数列"
    },
    
    # 字符串操作
    "Extra_IsPalindrome": {
        "func": extra_is_palindrome,
        "type": "string",
        "range": None,
        "description": "判断回文"
    },
    "Extra_Dec2Bin": {
        "func": extra_dec2bin,
        "type": "single_int",
        "range": (0, 256),
        "description": "十进制转二进制"
    },
    "Extra_CountChars": {
        "func": extra_count_chars,
        "type": "string",
        "range": None,
        "description": "统计字符串长度"
    },
    
    # 列表操作
    "Extra_ListSum": {
        "func": extra_list_sum,
        "type": "int_list",
        "range": (0, 20),
        "description": "列表求和"
    },
    "Extra_ListProduct": {
        "func": extra_list_product,
        "type": "int_list",
        "range": (1, 5),
        "description": "列表乘积"
    },
    "Extra_ReverseList": {
        "func": extra_reverse_list,
        "type": "int_list",
        "range": (0, 20),
        "description": "反转列表"
    },
    "Extra_IsSorted": {
        "func": extra_is_sorted,
        "type": "int_list",
        "range": (0, 20),
        "description": "判断是否有序"
    },
    "Extra_CountZeros": {
        "func": extra_count_zeros,
        "type": "int_list",
        "range": (0, 5),
        "description": "统计0的个数"
    },
    "Extra_AllPositive": {
        "func": extra_all_positive,
        "type": "int_list",
        "range": (-5, 10),
        "description": "判断是否全为正数"
    },
    
    # ============ 新增的重要任务 ============
    
    # 经典算法问题
    "Extra_BalancedParentheses": {
        "func": extra_balanced_parentheses,
        "type": "bit_seq",  # 0/1 序列，输出序列
        "range": (0, 2),
        "description": "平衡括号检测 (栈问题)"
    },
    "Extra_Dec2Roman": {
        "func": extra_dec2roman,
        "type": "single_int",
        "range": (1, 100),
        "description": "十进制转罗马数字"
    },
    "Extra_PolishRPN": {
        "func": extra_polish_rpn,
        "type": "rpn_seq",  # RPN 表达式序列
        "range": (0, 10),
        "description": "逆波兰表达式求值 (栈)"
    },
    "Extra_KthRoot": {
        "func": extra_kth_root,
        "type": "pair",
        "range": (1, 100),
        "description": "求 k 次根"
    },
    "Extra_BinaryAddition": {
        "func": extra_binary_addition,
        "type": "bit_pairs",  # [[a,b], [a,b], ...]
        "range": (0, 2),
        "description": "二进制加法"
    },
    "Extra_Multiply": {
        "func": extra_multiply,
        "type": "pair",
        "range": (0, 20),
        "description": "两数相乘"
    },
    "Extra_Power": {
        "func": extra_power,
        "type": "pair",
        "range": (1, 5),
        "description": "求幂 a^b"
    },
    "Extra_Modulo": {
        "func": extra_modulo,
        "type": "pair",
        "range": (1, 50),
        "description": "求余数"
    },
    "Extra_IntegerDivision": {
        "func": extra_integer_division,
        "type": "pair",
        "range": (1, 50),
        "description": "整数除法"
    },
    
    # 数字操作
    "Extra_Sign": {
        "func": extra_sign,
        "type": "single_int",
        "range": (-100, 100),
        "description": "符号函数"
    },
    "Extra_SumDigits": {
        "func": extra_sum_digits,
        "type": "single_int",
        "range": (0, 1000),
        "description": "数字各位之和"
    },
    "Extra_CountDigits": {
        "func": extra_count_digits,
        "type": "single_int",
        "range": (0, 10000),
        "description": "数字位数"
    },
    "Extra_ReverseNumber": {
        "func": extra_reverse_number,
        "type": "single_int",
        "range": (0, 10000),
        "description": "翻转数字"
    },
    
    # ============ 新增: Polish 变体和稀疏奇偶 ============
    "Extra_PolishRPN2": {
        "func": extra_polish_rpn2,
        "type": "rpn_seq2",  # RPN +/- 表达式
        "range": (0, 12),
        "description": "逆波兰表达式 (+/-)"
    },
    "Extra_PolishRPN3": {
        "func": extra_polish_rpn3,
        "type": "rpn_seq3",  # RPN +/-/* 表达式
        "range": (0, 13),
        "description": "逆波兰表达式 (+/-/*)"
    },
    "Extra_SparseParity": {
        "func": extra_sparse_parity,
        "type": "bit_list",
        "range": (0, 2),
        "description": "稀疏奇偶校验 (前3位)"
    },
    "Extra_HammingWeight": {
        "func": extra_hamming_weight,
        "type": "bit_list",
        "range": (0, 2),
        "description": "汉明权重 (1的个数)"
    },
    "Extra_LeadingZeros": {
        "func": extra_leading_zeros,
        "type": "bit_list",
        "range": (0, 2),
        "description": "前导零个数"
    },
    "Extra_TrailingZeros": {
        "func": extra_trailing_zeros,
        "type": "bit_list",
        "range": (0, 2),
        "description": "尾随零个数"
    },
    
    # ============ 新增: DP 问题和多位数运算 ============
    "Extra_KthRootMultidigit": {
        "func": extra_kth_root_multidigit,
        "type": "pair",
        "range": (1, 1000),
        "description": "k次根 (多位数输出)"
    },
    "Extra_HouseRobber": {
        "func": extra_house_robber,
        "type": "int_list",
        "range": (0, 20),
        "description": "打家劫舍 (DP)"
    },
    "Extra_ClimbStairs": {
        "func": extra_climb_stairs,
        "type": "single_int",
        "range": (0, 20),
        "description": "爬楼梯 (DP)"
    },
    "Extra_LongestIncreasingSubseqLen": {
        "func": extra_longest_increasing_subseq_len,
        "type": "int_list",
        "range": (0, 20),
        "description": "最长递增子序列长度"
    },
    # ============ 新增: LeetCode Geometry 任务 ============
    "LeetCode_Geo_ManhattanDistance": {
        "func": lc_geo_manhattan_distance,
        "type": "point_pair",
        "range": (-20, 20),
        "description": "两点曼哈顿距离"
    },
    "LeetCode_Geo_ChebyshevDistance": {
        "func": lc_geo_chebyshev_distance,
        "type": "point_pair",
        "range": (-20, 20),
        "description": "两点切比雪夫距离 (1266 局部距离)"
    },
    "LeetCode_Geo_EuclideanDistanceSquared": {
        "func": lc_geo_euclidean_distance_squared,
        "type": "point_pair",
        "range": (-20, 20),
        "description": "两点欧式距离平方"
    },
    "LeetCode_Geo_RectangleAreaByDiagonal": {
        "func": lc_geo_rectangle_area_by_diagonal,
        "type": "point_pair",
        "range": (-20, 20),
        "description": "对角点定义的轴对齐矩形面积"
    },
    "LeetCode_Geo_IsAxisAligned": {
        "func": lc_geo_is_axis_aligned,
        "type": "point_pair",
        "range": (-20, 20),
        "description": "两点连线是否平行于坐标轴"
    },
    "LeetCode_Geo_MidpointIntegerCheck": {
        "func": lc_geo_midpoint_integer_check,
        "type": "point_pair",
        "range": (-20, 20),
        "description": "两点中点坐标是否都是整数"
    },
    "LeetCode_Geo_TriangleArea2": {
        "func": lc_geo_triangle_area2,
        "type": "points_triplet",
        "range": (-20, 20),
        "description": "三角形面积的两倍 (整数输出)"
    },
    "LeetCode_Geo_Orientation": {
        "func": lc_geo_orientation,
        "type": "points_triplet",
        "range": (-20, 20),
        "description": "三点方向: 逆时针/顺时针/共线"
    },
    "LeetCode_Geo_1232_CheckStraightLine": {
        "func": lc_geo_check_straight_line,
        "type": "points_list",
        "range": (-20, 20),
        "description": "点集是否在同一直线 (LeetCode 1232)"
    },
    "LeetCode_Geo_BoundingBoxArea": {
        "func": lc_geo_bounding_box_area,
        "type": "points_list",
        "range": (-20, 20),
        "description": "点集外接轴对齐矩形面积"
    },
    "LeetCode_Geo_BoundingBoxPerimeter": {
        "func": lc_geo_bounding_box_perimeter,
        "type": "points_list",
        "range": (-20, 20),
        "description": "点集外接轴对齐矩形周长"
    },
    "LeetCode_Geo_593_ValidSquare": {
        "func": lc_geo_valid_square,
        "type": "points_list",
        "range": (-20, 20),
        "description": "四点是否构成正方形 (LeetCode 593)"
    },
    "LeetCode_Geo_812_LargestTriangleArea2": {
        "func": lc_geo_largest_triangle_area2,
        "type": "points_list",
        "range": (-20, 20),
        "description": "最大三角形面积的两倍 (LeetCode 812 相关)"
    },
    "LeetCode_Geo_447_NumberOfBoomerangs": {
        "func": lc_geo_number_of_boomerangs,
        "type": "points_list",
        "range": (-20, 20),
        "description": "回旋镖数量 (LeetCode 447)"
    },
    "LeetCode_Geo_149_MaxPointsOnLine": {
        "func": lc_geo_max_points_on_line,
        "type": "points_list",
        "range": (-20, 20),
        "description": "同一直线上的最大点数 (LeetCode 149)"
    },
    "LeetCode_Geo_HasDuplicatePoints": {
        "func": lc_geo_has_duplicate_points,
        "type": "points_list",
        "range": (-20, 20),
        "description": "点集中是否存在重复点"
    },
    "LeetCode_Geo_1266_MinTimeVisitPoints": {
        "func": lc_geo_min_time_visit_points,
        "type": "points_path",
        "range": (-20, 20),
        "description": "走访点集最短时间 (LeetCode 1266)"
    },
    "LeetCode_Geo_PathReturnsOrigin": {
        "func": lc_geo_path_returns_origin,
        "type": "points_path",
        "range": (-20, 20),
        "description": "路径终点是否回到起点"
    },

    # ============ Manually mutated tasks ============
}


def generate_extra_examples(task_name: str, num_examples: int = 10) -> List[Tuple[Any, Any]]:
    """为 Extra 任务生成测试用例"""
    if task_name not in EXTRA_TASK_REGISTRY:
        return None
    
    task_info = EXTRA_TASK_REGISTRY[task_name]
    func = task_info["func"]
    task_type = task_info["type"]
    val_range = task_info["range"]
    
    random.seed(42)
    examples = []
    
    if task_type == "single_int":
        lo, hi = val_range
        # 使用集合避免重复值
        used_values = set()
        
        # 添加关键边界值
        for val in [lo, lo + 1, (lo + hi) // 2, hi - 1, hi]:
            if val <= hi and val not in used_values:
                examples.append((val, func(val)))
                used_values.add(val)
        
        # 添加随机值，避免重复
        attempts = 0
        max_attempts = (hi - lo + 1) * 10  # 避免无限循环
        while len(examples) < num_examples and attempts < max_attempts:
            val = random.randint(lo, hi)
            if val not in used_values:
                examples.append((val, func(val)))
                used_values.add(val)
            attempts += 1
    
    elif task_type == "pair":
        lo, hi = val_range
        # 基本用例
        examples.append(((1, 1), func((1, 1))))
        examples.append(((lo, hi), func((lo, hi))))
        examples.append(((hi, lo), func((hi, lo))))
        examples.append(((hi//2, hi//3), func((hi//2, hi//3))))
        for _ in range(num_examples - len(examples)):
            a, b = random.randint(lo, hi), random.randint(lo, hi)
            examples.append(((a, b), func((a, b))))
    
    elif task_type == "string":
        test_strings = ["", "a", "aa", "aba", "abc", "abba", "abcba", "hello", "racecar", "level"]
        for s in test_strings[:num_examples]:
            examples.append((s, func(s)))
    
    elif task_type == "int_list":
        lo, hi = val_range
        # 空列表和单元素
        examples.append(([], func([])))
        examples.append(([lo], func([lo])))
        examples.append(([hi], func([hi])))
        # 多元素
        for length in [2, 3, 4, 5, 6]:
            xs = [random.randint(lo, hi) for _ in range(length)]
            examples.append((xs, func(xs)))
    
    elif task_type == "bit_seq":
        # 平衡括号: 0=(, 1=)
        test_seqs = [
            [0, 1],           # ()
            [0, 0, 1, 1],     # (())
            [0, 1, 0, 1],     # ()()
            [1, 0],           # )(  不平衡
            [0, 0, 1, 0, 1, 1],  # (()())
            [0, 1, 1, 0],     # ())(
        ]
        for seq in test_seqs[:num_examples]:
            examples.append((seq, func(seq)))
    
    elif task_type == "rpn_seq":
        # 逆波兰表达式: 数字0-9, 10表示+
        test_seqs = [
            [3, 4, 10],           # 3 4 + = 7
            [1, 2, 10, 3, 10],    # 1 2 + 3 + = 6
            [5, 5, 10, 2, 10],    # 5 5 + 2 + = 12
            [1, 2, 3, 10, 10],    # 1 2 3 + + = 6
            [9, 1, 10],           # 9 1 + = 10
        ]
        for seq in test_seqs[:num_examples]:
            examples.append((seq, func(seq)))
    
    elif task_type == "bit_pairs":
        # 二进制加法 [[a1,b1], [a2,b2], ...]
        test_cases = [
            [[0, 0]],                     # 0 + 0 = 0
            [[1, 0]],                     # 1 + 0 = 1
            [[1, 1]],                     # 1 + 1 = 10
            [[1, 1], [0, 0]],             # 01 + 01 = 10
            [[1, 1], [1, 1]],             # 11 + 11 = 110
            [[0, 1], [1, 0], [1, 1]],     # 110 + 011 = 1001
        ]
        for pairs in test_cases[:num_examples]:
            examples.append((pairs, func(pairs)))
    
    elif task_type == "rpn_seq2":
        # 逆波兰表达式 +/-: 10=+, 11=-
        test_seqs = [
            [3, 4, 10],           # 3 4 + = 7
            [5, 2, 11],           # 5 2 - = 3
            [1, 2, 10, 3, 11],    # 1 2 + 3 - = 0
            [9, 3, 11, 2, 10],    # 9 3 - 2 + = 8
            [8, 5, 11],           # 8 5 - = 3
        ]
        for seq in test_seqs[:num_examples]:
            examples.append((seq, func(seq)))
    
    elif task_type == "rpn_seq3":
        # 逆波兰表达式 +/-/*: 10=+, 11=-, 12=*
        test_seqs = [
            [3, 4, 10],           # 3 4 + = 7
            [3, 4, 12],           # 3 4 * = 12
            [5, 2, 11],           # 5 2 - = 3
            [2, 3, 12, 4, 10],    # 2 3 * 4 + = 10
            [6, 2, 11, 3, 12],    # 6 2 - 3 * = 12
        ]
        for seq in test_seqs[:num_examples]:
            examples.append((seq, func(seq)))
    
    elif task_type == "bit_list":
        # 二进制列表
        test_seqs = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 0],
        ]
        for seq in test_seqs[:num_examples]:
            examples.append((seq, func(seq)))

    elif task_type == "point_pair":
        lo, hi = val_range
        test_pairs = [
            [[0, 0], [0, 0]],
            [[0, 0], [1, 1]],
            [[-1, -1], [2, 3]],
            [[lo, hi], [hi, lo]],
            [[-5, 7], [8, -6]],
        ]
        for pair in test_pairs[:num_examples]:
            examples.append((pair, func(pair)))

    elif task_type == "points_triplet":
        lo, hi = val_range
        test_sets = [
            [[0, 0], [1, 1], [2, 2]],   # 共线
            [[0, 0], [1, 2], [2, 1]],   # 不共线
            [[-1, -1], [0, 0], [1, 0]], # 不共线
            [[lo, lo], [0, 0], [hi, hi]],  # 共线
            [[2, 3], [4, 7], [5, 8]],   # 不共线
        ]
        for points in test_sets[:num_examples]:
            examples.append((points, func(points)))

    elif task_type == "points_list":
        lo, hi = val_range
        test_sets = [
            [[0, 0], [1, 1], [2, 2], [3, 3]],      # 共线
            [[0, 0], [1, 2], [2, 4], [3, 6]],      # 共线
            [[0, 0], [1, 1], [2, 3]],              # 不共线
            [[-2, -1], [0, 0], [2, 1], [4, 2]],    # 共线
            [[lo, lo], [0, 0], [hi, hi - 1]],      # 不共线
        ]
        for points in test_sets[:num_examples]:
            examples.append((points, func(points)))

    elif task_type == "points_path":
        lo, hi = val_range
        test_paths = [
            [[0, 0]],
            [[0, 0], [1, 1]],
            [[1, 1], [3, 4], [-1, 0]],
            [[-2, -2], [-2, 2], [2, 2], [2, -2]],
            [[lo, hi], [hi, lo], [0, 0]],
        ]
        for path in test_paths[:num_examples]:
            examples.append((path, func(path)))

    elif task_type == "coin_change":
        lo, hi = val_range
        coin_sets = [
            [1],
            [1, 2],
            [1, 2, 5],
            [2, 3, 7],
            [1, 3, 4, 10],
        ]
        seed_cases = [
            (0, [1, 2, 5]),
            (1, [1]),
            (5, [1, 2, 5]),
            (10, [2, 5]),
            (12, [1, 3, 4]),
        ]
        for amount, coins in seed_cases:
            examples.append(((amount, coins), func((amount, coins))))

        for _ in range(max(0, num_examples - len(examples))):
            amount = random.randint(max(0, lo), max(1, hi))
            coins = random.choice(coin_sets)
            examples.append(((amount, coins), func((amount, coins))))
    
    return examples[:num_examples]


def generate_examples(task_name: str, num_examples: int = 10) -> List[Tuple[List, List]]:
    """为任务生成测试用例"""
    if task_name not in TASK_REGISTRY:
        return None
    
    func, dtype, (lo, hi) = TASK_REGISTRY[task_name]
    random.seed(42)
    
    examples = []
    
    # 处理 2D 输入类型 (Base-X Addition)
    if dtype.startswith("2d_"):
        base_type = dtype[3:]  # "2d_int" -> "int", "2d_bit" -> "bit"
        # 边界用例
        examples.append(([[0, 0]], func([[0, 0]])))
        
        # 简单用例
        if base_type == "bit":
            examples.append(([[1, 0]], func([[1, 0]])))
            examples.append(([[1, 1]], func([[1, 1]])))
            examples.append(([[0, 1], [1, 0]], func([[0, 1], [1, 0]])))
        else:
            examples.append(([[1, 1]], func([[1, 1]])))
            examples.append(([[lo, hi-1]], func([[lo, hi-1]])))
        
        # 多元素
        for length in [2, 3, 4, 5, 6]:
            if base_type == "bit":
                xs = [[random.randint(0, 1), random.randint(0, 1)] for _ in range(length)]
            else:
                xs = [[random.randint(lo, hi-1), random.randint(lo, hi-1)] for _ in range(length)]
            examples.append((xs, func(xs)))
        
        return examples[:num_examples]
    
    # 边界用例
    examples.append(([], func([])))
    
    # 单元素
    if dtype == "bit":
        examples.append(([0], func([0])))
        examples.append(([1], func([1])))
    elif dtype == "float":
        examples.append(([0.3], func([0.3])))
        examples.append(([0.7], func([0.7])))
    else:
        examples.append(([lo], func([lo])))
        examples.append(([hi-1], func([hi-1])))
    
    # 多元素
    for length in [2, 3, 4, 5, 6]:
        if dtype == "bit":
            xs = [random.randint(0, 1) for _ in range(length)]
        elif dtype == "float":
            xs = [round(random.random(), 2) for _ in range(length)]
        else:
            xs = [random.randint(lo, hi-1) for _ in range(length)]
        examples.append((xs, func(xs)))
    
    return examples[:num_examples]


def get_all_tasks(include_extra: bool = False) -> List[str]:
    """获取所有任务名称"""
    tasks = sorted(TASK_REGISTRY.keys())
    if include_extra:
        tasks.extend(sorted(EXTRA_TASK_REGISTRY.keys()))
    return tasks


def get_extra_tasks() -> List[str]:
    """获取所有 Extra 任务名称"""
    return sorted(EXTRA_TASK_REGISTRY.keys())


def run_single_extra_task(task_name: str, num_examples: int = 6, verbose: bool = True, max_retries: int = 10) -> dict:
    """在单个 Extra 任务上运行 IO2Code task evaluation
    
    Args:
        task_name: 任务名称
        num_examples: 示例数量
        verbose: 是否详细输出
        max_retries: 每步最大重试次数
    """
    print(f"\n{'='*60}")
    print(f"[Extra] 任务: {task_name}")
    print(f"{'='*60}")
    
    if task_name not in EXTRA_TASK_REGISTRY:
        return {
            "task": task_name,
            "success": False,
            "code": "",
            "llm_calls": 0,
            "time": 0,
            "error": "任务未定义"
        }
    
    task_info = EXTRA_TASK_REGISTRY[task_name]
    func = task_info["func"]
    
    # 生成训练数据
    train_examples = generate_extra_examples(task_name, num_examples)
    
    if verbose:
        print(f"任务描述: {task_info['description']}")
        print(f"类型: {task_info['type']}")
        print(f"样例数: {len(train_examples)}")
        for i, (x, y) in enumerate(train_examples[:4]):
            print(f"  {i+1}. {x} -> {y}")
    
    # 运行合成
    start_time = time.time()
    synthesizer = _build_synthesizer()
    
    try:
        result = synthesizer.synthesize(
            train_examples,
            func_name="solution",
            verbose=verbose,
            max_retries=max_retries
        )
        elapsed = time.time() - start_time
        
        # 额外验证（只有训练成功才会验证）
        extra_passed = False
        if result.success:
            extra_passed = True  # 假设通过，下面测试可能改为 False
            random.seed(123)
            test_inputs = []
            
            task_type = task_info["type"]
            val_range = task_info["range"]
            
            if task_type == "single_int":
                lo, hi = val_range
                for _ in range(5):
                    test_inputs.append(random.randint(lo, hi))
            elif task_type == "pair":
                lo, hi = val_range
                for _ in range(5):
                    test_inputs.append((random.randint(lo, hi), random.randint(lo, hi)))
            elif task_type == "string":
                test_strings = ["test", "level", "python", "noon", "world"]
                test_inputs = test_strings
            elif task_type == "int_list":
                lo, hi = val_range
                for _ in range(5):
                    length = random.randint(2, 6)
                    test_inputs.append([random.randint(lo, hi) for _ in range(length)])
            elif task_type == "coin_change":
                lo, hi = val_range
                candidate_coins = [[1], [1, 2], [1, 2, 5], [2, 3, 7], [1, 3, 4]]
                for _ in range(5):
                    amount = random.randint(max(0, lo), max(1, hi))
                    coins = random.choice(candidate_coins)
                    test_inputs.append((amount, coins))
            
            try:
                env = {}
                exec(result.code, env)
                synth_func = env["solution"]
                for inp in test_inputs:
                    expected = func(inp)
                    actual = synth_func(inp)
                    if actual != expected:
                        extra_passed = False
                        if verbose:
                            print(f"⚠️ 验证失败: {inp} -> {actual}, 期望 {expected}")
                        break
            except Exception as e:
                extra_passed = False
                if verbose:
                    print(f"⚠️ 验证出错: {e}")
        
        success = result.success and extra_passed
        
        status = "✅ 成功" if success else "❌ 失败"
        print(f"\n{status} (LLM: {result.llm_calls}次, {elapsed:.1f}s)")
        if result.code and verbose:
            print(f"代码:\n{result.code}")
        
        return {
            "task": task_name,
            "success": success,
            "code": result.code,
            "llm_calls": result.llm_calls,
            "time": elapsed,
            "verified": extra_passed,
            "is_extra": True,
            "token_usage": result.token_usage,
            "token_usage_per_iteration": result.token_usage_per_iteration,
            "history": result.history  # 保存交互历史
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ 错误: {e}")
        return {
            "task": task_name,
            "success": False,
            "code": "",
            "llm_calls": synthesizer.llm_calls,
            "time": elapsed,
            "error": str(e),
            "is_extra": True,
            "token_usage": synthesizer.token_usage,
            "token_usage_per_iteration": [],
            "history": synthesizer.history  # 保存交互历史
        }


def run_single_task(task_name: str, num_examples: int = 10, verbose: bool = True, max_retries: int = 10) -> dict:
    """在单个任务上运行 IO2Code task evaluation
    
    Args:
        task_name: 任务名称
        num_examples: 示例数量
        verbose: 是否详细输出
        max_retries: 每步最大重试次数
    """
    print(f"\n{'='*60}")
    print(f"任务: {task_name}")
    print(f"{'='*60}")
    
    if task_name not in TASK_REGISTRY:
        return {
            "task": task_name,
            "success": False,
            "code": "",
            "llm_calls": 0,
            "time": 0,
            "error": "任务未定义"
        }
    
    func, dtype, _ = TASK_REGISTRY[task_name]
    
    # 生成训练数据
    train_examples = generate_examples(task_name, num_examples)
    
    if verbose:
        print(f"样例数: {len(train_examples)}")
        for i, (x, y) in enumerate(train_examples[:3]):
            x_show = x[:6] if len(x) > 6 else x
            y_show = y[:6] if isinstance(y, list) and len(y) > 6 else y
            print(f"  {i+1}. {x_show} -> {y_show}")
    
    # 运行合成
    start_time = time.time()
    synthesizer = _build_synthesizer()
    
    try:
        result = synthesizer.synthesize(
            train_examples,
            func_name="solution",
            verbose=verbose,
            max_retries=max_retries
        )
        elapsed = time.time() - start_time
        
        # 额外验证（只有训练成功才会验证）
        extra_passed = False
        if result.success:
            extra_passed = True  # 假设通过，下面测试可能改为 False
            random.seed(123)
            test_inputs = []
            for _ in range(5):
                length = random.randint(3, 8)
                if dtype == "bit":
                    test_inputs.append([random.randint(0, 1) for _ in range(length)])
                elif dtype == "float":
                    test_inputs.append([round(random.random(), 2) for _ in range(length)])
                else:
                    lo, hi = TASK_REGISTRY[task_name][2]
                    test_inputs.append([random.randint(lo, hi-1) for _ in range(length)])
            
            try:
                env = {}
                exec(result.code, env)
                synth_func = env["solution"]
                for inp in test_inputs:
                    expected = func(inp)
                    actual = synth_func(inp)
                    if actual != expected:
                        extra_passed = False
                        if verbose:
                            print(f"⚠️ 验证失败: {inp[:5]}... -> {actual}, 期望 {expected}")
                        break
            except Exception as e:
                extra_passed = False
                if verbose:
                    print(f"⚠️ 验证出错: {e}")
        
        success = result.success and extra_passed
        
        status = "✅ 成功" if success else "❌ 失败"
        print(f"\n{status} (LLM: {result.llm_calls}次, {elapsed:.1f}s)")
        if result.code and verbose:
            print(f"代码:\n{result.code}")
        
        return {
            "task": task_name,
            "success": success,
            "code": result.code,
            "llm_calls": result.llm_calls,
            "time": elapsed,
            "verified": extra_passed,
            "token_usage": result.token_usage,
            "token_usage_per_iteration": result.token_usage_per_iteration,
            "history": result.history  # 保存交互历史
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ 错误: {e}")
        return {
            "task": task_name,
            "success": False,
            "code": "",
            "llm_calls": synthesizer.llm_calls,
            "time": elapsed,
            "error": str(e),
            "token_usage": synthesizer.token_usage,
            "token_usage_per_iteration": [],
            "history": synthesizer.history  # 保存交互历史
        }


def _save_single_result(result: dict, result_dir: str, history_dir: str):
    """保存单个任务结果和交互历史"""
    import os
    task_name = result["task"]
    
    # 保存任务结果
    result_file = os.path.join(result_dir, f"{task_name}.json")
    # 复制结果，排除 history（单独保存）
    result_copy = {k: v for k, v in result.items() if k != "history"}
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_copy, f, ensure_ascii=False, indent=2)
    
    # 保存交互历史（如果有）
    if result.get("history"):
        history_file = os.path.join(history_dir, f"{task_name}_history.json")
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump({
                "task": task_name,
                "llm_calls": result.get("llm_calls", 0),
                "interactions": result["history"]
            }, f, ensure_ascii=False, indent=2)


def _update_final_summary(all_results: list, result_dir: str):
    """更新最终的summary.json文件，包含所有任务结果"""
    success_count = sum(1 for r in all_results if r["success"])
    total_calls = sum(r.get("llm_calls", 0) for r in all_results)
    total_time = sum(r.get("time", 0) for r in all_results)
    total_prompt_tokens = sum(r.get("token_usage", {}).get("prompt_tokens", 0) for r in all_results)
    total_completion_tokens = sum(r.get("token_usage", {}).get("completion_tokens", 0) for r in all_results)
    total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in all_results)
    
    # 分别统计普通任务和 Extra 任务
    regular_results = [r for r in all_results if not r.get("is_extra")]
    extra_results = [r for r in all_results if r.get("is_extra")]
    
    # 保存总结文件
    summary_file = os.path.join(result_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(all_results),
                "success": success_count,
                "success_rate": success_count / len(all_results) * 100 if all_results else 0,
                "regular_tasks": len(regular_results),
                "regular_success": sum(1 for r in regular_results if r["success"]) if regular_results else 0,
                "extra_tasks": len(extra_results),
                "extra_success": sum(1 for r in extra_results if r["success"]) if extra_results else 0,
                "total_llm_calls": total_calls,
                "total_time": total_time,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
            },
            "results": all_results
        }, f, ensure_ascii=False, indent=2)


def run_all_tasks(task_names: List[str] = None, verbose: bool = True,
                  num_examples: int = 10, save: bool = True, 
                  include_extra: bool = False, extra_only: bool = False,
                  max_retries: int = 5, resume_dir: str = None):
    """在所有任务上运行测试
    
    Args:
        task_names: 指定任务名称列表
        verbose: 是否输出详细信息
        num_examples: 每任务样例数
        save: 是否保存结果
        include_extra: 是否包含 Extra 任务
        extra_only: 只运行 Extra 任务
        max_retries: 每步最大重试次数
        resume_dir: 续写模式：指定已有结果目录
    """
    import os
    
    # 创建或使用已有结果目录
    if resume_dir:
        result_dir = resume_dir
        history_dir = os.path.join(result_dir, "history")
        if not os.path.exists(result_dir):
            print(f"❌ 目录不存在: {result_dir}")
            return
        print(f"🔄 续写模式: {result_dir}/")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "extra" if extra_only else ("all" if include_extra else "62tasks")
        result_dir = f"results_{suffix}_{timestamp}"
        history_dir = os.path.join(result_dir, "history")
        
    if save:
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(history_dir, exist_ok=True)
        if not resume_dir:
            print(f"📁 结果保存目录: {result_dir}/")
    
    if extra_only:
        all_tasks = get_extra_tasks()
        title = "IO2Code extra task benchmark"
    elif include_extra:
        all_tasks = get_all_tasks(include_extra=True)
        title = "IO2Code task benchmark"
    else:
        all_tasks = get_all_tasks()
        title = "IO2Code task benchmark"
    
    if task_names:
        tasks = [t for t in all_tasks if t in task_names]
        # 也检查 extra 任务
        for t in task_names:
            if t in EXTRA_TASK_REGISTRY and t not in tasks:
                tasks.append(t)
        if not tasks:
            print(f"❌ 未找到任务: {task_names}")
            print(f"可用任务: {all_tasks[:10]}...")
            return
    else:
        tasks = all_tasks
    
    # 续写模式：跳过已完成的任务，并加载之前的结果
    previous_results = []
    if resume_dir:
        completed = []
        if os.path.exists(history_dir):
            for f in os.listdir(history_dir):
                if f.endswith('_history.json'):
                    task = f.replace('_history.json', '')
                    completed.append(task)
                    # 加载之前保存的任务结果
                    task_result_file = os.path.join(resume_dir, f"{task}.json")
                    if os.path.exists(task_result_file):
                        with open(task_result_file, 'r', encoding='utf-8') as rf:
                            previous_results.append(json.load(rf))
        tasks = [t for t in tasks if t not in completed]
        if completed:
            print(f"⏭️  跳过已完成: {len(completed)} 个任务")
            print(f"   已完成: {', '.join(sorted(completed)[:5])}..." if len(completed) > 5 else f"   已完成: {', '.join(sorted(completed))}")
        if not tasks:
            print(f"✅ 所有任务已完成！")
            # 更新summary以包含所有已完成的任务
            if save and previous_results:
                print(f"📝 更新 summary.json，包含所有 {len(previous_results)} 个任务...")
                _update_final_summary(previous_results, result_dir)
                print(f"✅ Summary 已更新！")
            return
    
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"任务数量: {len(tasks)}")
    
    results = []
    for i, task_name in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}]", end="")
        
        try:
            # 根据任务类型选择运行函数
            if task_name in EXTRA_TASK_REGISTRY:
                result = run_single_extra_task(task_name, num_examples, verbose, max_retries)
            else:
                result = run_single_task(task_name, num_examples, verbose, max_retries)
            
            # 每个任务完成后立即保存
            if save:
                _save_single_result(result, result_dir, history_dir)
                print(f"   💾 已保存: {task_name}")
            
            results.append(result)
            
        except Exception as e:
            # 捕获任务级别的异常
            error_result = {
                "task": task_name,
                "success": False,
                "error": str(e),
                "llm_calls": 0,
                "time": 0
            }
            results.append(error_result)
            if save:
                _save_single_result(error_result, result_dir, history_dir)
            print(f"\n❌ 任务异常: {task_name} - {e}")
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    # 合并之前的结果和当前的结果
    all_results = previous_results + results
    
    success_count = sum(1 for r in all_results if r["success"])
    total_calls = sum(r.get("llm_calls", 0) for r in all_results)
    total_time = sum(r.get("time", 0) for r in all_results)
    total_prompt_tokens = sum(r.get("token_usage", {}).get("prompt_tokens", 0) for r in all_results)
    total_completion_tokens = sum(r.get("token_usage", {}).get("completion_tokens", 0) for r in all_results)
    total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in all_results)
    
    # 分别统计普通任务和 Extra 任务
    regular_results = [r for r in all_results if not r.get("is_extra")]
    extra_results = [r for r in all_results if r.get("is_extra")]
    
    print(f"总体成功: {success_count}/{len(all_results)} ({success_count/len(all_results)*100:.1f}%)")
    if regular_results:
        regular_success = sum(1 for r in regular_results if r["success"])
        print(f"  62任务: {regular_success}/{len(regular_results)}")
    if extra_results:
        extra_success = sum(1 for r in extra_results if r["success"])
        print(f"  Extra任务: {extra_success}/{len(extra_results)}")
    print(f"总 LLM 调用: {total_calls} 次")
    print(f"总耗时: {total_time:.1f} 秒")
    print(f"总 Prompt Tokens: {total_prompt_tokens}")
    print(f"总 Completion Tokens: {total_completion_tokens}")
    print(f"总 Tokens: {total_tokens}")
    
    if previous_results:
        print(f"\n💡 本次运行: {len(results)} 个任务")
        print(f"   之前完成: {len(previous_results)} 个任务")
    print()
    
    print("详细结果:")
    for r in all_results:
        status = "✅" if r["success"] else "❌"
        extra_mark = "[Extra] " if r.get("is_extra") else ""
        # 标记是之前完成还是本次完成
        prefix = "   " if r in previous_results else "🆕 "
        print(f"  {prefix}{status} {extra_mark}{r['task']:25s} - {r.get('llm_calls', 0):2d}次, {r.get('time', 0):5.1f}s")
    
    if save:
        # 保存总结文件（包含所有结果）
        summary_file = os.path.join(result_dir, "summary.json")
        # 去掉 history 字段（已单独保存）
        all_results_no_history = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": len(all_results),
                    "success": success_count,
                    "success_rate": success_count / len(all_results) * 100,
                    "regular_tasks": len(regular_results),
                    "regular_success": sum(1 for r in regular_results if r["success"]) if regular_results else 0,
                    "extra_tasks": len(extra_results),
                    "extra_success": sum(1 for r in extra_results if r["success"]) if extra_results else 0,
                    "total_llm_calls": total_calls,
                    "total_time": total_time,
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                },
                "results": all_results_no_history
            }, f, ensure_ascii=False, indent=2)
        print(f"\n📁 结果保存目录: {result_dir}/")
        print(f"   summary.json        - 汇总结果 (共 {len(all_results)} 个任务)")
        print(f"   <task>.json         - 各任务结果")
        print(f"   history/<task>_history.json - 交互历史")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="在62个任务上测试 IO2Code")
    parser.add_argument("--tasks", "-t", nargs="+", help="指定任务名称")
    parser.add_argument("--quiet", "-q", action="store_true", help="静默模式")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有任务")
    parser.add_argument("--examples", "-n", type=int, default=8, help="每任务样例数")
    parser.add_argument("--retries", "-r", type=int, default=3, help="每步最大重试次数")
    parser.add_argument("--no-save", action="store_true", help="不保存结果")
    parser.add_argument("--extra", "-e", action="store_true", help="只运行 Extra 任务")
    parser.add_argument("--all", "-a", action="store_true", help="运行所有任务(包含 Extra)")
    parser.add_argument("--resume", type=str, help="续写模式：指定已有结果目录，跳过已完成任务")
    
    args = parser.parse_args()
    
    if args.list:
        print("=" * 40)
        print("62 任务 (序列到序列):")
        print("=" * 40)
        tasks = get_all_tasks()
        for i, t in enumerate(tasks):
            print(f"  {i+1:2d}. {t}")
        
        print()
        print("=" * 40)
        print("Extra 任务 (非序列):")
        print("=" * 40)
        extra_tasks = get_extra_tasks()
        for i, t in enumerate(extra_tasks):
            desc = EXTRA_TASK_REGISTRY[t]["description"]
            print(f"  {i+1:2d}. {t:25s} - {desc}")
        
        print()
        print(f"共 {len(tasks)} + {len(extra_tasks)} = {len(tasks) + len(extra_tasks)} 个任务")
        sys.exit(0)
    
    run_all_tasks(
        task_names=args.tasks,
        verbose=not args.quiet,
        num_examples=args.examples,
        save=not args.no_save,
        include_extra=args.all,
        extra_only=args.extra,
        max_retries=args.retries,
        resume_dir=args.resume
    )
