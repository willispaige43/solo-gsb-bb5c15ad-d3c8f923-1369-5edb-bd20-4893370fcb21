"""多尺码对称花样排针求解器。

排针规则：起针数 = 2×单侧边针数 + 中央针数 + 2×花样宽度×单侧重复数；
每侧使用相同且至少一次的完整花样重复（单侧重复数为正整数，左右两侧
相同）。实际片幅（厘米）= 起针数 ÷ 每10厘米针数 × 10。

给定按目标片幅从小到大排列的各尺码，为每个尺码选择单侧重复数，要求：
  - 全部尺码的重复数严格递增（按尺码顺序）；
  - 每个尺码的实际片幅与目标片幅之差不超过允许宽差。
在所有可行组合中最小化各尺码宽差绝对值之和，同分时取按尺码顺序
字典序最小的重复数组。

所有中间计算用 Fraction 精确进行（输入的十进制小数按字面还原），边界
比较不存在浮点误差。求解用动态规划：g[i][r] 为第 i 个尺码取重复数 r
时第 i..n 个尺码的最小宽差和，借助后缀最小值把每个尺码的转移压到
O(可行重复数)；回溯时从第一个尺码起依次取仍能达成全局最优的最小
重复数，保证同分解中字典序最小。
"""

import math
from fractions import Fraction


def _frac(value):
    """把 int/float/十进制字符串精确还原为 Fraction。"""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    # float 与字符串统一按十进制字面量还原（str(float) 是最短十进制表示）
    return Fraction(str(value))


def cast_on_for(edge, center, pattern_width, repeats):
    """起针数 = 2×单侧边针数 + 中央针数 + 2×花样宽度×单侧重复数。"""
    return 2 * edge + center + 2 * pattern_width * repeats


def width_for(cast_on, gauge):
    """实际片幅（厘米）= 起针数 ÷ 每10厘米针数 × 10，返回 Fraction。"""
    return Fraction(cast_on * 10, 1) / _frac(gauge)


def feasible_repeats(target, gauge, edge, center, pattern_width, tolerance):
    """单个尺码的可行单侧重复数区间 (最小, 最大)；无可行值时返回 None。

    可行指重复数为正整数（每侧至少一次完整重复），且对应实际片幅与
    目标片幅之差不超过允许宽差。边界用 Fraction 精确求 ceil/floor，
    不存在浮点误差。
    """
    g = _frac(gauge)
    t = _frac(target)
    tol = _frac(tolerance)
    base = 2 * edge + center
    step = 2 * pattern_width
    # |(base + step*r)*10/g - t| <= tol
    # <=> (t-tol)*g/10 <= base + step*r <= (t+tol)*g/10
    lo = ((t - tol) * g / 10 - base) / step
    hi = ((t + tol) * g / 10 - base) / step
    r_lo = max(1, math.ceil(lo))
    r_hi = math.floor(hi)
    if r_lo > r_hi:
        return None
    return (r_lo, r_hi)


def solve_layout(gauge, targets, edge, center, pattern_width, tolerance):
    """求解多尺码对称花样排针。

    参数：
      gauge         每10厘米针数（>0）
      targets       各尺码目标片幅列表（厘米，按从小到大排列）
      edge          单侧边针数（>=0 整数）
      center        中央针数（>=0 整数）
      pattern_width 花样宽度（针，>0 整数）
      tolerance     允许宽差（厘米，>=0）

    返回 dict：
      "ok"               是否存在可行组合
      "feasible"         各尺码可行重复数区间 [(lo, hi) | None, ...]
      "failure"          不可行原因：None 或
                         {"kind": "empty"|"increasing", "index", "required"}
                         empty      第 index 个尺码没有任何可用重复数
                         increasing 严格递增推进到第 index 个尺码时，需要
                                   重复数 >= required，超出其可行区间
      "repeats"          各尺码单侧重复数列表（不可行时为 None）
      "cast_on"          各尺码起针数列表（不可行时为 None）
      "widths"           各尺码实际片幅 Fraction 列表（不可行时为 None）
      "deviations"       各尺码宽差绝对值 Fraction 列表（不可行时为 None）
      "total_deviation"  宽差绝对值之和 Fraction（不可行时为 None）
    """
    g = _frac(gauge)
    base = 2 * edge + center
    step = 2 * pattern_width
    n = len(targets)

    feasible = [feasible_repeats(t, gauge, edge, center, pattern_width, tolerance)
                for t in targets]

    def failure(kind, index, required=None):
        return {
            "ok": False,
            "feasible": feasible,
            "failure": {"kind": kind, "index": index, "required": required},
            "repeats": None,
            "cast_on": None,
            "widths": None,
            "deviations": None,
            "total_deviation": None,
        }

    # 某尺码在允许宽差内没有任何可用重复数
    for i, f in enumerate(feasible):
        if f is None:
            return failure("empty", i)

    # 严格递增可行性：贪心取每个尺码的最小可行值即可判定
    prev = 0
    for i, (r_lo, r_hi) in enumerate(feasible):
        need = max(r_lo, prev + 1)
        if need > r_hi:
            return failure("increasing", i, required=prev + 1)
        prev = need

    def cost(i, r):
        width = Fraction((base + step * r) * 10, 1) / g
        return abs(width - _frac(targets[i]))

    # gvals[i][k]：第 i 个尺码取 r = feasible[i][0]+k 时，第 i..n 个尺码
    # 的最小宽差和；None 表示该取值没有可行后继
    gvals = [None] * n
    lo_n, hi_n = feasible[-1]
    gvals[n - 1] = [cost(n - 1, r) for r in range(lo_n, hi_n + 1)]
    for i in range(n - 2, -1, -1):
        lo_i, hi_i = feasible[i]
        lo_j = feasible[i + 1][0]
        nxt = gvals[i + 1]
        # 后缀最小值：sufmin[k] = min(nxt[k:])，O(区间长) 一次算好
        sufmin = [None] * len(nxt)
        sufmin[-1] = nxt[-1]
        for k in range(len(nxt) - 2, -1, -1):
            sufmin[k] = min(sufmin[k + 1], nxt[k])
        vals = []
        for r in range(lo_i, hi_i + 1):
            k = max(0, (r + 1) - lo_j)  # 后继中第一个 > r 的下标
            if k >= len(nxt):
                vals.append(None)  # 该取值之后没有可行后继
            else:
                vals.append(cost(i, r) + sufmin[k])
        gvals[i] = vals

    total = min(v for v in gvals[0] if v is not None)

    # 回溯：从第一个尺码起，依次取仍能达成全局最优的最小重复数，
    # 保证同分解中按尺码顺序字典序最小
    repeats = []
    remaining = total
    prev = 0
    for i in range(n):
        lo_i, hi_i = feasible[i]
        for r in range(max(lo_i, prev + 1), hi_i + 1):
            v = gvals[i][r - lo_i]
            if v is not None and v == remaining:
                repeats.append(r)
                remaining -= cost(i, r)
                prev = r
                break

    cast_ons = [cast_on_for(edge, center, pattern_width, r) for r in repeats]
    widths = [Fraction(c * 10, 1) / g for c in cast_ons]
    deviations = [abs(w - _frac(t)) for w, t in zip(widths, targets)]
    return {
        "ok": True,
        "feasible": feasible,
        "failure": None,
        "repeats": repeats,
        "cast_on": cast_ons,
        "widths": widths,
        "deviations": deviations,
        "total_deviation": total,
    }
