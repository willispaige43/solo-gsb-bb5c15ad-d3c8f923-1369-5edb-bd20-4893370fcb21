import unittest
from fractions import Fraction

from pattern_layout import (
    cast_on_for,
    feasible_repeats,
    solve_layout,
    width_for,
)


class TestHelpers(unittest.TestCase):
    def test_cast_on_formula(self):
        # 起针数 = 2×边针 + 中央针 + 2×花样宽度×单侧重复数
        self.assertEqual(cast_on_for(1, 4, 6, 3), 2 * 1 + 4 + 2 * 6 * 3)
        self.assertEqual(cast_on_for(0, 0, 2, 1), 4)

    def test_width_from_gauge(self):
        # 实际片幅 = 起针数 ÷ 每10厘米针数 × 10
        self.assertEqual(width_for(54, 20), Fraction(27))
        self.assertEqual(width_for(8, 22.5), Fraction(32, 9))

    def test_feasible_repeats_bounds(self):
        # width(r) = 3+6r；目标 27±1 -> 仅 r=4
        self.assertEqual(feasible_repeats(27, 20, 1, 4, 6, 1), (4, 4))
        # width(r) = 2r；目标 5±2.5 -> r ∈ {2, 3}
        self.assertEqual(feasible_repeats(5, 20, 0, 0, 2, 2.5), (2, 3))
        # 目标 27.6±0.1：width 总是整数，无可行重复数
        self.assertIsNone(feasible_repeats(27.6, 20, 1, 4, 6, 0.1))

    def test_feasible_repeats_at_least_one(self):
        # 每侧至少一次完整重复：目标再小，重复数也不小于 1
        lo, hi = feasible_repeats(1, 20, 0, 0, 2, 10)
        self.assertEqual(lo, 1)
        self.assertEqual(hi, 5)  # |2r-1| <= 10 -> r <= 5.5

    def test_feasible_repeats_fractional_gauge_exact(self):
        # 每10厘米 22.5 针：width(r) = (2+6r)*10/22.5，r=1 时恰为 32/9
        self.assertEqual(feasible_repeats(Fraction(32, 9), 22.5, 1, 0, 3, 0.1),
                         (1, 1))


class TestSolveLayout(unittest.TestCase):
    def solve(self, targets, **kw):
        params = dict(gauge=20, edge=1, center=4,
                      pattern_width=6, tolerance=1)
        params.update(kw)
        return solve_layout(params['gauge'], targets, params['edge'],
                            params['center'], params['pattern_width'],
                            params['tolerance'])

    def test_exact_match(self):
        # width(r) = 3+6r：目标 27/33/39 恰好对应 r = 4/5/6
        res = self.solve([27, 33, 39])
        self.assertTrue(res['ok'])
        self.assertIsNone(res['failure'])
        self.assertEqual(res['repeats'], [4, 5, 6])
        self.assertEqual(res['cast_on'], [54, 66, 78])
        self.assertEqual(res['widths'],
                         [Fraction(27), Fraction(33), Fraction(39)])
        self.assertEqual(res['deviations'], [0, 0, 0])
        self.assertEqual(res['total_deviation'], 0)

    def test_minimizes_total_deviation(self):
        # 目标 26.5/33/38.5：最优 r = 4/5/6，偏差 0.5/0/0.5
        res = self.solve([26.5, 33, 38.5])
        self.assertTrue(res['ok'])
        self.assertEqual(res['repeats'], [4, 5, 6])
        self.assertEqual(res['deviations'],
                         [Fraction(1, 2), 0, Fraction(1, 2)])
        self.assertEqual(res['total_deviation'], 1)

    def test_strictly_increasing_forces_compromise(self):
        # width(r) = 2r；两尺码目标都是 5：各自最优都是 r ∈ {2,3}（偏差 1），
        # 严格递增下只能取 [2, 3]，总偏差 2
        res = self.solve([5, 5], edge=0, center=0,
                         pattern_width=2, tolerance=2)
        self.assertTrue(res['ok'])
        self.assertEqual(res['repeats'], [2, 3])
        self.assertEqual(res['total_deviation'], 2)

    def test_tie_breaks_to_lexicographically_smallest(self):
        # width(r) = 2r；目标 5/9、宽差 2.5：可行组合 [2,4]/[2,5]/[3,4]/[3,5]
        # 的总偏差都是 2，取按尺码顺序字典序最小的 [2, 4]
        res = self.solve([5, 9], edge=0, center=0,
                         pattern_width=2, tolerance=2.5)
        self.assertTrue(res['ok'])
        self.assertEqual(res['repeats'], [2, 4])
        self.assertEqual(res['total_deviation'], 2)

    def test_single_size_picks_smallest_best_repeat(self):
        # 单尺码目标 5、宽差 2.5：r=2 与 r=3 偏差同为 1，取最小的 2
        res = self.solve([5], edge=0, center=0,
                         pattern_width=2, tolerance=2.5)
        self.assertTrue(res['ok'])
        self.assertEqual(res['repeats'], [2])

    def test_infeasible_when_size_has_no_repeats(self):
        # width(r) = 3+6r 总是整数，目标 27.6±0.1 没有任何可用重复数
        res = self.solve([27.6], tolerance=0.1)
        self.assertFalse(res['ok'])
        self.assertEqual(res['failure']['kind'], 'empty')
        self.assertEqual(res['failure']['index'], 0)
        self.assertIsNone(res['repeats'])
        self.assertIsNone(res['total_deviation'])
        self.assertEqual(res['feasible'], [None])

    def test_infeasible_when_cannot_increase(self):
        # width(r) = 2r、宽差 0.6：两尺码目标都是 4，各自只能 r=2，
        # 无法构成严格递增序列
        res = self.solve([4, 4], edge=0, center=0,
                         pattern_width=2, tolerance=0.6)
        self.assertFalse(res['ok'])
        self.assertEqual(res['failure']['kind'], 'increasing')
        self.assertEqual(res['failure']['index'], 1)
        self.assertEqual(res['failure']['required'], 3)
        self.assertEqual(res['feasible'], [(2, 2), (2, 2)])

    def test_fractional_gauge_exact(self):
        # 每10厘米 22.5 针：width(1) = 8×10/22.5 = 32/9，精确匹配目标
        res = solve_layout(22.5, [Fraction(32, 9)], 1, 0, 3, 0.1)
        self.assertTrue(res['ok'])
        self.assertEqual(res['repeats'], [1])
        self.assertEqual(res['total_deviation'], 0)

    def test_string_and_float_inputs(self):
        # 十进制字符串/浮点输入按字面还原，与分数计算一致
        res = self.solve(['26.5', 33.0, '38.5'], gauge='20', tolerance='1')
        self.assertTrue(res['ok'])
        self.assertEqual(res['repeats'], [4, 5, 6])
        self.assertEqual(res['total_deviation'], 1)


if __name__ == '__main__':
    unittest.main()
