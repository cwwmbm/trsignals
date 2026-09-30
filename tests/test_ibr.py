import unittest

import numpy as np
import pandas as pd

from indicators import internal_bar_range


class InternalBarRangeTests(unittest.TestCase):
    def test_one_bar_is_close_within_day_range(self):
        high = pd.Series([110.0, 108.0])
        low = pd.Series([100.0, 102.0])
        close = pd.Series([105.0, 104.0])
        result = internal_bar_range(high, low, close, period=1)
        np.testing.assert_allclose(result, [0.5, 1.0 / 3.0])

    def test_two_bar_uses_composite_candle(self):
        high = pd.Series([110.0, 108.0])
        low = pd.Series([100.0, 102.0])
        close = pd.Series([105.0, 104.0])
        result = internal_bar_range(high, low, close, period=2)
        self.assertTrue(np.isnan(result[0]))
        # Composite: high=110, low=100, close=104 → 0.4
        self.assertAlmostEqual(result[1], 0.4)
        daily_average = (0.5 + (104.0 - 102.0) / (108.0 - 102.0)) / 2
        self.assertNotAlmostEqual(result[1], daily_average)

    def test_three_bar_uses_composite_candle(self):
        high = pd.Series([110.0, 108.0, 112.0])
        low = pd.Series([100.0, 102.0, 101.0])
        close = pd.Series([105.0, 104.0, 107.0])
        result = internal_bar_range(high, low, close, period=3)
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        # Composite: high=112, low=100, close=107 → 7/12
        self.assertAlmostEqual(result[2], 7.0 / 12.0)

    def test_zero_range_composite_is_one(self):
        high = pd.Series([100.0, 100.0])
        low = pd.Series([100.0, 100.0])
        close = pd.Series([100.0, 100.0])
        result = internal_bar_range(high, low, close, period=2)
        self.assertAlmostEqual(result[1], 1.0)


if __name__ == "__main__":
    unittest.main()
