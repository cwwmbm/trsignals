import math
import unittest

import numpy as np
import pandas as pd

from api.indicator_catalog import is_market_wide_indicator, list_indicators
from indicator_sweep import PRICE_SWEEPS
from indicators import INDICATOR_CACHE_VERSION, squeeze_age


class SqueezeAgeTests(unittest.TestCase):
    def test_counts_up_inside_and_down_after_release(self):
        flags = pd.Series([-1, -1, 1, 1, 1, -1, -1, 1, -1])
        age = squeeze_age(flags)
        self.assertTrue(age.iloc[:2].isna().all())
        self.assertEqual(list(age.iloc[2:]), [1, 2, 3, -1, -2, 1, -1])

    def test_never_squeezed_is_blank(self):
        self.assertTrue(squeeze_age(pd.Series([-1, -1, -1])).isna().all())

    def test_keeps_the_index(self):
        index = pd.RangeIndex(10, 13)
        self.assertTrue(squeeze_age(pd.Series([1, -1, 1], index=index)).index.equals(index))

    def test_cache_version_was_bumped(self):
        self.assertGreaterEqual(INDICATOR_CACHE_VERSION, 6)


class SqueezeWiringTests(unittest.TestCase):
    COLUMNS = ["SqueezeRatio20", "SqueezeAge20"]

    def test_swept(self):
        swept = {spec[0] for spec in PRICE_SWEEPS}
        for column in self.COLUMNS:
            self.assertIn(column, swept)

    def test_listed_with_a_typical_range(self):
        items = {item["id"]: item for item in list_indicators(builder_only=True, include_regimes=False)}
        for column in self.COLUMNS:
            self.assertIn(column, items)
            self.assertEqual(items[column]["category"], "Volatility / risk")
            self.assertIn("typicalRange", items[column])
            self.assertFalse(is_market_wide_indicator(column))


if __name__ == "__main__":
    unittest.main()
