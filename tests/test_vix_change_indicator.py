import math
import unittest

import numpy as np
import pandas as pd

from api.custom_data import unavailable_indicator_ids
from api.indicator_catalog import is_market_wide_indicator, list_indicators
from api.strategy_compiler import compile_buy_mask
from indicators import INDICATOR_CACHE_VERSION, vix_daily_change_pct


class VixDailyChangeTests(unittest.TestCase):
    def test_is_the_close_to_close_percent_change_in_percent_points(self):
        change = vix_daily_change_pct(pd.Series([20.0, 22.0, 19.8]))
        self.assertTrue(math.isnan(change.iloc[0]))
        self.assertAlmostEqual(change.iloc[1], 10.0)
        self.assertAlmostEqual(change.iloc[2], -10.0)

    def test_a_missing_close_is_not_filled_across(self):
        change = vix_daily_change_pct(pd.Series([20.0, np.nan, 25.0, 30.0]))
        self.assertTrue(math.isnan(change.iloc[1]))
        # The bar after the gap reads the change since the last VIX close.
        self.assertAlmostEqual(change.iloc[2], 25.0)
        self.assertAlmostEqual(change.iloc[3], 20.0)

    def test_keeps_the_index_and_ignores_an_all_blank_column(self):
        index = pd.date_range("2024-01-01", periods=3)
        change = vix_daily_change_pct(pd.Series([np.nan, np.nan, np.nan], index=index))
        self.assertTrue(change.index.equals(index))
        self.assertTrue(change.isna().all())

    def test_cache_version_was_bumped_for_the_new_column(self):
        self.assertGreaterEqual(INDICATOR_CACHE_VERSION, 3)


class VixChangeCatalogTests(unittest.TestCase):
    def test_listed_next_to_vix_in_the_entry_condition_dropdown(self):
        items = {item["id"]: item for item in list_indicators(builder_only=True, include_regimes=False)}
        self.assertIn("VixChange", items)
        self.assertEqual(items["VixChange"]["category"], "Volatility / risk")
        self.assertEqual(items["VixChange"]["label"], "VIX daily change")
        self.assertEqual(items["VixChange"]["category"], items["Vix"]["category"])

    def test_is_a_market_wide_filter_like_vix(self):
        self.assertTrue(is_market_wide_indicator("VixChange"))

    def test_is_unavailable_for_custom_datasets(self):
        self.assertIn("VixChange", unavailable_indicator_ids())

    def test_conditions_on_the_overlapping_thresholds_compile(self):
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-02", periods=5, freq="B"),
                "VixChange": [np.nan, -16.0, -10.0, 3.0, 21.0],
            }
        )
        def mask(operator, value):
            return compile_buy_mask(
                data, [{"left": "VixChange", "operator": operator, "right": str(value), "logic": "AND"}]
            ).tolist()

        self.assertEqual(mask("<=", -15), [False, True, False, False, False])
        self.assertEqual(mask("<=", -10), [False, True, True, False, False])
        self.assertEqual(mask(">=", 0), [False, False, False, True, True])
        self.assertEqual(mask(">=", 20), [False, False, False, False, True])


if __name__ == "__main__":
    unittest.main()
