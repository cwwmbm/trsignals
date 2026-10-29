import math
import unittest

import numpy as np
import pandas as pd
import ta

from api.indicator_catalog import is_compare_indicator_allowed, is_market_wide_indicator, list_indicators
from api.strategy_compiler import compile_buy_mask
from indicator_sweep import PRICE_SWEEPS
from indicators import INDICATOR_CACHE_VERSION, directional_indicators

DI_COLUMNS = ["DIPlus14", "DIMinus14", "DIBalance14", "DICrossAge14", "SignedADX14"]


def _ohlc(closes):
    close = pd.Series(closes, dtype=float)
    return close + 1.0, close - 1.0, close


def _trend_then_reversal():
    # 40 bars up, then 40 bars down: one clear bearish DI cross in the middle.
    closes = list(np.linspace(100, 140, 40)) + list(np.linspace(139, 99, 40))
    return _ohlc(closes)


class DirectionalIndicatorTests(unittest.TestCase):
    def setUp(self):
        self.high, self.low, self.close = _trend_then_reversal()
        self.di = directional_indicators(self.high, self.low, self.close)

    def test_returns_the_five_columns(self):
        self.assertEqual(sorted(self.di), sorted(DI_COLUMNS))

    def test_dis_match_ta_once_warmed_up(self):
        ref = ta.trend.ADXIndicator(self.high, self.low, self.close, window=14)
        valid = self.di["DIPlus14"].notna()
        self.assertTrue(valid.iloc[-1])
        np.testing.assert_allclose(self.di["DIPlus14"][valid], ref.adx_pos()[valid])
        np.testing.assert_allclose(self.di["DIMinus14"][valid], ref.adx_neg()[valid])

    def test_warm_up_is_blank_not_zero(self):
        for column in DI_COLUMNS:
            self.assertTrue(math.isnan(self.di[column].iloc[0]), column)
        self.assertFalse((self.di["DIPlus14"].dropna() == 0).all())

    def test_balance_sign_follows_the_leading_di(self):
        plus, minus, balance = self.di["DIPlus14"], self.di["DIMinus14"], self.di["DIBalance14"]
        valid = balance.notna()
        np.testing.assert_array_equal(np.sign(balance[valid]), np.sign((plus - minus)[valid]))
        self.assertTrue(balance[valid].between(-100, 100).all())
        self.assertGreater(balance.iloc[35], 0)
        self.assertLess(balance.iloc[-1], 0)

    def test_signed_adx_is_adx_with_the_leading_side(self):
        signed = self.di["SignedADX14"]
        adx = ta.trend.ADXIndicator(self.high, self.low, self.close, window=14).adx()
        valid = signed.notna()
        np.testing.assert_allclose(signed[valid].abs(), adx[valid])
        self.assertGreater(signed.iloc[39], 0)
        self.assertLess(signed.iloc[-1], 0)

    def test_cross_age_counts_days_since_the_cross(self):
        age = self.di["DICrossAge14"]
        balance = self.di["DIBalance14"]
        # The opening uptrend began before the data, so its age is unknown.
        self.assertTrue(age.iloc[:40].isna().all())
        cross = int(np.flatnonzero((balance < 0).to_numpy())[0])
        self.assertEqual(age.iloc[cross], -1)
        self.assertEqual(age.iloc[cross + 4], -5)
        self.assertEqual(age.iloc[-1], -(len(age) - cross))

    def test_a_tie_keeps_the_previous_side(self):
        # Flat bars after a rise leave +DI and -DI equal-weighted only by decay, never a new side.
        high, low, close = _ohlc(list(np.linspace(100, 120, 30)) + [120.0] * 10 + list(np.linspace(121, 90, 30)))
        age = directional_indicators(high, low, close)["DICrossAge14"]
        values = age.dropna().to_numpy()
        # Within a run the magnitude only ever grows by one.
        for prev, cur in zip(values, values[1:]):
            if np.sign(prev) == np.sign(cur):
                self.assertEqual(abs(cur), abs(prev) + 1)
            else:
                self.assertEqual(abs(cur), 1)

    def test_cache_version_was_bumped_for_the_new_columns(self):
        self.assertGreaterEqual(INDICATOR_CACHE_VERSION, 5)


class DirectionalIndicatorWiringTests(unittest.TestCase):
    def test_every_column_is_swept(self):
        swept = {spec[0] for spec in PRICE_SWEEPS}
        for column in DI_COLUMNS:
            self.assertIn(column, swept)

    def test_listed_next_to_adx_with_a_typical_range(self):
        items = {item["id"]: item for item in list_indicators(builder_only=True, include_regimes=False)}
        for column in DI_COLUMNS:
            self.assertIn(column, items)
            self.assertEqual(items[column]["category"], items["ADX14"]["category"])
            self.assertIn("typicalRange", items[column])

    def test_symbol_specific_like_adx(self):
        for column in DI_COLUMNS:
            self.assertFalse(is_market_wide_indicator(column))

    def test_plus_di_above_minus_di_compiles(self):
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-02", periods=3, freq="B"),
                "DIPlus14": [20.0, 30.0, np.nan],
                "DIMinus14": [25.0, 15.0, 20.0],
            }
        )
        mask = compile_buy_mask(
            data, [{"left": "DIPlus14", "operator": ">", "right": "DIMinus14", "logic": "AND"}]
        )
        self.assertEqual(list(np.asarray(mask, dtype=bool)), [False, True, False])

    def test_dis_compare_only_with_each_other(self):
        self.assertTrue(is_compare_indicator_allowed("DIPlus14", "DIMinus14"))
        self.assertTrue(is_compare_indicator_allowed("DIMinus14", "DIPlus14"))
        self.assertFalse(is_compare_indicator_allowed("DIPlus14", "RSI14"))
        self.assertFalse(is_compare_indicator_allowed("DIPlus14", "Close"))
        self.assertFalse(is_compare_indicator_allowed("DIBalance14", "DIPlus14"))
        with self.assertRaises(ValueError):
            compile_buy_mask(
                pd.DataFrame({"DIPlus14": [1.0], "RSI14": [2.0]}),
                [{"left": "DIPlus14", "operator": ">", "right": "RSI14", "logic": "AND"}],
            )

    def test_dis_still_take_numeric_thresholds(self):
        data = pd.DataFrame({"DIPlus14": [20.0, 30.0]})
        mask = compile_buy_mask(data, [{"left": "DIPlus14", "operator": ">=", "right": "25", "logic": "AND"}])
        self.assertEqual(list(np.asarray(mask, dtype=bool)), [False, True])


if __name__ == "__main__":
    unittest.main()
