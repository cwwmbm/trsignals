import unittest

import pandas as pd

from api.indicator_catalog import list_indicators
from cross_pairs import (
    CROSS_PAIRS,
    cross_columns,
    cross_pairs_signature,
    down_column,
    down_prev_column,
    pair_columns,
    up_column,
    up_prev_column,
)
from indicator_sweep import PRICE_SWEEPS


class CrossFlagTests(unittest.TestCase):
    def setUp(self):
        # Close vs a flat KAMA: up on bar 2, down on bar 4, a touch on bar 6 keeps the side, up on bar 7.
        close = [9.0, 9.0, 11.0, 12.0, 8.0, 8.0, 10.0, 12.0]
        self.data = pd.DataFrame({"Close": close, "KAMA10": [10.0] * len(close)})
        self.pair = CROSS_PAIRS[[(p.left, p.right) for p in CROSS_PAIRS].index(("Close", "KAMA10"))]
        self.cols = cross_columns(self.data)

    def flags(self, column):
        return [i for i, v in enumerate(self.cols[column]) if v == 1]

    def test_today_flags(self):
        self.assertEqual(self.flags(up_column(self.pair)), [2, 7])
        self.assertEqual(self.flags(down_column(self.pair)), [4])

    def test_yesterday_flags_are_one_bar_later(self):
        self.assertEqual(self.flags(up_prev_column(self.pair)), [3])
        self.assertEqual(self.flags(down_prev_column(self.pair)), [5])

    def test_flags_are_plus_or_minus_one(self):
        for column in pair_columns(self.pair):
            self.assertTrue(set(self.cols[column].unique()) <= {1, -1})

    def test_first_bar_is_never_a_cross(self):
        # Nothing is known before the data starts, so bar 0 cannot be a cross; bar 1 can.
        cols = cross_columns(pd.DataFrame({"Close": [11.0, 9.0, 11.0], "KAMA10": [10.0] * 3}))
        self.assertEqual(cols[down_column(self.pair)].tolist(), [-1, 1, -1])
        self.assertEqual(cols[up_column(self.pair)].tolist(), [-1, -1, 1])

    def test_matches_the_builder_cross_operators(self):
        from api.strategy_compiler import _compare

        close, kama = self.data["Close"], self.data["KAMA10"]
        # A touch (close == KAMA on bar 6) is not a cross; the move off it on bar 7 is.
        self.assertEqual((self.cols[up_column(self.pair)] == 1).tolist(), _compare(close, "crosses above", kama).tolist())
        self.assertEqual((self.cols[down_column(self.pair)] == 1).tolist(), _compare(close, "crosses below", kama).tolist())

    def test_pairs_with_missing_inputs_are_skipped(self):
        self.assertNotIn("CrossUp_SMA50_SMA200", self.cols)


class CrossWiringTests(unittest.TestCase):
    def test_every_flag_is_swept_as_is_true_only(self):
        specs = {spec[0]: spec for spec in PRICE_SWEEPS}
        for pair in CROSS_PAIRS:
            for column in pair_columns(pair):
                self.assertEqual(specs[column][2:], ("more", 0, 0, 1))

    def test_every_flag_is_listed_as_a_flag(self):
        items = {i["id"]: i for i in list_indicators(builder_only=True, include_regimes=False)}
        for pair in CROSS_PAIRS:
            for column in pair_columns(pair):
                self.assertEqual(items[column]["category"], "Crosses")
                self.assertEqual(items[column]["valueType"], "flag")
        self.assertEqual(items["CrossUpPrev_Close_EMA20"]["label"], "Close crossed above EMA(20) yesterday")

    def test_pair_inputs_are_catalog_indicators(self):
        ids = {i["id"] for i in list_indicators(builder_only=False, include_regimes=False)}
        for pair in CROSS_PAIRS:
            self.assertIn(pair.left, ids)
            self.assertIn(pair.right, ids)

    def test_old_one_off_cross_flags_are_not_swept_twice(self):
        swept = {spec[0] for spec in PRICE_SWEEPS}
        self.assertFalse({"EMA8CrossUp", "EMA8CrossDown"} & swept)

    def test_signature_changes_with_the_pair_list(self):
        before = cross_pairs_signature()
        CROSS_PAIRS.append(CROSS_PAIRS[0]._replace(right="SMA10"))
        try:
            self.assertNotEqual(before, cross_pairs_signature())
        finally:
            CROSS_PAIRS.pop()

    def test_cache_meta_records_the_pair_signature(self):
        import inspect

        from api import market_data_cache

        self.assertIn("cross_pairs_signature", inspect.getsource(market_data_cache))


if __name__ == "__main__":
    unittest.main()
