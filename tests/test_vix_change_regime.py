import math
import unittest

import numpy as np
import pandas as pd

from api import market_regimes as m


def _calendar(vix_values):
    idx = pd.bdate_range("2024-01-01", periods=len(vix_values))
    vix = pd.Series(vix_values, index=idx, dtype=float)
    spy = pd.Series(100.0, index=idx)
    return m.build_regime_calendar(vix, vix, spy, spy), idx


class ClassifyVixChangeTests(unittest.TestCase):
    def test_missing_values_are_unlabeled(self):
        self.assertIsNone(m.classify_vix_change(None))
        self.assertIsNone(m.classify_vix_change(float("nan")))

    def test_keys_are_the_requested_overlapping_buckets(self):
        self.assertEqual(
            m.VIX_CHANGE_KEYS,
            ("le_neg_15", "le_neg_10", "le_neg_5", "le_0", "ge_0", "ge_5", "ge_10", "ge_15", "ge_20"),
        )

    def test_a_day_is_in_every_bucket_it_satisfies(self):
        self.assertEqual(m.classify_vix_change(-12), ("le_neg_10", "le_neg_5", "le_0"))
        self.assertEqual(m.classify_vix_change(-20), ("le_neg_15", "le_neg_10", "le_neg_5", "le_0"))
        self.assertEqual(m.classify_vix_change(2), ("ge_0",))
        self.assertEqual(m.classify_vix_change(-2), ("le_0",))
        self.assertEqual(m.classify_vix_change(25), ("ge_0", "ge_5", "ge_10", "ge_15", "ge_20"))

    def test_edges_are_inclusive_and_zero_is_in_both_sides(self):
        self.assertEqual(m.classify_vix_change(-15), ("le_neg_15", "le_neg_10", "le_neg_5", "le_0"))
        self.assertNotIn("le_neg_15", m.classify_vix_change(-14.99))
        self.assertIn("ge_20", m.classify_vix_change(20))
        self.assertNotIn("ge_20", m.classify_vix_change(19.99))
        self.assertEqual(m.classify_vix_change(0), ("le_0", "ge_0"))

    def test_every_finite_value_lands_in_a_bucket(self):
        for value in np.linspace(-60, 120, 361):
            self.assertTrue(m.classify_vix_change(float(value)))


class VixChangeCalendarTests(unittest.TestCase):
    def test_column_is_the_close_to_close_percent_change(self):
        calendar, _ = _calendar([20, 22, 19.8])
        self.assertTrue(math.isnan(calendar["vix_change"].iloc[0]))
        self.assertAlmostEqual(calendar["vix_change"].iloc[1], 10.0)
        self.assertAlmostEqual(calendar["vix_change"].iloc[2], -10.0)

    def test_a_missing_close_blanks_that_day_and_the_next_without_bridging(self):
        calendar, _ = _calendar([20, np.nan, 25, 30])
        change = calendar["vix_change"]
        self.assertTrue(math.isnan(change.iloc[1]))
        self.assertTrue(math.isnan(change.iloc[2]))
        self.assertAlmostEqual(change.iloc[3], 20.0)

    def test_labels_hold_a_tuple_of_keys_per_day(self):
        calendar, idx = _calendar([20, 22, 19.36, 19.36])
        labels = m._regime_label_sets(pd.DataFrame({"Date": idx}), calendar)["vix_change"]
        self.assertIsNone(labels.iloc[0])
        self.assertEqual(labels.iloc[1], ("ge_0", "ge_5", "ge_10"))
        self.assertEqual(labels.iloc[2], ("le_neg_10", "le_neg_5", "le_0"))
        self.assertEqual(labels.iloc[3], ("le_0", "ge_0"))

    def test_trade_record_field_is_a_json_friendly_list(self):
        calendar, idx = _calendar([20, 22])
        record = m.market_regimes_for_timestamp(idx[1], calendar)
        self.assertEqual(record["vix_change_regime"], ["ge_0", "ge_5", "ge_10"])
        self.assertIsNone(m.market_regimes_for_timestamp(idx[0], calendar)["vix_change_regime"])

    def test_flags_are_one_inside_minus_one_outside_zero_unlabeled(self):
        calendar, idx = _calendar([20, 22, 19.36])
        labels = m._regime_label_sets(pd.DataFrame({"Date": idx}), calendar)["vix_change"]
        self.assertEqual(m._flag_series(labels, "ge_5").tolist(), [0, 1, -1])
        self.assertEqual(m._flag_series(labels, "le_0").tolist(), [0, -1, 1])

    def test_indicator_ids_exist_for_every_bucket(self):
        for key in m.VIX_CHANGE_KEYS:
            self.assertTrue(m.is_regime_indicator_id(m.regime_indicator_id("vix_change", key)))

    def test_current_regime_lists_all_buckets_and_the_reading(self):
        calendar, idx = _calendar([20, 22, 19.36, 21])
        current = m.current_market_regimes(pd.DataFrame({"Date": idx}), calendar)
        self.assertEqual(current["regimes"]["vix_change"], ["ge_0", "ge_5"])
        self.assertAlmostEqual(current["readings"]["vix_change"], (21 / 19.36 - 1) * 100)


class VixChangeAnalysisTests(unittest.TestCase):
    def test_days_count_once_per_bucket_and_total_days_once(self):
        calendar, _ = _calendar([20, 22, 19.36, 19.36])
        rows = m.analyze_market_regime("vix_change", calendar)["rows"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["total_days"], 3)
        self.assertEqual(row["days"]["ge_10"], 1)
        self.assertEqual(row["days"]["le_neg_10"], 1)
        self.assertEqual(row["days"]["le_0"], 2)
        self.assertEqual(row["days"]["ge_0"], 2)
        self.assertGreater(sum(row["days"].values()), row["total_days"])

    def test_coverage_starts_on_the_first_defined_change(self):
        calendar, idx = _calendar([20, 22, 19.36])
        coverage = m.regime_coverage(pd.DataFrame({"Date": idx}), calendar)["vix_change"]
        self.assertEqual(coverage, {"start": idx[1].strftime("%Y-%m-%d"), "end": idx[2].strftime("%Y-%m-%d")})


def _strategy_frame(idx, entry_positions):
    """One-bar-hold longs: enter on each entry bar, hold the next bar, exit on the one after."""
    n = len(idx)
    long_in = np.zeros(n, bool)
    hold = np.zeros(n, bool)
    long_out = np.zeros(n, bool)
    pnl = np.full(n, np.nan)
    for position in entry_positions:
        long_in[position] = True
        hold[position + 1] = True
        long_out[position + 1] = True
        pnl[position + 1] = 0.02
    equity = np.cumprod(np.where(hold, 1.02, 1.0))
    return pd.DataFrame(
        {
            "Date": idx,
            "RollingPnL": equity,
            "LongTradeIn": long_in,
            "HoldLong": hold,
            "LongTradeOut": long_out,
            "TradePnL": pnl,
            "DaysInTrade": np.where(long_out, 1.0, np.nan),
        }
    )


class VixChangeTradeTests(unittest.TestCase):
    def setUp(self):
        # Changes: nan, +10%, 0, -12%, 0, +13.6%, 0. Trades enter on the +10% and -12% days.
        self.calendar, self.idx = _calendar([20, 22, 22, 19.36, 19.36, 22, 22])
        self.frame = _strategy_frame(self.idx, entry_positions=[1, 3])

    def test_a_trade_counts_in_every_bucket_its_entry_day_is_in(self):
        rows = {
            row["key"]: row
            for row in m.compute_market_regime_sharpe(self.frame, self.calendar)["vix_change"]
        }
        self.assertEqual(set(rows), set(m.VIX_CHANGE_KEYS))
        # Entry on the +10% day: ge_0, ge_5, ge_10. Entry on the -12% day: le_neg_10, le_neg_5, le_0.
        for key in ("ge_0", "ge_5", "ge_10", "le_neg_10", "le_neg_5", "le_0"):
            self.assertEqual(rows[key]["trades"], 1, key)
        for key in ("le_neg_15", "ge_15", "ge_20"):
            self.assertEqual(rows[key]["trades"], 0, key)

    def test_year_rows_count_a_trade_in_each_of_its_buckets_but_once_in_the_total(self):
        calendar_frame = self.frame
        analysis = m.analyze_strategy_regime(calendar_frame, self.calendar, "vix_change")
        row = analysis["rows"][0]
        self.assertEqual(row["total_trades"], 2)
        self.assertEqual(row["trades"]["le_0"]["trades"], 1)
        self.assertEqual(row["trades"]["ge_0"]["trades"], 1)
        self.assertEqual(row["trades"]["le_neg_5"]["trades"], 1)
        states = {state["key"]: state for state in analysis["granular"]["states"]}
        self.assertEqual(states["le_neg_10"]["days"], 1)
        self.assertEqual(states["ge_10"]["days"], 2)
        self.assertEqual(states["le_0"]["days"], 4)
        self.assertEqual(states["ge_0"]["days"], 5)

    def test_book_scoring_covers_the_new_regime(self):
        book = m.score_regime_book(self.frame, self.calendar)
        self.assertEqual(set(book["regimes"]["vix_change"]), set(m.VIX_CHANGE_KEYS))


if __name__ == "__main__":
    unittest.main()
