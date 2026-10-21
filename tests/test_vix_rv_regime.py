import math
import unittest

import numpy as np
import pandas as pd

from api import market_regimes as m


class ClassifyVixRvTests(unittest.TestCase):
    def test_missing_is_unlabeled(self):
        self.assertIsNone(m.classify_vix_rv(None))
        self.assertIsNone(m.classify_vix_rv(float("nan")))

    def test_five_buckets_with_shared_edges_in_the_lower_bucket(self):
        expected = {
            0: "le_10", 10: "le_10", 10.01: "10_30", 30: "10_30", 30.01: "30_70",
            70: "30_70", 70.01: "70_90", 90: "70_90", 90.01: "gt_90", 100: "gt_90",
        }
        for value, key in expected.items():
            self.assertEqual(m.classify_vix_rv(value), key, value)
        self.assertEqual(m.VIX_RV_KEYS, ("le_10", "10_30", "30_70", "70_90", "gt_90"))


class ExpandingPercentileTests(unittest.TestCase):
    def test_ranks_against_earlier_readings_only(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 2.5])
        ranks = m._expanding_percentile(series, min_history=2)
        self.assertTrue(math.isnan(ranks.iloc[0]))
        self.assertTrue(math.isnan(ranks.iloc[1]))
        self.assertAlmostEqual(ranks.iloc[2], 100.0)        # 3 vs [1, 2]
        self.assertAlmostEqual(ranks.iloc[3], 100.0)        # 4 vs [1, 2, 3]
        self.assertAlmostEqual(ranks.iloc[4], 50.0)         # 2.5 vs [1, 2, 3, 4]: 2 of 4 at or below

    def test_a_later_value_never_changes_an_earlier_rank(self):
        rng = np.random.default_rng(1)
        values = pd.Series(rng.normal(size=400))
        full = m._expanding_percentile(values, min_history=50)
        truncated = m._expanding_percentile(values.iloc[:300], min_history=50)
        pd.testing.assert_series_equal(full.iloc[:300], truncated)

    def test_missing_values_are_skipped_not_ranked(self):
        series = pd.Series([1.0, np.nan, 2.0, 3.0])
        ranks = m._expanding_percentile(series, min_history=2)
        self.assertTrue(math.isnan(ranks.iloc[1]))
        self.assertAlmostEqual(ranks.iloc[3], 100.0)        # history [1, 2], the NaN joined nothing

    def test_default_waits_for_a_year_of_history(self):
        ranks = m._expanding_percentile(pd.Series(np.arange(300, dtype=float)))
        self.assertTrue(ranks.iloc[:252].isna().all())
        self.assertTrue(ranks.iloc[252:].notna().all())


def _calendar(periods=400, vix_level=20.0, seed=3):
    idx = pd.bdate_range("2020-01-01", periods=periods)
    rng = np.random.default_rng(seed)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, periods))), index=idx)
    vix = pd.Series(vix_level, index=idx)
    return m.build_regime_calendar(vix, vix, spy, spy), idx, spy, vix


class VixRvCalendarTests(unittest.TestCase):
    def test_ratio_is_vix_over_annualized_21_day_realized_volatility(self):
        calendar, idx, spy, vix = _calendar()
        realized = np.log(spy).diff().rolling(21).std() * math.sqrt(252) * 100
        self.assertTrue(calendar["vix_rv_ratio"].iloc[:21].isna().all())
        self.assertAlmostEqual(calendar["vix_rv_ratio"].iloc[100], vix.iloc[100] / realized.iloc[100])

    def test_zero_realized_volatility_is_blank_not_infinite(self):
        idx = pd.bdate_range("2020-01-01", periods=60)
        flat = pd.Series(100.0, index=idx)
        calendar = m.build_regime_calendar(pd.Series(20.0, index=idx), flat, flat, flat)
        self.assertFalse(np.isinf(calendar["vix_rv_ratio"]).any())
        self.assertTrue(calendar["vix_rv_ratio"].isna().all())

    def test_labels_start_after_the_percentile_history_and_cover_every_bucket(self):
        calendar, idx, _, _ = _calendar(periods=900, seed=5)
        labels = m._regime_label_sets(pd.DataFrame({"Date": idx}), calendar)["vix_rv"]
        first = next(i for i, value in enumerate(labels) if value is not None)
        self.assertGreaterEqual(first, 21 + m.VIX_RV_MIN_HISTORY)
        self.assertEqual(set(labels.dropna()), set(m.VIX_RV_KEYS))

    def test_a_calmer_market_than_ever_before_reads_as_the_top_bucket(self):
        # Volatility keeps falling under a constant VIX, so the ratio hits a new high every day.
        periods = 400
        idx = pd.bdate_range("2020-01-01", periods=periods)
        rng = np.random.default_rng(7)
        scale = np.linspace(0.02, 0.002, periods)
        spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 1, periods) * scale)), index=idx)
        calendar = m.build_regime_calendar(pd.Series(20.0, index=idx), pd.Series(20.0, index=idx), spy, spy)
        tail = m._regime_label_sets(pd.DataFrame({"Date": idx}), calendar)["vix_rv"].iloc[-30:]
        self.assertEqual(set(tail), {"gt_90"})

    def test_trade_record_current_regime_and_analysis(self):
        calendar, idx, _, _ = _calendar(periods=900, seed=5)
        record = m.market_regimes_for_timestamp(idx[-1], calendar)
        self.assertIn(record["vix_rv_regime"], m.VIX_RV_KEYS)
        self.assertIsNone(m.market_regimes_for_timestamp(idx[5], calendar)["vix_rv_regime"])
        current = m.current_market_regimes(pd.DataFrame({"Date": idx}), calendar)
        self.assertEqual(current["regimes"]["vix_rv"], record["vix_rv_regime"])
        self.assertAlmostEqual(current["readings"]["vix_rv"], calendar["vix_rv_pctile"].iloc[-1])
        rows = m.analyze_market_regime("vix_rv", calendar)["rows"]
        self.assertEqual(sum(row["total_days"] for row in rows), calendar["vix_rv_pctile"].notna().sum())
        for row in rows:
            self.assertEqual(sum(row["days"].values()), row["total_days"])

    def test_flags_exist_for_every_bucket(self):
        for key in m.VIX_RV_KEYS:
            self.assertTrue(m.is_regime_indicator_id(m.regime_indicator_id("vix_rv", key)))


if __name__ == "__main__":
    unittest.main()
