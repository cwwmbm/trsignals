import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import backtest as bt
import config
import stats
from api import fast_metrics as fm
from api import vehicles
from tests.test_fast_sweep import _frame


def _slow(frame, days, profit, is_long, *, pnl_column=None, include_yearly=True):
    executed = bt.execute_strategy(frame.copy(), days, profit, is_long, pnl_column=pnl_column)
    return bt._ranking_metrics(executed, include_yearly=include_yearly)


class RankingMetricsParityTests(unittest.TestCase):
    def _assert_same(self, frame, days, profit, is_long, **kwargs):
        slow = _slow(frame, days, profit, is_long, **kwargs)
        fast = fm.ranking_metrics(frame, days, profit, is_long, **kwargs)
        self.assertIsNotNone(fast)
        self.assertEqual(set(slow), set(fast))
        for key, expected in slow.items():
            got = fast[key]
            if key == "Yearly":
                self.assertEqual(expected, got)
            elif key in ("Trades",):
                self.assertEqual(int(expected), int(got))
            elif key == "CAGR":
                self.assertEqual(expected, got)
            else:
                self.assertTrue(np.isclose(float(expected), float(got), rtol=1e-9, atol=1e-9), (key, expected, got))

    def test_long_variants(self):
        for seed in range(6):
            for days, profit in ((1, 1), (3, 2), (5, 5), (2, 0)):
                with self.subTest(seed=seed, days=days, profit=profit):
                    self._assert_same(_frame(seed), days, profit, True)

    def test_short(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                self._assert_same(_frame(seed + 20), 3, 2, False)

    def test_hold_on_buy(self):
        original = bt.HoldOnBuySignal
        bt.HoldOnBuySignal = True
        try:
            for seed in range(4):
                with self.subTest(seed=seed):
                    self._assert_same(_frame(seed + 30), 3, 1, True)
        finally:
            bt.HoldOnBuySignal = original

    def test_proxy_pnl_column(self):
        self._assert_same(_frame(40), 3, 2, True, pnl_column="PROXY")

    def test_short_with_proxy(self):
        self._assert_same(_frame(41), 3, 2, False, pnl_column="PROXY")

    def test_entry_ready_trims_the_leading_bars(self):
        frame = _frame(42)
        frame["EntryReady"] = np.arange(len(frame)) >= 400
        self._assert_same(frame, 4, 2, True)

    def test_best_year_exclusion_off(self):
        with patch.object(stats, "ExcludeBestReturnYear", False):
            self._assert_same(_frame(43), 4, 2, True)

    def test_without_yearly(self):
        frame = _frame(44)
        fast = fm.ranking_metrics(frame, 3, 2, True, include_yearly=False)
        self.assertNotIn("Yearly", fast)
        slow = _slow(frame, 3, 2, True, include_yearly=False)
        self.assertEqual(set(slow), set(fast))

    def test_leverage_vehicle_stream(self):
        frame = _frame(45)
        frame.attrs["symbol"] = "SPY"
        rng = np.random.default_rng(1)
        vehicle = rng.normal(0.001, 0.02, len(frame))
        vehicle[:300] = np.nan
        stream = vehicles.TradedStream(
            symbol="SPY", vehicle_symbol="UPRO", leverage=3.0,
            synthetic=rng.normal(0.0008, 0.015, len(frame)), vehicle=vehicle,
            vehicle_ok=np.arange(len(frame)) >= 350, first_vehicle_date=frame["Date"].iloc[300],
        )
        with patch.object(vehicles, "stream_for", return_value=stream):
            self._assert_same(frame, 3, 2, True)

    def test_a_missing_close_marks_to_the_last_valid_price(self):
        frame = _frame(46)
        frame.loc[frame.index[700:704], "Close"] = np.nan
        frame["%Change"] = frame["%Change"].fillna(0)
        for is_long in (True, False):
            with self.subTest(is_long=is_long):
                self._assert_same(frame, 3, 2, is_long)


class FallbackTests(unittest.TestCase):
    def test_returns_none_when_the_real_path_should_decide(self):
        frame = _frame(50)
        self.assertIsNone(fm.ranking_metrics(frame, 0, 0, True))  # zero-day "og" strategy
        self.assertIsNone(fm.ranking_metrics(frame.head(2), 3, 2, True))
        with patch.object(fm.fast_sweep, "NUMBA_AVAILABLE", False):
            self.assertIsNone(fm.ranking_metrics(frame, 3, 2, True))

    def test_blown_up_account_falls_back(self):
        frame = _frame(51)
        frame["Buy"] = True
        frame["%Change"] = -0.9  # every held day loses 90% at 3x: equity goes non-positive
        self.assertIsNone(fm.ranking_metrics(frame, 3, 1000, True))

    def test_single_positive_year_falls_back(self):
        frame = _frame(52, bars=200)  # one calendar year: excluding the best year leaves nothing
        slow_raises = False
        try:
            _slow(frame, 3, 2, True)
        except Exception:
            slow_raises = True
        if fm.ranking_metrics(frame, 3, 2, True) is None:
            return
        self.assertFalse(slow_raises)


class WiringTests(unittest.TestCase):
    def test_indicator_threshold_matches_the_slow_row(self):
        frame = _frame(60)
        row = bt._run_indicator_threshold(frame, 4, 2, True, "A", "Buy", "less", 0.4, include_yearly=True)
        with patch.object(fm.fast_sweep, "NUMBA_AVAILABLE", False):
            slow = bt._run_indicator_threshold(frame, 4, 2, True, "A", "Buy", "less", 0.4, include_yearly=True)
        self.assertEqual(row["Trades"], slow["Trades"])
        self.assertEqual(row["Yearly"], slow["Yearly"])
        self.assertEqual(row["CAGR"], slow["CAGR"])
        self.assertAlmostEqual(row["Sharpe"], slow["Sharpe"], places=9)


if __name__ == "__main__":
    unittest.main()
