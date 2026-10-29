import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import getdata as dt
from api import scan_service as ss

CONTEXT = ["^VIX", "RSP", "SPY", "QQQ", "SMH", "XLF", "XLE", "XLU", "XLI", "GLD", "TLT", "SOXX"]


def _bulk(symbols=CONTEXT, bars=320, seed=0) -> pd.DataFrame:
    """A yfinance-shaped (field, symbol) frame ending on a recent business day."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=3), periods=bars)
    columns = {}
    for symbol in symbols:
        close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, bars))
        columns[("Close", symbol)] = close
        columns[("High", symbol)] = close * 1.01
        columns[("Low", symbol)] = close * 0.99
        columns[("Open", symbol)] = close * (1 + rng.normal(0, 0.002, bars))
        columns[("Volume", symbol)] = rng.integers(1_000_000, 5_000_000, bars).astype(float)
    frame = pd.DataFrame(columns, index=dates)
    frame.index.name = "Date"
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


class HurstSkipTests(unittest.TestCase):
    def test_skipping_hurst_changes_only_the_hurst_column(self):
        full = _bulk()
        slow = ss._prepare_symbol_frame(full.copy(), "SPY", "SPY", compute_hurst=True)
        fast = ss._prepare_symbol_frame(full.copy(), "SPY", "SPY", compute_hurst=False)
        self.assertEqual(list(slow.columns), list(fast.columns))
        self.assertTrue(slow["Hurst"].notna().any())
        self.assertTrue(fast["Hurst"].isna().all())
        other = [c for c in slow.columns if c != "Hurst"]
        pd.testing.assert_frame_equal(slow[other], fast[other])

    def test_default_still_computes_hurst(self):
        frame = ss._prepare_symbol_frame(_bulk(), "SPY", "SPY")
        self.assertTrue(frame["Hurst"].notna().any())

    def test_need_hurst_detection(self):
        def strategy(left, right="", sell=()):
            return SimpleNamespace(
                conditions=[{"left": left, "right": right}],
                sell_conditions=[{"left": item, "right": ""} for item in sell],
            )

        self.assertFalse(ss._strategies_need_hurst([strategy("RSI2", "15")]))
        self.assertTrue(ss._strategies_need_hurst([strategy("Hurst", "0.4")]))
        self.assertTrue(ss._strategies_need_hurst([strategy("Close", "Hurst")]))
        self.assertTrue(ss._strategies_need_hurst([strategy("RSI2", "15", sell=["Hurst"])]))
        self.assertFalse(ss._strategies_need_hurst([]))


class HolidayCalendarTests(unittest.TestCase):
    def setUp(self):
        dt._TRADING_DAYS_CACHE.clear()

    def test_calendar_is_built_once_per_date_range_with_same_result(self):
        dates = pd.bdate_range("2025-11-20", "2025-12-31")
        frame = pd.DataFrame({"Date": dates, "Close": 1.0})
        real = dt.mcal.get_calendar
        with patch.object(dt.mcal, "get_calendar", side_effect=real) as build:
            first = dt.clean_holidays(frame.copy())
            second = dt.clean_holidays(frame.copy())
        self.assertEqual(build.call_count, 1)
        pd.testing.assert_frame_equal(first, second)
        # Thanksgiving and Christmas are gone, ordinary days stay.
        self.assertNotIn(pd.Timestamp("2025-11-27"), set(first["Date"]))
        self.assertNotIn(pd.Timestamp("2025-12-25"), set(first["Date"]))
        self.assertIn(pd.Timestamp("2025-12-24"), set(first["Date"]))

    def test_different_ranges_do_not_share_an_entry(self):
        a = pd.DataFrame({"Date": pd.bdate_range("2025-01-02", "2025-01-31"), "Close": 1.0})
        b = pd.DataFrame({"Date": pd.bdate_range("2025-02-03", "2025-02-28"), "Close": 1.0})
        dt.clean_holidays(a)
        dt.clean_holidays(b)
        self.assertEqual(len(dt._TRADING_DAYS_CACHE), 2)

    def test_cache_is_bounded(self):
        with patch.object(dt, "_TRADING_DAYS_CACHE_SIZE", 3):
            for offset in range(6):
                start = pd.Timestamp("2025-01-02") + pd.Timedelta(days=offset)
                dt.clean_holidays(pd.DataFrame({"Date": pd.bdate_range(start, periods=5), "Close": 1.0}))
        self.assertLessEqual(len(dt._TRADING_DAYS_CACHE), 3)


class ScanCacheTests(unittest.TestCase):
    def setUp(self):
        ss.clear_scan_cache()
        self.prepared = 0
        self.bulk = _bulk(["SPY"], bars=30)

        def fake_prepare(full, symbol, yf_symbol, **kwargs):
            self.prepared += 1
            return pd.DataFrame({"Date": [pd.Timestamp("2025-01-02")], "Close": [1.0]})

        self.patches = [
            patch.object(ss, "list_strategies", return_value=[]),
            patch.object(ss, "list_portfolios", return_value=[]),
            patch.object(ss, "_prepare_symbol_frame", side_effect=fake_prepare),
            patch.object(ss, "_legacy_scan_row", side_effect=lambda symbol, signal, data: {"symbol": symbol, "signal": str(signal)}),
            patch.object(ss.dt, "get_bulk_data", side_effect=lambda *a, **k: self.bulk.copy()),
        ]
        for item in self.patches:
            item.start()
        self.addCleanup(lambda: [item.stop() for item in self.patches])
        self.addCleanup(ss.clear_scan_cache)

    def test_second_scan_with_same_inputs_is_served_from_cache(self):
        first = ss.run_scan()
        used = self.prepared
        self.assertGreater(used, 0)
        second = ss.run_scan()
        self.assertEqual(self.prepared, used)
        self.assertEqual(first, second)

    def test_cached_rows_are_copies(self):
        first = ss.run_scan()
        first[0]["symbol"] = "MUTATED"
        second = ss.run_scan()
        second_again = ss.run_scan()
        self.assertNotEqual(second[0]["symbol"], "MUTATED")
        second[0]["symbol"] = "ALSO"
        self.assertNotEqual(second_again[0]["symbol"], "ALSO")

    def test_download_float_noise_does_not_invalidate(self):
        ss.run_scan()
        used = self.prepared
        noisy = self.bulk * (1 + 3e-8)
        self.bulk = noisy
        ss.run_scan()
        self.assertEqual(self.prepared, used)

    def test_a_real_price_change_invalidates(self):
        ss.run_scan()
        used = self.prepared
        changed = self.bulk.copy()
        changed.iloc[-1] = changed.iloc[-1] * 1.001
        self.bulk = changed
        ss.run_scan()
        self.assertGreater(self.prepared, used)

    def test_a_three_percent_intraday_move_in_the_last_bar_invalidates(self):
        ss.run_scan()
        used = self.prepared
        moved = self.bulk.copy()
        moved.iloc[-1] = moved.iloc[-1] * 1.03
        self.bulk = moved
        ss.run_scan()
        self.assertGreater(self.prepared, used)

    def test_a_new_bar_invalidates(self):
        ss.run_scan()
        used = self.prepared
        extra = self.bulk.iloc[[-1]].copy()
        extra.index = extra.index + pd.Timedelta(days=1)
        self.bulk = pd.concat([self.bulk, extra])
        ss.run_scan()
        self.assertGreater(self.prepared, used)

    def test_changed_strategies_invalidate(self):
        ss.run_scan()
        used = self.prepared
        extra = SimpleNamespace(
            id="x", name="X", symbol="ZZZ", direction="long", hold_days=1, profit=1, description="",
            proxy_symbol=None, confirm_symbols=[], conditions=[], sell_conditions=[],
        )
        with patch.object(ss, "list_strategies", return_value=[extra]), patch.object(ss, "_builder_scan_row", return_value={"symbol": "ZZZ", "signal": "X"}):
            ss.run_scan()
        self.assertGreater(self.prepared, used)

    def test_changed_leverage_settings_invalidate(self):
        ss.run_scan()
        used = self.prepared
        with patch.object(ss, "_file_stamp", return_value=(1, 2)):
            ss.run_scan()
        self.assertGreater(self.prepared, used)

    def test_unreadable_frames_never_match(self):
        self.assertFalse(ss._frames_match(pd.DataFrame({"a": ["x"]}), pd.DataFrame({"a": ["x"]})))

    def test_expires_after_the_ttl(self):
        with patch.object(ss.time, "monotonic", return_value=1000.0):
            ss.run_scan()
        used = self.prepared
        with patch.object(ss.time, "monotonic", return_value=1000.0 + ss.SCAN_CACHE_TTL_SECONDS - 1):
            ss.run_scan()
        self.assertEqual(self.prepared, used)
        with patch.object(ss.time, "monotonic", return_value=1000.0 + ss.SCAN_CACHE_TTL_SECONDS + 1):
            ss.run_scan()
        self.assertGreater(self.prepared, used)

    def test_calls_with_supplied_frames_bypass_the_cache(self):
        frames = {"SPY": pd.DataFrame({"Date": [pd.Timestamp("2025-01-02")], "Close": [1.0]})}
        ss.run_scan(symbol_frames=frames)
        ss.run_scan(symbol_frames=frames)
        self.assertEqual(self.prepared, 0)
        self.assertIsNone(ss._scan_cache["signature"])


if __name__ == "__main__":
    unittest.main()
