import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from api import vehicle_data as vd

EASTERN = ZoneInfo("America/New_York")
REAL_SESSION_OPEN = vd.session_open


def _series(last_close: float, end: str = "2026-10-21") -> pd.Series:
    index = pd.bdate_range(end=end, periods=6)
    return pd.Series([90.0, 91.0, 92.0, 97.55, 98.65, last_close], index=index)


class IntradayTailTests(unittest.TestCase):
    def setUp(self):
        vd.clear_memory()
        self.addCleanup(vd.clear_memory)
        self.clock = 1000.0
        self.tail_calls = 0
        self.tail_close = 100.93
        self.today = datetime(2026, 10, 21, 10, 30, tzinfo=EASTERN)

        def tail(symbol):
            self.tail_calls += 1
            return pd.Series([98.65, self.tail_close], index=pd.to_datetime(["2026-10-20", "2026-10-21"]))

        self.patches = [
            patch.object(vd, "download_close", side_effect=lambda symbol: _series(100.93)),
            patch.object(vd, "download_tail", side_effect=tail),
            patch.object(vd, "_today", return_value=self.today.date()),
            patch.object(vd, "session_open", return_value=True),
            patch.object(vd.time, "monotonic", side_effect=lambda: self.clock),
            patch.object(vd, "_fresh", return_value=False),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_todays_bar_moves_during_the_session_but_not_more_than_once_a_minute(self):
        first = vd.load_close("SOXL")
        self.assertEqual(first.iloc[-1], 100.93)
        calls = self.tail_calls  # the first load already refreshed the tail once
        self.tail_close = 104.07
        self.clock += vd.INTRADAY_REFRESH_SECONDS - 1
        self.assertEqual(vd.load_close("SOXL").iloc[-1], 100.93)
        self.assertEqual(self.tail_calls, calls)
        self.clock += 2
        self.assertEqual(vd.load_close("SOXL").iloc[-1], 104.07)
        self.assertEqual(self.tail_calls, calls + 1)

    def test_history_is_kept_and_not_duplicated(self):
        vd.load_close("SOXL")
        self.tail_close = 104.07
        self.clock += 120
        series = vd.load_close("SOXL")
        self.assertTrue(series.index.is_unique and series.index.is_monotonic_increasing)
        self.assertEqual(len(series), 6)
        self.assertEqual(series.iloc[0], 90.0)

    def test_a_cache_that_ends_yesterday_gets_todays_bar(self):
        with patch.object(vd, "download_close", side_effect=lambda symbol: _series(98.65, end="2026-10-20")):
            series = vd.load_close("SOXL")
        self.assertEqual(series.index[-1], pd.Timestamp("2026-10-21"))
        self.assertEqual(series.iloc[-1], 100.93)

    def test_nothing_refreshes_outside_the_session(self):
        with patch.object(vd, "session_open", return_value=False):
            series = vd.load_close("SOXL")
            self.clock += 600
            vd.load_close("SOXL")
        self.assertEqual(self.tail_calls, 0)
        self.assertEqual(series.iloc[-1], 100.93)

    def test_the_t_bill_rate_is_never_refreshed(self):
        vd.load_close(vd.RATE_SYMBOL)
        self.clock += 600
        vd.load_close(vd.RATE_SYMBOL)
        self.assertEqual(self.tail_calls, 0)

    def test_a_failed_refresh_keeps_the_series_and_backs_off(self):
        vd.load_close("SOXL")
        calls = self.tail_calls
        with patch.object(vd, "download_tail", side_effect=RuntimeError("yahoo down")) as broken:
            self.clock += 120
            series = vd.load_close("SOXL")
            self.assertEqual(series.iloc[-1], 100.93)
            vd.load_close("SOXL")  # right after: no second attempt
            self.assertEqual(broken.call_count, 1)
        self.assertEqual(self.tail_calls, calls)

    def test_session_hours(self):
        def at(day, hour, minute):
            return datetime(2026, 10, day, hour, minute, tzinfo=EASTERN)

        self.assertTrue(REAL_SESSION_OPEN(at(21, 10, 30)))  # Wednesday morning
        self.assertTrue(REAL_SESSION_OPEN(at(21, 16, 30)))  # just after the close
        self.assertFalse(REAL_SESSION_OPEN(at(21, 8, 0)))
        self.assertFalse(REAL_SESSION_OPEN(at(21, 17, 0)))
        self.assertFalse(REAL_SESSION_OPEN(at(24, 11, 0)))  # Saturday


def _yf_frame(closes: dict[str, list[float]]) -> pd.DataFrame:
    """What ``yf.download`` returns: (Price, Ticker) columns."""
    index = pd.bdate_range(end="2026-10-21", periods=3)
    columns = pd.MultiIndex.from_tuples([("Close", ticker) for ticker in closes], names=["Price", "Ticker"])
    return pd.DataFrame({col: closes[col[1]] for col in columns}, index=index)


class DownloadTests(unittest.TestCase):
    def test_the_requested_ticker_is_returned(self):
        frame = _yf_frame({"SOXL": [70.0, 71.0, 72.0]})
        with patch("getdata.yf.download", return_value=frame):
            close = vd.download_tail("SOXL")
        self.assertEqual(close.tolist(), [70.0, 71.0, 72.0])

    def test_another_downloads_tickers_are_refused(self):
        # A concurrent yfinance call can leave other tickers' data in this call's result.
        mixed = _yf_frame({"SOXX": [470.0, 471.0, 472.0], "SOXL": [70.0, 71.0, 72.0]})
        only_other = _yf_frame({"SOXX": [470.0, 471.0, 472.0]})
        for frame in (mixed, only_other):
            with self.subTest(columns=list(frame.columns)), patch("getdata.yf.download", return_value=frame):
                self.assertIsNone(vd.download_tail("SOXL"))

    def test_downloads_run_one_at_a_time(self):
        import threading
        import time

        import getdata

        active, peak = [0], [0]
        guard = threading.Lock()

        def slow_download(*args, **kwargs):
            with guard:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            with guard:
                active[0] -= 1
            return _yf_frame({"SOXL": [70.0, 71.0, 72.0]})

        with patch("getdata.yf.download", side_effect=slow_download):
            threads = [threading.Thread(target=getdata.yf_download, args=(["SOXL"],)) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(peak[0], 1)


if __name__ == "__main__":
    unittest.main()
