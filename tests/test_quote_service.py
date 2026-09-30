from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import ta
import unittest

from api.quote_service import (
    EASTERN,
    build_quote_snapshot,
    metrics_from_ohlcv,
    missing_weekdays,
)
from indicators import internal_bar_range

ET = EASTERN


def _close(pairs: list[tuple[str, float | None]]) -> pd.Series:
    index = pd.to_datetime([day for day, _ in pairs])
    values = [value for _, value in pairs]
    series = pd.Series(values, index=index, dtype=float)
    series.index.name = "Date"
    return series


def _ohlcv(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    index = pd.to_datetime([day for day, _, _, _ in rows])
    frame = pd.DataFrame(
        {
            "High": [high for _, high, _, _ in rows],
            "Low": [low for _, _, low, _ in rows],
            "Close": [close for _, _, _, close in rows],
        },
        index=index,
    )
    frame.index.name = "Date"
    return frame


def _bulk(dates: list[str], closes: dict[str, list[float | None]]) -> pd.DataFrame:
    index = pd.to_datetime(dates)
    tickers = list(closes)
    columns = pd.MultiIndex.from_product(
        [["Close", "High", "Low", "Open", "Volume"], tickers],
        names=["Price", "Ticker"],
    )
    records = []
    for i, _date in enumerate(dates):
        row = []
        for price in ["Close", "High", "Low", "Open", "Volume"]:
            for ticker in tickers:
                close = closes[ticker][i]
                if close is None:
                    row.append(np.nan)
                    continue
                if price == "Close":
                    row.append(close)
                elif price == "High":
                    row.append(close + 1.0)
                elif price == "Low":
                    row.append(close - 1.0)
                elif price == "Open":
                    row.append(close)
                else:
                    row.append(1_000_000.0)
        records.append(row)
    data = pd.DataFrame(records, index=index, columns=columns)
    data.index.name = "Date"
    return data


class MissingWeekdaysTests(unittest.TestCase):
    def test_weekend_in_window_is_not_flagged(self):
        # Mon Aug 24 2026 after close: window is Aug 20–24 (Thu–Mon), including Sat/Sun.
        now = datetime(2026, 8, 24, 17, 0, tzinfo=ET)
        close = _close(
            [
                ("2026-08-20", 100.0),
                ("2026-08-21", 101.0),
                ("2026-08-24", 102.0),
            ]
        )
        self.assertEqual(missing_weekdays(close, now=now), [])

    def test_missing_weekday_is_flagged(self):
        now = datetime(2026, 8, 21, 17, 0, tzinfo=ET)
        close = _close(
            [
                ("2026-08-17", 100.0),
                ("2026-08-18", 101.0),
                ("2026-08-20", 103.0),
                ("2026-08-21", 104.0),
            ]
        )
        self.assertEqual(missing_weekdays(close, now=now), ["2026-08-19"])

    def test_holiday_weekday_is_flagged(self):
        # New Year's Day 2026 is Thursday. Window from Fri Jan 2 after close.
        now = datetime(2026, 1, 2, 17, 0, tzinfo=ET)
        close = _close(
            [
                ("2025-12-29", 100.0),
                ("2025-12-30", 101.0),
                ("2025-12-31", 102.0),
                ("2026-01-02", 103.0),
            ]
        )
        self.assertEqual(missing_weekdays(close, now=now), ["2026-01-01"])

    def test_nan_close_counts_as_missing(self):
        now = datetime(2026, 8, 21, 17, 0, tzinfo=ET)
        close = _close(
            [
                ("2026-08-17", 100.0),
                ("2026-08-18", 101.0),
                ("2026-08-19", None),
                ("2026-08-20", 103.0),
                ("2026-08-21", 104.0),
            ]
        )
        self.assertEqual(missing_weekdays(close, now=now), ["2026-08-19"])

    def test_today_before_close_is_not_flagged(self):
        now = datetime(2026, 8, 21, 10, 0, tzinfo=ET)
        close = _close(
            [
                ("2026-08-17", 100.0),
                ("2026-08-18", 101.0),
                ("2026-08-19", 102.0),
                ("2026-08-20", 103.0),
            ]
        )
        self.assertEqual(missing_weekdays(close, now=now), [])

    def test_today_after_close_is_flagged(self):
        now = datetime(2026, 8, 21, 16, 0, tzinfo=ET)
        close = _close(
            [
                ("2026-08-17", 100.0),
                ("2026-08-18", 101.0),
                ("2026-08-19", 102.0),
                ("2026-08-20", 103.0),
            ]
        )
        self.assertEqual(missing_weekdays(close, now=now), ["2026-08-21"])


class MetricsFromOhlcvTests(unittest.TestCase):
    def test_last_row_ibr_rsi_stoch_and_pct_change(self):
        closes = [
            100.0, 101.0, 99.0, 102.0, 103.0, 101.0, 104.0, 105.0,
            106.0, 104.0, 107.0, 108.0, 106.0, 109.0, 110.0, 108.0,
        ]
        rows = []
        for i, close in enumerate(closes):
            day = f"2026-07-{10 + i:02d}"
            rows.append((day, close + 10.0, close - 10.0, close))
        frame = _ohlcv(rows)

        result = metrics_from_ohlcv(frame)
        expected_ibr = internal_bar_range(frame["High"], frame["Low"], frame["Close"], period=1)[-1]
        expected_rsi2 = ta.momentum.RSIIndicator(frame["Close"], window=2).rsi().iloc[-1]
        expected_rsi5 = ta.momentum.RSIIndicator(frame["Close"], window=5).rsi().iloc[-1]
        expected_stoch = ta.momentum.stoch(
            frame["High"], frame["Low"], frame["Close"], window=14, smooth_window=3
        ).iloc[-1]

        self.assertEqual(result["as_of"], "2026-07-25")
        self.assertAlmostEqual(result["close"], 108.0)
        self.assertAlmostEqual(result["pct_change"], 108.0 / 110.0 - 1.0)
        self.assertAlmostEqual(result["ibr"], float(expected_ibr))
        self.assertAlmostEqual(result["ibr"], (108.0 - 98.0) / (118.0 - 98.0))
        self.assertAlmostEqual(result["rsi2"], float(expected_rsi2))
        self.assertAlmostEqual(result["rsi5"], float(expected_rsi5))
        self.assertAlmostEqual(result["stoch"], float(expected_stoch))


class BuildQuoteSnapshotTests(unittest.TestCase):
    def test_labels_vix_and_reports_per_symbol_gaps(self):
        dates = ["2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21"]
        bulk = _bulk(
            dates,
            {
                "SPY": [500.0, 501.0, 502.0, 503.0, 504.0],
                "QQQ": [400.0, 401.0, None, 403.0, 404.0],
                "SOXX": [200.0, 201.0, 202.0, 203.0, 204.0],
                "^VIX": [16.0, 16.5, 16.2, 15.8, 15.5],
            },
        )
        now = datetime(2026, 8, 21, 17, 0, tzinfo=ET)
        snapshot = build_quote_snapshot(full_data=bulk, now=now)
        by_symbol = {row["symbol"]: row for row in snapshot["quotes"]}

        self.assertEqual(snapshot["as_of"], "2026-08-21")
        self.assertEqual(list(by_symbol), ["SPY", "QQQ", "SOXX", "VIX"])
        self.assertEqual(by_symbol["SPY"]["missing_days"], [])
        self.assertEqual(by_symbol["QQQ"]["missing_days"], ["2026-08-19"])
        self.assertAlmostEqual(by_symbol["VIX"]["close"], 15.5)
        self.assertAlmostEqual(by_symbol["VIX"]["pct_change"], 15.5 / 15.8 - 1.0)


if __name__ == "__main__":
    unittest.main()
