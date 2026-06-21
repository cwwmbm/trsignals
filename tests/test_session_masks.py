import unittest
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent

from api.custom_data import parse_intraday_csv
from api.session_masks import (
    infer_market_timestamp_timezone,
    last_rth_bar_mask,
    regular_trading_hours_mask,
)


def _utc_series(values: list[str]) -> pd.Series:
    return pd.Series(pd.to_datetime(values, utc=True))


class SessionMaskTests(unittest.TestCase):
    def test_regular_trading_hours_mask_allows_open_and_blocks_premarket(self):
        # 2020-07-27 is EDT (UTC-4): 13:30 UTC = 9:30 ET, 12:45 UTC = 8:45 ET pre-market
        dates = _utc_series(
            [
                "2020-07-27 12:45:00+00:00",
                "2020-07-27 13:30:00+00:00",
                "2020-07-27 19:55:00+00:00",
                "2020-07-27 20:05:00+00:00",
            ]
        )
        mask = regular_trading_hours_mask(dates, source_timezone="UTC")

        self.assertFalse(mask.iloc[0])
        self.assertTrue(mask.iloc[1])
        self.assertTrue(mask.iloc[2])
        self.assertFalse(mask.iloc[3])

    def test_last_rth_bar_mask_marks_final_regular_hours_bar(self):
        dates = _utc_series(
            [
                "2020-07-27 13:30:00+00:00",
                "2020-07-27 19:50:00+00:00",
                "2020-07-27 19:55:00+00:00",
                "2020-07-27 20:05:00+00:00",
                "2020-07-28 13:30:00+00:00",
                "2020-07-28 19:55:00+00:00",
            ]
        )
        mask = last_rth_bar_mask(dates, source_timezone="UTC")

        self.assertFalse(mask.iloc[0])
        self.assertFalse(mask.iloc[1])
        self.assertTrue(mask.iloc[2])
        self.assertFalse(mask.iloc[3])
        self.assertFalse(mask.iloc[4])
        self.assertTrue(mask.iloc[5])

    def test_regular_trading_hours_mask_handles_naive_timestamps_as_utc(self):
        dates = pd.Series(pd.to_datetime(["2020-07-27 13:30:00", "2020-07-27 12:45:00"]))
        mask = regular_trading_hours_mask(dates, source_timezone="UTC")

        self.assertTrue(mask.iloc[0])
        self.assertFalse(mask.iloc[1])

    def test_infer_market_timestamp_timezone_detects_canonical_utc(self):
        with open(_REPO_ROOT / "IntradayData" / "AAPL_60min.csv", "rb") as handle:
            frame = parse_intraday_csv(handle.read(), filename="AAPL_60min.csv")
        self.assertEqual(infer_market_timestamp_timezone(frame["Date"]), "UTC")
        self.assertEqual(frame.attrs["timezone"], "UTC")

    def test_infer_market_timestamp_timezone_detects_eastern_wall_clock_labels(self):
        dates = _utc_series(
            [
                "2020-07-27 09:30:00+00:00",
                "2020-07-27 10:30:00+00:00",
                "2020-07-27 11:30:00+00:00",
                "2020-07-27 12:30:00+00:00",
                "2020-07-27 13:30:00+00:00",
                "2020-07-27 14:30:00+00:00",
                "2020-07-27 15:30:00+00:00",
            ]
        )
        self.assertEqual(infer_market_timestamp_timezone(dates), "America/New_York")

    def test_aapl_60min_rth_masks_with_detected_utc(self):
        with open(_REPO_ROOT / "IntradayData" / "AAPL_60min.csv", "rb") as handle:
            frame = parse_intraday_csv(handle.read(), filename="AAPL_60min.csv")
        day = frame[frame["Date"].dt.date == pd.Timestamp("2020-07-27").date()]
        source_timezone = frame.attrs["timezone"]
        rth = regular_trading_hours_mask(day["Date"], source_timezone=source_timezone)
        eod = last_rth_bar_mask(day["Date"], source_timezone=source_timezone)

        self.assertFalse(bool(rth.iloc[0]))
        self.assertTrue(bool(rth.iloc[1]))
        self.assertTrue(bool(eod.iloc[-1]))


if __name__ == "__main__":
    unittest.main()
