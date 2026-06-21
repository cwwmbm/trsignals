import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from api.custom_data import parse_intraday_csv, prepare_custom_intraday_frame, detect_interval
from indicators import VWAP_INDICATOR_COLUMN_IDS, add_vwap_indicators, stub_vwap_columns


class VwapIndicatorTests(unittest.TestCase):
    def _load_qqq_sample(self, rows: int = 300) -> pd.DataFrame:
        csv_path = Path(__file__).resolve().parent.parent / "IntradayData" / "QQQ_60min.csv"
        content = csv_path.read_bytes()
        frame = parse_intraday_csv(content, filename="QQQ_60min.csv")
        return frame.iloc[:rows].copy()

    def test_stub_vwap_columns_on_daily_path(self):
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        frame = pd.DataFrame(
            {
                "Date": dates,
                "Open": np.linspace(100, 110, len(dates)),
                "High": np.linspace(101, 111, len(dates)),
                "Low": np.linspace(99, 109, len(dates)),
                "Close": np.linspace(100.5, 110.5, len(dates)),
                "Volume": 1000,
            }
        )
        result = add_vwap_indicators(frame)
        for column_id in VWAP_INDICATOR_COLUMN_IDS:
            self.assertIn(column_id, result.columns)
            self.assertTrue(result[column_id].isna().all())

    def test_close_vwap_percent_from_csv(self):
        frame = self._load_qqq_sample()
        _, _, periods_per_year = detect_interval(frame)
        prepared = prepare_custom_intraday_frame(frame, periods_per_year)

        self.assertIn("Close_VWAP", prepared.columns)
        manual = (prepared["Close"] - prepared["VWAP"]) / prepared["Close"] * 100
        pd.testing.assert_series_equal(
            prepared["Close_VWAP"],
            manual,
            check_names=False,
            rtol=1e-9,
            atol=1e-9,
        )
        self.assertLessEqual(prepared["Close_VWAP"].abs().max(), 5.0)

    def test_vwap_cross_flags_fire(self):
        frame = self._load_qqq_sample()
        result = add_vwap_indicators(frame, source_timezone=frame.attrs.get("timezone", "UTC"))

        cross_up = result[result["VWAPCrossUp"] == 1]
        cross_down = result[result["VWAPCrossDown"] == 1]
        self.assertGreater(len(cross_up), 0)
        self.assertGreater(len(cross_down), 0)

    def test_vwap_std_bands_are_valid(self):
        frame = self._load_qqq_sample(rows=80)
        result = add_vwap_indicators(frame, source_timezone=frame.attrs.get("timezone", "UTC"))

        std_values = result["VWAPStd"].dropna()
        self.assertGreater(len(std_values), 1)
        self.assertGreater(std_values.max(), 0)
        self.assertTrue((result["VWAPUpper2"] >= result["VWAPLower2"]).all())

    def test_vwap_percent_b_is_finite(self):
        frame = self._load_qqq_sample()
        result = add_vwap_indicators(frame, source_timezone=frame.attrs.get("timezone", "UTC"))
        percent_b = result["VWAPPercentB"].dropna()
        self.assertGreater(len(percent_b), 0)
        self.assertFalse(np.isinf(percent_b).any())
        self.assertLess(percent_b.abs().max(), 10)

    def test_all_vwap_columns_present_on_custom_prepare(self):
        frame = self._load_qqq_sample()
        _, _, periods_per_year = detect_interval(frame)
        prepared = prepare_custom_intraday_frame(frame, periods_per_year)
        for column_id in VWAP_INDICATOR_COLUMN_IDS:
            self.assertIn(column_id, prepared.columns)
            self.assertFalse(prepared[column_id].isna().all())


if __name__ == "__main__":
    unittest.main()
