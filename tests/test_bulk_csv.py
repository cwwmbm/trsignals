import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import getdata as dt


class BulkCsvTests(unittest.TestCase):
    def _sample_bulk(self) -> pd.DataFrame:
        index = pd.to_datetime(["2026-07-23", "2026-07-24"])
        columns = pd.MultiIndex.from_product(
            [["Close", "High", "Low", "Open", "Volume"], ["SPY", "^VIX"]],
            names=["Price", "Ticker"],
        )
        data = pd.DataFrame(
            [
                [600.0, 15.0, 601.0, 15.5, 599.0, 14.5, 600.5, 15.2, 1_000_000, 0.0],
                [602.0, 16.0, 603.0, 16.5, 601.0, 15.5, 601.5, 15.8, 1_100_000, 0.0],
            ],
            index=index,
            columns=columns,
        )
        data.index.name = "Date"
        return data

    def test_load_bulk_csv_roundtrip(self):
        bulk = self._sample_bulk()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bulk.csv"
            bulk.to_csv(path)
            loaded = dt.load_bulk_csv(path)
            self.assertIsInstance(loaded.columns, pd.MultiIndex)
            self.assertEqual(float(loaded["Close"]["SPY"].iloc[-1]), 602.0)
            self.assertEqual(float(loaded["Close"]["^VIX"].iloc[-1]), 16.0)

    def test_get_bulk_data_uses_csv_path(self):
        bulk = self._sample_bulk()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bulk.csv"
            bulk.to_csv(path)
            loaded = dt.get_bulk_data(["SPY", "^VIX"], years=1, csv_path=str(path))
            self.assertEqual(list(loaded.columns.get_level_values(1).unique()), ["SPY", "^VIX"])
            self.assertEqual(float(loaded["Close"]["SPY"].loc["2026-07-24"]), 602.0)

    def test_default_scan_bulk_csv_path_uses_env(self):
        previous = os.environ.get("SCAN_BULK_CSV")
        os.environ["SCAN_BULK_CSV"] = "/tmp/custom_scan.csv"
        try:
            self.assertEqual(str(dt.default_scan_bulk_csv_path()), "/tmp/custom_scan.csv")
        finally:
            if previous is None:
                os.environ.pop("SCAN_BULK_CSV", None)
            else:
                os.environ["SCAN_BULK_CSV"] = previous


if __name__ == "__main__":
    unittest.main()
