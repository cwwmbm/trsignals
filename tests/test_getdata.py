import unittest

import numpy as np
import pandas as pd

from getdata import extract_market_context, symbol_frame_from_bulk


class BulkSymbolFrameTests(unittest.TestCase):
    def test_symbol_frame_drops_pre_inception_primary_rows(self):
        dates = pd.date_range("2024-01-02", periods=4, freq="B")
        dates.name = "Date"
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["XBI"]],
        )
        full_data = pd.DataFrame(index=dates, columns=columns, dtype=float)
        full_data[("Open", "XBI")] = [np.nan, np.nan, 10.0, 11.0]
        full_data[("High", "XBI")] = [np.nan, np.nan, 11.0, 12.0]
        full_data[("Low", "XBI")] = [np.nan, np.nan, 9.0, 10.0]
        full_data[("Close", "XBI")] = [np.nan, np.nan, 10.5, 11.5]
        full_data[("Volume", "XBI")] = [np.nan, np.nan, 1000.0, 1100.0]
        context = {
            "vix_close": pd.Series([20.0, 21.0, 22.0, 23.0], index=dates),
            "breadth": pd.Series([1.0, 1.0, 1.0, 1.0], index=dates),
            "qqq_to_spy": pd.Series([1.0, 1.0, 1.0, 1.0], index=dates),
            "smh_to_spy": pd.Series([1.0, 1.0, 1.0, 1.0], index=dates),
            "xlf_to_spy": pd.Series([1.0, 1.0, 1.0, 1.0], index=dates),
            "xle_to_spy": pd.Series([1.0, 1.0, 1.0, 1.0], index=dates),
            "xlu_to_spy": pd.Series([1.0, 1.0, 1.0, 1.0], index=dates),
            "xli_to_spy": pd.Series([1.0, 1.0, 1.0, 1.0], index=dates),
            "gold_to_spy": pd.Series([1.0, 1.0, 1.0, 1.0], index=dates),
            "bond_breadth": pd.Series([1.0, 1.0, 1.0, 1.0], index=dates),
            "soxx": pd.Series([100.0, 101.0, 102.0, 103.0], index=dates),
            "qqq": pd.Series([100.0, 101.0, 102.0, 103.0], index=dates),
            "spy_bull": pd.Series([-1, -1, 1, 1], index=dates),
        }

        frame = symbol_frame_from_bulk(full_data, "XBI", context)

        self.assertEqual(frame["Date"].tolist(), list(dates[2:]))
        self.assertFalse(frame[["Open", "High", "Low", "Close"]].isna().any().any())

    def test_extract_market_context_without_vix(self):
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        columns = pd.MultiIndex.from_product([["Close"], ["SPY", "RSP", "QQQ", "SMH", "XLF", "XLE", "XLU", "XLI", "GLD", "TLT", "SOXX"]])
        full_data = pd.DataFrame(index=dates, columns=columns, dtype=float)
        for symbol in columns.get_level_values(1):
            full_data[("Close", symbol)] = 100.0

        context = extract_market_context(
            full_data,
            {symbol: symbol for symbol in columns.get_level_values(1).unique()} | {"^VIX": "^VIX", "SPY": "SPY"},
        )

        self.assertTrue(context["vix_close"].isna().all())
        self.assertEqual(len(context["breadth"]), 3)


if __name__ == "__main__":
    unittest.main()
