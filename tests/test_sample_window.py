import unittest

import numpy as np
import pandas as pd

from api.sample_window import (
    in_sample_end_timestamp,
    in_sample_fraction,
    sample_window_meta,
    slice_frame_to_end,
    slice_symbol_data_to_end,
)


class SampleWindowTests(unittest.TestCase):
    def test_in_sample_fraction_reads_config_default(self):
        self.assertEqual(in_sample_fraction(), 0.7)

    def test_in_sample_end_timestamp_uses_row_fraction(self):
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "Close": np.arange(10),
            }
        )
        end = in_sample_end_timestamp(frame, fraction=0.7)
        # int(10 * 0.7) - 1 = 6 → 2020-01-07
        self.assertEqual(end, pd.Timestamp("2020-01-07"))

    def test_slice_frame_to_end(self):
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "Close": np.arange(10),
            }
        )
        sliced = slice_frame_to_end(frame, pd.Timestamp("2020-01-07"))
        self.assertEqual(len(sliced), 7)
        self.assertEqual(pd.Timestamp(sliced["Date"].iloc[-1]), pd.Timestamp("2020-01-07"))

    def test_slice_symbol_data_to_end(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        symbol_data = {
            "SOXX": pd.DataFrame({"Date": dates, "Close": range(5)}),
            "SMH": pd.DataFrame({"Date": dates, "Close": range(5, 10)}),
        }
        sliced = slice_symbol_data_to_end(symbol_data, pd.Timestamp("2020-01-03"))
        self.assertEqual(len(sliced["SOXX"]), 3)
        self.assertEqual(len(sliced["SMH"]), 3)

    def test_sample_window_meta(self):
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "Close": np.arange(10),
            }
        )
        meta = sample_window_meta(frame, sample="in_sample", fraction=0.7)
        self.assertEqual(meta["sample"], "in_sample")
        self.assertEqual(meta["in_sample_fraction"], 0.7)
        self.assertEqual(meta["in_sample_end"], "2020-01-07")
        self.assertEqual(meta["period_start"], "2020-01-01")
        self.assertEqual(meta["period_end"], "2020-01-10")


if __name__ == "__main__":
    unittest.main()
