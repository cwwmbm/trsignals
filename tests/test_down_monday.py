import unittest

import numpy as np
import pandas as pd

from indicators import down_monday_flag


class DownMondayTests(unittest.TestCase):
    def test_true_when_monday_close_below_prior_friday(self):
        dates = pd.to_datetime(["2024-01-05", "2024-01-08"])
        close = pd.Series([100.0, 99.0])
        result = down_monday_flag(dates, close)
        self.assertEqual(result.tolist(), [-1, 1])

    def test_false_when_monday_close_above_prior_friday(self):
        dates = pd.to_datetime(["2024-01-05", "2024-01-08"])
        close = pd.Series([100.0, 101.0])
        result = down_monday_flag(dates, close)
        self.assertEqual(result.tolist(), [-1, -1])

    def test_tuesday_first_trading_day_does_not_count(self):
        dates = pd.to_datetime(["2024-01-12", "2024-01-16"])
        close = pd.Series([100.0, 98.0])
        result = down_monday_flag(dates, close)
        self.assertEqual(result.tolist(), [-1, -1])

    def test_non_monday_days_are_false(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        close = pd.Series([100.0, 99.0, 98.0, 97.0])
        result = down_monday_flag(dates, close)
        self.assertTrue(np.all(result == -1))

    def test_uses_last_friday_close_not_prior_trading_day(self):
        dates = pd.to_datetime(["2024-01-04", "2024-01-08"])
        close = pd.Series([100.0, 99.0])
        result = down_monday_flag(dates, close)
        self.assertEqual(result.tolist(), [-1, -1])

    def test_add_indicators_includes_down_monday(self):
        from backtest_runners import load_ticker_data

        result = load_ticker_data("SPY", years=2)
        self.assertIn("DownMonday", result.columns)
        mondays = pd.to_datetime(result["Date"]).dt.dayofweek == 0
        if mondays.any():
            self.assertTrue(result.loc[mondays, "DownMonday"].isin([1, -1]).all())


if __name__ == "__main__":
    unittest.main()
