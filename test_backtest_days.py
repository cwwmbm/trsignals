import re
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import backtest as bt


def _sample_signal_data(rows: int = 120) -> pd.DataFrame:
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, rows))
    buy = np.zeros(rows, dtype=bool)
    buy[5] = True
    buy[30] = True
    buy[60] = True
    buy[90] = True
    return pd.DataFrame(
        {
            "Date": pd.date_range("2018-01-01", periods=rows, freq="B"),
            "Close": close,
            "%Change": np.r_[0.0, np.diff(close) / close[:-1]],
            "Buy": buy,
            "Sell": False,
        }
    )


def _parse_dollar_pnl(value) -> int:
    return int(float(re.sub(r"[$,]", "", str(value))))


class BacktestDaysTests(unittest.TestCase):
    @patch("stats.ExcludeBestReturnYear", False)
    def test_hold_days_row_matches_single_execute(self):
        data = _sample_signal_data()
        days, profit = 4, 1
        is_long = True

        executed = bt.execute_strategy(data.copy(), days, profit, is_long)
        expected_pnl = int(executed["RollingPnL"].iloc[-1])
        expected_trades = int(executed["LongTradeOut"].sum())

        results = bt.backtest_days(data.copy(), max_days=7, is_long=is_long)
        row = results[(results["Days"] == days) & (results["Prf"] == profit)].iloc[0]

        self.assertEqual(int(row["Trades"]), expected_trades)
        self.assertEqual(_parse_dollar_pnl(row["PnL"]), expected_pnl)


if __name__ == "__main__":
    unittest.main()
