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

    def test_open_trade_pnl_uses_last_valid_close_when_final_bar_is_nan(self):
        rows = 40
        close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, rows))
        buy = np.zeros(rows, dtype=bool)
        buy[rows - 2] = True
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2018-01-01", periods=rows, freq="B"),
                "Close": close,
                "%Change": np.r_[0.0, np.diff(close) / close[:-1]],
                "Buy": buy,
                "Sell": False,
            }
        )
        data.loc[data.index[-1], "Close"] = np.nan
        data.loc[data.index[-1], "%Change"] = np.nan

        executed = bt.execute_strategy(data.copy(), days=10, profit=1, is_long=True)
        last = executed.iloc[-1]

        self.assertTrue(bool(last["HoldLong"]))
        self.assertFalse(pd.isna(last["TradePnL"]))

    @patch("stats.ExcludeBestReturnYear", False)
    def test_indicator_sweep_yearly_uses_proxy_pnl_column(self):
        rows = 1500
        data = _sample_signal_data(rows)
        buy = np.zeros(rows, dtype=bool)
        buy[10::25] = True
        data["Buy"] = buy
        data["Vix"] = np.random.default_rng(1).uniform(10, 30, rows)
        data["PROXY"] = 50 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.001, 0.03, rows)))

        sweep = bt.backtest_sell_ind(
            data, 20, 100, True, "Vix", "less", 15, 15, 1,
            include_yearly=False, pnl_column="PROXY",
        )
        top = sweep.head(1)
        expected = bt._run_indicator_threshold(
            data, 20, 100, True, top.iloc[0]["Indicator"], top.iloc[0]["Buysell"],
            top.iloc[0]["Condition"], top.iloc[0]["Value"], pnl_column="PROXY",
        )["Yearly"]

        with_proxy = bt.add_yearly_to_indicator_rows(
            top, data, 20, 100, True, pnl_column="PROXY"
        ).iloc[0]["Yearly"]
        without_proxy = bt.add_yearly_to_indicator_rows(
            top, data, 20, 100, True
        ).iloc[0]["Yearly"]

        self.assertEqual(with_proxy, expected)
        self.assertNotEqual(with_proxy, without_proxy)


if __name__ == "__main__":
    unittest.main()
