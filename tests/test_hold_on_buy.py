import unittest

import numpy as np
import pandas as pd

import backtest as bt


def _frame(*, buy, sell, close=None, rows=10):
    if close is None:
        close = 100 + np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=rows, freq="B"),
            "Close": close,
            "%Change": np.r_[0.0, np.diff(close) / close[:-1]],
            "Buy": buy,
            "Sell": sell,
        }
    )


class HoldOnBuySignalTests(unittest.TestCase):
    def setUp(self):
        self._original = bt.HoldOnBuySignal

    def tearDown(self):
        bt.HoldOnBuySignal = self._original

    def test_default_exits_when_buy_and_sell_overlap(self):
        buy = [False, True, True, False, False, False, False, False, False, False]
        sell = [False, False, True, False, False, False, False, False, False, False]
        data = _frame(buy=buy, sell=sell)

        bt.HoldOnBuySignal = False
        executed = bt.execute_strategy(data.copy(), days=5, profit=99, is_long=True)

        self.assertTrue(bool(executed["LongTradeOut"].iloc[2]))

    def test_enabled_stays_in_when_buy_and_sell_overlap(self):
        buy = [False, True, True, True, False, False, False, False, False, False]
        sell = [False, False, True, False, False, False, False, False, False, False]
        data = _frame(buy=buy, sell=sell)

        bt.HoldOnBuySignal = True
        executed = bt.execute_strategy(data.copy(), days=5, profit=99, is_long=True)

        self.assertFalse(bool(executed["LongTradeOut"].iloc[2]))
        self.assertTrue(bool(executed["HoldLong"].iloc[3]))

    def test_enabled_suppresses_hold_days_exit_while_buy_active(self):
        buy = [False, True, True, True, True, False, False, False, False, False]
        sell = [False] * 10
        data = _frame(buy=buy, sell=sell)

        bt.HoldOnBuySignal = True
        executed = bt.execute_strategy(data.copy(), days=2, profit=99, is_long=True)

        self.assertFalse(bool(executed["LongTradeOut"].iloc[3]))
        self.assertEqual(int(executed["DaysInTrade"].iloc[3]), 0)
        self.assertTrue(bool(executed["HoldLong"].iloc[3]))

    def test_enabled_suppresses_profitable_closes_exit_while_buy_active(self):
        close = np.array([100.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0])
        buy = [False, True, True, True, True, False, False, False, False, False]
        sell = [False] * 10
        data = _frame(buy=buy, sell=sell, close=close)

        bt.HoldOnBuySignal = True
        executed = bt.execute_strategy(data.copy(), days=99, profit=1, is_long=True)

        self.assertFalse(bool(executed["LongTradeOut"].iloc[2]))
        self.assertEqual(int(executed["ProfitableCloses"].iloc[2]), 0)
        self.assertTrue(bool(executed["HoldLong"].iloc[3]))

    def test_enabled_still_exits_when_buy_not_active(self):
        buy = [False, True, False, False, False, False, False, False, False, False]
        sell = [False, False, True, False, False, False, False, False, False, False]
        data = _frame(buy=buy, sell=sell)

        bt.HoldOnBuySignal = True
        executed = bt.execute_strategy(data.copy(), days=5, profit=99, is_long=True)

        self.assertTrue(bool(executed["LongTradeOut"].iloc[2]))

    def test_no_exit_day_after_buy_stops_when_profit_was_reset(self):
        close = np.array([100.0, 100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        buy = [False, True, True, False, False, False, False, False, False, False]
        sell = [False] * 10
        data = _frame(buy=buy, sell=sell, close=close)

        bt.HoldOnBuySignal = True
        executed = bt.execute_strategy(data.copy(), days=4, profit=1, is_long=True)

        self.assertFalse(bool(executed["LongTradeOut"].iloc[2]))
        self.assertEqual(int(executed["ProfitableCloses"].iloc[2]), 0)
        self.assertFalse(bool(executed["LongTradeOut"].iloc[3]))
        self.assertTrue(bool(executed["HoldLong"].iloc[3]))

    def test_hold_days_must_be_re_earned_after_reset(self):
        buy = [False, True, True, True, False, False, False, False, False, False]
        sell = [False] * 10
        data = _frame(buy=buy, sell=sell)

        bt.HoldOnBuySignal = True
        executed = bt.execute_strategy(data.copy(), days=2, profit=99, is_long=True)

        self.assertFalse(bool(executed["LongTradeOut"].iloc[3]))
        self.assertEqual(int(executed["DaysInTrade"].iloc[3]), 0)
        self.assertFalse(bool(executed["LongTradeOut"].iloc[4]))
        self.assertEqual(int(executed["DaysInTrade"].iloc[4]), 1)
        self.assertTrue(bool(executed["LongTradeOut"].iloc[5]))
        self.assertEqual(int(executed["DaysInTrade"].iloc[5]), 2)


if __name__ == "__main__":
    unittest.main()
