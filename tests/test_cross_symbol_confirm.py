import unittest

import numpy as np
import pandas as pd

import backtest as bt


def _frame(dates, closes):
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": np.full(len(dates), 1000.0),
            "%Change": np.zeros(len(dates)),
        }
    )


def _stub_signal(buy_masks, *, days=3, profit=1):
    def signal(data, symbol):
        mask = buy_masks[symbol]
        length = len(data)
        buy = pd.Series(mask[:length], index=data.index)
        sell = pd.Series([False] * length, index=data.index)
        return buy, sell, days, profit, "test", "", True, False

    signal.__name__ = "stub_signal"
    return signal


class CrossSymbolConfirmTests(unittest.TestCase):
    def setUp(self):
        self.dates = ["2024-01-05", "2024-01-08", "2024-01-09"]
        self.primary = "SPY"
        self.confirm = "SMH"
        self.symbol_data = {
            self.primary: _frame(self.dates, [100.0, 101.0, 102.0]),
            self.confirm: _frame(self.dates, [50.0, 51.0, 52.0]),
        }

    def test_primary_confirms_when_confirm_still_holding_from_prior_buy(self):
        signal = _stub_signal(
            {
                self.primary: [False, True, False],
                self.confirm: [True, False, False],
            }
        )
        frame, _, _, _, _, _, _ = bt.apply_cross_symbol_signal(
            signal,
            self.primary,
            [self.confirm],
            self.symbol_data,
        )
        self.assertFalse(frame["Buy"].iloc[0])
        self.assertTrue(frame["Buy"].iloc[1])
        self.assertFalse(frame["Buy"].iloc[2])

    def test_primary_not_confirmed_when_confirm_exited(self):
        signal = _stub_signal(
            {
                self.primary: [False, False, True],
                self.confirm: [True, False, False],
            },
            days=1,
            profit=1,
        )
        frame, _, _, _, _, _, _ = bt.apply_cross_symbol_signal(
            signal,
            self.primary,
            [self.confirm],
            self.symbol_data,
        )
        self.assertFalse(frame["Buy"].iloc[2])

    def test_primary_not_confirmed_when_confirm_holding_but_sell_signal(self):
        def signal(data, symbol):
            length = len(data)
            if symbol == self.primary:
                buy = pd.Series([False, True, False], index=data.index)
                sell = pd.Series([False] * length, index=data.index)
            else:
                buy = pd.Series([True, False, False], index=data.index)
                sell = pd.Series([False, True, False], index=data.index)
            return buy, sell, 3, 1, "test", "", True, False

        signal.__name__ = "stub_signal"
        frame, _, _, _, _, _, _ = bt.apply_cross_symbol_signal(
            signal,
            self.primary,
            [self.confirm],
            self.symbol_data,
        )
        self.assertFalse(frame["Buy"].iloc[1])

    def test_same_day_buy_on_both_still_confirms(self):
        signal = _stub_signal(
            {
                self.primary: [False, True, False],
                self.confirm: [False, True, False],
            }
        )
        frame, _, _, _, _, _, _ = bt.apply_cross_symbol_signal(
            signal,
            self.primary,
            [self.confirm],
            self.symbol_data,
        )
        self.assertTrue(frame["Buy"].iloc[1])

    def test_confirm_active_handles_scalar_sell_false(self):
        signal = _stub_signal(
            {
                self.primary: [False, True, False],
                self.confirm: [True, False, False],
            }
        )

        def signal_with_false_sell(data, symbol):
            buy, _, days, profit, desc, verdict, is_long, ignore = signal(data, symbol)
            return buy, False, days, profit, desc, verdict, is_long, ignore

        signal_with_false_sell.__name__ = "stub_signal"
        frame, _, _, _, _, _, _ = bt.apply_cross_symbol_signal(
            signal_with_false_sell,
            self.primary,
            [self.confirm],
            self.symbol_data,
        )
        self.assertTrue(frame["Buy"].iloc[1])

    def test_requires_all_confirm_symbols_active(self):
        signal = _stub_signal(
            {
                self.primary: [False, True, False],
                self.confirm: [True, False, False],
                "QQQ": [False, False, False],
            },
            days=3,
        )
        self.symbol_data["QQQ"] = _frame(self.dates, [200.0, 201.0, 202.0])
        frame, _, _, _, _, _, _ = bt.apply_cross_symbol_signal(
            signal,
            self.primary,
            [self.confirm, "QQQ"],
            self.symbol_data,
        )
        self.assertFalse(frame["Buy"].iloc[1])


if __name__ == "__main__":
    unittest.main()
