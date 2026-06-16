import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from api.scan_service import compute_kelly, run_scan, _legacy_scan_row, _scan_sort_key, SCAN_SYMBOL_ORDER, SIGNAL_ORDER


def _sample_executed(*, trade_out: bool, trade_pnl: float, hold_long: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "LongTradeIn": [False, trade_out],
            "HoldLong": [False, hold_long],
            "LongTradeOut": [False, trade_out],
            "TradePnL": [0.0, trade_pnl],
        }
    )


class ScanServiceTests(unittest.TestCase):
    def test_compute_kelly_returns_none_without_trades(self):
        executed = pd.DataFrame(
            {
                "LongTradeOut": [False, False],
                "TradePnL": [0.0, 0.0],
            }
        )
        self.assertIsNone(compute_kelly(executed))

    def test_compute_kelly_with_mixed_trades(self):
        executed = pd.DataFrame(
            {
                "LongTradeOut": [False, True, True, True],
                "TradePnL": [0.0, 0.05, -0.02, 0.03],
            }
        )
        kelly = compute_kelly(executed)
        self.assertIsNotNone(kelly)
        self.assertIsInstance(kelly, float)

    def test_legacy_scan_row_skips_when_ignore_true(self):
        data = pd.DataFrame({"Close": [100.0, 101.0]})

        def ignored_signal(_data, _symbol):
            return False, False, 2, 1, "ignored", "", True, True

        self.assertIsNone(_legacy_scan_row("SPY", ignored_signal, data))

    def test_legacy_scan_row_returns_flags(self):
        data = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})

        def active_signal(frame, _symbol):
            frame["Buy"] = True
            frame["Sell"] = False
            return frame["Buy"], frame["Sell"], 2, 1, "Active signal", "", True, False

        with patch("api.scan_service.bt.execute_strategy") as execute_strategy:
            execute_strategy.return_value = _sample_executed(trade_out=True, trade_pnl=0.018, hold_long=True)
            row = _legacy_scan_row("SPY", active_signal, data)

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["source"], "legacy")
        self.assertEqual(row["signal"], "active_signal")
        self.assertTrue(row["buy_signal"])
        self.assertTrue(row["hold_long"])
        self.assertEqual(row["trade_pnl"], 1.8)

    def test_run_scan_uses_injected_symbol_frames(self):
        data = pd.DataFrame({"Close": np.linspace(100, 105, 30)})

        def active_signal(frame, symbol):
            frame["Buy"] = frame["Close"] < 200
            frame["Sell"] = False
            return frame["Buy"], frame["Sell"], 2, 1, f"{symbol} legacy", "", True, False

        with patch("api.scan_service.LEGACY_BUY_SIGNALS", [active_signal]):
            with patch("api.scan_service.list_strategies", return_value=[]):
                with patch("api.scan_service.bt.execute_strategy") as execute_strategy:
                    execute_strategy.return_value = _sample_executed(trade_out=False, trade_pnl=0.0)
                    rows = run_scan(symbol_frames={"SPY": data})

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "SPY")
        self.assertEqual(rows[0]["description"], "SPY legacy")

    def test_scan_sort_key_orders_symbol_before_signal(self):
        rows = [
            {"symbol": "QQQ", "signal": "buy_signal7"},
            {"symbol": "SPY", "signal": "buy_signal10"},
            {"symbol": "SPY", "signal": "buy_signal7"},
            {"symbol": "SMH", "signal": "buy_signal16"},
        ]
        rows.sort(key=_scan_sort_key)
        self.assertEqual(
            [(row["symbol"], row["signal"]) for row in rows],
            [
                ("SPY", "buy_signal7"),
                ("SPY", "buy_signal10"),
                ("SMH", "buy_signal16"),
                ("QQQ", "buy_signal7"),
            ],
        )

    def test_scan_symbol_order_matches_signal_check(self):
        self.assertEqual(
            SCAN_SYMBOL_ORDER,
            ["SPY", "SMH", "QQQ", "SOXX", "IWM", "FXI", "AAPL", "GDX", "MSFT", "GLD", "XBI", "TLT"],
        )
        self.assertEqual(SIGNAL_ORDER[-2:], ["og_buy_signal", "og_new_buy_signal"])


if __name__ == "__main__":
    unittest.main()
