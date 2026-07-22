import unittest
from types import SimpleNamespace
from unittest.mock import ANY, patch

import numpy as np
import pandas as pd

from api.scan_service import compute_kelly, run_scan, _builder_scan_row, _legacy_scan_row, _portfolio_scan_row, _scan_sort_key, _scan_trade_pnl_pct, SCAN_SYMBOL_ORDER, SIGNAL_ORDER


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

    def test_scan_trade_pnl_pct_falls_back_for_open_trade_with_nan_close(self):
        executed = pd.DataFrame(
            {
                "HoldLong": [False, True, True],
                "Close": [100.0, 102.0, np.nan],
                "TradeEntry": [0.0, 100.0, 100.0],
                "TradePnL": [0.0, 0.02, np.nan],
            }
        )
        self.assertEqual(_scan_trade_pnl_pct(executed, is_long=True), 6.0)

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
                with patch("api.scan_service.list_portfolios", return_value=[]):
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
            ["SPY", "SMH", "QQQ", "SOXX", "IWM", "FXI", "GDX", "GLD", "XBI", "TLT"],
        )
        self.assertEqual(SIGNAL_ORDER[-2:], ["og_buy_signal", "og_new_buy_signal"])

    def test_run_scan_downloads_yfinance_once_for_builder_strategies(self):
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-02", periods=30, freq="B"),
                "Close": np.linspace(100, 105, 30),
                "SMA200": np.linspace(99, 104, 30),
            }
        )
        strategy = SimpleNamespace(
            id="s1",
            name="Test Strategy",
            symbol="SPY",
            direction="long",
            hold_days=2,
            profit=1,
            description="test",
            proxy_symbol="SOXX",
            confirm_symbols=["SMH"],
            conditions=[SimpleNamespace(model_dump=lambda: {"left": "Close", "operator": ">", "right": "SMA200", "logic": "AND"})],
            sell_conditions=[],
        )
        bulk_data = pd.DataFrame()

        with patch("api.scan_service.LEGACY_BUY_SIGNALS", []):
            with patch("api.scan_service.dt.get_bulk_data", return_value=bulk_data) as get_bulk_data:
                with patch("api.scan_service._prepare_symbol_frame", return_value=data):
                    with patch("api.scan_service.list_strategies", return_value=[strategy]):
                        with patch("api.scan_service.list_portfolios", return_value=[]):
                            with patch("api.scan_service.bt.build_symbol_dataset", return_value={"SPY": data, "SMH": data}):
                                with patch("api.scan_service.execute_saved_strategy") as execute_saved:
                                    execute_saved.return_value = _sample_executed(trade_out=False, trade_pnl=0.0)
                                    run_scan()

        get_bulk_data.assert_called_once()
        execute_saved.assert_called_once()
        self.assertIs(execute_saved.call_args.kwargs.get("bulk_data"), bulk_data)
        self.assertIsNotNone(execute_saved.call_args.kwargs.get("symbol_data"))

    def test_run_scan_includes_saved_strategy_symbols_outside_scan_list(self):
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-02", periods=30, freq="B"),
                "Close": np.linspace(100, 105, 30),
                "IBR": np.full(30, 0.4),
            }
        )
        strategy = SimpleNamespace(
            id="jepq-1",
            name="JEPQ Test",
            symbol="JEPQ",
            direction="long",
            hold_days=3,
            profit=1,
            description="",
            proxy_symbol=None,
            confirm_symbols=[],
            conditions=[SimpleNamespace(model_dump=lambda: {"left": "IBR", "operator": "<=", "right": "0.5", "logic": "AND"})],
            sell_conditions=[],
        )
        bulk_data = pd.DataFrame()

        with patch("api.scan_service.LEGACY_BUY_SIGNALS", []):
            with patch("api.scan_service.dt.get_bulk_data", return_value=bulk_data) as get_bulk_data:
                with patch("api.scan_service._prepare_symbol_frame", return_value=data) as prepare_frame:
                    with patch("api.scan_service.list_strategies", return_value=[strategy]):
                        with patch("api.scan_service.list_portfolios", return_value=[]):
                            with patch("api.scan_service.execute_saved_strategy") as execute_saved:
                                execute_saved.return_value = _sample_executed(trade_out=False, trade_pnl=0.0)
                                rows = run_scan()

        get_bulk_data.assert_called_once()
        self.assertIn("JEPQ", get_bulk_data.call_args.args[0])
        prepare_frame.assert_any_call(bulk_data, "JEPQ", "JEPQ")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "JEPQ")
        self.assertEqual(rows[0]["signal"], "JEPQ Test")

    def test_portfolio_scan_row_uses_overlay_last_bar(self):
        portfolio = SimpleNamespace(
            id="p1",
            name="My Portfolio",
            strategy_ids=["a", "b"],
            overlap_mode="first_signal_only",
            proxy_symbol=None,
        )
        frame = pd.DataFrame(
            {
                "LongTradeIn": [False, True],
                "HoldLong": [False, True],
                "LongTradeOut": [False, False],
                "TradePnL": [0.0, 0.04],
            }
        )

        with patch("api.portfolio_service.build_portfolio_overlay_frame") as build_overlay:
            build_overlay.return_value = (
                frame,
                [
                    SimpleNamespace(symbol="SPY", hold_days=3, profit=2),
                    SimpleNamespace(symbol="SOXX", hold_days=5, profit=1),
                ],
                "Portfolio (2 strategies, first signal only): A, B",
                5,
                2,
            )
            row = _portfolio_scan_row(portfolio)

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["source"], "portfolio")
        self.assertEqual(row["portfolio_id"], "p1")
        self.assertIsNone(row["strategy_id"])
        self.assertEqual(row["symbol"], "SOXX+SPY")
        self.assertEqual(row["signal"], "My Portfolio")
        self.assertTrue(row["buy_signal"])
        self.assertTrue(row["hold_long"])
        self.assertFalse(row["sell_signal"])
        self.assertEqual(row["days"], 5)
        self.assertEqual(row["profit"], 2)
        self.assertEqual(row["trade_pnl"], 4.0)

    def test_run_scan_includes_portfolio_rows(self):
        data = pd.DataFrame({"Close": np.linspace(100, 105, 30)})
        portfolio = SimpleNamespace(
            id="p1",
            name="Combo",
            strategy_ids=["s1"],
            overlap_mode="first_signal_only",
            proxy_symbol=None,
        )

        with patch("api.scan_service.LEGACY_BUY_SIGNALS", []):
            with patch("api.scan_service.list_strategies", return_value=[]):
                with patch("api.scan_service.list_portfolios", return_value=[portfolio]):
                    with patch("api.scan_service._portfolio_scan_row") as portfolio_row:
                        portfolio_row.return_value = {
                            "id": "portfolio:p1",
                            "source": "portfolio",
                            "strategy_id": None,
                            "portfolio_id": "p1",
                            "symbol": "SPY",
                            "signal": "Combo",
                            "buy_signal": False,
                            "hold_long": False,
                            "sell_signal": False,
                            "days": 2,
                            "profit": 1,
                            "trade_pnl": 0.0,
                            "kelly": None,
                            "description": "Portfolio",
                        }
                        rows = run_scan(symbol_frames={"SPY": data})

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "portfolio")
        portfolio_row.assert_called_once_with(
            portfolio,
            bulk_data=ANY,
            symbol_dataset=ANY,
        )

    def test_run_scan_downloads_yfinance_once_with_portfolios(self):
        data = pd.DataFrame({"Close": np.linspace(100, 105, 30)})
        strategy = SimpleNamespace(
            id="s1",
            name="Test Strategy",
            symbol="SPY",
            direction="long",
            hold_days=2,
            profit=1,
            description="test",
            proxy_symbol=None,
            confirm_symbols=[],
            conditions=[],
            sell_conditions=[],
        )
        portfolio = SimpleNamespace(
            id="p1",
            name="Combo",
            strategy_ids=["s1"],
            overlap_mode="first_signal_only",
            proxy_symbol=None,
        )
        bulk_data = pd.DataFrame()

        with patch("api.scan_service.LEGACY_BUY_SIGNALS", []):
            with patch("api.scan_service.dt.get_bulk_data", return_value=bulk_data) as get_bulk_data:
                with patch("api.scan_service._prepare_symbol_frame", return_value=data):
                    with patch("api.scan_service.list_strategies", return_value=[strategy]):
                        with patch("api.scan_service.list_portfolios", return_value=[portfolio]):
                            with patch("api.scan_service.bt.build_symbol_dataset", return_value={"SPY": data}):
                                with patch("api.scan_service._portfolio_scan_row") as portfolio_row:
                                    portfolio_row.return_value = {
                                        "id": "portfolio:p1",
                                        "source": "portfolio",
                                        "strategy_id": None,
                                        "portfolio_id": "p1",
                                        "symbol": "SPY",
                                        "signal": "Combo",
                                        "buy_signal": False,
                                        "hold_long": False,
                                        "sell_signal": False,
                                        "days": 2,
                                        "profit": 1,
                                        "trade_pnl": 0.0,
                                        "kelly": None,
                                        "description": "Portfolio",
                                    }
                                    with patch("api.scan_service.execute_saved_strategy") as execute_saved:
                                        execute_saved.return_value = _sample_executed(
                                            trade_out=False,
                                            trade_pnl=0.0,
                                        )
                                        run_scan()

        get_bulk_data.assert_called_once()

    def test_builder_scan_row_includes_condition_snapshot(self):
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-02", periods=3, freq="B"),
                "Close": [100.0, 99.0, 102.0],
                "SMA200": [101.0, 100.5, 99.5],
                "RSI2": [18.0, 22.0, 30.0],
            }
        )
        strategy = SimpleNamespace(
            id="s1",
            name="Test Strategy",
            symbol="SPY",
            direction="long",
            hold_days=2,
            profit=1,
            description="",
            proxy_symbol=None,
            confirm_symbols=[],
            hold_on_buy_signal=False,
            conditions=[
                SimpleNamespace(
                    model_dump=lambda: {"left": "Close", "operator": "<", "right": "SMA200", "logic": "AND"}
                ),
                SimpleNamespace(
                    model_dump=lambda: {"left": "RSI2", "operator": "<=", "right": "30", "logic": "AND"}
                ),
            ],
            sell_conditions=[],
        )

        with patch("api.scan_service.execute_saved_strategy") as execute_saved:
            execute_saved.return_value = _sample_executed(trade_out=False, trade_pnl=0.0)
            with patch("api.scan_service.list_strategies", return_value=[strategy]):
                row = _builder_scan_row(strategy, data)

        self.assertEqual(row["source"], "builder")
        self.assertEqual(row["condition_total_count"], 2)
        self.assertEqual(row["condition_passed_count"], 1)
        self.assertEqual(row["condition_as_of"], "2024-01-04")
        self.assertEqual(len(row["condition_snapshot"]), 2)
        self.assertFalse(row["condition_snapshot"][0]["passed"])
        self.assertTrue(row["condition_snapshot"][1]["passed"])
        self.assertIsNone(row["sell_condition_snapshot"])
        self.assertIsNone(row["sell_condition_passed_count"])
        self.assertIsNone(row["sell_condition_total_count"])

    def test_builder_scan_row_includes_sell_condition_snapshot(self):
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-02", periods=3, freq="B"),
                "Close": [100.0, 99.0, 102.0],
                "SMA200": [101.0, 100.5, 99.5],
                "RSI2": [18.0, 22.0, 85.0],
            }
        )
        strategy = SimpleNamespace(
            id="s2",
            name="Entry Exit Strategy",
            symbol="SPY",
            direction="long",
            hold_days=2,
            profit=1,
            description="",
            proxy_symbol=None,
            confirm_symbols=[],
            hold_on_buy_signal=False,
            conditions=[
                SimpleNamespace(
                    model_dump=lambda: {"left": "RSI2", "operator": "<=", "right": "30", "logic": "AND"}
                ),
            ],
            sell_conditions=[
                SimpleNamespace(
                    model_dump=lambda: {"left": "RSI2", "operator": ">=", "right": "80", "logic": "AND"}
                ),
                SimpleNamespace(
                    model_dump=lambda: {"left": "Close", "operator": ">", "right": "SMA200", "logic": "AND"}
                ),
            ],
        )

        with patch("api.scan_service.execute_saved_strategy") as execute_saved:
            execute_saved.return_value = _sample_executed(trade_out=False, trade_pnl=0.0)
            with patch("api.scan_service.list_strategies", return_value=[strategy]):
                row = _builder_scan_row(strategy, data)

        self.assertEqual(row["sell_condition_total_count"], 2)
        self.assertEqual(row["sell_condition_passed_count"], 2)
        self.assertEqual(len(row["sell_condition_snapshot"]), 2)
        self.assertTrue(row["sell_condition_snapshot"][0]["passed"])
        self.assertTrue(row["sell_condition_snapshot"][1]["passed"])
        self.assertEqual(row["condition_total_count"], 1)
        self.assertFalse(row["condition_snapshot"][0]["passed"])

if __name__ == "__main__":
    unittest.main()
