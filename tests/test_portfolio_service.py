import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from api.portfolio_service import (
    StrategySignals,
    simulate_portfolio,
    simulate_portfolio_overlay,
)


def _strategy(strategy_id: str, name: str, symbol: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=strategy_id,
        name=name,
        symbol=symbol,
        direction="long",
        hold_days=2,
        profit=1,
        confirm_symbols=[],
        proxy_symbol=None,
    )


def _signals(
    strategy_id: str,
    name: str,
    symbol: str,
    events: list[tuple[str, str]],
) -> StrategySignals:
    dates = pd.to_datetime([event[0] for event in events])
    trade_in = np.array([event[1] == "buy" for event in events], dtype=bool)
    trade_out = np.array([event[1] == "sell" for event in events], dtype=bool)
    hold = np.zeros(len(events), dtype=bool)
    in_trade = False
    for index, event in enumerate(events):
        if index > 0:
            in_trade = (in_trade and not trade_out[index - 1]) or trade_in[index - 1]
        hold[index] = in_trade
    return StrategySignals(
        strategy_id=strategy_id,
        name=name,
        symbol=symbol,
        dates=dates,
        long_trade_in=trade_in,
        hold_long=hold,
        long_trade_out=trade_out,
    )


class PortfolioOverlayTests(unittest.TestCase):
    def setUp(self):
        self.strategy_a = _strategy("a", "Strategy A", "AAA")
        self.strategy_b = _strategy("b", "Strategy B", "BBB")
        self.dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
        self.signals_by_id = {
            "a": _signals(
                "a",
                "Strategy A",
                "AAA",
                [
                    ("2024-01-01", "buy"),
                    ("2024-01-02", "hold"),
                    ("2024-01-03", "hold"),
                    ("2024-01-04", "sell"),
                ],
            ),
            "b": _signals(
                "b",
                "Strategy B",
                "BBB",
                [
                    ("2024-01-01", "hold"),
                    ("2024-01-02", "hold"),
                    ("2024-01-03", "buy"),
                    ("2024-01-04", "hold"),
                ],
            ),
        }
        track = pd.Series([0.01, 0.0, 0.02, -0.01], index=self.dates)
        self.track_changes = {"AAA": track, "BBB": track * 2}
        self.close_series = {
            "AAA": pd.Series([100.0, 100.0, 102.0, 101.0], index=self.dates),
            "BBB": pd.Series([50.0, 50.0, 51.0, 51.0], index=self.dates),
            "SPY": pd.Series([400.0, 401.0, 402.0, 401.0], index=self.dates),
        }

    def test_first_signal_only_exits_when_leader_exits(self):
        frame = simulate_portfolio_overlay(
            [self.strategy_a, self.strategy_b],
            self.signals_by_id,
            overlap_mode="first_signal_only",
            global_proxy=None,
            track_changes=self.track_changes,
            close_series=self.close_series,
            strategy_order=["a", "b"],
        )
        self.assertFalse(frame["HoldLong"].iloc[0])
        self.assertTrue(frame["HoldLong"].iloc[2])
        self.assertTrue(frame["HoldLong"].iloc[3])
        self.assertTrue(frame["LongTradeOut"].iloc[3])

    def test_hold_until_all_exit_keeps_position_when_one_strategy_still_active(self):
        frame = simulate_portfolio_overlay(
            [self.strategy_a, self.strategy_b],
            self.signals_by_id,
            overlap_mode="hold_until_all_exit",
            global_proxy=None,
            track_changes=self.track_changes,
            close_series=self.close_series,
            strategy_order=["a", "b"],
        )
        self.assertTrue(frame["HoldLong"].iloc[2])
        self.assertTrue(frame["HoldLong"].iloc[3])
        self.assertFalse(frame["LongTradeOut"].iloc[3])

    def test_same_day_exit_and_entry_stays_in_trade(self):
        signals_by_id = {
            "a": _signals(
                "a",
                "Strategy A",
                "AAA",
                [
                    ("2024-01-01", "buy"),
                    ("2024-01-02", "sell"),
                ],
            ),
            "b": _signals(
                "b",
                "Strategy B",
                "BBB",
                [
                    ("2024-01-01", "hold"),
                    ("2024-01-02", "buy"),
                ],
            ),
        }
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        track = pd.Series([0.01, 0.01], index=dates)
        frame = simulate_portfolio_overlay(
            [self.strategy_a, self.strategy_b],
            signals_by_id,
            overlap_mode="hold_until_all_exit",
            global_proxy=None,
            track_changes={"AAA": track, "BBB": track},
            close_series={
                "AAA": pd.Series([100.0, 101.0], index=dates),
                "BBB": pd.Series([50.0, 51.0], index=dates),
            },
            strategy_order=["a", "b"],
        )
        self.assertTrue(frame["HoldLong"].iloc[1])
        self.assertFalse(frame["LongTradeOut"].iloc[1])

    def test_global_proxy_used_for_returns(self):
        frame = simulate_portfolio_overlay(
            [self.strategy_a],
            {"a": self.signals_by_id["a"]},
            overlap_mode="first_signal_only",
            global_proxy="SPY",
            track_changes={
                "SPY": pd.Series([0.0, 0.05, 0.0, 0.0], index=self.dates),
                "AAA": pd.Series([0.01, 0.0, 0.0, 0.0], index=self.dates),
            },
            close_series=self.close_series,
            strategy_order=["a"],
        )
        self.assertAlmostEqual(frame["TrackChange"].iloc[1], 0.05)
        self.assertGreater(frame["RollingPnL"].iloc[1], 15000)

    def test_saved_strategy_proxy_used_when_no_global_proxy(self):
        strategy = _strategy("a", "Strategy A", "AAA")
        strategy.proxy_symbol = "SPY"
        frame = simulate_portfolio_overlay(
            [strategy],
            {"a": self.signals_by_id["a"]},
            overlap_mode="first_signal_only",
            global_proxy=None,
            track_changes={
                "AAA": pd.Series([0.01, 0.0, 0.0, 0.0], index=self.dates),
                "SPY": pd.Series([0.0, 0.05, 0.0, 0.0], index=self.dates),
            },
            close_series=self.close_series,
            strategy_order=["a"],
        )
        self.assertAlmostEqual(frame["TrackChange"].iloc[1], 0.05)
        self.assertGreater(frame["RollingPnL"].iloc[1], 15000)

    def test_days_in_trade_starts_after_buy_signal(self):
        frame = simulate_portfolio_overlay(
            [self.strategy_a],
            {"a": self.signals_by_id["a"]},
            overlap_mode="first_signal_only",
            global_proxy=None,
            track_changes=self.track_changes,
            close_series=self.close_series,
            strategy_order=["a"],
        )
        self.assertEqual(frame["DaysInTrade"].tolist(), [0, 1, 2, 3])
        self.assertTrue(frame["LongTradeIn"].iloc[0])
        self.assertFalse(frame["HoldLong"].iloc[0])
        self.assertTrue(frame["HoldLong"].iloc[1])

    def test_mixed_symbols_use_portfolio_equity_in_trades_table(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
        strategy_a = _strategy("a", "Strategy A", "AAA")
        strategy_b = _strategy("b", "Strategy B", "BBB")
        signals_by_id = {
            "a": _signals(
                "a",
                "Strategy A",
                "AAA",
                [
                    ("2024-01-01", "buy"),
                    ("2024-01-02", "sell"),
                    ("2024-01-03", "hold"),
                    ("2024-01-04", "hold"),
                ],
            ),
            "b": _signals(
                "b",
                "Strategy B",
                "BBB",
                [
                    ("2024-01-01", "hold"),
                    ("2024-01-02", "hold"),
                    ("2024-01-03", "buy"),
                    ("2024-01-04", "hold"),
                ],
            ),
        }
        track = pd.Series([0.0, 0.10, 0.0, 0.20], index=dates)
        frame = simulate_portfolio_overlay(
            [strategy_a, strategy_b],
            signals_by_id,
            overlap_mode="first_signal_only",
            global_proxy=None,
            track_changes={"AAA": track, "BBB": pd.Series([0.0, 0.0, 0.0, 0.20], index=dates)},
            close_series={
                "AAA": pd.Series([100.0, 110.0, 110.0, 110.0], index=dates),
                "BBB": pd.Series([50.0, 50.0, 50.0, 60.0], index=dates),
            },
            strategy_order=["a", "b"],
        )
        from api.serializers import trade_payload

        trades = trade_payload(frame, portfolio_equity=True)
        closed = [trade for trade in trades if trade["status"] == "Closed"]
        self.assertEqual(len(closed), 1)
        trade = closed[0]
        self.assertGreater(trade["entry_price"], 1000)
        self.assertGreater(trade["exit_price"], 1000)
        self.assertAlmostEqual(trade["trade_pnl"], 0.10, places=4)
        self.assertAlmostEqual(
            trade["exit_price"] / trade["entry_price"] - 1,
            trade["trade_pnl"],
            places=4,
        )


class PortfolioSimulationTests(unittest.TestCase):
    @patch("api.portfolio_service._load_portfolio_market_data")
    @patch("api.portfolio_service.get_strategy_by_id")
    def test_rejects_short_strategy(self, mock_get_strategy, mock_load_market_data):
        mock_get_strategy.return_value = SimpleNamespace(
            id="short-1",
            name="Short",
            symbol="SPY",
            direction="short",
            hold_days=2,
            profit=1,
            confirm_symbols=[],
            proxy_symbol=None,
        )
        request = SimpleNamespace(
            strategy_ids=["short-1"],
            overlap_mode="first_signal_only",
            proxy_symbol=None,
            years=25,
        )
        with self.assertRaisesRegex(ValueError, "Short strategies are not supported"):
            simulate_portfolio(request)
        mock_load_market_data.assert_not_called()

    @patch("api.portfolio_service._load_strategy_signals")
    @patch("api.portfolio_service._load_portfolio_market_data")
    @patch("api.portfolio_service.get_strategy_by_id")
    def test_returns_detailed_payload(
        self,
        mock_get_strategy,
        mock_load_market_data,
        mock_load_signals,
    ):
        dates = pd.to_datetime(
            ["2023-01-03", "2023-06-01", "2024-01-03", "2024-06-01"]
        )
        strategy = _strategy("a", "Strategy A", "AAA")
        mock_get_strategy.return_value = strategy
        mock_load_market_data.return_value = (
            pd.DataFrame(),
            {"AAA": pd.DataFrame({"Date": dates, "Close": [100, 101, 103, 102]})},
            {"AAA": pd.Series([0.01, 0.01, 0.02, -0.01], index=dates)},
            {"AAA": pd.Series([100.0, 101.0, 103.0, 102.0], index=dates)},
        )
        mock_load_signals.return_value = StrategySignals(
            strategy_id="a",
            name="Strategy A",
            symbol="AAA",
            dates=dates,
            long_trade_in=np.array([True, False, False, False]),
            hold_long=np.array([True, True, True, False]),
            long_trade_out=np.array([False, False, False, True]),
        )
        request = SimpleNamespace(
            strategy_ids=["a"],
            overlap_mode="first_signal_only",
            proxy_symbol=None,
            years=25,
        )
        payload = simulate_portfolio(request)
        self.assertIn("summary", payload)
        self.assertIn("equity_curve", payload)
        self.assertIn("Portfolio", payload["summary"]["description"])
        mock_load_market_data.assert_called_once()

    @patch("api.portfolio_service._load_strategy_signals")
    @patch("api.portfolio_service._load_portfolio_market_data")
    @patch("api.portfolio_service.get_strategy_by_id")
    def test_downloads_market_data_once_for_multiple_strategies(
        self,
        mock_get_strategy,
        mock_load_market_data,
        mock_load_signals,
    ):
        dates = pd.to_datetime(["2023-01-03", "2024-01-03"])
        strategy_a = _strategy("a", "Strategy A", "AAA")
        strategy_b = _strategy("b", "Strategy B", "BBB")
        mock_get_strategy.side_effect = [strategy_a, strategy_b]
        mock_load_market_data.return_value = (
            pd.DataFrame(),
            {},
            {"AAA": pd.Series([0.01, 0.0], index=dates), "BBB": pd.Series([0.02, 0.0], index=dates), "SOXX": pd.Series([0.03, 0.0], index=dates)},
            {"AAA": pd.Series([100.0, 101.0], index=dates), "BBB": pd.Series([50.0, 51.0], index=dates), "SOXX": pd.Series([200.0, 201.0], index=dates)},
        )
        mock_load_signals.side_effect = [
            StrategySignals("a", "A", "AAA", dates, np.array([True, False]), np.array([True, False]), np.array([False, True])),
            StrategySignals("b", "B", "BBB", dates, np.array([False, True]), np.array([False, True]), np.array([False, False])),
        ]
        request = SimpleNamespace(
            strategy_ids=["a", "b"],
            overlap_mode="hold_until_all_exit",
            proxy_symbol="SOXX",
            years=25,
        )
        simulate_portfolio(request)
        mock_load_market_data.assert_called_once()
        self.assertEqual(mock_load_signals.call_count, 2)

    @patch("api.portfolio_service.get_strategy_by_id")
    def test_single_strategy_matches_backtest_metrics(self, mock_get_strategy):
        from api.portfolio_service import _load_portfolio_market_data
        from api.scan_service import execute_saved_strategy
        from api.serializers import detailed_backtest_payload
        from api.strategy_store import list_strategies

        strat = next(
            s for s in list_strategies() if "signal16" in s.name.lower() and "confirm" in s.name.lower()
        )
        mock_get_strategy.return_value = strat
        years = 25
        symbols = {strat.symbol.strip().upper(), *(c.strip().upper() for c in (strat.confirm_symbols or []))}
        market_data = _load_portfolio_market_data(symbols, years=years)
        full_data, symbol_dataset, _, _ = market_data

        data = symbol_dataset[strat.symbol.strip().upper()].copy()
        executed = execute_saved_strategy(
            data,
            strat,
            years=years,
            bulk_data=full_data,
            symbol_data=symbol_dataset,
        )
        bt = detailed_backtest_payload(executed, strat.hold_days, strat.profit, strat.name)

        request = SimpleNamespace(
            strategy_ids=[strat.id],
            overlap_mode="first_signal_only",
            proxy_symbol=None,
            years=years,
        )
        with patch(
            "api.portfolio_service._load_portfolio_market_data",
            return_value=market_data,
        ):
            port = simulate_portfolio(request)

        self.assertAlmostEqual(port["summary"]["rolling_pnl"], bt["summary"]["rolling_pnl"], delta=1.0)
        self.assertEqual(port["summary"]["trades"], bt["summary"]["trades"])
        self.assertAlmostEqual(port["summary"]["pct_positive"], bt["summary"]["pct_positive"], delta=0.1)

    @patch("api.portfolio_service.get_strategy_by_id")
    def test_single_strategy_hold_until_all_exit_matches_backtest(self, mock_get_strategy):
        from api.portfolio_service import _load_portfolio_market_data
        from api.scan_service import execute_saved_strategy
        from api.serializers import detailed_backtest_payload
        from api.strategy_store import list_strategies

        strat = next(
            s for s in list_strategies() if "signal16" in s.name.lower() and "confirm" in s.name.lower()
        )
        mock_get_strategy.return_value = strat
        years = 25
        symbols = {strat.symbol.strip().upper(), *(c.strip().upper() for c in (strat.confirm_symbols or []))}
        market_data = _load_portfolio_market_data(symbols, years=years)
        full_data, symbol_dataset, _, _ = market_data

        data = symbol_dataset[strat.symbol.strip().upper()].copy()
        executed = execute_saved_strategy(
            data,
            strat,
            years=years,
            bulk_data=full_data,
            symbol_data=symbol_dataset,
        )
        bt = detailed_backtest_payload(executed, strat.hold_days, strat.profit, strat.name)

        request = SimpleNamespace(
            strategy_ids=[strat.id],
            overlap_mode="hold_until_all_exit",
            proxy_symbol=None,
            years=years,
        )
        with patch(
            "api.portfolio_service._load_portfolio_market_data",
            return_value=market_data,
        ):
            port = simulate_portfolio(request)

        self.assertAlmostEqual(port["summary"]["rolling_pnl"], bt["summary"]["rolling_pnl"], delta=1.0)
        self.assertEqual(port["summary"]["trades"], bt["summary"]["trades"])
        self.assertAlmostEqual(port["summary"]["pct_positive"], bt["summary"]["pct_positive"], delta=0.1)


if __name__ == "__main__":
    unittest.main()
