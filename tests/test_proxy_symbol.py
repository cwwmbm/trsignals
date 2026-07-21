import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import backtest as bt
from api.builder_strategy import draft_to_saved_strategy
from api.proxy_symbol import execute_with_proxy, proxy_column, with_proxy_description
from api.scan_service import execute_saved_strategy
from api.schemas import BuilderBacktestRequest, BuilderCondition
from api.strategy_store import _normalize_proxy_symbol
from backtest_runners import attach_proxy_column


def _signal_frame(*, signal_close, proxy_close, buy_at=(5,)) -> pd.DataFrame:
    rows = len(signal_close)
    buy = np.zeros(rows, dtype=bool)
    for index in buy_at:
        buy[index] = True
    signal = np.asarray(signal_close, dtype=float)
    proxy = np.asarray(proxy_close, dtype=float)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=rows, freq="B"),
            "Close": signal,
            "%Change": np.r_[0.0, np.diff(signal) / signal[:-1]],
            "SOXX": proxy,
            "Buy": buy,
            "Sell": False,
        }
    )


class ProxySymbolHelperTests(unittest.TestCase):
    def test_normalize_proxy_symbol_returns_none_when_empty_or_same(self):
        self.assertIsNone(_normalize_proxy_symbol("SPY", None))
        self.assertIsNone(_normalize_proxy_symbol("SPY", ""))
        self.assertIsNone(_normalize_proxy_symbol("SPY", "spy"))

    def test_normalize_proxy_symbol_uppercases_value(self):
        self.assertEqual(_normalize_proxy_symbol("SPY", "soxx"), "SOXX")

    def test_proxy_column_reads_strategy_field(self):
        strategy = SimpleNamespace(symbol="SPY", proxy_symbol="SOXX")
        self.assertEqual(proxy_column(strategy), "SOXX")

    def test_with_proxy_description_adds_tag(self):
        strategy = SimpleNamespace(symbol="SPY", proxy_symbol="SOXX")
        description = with_proxy_description("Close > SMA200", strategy)
        self.assertIn("[SPY signals, SOXX trade]", description)
        self.assertIn("Close > SMA200", description)


class LongStratProxyTests(unittest.TestCase):
    def test_long_strat_uses_proxy_prices_for_pnl(self):
        signal_close = [100, 100, 100, 100, 100, 100, 100, 100]
        proxy_close = [50, 50, 50, 55, 55, 55, 60, 60]
        data = _signal_frame(signal_close=signal_close, proxy_close=proxy_close, buy_at=(3,))

        without_proxy = bt.long_strat(data.copy(), days=4, prof_closes=99, is_long=True)
        with_proxy = bt.long_strat(data.copy(), days=4, prof_closes=99, is_long=True, pnl_column="SOXX")

        self.assertEqual(without_proxy["RollingPnL"].iloc[-1], 15000)
        self.assertGreater(with_proxy["RollingPnL"].iloc[-1], 15000)

    def test_execute_strategy_without_proxy_matches_close_based_pnl(self):
        data = _signal_frame(
            signal_close=[100, 101, 102, 103, 104, 105],
            proxy_close=[50, 60, 70, 80, 90, 100],
            buy_at=(1,),
        )
        baseline = bt.execute_strategy(data.copy(), days=3, profit=99, is_long=True)
        explicit = bt.execute_strategy(data.copy(), days=3, profit=99, is_long=True, pnl_column=None)
        self.assertEqual(baseline["RollingPnL"].iloc[-1], explicit["RollingPnL"].iloc[-1])

    @patch("stats.ExcludeBestReturnYear", False)
    def test_run_indicator_threshold_keeps_pnl_column_when_hold_days_positive(self):
        data = _signal_frame(
            signal_close=[100, 100, 100, 100, 105, 105, 110, 110],
            proxy_close=[50, 50, 50, 55, 55, 60, 60, 65],
            buy_at=(3,),
        )
        data["RSI2"] = [10, 10, 10, 10, 40, 40, 40, 40]
        row = bt._run_indicator_threshold(
            data,
            days_in_trade=2,
            profitable_close=99,
            is_long=True,
            column_name="RSI2",
            buy_sell="Buy",
            condition="less",
            value=20,
            include_yearly=False,
            pnl_column="SOXX",
        )
        self.assertEqual(row["Indicator"], "RSI2")
        self.assertGreater(row["Trades"], 0)


class ExecuteSavedStrategyProxyTests(unittest.TestCase):
    def test_execute_saved_strategy_uses_proxy_execution(self):
        data = _sample_builder_data()
        strategy = SimpleNamespace(
            symbol="SPY",
            proxy_symbol="SOXX",
            direction="long",
            hold_days=2,
            profit=1,
            conditions=[SimpleNamespace(model_dump=lambda: {"left": "RSI2", "operator": "<=", "right": "20", "logic": "AND"})],
            sell_conditions=[],
            confirm_symbols=[],
        )

        with patch("api.scan_service.execute_with_proxy") as execute_with_proxy:
            execute_with_proxy.return_value = data
            result = execute_saved_strategy(data, strategy, years=25)

        execute_with_proxy.assert_called_once()
        self.assertIs(result, data)

    def test_draft_to_saved_strategy_preserves_proxy_symbol(self):
        saved = draft_to_saved_strategy(
            BuilderBacktestRequest(
                symbol="SPY",
                proxy_symbol="SOXX",
                conditions=[BuilderCondition(left="Close", operator=">", right="SMA200", logic="AND")],
            )
        )
        self.assertEqual(saved.proxy_symbol, "SOXX")


def _sample_builder_data(rows: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=rows, freq="B"),
            "Close": np.linspace(100, 110, rows),
            "RSI2": np.linspace(10, 30, rows),
            "Buy": False,
            "Sell": False,
        }
    )


class ExecuteWithProxyTests(unittest.TestCase):
    def test_attach_proxy_column_aligns_by_date_not_index_labels(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        # Non-zero-based index, like scan frames after bulk extract.
        data = pd.DataFrame(
            {
                "Date": dates,
                "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
            },
            index=[10, 11, 12, 13, 14],
        )
        proxy_close = pd.Series(
            [50.0, 51.0, 52.0, 53.0, 54.0],
            index=dates,
            name="SOXX",
        )

        with patch("api.market_data_cache.load_close_column", return_value=None):
            with patch("getdata._bulk_close", return_value=proxy_close):
                merged = attach_proxy_column(data.copy(), "SOXX", years=1, bulk_data=object(), use_cache=False)

        pd.testing.assert_series_equal(
            merged["SOXX"],
            pd.Series([50.0, 51.0, 52.0, 53.0, 54.0], index=[10, 11, 12, 13, 14], name="SOXX"),
        )

    def test_symbol_confirmation_sweep_attaches_proxy_column(self):
        rows = 8
        dates = pd.date_range("2020-01-01", periods=rows, freq="B")
        symbol_data = {
            "SPY": pd.DataFrame(
                {
                    "Date": dates,
                    "Close": np.linspace(100, 108, rows),
                    "%Change": np.r_[0.0, np.diff(np.linspace(100, 108, rows)) / np.linspace(100, 108, rows)[:-1]],
                }
            ),
            "QQQ": pd.DataFrame(
                {
                    "Date": dates,
                    "Close": np.linspace(200, 208, rows),
                    "%Change": np.r_[0.0, np.diff(np.linspace(200, 208, rows)) / np.linspace(200, 208, rows)[:-1]],
                }
            ),
        }

        def buy_signal(data, symbol):
            buy = pd.Series(False, index=data.index)
            buy.iloc[2] = True
            return buy, False, 2, 1, "test", "", True, False

        executed = symbol_data["SPY"].copy()
        executed["RollingPnL"] = 15000.0
        executed["LongTradeOut"] = False
        executed["TradePnL"] = 0.0
        executed["Drawdown"] = 0.0
        executed["Buy"] = False
        executed["Sell"] = False

        with patch("backtest.load_symbol_dataset", return_value=symbol_data):
            with patch("backtest.execute_strategy", return_value=executed) as execute_strategy:
                with patch("backtest_runners.attach_proxy_column") as attach_proxy:
                    attach_proxy.side_effect = lambda frame, proxy, **_kwargs: frame.assign(**{proxy: np.linspace(50, 58, len(frame))})
                    bt.backtest_symbol_confirmation_sweep(
                        buy_signal,
                        "SPY",
                        ["SPY", "QQQ"],
                        years=1,
                        confirm_sets=[[]],
                        pnl_column="SOXX",
                    )

        attach_proxy.assert_called_once()
        self.assertEqual(attach_proxy.call_args.args[1], "SOXX")
        execute_strategy.assert_called_once()
        self.assertEqual(execute_strategy.call_args.kwargs["pnl_column"], "SOXX")

    def test_execute_with_proxy_passes_pnl_column(self):
        data = _signal_frame(
            signal_close=[100, 100, 100, 100, 100],
            proxy_close=[50, 50, 55, 55, 60],
            buy_at=(1,),
        )
        strategy = SimpleNamespace(symbol="SPY", proxy_symbol="SOXX")

        with patch("api.proxy_symbol.attach_proxy_column", side_effect=lambda frame, *_args, **_kwargs: frame):
            with patch("api.proxy_symbol.bt.execute_strategy") as execute_strategy:
                execute_strategy.return_value = data
                execute_with_proxy(data.copy(), strategy, 2, 1, True, years=25)

        execute_strategy.assert_called_once()
        self.assertEqual(execute_strategy.call_args.kwargs["pnl_column"], "SOXX")


if __name__ == "__main__":
    unittest.main()
