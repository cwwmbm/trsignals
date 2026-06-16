import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from api.builder_strategy import (
    backtest_builder_signal_combinations,
    backtest_builder_signal_sweep,
    combine_builder_buy_masks,
    draft_to_saved_strategy,
)
from api.schemas import BuilderBacktestRequest, BuilderCondition, BuilderRefineRequest, SaveStrategyRequest
from api.services import run_builder_refine
from api.strategy_store import create_strategy, get_strategy_by_id


def _draft_request(**overrides) -> BuilderBacktestRequest:
    payload = {
        "symbol": "SPY",
        "years": 25,
        "direction": "long",
        "hold_days": 2,
        "profit": 1,
        "name": "Draft RSI",
        "description": "",
        "conditions": [
            BuilderCondition(left="RSI2", operator="<=", right="20", logic="AND"),
        ],
        "sell_conditions": [],
    }
    payload.update(overrides)
    return BuilderBacktestRequest(**payload)


def _sample_data(rows: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=rows, freq="B"),
            "Close": np.linspace(100, 110, rows),
            "RSI2": np.linspace(10, 30, rows),
            "SMA200": np.linspace(99, 108, rows),
        }
    )


class BuilderStrategyHelperTests(unittest.TestCase):
    def test_draft_to_saved_strategy_uses_name_and_conditions(self):
        saved = draft_to_saved_strategy(_draft_request())
        self.assertEqual(saved.id, "draft")
        self.assertEqual(saved.name, "Draft RSI")
        self.assertEqual(saved.symbol, "SPY")
        self.assertIn("RSI(2)", saved.description)

    def test_combine_builder_buy_masks_and_mode(self):
        data = _sample_data()
        primary = draft_to_saved_strategy(_draft_request())
        secondary = draft_to_saved_strategy(
            _draft_request(name="Other", conditions=[BuilderCondition(left="RSI2", operator="<=", right="25", logic="AND")])
        )
        buy_and, _, days, profit, description, _, is_long, _ = combine_builder_buy_masks(
            primary, secondary, data, "and"
        )
        buy_or, *_ = combine_builder_buy_masks(primary, secondary, data, "or")
        self.assertEqual(days, 2)
        self.assertEqual(profit, 1)
        self.assertTrue(is_long)
        self.assertIn("AND", description)
        self.assertTrue(buy_or.sum() >= buy_and.sum())


class BuilderRefineServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.temp_dir.name) / "strategies.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_combo_sweep_requires_same_symbol_saved_strategies(self):
        request = BuilderRefineRequest(
            mode="signal-combo-sweep",
            strategy=_draft_request(),
        )
        with patch("api.services.list_strategies", return_value=[]):
            with self.assertRaisesRegex(ValueError, "No saved strategies found for SPY"):
                run_builder_refine(request)

    def test_get_strategy_by_id(self):
        saved = create_strategy(
            SaveStrategyRequest(
                name="Saved RSI",
                symbol="SPY",
                direction="long",
                hold_days=2,
                profit=1,
                conditions=[BuilderCondition(left="RSI2", operator="<=", right="20", logic="AND")],
            ),
            store_path=self.store_path,
        )
        loaded = get_strategy_by_id(saved.id, store_path=self.store_path)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.name, "Saved RSI")

    def test_combo_sweep_uses_all_same_symbol_strategies(self):
        secondary_a = create_strategy(
            SaveStrategyRequest(
                name="Secondary A",
                symbol="SPY",
                direction="long",
                hold_days=2,
                profit=1,
                conditions=[BuilderCondition(left="RSI2", operator="<=", right="25", logic="AND")],
            ),
            store_path=self.store_path,
        )
        secondary_b = create_strategy(
            SaveStrategyRequest(
                name="Secondary B",
                symbol="SPY",
                direction="long",
                hold_days=2,
                profit=1,
                conditions=[BuilderCondition(left="Close", operator="<", right="SMA200", logic="AND")],
            ),
            store_path=self.store_path,
        )
        other_symbol = create_strategy(
            SaveStrategyRequest(
                name="Other Symbol",
                symbol="QQQ",
                direction="long",
                hold_days=2,
                profit=1,
                conditions=[BuilderCondition(left="RSI2", operator="<=", right="25", logic="AND")],
            ),
            store_path=self.store_path,
        )
        request = BuilderRefineRequest(
            mode="signal-combo-sweep",
            strategy=_draft_request(),
        )
        data = _sample_data()

        captured_secondaries = []

        def fake_sweep(_primary, secondaries, *_args, **_kwargs):
            captured_secondaries.extend(secondaries)
            return pd.DataFrame(
                [
                    {"Primary": "Draft RSI", "Secondary": "Secondary A", "Mode": "AND", "Sharpe": 1.2},
                    {"Primary": "Draft RSI", "Secondary": "Secondary A", "Mode": "OR", "Sharpe": 1.0},
                    {"Primary": "Draft RSI", "Secondary": "Secondary B", "Mode": "AND", "Sharpe": 0.9},
                    {"Primary": "Draft RSI", "Secondary": "Secondary B", "Mode": "OR", "Sharpe": 0.8},
                ]
            )

        with patch("api.services.load_ticker_data", return_value=data.copy()):
            with patch("api.services.list_strategies", return_value=[secondary_a, secondary_b, other_symbol]):
                with patch("api.services.backtest_builder_signal_sweep", side_effect=fake_sweep) as combo:
                    rows = run_builder_refine(request)

        self.assertEqual(len(rows), 4)
        combo.assert_called_once()
        self.assertEqual([item.id for item in captured_secondaries], [secondary_a.id, secondary_b.id])

    def test_indicator_sweep_returns_rows(self):
        request = BuilderRefineRequest(
            mode="indicator-sweep",
            strategy=_draft_request(),
            check_both=False,
            check_breadth=False,
        )
        data = _sample_data()

        with patch("api.services.load_ticker_data", return_value=data.copy()):
            with patch("api.services.indicator_tryout") as tryout:
                tryout.return_value = pd.DataFrame(
                    [
                        {"Indicator": "RSI2", "Value": 20, "Sharpe": 1.1},
                        {"Indicator": "RSI5", "Value": 25, "Sharpe": 0.9},
                    ]
                )
                rows = run_builder_refine(request)

        self.assertEqual(len(rows), 2)
        tryout.assert_called_once()


class BuilderComboSweepTests(unittest.TestCase):
    def test_backtest_builder_signal_combinations_produces_four_rows(self):
        primary = draft_to_saved_strategy(_draft_request())
        secondary = draft_to_saved_strategy(
            _draft_request(name="Secondary", conditions=[BuilderCondition(left="RSI2", operator="<=", right="25", logic="AND")])
        )
        data = _sample_data()

        with patch("api.builder_strategy.bt.execute_strategy") as execute_strategy:
            with patch("api.builder_strategy.bt._ranking_metrics") as ranking:
                execute_strategy.side_effect = lambda frame, *_args: frame
                ranking.return_value = {
                    "PnL": 1000,
                    "MaxDD": 10.0,
                    "Trades": 5,
                    "%Pstv": 60.0,
                    "CAGR": 12.0,
                    "Sharpe": 1.0,
                    "Sortino": 1.1,
                    "Yearly": [],
                }
                results = backtest_builder_signal_combinations(primary, secondary, data, "SPY")

        self.assertEqual(len(results), 4)
        self.assertEqual(set(results["Mode"]), {"AND", "OR"})

    def test_backtest_builder_signal_sweep_produces_two_rows_per_secondary(self):
        primary = draft_to_saved_strategy(_draft_request())
        secondaries = [
            draft_to_saved_strategy(
                _draft_request(name="Secondary A", conditions=[BuilderCondition(left="RSI2", operator="<=", right="25", logic="AND")])
            ),
            draft_to_saved_strategy(
                _draft_request(name="Secondary B", conditions=[BuilderCondition(left="Close", operator="<", right="SMA200", logic="AND")])
            ),
        ]
        data = _sample_data()

        with patch("api.builder_strategy.bt.execute_strategy") as execute_strategy:
            with patch("api.builder_strategy.bt._ranking_metrics") as ranking:
                execute_strategy.side_effect = lambda frame, *_args: frame
                ranking.return_value = {
                    "PnL": 1000,
                    "MaxDD": 10.0,
                    "Trades": 5,
                    "%Pstv": 60.0,
                    "CAGR": 12.0,
                    "Sharpe": 1.0,
                    "Sortino": 1.1,
                    "Yearly": [],
                }
                results = backtest_builder_signal_sweep(primary, secondaries, data, "SPY")

        self.assertEqual(len(results), 4)
        self.assertEqual(set(results["Primary"]), {"Draft RSI"})
        self.assertEqual(set(results["Secondary"]), {"Secondary A", "Secondary B"})
        self.assertEqual(set(results["Mode"]), {"AND", "OR"})


if __name__ == "__main__":
    unittest.main()
