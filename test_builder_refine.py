import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from api.builder_strategy import (
    backtest_builder_signal_combinations,
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
        self.assertIn("RSI2", saved.description)

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

    def test_combo_sweep_requires_secondary_strategy(self):
        request = BuilderRefineRequest(
            mode="signal-combo-sweep",
            strategy=_draft_request(),
        )
        with self.assertRaisesRegex(ValueError, "Secondary strategy"):
            run_builder_refine(request)

    def test_combo_sweep_unknown_secondary_strategy(self):
        request = BuilderRefineRequest(
            mode="signal-combo-sweep",
            strategy=_draft_request(),
            secondary_strategy_id="missing-id",
        )
        with self.assertRaisesRegex(ValueError, "Unknown strategy"):
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

    def test_combo_sweep_returns_four_rows(self):
        secondary = create_strategy(
            SaveStrategyRequest(
                name="Secondary",
                symbol="SPY",
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
            secondary_strategy_id=secondary.id,
        )
        data = _sample_data()

        with patch("api.services.load_ticker_data", return_value=data.copy()):
            with patch("api.services.backtest_builder_signal_combinations") as combo:
                combo.return_value = pd.DataFrame(
                    [
                        {"Primary": "Draft RSI", "Secondary": "Secondary", "Mode": "AND", "Sharpe": 1.2},
                        {"Primary": "Draft RSI", "Secondary": "Secondary", "Mode": "OR", "Sharpe": 1.0},
                        {"Primary": "Secondary", "Secondary": "Draft RSI", "Mode": "AND", "Sharpe": 0.9},
                        {"Primary": "Secondary", "Secondary": "Draft RSI", "Mode": "OR", "Sharpe": 0.8},
                    ]
                )
                rows = run_builder_refine(request)

        self.assertEqual(len(rows), 4)
        combo.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()
