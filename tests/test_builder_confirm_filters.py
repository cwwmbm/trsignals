import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import backtest as bt
from api.builder_strategy import (
    _confirm_strategy_for_cross_symbol,
    apply_primary_entry_filters,
    prepare_builder_refine_frame,
)
from api.schemas import BuilderCondition, SavedStrategy
from api.strategy_compiler import split_entry_conditions_for_confirm


def _frame(dates, rsi_values):
    closes = np.linspace(100.0, 100.0 + len(dates) - 1, len(dates))
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": np.full(len(dates), 1000.0),
            "%Change": np.zeros(len(dates)),
            "RSI2RiskBreadth": rsi_values,
        }
    )


def _stub_signal(buy_masks, *, days=3, profit=1):
    def signal(data, symbol):
        mask = buy_masks[symbol]
        length = len(data)
        buy = pd.Series(mask[:length], index=data.index)
        sell = pd.Series([False] * length, index=data.index)
        return buy, sell, days, profit, "test", None, True, False

    signal.__name__ = "stub_signal"
    return signal


def _saved_strategy(**overrides) -> SavedStrategy:
    payload = {
        "id": "draft",
        "name": "Confirm split test",
        "symbol": "SOXX",
        "direction": "long",
        "hold_days": 3,
        "profit": 1,
        "description": "",
        "conditions": [
            BuilderCondition(left="strategy:sig16", operator="is true", right="", logic="AND"),
            BuilderCondition(left="RSI2RiskBreadth", operator=">=", right="30", logic="AND"),
        ],
        "sell_conditions": [],
        "confirm_symbols": ["SMH", "QQQ"],
        "created_at": "",
        "updated_at": "",
    }
    payload.update(overrides)
    return SavedStrategy(**payload)


class BuilderConfirmPrimaryFilterTests(unittest.TestCase):
    def test_split_entry_conditions_for_confirm(self):
        conditions = [
            BuilderCondition(left="strategy:sig16", operator="is true", right="", logic="AND"),
            BuilderCondition(left="RSI2RiskBreadth", operator=">=", right="30", logic="AND"),
        ]
        confirm, primary = split_entry_conditions_for_confirm(conditions)
        self.assertEqual(confirm[0]["left"], "strategy:sig16")
        self.assertEqual(primary[0]["left"], "RSI2RiskBreadth")

    def test_split_keeps_symbol_local_conditions_in_confirm(self):
        conditions = [
            BuilderCondition(left="High", operator=">", right="CloseLag1", logic="AND"),
            BuilderCondition(left="IBR", operator="<=", right="0.5", logic="AND"),
            BuilderCondition(left="RSI2RiskBreadth", operator=">=", right="30", logic="AND"),
        ]
        confirm, primary = split_entry_conditions_for_confirm(conditions)
        self.assertEqual([c["left"] for c in confirm], ["High", "IBR"])
        self.assertEqual([c["left"] for c in primary], ["RSI2RiskBreadth"])

    def test_split_keeps_adx_on_confirm_path(self):
        conditions = [
            BuilderCondition(left="CloseLag1", operator="<=", right="CloseLag3", logic="AND"),
            BuilderCondition(left="IBR", operator="<=", right="0.4", logic="AND"),
            BuilderCondition(left="ADX14", operator=">=", right="25", logic="AND"),
        ]
        confirm, primary = split_entry_conditions_for_confirm(conditions)
        self.assertEqual([c["left"] for c in confirm], ["CloseLag1", "IBR", "ADX14"])
        self.assertEqual(primary, [])

    def test_confirm_strategy_keeps_only_signal_conditions(self):
        strategy = _saved_strategy()
        confirm_strategy, primary_filters = _confirm_strategy_for_cross_symbol(strategy)
        self.assertEqual(len(confirm_strategy.conditions), 1)
        self.assertEqual(confirm_strategy.conditions[0].left, "strategy:sig16")
        self.assertEqual(len(primary_filters), 1)
        self.assertEqual(primary_filters[0]["left"], "RSI2RiskBreadth")

    def test_confirm_strategy_splits_raw_signal16_plus_breadth(self):
        strategy = _saved_strategy(
            conditions=[
                BuilderCondition(left="High", operator=">", right="CloseLag1", logic="AND"),
                BuilderCondition(left="IBR", operator="<=", right="0.5", logic="AND"),
                BuilderCondition(left="RSI2RiskBreadth", operator=">=", right="30", logic="AND"),
            ]
        )
        confirm_strategy, primary_filters = _confirm_strategy_for_cross_symbol(strategy)
        self.assertEqual(
            [c.left for c in confirm_strategy.conditions],
            ["High", "IBR"],
        )
        self.assertEqual([c["left"] for c in primary_filters], ["RSI2RiskBreadth"])

    def test_primary_filter_applied_after_cross_symbol_confirm(self):
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        symbol_data = {
            "SOXX": _frame(dates, [25, 35, 40]),
            "SMH": _frame(dates, [25, 35, 40]),
            "QQQ": _frame(dates, [40, 40, 40]),
        }
        signal = _stub_signal(
            {
                "SOXX": [True, True, True],
                "SMH": [True, False, False],
                "QQQ": [True, True, True],
            }
        )

        frame, *_ = bt.apply_cross_symbol_signal(signal, "SOXX", ["SMH", "QQQ"], symbol_data)
        self.assertTrue(frame["Buy"].tolist())

        filtered = apply_primary_entry_filters(
            frame,
            [{"left": "RSI2RiskBreadth", "operator": ">=", "right": "30", "logic": "AND"}],
        )
        self.assertEqual(filtered["Buy"].tolist(), [False, True, False])

    def test_embedded_filter_changes_confirm_activity(self):
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        symbol_data = {
            "SOXX": _frame(dates, [25, 35, 40]),
            "SMH": _frame(dates, [25, 35, 40]),
            "QQQ": _frame(dates, [40, 40, 40]),
        }

        def signal_with_filter(data, symbol):
            length = len(data)
            base = {
                "SOXX": [True, True, True],
                "SMH": [True, False, False],
                "QQQ": [True, True, True],
            }[symbol]
            buy = pd.Series(base[:length], index=data.index)
            buy = buy & (data["RSI2RiskBreadth"] >= 30)
            sell = pd.Series([False] * length, index=data.index)
            return buy, sell, 3, 1, "test", None, True, False

        signal_with_filter.__name__ = "embedded_filter"
        embedded, *_ = bt.apply_cross_symbol_signal(
            signal_with_filter,
            "SOXX",
            ["SMH", "QQQ"],
            symbol_data,
        )
        self.assertEqual(embedded["Buy"].tolist(), [False, False, False])

    def test_prepare_builder_refine_frame_matches_sweep_filter_order(self):
        strategy = _saved_strategy()
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        symbol_data = {
            "SOXX": _frame(dates, [25, 35, 40]),
            "SMH": _frame(dates, [25, 35, 40]),
            "QQQ": _frame(dates, [40, 40, 40]),
        }
        signal = _stub_signal(
            {
                "SOXX": [True, True, True],
                "SMH": [True, False, False],
                "QQQ": [True, True, True],
            }
        )

        with patch("api.builder_strategy.bt.load_symbol_dataset", return_value=symbol_data):
            with patch("api.builder_strategy.builder_signal_callable", return_value=signal):
                frame, *_ = prepare_builder_refine_frame(strategy)

        self.assertEqual(frame["Buy"].tolist(), [False, True, False])


if __name__ == "__main__":
    unittest.main()
