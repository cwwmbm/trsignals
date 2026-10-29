"""Scan lanes: Live is the first section, and strategies and portfolios can be filed in it."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from api.portfolio_store import create_portfolio, list_portfolios, update_portfolio
from api.schemas import (
    BuilderCondition,
    SavePortfolioRequest,
    SaveStrategyRequest,
    UpdatePortfolioRequest,
    UpdateStrategyRequest,
)
from api.strategy_store import create_strategy, list_strategies, update_strategy

LANES = ("live", "active", "testing", "archived")


class LaneValueTests(unittest.TestCase):
    def test_every_lane_is_accepted_and_others_are_not(self):
        for lane in LANES:
            self.assertEqual(UpdateStrategyRequest(scan_lane=lane).scan_lane, lane)
            self.assertEqual(UpdatePortfolioRequest(scan_lane=lane).scan_lane, lane)
        for bad in ("Live", "paper", ""):
            with self.assertRaises(ValueError):
                UpdateStrategyRequest(scan_lane=bad)
            with self.assertRaises(ValueError):
                UpdatePortfolioRequest(scan_lane=bad)

    def test_new_things_still_start_in_testing(self):
        with tempfile.TemporaryDirectory() as folder:
            strategy = create_strategy(
                SaveStrategyRequest(
                    name="RSI Dip", symbol="SPY", direction="long", hold_days=2, profit=1,
                    conditions=[BuilderCondition(left="RSI2", operator="<=", right="20", logic="AND")],
                ),
                store_path=Path(folder) / "strategies.json",
            )
        self.assertEqual(strategy.scan_lane, "testing")


class StrategyLiveTests(unittest.TestCase):
    def test_a_strategy_can_be_moved_to_live_and_stays_there(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "strategies.json"
            saved = create_strategy(
                SaveStrategyRequest(
                    name="RSI Dip", symbol="SPY", direction="long", hold_days=2, profit=1, description="Keep me",
                    conditions=[BuilderCondition(left="RSI2", operator="<=", right="20", logic="AND")],
                ),
                store_path=path,
            )
            updated = update_strategy(
                saved.id, UpdateStrategyRequest(scan_lane="live", scan_sort_order=2), store_path=path
            )
            self.assertEqual((updated.scan_lane, updated.scan_sort_order, updated.description), ("live", 2, "Keep me"))
            (loaded,) = list_strategies(store_path=path)
        self.assertEqual(loaded.scan_lane, "live")


class PortfolioLiveTests(unittest.TestCase):
    def test_a_portfolio_can_be_moved_to_live_and_stays_there(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "portfolios.json"
            with patch(
                "api.portfolio_store.get_strategy_by_id",
                return_value=SimpleNamespace(id="s1", name="Long SPY", direction="long"),
            ):
                saved = create_portfolio(SavePortfolioRequest(name="Book", strategy_ids=["s1"]), store_path=path)
                updated = update_portfolio(saved.id, UpdatePortfolioRequest(scan_lane="live"), store_path=path)
                self.assertEqual(updated.scan_lane, "live")
                (loaded,) = list_portfolios(store_path=path)
        self.assertEqual((loaded.scan_lane, loaded.name), ("live", "Book"))


if __name__ == "__main__":
    unittest.main()
