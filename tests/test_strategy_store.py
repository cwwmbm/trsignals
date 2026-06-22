import tempfile
import unittest
from pathlib import Path

from api.schemas import BuilderCondition, SaveStrategyRequest, UpdateStrategyRequest
from api.strategy_store import create_strategy, delete_strategy, list_strategies, update_strategy


class StrategyStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.temp_dir.name) / "strategies.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_and_list_strategies(self):
        request = self._request()
        saved = create_strategy(request, store_path=self.store_path)
        self.assertTrue(saved.id)
        self.assertEqual(saved.name, "RSI Dip")
        self.assertEqual(saved.symbol, "SPY")

        strategies = list_strategies(store_path=self.store_path)
        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0].id, saved.id)

    def test_update_strategy_description(self):
        saved = create_strategy(self._request(), store_path=self.store_path)
        updated = update_strategy(
            saved.id,
            UpdateStrategyRequest(description="Updated description"),
            store_path=self.store_path,
        )
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.description, "Updated description")
        self.assertEqual(updated.name, "RSI Dip")
        self.assertTrue(updated.updated_at)

    def test_update_unknown_strategy_returns_none(self):
        updated = update_strategy(
            "missing-id",
            UpdateStrategyRequest(description="Updated description"),
            store_path=self.store_path,
        )
        self.assertIsNone(updated)

    def test_delete_strategy(self):
        saved = create_strategy(self._request(), store_path=self.store_path)
        self.assertTrue(delete_strategy(saved.id, store_path=self.store_path))
        self.assertEqual(list_strategies(store_path=self.store_path), [])
        self.assertFalse(delete_strategy(saved.id, store_path=self.store_path))

    def test_hold_on_buy_signal_persists(self):
        request = SaveStrategyRequest(
            name="Hold On Buy",
            symbol="SPY",
            direction="long",
            hold_days=2,
            profit=1,
            description="",
            conditions=[
                BuilderCondition(left="RSI2", operator="<=", right="20", logic="AND"),
            ],
            hold_on_buy_signal=True,
        )
        saved = create_strategy(request, store_path=self.store_path)
        self.assertTrue(saved.hold_on_buy_signal)
        loaded = list_strategies(store_path=self.store_path)[0]
        self.assertTrue(loaded.hold_on_buy_signal)

    def test_confirm_symbols_persist_and_normalize(self):
        request = SaveStrategyRequest(
            name="Confirm Test",
            symbol="SPY",
            direction="long",
            hold_days=2,
            profit=1,
            description="",
            conditions=[
                BuilderCondition(left="RSI2", operator="<=", right="20", logic="AND"),
            ],
            confirm_symbols=["smh", "SPY", "QQQ", "smh"],
        )
        saved = create_strategy(request, store_path=self.store_path)
        self.assertEqual(saved.confirm_symbols, ["SMH", "QQQ"])

        loaded = list_strategies(store_path=self.store_path)[0]
        self.assertEqual(loaded.confirm_symbols, ["SMH", "QQQ"])

    def test_saved_strategy_without_confirm_symbols_defaults_empty(self):
        from api.schemas import SavedStrategy

        payload = {
            "id": "legacy-id",
            "name": "Legacy",
            "symbol": "SPY",
            "direction": "long",
            "hold_days": 2,
            "profit": 1,
            "description": "",
            "conditions": [
                {"left": "RSI2", "operator": "<=", "right": "20", "logic": "AND"},
            ],
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        if hasattr(SavedStrategy, "model_validate"):
            strategy = SavedStrategy.model_validate(payload)
        else:
            strategy = SavedStrategy.parse_obj(payload)
        self.assertEqual(strategy.confirm_symbols, [])

    def test_create_defaults_scan_lane_to_testing(self):
        saved = create_strategy(self._request(), store_path=self.store_path)
        self.assertEqual(saved.scan_lane, "testing")
        self.assertEqual(saved.scan_sort_order, 0)

    def test_update_scan_lane_preserves_description(self):
        saved = create_strategy(self._request(), store_path=self.store_path)
        updated = update_strategy(
            saved.id,
            UpdateStrategyRequest(scan_lane="active"),
            store_path=self.store_path,
        )
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.scan_lane, "active")
        self.assertEqual(updated.description, "Test strategy")

    def test_update_scan_sort_order(self):
        saved = create_strategy(self._request(), store_path=self.store_path)
        updated = update_strategy(
            saved.id,
            UpdateStrategyRequest(scan_sort_order=5),
            store_path=self.store_path,
        )
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.scan_sort_order, 5)

    def test_saved_strategy_without_scan_lane_defaults_testing(self):
        from api.schemas import SavedStrategy

        payload = {
            "id": "legacy-id",
            "name": "Legacy",
            "symbol": "SPY",
            "direction": "long",
            "hold_days": 2,
            "profit": 1,
            "description": "",
            "conditions": [
                {"left": "RSI2", "operator": "<=", "right": "20", "logic": "AND"},
            ],
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        if hasattr(SavedStrategy, "model_validate"):
            strategy = SavedStrategy.model_validate(payload)
        else:
            strategy = SavedStrategy.parse_obj(payload)
        self.assertEqual(strategy.scan_lane, "testing")
        self.assertEqual(strategy.scan_sort_order, 0)

    def _request(self):
        return SaveStrategyRequest(
            name="RSI Dip",
            symbol="SPY",
            direction="long",
            hold_days=2,
            profit=1,
            description="Test strategy",
            conditions=[
                BuilderCondition(left="RSI2", operator="<=", right="20", logic="AND"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
