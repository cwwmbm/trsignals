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
