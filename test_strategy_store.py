import tempfile
import unittest
from pathlib import Path

from api.schemas import BuilderCondition, SaveStrategyRequest
from api.strategy_store import create_strategy, list_strategies


class StrategyStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.temp_dir.name) / "strategies.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_and_list_strategies(self):
        request = SaveStrategyRequest(
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
        saved = create_strategy(request, store_path=self.store_path)
        self.assertTrue(saved.id)
        self.assertEqual(saved.name, "RSI Dip")
        self.assertEqual(saved.symbol, "SPY")

        strategies = list_strategies(store_path=self.store_path)
        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0].id, saved.id)


if __name__ == "__main__":
    unittest.main()
