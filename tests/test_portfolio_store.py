import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api.schemas import SavePortfolioRequest, UpdatePortfolioRequest
from api.portfolio_store import (
    create_portfolio,
    delete_portfolio,
    list_portfolios,
    update_portfolio,
)


class PortfolioStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.temp_dir.name) / "portfolios.json"
        self.strategy_store_path = Path(self.temp_dir.name) / "strategies.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("api.portfolio_store.get_strategy_by_id")
    def test_create_and_list_portfolios(self, mock_get_strategy):
        from types import SimpleNamespace

        mock_get_strategy.return_value = SimpleNamespace(
            id="s1",
            name="Long SPY",
            direction="long",
        )
        request = SavePortfolioRequest(
            name="Core combo",
            description="Two strategies",
            strategy_ids=["s1", "s1"],
            overlap_mode="first_signal_only",
            proxy_symbol="spy",
        )
        saved = create_portfolio(request, store_path=self.store_path)
        self.assertTrue(saved.id)
        self.assertEqual(saved.name, "Core combo")
        self.assertEqual(saved.strategy_ids, ["s1", "s1"])
        self.assertEqual(saved.overlap_mode, "first_signal_only")
        self.assertEqual(saved.proxy_symbol, "SPY")
        self.assertEqual(saved.scan_lane, "testing")
        self.assertEqual(saved.scan_sort_order, 0)

        portfolios = list_portfolios(store_path=self.store_path)
        self.assertEqual(len(portfolios), 1)
        self.assertEqual(portfolios[0].id, saved.id)

    @patch("api.portfolio_store.get_strategy_by_id")
    def test_update_portfolio_partial_patch(self, mock_get_strategy):
        from types import SimpleNamespace

        mock_get_strategy.return_value = SimpleNamespace(
            id="s1",
            name="Long SPY",
            direction="long",
        )
        saved = create_portfolio(
            SavePortfolioRequest(name="Combo", strategy_ids=["s1"]),
            store_path=self.store_path,
        )
        updated = update_portfolio(
            saved.id,
            UpdatePortfolioRequest(description="Updated", scan_lane="active", scan_sort_order=3),
            store_path=self.store_path,
        )
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.description, "Updated")
        self.assertEqual(updated.scan_lane, "active")
        self.assertEqual(updated.scan_sort_order, 3)
        self.assertEqual(updated.name, "Combo")

    @patch("api.portfolio_store.get_strategy_by_id")
    def test_delete_portfolio(self, mock_get_strategy):
        from types import SimpleNamespace

        mock_get_strategy.return_value = SimpleNamespace(
            id="s1",
            name="Long SPY",
            direction="long",
        )
        saved = create_portfolio(
            SavePortfolioRequest(name="Combo", strategy_ids=["s1"]),
            store_path=self.store_path,
        )
        self.assertTrue(delete_portfolio(saved.id, store_path=self.store_path))
        self.assertEqual(list_portfolios(store_path=self.store_path), [])
        self.assertFalse(delete_portfolio(saved.id, store_path=self.store_path))

    @patch("api.portfolio_store.get_strategy_by_id")
    def test_rejects_short_strategy(self, mock_get_strategy):
        from types import SimpleNamespace

        mock_get_strategy.return_value = SimpleNamespace(
            id="short-1",
            name="Short SPY",
            direction="short",
        )
        with self.assertRaisesRegex(ValueError, "Short strategies are not supported"):
            create_portfolio(
                SavePortfolioRequest(name="Bad combo", strategy_ids=["short-1"]),
                store_path=self.store_path,
            )

    @patch("api.portfolio_store.get_strategy_by_id")
    def test_concurrent_writes_use_file_lock(self, mock_get_strategy):
        from types import SimpleNamespace

        mock_get_strategy.return_value = SimpleNamespace(
            id="s1",
            name="Long SPY",
            direction="long",
        )
        first = create_portfolio(
            SavePortfolioRequest(name="One", strategy_ids=["s1"]),
            store_path=self.store_path,
        )
        second = create_portfolio(
            SavePortfolioRequest(name="Two", strategy_ids=["s1"]),
            store_path=self.store_path,
        )
        loaded = list_portfolios(store_path=self.store_path)
        self.assertEqual({item.id for item in loaded}, {first.id, second.id})


if __name__ == "__main__":
    unittest.main()
