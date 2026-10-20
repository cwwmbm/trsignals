"""Tiered allocation: one strategy active takes a set share of the book, two or more take all of it."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from api.allocation import (
    DEFAULT_TIER_SINGLE_PCT,
    AllInModel,
    SignalState,
    TieredModel,
    comparison_models,
    resolve_allocation_model,
    tiered_model,
)
from api.portfolio_service import (
    allocation_settings,
    simulate_portfolio,
    simulate_portfolio_allocation_comparison,
)
from api.portfolio_store import create_portfolio, list_portfolios, update_portfolio
from api.schemas import PortfolioSimulateRequest, SavePortfolioRequest, UpdatePortfolioRequest
from tests.test_allocation import _patched_portfolio, _run, _two_strategy_case


class TieredModelTests(unittest.TestCase):
    def test_exposure_by_how_many_are_active(self):
        model = tiered_model(55)
        self.assertEqual(model.target_exposure(SignalState(0, 4)), 0.0)
        self.assertAlmostEqual(model.target_exposure(SignalState(1, 4)), 0.55)
        for active in (2, 3, 4):
            self.assertEqual(model.target_exposure(SignalState(active, 4)), 1.0)

    def test_two_or_more_is_always_all_of_the_book_whatever_the_single_weight(self):
        for pct in (0, 10, 55, 100):
            model = tiered_model(pct)
            self.assertEqual(model.target_exposure(SignalState(2, 2)), 1.0)
            self.assertAlmostEqual(model.target_exposure(SignalState(1, 2)), pct / 100)

    def test_default_is_55_percent(self):
        self.assertEqual(DEFAULT_TIER_SINGLE_PCT, 55.0)
        self.assertAlmostEqual(tiered_model().single, 0.55)
        self.assertAlmostEqual(resolve_allocation_model("tiered").single, 0.55)

    def test_resolve_takes_the_single_weight_in_percent(self):
        model = resolve_allocation_model("tiered", None, 40)
        self.assertIsInstance(model, TieredModel)
        self.assertAlmostEqual(model.single, 0.40)
        # The weight belongs to this model only: others ignore it.
        self.assertIsInstance(resolve_allocation_model("all_in", None, 40), AllInModel)

    def test_rejects_weights_outside_zero_to_one_hundred(self):
        for bad in (-1, 100.5, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tiered_model(bad)
        with self.assertRaises(ValueError):
            TieredModel(single=1.2)

    def test_label_shows_the_numbers(self):
        self.assertEqual(tiered_model(55).label, "Tiered (55% / 100%)")
        self.assertEqual(tiered_model(37.5).label, "Tiered (37.5% / 100%)")

    def test_a_change_in_who_is_eligible_does_not_count_as_a_new_state(self):
        self.assertFalse(tiered_model().reacts_to_eligibility)


class TieredEngineTests(unittest.TestCase):
    def test_every_rebalance_targets_the_tier_for_the_active_count(self):
        strategies, signals = _two_strategy_case()
        result = _run(tiered_model(55), strategies, signals)
        targets = result.target_exposure
        counts = result.active_count
        seen = False
        for bar, target in enumerate(targets):
            if np.isnan(target):
                continue
            seen = True
            expected = 0.0 if counts[bar] == 0 else (0.55 if counts[bar] == 1 else 1.0)
            self.assertAlmostEqual(target, expected, msg=f"bar {bar} active {counts[bar]}")
        self.assertTrue(seen)
        self.assertGreater(counts.max(), 1)  # the fixture does have a stretch with both active

    def test_a_single_weight_of_100_is_all_in(self):
        strategies, signals = _two_strategy_case()
        tiered = _run(tiered_model(100), strategies, signals)
        all_in = _run(AllInModel(), strategies, signals)
        np.testing.assert_allclose(tiered.equity, all_in.equity)

    def test_a_lower_single_weight_changes_the_result(self):
        strategies, signals = _two_strategy_case()
        full = _run(tiered_model(100), strategies, signals)
        half = _run(tiered_model(50), strategies, signals)
        self.assertFalse(np.allclose(full.equity, half.equity))


class TieredSettingsTests(unittest.TestCase):
    def test_request_defaults_and_bounds(self):
        self.assertEqual(PortfolioSimulateRequest(strategy_ids=["a"]).tier_single_pct, 55.0)
        for bad in (-0.1, 100.1):
            with self.assertRaises(ValueError):
                PortfolioSimulateRequest(strategy_ids=["a"], tier_single_pct=bad)
            with self.assertRaises(ValueError):
                SavePortfolioRequest(name="x", strategy_ids=["a"], tier_single_pct=bad)
            with self.assertRaises(ValueError):
                UpdatePortfolioRequest(tier_single_pct=bad)
        with self.assertRaises(ValueError):
            PortfolioSimulateRequest(strategy_ids=["a"], allocation_model="tiered", tier_single_pct=150)

    def test_allocation_settings_reads_the_weight_and_defaults_when_absent(self):
        request = SimpleNamespace(allocation_model="tiered", tier_single_pct=35)
        self.assertAlmostEqual(allocation_settings(request).model.single, 0.35)
        self.assertAlmostEqual(allocation_settings(SimpleNamespace(allocation_model="tiered")).model.single, 0.55)

    def test_comparison_models_end_with_tiered_at_the_requested_weight(self):
        models = comparison_models(40)
        self.assertEqual(models[-1].id, "tiered")
        self.assertAlmostEqual(models[-1].single, 0.40)
        self.assertAlmostEqual(comparison_models()[-1].single, 0.55)


class TieredStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "portfolios.json"

    def strategy(self):
        return SimpleNamespace(id="s1", name="Long SPY", direction="long")

    @patch("api.portfolio_store.get_strategy_by_id")
    def test_round_trips_updates_and_defaults(self, mock_get):
        mock_get.return_value = self.strategy()
        saved = create_portfolio(SavePortfolioRequest(name="A", strategy_ids=["s1"], allocation_model="tiered",
                                                      tier_single_pct=45), store_path=self.path)
        (loaded,) = list_portfolios(store_path=self.path)
        self.assertEqual((loaded.allocation_model, loaded.tier_single_pct), ("tiered", 45))
        updated = update_portfolio(saved.id, UpdatePortfolioRequest(tier_single_pct=60), store_path=self.path)
        self.assertEqual((updated.allocation_model, updated.tier_single_pct), ("tiered", 60))
        other = create_portfolio(SavePortfolioRequest(name="B", strategy_ids=["s1"]), store_path=self.path)
        self.assertEqual(other.tier_single_pct, 55.0)

    def test_records_saved_before_the_model_existed_load_with_the_default(self):
        self.path.write_text('[{"id": "p1", "name": "Old", "strategy_ids": ["s1"], "overlap_mode": '
                             '"hold_until_all_exit", "created_at": "2024-01-01", "updated_at": "2024-01-01"}]')
        with patch("api.portfolio_store.get_strategy_by_id", return_value=self.strategy()):
            (loaded,) = list_portfolios(store_path=self.path)
        self.assertEqual(loaded.tier_single_pct, 55.0)


class TieredPortfolioTests(unittest.TestCase):
    def test_a_portfolio_runs_tiered_and_names_it(self):
        request = _patched_portfolio(self, dict(allocation_model="tiered", tier_single_pct=40))
        payload = simulate_portfolio(request)
        self.assertEqual(payload["allocation"]["model"], "tiered")
        self.assertEqual(payload["allocation"]["label"], "Tiered (40% / 100%)")
        self.assertIn("tiered (40% / 100%)", payload["summary"]["description"])

    def test_comparison_adds_a_tiered_row_with_the_requested_weight(self):
        request = _patched_portfolio(self, dict(tier_single_pct=40))
        rows = simulate_portfolio_allocation_comparison(request)["models"]
        self.assertEqual(rows[-1]["model"], "tiered")
        self.assertEqual(rows[-1]["label"], "Tiered (40% / 100%)")
        self.assertIsNotNone(rows[-1]["attribution"])


if __name__ == "__main__":
    unittest.main()
