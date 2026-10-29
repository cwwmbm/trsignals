import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from api import leverage_settings as ls
from api import services
from api.schemas import LeverageEstimateRequest, LeverageSettingsModel, SymbolLeverageModel


class LeverageApiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.original = ls.DEFAULT_STORE_PATH
        ls.DEFAULT_STORE_PATH = Path(self.folder.name) / "settings.json"
        self.addCleanup(setattr, ls, "DEFAULT_STORE_PATH", self.original)

    def request(self, general=3, **entry):
        symbols = [SymbolLeverageModel(symbol="soxx", leverage_symbol="soxl", leverage=3, **entry)]
        return LeverageSettingsModel(general_leverage=general, symbols=symbols)

    def test_reading_before_anything_is_saved_gives_the_defaults(self):
        self.assertEqual(services.read_leverage_settings(), {"general_leverage": 3.0, "symbols": []})

    def test_saving_normalizes_stores_and_is_what_the_engines_then_read(self):
        saved = services.write_leverage_settings(self.request(2.5, annual_fee=0.019, financing_multiple=2.6))
        self.assertEqual(saved["general_leverage"], 2.5)
        self.assertEqual(saved["symbols"][0]["symbol"], "SOXX")
        self.assertEqual(saved["symbols"][0]["leverage_symbol"], "SOXL")
        self.assertEqual(services.read_leverage_settings(), saved)
        self.assertEqual(ls.general_leverage(), 2.5)
        self.assertEqual(ls.symbol_setting("SOXX").financing_multiple, 2.6)

    def test_saving_replaces_the_whole_list(self):
        services.write_leverage_settings(self.request())
        services.write_leverage_settings(LeverageSettingsModel(general_leverage=3, symbols=[]))
        self.assertEqual(services.read_leverage_settings()["symbols"], [])

    def test_invalid_settings_are_refused_with_a_message_and_nothing_is_saved(self):
        services.write_leverage_settings(self.request())
        bad = LeverageSettingsModel(general_leverage=0, symbols=[])
        with self.assertRaisesRegex(ValueError, "General leverage"):
            services.write_leverage_settings(bad)
        twice = LeverageSettingsModel(
            general_leverage=3,
            symbols=[SymbolLeverageModel(symbol="A", leverage_symbol="B", leverage=3)] * 2,
        )
        with self.assertRaisesRegex(ValueError, "more than once"):
            services.write_leverage_settings(twice)
        self.assertEqual(services.read_leverage_settings()["symbols"][0]["symbol"], "SOXX")


class EstimateTests(unittest.TestCase):
    def prices(self):
        rng = np.random.default_rng(3)
        index = pd.bdate_range("2012-01-01", periods=1200)
        u = pd.Series(rng.normal(0.0005, 0.017, len(index)), index=index)
        rate = pd.Series(np.linspace(0.005, 0.05, len(index)), index=index)
        v = 2.95 * u - (0.02 + 2.5 * rate) / 252 + rng.normal(0, 0.0003, len(index))
        under = 100 * (1 + u).cumprod()
        veh = (50 * (1 + v).cumprod())[200:]  # the leverage symbol starts later
        return under, veh, rate

    def test_estimates_the_cost_model_from_the_overlap(self):
        under, veh, rate = self.prices()
        closes = {"SOXX": under, "SOXL": veh}
        with patch("api.vehicle_data.load_close", side_effect=lambda s: closes.get(s)), \
             patch("api.vehicle_data.load_rates", return_value=rate):
            fit = services.estimate_leverage_costs(LeverageEstimateRequest(symbol="soxx", leverage_symbol="soxl"))
        self.assertAlmostEqual(fit["beta"], 2.95, delta=0.03)
        self.assertAlmostEqual(fit["fee"], 0.02, delta=0.015)
        self.assertGreater(fit["correlation"], 0.99)
        self.assertEqual(fit["days"], 999)
        self.assertEqual(fit["leverage_symbol_first_date"], veh.index.min().strftime("%Y-%m-%d"))
        self.assertEqual(fit["symbol_first_date"], under.index.min().strftime("%Y-%m-%d"))

    def test_a_flat_rate_is_left_out_instead_of_splitting_the_fee_arbitrarily(self):
        under, veh, _ = self.prices()
        flat = pd.Series(0.03, index=under.index)
        closes = {"SOXX": under, "SOXL": veh}
        with patch("api.vehicle_data.load_close", side_effect=lambda s: closes.get(s)), \
             patch("api.vehicle_data.load_rates", return_value=flat):
            fit = services.estimate_leverage_costs(LeverageEstimateRequest(symbol="SOXX", leverage_symbol="SOXL"))
        self.assertEqual(fit["financing_multiple"], 0.0)
        self.assertAlmostEqual(fit["beta"], 2.95, delta=0.03)

    def test_says_which_symbol_has_no_history(self):
        under, veh, rate = self.prices()
        with patch("api.vehicle_data.load_close", side_effect=lambda s: {"SOXX": under}.get(s)):
            with self.assertRaisesRegex(ValueError, "SOXL"):
                services.estimate_leverage_costs(LeverageEstimateRequest(symbol="SOXX", leverage_symbol="SOXL"))
        with patch("api.vehicle_data.load_close", side_effect=lambda s: {"SOXL": veh}.get(s)):
            with self.assertRaisesRegex(ValueError, "SOXX"):
                services.estimate_leverage_costs(LeverageEstimateRequest(symbol="SOXX", leverage_symbol="SOXL"))

    def test_rejects_a_bad_symbol(self):
        with self.assertRaises(ValueError):
            services.estimate_leverage_costs(LeverageEstimateRequest(symbol="not a ticker", leverage_symbol="SOXL"))


if __name__ == "__main__":
    unittest.main()
