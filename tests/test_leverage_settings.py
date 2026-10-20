import json
import tempfile
import unittest
from pathlib import Path

import config
from api import leverage_settings as ls
from api.leverage_settings import (
    LeverageSettings,
    SymbolLeverage,
    load_settings,
    normalize_settings,
    save_settings,
)


def entry(symbol="SOXX", vehicle="SOXL", leverage=3, **kwargs):
    return dict(symbol=symbol, leverage_symbol=vehicle, leverage=leverage, **kwargs)


class DefaultsAndIsolationTests(unittest.TestCase):
    def test_tests_never_read_the_users_settings(self):
        real = Path(ls.__file__).resolve().parent.parent / "data" / "settings.json"
        self.assertNotEqual(Path(ls.DEFAULT_STORE_PATH).resolve(), real)

    def test_missing_file_means_the_behavior_before_this_feature(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = load_settings(Path(folder) / "nothing.json")
        self.assertEqual(settings.general_leverage, float(config.Leverage))
        self.assertEqual(settings.symbols, ())
        self.assertIsNone(settings.for_symbol("SOXX"))

    def test_unreadable_file_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text("{not json")
            self.assertEqual(load_settings(path), ls.default_settings())


class ValidationTests(unittest.TestCase):
    def test_normalizes_case_and_whitespace(self):
        settings = normalize_settings("3", [entry(" soxx ", "soxl", "3")])
        self.assertEqual(settings.symbols[0].symbol, "SOXX")
        self.assertEqual(settings.symbols[0].leverage_symbol, "SOXL")
        self.assertEqual(settings.general_leverage, 3.0)
        self.assertIs(settings.for_symbol("soxx"), settings.symbols[0])

    def test_rejects_what_cannot_run(self):
        bad = [
            (0, [entry()]),  # general leverage
            (11, [entry()]),
            ("x", [entry()]),
            (3, [entry(leverage=0)]),
            (3, [entry(leverage=-3)]),
            (3, [entry(leverage=11)]),
            (3, [entry(leverage="three")]),
            (3, [entry(vehicle="SOXX")]),  # the leverage symbol must differ
            (3, [entry(), entry(vehicle="USD")]),  # symbol listed twice
            (3, [entry(), entry("SMH", "SOXL")]),  # one leverage symbol for two symbols
            (3, [entry("SOXX", "SOXL"), entry("SOXL", "XYZ")]),  # chain
            (3, [entry(symbol="")]),
            (3, [entry(symbol="not a ticker")]),
            (3, [entry(beta=0)]),
            (3, [entry(annual_fee=0.5)]),
            (3, [entry(annual_fee=-0.01)]),
            (3, [entry(financing_multiple=-1)]),
        ]
        for general, entries in bad:
            with self.subTest(general=general, entries=entries), self.assertRaises(ValueError):
                normalize_settings(general, entries)

    def test_cost_model_fields_are_optional_and_reported(self):
        plain = normalize_settings(3, [entry()]).symbols[0]
        self.assertFalse(plain.has_cost_model)
        self.assertEqual(plain.effective_beta, 3.0)
        costed = normalize_settings(3, [entry(beta=2.96, annual_fee=0.019, financing_multiple=2.6)]).symbols[0]
        self.assertTrue(costed.has_cost_model)
        self.assertEqual(costed.effective_beta, 2.96)

    def test_fractional_leverage_is_allowed(self):
        self.assertEqual(normalize_settings(1.5, [entry(leverage=0.5)]).symbols[0].leverage, 0.5)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "settings.json"

    def test_round_trip_and_cache_follow_the_file(self):
        saved = save_settings(normalize_settings(2.5, [entry(annual_fee=0.02)]), self.path)
        self.assertEqual(load_settings(self.path), saved)
        self.assertEqual(json.loads(self.path.read_text())["general_leverage"], 2.5)
        self.assertEqual(load_settings(self.path).symbols[0].annual_fee, 0.02)
        save_settings(normalize_settings(4, []), self.path)
        self.assertEqual(load_settings(self.path).general_leverage, 4.0)
        self.assertEqual(load_settings(self.path).symbols, ())

    def test_an_edit_made_outside_the_app_is_picked_up(self):
        save_settings(normalize_settings(3, []), self.path)
        self.assertEqual(load_settings(self.path).general_leverage, 3.0)
        self.path.write_text(json.dumps({"general_leverage": 5, "symbols": []}))
        self.assertEqual(load_settings(self.path).general_leverage, 5.0)

    def test_save_refuses_invalid_settings_and_leaves_the_file_alone(self):
        save_settings(normalize_settings(3, [entry()]), self.path)
        before = self.path.read_text()
        invalid = LeverageSettings(general_leverage=3.0, symbols=(SymbolLeverage("A", "A", 3.0),))
        with self.assertRaises(ValueError):
            save_settings(invalid, self.path)
        self.assertEqual(self.path.read_text(), before)

    def test_module_helpers_read_the_default_path(self):
        original = ls.DEFAULT_STORE_PATH
        try:
            ls.DEFAULT_STORE_PATH = self.path
            save_settings(normalize_settings(2, [entry()]))
            self.assertEqual(ls.general_leverage(), 2.0)
            self.assertEqual(ls.symbol_setting("soxx").leverage_symbol, "SOXL")
            self.assertIsNone(ls.symbol_setting("SPY"))
        finally:
            ls.DEFAULT_STORE_PATH = original


if __name__ == "__main__":
    unittest.main()
