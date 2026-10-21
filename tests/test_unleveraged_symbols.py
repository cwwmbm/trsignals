"""A symbol with no leverage symbol is traded itself at the entry's leverage (1 for a plain stock)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import backtest as bt
from api import leverage_settings as ls
from api import services, vehicles
from api.allocation import AllInModel
from api.instruments import InstrumentBook
from api.leverage_settings import normalize_entry, normalize_settings
from api.schemas import LeverageEstimateRequest, LeverageSettingsModel, SymbolLeverageModel
from tests.test_vehicle_engines import DATES as ENGINE_DATES
from tests.test_vehicle_engines import R, child_n, market, run, signals_for
from tests.test_allocation import _strategy
from tests.test_vehicles import DATES, frame, settings, underlying_close

PLAIN = dict(symbol="AAA", leverage_symbol=None, leverage=1)


class NormalizationTests(unittest.TestCase):
    def test_blank_missing_or_the_symbol_itself_all_mean_no_leverage_symbol(self):
        for value in (None, "", "   ", "AAA", " aaa ", "Aaa"):
            with self.subTest(value=value):
                entry = normalize_entry(dict(symbol="AAA", leverage_symbol=value, leverage=1))
                self.assertIsNone(entry.leverage_symbol)
                self.assertFalse(entry.has_vehicle)
                self.assertEqual((entry.symbol, entry.leverage), ("AAA", 1.0))

    def test_a_different_symbol_is_still_a_leverage_symbol(self):
        entry = normalize_entry(dict(symbol="soxx", leverage_symbol=" soxl ", leverage=3))
        self.assertEqual(entry.leverage_symbol, "SOXL")
        self.assertTrue(entry.has_vehicle)

    def test_a_bad_leverage_symbol_is_still_refused(self):
        with self.assertRaisesRegex(ValueError, "Leverage symbol"):
            normalize_entry(dict(symbol="AAA", leverage_symbol="not a ticker", leverage=1))

    def test_several_symbols_without_a_leverage_symbol_do_not_clash(self):
        entries = [PLAIN, dict(symbol="BBB", leverage_symbol="", leverage=1), dict(symbol="SOXX", leverage_symbol="SOXL", leverage=3)]
        settings_ = normalize_settings(3, entries)
        self.assertEqual([e.leverage_symbol for e in settings_.symbols], [None, None, "SOXL"])

    def test_a_symbol_listed_twice_is_still_refused_with_or_without_one(self):
        with self.assertRaisesRegex(ValueError, "more than once"):
            normalize_settings(3, [PLAIN, dict(PLAIN, leverage=2)])

    def test_a_leverage_symbol_that_is_itself_listed_is_still_a_chain(self):
        with self.assertRaisesRegex(ValueError, "chain"):
            normalize_settings(3, [dict(symbol="SOXX", leverage_symbol="SOXL", leverage=3), dict(symbol="SOXL", leverage_symbol=None, leverage=1)])

    def test_saved_and_loaded_with_a_null_leverage_symbol(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            ls.save_settings(normalize_settings(3, [PLAIN]), path)
            self.assertIsNone(json.loads(path.read_text())["symbols"][0]["leverage_symbol"])
            ls._cache["key"] = None
            loaded = ls.load_settings(path)
        self.assertEqual(loaded.symbols[0].leverage_symbol, None)
        self.assertIs(loaded.for_symbol("aaa"), loaded.symbols[0])

    def test_existing_entries_with_a_leverage_symbol_are_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps({"general_leverage": 3, "symbols": [
                {"symbol": "SOXX", "leverage_symbol": "SOXL", "leverage": 3, "beta": 2.96, "annual_fee": 0.019, "financing_multiple": 2.35}]}))
            ls._cache["key"] = None
            loaded = ls.load_settings(path)
        entry = loaded.symbols[0]
        self.assertEqual((entry.leverage_symbol, entry.beta, entry.annual_fee), ("SOXL", 2.96, 0.019))


class StreamTests(unittest.TestCase):
    close = underlying_close()

    def test_the_stand_in_is_the_symbols_own_return_times_the_leverage_for_the_whole_history(self):
        with settings(entries=[dict(symbol="SOXX", leverage_symbol=None, leverage=1)]), \
             patch("api.vehicle_data.load_close", side_effect=AssertionError("no product to download")):
            stream = vehicles.stream_for(DATES, self.close.to_numpy(), "SOXX")
        self.assertIsNone(stream.vehicle_symbol)
        self.assertFalse(stream.vehicle_available)
        self.assertFalse(stream.vehicle_ok.any())
        self.assertIsNone(stream.first_vehicle_date)
        np.testing.assert_allclose(stream.synthetic[1:], self.close.pct_change().to_numpy()[1:], atol=1e-12)

    def test_a_cost_model_applies_to_every_day_when_there_is_no_product(self):
        entry = dict(symbol="SOXX", leverage_symbol="", leverage=2, annual_fee=0.02)
        with settings(entries=[entry]):
            stream = vehicles.stream_for(DATES, self.close.to_numpy(), "SOXX")
        expected = 2 * self.close.pct_change().to_numpy()[1:] - 0.02 / 252
        np.testing.assert_allclose(stream.synthetic[1:], expected, atol=1e-12)


class SingleBacktestTests(unittest.TestCase):
    def run_backtest(self, entries, general=3):
        with settings(general=general, entries=entries):
            return bt.execute_strategy(frame(underlying_close()), 4, 99, True)

    def test_leverage_one_earns_what_a_general_leverage_of_one_would(self):
        entry_one = self.run_backtest([dict(symbol="SOXX", leverage_symbol=None, leverage=1)], general=3)
        general_one = self.run_backtest([], general=1)
        np.testing.assert_allclose(entry_one["RollingPnL"], general_one["RollingPnL"], rtol=1e-12)
        np.testing.assert_allclose(entry_one["TradePnL"], general_one["TradePnL"], atol=1e-12)

    def test_it_differs_from_the_general_leverage_it_overrides(self):
        entry_one = self.run_backtest([dict(symbol="SOXX", leverage_symbol=None, leverage=1)], general=3)
        general_three = self.run_backtest([], general=3)
        self.assertFalse(np.allclose(entry_one["RollingPnL"], general_three["RollingPnL"]))
        # other symbols keep the general leverage
        other = self.run_backtest([dict(symbol="ZZZ", leverage_symbol=None, leverage=1)], general=3)
        np.testing.assert_allclose(other["RollingPnL"], general_three["RollingPnL"], rtol=1e-12)

    def test_the_payload_says_the_symbol_was_traded_itself(self):
        from api.serializers import detailed_backtest_payload

        executed = self.run_backtest([dict(symbol="SOXX", leverage_symbol=None, leverage=1)])
        info = detailed_backtest_payload(executed, 4, 99, "plain")["vehicle"]
        self.assertEqual((info["symbol"], info["leverage_symbol"], info["leverage"]), ("SOXX", None, 1.0))
        self.assertFalse(info["available"])
        self.assertEqual(info["trades_on_leverage_symbol"], 0)


class EngineTests(unittest.TestCase):
    def book(self, entries, general=3):
        strategy = _strategy("a", "AAA")
        states = {"a": signals_for("AAA", "a", (2, 6), (8, 16), (18, 25))}
        with settings(general=general, entries=entries):
            track, close = market()
            return run([strategy], states, AllInModel(), track, close), track

    def test_the_flat_engine_earns_the_symbols_own_return_at_leverage_one(self):
        simulation, _ = self.book([PLAIN], general=3)
        reference, _ = self.book([], general=1)
        np.testing.assert_allclose(simulation.equity, reference.equity, rtol=1e-12)
        self.assertEqual({t.symbol for t in simulation.transactions}, {"AAA"})

    def test_the_book_reports_the_entry_without_pretending_there_is_a_product(self):
        simulation, _ = self.book([PLAIN])
        self.assertEqual(simulation.instruments["AAA"]["leverage"], 1.0)
        self.assertTrue(simulation.instruments["AAA"]["configured"])
        (row,) = simulation.vehicle_report
        self.assertEqual((row["symbol"], row["leverage_symbol"], row["available"], row["first_date"]), ("AAA", None, False, None))

    def test_the_instrument_book_never_looks_for_a_product(self):
        with settings(entries=[PLAIN]), patch("api.vehicle_data.load_close", side_effect=AssertionError("no product")):
            track, close = market()
            book = InstrumentBook(["AAA", "BBB"], track, close, ENGINE_DATES)
        self.assertEqual(sorted(book.keys), ["AAA", "BBB"])

    def test_other_symbols_keep_the_general_leverage(self):
        with settings(general=3, entries=[PLAIN]):
            track, _ = market()
        np.testing.assert_allclose(track["AAA"].iloc[1:], R["AAA"].iloc[1:].to_numpy(), atol=1e-12)        # 1x
        np.testing.assert_allclose(track["BBB"].iloc[1:], 3 * R["BBB"].iloc[1:].to_numpy(), atol=1e-12)    # general 3x

    def test_a_nested_book_agrees_with_the_flat_one(self):
        a = _strategy("a", "AAA")
        states = {"a": signals_for("AAA", "a", (2, 6), (18, 25))}
        with settings(entries=[PLAIN]):
            track, close = market()
            flat = run([a], states, AllInModel(), track, close)
            wrapped, state = child_n([a], states, AllInModel(), "A")
            nested = run([wrapped], {wrapped.id: state}, AllInModel(), track, close)
        np.testing.assert_allclose(nested.equity, flat.equity, rtol=1e-9)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.original = ls.DEFAULT_STORE_PATH
        ls.DEFAULT_STORE_PATH = Path(self.folder.name) / "settings.json"
        self.addCleanup(setattr, ls, "DEFAULT_STORE_PATH", self.original)

    def test_the_server_accepts_a_missing_or_blank_leverage_symbol_and_stores_none(self):
        for value in (None, "", "RIVN"):
            with self.subTest(value=value):
                saved = services.write_leverage_settings(LeverageSettingsModel(
                    general_leverage=3, symbols=[SymbolLeverageModel(symbol="rivn", leverage_symbol=value, leverage=1)]))
                self.assertEqual(saved["symbols"][0]["leverage_symbol"], None)
                self.assertEqual(services.read_leverage_settings(), saved)
                self.assertIsNone(ls.symbol_setting("RIVN").leverage_symbol)
                self.assertEqual(ls.symbol_setting("RIVN").leverage, 1.0)

    def test_there_is_nothing_to_fit_a_cost_model_against(self):
        for value in ("", "RIVN"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "leverage symbol"):
                services.estimate_leverage_costs(LeverageEstimateRequest(symbol="RIVN", leverage_symbol=value))


if __name__ == "__main__":
    unittest.main()
