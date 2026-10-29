from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api.quote_service import QUOTE_SYMBOLS
from api.quote_store import (
    add_quote_symbol,
    list_quote_symbols,
    normalize_quote_symbol,
    remove_quote_symbol,
)
from api.services import add_saved_quote_symbol, remove_saved_quote_symbol


class NormalizeQuoteSymbolTests(unittest.TestCase):
    def test_vix_futures_and_index_tickers(self):
        self.assertEqual(normalize_quote_symbol(" vix "), ("VIX", "^VIX"))
        self.assertEqual(normalize_quote_symbol("^VIX"), ("VIX", "^VIX"))
        self.assertEqual(normalize_quote_symbol("nq"), ("NQ", "NQ=F"))
        self.assertEqual(normalize_quote_symbol("^tnx"), ("^TNX", "^TNX"))
        self.assertEqual(normalize_quote_symbol("brk-b"), ("BRK-B", "BRK-B"))

    def test_rejects_blank_and_garbage(self):
        with self.assertRaisesRegex(ValueError, "Enter a symbol"):
            normalize_quote_symbol("  ")
        with self.assertRaisesRegex(ValueError, "ticker"):
            normalize_quote_symbol("NOT A TICKER")


class QuoteStoreTests(unittest.TestCase):
    def test_missing_file_uses_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quote_symbols.json"
            self.assertEqual(list_quote_symbols(path), list(QUOTE_SYMBOLS))
            self.assertFalse(path.exists())

    def test_add_and_remove_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quote_symbols.json"
            added = add_quote_symbol("iwm", store_path=path)
            self.assertEqual(added[-1], ("IWM", "IWM"))
            self.assertTrue(path.exists())

            with self.assertRaisesRegex(ValueError, "already"):
                add_quote_symbol("IWM", store_path=path)

            removed = remove_quote_symbol("iwm", store_path=path)
            self.assertNotIn(("IWM", "IWM"), removed)
            self.assertEqual([symbol for symbol, _yf in list_quote_symbols(path)], ["SPY", "QQQ", "SOXX", "VIX"])

            with self.assertRaisesRegex(ValueError, "not on the quote list"):
                remove_quote_symbol("IWM", store_path=path)


class QuoteSymbolServiceTests(unittest.TestCase):
    def _snapshot(self, *symbols: str) -> dict:
        return {
            "as_of": "2026-08-21",
            "quotes": [{"symbol": symbol, "close": 10.0} for symbol in symbols],
        }

    def test_add_keeps_symbol_when_yahoo_has_a_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quote_symbols.json"
            snapshot = self._snapshot("SPY", "QQQ", "SOXX", "VIX", "IWM")
            with patch("api.services.run_quotes", return_value=snapshot):
                result = add_saved_quote_symbol("iwm", store_path=path)
            self.assertEqual(result["quotes"][-1]["symbol"], "IWM")
            self.assertEqual(
                [symbol for symbol, _yf in list_quote_symbols(path)],
                ["SPY", "QQQ", "SOXX", "VIX", "IWM"],
            )

    def test_add_rolls_back_when_symbol_has_no_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quote_symbols.json"
            snapshot = {
                "as_of": None,
                "quotes": [{"symbol": "ZZZZ", "close": None}],
            }
            with patch("api.services.run_quotes", return_value=snapshot):
                with self.assertRaisesRegex(ValueError, "No Yahoo data for ZZZZ"):
                    add_saved_quote_symbol("ZZZZ", store_path=path)
            self.assertNotIn("ZZZZ", [symbol for symbol, _yf in list_quote_symbols(path)])

    def test_add_rolls_back_when_snapshot_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quote_symbols.json"
            with patch("api.services.run_quotes", side_effect=RuntimeError("yahoo down")):
                with self.assertRaisesRegex(ValueError, "Couldn't download"):
                    add_saved_quote_symbol("IWM", store_path=path)
            self.assertNotIn("IWM", [symbol for symbol, _yf in list_quote_symbols(path)])

    def test_remove_then_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quote_symbols.json"
            add_quote_symbol("IWM", store_path=path)
            with patch("api.services.run_quotes", return_value=self._snapshot("SPY", "QQQ", "SOXX", "VIX")):
                remove_saved_quote_symbol("IWM", store_path=path)
            self.assertEqual(
                [symbol for symbol, _yf in list_quote_symbols(path)],
                ["SPY", "QQQ", "SOXX", "VIX"],
            )


if __name__ == "__main__":
    unittest.main()
