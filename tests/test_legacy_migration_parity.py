"""Compare migrated builder strategies against legacy buy_signal functions."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import indicators as ind
import numpy as np
import pandas as pd

from api.legacy_strategy_migration import (
    LEGACY_STRATEGY_TEMPLATES,
    MIGRATED_LEGACY_SIGNALS,
    build_migrated_strategies,
)
from api.scan_service import (
    SKIP_SCAN_SYMBOLS,
    _legacy_scan_row,
    _prepare_symbol_frame,
    _builder_scan_row,
    compute_kelly,
)


def _legacy_signal_by_name(name: str):
    return getattr(ind, name)


SCAN_COMPARE_FIELDS = (
    "buy_signal",
    "hold_long",
    "sell_signal",
    "days",
    "profit",
    "trade_pnl",
    "kelly",
)


class LegacyMigrationParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.symbol_frames = cls._build_symbol_frames()

    @classmethod
    def _build_symbol_frames(cls) -> dict[str, pd.DataFrame]:
        import getdata as dt

        scan_symbols = [
            "SPY",
            "SMH",
            "QQQ",
            "SOXX",
            "^VIX",
            "XLI",
            "XLU",
            "XLE",
            "XLF",
            "RSP",
            "IWM",
            "FXI",
            "GDX",
            "GLD",
            "XBI",
            "TLT",
        ]
        yf_symbols = [symbol + "=F" if symbol in ["NQ", "ES", "RTY", "CL", "GC", "SI", "HG"] else symbol for symbol in scan_symbols]
        symbol_mapping = dict(zip(scan_symbols, yf_symbols))
        full_data = dt.get_bulk_data(yf_symbols, years=1)

        frames: dict[str, pd.DataFrame] = {}
        for symbol, yf_symbol in symbol_mapping.items():
            if symbol in SKIP_SCAN_SYMBOLS:
                continue
            frames[symbol] = _prepare_symbol_frame(full_data, symbol, yf_symbol)
        return frames

    def test_every_migrated_template_has_legacy_function(self):
        for template in LEGACY_STRATEGY_TEMPLATES:
            self.assertTrue(hasattr(ind, template.legacy_signal), template.legacy_signal)

    def test_migrated_strategies_match_legacy_scan_rows(self):
        failures: list[str] = []
        strategies = build_migrated_strategies(include_non_scan=False)

        for strategy in strategies:
            legacy_fn = _legacy_signal_by_name(strategy.legacy_signal)
            symbol = strategy.symbol
            data = self.symbol_frames.get(symbol)
            if data is None:
                continue

            legacy_row = _legacy_scan_row(symbol, legacy_fn, data)
            builder_row = _builder_scan_row(strategy, data)

            if legacy_row is None:
                failures.append(f"{strategy.legacy_signal}/{symbol}: legacy ignored but builder produced a row")
                continue

            for field in SCAN_COMPARE_FIELDS:
                legacy_value = legacy_row[field]
                builder_value = builder_row[field]
                if field == "kelly":
                    if legacy_value is None and builder_value is None:
                        continue
                    if legacy_value is None or builder_value is None:
                        failures.append(
                            f"{strategy.legacy_signal}/{symbol}.{field}: legacy={legacy_value!r} builder={builder_value!r}"
                        )
                        continue
                    if abs(float(legacy_value) - float(builder_value)) > 0.01:
                        failures.append(
                            f"{strategy.legacy_signal}/{symbol}.{field}: legacy={legacy_value!r} builder={builder_value!r}"
                        )
                    continue

                if legacy_value != builder_value:
                    failures.append(
                        f"{strategy.legacy_signal}/{symbol}.{field}: legacy={legacy_value!r} builder={builder_value!r}"
                    )

        if failures:
            self.fail("Migration parity failures:\n" + "\n".join(failures))

    def test_og_signals_remain_legacy_only(self):
        from api.scan_service import LEGACY_BUY_SIGNALS

        legacy_names = {fn.__name__ for fn in LEGACY_BUY_SIGNALS}
        self.assertEqual(legacy_names, {"og_buy_signal", "og_new_buy_signal"})
        self.assertTrue(MIGRATED_LEGACY_SIGNALS.isdisjoint(legacy_names))


if __name__ == "__main__":
    unittest.main()
