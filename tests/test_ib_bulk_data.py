import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

import getdata as dt


class IbBulkDataTests(unittest.TestCase):
    def test_ib_canonical_and_contract(self):
        self.assertEqual(dt.ib_canonical_symbol("^VIX"), "^VIX")
        self.assertEqual(dt.ib_canonical_symbol("vix"), "^VIX")
        self.assertEqual(dt.ib_canonical_symbol("spy"), "SPY")

        with patch.dict("sys.modules", {"ib_insync": MagicMock()}):
            # Import path uses real ib_insync if installed; contract types still construct.
            pass
        contract, canonical = dt.ib_contract_for_symbol("^VIX")
        self.assertEqual(canonical, "^VIX")
        self.assertEqual(contract.symbol, "VIX")
        stock, stock_symbol = dt.ib_contract_for_symbol("qqq")
        self.assertEqual(stock_symbol, "QQQ")
        self.assertEqual(stock.symbol, "QQQ")

    def test_ib_bars_to_frame_and_multiindex(self):
        bars = [
            SimpleNamespace(
                date="2026-07-24",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1_000_000,
            ),
            SimpleNamespace(
                date="2026-07-27",
                open=100.5,
                high=102.0,
                low=100.0,
                close=101.0,
                volume=1_100_000,
            ),
        ]
        spy = dt.ib_bars_to_frame(bars)
        vix = dt.ib_bars_to_frame(
            [
                SimpleNamespace(
                    date="2026-07-24",
                    open=18.0,
                    high=19.0,
                    low=17.5,
                    close=18.5,
                    volume=0,
                ),
                SimpleNamespace(
                    date="2026-07-27",
                    open=18.5,
                    high=19.2,
                    low=18.0,
                    close=18.7,
                    volume=0,
                ),
            ]
        )
        bulk = dt.build_bulk_multiindex({"SPY": spy, "^VIX": vix})
        self.assertIsInstance(bulk.columns, pd.MultiIndex)
        self.assertEqual(float(bulk["Close"]["SPY"].loc["2026-07-27"]), 101.0)
        self.assertEqual(float(bulk["Close"]["^VIX"].loc["2026-07-27"]), 18.7)
        self.assertEqual(list(bulk.columns.get_level_values(1).unique()), ["SPY", "^VIX"])

    def test_build_bulk_multiindex_requires_data(self):
        with self.assertRaises(ValueError):
            dt.build_bulk_multiindex({"SPY": pd.DataFrame()})

    def test_ensure_ib_event_loop_in_thread(self):
        import asyncio
        import threading

        result = {}

        def worker():
            try:
                asyncio.get_event_loop()
                had_loop = True
            except RuntimeError:
                had_loop = False
            result["had_loop_before"] = had_loop
            loop = dt._ensure_ib_event_loop()
            result["loop"] = loop
            result["same"] = asyncio.get_event_loop() is loop

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)
        self.assertFalse(result.get("had_loop_before", True))
        self.assertTrue(result.get("same"))
        self.assertIsNotNone(result.get("loop"))
        previous = {
            key: os.environ.get(key) for key in ("IB_HOST", "IB_PORT", "IB_CLIENT_ID")
        }
        os.environ["IB_HOST"] = "10.0.0.2"
        os.environ["IB_PORT"] = "7497"
        os.environ["IB_CLIENT_ID"] = "99"
        try:
            self.assertEqual(dt.ib_connection_settings(), ("10.0.0.2", 7497, 99))
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_get_bulk_data_ib_connect_failure(self):
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = False
        mock_ib.connect.side_effect = ConnectionRefusedError("refused")

        with patch("ib_insync.IB", return_value=mock_ib):
            with self.assertRaises(ValueError) as ctx:
                dt.get_bulk_data_ib(["SPY"], years=1)
        self.assertIn("Could not connect to Interactive Brokers", str(ctx.exception))

    def test_get_bulk_data_ib_success(self):
        bars = [
            SimpleNamespace(
                date="2026-07-27",
                open=1.0,
                high=2.0,
                low=0.5,
                close=1.5,
                volume=10,
            )
        ]
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.qualifyContracts.side_effect = lambda contract: [contract]
        mock_ib.reqHistoricalData.return_value = bars

        with patch("ib_insync.IB", return_value=mock_ib):
            bulk = dt.get_bulk_data_ib(["SPY", "^VIX"], years=1)

        self.assertEqual(float(bulk["Close"]["SPY"].iloc[-1]), 1.5)
        self.assertEqual(float(bulk["Close"]["^VIX"].iloc[-1]), 1.5)
        mock_ib.disconnect.assert_called_once()
        self.assertEqual(mock_ib.reqHistoricalData.call_count, 2)


class BulkCsvHelpersStillWork(unittest.TestCase):
    """CSV helpers remain for offline scripts; Scan UI no longer uses them."""

    def test_default_scan_bulk_csv_path_uses_env(self):
        previous = os.environ.get("SCAN_BULK_CSV")
        os.environ["SCAN_BULK_CSV"] = "/tmp/custom_scan.csv"
        try:
            self.assertEqual(str(dt.default_scan_bulk_csv_path()), "/tmp/custom_scan.csv")
        finally:
            if previous is None:
                os.environ.pop("SCAN_BULK_CSV", None)
            else:
                os.environ["SCAN_BULK_CSV"] = previous


if __name__ == "__main__":
    unittest.main()
