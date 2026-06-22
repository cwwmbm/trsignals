import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import api.market_data_cache as cache
import backtest as bt
from api.market_data_cache import PROFILE_BULK, PROFILE_SINGLE, load, save
from backtest_runners import attach_proxy_column, load_ticker_data


def _sample_frame(symbol: str = "SPY", rows: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": np.arange(rows, dtype=float) + 100,
            "High": np.arange(rows, dtype=float) + 101,
            "Low": np.arange(rows, dtype=float) + 99,
            "Close": np.arange(rows, dtype=float) + 100.5,
            "Volume": np.full(rows, 1000.0),
            "RSI2": np.full(rows, 50.0),
        }
    )


class MarketDataCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_root = Path(self.temp_dir.name)
        self.cache_root_patcher = patch.object(cache, "CACHE_ROOT", self.cache_root)
        self.cache_root_patcher.start()

    def tearDown(self):
        self.cache_root_patcher.stop()
        self.temp_dir.cleanup()

    def test_save_and_load_roundtrip(self):
        frame = _sample_frame()
        save("SPY", 25, PROFILE_SINGLE, frame)
        loaded = load("SPY", 25, PROFILE_SINGLE)
        self.assertIsNotNone(loaded)
        pd.testing.assert_frame_equal(loaded, frame)
        self.assertTrue((self.cache_root / PROFILE_SINGLE / "SPY_25y.pkl").exists())
        self.assertTrue((self.cache_root / PROFILE_SINGLE / "SPY_25y.meta.json").exists())

    def test_load_returns_copy(self):
        frame = _sample_frame()
        save("SPY", 25, PROFILE_SINGLE, frame)
        loaded = load("SPY", 25, PROFILE_SINGLE)
        loaded.loc[0, "Close"] = 999.0
        reloaded = load("SPY", 25, PROFILE_SINGLE)
        self.assertNotEqual(reloaded.loc[0, "Close"], 999.0)

    def test_stale_entry_is_ignored(self):
        frame = _sample_frame()
        cache_path = cache._cache_path("SPY", 25, PROFILE_SINGLE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            import pickle

            pickle.dump(frame, handle)
        stale_time = datetime.now(cache.EASTERN) - timedelta(days=1)
        stale_ts = stale_time.timestamp()
        import os

        os.utime(cache_path, (stale_ts, stale_ts))
        self.assertIsNone(load("SPY", 25, PROFILE_SINGLE))

    def test_incompatible_pickle_is_ignored(self):
        frame = _sample_frame()
        cache_path = cache._cache_path("SPY", 25, PROFILE_SINGLE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            handle.write(b"not-a-valid-pickle")
        cache._meta_path(cache_path).write_text(
            json.dumps(
                {
                    "symbol": "SPY",
                    "years": 25,
                    "profile": PROFILE_SINGLE,
                    "pandas_version": pd.__version__,
                    "cached_date": cache._eastern_today().isoformat(),
                }
            )
        )
        self.assertIsNone(load("SPY", 25, PROFILE_SINGLE))
        self.assertFalse(cache_path.exists())

    def test_pandas_version_mismatch_is_ignored(self):
        frame = _sample_frame()
        save("SPY", 25, PROFILE_SINGLE, frame)
        meta_path = cache._meta_path(cache._cache_path("SPY", 25, PROFILE_SINGLE))
        meta = json.loads(meta_path.read_text())
        meta["pandas_version"] = "0.0.0"
        meta_path.write_text(json.dumps(meta))
        self.assertIsNone(load("SPY", 25, PROFILE_SINGLE))

    def test_load_close_column_checks_both_profiles(self):
        frame = _sample_frame("QQQ")
        save("QQQ", 25, PROFILE_BULK, frame)
        close = cache.load_close_column("QQQ", 25)
        self.assertIsNotNone(close)
        self.assertEqual(len(close), len(frame))

    @patch("backtest_runners.ind.add_indicators")
    @patch("backtest_runners.dt.get_data_yf")
    def test_load_ticker_data_uses_cache_on_second_call(self, mock_get_data_yf, mock_add_indicators):
        prepared = _sample_frame()
        mock_get_data_yf.return_value = prepared.copy()
        mock_add_indicators.return_value = prepared

        first = load_ticker_data("SPY", years=25)
        second = load_ticker_data("SPY", years=25)

        mock_get_data_yf.assert_called_once()
        mock_add_indicators.assert_called_once()
        pd.testing.assert_frame_equal(first, second)

    @patch("backtest_runners.dt.get_bulk_data")
    def test_attach_proxy_column_uses_cached_proxy(self, mock_get_bulk_data):
        signal_frame = _sample_frame("XBI")
        proxy_frame = _sample_frame("SPY")
        save("SPY", 25, PROFILE_BULK, proxy_frame)

        result = attach_proxy_column(signal_frame.copy(), "SPY", years=25)

        mock_get_bulk_data.assert_not_called()
        self.assertIn("SPY", result.columns)
        self.assertFalse(result["SPY"].isna().all())

    @patch("backtest.build_symbol_dataset")
    @patch("backtest.dt.get_bulk_data")
    def test_load_symbol_dataset_partial_cache_hit(
        self,
        mock_get_bulk_data,
        mock_build_symbol_dataset,
    ):
        primary = _sample_frame("SPY")
        confirm = _sample_frame("QQQ")
        save("SPY", 25, PROFILE_BULK, primary)
        mock_get_bulk_data.return_value = pd.DataFrame()
        mock_build_symbol_dataset.return_value = {"QQQ": confirm}

        dataset = bt.load_symbol_dataset(["SPY", "QQQ"], years=25)

        mock_get_bulk_data.assert_called_once()
        mock_build_symbol_dataset.assert_called_once()
        self.assertIn("SPY", dataset)
        self.assertIn("QQQ", dataset)
        pd.testing.assert_frame_equal(dataset["SPY"], primary)
        pd.testing.assert_frame_equal(dataset["QQQ"], confirm)

    @patch("api.scan_service.execute_with_proxy")
    @patch("api.scan_service.bt.apply_cross_symbol_signal")
    @patch("api.builder_strategy.builder_signal_callable")
    @patch("api.scan_service.bt.load_symbol_dataset")
    def test_execute_saved_strategy_scan_disables_cache(
        self,
        mock_load_symbol_dataset,
        mock_builder_signal,
        mock_apply_cross_symbol,
        mock_execute_with_proxy,
    ):
        from api.scan_service import execute_saved_strategy

        primary = _sample_frame("SPY")
        confirm = _sample_frame("QQQ")
        mock_builder_signal.return_value = lambda *_args, **_kwargs: (
            pd.Series([False] * len(primary)),
            pd.Series([False] * len(primary)),
            1,
            1,
            "test",
            None,
            True,
            None,
        )
        mock_load_symbol_dataset.return_value = {"SPY": primary, "QQQ": confirm}
        mock_apply_cross_symbol.return_value = (primary, 1, 1, "test", None, True, None)
        mock_execute_with_proxy.side_effect = lambda frame, *_args, **_kwargs: frame

        strategy = unittest.mock.Mock()
        strategy.symbol = "SPY"
        strategy.confirm_symbols = ["QQQ"]
        strategy.conditions = []
        strategy.hold_on_buy_signal = False

        execute_saved_strategy(
            primary,
            strategy,
            years=1,
            use_cache=False,
        )

        mock_load_symbol_dataset.assert_called_once_with(["SPY", "QQQ"], years=1, use_cache=False)


if __name__ == "__main__":
    unittest.main()
