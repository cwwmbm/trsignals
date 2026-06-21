import io
import unittest

import numpy as np
import pandas as pd

from api.custom_data import (
    CUSTOM_INTRADAY_BACKTEST_LIMIT,
    detect_interval,
    parse_intraday_csv,
    prepare_custom_intraday_frame,
    slice_intraday_backtest_window,
    stub_market_context_columns,
    unavailable_indicator_ids,
)
from stats import monthly_performance


def _sample_csv(rows: list[str]) -> bytes:
    header = "symbol,timestamp,open,high,low,close,volume"
    body = "\n".join([header, *rows])
    return body.encode("utf-8")


class CustomDataTests(unittest.TestCase):
    def test_parse_intraday_csv_maps_columns_and_symbol(self):
        content = _sample_csv(
            [
                "SPY,2024-01-02 09:35:00+00:00,100,101,99,100.5,1000",
                "SPY,2024-01-02 09:40:00+00:00,100.5,102,100,101.5,1100",
            ]
        )
        frame = parse_intraday_csv(content, filename="SPY_5min.csv")
        self.assertEqual(frame.attrs["symbol"], "SPY")
        self.assertEqual(list(frame.columns), ["Date", "Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(frame.shape[0], 2)
        self.assertIn("timezone", frame.attrs)

    def test_parse_intraday_csv_maps_vwap_column(self):
        header = "symbol,timestamp,open,high,low,close,volume,vwap"
        body = "\n".join(
            [
                header,
                "QQQ,2024-01-02 09:35:00+00:00,100,101,99,100.5,1000,100.25",
                "QQQ,2024-01-02 09:40:00+00:00,100.5,102,100,101.5,1100,100.75",
            ]
        )
        frame = parse_intraday_csv(body.encode("utf-8"), filename="QQQ_5min.csv")
        self.assertIn("VWAP", frame.columns)
        self.assertTrue(frame["VWAP"].notna().all())

    def test_add_from_csv_sets_has_vwap_metadata(self):
        from api.custom_data import custom_dataset_store

        header = "symbol,timestamp,open,high,low,close,volume,vwap"
        rows = []
        for index in range(60):
            minute = 35 + index * 5
            hour = 9 + minute // 60
            minute = minute % 60
            rows.append(
                f"QQQ,2024-01-02 {hour:02d}:{minute:02d}:00+00:00,"
                f"{100 + index},{101 + index},{99 + index},{100.5 + index},1000,{100 + index * 0.1}"
            )
        content = "\n".join([header, *rows]).encode("utf-8")
        dataset = custom_dataset_store.add_from_csv(content, filename="QQQ_5min.csv")
        metadata = custom_dataset_store.metadata(dataset)
        self.assertTrue(dataset.has_vwap)
        self.assertTrue(metadata["has_vwap"])
        self.assertIn("Close_VWAP", metadata["custom_data_only_indicator_ids"])
        custom_dataset_store.delete(dataset.id)

    def test_prepare_custom_intraday_frame_adds_vwap_columns(self):
        header = "symbol,timestamp,open,high,low,close,volume,vwap"
        rows = []
        for index in range(60):
            minute = 35 + index * 5
            hour = 9 + minute // 60
            minute = minute % 60
            rows.append(
                f"QQQ,2024-01-02 {hour:02d}:{minute:02d}:00+00:00,"
                f"{100 + index},{101 + index},{99 + index},{100.5 + index},1000,{100 + index * 0.1}"
            )
        content = "\n".join([header, *rows]).encode("utf-8")
        parsed = parse_intraday_csv(content)
        _, _, periods_per_year = detect_interval(parsed)
        prepared = prepare_custom_intraday_frame(parsed, periods_per_year)
        self.assertIn("Close_VWAP", prepared.columns)
        self.assertIn("VWAPPercentB", prepared.columns)
        self.assertFalse(prepared["Close_VWAP"].isna().all())

    def test_detect_interval_five_minutes(self):
        dates = pd.date_range("2024-01-02 09:30", periods=6, freq="5min")
        frame = pd.DataFrame(
            {
                "Date": dates,
                "Open": np.arange(6) + 100,
                "High": np.arange(6) + 101,
                "Low": np.arange(6) + 99,
                "Close": np.arange(6) + 100.5,
                "Volume": 1000,
            }
        )
        minutes, label, periods_per_year = detect_interval(frame)
        self.assertEqual(minutes, 5)
        self.assertEqual(label, "5min")
        self.assertGreater(periods_per_year, 252)

    def test_detect_interval_sixty_minutes(self):
        dates = pd.date_range("2024-01-02 09:00", periods=4, freq="60min")
        frame = pd.DataFrame(
            {
                "Date": dates,
                "Open": np.arange(4) + 100,
                "High": np.arange(4) + 101,
                "Low": np.arange(4) + 99,
                "Close": np.arange(4) + 100.5,
                "Volume": 1000,
            }
        )
        minutes, label, _ = detect_interval(frame)
        self.assertEqual(minutes, 60)
        self.assertEqual(label, "1h")

    def test_stub_and_prepare_allow_add_indicators(self):
        rows = []
        for index in range(60):
            minute = 35 + index * 5
            hour = 9 + minute // 60
            minute = minute % 60
            rows.append(
                f"QQQ,2024-01-02 {hour:02d}:{minute:02d}:00+00:00,"
                f"{100 + index},{101 + index},{99 + index},{100.5 + index},1000"
            )
        content = _sample_csv(rows)
        parsed = parse_intraday_csv(content)
        stubbed = stub_market_context_columns(parsed)
        for column in ["Spybull", "Breadth", "Riskbreadth"]:
            self.assertIn(column, stubbed.columns)
        _, _, periods_per_year = detect_interval(parsed)
        prepared = prepare_custom_intraday_frame(parsed, periods_per_year)
        self.assertIn("RSI2", prepared.columns)
        self.assertIn("SMA20", prepared.columns)

    def test_add_from_csv_defers_indicator_computation(self):
        rows = []
        for index in range(60):
            minute = 35 + index * 5
            hour = 9 + minute // 60
            minute = minute % 60
            rows.append(
                f"SPY,2024-01-02 {hour:02d}:{minute:02d}:00+00:00,"
                f"{100 + index},{101 + index},{99 + index},{100.5 + index},1000"
            )
        content = _sample_csv(rows)
        from api.custom_data import custom_dataset_store, is_indicators_prepared

        dataset = custom_dataset_store.add_from_csv(content, filename="SPY_5min.csv")
        self.assertFalse(is_indicators_prepared(dataset.data))
        prepared = custom_dataset_store.ensure_prepared(dataset.id)
        self.assertTrue(is_indicators_prepared(prepared.prepared_data))
        self.assertFalse(is_indicators_prepared(prepared.data))
        custom_dataset_store.delete(dataset.id)

    def test_load_backtest_frame_prepares_only_window_when_limited(self):
        from unittest.mock import patch

        from api.custom_data import custom_dataset_store, is_indicators_prepared

        dates = pd.date_range("2024-01-02 09:30", periods=6000, freq="5min", tz="UTC")
        rows = [
            f"SPY,{date.isoformat()},{100 + index},{101 + index},{99 + index},{100.5 + index},1000"
            for index, date in enumerate(dates)
        ]
        content = _sample_csv(rows)
        dataset = custom_dataset_store.add_from_csv(content, filename="SPY_5min.csv")

        with patch("api.custom_data.prepare_custom_intraday_frame") as prepare:
            prepare.side_effect = lambda frame, periods_per_year: frame.assign(RSI2=1.0)
            frame = custom_dataset_store.load_backtest_frame(dataset.id, backtest_all_data=False)

        self.assertEqual(len(frame), CUSTOM_INTRADAY_BACKTEST_LIMIT)
        self.assertTrue(is_indicators_prepared(frame))
        self.assertIsNone(dataset.prepared_data)
        prepare.assert_called_once()
        prepared_input = prepare.call_args.args[0]
        self.assertGreaterEqual(len(prepared_input), CUSTOM_INTRADAY_BACKTEST_LIMIT)
        custom_dataset_store.delete(dataset.id)

    def test_unavailable_indicator_ids_include_market_context(self):
        unavailable = unavailable_indicator_ids()
        self.assertIn("Vix", unavailable)
        self.assertIn("Breadth", unavailable)
        self.assertIn("SPYBull", unavailable)
        self.assertNotIn("Close", unavailable)

    def test_slice_intraday_backtest_window_returns_latest_bars(self):
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-02 09:30", periods=6000, freq="5min", tz="UTC"),
                "Close": np.arange(6000),
            }
        )
        frame.attrs["timezone"] = "UTC"
        sliced = slice_intraday_backtest_window(frame, backtest_all_data=False)
        self.assertEqual(len(sliced), CUSTOM_INTRADAY_BACKTEST_LIMIT)
        self.assertEqual(sliced["Close"].iloc[0], 1000)
        self.assertEqual(sliced["Close"].iloc[-1], 5999)
        self.assertEqual(sliced.attrs["timezone"], "UTC")

        full = slice_intraday_backtest_window(frame, backtest_all_data=True)
        self.assertEqual(len(full), 6000)

    def test_monthly_performance_labels_months(self):
        dates = pd.date_range("2024-01-02", periods=40, freq="B")
        rolling = np.linspace(10000, 11000, len(dates))
        frame = pd.DataFrame(
            {
                "Date": dates,
                "RollingPnL": rolling,
                "Drawdown": np.zeros(len(dates)),
                "LongTradeOut": [False] * (len(dates) - 1) + [True],
                "TradePnL": [0.0] * (len(dates) - 1) + [0.01],
            }
        )
        monthly = monthly_performance(frame)
        self.assertGreaterEqual(monthly.shape[0], 2)

    def test_session_masks_restrict_entries_and_force_eod_exit(self):
        from api.builder_strategy import _compile_strategy_masks
        from api.schemas import BuilderCondition, SavedStrategy

        dates = pd.Series(
            pd.to_datetime(
                [
                    "2020-07-27 12:45:00+00:00",
                    "2020-07-27 13:30:00+00:00",
                    "2020-07-27 19:55:00+00:00",
                    "2020-07-27 20:05:00+00:00",
                ],
                utc=True,
            )
        )
        data = pd.DataFrame(
            {
                "Date": dates,
                "Close": [100.0, 100.0, 100.0, 100.0],
            }
        )
        strategy = SavedStrategy(
            id="test",
            name="test",
            symbol="SPY",
            direction="long",
            hold_days=2,
            profit=1,
            description="",
            conditions=[BuilderCondition(left="Close", operator=">", right="0", logic="AND")],
            sell_conditions=[],
            rth_entries_only=True,
            eod_exit=True,
            created_at="",
            updated_at="",
        )
        buy, sell = _compile_strategy_masks(data, strategy)
        self.assertFalse(bool(buy.iloc[0]))
        self.assertTrue(bool(buy.iloc[1]))
        self.assertTrue(bool(sell.iloc[2]))
        self.assertFalse(bool(sell.iloc[3]))


from stats import monthly_performance


class EquityCurvePayloadTests(unittest.TestCase):
    def test_equity_curve_downsamples_large_series(self):
        from api.serializers import MAX_EQUITY_CURVE_POINTS, equity_curve_payload

        rows = 5000
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-02 09:30", periods=rows, freq="5min"),
                "RollingPnL": np.linspace(10000, 12000, rows),
                "Drawdown": np.zeros(rows),
            }
        )
        payload = equity_curve_payload(frame, is_intraday=True)
        self.assertEqual(len(payload), MAX_EQUITY_CURVE_POINTS)


if __name__ == "__main__":
    unittest.main()
