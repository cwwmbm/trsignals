import unittest

import numpy as np
import pandas as pd

import backtest as bt
from api import benchmark
from api.leverage_settings import SymbolLeverage
from api.serializers import detailed_backtest_payload
from tests.test_vehicles import (
    DATES,
    LISTED,
    N,
    frame,
    settings,
    underlying_close,
    vehicle_close_from,
)


def scored_frame(equity, dates=None):
    dates = DATES[: len(equity)] if dates is None else dates
    frame_ = pd.DataFrame({"Date": dates, "RollingPnL": np.asarray(equity, dtype=float)})
    frame_["HoldLong"] = [i % 2 == 0 for i in range(len(frame_))]
    return frame_


class EquityMathTests(unittest.TestCase):
    def test_underlying_starts_at_the_strategy_start_and_compounds_the_close(self):
        close = pd.Series([100.0, 110.0, 121.0, 108.9], index=DATES[:4])
        inputs = benchmark.BenchmarkInputs(
            symbol="XYZ", close=close, synthetic=close.pct_change() * 1.0, vehicle=None, leverage=1.0
        )
        columns, meta = benchmark.benchmark_for(scored_frame([5000.0, 5100.0, 5200.0, 5300.0]), inputs)
        np.testing.assert_allclose(columns["Benchmark_underlying"], [5000.0, 5500.0, 6050.0, 5445.0])
        # Unleveraged with no product: the vehicle benchmark would be the same line, so it is left out.
        self.assertEqual([item["id"] for item in meta["series"]], ["underlying"])

    def test_vehicle_uses_real_returns_where_they_exist_and_the_stand_in_before(self):
        close = pd.Series(np.linspace(100, 130, 6), index=DATES[:6])
        synthetic = pd.Series([np.nan, 0.03, 0.03, 0.03, 0.03, 0.03], index=DATES[:6])
        vehicle = pd.Series([np.nan, np.nan, np.nan, 0.05, -0.02, 0.01], index=DATES[:6])
        inputs = benchmark.BenchmarkInputs(
            symbol="SOXX",
            close=close,
            synthetic=synthetic,
            vehicle=vehicle,
            leverage=3.0,
            leverage_symbol="SOXL",
            first_vehicle_date=DATES[2],
        )
        columns, meta = benchmark.benchmark_for(scored_frame([1000.0] * 6), inputs)
        expected = 1000.0 * np.cumprod([1.0, 1.03, 1.03, 1.05, 0.98, 1.01])
        np.testing.assert_allclose(columns["Benchmark_vehicle"], expected)
        labels = {item["id"]: item["label"] for item in meta["series"]}
        self.assertEqual(labels["underlying"], "SOXX buy & hold")
        self.assertIn("SOXL buy & hold", labels["vehicle"])
        self.assertIn("SOXX 3x before", labels["vehicle"])

    def test_metrics_show_the_full_period_drawdown_and_total_return(self):
        # Quiet climb, a 60% crash, then a recovery well past the old peak.
        returns = np.r_[np.full(300, 0.001), np.full(10, -0.09), np.full(300, 0.004)]
        dates = pd.bdate_range("2018-01-01", periods=len(returns) + 1)
        close = pd.Series(100 * np.cumprod(np.r_[1.0, 1.0 + returns]), index=dates)
        inputs = benchmark.BenchmarkInputs(
            symbol="XYZ", close=close, synthetic=close.pct_change(), vehicle=None, leverage=1.0
        )
        _, meta = benchmark.benchmark_for(scored_frame([1000.0] * len(close), dates=dates), inputs)
        metrics = meta["series"][0]["metrics"]
        self.assertGreater(metrics["max_drawdown"], 0.55)
        self.assertAlmostEqual(metrics["total_return"], close.iloc[-1] / close.iloc[0] - 1.0, places=9)

    def test_time_in_market_is_the_share_of_held_bars(self):
        close = pd.Series(np.linspace(100, 110, 10), index=DATES[:10])
        inputs = benchmark.BenchmarkInputs(
            symbol="XYZ", close=close, synthetic=close.pct_change(), vehicle=None, leverage=1.0
        )
        _, meta = benchmark.benchmark_for(scored_frame([1000.0] * 10), inputs)
        self.assertAlmostEqual(meta["time_in_market_percent"], 50.0)

    def test_nothing_to_compare_returns_none(self):
        self.assertIsNone(benchmark.benchmark_for(scored_frame([1000.0] * 5), None))
        close = pd.Series(np.nan, index=DATES[:5])
        inputs = benchmark.BenchmarkInputs(
            symbol="XYZ", close=close, synthetic=close.pct_change(), vehicle=None, leverage=1.0
        )
        self.assertIsNone(benchmark.benchmark_for(scored_frame([1000.0] * 5), inputs))


class InputsFromFrameTests(unittest.TestCase):
    def test_a_symbol_with_a_leverage_product_uses_its_real_returns_once_listed(self):
        close = underlying_close()
        vehicle = vehicle_close_from(close)
        with settings(entries=[SymbolLeverage("SOXX", "SOXL", 3.0)], vehicle_closes={"SOXL": vehicle}):
            inputs = benchmark.inputs_from_frame(frame(close))
            self.assertEqual(inputs.symbol, "SOXX")
            self.assertEqual(inputs.leverage_symbol, "SOXL")
            self.assertEqual(inputs.leverage, 3.0)
            self.assertEqual(inputs.first_vehicle_date, DATES[LISTED])
            held = benchmark._held_returns(inputs)
            # Before the product exists the stand-in earns, after it the product's own returns.
            self.assertAlmostEqual(held.iloc[10], inputs.synthetic.iloc[10])
            self.assertAlmostEqual(held.iloc[LISTED + 5], vehicle.pct_change().iloc[5])

    def test_a_symbol_without_an_entry_earns_the_general_leverage(self):
        close = underlying_close()
        with settings(general=2):
            inputs = benchmark.inputs_from_frame(frame(close))
            self.assertIsNone(inputs.leverage_symbol)
            self.assertEqual(inputs.leverage, 2)
            np.testing.assert_allclose(inputs.synthetic.iloc[1:], 2 * close.pct_change().iloc[1:])

    def test_a_proxy_symbol_is_benchmarked_on_its_own_prices(self):
        close = underlying_close()
        data = frame(close)
        proxy = (close * 2.0).to_numpy()
        data["QQQ"] = proxy
        with settings(general=1):
            inputs = benchmark.inputs_from_frame(data, "qqq")
            self.assertEqual(inputs.symbol, "QQQ")
            np.testing.assert_allclose(inputs.close.to_numpy(), proxy)

    def test_an_unknown_proxy_column_gives_no_benchmark(self):
        with settings():
            self.assertIsNone(benchmark.inputs_from_frame(frame(underlying_close()), "ZZZ"))


class PayloadTests(unittest.TestCase):
    def _executed(self, close):
        data = frame(close)
        return bt.execute_strategy(data, 3, 2, True)

    def test_curve_rows_carry_both_benchmarks_and_the_payload_describes_them(self):
        close = underlying_close()
        with settings(entries=[SymbolLeverage("SOXX", "SOXL", 3.0)], vehicle_closes={"SOXL": vehicle_close_from(close)}):
            executed = self._executed(close)
            payload = detailed_backtest_payload(
                executed, 3, 2, "x", benchmark=benchmark.inputs_from_frame(executed)
            )
        first = payload["equity_curve"][0]
        self.assertEqual(first["benchmark_underlying"], first["rolling_pnl"])
        self.assertEqual(first["benchmark_vehicle"], first["rolling_pnl"])
        self.assertEqual([item["id"] for item in payload["benchmark"]["series"]], ["underlying", "vehicle"])
        self.assertEqual(payload["benchmark"]["symbol"], "SOXX")
        last = payload["equity_curve"][-1]
        underlying = payload["benchmark"]["series"][0]["metrics"]["total_return"]
        self.assertAlmostEqual(last["benchmark_underlying"] / first["benchmark_underlying"] - 1.0, underlying)

    def test_no_benchmark_input_means_no_benchmark_block_or_columns(self):
        with settings():
            payload = detailed_backtest_payload(self._executed(underlying_close()), 3, 2, "x")
        self.assertNotIn("benchmark", payload)
        self.assertNotIn("benchmark_underlying", payload["equity_curve"][0])

    def test_intraday_results_are_not_benchmarked(self):
        close = underlying_close()
        with settings():
            executed = self._executed(close)
            payload = detailed_backtest_payload(
                executed, 3, 2, "x", is_intraday=True, benchmark=benchmark.inputs_from_frame(executed)
            )
        self.assertNotIn("benchmark", payload)

    def test_the_chart_rows_still_line_up_when_the_curve_is_downsampled(self):
        close = underlying_close()
        with settings():
            executed = self._executed(close)
            payload = detailed_backtest_payload(
                executed, 3, 2, "x", benchmark=benchmark.inputs_from_frame(executed)
            )
        self.assertEqual(len(payload["equity_curve"]), N)
        self.assertTrue(all("benchmark_underlying" in row for row in payload["equity_curve"]))


if __name__ == "__main__":
    unittest.main()
