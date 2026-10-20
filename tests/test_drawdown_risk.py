"""Drawdown probabilities from a block bootstrap of the book's own daily returns."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from api import drawdown_risk as dr


def curve(returns, start=100.0):
    return start * np.cumprod(1.0 + np.asarray(returns, dtype=float))


def naive_worst_fall(returns, starts, block):
    """Reference: build each path day by day in plain Python."""
    length = len(returns)
    worst = []
    for row in starts:
        sequence = [returns[(s + k) % length] for s in row for k in range(block)][:length]
        equity, peak, fall = 1.0, 1.0, 0.0
        for r in sequence:
            equity *= 1.0 + r
            peak = max(peak, equity)
            fall = max(fall, 1.0 - equity / peak)
        worst.append(fall)
    return np.array(worst)


class BootstrapMechanicsTests(unittest.TestCase):
    def test_matches_a_day_by_day_reference_on_the_same_blocks(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0.001, 0.03, 130)
        block, paths = 7, 40
        starts = np.random.default_rng(11).integers(0, len(returns), size=(paths, -(-len(returns) // block)))
        expected = naive_worst_fall(returns, starts, block)
        with patch("numpy.random.default_rng", lambda seed: np.random.Generator(np.random.PCG64(11))):
            got = dr.bootstrap_max_drawdowns(returns, paths=paths, block=block, seed=11)
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_result_does_not_depend_on_chunking(self):
        returns = np.random.default_rng(5).normal(0.0005, 0.02, 200)
        whole = dr.bootstrap_max_drawdowns(returns, paths=300, seed=9)
        with patch.object(dr, "_CELLS_PER_CHUNK", 600):  # three paths at a time
            chunked = dr.bootstrap_max_drawdowns(returns, paths=300, seed=9)
        np.testing.assert_array_equal(whole, chunked)

    def test_same_seed_repeats_and_other_seed_differs(self):
        returns = np.random.default_rng(1).normal(0.0, 0.02, 150)
        a = dr.bootstrap_max_drawdowns(returns, paths=200, seed=1)
        np.testing.assert_array_equal(a, dr.bootstrap_max_drawdowns(returns, paths=200, seed=1))
        self.assertFalse(np.array_equal(a, dr.bootstrap_max_drawdowns(returns, paths=200, seed=2)))

    def test_a_fall_from_the_first_bar_counts(self):
        # Every day loses 10%: each path falls from its starting value of 1.
        worst = dr.bootstrap_max_drawdowns(np.full(100, -0.10), paths=20, seed=0)
        np.testing.assert_allclose(worst, 1 - 0.9**100)

    def test_a_total_wipeout_day_does_not_break_the_logarithm(self):
        worst = dr.bootstrap_max_drawdowns(np.r_[np.zeros(80), -1.0], paths=50, seed=0)
        self.assertTrue(np.isfinite(worst).all())
        self.assertTrue((worst <= 1.0).all())


class DrawdownRiskTests(unittest.TestCase):
    def test_flat_equity_has_no_drawdown_probability(self):
        out = dr.drawdown_risk(np.full(300, 15000.0))
        self.assertEqual([row["probability"] for row in out["thresholds"]], [0.0, 0.0, 0.0])
        self.assertEqual(out["median_max_drawdown"], 0.0)
        self.assertEqual(out["historical_max_drawdown"], 0.0)

    def test_always_rising_equity_never_breaches(self):
        out = dr.drawdown_risk(curve(np.full(300, 0.01)))
        self.assertTrue(all(row["probability"] == 0.0 for row in out["thresholds"]))

    def test_certain_collapse_breaches_every_threshold(self):
        out = dr.drawdown_risk(curve(np.full(200, -0.05)))  # ends 99.99% down
        self.assertEqual([row["probability"] for row in out["thresholds"]], [1.0, 1.0, 1.0])
        self.assertEqual([row["std_error"] for row in out["thresholds"]], [0.0, 0.0, 0.0])

    def test_probabilities_fall_as_the_threshold_rises(self):
        returns = np.random.default_rng(2).normal(0.0015, 0.04, 800)
        out = dr.drawdown_risk(curve(returns), thresholds=(0.1, 0.2, 0.3, 0.5, 0.8), paths=1000)
        probs = [row["probability"] for row in out["thresholds"]]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertGreater(probs[0], probs[-1])

    def test_reports_the_realized_drawdown_percentiles_and_settings(self):
        equity = np.array([100.0] * 30 + [50.0] + [60.0] * 40)
        out = dr.drawdown_risk(equity, paths=500, block=10, seed=7)
        self.assertAlmostEqual(out["historical_max_drawdown"], 0.5)
        self.assertEqual((out["paths"], out["block_days"], out["seed"], out["horizon_bars"]), (500, 10, 7, 70))
        self.assertLessEqual(out["median_max_drawdown"], out["p95_max_drawdown"])
        self.assertEqual([row["drawdown"] for row in out["thresholds"]], [0.4, 0.5, 0.75])

    def test_standard_error_is_the_binomial_one(self):
        out = dr.drawdown_risk(curve(np.random.default_rng(4).normal(0.001, 0.05, 600)), paths=800)
        for row in out["thresholds"]:
            p = row["probability"]
            self.assertAlmostEqual(row["std_error"], (p * (1 - p) / 800) ** 0.5)

    def test_too_little_or_unusable_history_gives_nothing(self):
        self.assertIsNone(dr.drawdown_risk(curve(np.full(dr.MIN_BARS - 1, 0.01))))
        self.assertIsNone(dr.drawdown_risk(np.r_[np.full(100, 100.0), 0.0, 5.0]))  # equity reached zero
        self.assertIsNone(dr.drawdown_risk([]))

    def test_block_length_is_capped_by_the_history(self):
        out = dr.drawdown_risk(curve(np.full(80, 0.001)), block=500, paths=50)
        self.assertEqual(out["block_days"], 79)  # 80 equity values make 79 returns

    def test_frame_version_starts_where_entries_could_first_be_read(self):
        n = 200
        equity = np.r_[np.full(50, 15000.0), curve(np.random.default_rng(6).normal(0.001, 0.03, n - 50), 15000.0)]
        frame = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=n), "RollingPnL": equity,
                              "EntryReady": [False] * 50 + [True] * (n - 50)})
        out = dr.drawdown_risk_for_frame(frame, paths=100)
        self.assertEqual(out["horizon_bars"], n - 50 - 1)
        self.assertIsNone(dr.drawdown_risk_for_frame(None))
        self.assertIsNone(dr.drawdown_risk_for_frame(frame.drop(columns="RollingPnL")))


class IntradayScalingTests(unittest.TestCase):
    def test_blocks_span_days_and_long_histories_get_fewer_paths(self):
        equity = curve(np.random.default_rng(9).normal(0.0, 0.002, 20_000))
        out = dr.drawdown_risk(equity, periods_per_year=252 * 78)   # five-minute bars
        self.assertEqual(out["block_days"], 20)
        self.assertEqual(out["block_bars"], 20 * 78)
        self.assertEqual(out["paths"], dr.CELL_BUDGET // 20_000)
        self.assertEqual(out["periods_per_year"], 252 * 78)

    def test_a_daily_history_keeps_the_full_path_count_and_day_blocks(self):
        out = dr.drawdown_risk(curve(np.random.default_rng(9).normal(0.0, 0.01, 6500)))
        self.assertEqual((out["paths"], out["block_bars"], out["periods_per_year"]), (dr.PATHS, 20, 252))

    def test_path_count_never_drops_below_the_floor(self):
        out = dr.drawdown_risk(curve(np.random.default_rng(9).normal(0.0, 0.001, 200_000)), periods_per_year=252 * 390)
        self.assertEqual(out["paths"], dr.MIN_PATHS)


class SingleStrategyPayloadTests(unittest.TestCase):
    def frame(self, bars=500):
        returns = np.random.default_rng(12).normal(0.001, 0.02, bars)
        equity = 15000 * np.cumprod(1 + returns)
        peak = np.maximum.accumulate(equity)
        close = 100 * np.cumprod(1 + returns / 2)
        return pd.DataFrame({
            "Date": pd.bdate_range("2020-01-01", periods=bars), "Close": close, "RollingPnL": equity,
            "RunningMax": peak, "Drawdown": (peak - equity) / peak,
            "LongTradeIn": False, "HoldLong": False, "LongTradeOut": False, "TradePnL": 0.0,
        })

    def test_any_backtest_payload_carries_the_panel(self):
        from api.serializers import detailed_backtest_payload

        risk = detailed_backtest_payload(self.frame(), 2, 1, "single strategy")["drawdown_risk"]
        self.assertEqual([row["drawdown"] for row in risk["thresholds"]], [0.4, 0.5, 0.75])
        self.assertEqual(risk["horizon_bars"], 499)

    def test_the_bars_per_year_of_the_data_are_respected(self):
        from api.serializers import detailed_backtest_payload

        risk = detailed_backtest_payload(self.frame(), 2, 1, "intraday", periods_per_year=252 * 78,
                                         is_intraday=True)["drawdown_risk"]
        self.assertEqual(risk["block_bars"], 20 * 78 if 20 * 78 < 499 else 499)
        self.assertEqual(risk["periods_per_year"], 252 * 78)

    def test_a_short_backtest_has_no_panel_and_the_payload_still_builds(self):
        from api.serializers import detailed_backtest_payload

        payload = detailed_backtest_payload(self.frame(30), 2, 1, "short")
        self.assertIsNone(payload["drawdown_risk"])
        self.assertIn("summary", payload)


class PortfolioPayloadTests(unittest.TestCase):
    def payload(self, bars):
        from types import SimpleNamespace

        from api.portfolio_service import StrategySignals, simulate_portfolio
        from tests.test_portfolio_service import _strategy

        dates = pd.bdate_range("2022-01-03", periods=bars)
        returns = np.random.default_rng(8).normal(0.002, 0.03, bars)
        close = 100 * np.cumprod(1 + returns)
        enter = np.zeros(bars, dtype=bool); enter[::10] = True
        leave = np.zeros(bars, dtype=bool); leave[5::10] = True
        hold = np.zeros(bars, dtype=bool)
        for i in range(bars):
            hold[i] = (hold[i - 1] and not leave[i - 1] or enter[i - 1]) if i else False
        signals = StrategySignals(strategy_id="a", name="A", symbol="AAA", dates=dates,
                                  long_trade_in=enter, hold_long=hold, long_trade_out=leave)
        request = SimpleNamespace(strategy_ids=["a"], overlap_mode="first_signal_only",
                                  proxy_symbol=None, years=25)
        with patch("api.portfolio_service.get_strategy_by_id", return_value=_strategy("a", "A", "AAA")), \
             patch("api.portfolio_service._load_portfolio_market_data", return_value=(
                 pd.DataFrame(), {"AAA": pd.DataFrame({"Date": dates, "Close": close})},
                 {"AAA": pd.Series(returns, index=dates)}, {"AAA": pd.Series(close, index=dates)})), \
             patch("api.portfolio_service._load_strategy_signals", return_value=signals):
            return simulate_portfolio(request)

    def test_a_portfolio_result_carries_the_panel(self):
        risk = self.payload(400)["drawdown_risk"]
        self.assertEqual([row["drawdown"] for row in risk["thresholds"]], [0.4, 0.5, 0.75])
        self.assertEqual(risk["paths"], dr.PATHS)
        self.assertTrue(all(0.0 <= row["probability"] <= 1.0 for row in risk["thresholds"]))

    def test_a_result_with_too_little_history_has_no_panel_and_still_works(self):
        payload = self.payload(20)
        self.assertIsNone(payload["drawdown_risk"])
        self.assertIn("summary", payload)


if __name__ == "__main__":
    unittest.main()
