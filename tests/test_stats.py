import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from stats import (
    best_year_concentration,
    compute_aggregate_metrics,
    metrics_start_label,
    performance_frame,
    yearly_returns,
)


def _backtest_frame(*, years=(2018, 2019, 2020), trades_per_year=(2, 2, 5)) -> pd.DataFrame:
    rows = []
    rolling = 10_000.0
    trade_idx = 0
    for year, trade_count in zip(years, trades_per_year):
        dates = pd.date_range(f"{year}-01-02", periods=60, freq="B")
        year_start = rolling
        for index, date in enumerate(dates):
            trade_out = False
            trade_pnl = 0.0
            if index % max(1, 60 // max(trade_count, 1)) == 0 and trade_idx < sum(trades_per_year):
                trade_out = True
                trade_pnl = 0.02 if year == 2020 else 0.01
                trade_idx += 1
            if index == len(dates) - 1:
                if year == 2020:
                    rolling = year_start * 2.0
                else:
                    rolling = year_start * 1.1
            rows.append(
                {
                    "Date": date,
                    "RollingPnL": rolling,
                    "Drawdown": 0.05 if year == 2020 else 0.02,
                    "LongTradeOut": trade_out,
                    "TradePnL": trade_pnl if trade_out else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _daily_returns(data: pd.DataFrame, drop_year=None) -> pd.Series:
    """Daily returns from the unbroken equity series, leaving one year's own returns out."""
    returns = data["RollingPnL"].pct_change()
    if drop_year is not None:
        returns = returns[data["Date"].dt.year != drop_year]
    return returns.dropna()


def _sharpe(returns: pd.Series) -> float:
    return float(np.sqrt(252) * returns.mean() / returns.std())


def _sortino(returns: pd.Series) -> float:
    downside = np.sqrt((np.minimum(returns, 0.0) ** 2).mean())
    return float(np.sqrt(252) * returns.mean() / downside)


def _noisy_frame(years=(2018, 2019, 2020, 2021, 2022), best_year=2020, seed=3) -> pd.DataFrame:
    """Daily equity with real day-to-day noise and one runaway year in the middle."""
    rng = np.random.default_rng(seed)
    dates = pd.concat(
        [pd.Series(pd.bdate_range(f"{year}-01-02", periods=252)) for year in years],
        ignore_index=True,
    )
    drift = np.where(dates.dt.year == best_year, 0.012, 0.0006)
    returns = rng.normal(drift, 0.01)
    returns[0] = 0.0
    equity = 10_000.0 * np.cumprod(1.0 + returns)
    frame = pd.DataFrame({"Date": dates, "RollingPnL": equity})
    frame["Drawdown"] = (frame["RollingPnL"].cummax() - frame["RollingPnL"]) / frame["RollingPnL"].cummax()
    frame["LongTradeOut"] = False
    frame["TradePnL"] = 0.0
    return frame


class BestYearExclusionDoesNotBridgeTheGapTests(unittest.TestCase):
    """Dropping the best year must not turn its whole gain into one phantom daily return."""

    @patch("stats.ExcludeBestReturnYear", True)
    def test_sharpe_and_sortino_use_real_daily_returns(self):
        data = _noisy_frame()
        metrics = compute_aggregate_metrics(data)
        self.assertEqual(metrics["excluded_year"], 2020)
        returns = _daily_returns(data, drop_year=2020)
        self.assertAlmostEqual(metrics["sharpe"], _sharpe(returns), places=9)
        self.assertAlmostEqual(metrics["sortino"], _sortino(returns), places=9)

    @patch("stats.ExcludeBestReturnYear", True)
    def test_no_phantom_return_and_the_first_session_after_the_gap_is_kept(self):
        data = _noisy_frame()
        returns = _daily_returns(data, drop_year=2020)
        # Largest single day is an ordinary move, not a 1,000%+ jump across 2020.
        self.assertLess(returns.abs().max(), 0.10)
        first_after = data.index[data["Date"].dt.year == 2021][0]
        self.assertIn(first_after, returns.index)  # measured against 2020's last close
        # The old way, returns taken after the rows were dropped, would have been wrecked.
        old_way = _sharpe(data[data["Date"].dt.year != 2020]["RollingPnL"].pct_change().dropna())
        self.assertLess(old_way, _sharpe(returns) * 0.5)
        self.assertAlmostEqual(compute_aggregate_metrics(data)["sharpe"], _sharpe(returns), places=9)

    @patch("stats.ExcludeBestReturnYear", False)
    def test_without_exclusion_the_full_series_is_used(self):
        data = _noisy_frame()
        metrics = compute_aggregate_metrics(data)
        self.assertIsNone(metrics["excluded_year"])
        returns = _daily_returns(data)
        self.assertAlmostEqual(metrics["sharpe"], _sharpe(returns), places=9)
        self.assertAlmostEqual(metrics["sortino"], _sortino(returns), places=9)

    @patch("stats.ExcludeBestReturnYear", True)
    def test_best_year_at_either_end_gives_the_same_answer_as_dropping_its_returns(self):
        for years, best in (((2018, 2019, 2020), 2018), ((2018, 2019, 2020), 2020)):
            with self.subTest(best=best):
                data = _noisy_frame(years=years, best_year=best)
                metrics = compute_aggregate_metrics(data)
                self.assertEqual(metrics["excluded_year"], best)
                self.assertAlmostEqual(
                    metrics["sharpe"], _sharpe(_daily_returns(data, drop_year=best)), places=9
                )

    @patch("stats.ExcludeBestReturnYear", True)
    def test_the_callers_frame_is_not_modified(self):
        data = _noisy_frame()
        before = data.copy()
        compute_aggregate_metrics(data)
        pd.testing.assert_frame_equal(data, before)

    def test_ratio_functions_are_unchanged_without_exclude_year(self):
        import indicators as ind

        data = _noisy_frame()
        expected = _sharpe(_daily_returns(data))
        self.assertAlmostEqual(ind.sharpes_ratio(data.copy()), expected, places=9)
        self.assertAlmostEqual(ind.sortino_ratio(data.copy()), _sortino(_daily_returns(data)), places=9)


class ComputeAggregateMetricsTests(unittest.TestCase):
    @patch("stats.ExcludeBestReturnYear", True)
    def test_trade_count_includes_excluded_best_year(self):
        data = _backtest_frame()
        metrics = compute_aggregate_metrics(data)

        self.assertEqual(metrics["excluded_year"], 2020)
        self.assertEqual(metrics["trades"], 9)
        self.assertNotEqual(metrics["sharpe"], 0.0)

    @patch("stats.ExcludeBestReturnYear", True)
    def test_risk_metrics_still_exclude_best_year(self):
        data = _backtest_frame()
        metrics = compute_aggregate_metrics(data)
        excluded = data[data["Date"].dt.year != metrics["excluded_year"]]

        self.assertEqual(metrics["excluded_year"], 2020)
        self.assertEqual(metrics["max_drawdown"], excluded["Drawdown"].max())
        expected = _sharpe(_daily_returns(data, drop_year=2020))
        self.assertAlmostEqual(metrics["sharpe"], expected)

    @patch("stats.ExcludeBestReturnYear", True)
    def test_ratios_skip_the_stretch_before_an_entry_is_possible(self):
        def block(year, ready, end_pnl):
            dates = pd.bdate_range(f"{year}-01-02", periods=20)
            rows = []
            for index, date in enumerate(dates):
                rows.append(
                    {
                        "Date": date,
                        "RollingPnL": end_pnl if index == len(dates) - 1 else 10_000.0,
                        "Drawdown": 0.0,
                        "LongTradeOut": index == 10 and ready,
                        "TradePnL": 0.1 if index == 10 and ready else 0.0,
                        "EntryReady": ready,
                    }
                )
            return rows

        data = pd.DataFrame(block(2018, False, 10_000.0) + block(2019, True, 10_000.0) + block(2020, True, 12_000.0))
        scored = performance_frame(data)
        self.assertEqual(scored["Date"].dt.year.min(), 2019)
        self.assertIn(2019, yearly_returns(scored))
        self.assertNotIn(2018, yearly_returns(scored))

        metrics = compute_aggregate_metrics(data)
        expected = compute_aggregate_metrics(scored.drop(columns=["EntryReady"]))
        self.assertEqual(metrics_start_label(data), "2019-01-02")
        self.assertAlmostEqual(metrics["sharpe"], expected["sharpe"])
        self.assertAlmostEqual(metrics["cagr_decimal"], expected["cagr_decimal"])
        self.assertEqual(metrics["trades"], 2)
        self.assertEqual(metrics["rolling_pnl"], 12_000.0)


def _crash_year_frame() -> pd.DataFrame:
    """Two quiet years, then a best year that opens with a crash and recovers far past it."""
    dates = pd.concat(
        [pd.Series(pd.bdate_range(f"{year}-01-02", periods=252)) for year in (2018, 2019, 2020)],
        ignore_index=True,
    )
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0004, 0.004, len(dates))
    crash = np.flatnonzero(dates.dt.year == 2020)
    returns[crash[:40]] = -0.015
    returns[crash[40:]] = 0.012
    returns[0] = 0.0
    frame = pd.DataFrame({"Date": dates, "RollingPnL": 10_000.0 * np.cumprod(1.0 + returns)})
    peak = frame["RollingPnL"].cummax()
    frame["Drawdown"] = (peak - frame["RollingPnL"]) / peak
    frame["LongTradeOut"] = False
    frame["TradePnL"] = 0.0
    return frame


class FullPeriodMetricsTests(unittest.TestCase):
    @patch("stats.ExcludeBestReturnYear", True)
    def test_full_block_is_off_unless_asked_for(self):
        self.assertNotIn("full", compute_aggregate_metrics(_crash_year_frame()))

    @patch("stats.ExcludeBestReturnYear", True)
    def test_full_drawdown_keeps_the_crash_in_the_best_year(self):
        data = _crash_year_frame()
        metrics = compute_aggregate_metrics(data, include_full=True)
        self.assertEqual(metrics["excluded_year"], 2020)
        self.assertAlmostEqual(metrics["full"]["max_drawdown"], data["Drawdown"].max())
        self.assertGreater(metrics["full"]["max_drawdown"], 0.35)
        # The ranking figure leaves 2020 out, and with it the crash.
        self.assertLess(metrics["max_drawdown"], 0.10)

    @patch("stats.ExcludeBestReturnYear", True)
    def test_full_ratios_use_every_day(self):
        data = _crash_year_frame()
        full = compute_aggregate_metrics(data, include_full=True)["full"]
        returns = _daily_returns(data)
        self.assertAlmostEqual(full["sharpe"], _sharpe(returns), places=9)
        self.assertAlmostEqual(full["sortino"], _sortino(returns), places=9)
        expected_cagr = (data["RollingPnL"].iloc[-1] / data["RollingPnL"].iloc[0]) ** (252 / len(data)) - 1
        self.assertAlmostEqual(full["cagr_decimal"], expected_cagr, places=9)
        self.assertAlmostEqual(full["calmar"], full["cagr_percent"] / (full["max_drawdown"] * 100.0), places=9)

    @patch("stats.ExcludeBestReturnYear", True)
    def test_the_ranking_figures_do_not_change(self):
        data = _crash_year_frame()
        plain = compute_aggregate_metrics(data)
        with_full = compute_aggregate_metrics(data, include_full=True)
        for key, value in plain.items():
            if pd.isna(value):  # no closed trades in this frame, so the averages are NaN
                self.assertTrue(pd.isna(with_full[key]), key)
            else:
                self.assertEqual(with_full[key], value, key)

    @patch("stats.ExcludeBestReturnYear", False)
    def test_without_exclusion_full_matches_the_headline_ratios(self):
        data = _crash_year_frame()
        metrics = compute_aggregate_metrics(data, include_full=True)
        self.assertAlmostEqual(metrics["full"]["sharpe"], metrics["sharpe"], places=9)
        self.assertAlmostEqual(metrics["full"]["sortino"], metrics["sortino"], places=9)
        self.assertAlmostEqual(metrics["full"]["max_drawdown"], metrics["max_drawdown"], places=12)
        self.assertAlmostEqual(metrics["full"]["cagr_decimal"], metrics["cagr_decimal"], places=9)

    def test_best_year_concentration(self):
        year, best, share = best_year_concentration({2018: 0.10, 2019: 0.10, 2020: 1.00})
        self.assertEqual((year, best), (2020, 1.00))
        self.assertAlmostEqual(share, np.log(2.0) / (2 * np.log(1.1) + np.log(2.0)), places=12)

    def test_best_year_share_is_none_without_a_gain_to_share(self):
        self.assertEqual(best_year_concentration({}), (None, None, None))
        self.assertIsNone(best_year_concentration({2018: -0.1, 2019: -0.2})[2])
        # Other years net-lose: the best year supplies more than all of the return.
        self.assertGreater(best_year_concentration({2018: -0.2, 2019: -0.2, 2020: 0.8})[2], 1.0)

    @patch("stats.ExcludeBestReturnYear", True)
    def test_summary_payload_headlines_full_period_figures(self):
        from api.serializers import summary_payload

        data = _crash_year_frame()
        payload = summary_payload(data, 0, 0, "x")
        self.assertEqual(payload["excluded_year"], 2020)
        self.assertAlmostEqual(payload["full_max_drawdown"], data["Drawdown"].max())
        self.assertLess(payload["max_drawdown"], payload["full_max_drawdown"])
        self.assertEqual(payload["full_best_year"], 2020)
        for key in ("cagr_percent", "sharpe", "sortino", "calmar", "ulcer_index", "best_year_share"):
            self.assertIsNotNone(payload[f"full_{key}"], key)


if __name__ == "__main__":
    unittest.main()
