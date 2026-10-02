import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from stats import compute_aggregate_metrics, metrics_start_label, performance_frame, yearly_returns


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
        self.assertAlmostEqual(
            metrics["sharpe"],
            __import__("indicators").sharpes_ratio(excluded),
        )

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


if __name__ == "__main__":
    unittest.main()
