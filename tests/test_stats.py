import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from stats import compute_aggregate_metrics


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


if __name__ == "__main__":
    unittest.main()
