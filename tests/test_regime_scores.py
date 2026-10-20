import math
import unittest

import numpy as np
import pandas as pd

from api.market_regimes import (
    REGIME_SCORE_SORTINO_FULL_MARKS,
    _book_metrics_for_mask,
    _bucket_rows,
    _summary_metrics_of_trades,
    _trade_arrays,
    invested_bar_sortino,
    regime_score,
    robustness_regime_score,
)


class InvestedBarSortinoTests(unittest.TestCase):
    def test_matches_a_direct_calculation_on_the_selected_bars(self):
        returns = np.array([0.0, 0.02, -0.01, 0.03, 0.0, 0.0, -0.02, 0.01])
        mask = np.array([False, True, True, True, False, False, True, True])
        selected = returns[mask]
        downside = math.sqrt(np.mean(np.minimum(selected, 0) ** 2))
        expected = math.sqrt(252) * selected.mean() / downside
        self.assertAlmostEqual(invested_bar_sortino(returns, mask), expected)

    def test_flat_bars_do_not_change_it(self):
        # Same trades, many more flat bars around them: the old full-path Sortino shrank with
        # the extra flat time; this one must not.
        trade = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        short = np.concatenate(([0.0] * 5, trade))
        long = np.concatenate(([0.0] * 500, trade))
        a = invested_bar_sortino(short, np.r_[np.zeros(5, bool), np.ones(5, bool)])
        b = invested_bar_sortino(long, np.r_[np.zeros(500, bool), np.ones(5, bool)])
        self.assertAlmostEqual(a, b)

    def test_old_full_path_value_scaled_with_the_square_root_of_time_invested(self):
        trade = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        returns = np.concatenate(([0.0] * 95, trade))
        mask = np.r_[np.zeros(95, bool), np.ones(5, bool)]
        full_path = math.sqrt(252) * returns.mean() / math.sqrt(np.mean(np.minimum(returns, 0) ** 2))
        self.assertAlmostEqual(full_path / invested_bar_sortino(returns, mask), math.sqrt(5 / 100))

    def test_no_invested_bars_and_no_losing_bars(self):
        self.assertIsNone(invested_bar_sortino(np.zeros(5), np.zeros(5, bool)))
        self.assertEqual(invested_bar_sortino(np.array([0.01, 0.02]), np.ones(2, bool)), 0.0)


def _frame(n=500, seed=4):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    hold = np.zeros(n, bool)
    long_in = np.zeros(n, bool)
    long_out = np.zeros(n, bool)
    i = 5
    while i < n - 8:
        length = int(rng.integers(2, 6))
        long_in[i] = True
        hold[i + 1 : i + 1 + length] = True
        long_out[i + length] = True
        i += length + int(rng.integers(3, 12))
    returns = np.where(hold, rng.normal(0.004, 0.02, n), 0.0)
    equity = 10_000 * np.cumprod(1 + returns)
    pnl = np.zeros(n)
    entry_equity = None
    for k in range(n):
        if long_in[k]:
            entry_equity = equity[k]
        if long_out[k] and entry_equity:
            pnl[k] = equity[k] / entry_equity - 1
    return pd.DataFrame(
        {
            "Date": dates,
            "RollingPnL": equity,
            "LongTradeIn": long_in,
            "HoldLong": hold,
            "LongTradeOut": long_out,
            "TradePnL": pnl,
            "DaysInTrade": 0,
        }
    )


class SummaryMetricsTests(unittest.TestCase):
    def test_score_sortino_uses_invested_bars_and_summary_sortino_keeps_the_flat_path(self):
        data = _frame()
        arrays = _trade_arrays(data)
        mask = arrays["hold"]
        summary = _summary_metrics_of_trades(data["Date"], arrays["bar_returns"], mask, 252)
        self.assertAlmostEqual(
            summary["score_sortino"], invested_bar_sortino(arrays["bar_returns"], mask)
        )
        # The card Sortino still compounds the whole path (flat bars included), so it differs.
        self.assertNotAlmostEqual(summary["sortino"], summary["score_sortino"], places=3)

    def test_base_book_scores_over_bars_in_trade(self):
        data = _frame()
        arrays = _trade_arrays(data)
        closed = arrays["long_out"] & np.isfinite(arrays["trade_pnl"])
        base = _book_metrics_for_mask(
            data["Date"], arrays["bar_returns"], np.ones(len(data), bool),
            arrays["trade_pnl"], np.zeros(len(data)), closed, 252, invested_mask=arrays["hold"],
        )
        self.assertAlmostEqual(base["sortino"], invested_bar_sortino(arrays["bar_returns"], arrays["hold"]))


class RobustnessScoreTests(unittest.TestCase):
    YEARS = {2018: 0.20, 2019: 0.15, 2020: 0.30, 2021: 0.10}

    def test_formula(self):
        result = robustness_regime_score(0.015, self.YEARS)
        logs = [math.log(1 + v) for v in self.YEARS.values()]
        rob = (sum(logs) - max(logs)) / (len(logs) - 1) / (sum(logs) / len(logs))
        self.assertAlmostEqual(result["return"], 0.5)
        self.assertAlmostEqual(result["robustness"], rob)
        expected = 100 * 0.5 ** (0.25 / 0.55) * rob ** (0.30 / 0.55)
        self.assertAlmostEqual(result["score"], expected)

    def test_bounds_and_floors(self):
        top = robustness_regime_score(0.10, {2018: 0.1, 2019: 0.1, 2020: 0.1})
        self.assertAlmostEqual(top["score"], 100.0)
        bottom = robustness_regime_score(-0.05, {2018: 0.1})          # one year: robustness 0
        self.assertAlmostEqual(bottom["score"], 5.0)                  # both parts at the 0.05 floor
        self.assertEqual(bottom["robustness"], 0.0)

    def test_missing_trade_return_gives_no_score(self):
        self.assertIsNone(robustness_regime_score(None, self.YEARS)["score"])
        self.assertIsNone(robustness_regime_score(float("nan"), self.YEARS)["score"])

    def test_ignores_sortino_and_drawdown(self):
        # Same average trade and yearly returns, wildly different drawdown: identical score.
        a = robustness_regime_score(0.02, self.YEARS)["score"]
        low = regime_score(5.0, 0.02, 0.10, self.YEARS)["score"]
        high = regime_score(0.1, 0.02, 0.90, self.YEARS)["score"]
        self.assertNotAlmostEqual(low, high, places=1)
        self.assertAlmostEqual(a, robustness_regime_score(0.02, self.YEARS)["score"])

    def test_bucket_rows_carry_both_scores(self):
        data = _frame()
        arrays = _trade_arrays(data)
        labels = pd.Series(np.where(np.arange(len(data)) % 2 == 0, "a", "b"), index=data.index)
        rows = _bucket_rows(data, arrays, labels, ("a", "b"), 252)
        for row in rows:
            self.assertIn("robustness_score", row)
            self.assertIn("regime_score", row)
            if row["robustness_score"] is not None:
                self.assertGreaterEqual(row["robustness_score"], 5.0)
                self.assertLessEqual(row["robustness_score"], 100.0)

    def test_sortino_full_marks_is_the_documented_three(self):
        self.assertEqual(REGIME_SCORE_SORTINO_FULL_MARKS, 3.0)


if __name__ == "__main__":
    unittest.main()
