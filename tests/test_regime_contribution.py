"""Portfolio regime quality and leave-one-out contribution."""

import unittest

import numpy as np
import pandas as pd

from api.market_regimes import (
    _strategy_bar_returns,
    _summary_metrics_of_trades,
    build_regime_comparison,
    compute_market_regime_sharpe,
    regime_metric_delta,
    regime_score,
    score_regime_book,
)
from api.portfolio_service import _portfolio_atr_symbol


def _calendar(dates, vix):
    return pd.DataFrame(
        {
            "vix": vix,
            "vxn": [18.0] * len(dates),
            "spy_sma50": [2.0] * len(dates),
            "spy_sma200": [1.0] * len(dates),
            "breadth": [50.0] * len(dates),
        },
        index=dates,
    )


def _book_frame():
    """Two closed trades. Trade 1 is entered at VIX <= 15 and loses 10% over 2 days.
    Trade 2 is entered at VIX > 30 and gains 2% over 2 days.
    """
    dates = pd.bdate_range("2024-01-02", periods=7)
    return pd.DataFrame(
        {
            "Date": dates,
            "RollingPnL": [1.0, 1.0, 0.9, 0.945, 0.945, 0.9639, 0.983178],
            "LongTradeIn": [False, True, False, False, True, False, False],
            "HoldLong": [False, False, True, True, False, True, True],
            "LongTradeOut": [False, False, False, True, False, False, True],
            "TradePnL": [0.0, 0.0, 0.0, -0.10, 0.0, 0.0, 0.02],
            "DaysInTrade": [0, 0, 1, 2, 0, 1, 2],
            "ATR20": [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            "ATR50": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        }
    )


class ScoreRegimeBookTests(unittest.TestCase):
    def setUp(self):
        self.data = _book_frame()
        self.calendar = _calendar(self.data["Date"], [10.0, 10.0, 40.0, 40.0, 40.0, 10.0, 10.0])
        self.book = score_regime_book(self.data, self.calendar)

    def test_base_matches_regime_score_on_the_full_path(self):
        returns = _strategy_bar_returns(self.data["RollingPnL"])
        summary = _summary_metrics_of_trades(
            self.data["Date"],
            returns,
            np.ones(len(self.data), dtype=bool),
            252,
        )
        closed = self.data["LongTradeOut"].to_numpy(dtype=bool)
        pnl = self.data["TradePnL"].to_numpy(dtype=float)
        avg_trade = float(np.mean(pnl[closed]))
        expected = regime_score(
            summary["score_sortino"],
            avg_trade,
            summary["score_max_drawdown"],
            summary["year_returns"],
        )
        base = self.book["base"]
        self.assertIsNotNone(expected["score"])
        self.assertAlmostEqual(base["regime_score"], expected["score"])
        self.assertAlmostEqual(base["sortino"], summary["score_sortino"])
        self.assertAlmostEqual(base["avg_trade_return"], avg_trade)
        self.assertAlmostEqual(base["max_drawdown"], summary["score_max_drawdown"])
        self.assertAlmostEqual(base["robustness"], expected["robustness"])
        self.assertAlmostEqual(base["cagr"], summary["cagr"])
        self.assertEqual(base["trades"], 2)

    def test_bucket_score_matches_overview_regime_score(self):
        overview = compute_market_regime_sharpe(self.data, self.calendar)
        for row in overview["vix"]:
            bucket = self.book["regimes"]["vix"][row["key"]]
            if row["regime_score"] is None:
                self.assertIsNone(bucket["regime_score"])
            else:
                self.assertAlmostEqual(bucket["regime_score"], row["regime_score"])
            if row["cagr"] is None:
                self.assertIsNone(bucket["cagr"])
            else:
                self.assertAlmostEqual(bucket["cagr"], row["cagr"])
        self.assertEqual(self.book["regimes"]["vix"]["le_15"]["trades"], 1)
        self.assertEqual(self.book["regimes"]["vix"]["gt_30"]["trades"], 1)
        self.assertEqual(self.book["regimes"]["vix"]["15_20"]["trades"], 0)

    def test_exposure_is_pnl_per_day_held(self):
        self.assertAlmostEqual(self.book["regimes"]["vix"]["le_15"]["exposure"], -0.10 / 2)
        self.assertAlmostEqual(self.book["regimes"]["vix"]["gt_30"]["exposure"], 0.02 / 2)
        self.assertAlmostEqual(self.book["base"]["exposure"], (-0.10 + 0.02) / 4)
        self.assertIsNone(self.book["regimes"]["vix"]["15_20"]["exposure"])

    def test_zero_day_trades_do_not_change_exposure(self):
        data = self.data.copy()
        data.loc[data.index[-1], "DaysInTrade"] = 0
        book = score_regime_book(data, self.calendar)
        self.assertAlmostEqual(book["base"]["exposure"], -0.10 / 2)
        self.assertAlmostEqual(book["base"]["avg_trade_return"], (-0.10 + 0.02) / 2)
        self.assertIsNone(book["regimes"]["vix"]["gt_30"]["exposure"])


class ContributionDeltaTests(unittest.TestCase):
    def test_contribution_is_full_minus_leave_one_out(self):
        dates = pd.bdate_range("2023-01-02", periods=8)
        full = _book_frame()
        full["Date"] = dates[:7]
        reduced = full.copy()
        reduced["RollingPnL"] = [1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8]
        reduced["TradePnL"] = [0.0, 0.0, 0.0, -0.20, 0.0, 0.0, 0.0]
        reduced["LongTradeOut"] = [False, False, False, True, False, False, False]
        reduced["HoldLong"] = [False, False, True, True, False, False, False]
        calendar = _calendar(full["Date"], [10.0, 10.0, 40.0, 40.0, 40.0, 10.0, 10.0])
        full_book = score_regime_book(full, calendar)
        reduced_book = score_regime_book(reduced, calendar)

        delta = regime_metric_delta(full_book["base"], reduced_book["base"])
        self.assertIsNotNone(full_book["base"]["regime_score"])
        self.assertIsNotNone(reduced_book["base"]["regime_score"])
        self.assertAlmostEqual(
            delta["regime_score"],
            full_book["base"]["regime_score"] - reduced_book["base"]["regime_score"],
        )
        self.assertAlmostEqual(
            delta["sortino"],
            full_book["base"]["sortino"] - reduced_book["base"]["sortino"],
        )
        self.assertAlmostEqual(
            delta["max_drawdown"],
            full_book["base"]["max_drawdown"] - reduced_book["base"]["max_drawdown"],
        )
        self.assertLess(delta["max_drawdown"], 0)
        self.assertAlmostEqual(
            delta["exposure"],
            full_book["base"]["exposure"] - reduced_book["base"]["exposure"],
        )
        self.assertAlmostEqual(
            delta["cagr"],
            full_book["base"]["cagr"] - reduced_book["base"]["cagr"],
        )

        comparison = build_regime_comparison(full_book, reduced_book)
        self.assertAlmostEqual(
            comparison["base"]["delta"]["regime_score"],
            delta["regime_score"],
        )
        self.assertEqual(
            comparison["base"]["with"]["regime_score"],
            full_book["base"]["regime_score"],
        )
        self.assertEqual(
            comparison["base"]["without"]["regime_score"],
            reduced_book["base"]["regime_score"],
        )
        bucket = comparison["regimes"]["vix"]["le_15"]
        full_bucket = full_book["regimes"]["vix"]["le_15"]["regime_score"]
        reduced_bucket = reduced_book["regimes"]["vix"]["le_15"]["regime_score"]
        self.assertAlmostEqual(bucket["delta"]["regime_score"], full_bucket - reduced_bucket)

    def test_delta_is_blank_when_either_score_is_missing(self):
        full = {"regime_score": 70.0, "sortino": 2.0, "avg_trade_return": 0.01, "max_drawdown": 0.2, "robustness": 0.5, "exposure": 0.004}
        empty = {"regime_score": None, "sortino": None, "avg_trade_return": None, "max_drawdown": None, "robustness": None, "exposure": None}
        delta = regime_metric_delta(full, empty)
        self.assertTrue(all(value is None for value in delta.values()))


class PortfolioAtrSymbolTests(unittest.TestCase):
    def test_single_symbol_and_proxy(self):
        one = _FakeStrategy("spy")
        other = _FakeStrategy("qqq")
        self.assertEqual(_portfolio_atr_symbol([one, one], None), "SPY")
        self.assertIsNone(_portfolio_atr_symbol([one, other], None))
        self.assertEqual(_portfolio_atr_symbol([one, other], "IWM"), "IWM")


class _FakeStrategy:
    def __init__(self, symbol):
        self.symbol = symbol
        self.proxy_symbol = None


if __name__ == "__main__":
    unittest.main()
