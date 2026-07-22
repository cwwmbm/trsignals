import math
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from api.portfolio_service import (
    START_CAPITAL,
    StrategySignals,
    build_strategy_contribution,
    clamp_shapley_samples,
    compute_portfolio_contribution,
    compute_portfolio_shapley,
    exposure_adjusted_return,
    extract_contribution_metrics,
    flat_cash_frame,
    holding_overlap,
    marginal_cagr_per_10pp_exposure,
    portfolio_utility,
    simulate_portfolio_overlay,
    _metric_margin,
    _shapley_orderings,
)
from stats import calmar_ratio, time_under_water_percent, ulcer_index


def _strategy(strategy_id: str, name: str, symbol: str = "AAA") -> SimpleNamespace:
    return SimpleNamespace(
        id=strategy_id,
        name=name,
        symbol=symbol,
        direction="long",
        hold_days=2,
        profit=1,
        confirm_symbols=[],
        proxy_symbol=None,
    )


def _hold_signals(
    strategy_id: str,
    name: str,
    dates: pd.DatetimeIndex,
    hold: list[int],
    symbol: str = "AAA",
) -> StrategySignals:
    """Build signals from an explicit HoldLong mask (same-bar hold semantics)."""
    hold_arr = np.array(hold, dtype=bool)
    trade_in = np.zeros(len(hold), dtype=bool)
    trade_out = np.zeros(len(hold), dtype=bool)
    prev = False
    for index, is_hold in enumerate(hold_arr):
        if is_hold and not prev:
            trade_in[index] = True
        if prev and not is_hold:
            trade_out[index] = True
        prev = bool(is_hold)
    return StrategySignals(
        strategy_id=strategy_id,
        name=name,
        symbol=symbol,
        dates=dates,
        long_trade_in=trade_in,
        hold_long=hold_arr,
        long_trade_out=trade_out,
    )


class HoldingOverlapTests(unittest.TestCase):
    def test_unique_versus_overlapping_days(self):
        candidate = np.array([1, 1, 1, 0, 0], dtype=bool)
        reduced = np.array([0, 1, 1, 1, 0], dtype=bool)
        overlap = holding_overlap(candidate, reduced)
        self.assertEqual(overlap["candidate_holding_days"], 3)
        self.assertEqual(overlap["overlapping_holding_days"], 2)
        self.assertEqual(overlap["unique_holding_days"], 1)
        self.assertAlmostEqual(overlap["unique_holding_percent"], 1 / 3)
        self.assertAlmostEqual(overlap["redundant_holding_percent"], 2 / 3)

    def test_zero_candidate_holding_days(self):
        overlap = holding_overlap(np.zeros(4, dtype=bool), np.array([1, 0, 1, 0], dtype=bool))
        self.assertEqual(overlap["candidate_holding_days"], 0)
        self.assertEqual(overlap["overlapping_holding_days"], 0)
        self.assertEqual(overlap["unique_holding_days"], 0)
        self.assertIsNone(overlap["unique_holding_percent"])
        self.assertIsNone(overlap["redundant_holding_percent"])

    def test_fully_redundant(self):
        candidate = np.array([0, 1, 0], dtype=bool)
        reduced = np.array([1, 1, 1], dtype=bool)
        overlap = holding_overlap(candidate, reduced)
        self.assertEqual(overlap["unique_holding_days"], 0)
        self.assertAlmostEqual(overlap["unique_holding_percent"], 0.0)
        self.assertAlmostEqual(overlap["redundant_holding_percent"], 1.0)


class ContributionDeltaTests(unittest.TestCase):
    def test_percentage_point_delta_directions(self):
        full = {
            "cagr_percent": 30.0,
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": 0.35,
            "rolling_pnl": 20000.0,
            "time_in_market_percent": 60.0,
            "trades": 10,
            "worst_calendar_year_pnl_percent": -10.0,
            "avg_days_in_trade": 5.0,
            "median_days_in_trade": 4.0,
            "calmar": 30.0 / 35.0,
            "ulcer_index": 12.0,
            "time_under_water_percent": 40.0,
            "trades_per_year": 8.0,
            "utility": 10.0,
        }
        reduced = {
            "cagr_percent": 27.0,
            "sharpe": 1.2,
            "sortino": 1.8,
            "max_drawdown": 0.29,
            "rolling_pnl": 18000.0,
            "time_in_market_percent": 48.0,
            "trades": 7,
            "worst_calendar_year_pnl_percent": -8.0,
            "avg_days_in_trade": 4.0,
            "median_days_in_trade": 3.0,
            "calmar": 27.0 / 29.0,
            "ulcer_index": 10.0,
            "time_under_water_percent": 35.0,
            "trades_per_year": 6.0,
            "utility": 8.0,
        }
        row = build_strategy_contribution(
            strategy_id="a",
            strategy_name="A",
            full_metrics=full,
            reduced_metrics=reduced,
            overlap=holding_overlap(np.array([1, 0]), np.array([0, 0])),
        )
        self.assertAlmostEqual(row["cagr_contribution_pp"], 3.0)
        self.assertAlmostEqual(row["sharpe_delta"], 0.3)
        self.assertAlmostEqual(row["sortino_delta"], 0.2)
        self.assertAlmostEqual(row["calmar_delta"], full["calmar"] - reduced["calmar"])
        self.assertAlmostEqual(row["ulcer_index_delta"], 2.0)
        self.assertAlmostEqual(row["time_under_water_delta_pp"], 5.0)
        self.assertAlmostEqual(row["max_drawdown_effect_pp"], 6.0)
        self.assertAlmostEqual(row["added_exposure_pp"], 12.0)
        self.assertEqual(row["added_portfolio_trades"], 3)
        self.assertAlmostEqual(row["marginal_utility"], 2.0)

    def test_improves_drawdown_while_lowering_cagr(self):
        full = {
            "cagr_percent": 20.0,
            "sharpe": 1.0,
            "sortino": 1.0,
            "max_drawdown": 0.15,
            "rolling_pnl": 16000.0,
            "time_in_market_percent": 40.0,
            "trades": 8,
            "worst_calendar_year_pnl_percent": -5.0,
            "avg_days_in_trade": 3.0,
            "median_days_in_trade": 3.0,
            "calmar": None,
            "ulcer_index": 8.0,
            "time_under_water_percent": 30.0,
            "trades_per_year": 5.0,
            "utility": 5.0,
        }
        reduced = {
            "cagr_percent": 25.0,
            "sharpe": 1.2,
            "sortino": 1.2,
            "max_drawdown": 0.22,
            "rolling_pnl": 17000.0,
            "time_in_market_percent": 35.0,
            "trades": 6,
            "worst_calendar_year_pnl_percent": -4.0,
            "avg_days_in_trade": 2.0,
            "median_days_in_trade": 2.0,
            "calmar": None,
            "ulcer_index": 11.0,
            "time_under_water_percent": 40.0,
            "trades_per_year": 4.0,
            "utility": 7.0,
        }
        row = build_strategy_contribution(
            strategy_id="defensive",
            strategy_name="Defensive",
            full_metrics=full,
            reduced_metrics=reduced,
            overlap=holding_overlap(np.array([1, 1, 0]), np.array([1, 0, 0])),
        )
        self.assertLess(row["cagr_contribution_pp"], 0)
        self.assertLess(row["max_drawdown_effect_pp"], 0)
        self.assertLess(row["ulcer_index_delta"], 0)
        self.assertLess(row["time_under_water_delta_pp"], 0)

    def test_undefined_risk_adjusted_deltas_are_null(self):
        full = {
            "cagr_percent": 10.0,
            "sharpe": None,
            "sortino": None,
            "max_drawdown": 0.1,
            "rolling_pnl": 16000.0,
            "time_in_market_percent": 20.0,
            "trades": 2,
            "worst_calendar_year_pnl_percent": None,
            "avg_days_in_trade": None,
            "median_days_in_trade": None,
            "calmar": None,
            "ulcer_index": None,
            "time_under_water_percent": None,
            "trades_per_year": None,
            "utility": None,
        }
        reduced = {
            "cagr_percent": 8.0,
            "sharpe": 1.0,
            "sortino": 1.0,
            "max_drawdown": 0.08,
            "rolling_pnl": 15500.0,
            "time_in_market_percent": 10.0,
            "trades": 1,
            "worst_calendar_year_pnl_percent": None,
            "avg_days_in_trade": None,
            "median_days_in_trade": None,
            "calmar": 1.0,
            "ulcer_index": 5.0,
            "time_under_water_percent": 20.0,
            "trades_per_year": 2.0,
            "utility": 3.0,
        }
        row = build_strategy_contribution(
            strategy_id="a",
            strategy_name="A",
            full_metrics=full,
            reduced_metrics=reduced,
            overlap=holding_overlap(np.zeros(2, dtype=bool), np.zeros(2, dtype=bool)),
        )
        self.assertIsNone(row["sharpe_delta"])
        self.assertIsNone(row["sortino_delta"])
        self.assertIsNone(row["calmar_delta"])
        self.assertIsNone(row["ulcer_index_delta"])
        self.assertIsNone(row["time_under_water_delta_pp"])
        self.assertIsNone(row["marginal_utility"])
        self.assertIsNone(row["worst_calendar_year_pnl_delta_pp"])


class UtilityAndDrawdownMetricTests(unittest.TestCase):
    def test_portfolio_utility_formula(self):
        value = portfolio_utility(
            cagr_percent=30.0,
            sharpe=1.5,
            max_drawdown=0.35,
            time_in_market_percent=60.0,
            trades_per_year=20.0,
        )
        # 1*30 + 4*1.5 - 0.5*35 - 0.1*60 - 0.02*20 = 30 + 6 - 17.5 - 6 - 0.4 = 12.1
        self.assertAlmostEqual(value, 12.1)

    def test_portfolio_utility_null_when_inputs_missing(self):
        self.assertIsNone(
            portfolio_utility(
                cagr_percent=30.0,
                sharpe=None,
                max_drawdown=0.2,
                time_in_market_percent=40.0,
                trades_per_year=10.0,
            )
        )

    def test_marginal_utility_is_full_minus_without(self):
        full = {
            "cagr_percent": 30.0,
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": 0.35,
            "time_in_market_percent": 60.0,
            "trades_per_year": 20.0,
            "utility": portfolio_utility(
                cagr_percent=30.0,
                sharpe=1.5,
                max_drawdown=0.35,
                time_in_market_percent=60.0,
                trades_per_year=20.0,
            ),
            "calmar": calmar_ratio(30.0, 0.35),
            "ulcer_index": 10.0,
            "time_under_water_percent": 40.0,
            "rolling_pnl": 20000.0,
            "trades": 10,
            "worst_calendar_year_pnl_percent": None,
            "avg_days_in_trade": None,
            "median_days_in_trade": None,
        }
        reduced = {
            "cagr_percent": 27.0,
            "sharpe": 1.2,
            "sortino": 1.8,
            "max_drawdown": 0.29,
            "time_in_market_percent": 48.0,
            "trades_per_year": 15.0,
            "utility": portfolio_utility(
                cagr_percent=27.0,
                sharpe=1.2,
                max_drawdown=0.29,
                time_in_market_percent=48.0,
                trades_per_year=15.0,
            ),
            "calmar": calmar_ratio(27.0, 0.29),
            "ulcer_index": 8.0,
            "time_under_water_percent": 30.0,
            "rolling_pnl": 18000.0,
            "trades": 7,
            "worst_calendar_year_pnl_percent": None,
            "avg_days_in_trade": None,
            "median_days_in_trade": None,
        }
        row = build_strategy_contribution(
            strategy_id="a",
            strategy_name="A",
            full_metrics=full,
            reduced_metrics=reduced,
            overlap=holding_overlap(np.array([1]), np.array([0])),
        )
        self.assertAlmostEqual(row["marginal_utility"], full["utility"] - reduced["utility"])

    def test_ulcer_and_time_under_water_on_path(self):
        dates = pd.to_datetime(
            ["2023-01-02", "2023-01-03", "2023-01-04", "2024-01-02", "2024-01-03"]
        )
        # Peak then underwater then recover partially
        rolling = [100.0, 110.0, 99.0, 105.0, 105.0]
        running_max = np.maximum.accumulate(rolling)
        drawdown = (running_max - rolling) / running_max
        frame = pd.DataFrame(
            {
                "Date": dates,
                "RollingPnL": rolling,
                "Drawdown": drawdown,
                "HoldLong": [False, True, True, True, False],
                "LongTradeIn": [True, False, False, False, False],
                "LongTradeOut": [False, False, False, False, True],
                "TradePnL": [0.0, 0.0, 0.0, 0.0, 0.05],
                "DaysInTrade": [0, 1, 2, 3, 4],
            }
        )
        ui = ulcer_index(frame)
        tuw = time_under_water_percent(frame)
        self.assertIsNotNone(ui)
        self.assertGreater(ui, 0)
        # 3 of 5 bars have drawdown > 0 after peaking at 110
        self.assertAlmostEqual(tuw, 60.0)
        self.assertAlmostEqual(calmar_ratio(20.0, 0.1), 2.0)
        self.assertIsNone(calmar_ratio(20.0, 0.0))

        metrics = extract_contribution_metrics(frame)
        self.assertIn("utility", metrics)
        self.assertIn("calmar", metrics)
        self.assertIn("ulcer_index", metrics)
        self.assertIn("time_under_water_percent", metrics)
        self.assertIn("trades_per_year", metrics)
        self.assertIn("exposure_adjusted_return", metrics)

    def test_exposure_adjusted_return_and_marginal_cagr_per_10pp(self):
        self.assertAlmostEqual(exposure_adjusted_return(30.0, 60.0), 0.5)
        self.assertIsNone(exposure_adjusted_return(30.0, 0.0))
        self.assertIsNone(exposure_adjusted_return(None, 60.0))

        # +3 pp CAGR from +12 pp exposure => 3/12*10 = 2.5 pp CAGR per 10pp exposure
        self.assertAlmostEqual(marginal_cagr_per_10pp_exposure(3.0, 12.0), 2.5)
        self.assertIsNone(marginal_cagr_per_10pp_exposure(3.0, 0.0))

        full = {
            "cagr_percent": 30.0,
            "time_in_market_percent": 60.0,
            "exposure_adjusted_return": exposure_adjusted_return(30.0, 60.0),
            "sharpe": 1.0,
            "sortino": 1.0,
            "max_drawdown": 0.2,
            "rolling_pnl": 20000.0,
            "trades": 10,
            "worst_calendar_year_pnl_percent": None,
            "avg_days_in_trade": None,
            "median_days_in_trade": None,
            "calmar": None,
            "ulcer_index": None,
            "time_under_water_percent": None,
            "trades_per_year": 5.0,
            "utility": None,
        }
        reduced = {
            "cagr_percent": 27.0,
            "time_in_market_percent": 48.0,
            "exposure_adjusted_return": exposure_adjusted_return(27.0, 48.0),
            "sharpe": 1.0,
            "sortino": 1.0,
            "max_drawdown": 0.2,
            "rolling_pnl": 18000.0,
            "trades": 7,
            "worst_calendar_year_pnl_percent": None,
            "avg_days_in_trade": None,
            "median_days_in_trade": None,
            "calmar": None,
            "ulcer_index": None,
            "time_under_water_percent": None,
            "trades_per_year": 4.0,
            "utility": None,
        }
        row = build_strategy_contribution(
            strategy_id="a",
            strategy_name="A",
            full_metrics=full,
            reduced_metrics=reduced,
            overlap=holding_overlap(np.array([1]), np.array([0])),
        )
        self.assertAlmostEqual(row["cagr_contribution_pp"], 3.0)
        self.assertAlmostEqual(row["added_exposure_pp"], 12.0)
        self.assertAlmostEqual(row["marginal_cagr_per_10pp_exposure"], 2.5)
        self.assertAlmostEqual(
            row["exposure_adjusted_return_delta"],
            full["exposure_adjusted_return"] - reduced["exposure_adjusted_return"],
        )


class PortfolioContributionIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Span two calendar years so ExcludeBestReturnYear still leaves metric rows.
        self.dates = pd.to_datetime(
            [
                "2023-12-28",
                "2023-12-29",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
            ]
        )
        # Day:       1 2 3 4 5 6 7
        # Strategy A 1 1 1 0 0 0 0
        # Strategy B 0 0 1 1 1 0 0
        # Strategy C 0 0 1 0 0 0 0
        self.hold_a = [1, 1, 1, 0, 0, 0, 0]
        self.hold_b = [0, 0, 1, 1, 1, 0, 0]
        self.hold_c = [0, 0, 1, 0, 0, 0, 0]
        self.strategy_a = _strategy("a", "Strategy A")
        self.strategy_b = _strategy("b", "Strategy B")
        self.strategy_c = _strategy("c", "Strategy C")
        self.signals_by_id = {
            "a": _hold_signals("a", "Strategy A", self.dates, self.hold_a),
            "b": _hold_signals("b", "Strategy B", self.dates, self.hold_b),
            "c": _hold_signals("c", "Strategy C", self.dates, self.hold_c),
        }
        # Positive then negative path so drawdowns / returns differ when B extends.
        track = pd.Series([0.02, 0.02, 0.02, -0.03, -0.03, 0.01, 0.0], index=self.dates)
        close = pd.Series([100.0, 102.0, 104.0, 101.0, 98.0, 99.0, 99.0], index=self.dates)
        self.track_changes = {"AAA": track}
        self.close_series = {"AAA": close}
        self.strategies = [self.strategy_a, self.strategy_b, self.strategy_c]

    def _overlay(self, strategies, order):
        return simulate_portfolio_overlay(
            strategies,
            {strategy.id: self.signals_by_id[strategy.id] for strategy in strategies},
            overlap_mode="hold_until_all_exit",
            global_proxy=None,
            track_changes=self.track_changes,
            close_series=self.close_series,
            strategy_order=order,
            master_dates=self.dates,
        )

    def test_abc_timeline_redundant_and_extender(self):
        full = self._overlay(self.strategies, ["a", "b", "c"])
        without_b = self._overlay([self.strategy_a, self.strategy_c], ["a", "c"])
        without_c = self._overlay([self.strategy_a, self.strategy_b], ["a", "b"])

        # Union hold: days 1-5
        self.assertEqual(full["HoldLong"].astype(int).tolist(), [1, 1, 1, 1, 1, 0, 0])
        # Removing B shortens the portfolio position to A's span
        self.assertEqual(without_b["HoldLong"].astype(int).tolist(), [1, 1, 1, 0, 0, 0, 0])
        # Removing C does not change the portfolio position
        self.assertEqual(
            without_c["HoldLong"].astype(int).tolist(),
            full["HoldLong"].astype(int).tolist(),
        )

        contribution = compute_portfolio_contribution(
            self.strategies,
            self.signals_by_id,
            full_frame=full,
            overlap_mode="hold_until_all_exit",
            global_proxy=None,
            track_changes=self.track_changes,
            close_series=self.close_series,
            strategy_order=["a", "b", "c"],
        )
        by_id = {row["strategy_id"]: row for row in contribution}

        self.assertEqual(by_id["c"]["unique_holding_days"], 0)
        self.assertAlmostEqual(by_id["c"]["unique_holding_percent"], 0.0)
        self.assertAlmostEqual(by_id["c"]["redundant_holding_percent"], 1.0)
        self.assertAlmostEqual(by_id["c"]["added_exposure_pp"], 0.0)

        # B extends beyond A: unique days on 4 and 5
        self.assertEqual(by_id["b"]["unique_holding_days"], 2)
        self.assertGreater(by_id["b"]["added_exposure_pp"], 0)

    def test_strategy_adds_entirely_new_positions(self):
        dates = self.dates
        signals = {
            "a": _hold_signals("a", "A", dates, [1, 1, 0, 0, 0, 0, 0]),
            "b": _hold_signals("b", "B", dates, [0, 0, 0, 1, 1, 0, 0]),
        }
        strategies = [_strategy("a", "A"), _strategy("b", "B")]
        full = simulate_portfolio_overlay(
            strategies,
            signals,
            overlap_mode="hold_until_all_exit",
            global_proxy=None,
            track_changes=self.track_changes,
            close_series=self.close_series,
            strategy_order=["a", "b"],
            master_dates=dates,
        )
        contribution = compute_portfolio_contribution(
            strategies,
            signals,
            full_frame=full,
            overlap_mode="hold_until_all_exit",
            global_proxy=None,
            track_changes=self.track_changes,
            close_series=self.close_series,
            strategy_order=["a", "b"],
        )
        by_id = {row["strategy_id"]: row for row in contribution}
        self.assertEqual(by_id["b"]["overlapping_holding_days"], 0)
        self.assertEqual(by_id["b"]["unique_holding_days"], 2)
        self.assertAlmostEqual(by_id["b"]["unique_holding_percent"], 1.0)

    def test_single_strategy_reduced_is_flat_cash(self):
        strategies = [self.strategy_a]
        signals = {"a": self.signals_by_id["a"]}
        full = self._overlay(strategies, ["a"])
        contribution = compute_portfolio_contribution(
            strategies,
            signals,
            full_frame=full,
            overlap_mode="hold_until_all_exit",
            global_proxy=None,
            track_changes=self.track_changes,
            close_series=self.close_series,
            strategy_order=["a"],
        )
        self.assertEqual(len(contribution), 1)
        row = contribution[0]
        self.assertAlmostEqual(row["without_strategy"]["rolling_pnl"], START_CAPITAL)
        self.assertAlmostEqual(row["without_strategy"]["time_in_market_percent"], 0.0)
        self.assertEqual(row["without_strategy"]["trades"], 0)
        self.assertEqual(row["unique_holding_days"], row["candidate_holding_days"])
        self.assertAlmostEqual(row["unique_holding_percent"], 1.0)
        # Contribution equals standalone effect vs cash
        standalone = extract_contribution_metrics(full)
        cash = extract_contribution_metrics(flat_cash_frame(self.dates))
        self.assertAlmostEqual(
            row["cagr_contribution_pp"],
            standalone["cagr_percent"] - cash["cagr_percent"],
        )

    def test_flat_cash_frame_has_no_exposure(self):
        frame = flat_cash_frame(self.dates)
        self.assertFalse(frame["HoldLong"].any())
        self.assertTrue((frame["RollingPnL"] == START_CAPITAL).all())
        metrics = extract_contribution_metrics(frame)
        self.assertEqual(metrics["trades"], 0)
        self.assertAlmostEqual(metrics["time_in_market_percent"], 0.0)
        self.assertTrue(math.isfinite(metrics["sharpe"] or 0.0) or metrics["sharpe"] is None)


class ShapleyContributionTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(
            [
                "2023-12-28",
                "2023-12-29",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
            ]
        )
        self.hold_a = [1, 1, 1, 0, 0, 0, 0]
        self.hold_b = [0, 0, 1, 1, 1, 0, 0]
        self.hold_c = [0, 0, 1, 0, 0, 0, 0]
        self.strategy_a = _strategy("a", "Strategy A")
        self.strategy_b = _strategy("b", "Strategy B")
        self.strategy_c = _strategy("c", "Strategy C")
        self.signals_by_id = {
            "a": _hold_signals("a", "Strategy A", self.dates, self.hold_a),
            "b": _hold_signals("b", "Strategy B", self.dates, self.hold_b),
            "c": _hold_signals("c", "Strategy C", self.dates, self.hold_c),
        }
        track = pd.Series([0.02, 0.02, 0.02, -0.03, -0.03, 0.01, 0.0], index=self.dates)
        close = pd.Series([100.0, 102.0, 104.0, 101.0, 98.0, 99.0, 99.0], index=self.dates)
        self.track_changes = {"AAA": track}
        self.close_series = {"AAA": close}
        self.strategies = [self.strategy_a, self.strategy_b, self.strategy_c]

    def _shapley(self, strategies, order, samples=64, seed=0):
        return compute_portfolio_shapley(
            strategies,
            {strategy.id: self.signals_by_id[strategy.id] for strategy in strategies},
            master_dates=self.dates,
            overlap_mode="hold_until_all_exit",
            global_proxy=None,
            track_changes=self.track_changes,
            close_series=self.close_series,
            strategy_order=order,
            samples=samples,
            seed=seed,
        )

    def test_clamp_samples_and_exact_flag(self):
        self.assertEqual(clamp_shapley_samples(1), 8)
        self.assertEqual(clamp_shapley_samples(64), 64)
        self.assertEqual(clamp_shapley_samples(1000), 512)

        import random

        exact_orders, exact, used = _shapley_orderings(["a", "b"], samples=64, rng=random.Random(0))
        self.assertTrue(exact)
        self.assertEqual(used, 2)
        self.assertEqual(len(exact_orders), 2)

        approx_orders, exact_flag, used_approx = _shapley_orderings(
            ["a", "b", "c", "d"],
            samples=8,
            rng=random.Random(1),
        )
        self.assertFalse(exact_flag)
        self.assertEqual(used_approx, 8)
        self.assertEqual(len(approx_orders), 8)

    def test_empty_coalition_cash_baseline(self):
        result = self._shapley([self.strategy_a], ["a"], samples=8)
        cash = extract_contribution_metrics(flat_cash_frame(self.dates))
        full = extract_contribution_metrics(
            simulate_portfolio_overlay(
                [self.strategy_a],
                {"a": self.signals_by_id["a"]},
                overlap_mode="hold_until_all_exit",
                global_proxy=None,
                track_changes=self.track_changes,
                close_series=self.close_series,
                strategy_order=["a"],
                master_dates=self.dates,
            )
        )
        row = result["shapley"][0]
        self.assertTrue(result["exact"])
        self.assertAlmostEqual(
            row["cagr_contribution_pp"],
            full["cagr_percent"] - cash["cagr_percent"],
        )
        self.assertAlmostEqual(
            row["added_exposure_pp"],
            full["time_in_market_percent"] - cash["time_in_market_percent"],
        )

    def test_two_strategy_exact_efficiency(self):
        strategies = [self.strategy_a, self.strategy_b]
        result = self._shapley(strategies, ["a", "b"], samples=64)
        self.assertTrue(result["exact"])
        self.assertEqual(result["samples_used"], 2)

        full = extract_contribution_metrics(
            simulate_portfolio_overlay(
                strategies,
                {"a": self.signals_by_id["a"], "b": self.signals_by_id["b"]},
                overlap_mode="hold_until_all_exit",
                global_proxy=None,
                track_changes=self.track_changes,
                close_series=self.close_series,
                strategy_order=["a", "b"],
                master_dates=self.dates,
            )
        )
        cash = extract_contribution_metrics(flat_cash_frame(self.dates))
        by_id = {row["strategy_id"]: row for row in result["shapley"]}

        self.assertAlmostEqual(
            by_id["a"]["added_exposure_pp"] + by_id["b"]["added_exposure_pp"],
            full["time_in_market_percent"] - cash["time_in_market_percent"],
            places=6,
        )
        self.assertAlmostEqual(
            by_id["a"]["cagr_contribution_pp"] + by_id["b"]["cagr_contribution_pp"],
            full["cagr_percent"] - cash["cagr_percent"],
            places=6,
        )
        self.assertAlmostEqual(
            by_id["a"]["final_equity_delta"] + by_id["b"]["final_equity_delta"],
            full["rolling_pnl"] - cash["rolling_pnl"],
            places=6,
        )

    def test_redundant_versus_extender_exposure(self):
        result = self._shapley(self.strategies, ["a", "b", "c"], samples=64)
        self.assertTrue(result["exact"])
        self.assertEqual(result["samples_used"], 6)
        by_id = {row["strategy_id"]: row for row in result["shapley"]}

        # C only adds exposure when ordered first (2 of 6 permutations).
        self.assertAlmostEqual(by_id["c"]["added_exposure_pp"], (100.0 / 7.0) * (2.0 / 6.0), places=6)
        self.assertGreater(by_id["b"]["added_exposure_pp"], by_id["c"]["added_exposure_pp"])
        self.assertGreater(by_id["b"]["added_exposure_pp"], 0.0)

        leave_one_out = compute_portfolio_contribution(
            self.strategies,
            self.signals_by_id,
            full_frame=simulate_portfolio_overlay(
                self.strategies,
                self.signals_by_id,
                overlap_mode="hold_until_all_exit",
                global_proxy=None,
                track_changes=self.track_changes,
                close_series=self.close_series,
                strategy_order=["a", "b", "c"],
                master_dates=self.dates,
            ),
            overlap_mode="hold_until_all_exit",
            global_proxy=None,
            track_changes=self.track_changes,
            close_series=self.close_series,
            strategy_order=["a", "b", "c"],
        )
        loo_c = next(row for row in leave_one_out if row["strategy_id"] == "c")
        self.assertAlmostEqual(loo_c["added_exposure_pp"], 0.0)

    def test_null_safe_metric_margins(self):
        self.assertIsNone(_metric_margin({"sharpe": None}, {"sharpe": 1.0}, "sharpe"))
        self.assertIsNone(_metric_margin({"sharpe": 1.0}, {"sharpe": None}, "sharpe"))
        self.assertIsNone(_metric_margin({"sharpe": float("nan")}, {"sharpe": 1.0}, "sharpe"))
        self.assertAlmostEqual(
            _metric_margin({"cagr_percent": 12.0}, {"cagr_percent": 10.0}, "cagr_percent"),
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
