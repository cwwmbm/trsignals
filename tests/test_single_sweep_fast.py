import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import backtest as bt
import indicator_sweep as sw
import stats
from api import builder_strategy as builder
from api import confirm_sweep as cs
from api import fast_sweep as fs
from api.builder_strategy import _run_confirm_embedded_buy_threshold, prepare_builder_refine_frame
from api.schemas import BuilderCondition, SavedStrategy
from api.strategy_store import get_strategy_by_id
from tests.test_fast_sweep import _frame


class HoldFlagsTests(unittest.TestCase):
    def test_matches_the_engine_hold_state(self):
        for seed, is_long, hob in ((0, True, False), (1, False, False), (2, True, True), (3, False, True)):
            with self.subTest(seed=seed, is_long=is_long, hob=hob):
                frame = _frame(seed)
                original = bt.HoldOnBuySignal
                bt.HoldOnBuySignal = hob
                try:
                    executed = bt.execute_strategy(frame.copy(), 4, 2, is_long)
                    flags = fs.hold_flags(
                        frame["Buy"].to_numpy(), frame["Sell"].to_numpy(), frame["Close"].to_numpy(), 4, 2, is_long
                    )
                finally:
                    bt.HoldOnBuySignal = original
                np.testing.assert_array_equal(flags, executed["HoldLong"].to_numpy())


class BestCandidatesTests(unittest.TestCase):
    def test_order_ties_nan_and_min_trades(self):
        sharpe = np.array([0.5, np.nan, 1.2, 1.2, 0.9, 2.0])
        trades = np.array([50, 80, 40, 60, 5, 9])
        self.assertEqual(fs.best_candidates(sharpe, trades, limit=3), [5, 2, 3])  # ties keep input order
        self.assertEqual(fs.best_candidates(sharpe, trades, limit=3, min_trades=10), [2, 3, 0])
        self.assertEqual(fs.best_candidates(sharpe, trades, limit=10, min_trades=1000), [])
        self.assertEqual(fs.best_candidates(np.array([np.nan]), np.array([5])), [])


class ScorerTests(unittest.TestCase):
    def test_matches_the_engine_with_and_without_a_trimmed_start(self):
        frame = _frame(5)
        scorer = fs.Scorer(frame, 4, 2, True)
        sharpe, trades = scorer.score(scorer.base_buy)
        exact = stats.compute_aggregate_metrics(bt.execute_strategy(frame.copy(), 4, 2, True))
        self.assertAlmostEqual(sharpe, exact["sharpe"], places=9)
        self.assertEqual(trades, exact["trades"])
        trimmed = frame.copy()
        trimmed["EntryReady"] = np.arange(len(frame)) >= 300
        scorer = fs.Scorer(trimmed, 4, 2, True)
        sharpe, trades = scorer.score(scorer.base_buy, first=300)
        exact = stats.compute_aggregate_metrics(bt.execute_strategy(trimmed.copy(), 4, 2, True))
        self.assertAlmostEqual(sharpe, exact["sharpe"], places=9)

    def test_custom_exits(self):
        frame = _frame(6)
        scorer = fs.Scorer(frame, 4, 2, True)
        sell = frame["Sell"].to_numpy() | (frame["A"].to_numpy() > 0.8)
        sharpe, _ = scorer.score(scorer.base_buy, sell)
        other = frame.copy()
        other["Sell"] = sell
        exact = stats.compute_aggregate_metrics(bt.execute_strategy(other, 4, 2, True))["sharpe"]
        self.assertAlmostEqual(sharpe, exact, places=9)


SPECS = [("A", "A", "both", 0.1, 0.9, 0.1), ("B", "B", "both", 0.2, 0.8, 0.2), ("C", "C", "both", 0.3, 0.7, 0.1)]


def _records(frame):
    return {(r["Indicator"], r["Condition"], round(float(r["Value"]), 8)): r for r in frame.to_dict(orient="records")}


class CollectSweepsTests(unittest.TestCase):
    def _both(self, frame, *, is_sell, min_trades=0):
        fast = pd.DataFrame(
            sw._collect_sweeps([], frame, 4, 2, True, is_sell, False, SPECS, verbose=False, min_trades=min_trades)
        )
        with patch.object(fs, "NUMBA_AVAILABLE", False):
            slow = pd.DataFrame(sw._collect_sweeps([], frame, 4, 2, True, is_sell, False, SPECS, verbose=False))
        return fast, slow

    def test_buy_and_sell_sweeps_return_the_slow_path_rows(self):
        for is_sell in (False, True):
            with self.subTest(is_sell=is_sell):
                fast, slow = self._both(_frame(7), is_sell=is_sell)
                a, b = _records(fast), _records(slow)
                self.assertEqual(set(a), set(b))
                for key in a:
                    for column in ("Trades", "Sharpe", "Sortino", "%Pstv", "CAGR", "MaxDD", "PnL"):
                        self.assertEqual(str(a[key][column]), str(b[key][column]), (key, column))
                    self.assertEqual(a[key]["Yearly"], b[key]["Yearly"])

    def test_three_rows_per_spec_and_yearly_attached(self):
        fast, _ = self._both(_frame(8), is_sell=False)
        self.assertEqual(fast.groupby("Indicator").size().max(), 3)
        self.assertTrue(all(isinstance(y, list) and y for y in fast["Yearly"]))

    def test_min_trades_keeps_thin_thresholds_out_of_the_top_places(self):
        frame = _frame(9)
        thin, _ = self._both(frame, is_sell=False, min_trades=0)
        eligible, _ = self._both(frame, is_sell=False, min_trades=120)
        self.assertTrue((eligible["Trades"] >= 120).all())
        self.assertTrue(len(eligible) >= len(thin[thin["Trades"] >= 120]))

    def test_zero_day_strategies_keep_the_old_path(self):
        frame = _frame(10)
        with patch.object(sw, "_collect_sweeps_fast") as fast, patch.object(sw, "_run_one_sweep", side_effect=RuntimeError("slow path")):
            with self.assertRaises(RuntimeError):
                sw._collect_sweeps([], frame, 0, 0, True, False, True, SPECS, verbose=False)
        fast.assert_not_called()


def _symbol_frame(seed: int, bars: int = 1500, nan_prefix: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, bars))
    frame = pd.DataFrame({"Date": pd.bdate_range("2016-01-04", periods=bars), "Close": close})
    frame["High"], frame["Low"], frame["Open"] = close * 1.005, close * 0.995, close
    frame["%Change"] = frame["Close"].pct_change().fillna(0) * 3
    frame["RSI2"] = rng.uniform(0, 100, bars)
    frame["IBR"] = rng.uniform(0, 1, bars)
    frame["CCI"] = rng.normal(0, 100, bars)
    frame["Stoch"] = rng.uniform(0, 100, bars)
    frame["ER"] = rng.uniform(0, 1, bars)
    frame.loc[frame.index[:nan_prefix], "ER"] = np.nan
    return frame


def _confirm_strategy(**overrides) -> SavedStrategy:
    payload = dict(
        id="s", name="n", symbol="AAA", direction="long", hold_days=3, profit=2, description="",
        conditions=[BuilderCondition(left="RSI2", operator="<=", right="45", logic="AND")],
        confirm_symbols=["BBB", "CCC"], hold_on_buy_signal=False, created_at="", updated_at="",
    )
    payload.update(overrides)
    return SavedStrategy(**payload)


class ConfirmContextTests(unittest.TestCase):
    def setUp(self):
        self.symbol_data = {"AAA": _symbol_frame(1), "BBB": _symbol_frame(2), "CCC": _symbol_frame(3, nan_prefix=150)}
        self._hob = bt.HoldOnBuySignal

    def tearDown(self):
        bt.HoldOnBuySignal = self._hob

    def _context(self, strategy):
        bt.HoldOnBuySignal = strategy.hold_on_buy_signal
        frame, days, profit, is_long, pnl = prepare_builder_refine_frame(
            strategy, years=25, symbol_data=self.symbol_data, strategy_resolver=get_strategy_by_id, labels={}
        )
        return cs.build_confirm_context(
            strategy, frame, days, profit, is_long, pnl, self.symbol_data, strategy_resolver=get_strategy_by_id, labels={}
        ), frame

    def _slow(self, strategy, column, condition, value):
        bt.HoldOnBuySignal = strategy.hold_on_buy_signal
        return _run_confirm_embedded_buy_threshold(
            strategy, years=25, symbol_data=self.symbol_data, column_name=column, condition=condition,
            value=value, include_yearly=True, strategy_resolver=get_strategy_by_id, labels={},
        )

    def test_composed_rows_equal_the_slow_path_rows(self):
        for hold_on_buy in (False, True):
            strategy = _confirm_strategy(hold_on_buy_signal=hold_on_buy)
            context, _ = self._context(strategy)
            self.assertIsNotNone(context)
            for column, condition, value in (
                ("IBR", "less", 0.5), ("IBR", "more", 0.3), ("CCI", "less", 20.0), ("Stoch", "more", 40.0),
                ("ER", "more", 0.4),  # a column that is missing on one confirm symbol's first 150 bars
                ("IBR", "less", 0.30000000000000004),  # grid noise: the slow path rounds to 4 decimals
            ):
                with self.subTest(hold_on_buy=hold_on_buy, column=column, condition=condition, value=value):
                    fast, slow = context.exact_row(column, condition, value), self._slow(strategy, column, condition, value)
                    self.assertEqual(set(fast), set(slow))
                    for key in slow:
                        if key in ("Yearly", "Value"):
                            self.assertEqual(fast[key], slow[key], key)
                        elif isinstance(slow[key], str):
                            self.assertEqual(fast[key], slow[key], key)
                        else:
                            self.assertTrue(np.isclose(float(fast[key]), float(slow[key]), rtol=1e-9, atol=1e-9), key)

    def test_screen_picks_the_same_best_as_exact_scoring(self):
        strategy = _confirm_strategy()
        context, _ = self._context(strategy)
        candidates = [(cond, v) for v in (0.2, 0.4, 0.6, 0.8) for cond in ("more", "less")]
        picked = context.top_candidates("IBR", candidates, limit=3, min_trades=10)
        exact = [(context.exact_row("IBR", c, v, include_yearly=False), i) for i, (c, v) in enumerate(candidates)]
        exact = [(r, i) for r, i in exact if r["Trades"] >= 10 and np.isfinite(r["Sharpe"])]
        expected = [i for _, i in sorted(exact, key=lambda item: -item[0]["Sharpe"])][:3]
        self.assertEqual(picked, expected)

    def test_falls_back_when_it_cannot_be_sure(self):
        strategy = _confirm_strategy()
        context, frame = self._context(strategy)
        self.assertIsNotNone(context)
        bt.HoldOnBuySignal = False
        tampered = frame.copy()
        tampered["Buy"] = ~tampered["Buy"]
        self.assertIsNone(cs.build_confirm_context(strategy, tampered, 3, 2, True, None, self.symbol_data, strategy_resolver=get_strategy_by_id, labels={}))
        with patch.object(fs, "NUMBA_AVAILABLE", False):
            self.assertIsNone(cs.build_confirm_context(strategy, frame, 3, 2, True, None, self.symbol_data, strategy_resolver=get_strategy_by_id, labels={}))
        self.assertIsNone(cs.build_confirm_context(strategy, frame, 0, 0, True, None, self.symbol_data, strategy_resolver=get_strategy_by_id, labels={}))
        self.assertIsNone(cs.build_confirm_context(_confirm_strategy(confirm_symbols=[]), frame, 3, 2, True, None, self.symbol_data, strategy_resolver=get_strategy_by_id, labels={}))
        duplicated = dict(self.symbol_data)
        duplicated["BBB"] = pd.concat([duplicated["BBB"], duplicated["BBB"].tail(5)], ignore_index=True)
        self.assertIsNone(cs.build_confirm_context(strategy, frame, 3, 2, True, None, duplicated, strategy_resolver=get_strategy_by_id, labels={}))

    def test_builder_tryout_gives_the_same_rows_fast_and_slow(self):
        strategy = _confirm_strategy()
        bt.HoldOnBuySignal = False
        specs = [("IBR", "IBR", "both", 0.2, 0.8, 0.2), ("CCI", "CCI", "both", -100, 100, 50), ("Stoch", "Stoch", "both", 20, 80, 20), ("ER", "ER", "both", 0.2, 0.8, 0.2)]
        frame, days, profit, is_long, pnl = prepare_builder_refine_frame(
            strategy, years=25, symbol_data=self.symbol_data, strategy_resolver=get_strategy_by_id, labels={}
        )

        def run():
            with patch("indicator_sweep.PRICE_SWEEPS", specs), patch("indicator_sweep.BREADTH_SWEEPS", []), \
                    patch("indicator_sweep.POST_2003_BREADTH_SWEEPS", []), patch("indicator_sweep.VFI_SWEEPS", []), \
                    patch("indicator_sweep.VWAP_SWEEPS", []), patch.object(bt, "load_symbol_dataset", return_value=self.symbol_data):
                return builder.builder_indicator_tryout(
                    strategy, frame, days, profit, is_long, check_breadth=False, check_both=True,
                    strategy_resolver=get_strategy_by_id, labels={}, min_trades=0,
                )

        fast = run()
        with patch.object(fs, "NUMBA_AVAILABLE", False):
            slow = run()
        a, b = _records(fast), _records(slow)
        self.assertEqual(set(a), set(b))
        for key in a:
            for column in ("Trades", "Sharpe", "CAGR", "MaxDD", "PnL"):
                self.assertEqual(str(a[key][column]), str(b[key][column]), (key, column))
            self.assertEqual(a[key]["Yearly"], b[key]["Yearly"])


if __name__ == "__main__":
    unittest.main()
