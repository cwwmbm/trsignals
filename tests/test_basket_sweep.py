import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import backtest as bt
import config
import stats
from api import basket_sweep as bs
from api import fast_sweep as fs
from api.schemas import (
    BuilderBacktestRequest,
    BuilderCondition,
    BuilderRefineOutcomeRequest,
    BuilderRefineRequest,
    SavedStrategy,
)
from api.services import run_builder_refine, run_builder_refine_outcome
from tests.test_fast_sweep import _frame


def _strategy(**overrides) -> SavedStrategy:
    payload = dict(
        id="s1", name="Draft", symbol="AAPL", direction="long", hold_days=3, profit=2, description="",
        conditions=[BuilderCondition(left="RSI2", operator="<=", right="15", logic="AND")],
        sell_conditions=[], confirm_symbols=[], proxy_symbol=None, hold_on_buy_signal=False,
        created_at="", updated_at="",
    )
    payload.update(overrides)
    return SavedStrategy(**payload)


def _planted(seed: int) -> pd.DataFrame:
    """Entries that always precede an up day: a strategy with a real (look-ahead) edge."""
    frame = _frame(seed)
    up_next = (frame["Close"].shift(-1) > frame["Close"]).fillna(False).to_numpy()
    rng = np.random.default_rng(seed)
    frame["Buy"] = up_next & (rng.uniform(size=len(frame)) < 0.3)
    return frame


def _random(seed: int) -> pd.DataFrame:
    return _frame(seed)


class NormalizeTests(unittest.TestCase):
    def test_cleans_dedupes_and_drops_the_draft_symbol(self):
        got = bs.normalize_basket(["msft, nvda", "NVDA", " aapl ", "", "goog  amzn"], "AAPL")
        self.assertEqual(got, ["MSFT", "NVDA", "GOOG", "AMZN"])

    def test_cap(self):
        with self.assertRaises(ValueError):
            bs.normalize_basket([f"S{i}" for i in range(bs.MAX_BASKET_SYMBOLS + 1)], "AAPL")

    def test_empty_is_an_error(self):
        with self.assertRaises(ValueError):
            bs.run_basket_sweep(_strategy(), [], years=25)
        with self.assertRaises(ValueError):
            bs.run_basket_sweep(_strategy(), ["aapl"], years=25)


class RetargetTests(unittest.TestCase):
    def test_symbol_confirm_and_proxy_are_adapted(self):
        base = _strategy(confirm_symbols=["SMH", "QQQ"], proxy_symbol="SOXX")
        for_smh = bs.strategy_for_symbol(base, "SMH")
        self.assertEqual(for_smh.symbol, "SMH")
        self.assertEqual(for_smh.confirm_symbols, ["QQQ"])  # never confirms itself
        self.assertIsNone(for_smh.proxy_symbol)  # the proxy named the draft's traded instrument
        for_msft = bs.strategy_for_symbol(base, "MSFT")
        self.assertEqual(for_msft.confirm_symbols, ["SMH", "QQQ"])
        self.assertEqual(base.symbol, "AAPL")  # the draft itself is untouched
        self.assertEqual(base.proxy_symbol, "SOXX")


class ShiftNullTests(unittest.TestCase):
    def test_actual_sharpe_matches_the_engine_with_entry_ready_and_hold_on_buy(self):
        frame = _planted(1)
        frame["EntryReady"] = np.arange(len(frame)) >= 150
        for hold_on_buy in (False, True):
            original = bt.HoldOnBuySignal
            bt.HoldOnBuySignal = hold_on_buy
            try:
                result = fs.shift_null(frame, 4, 2, True, shifts=20)
                exact = stats.compute_aggregate_metrics(bt.execute_strategy(frame.copy(), 4, 2, True))["sharpe"]
            finally:
                bt.HoldOnBuySignal = original
            self.assertAlmostEqual(result["sharpe"], exact, places=9)

    def test_a_planted_edge_is_separated_from_random_timing(self):
        planted = fs.shift_null(_planted(2), 3, 1, True, shifts=200)
        self.assertLess(planted["p"], 0.02)
        self.assertGreater(planted["sharpe"], planted["null_p95"])
        # no edge: p-values spread over (0, 1) instead of piling up near zero
        ps = [fs.shift_null(_random(seed), 3, 1, True, shifts=200)["p"] for seed in range(10, 22)]
        self.assertGreater(np.median(ps), 0.15)
        self.assertLess(sum(p < 0.05 for p in ps), 4)

    def test_p_is_never_zero_and_counts_the_real_run(self):
        result = fs.shift_null(_planted(3), 3, 1, True, shifts=50)
        self.assertGreaterEqual(result["p"], 1 / 51)
        self.assertEqual(result["shifts"], 50)

    def test_too_short_or_never_trading_returns_none(self):
        self.assertIsNone(fs.shift_null(_frame(4, bars=100), 3, 1, True))
        never = _frame(5)
        never["Buy"] = False
        self.assertIsNone(fs.shift_null(never, 3, 1, True))

    def test_deterministic_for_a_seed(self):
        a = fs.shift_null(_planted(6), 3, 1, True, shifts=40, seed=9)
        b = fs.shift_null(_planted(6), 3, 1, True, shifts=40, seed=9)
        self.assertEqual(a, b)


class RunBasketSweepTests(unittest.TestCase):
    def setUp(self):
        frames = {"AAPL": _random(30), "MSFT": _planted(31), "NVDA": _random(32), "AMD": _random(33)}
        self.requested = []

        def fake_load(variant, years, in_sample_end=None):
            self.requested.append(variant)
            if variant.symbol == "BAD":
                raise ValueError("Could not load price data for BAD")
            if variant.symbol == "TINY":
                return _frame(40, bars=120), 3, 2, True, None
            return frames[variant.symbol], 3, 2, True, None

        patcher = patch.object(bs, "_load_frame", side_effect=fake_load)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_draft_first_basket_ranked_failures_last(self):
        rows, summary = bs.run_basket_sweep(_strategy(), ["MSFT", "NVDA", "BAD", "TINY", "AMD"])
        symbols = [r["Symbol"] for r in rows]
        self.assertEqual(symbols[0], "AAPL")
        self.assertEqual(rows[0]["Source"], "Draft symbol")
        ok = [r for r in rows if "Error" not in r]
        basket = [r for r in ok if r["Source"] == "Basket"]
        sharpes = [r["Sharpe"] for r in basket]
        self.assertEqual(sharpes, sorted(sharpes, reverse=True))
        self.assertEqual(symbols[-2:], ["BAD", "TINY"])
        errors = {r["Symbol"]: r["Error"] for r in rows if "Error" in r}
        self.assertIn("BAD", errors)
        self.assertIn("120 bars", errors["TINY"])
        self.assertEqual([f["symbol"] for f in summary["failed"]], ["BAD", "TINY"])

    def test_row_fields_and_types(self):
        rows, _ = bs.run_basket_sweep(_strategy(), ["MSFT"])
        row = rows[1]
        self.assertIsInstance(row["Trades"], int)
        for key in ("Sharpe", "NullSharpe", "NullP", "CAGR", "MaxDD", "PnL", "Sortino", "From", "Yearly"):
            self.assertIn(key, row)
        self.assertTrue(row["Yearly"])
        self.assertTrue(0 < row["NullP"] <= 1)

    def test_planted_symbol_beats_its_null_and_summary_counts_only_the_basket(self):
        rows, summary = bs.run_basket_sweep(_strategy(), ["MSFT", "NVDA", "AMD"])
        by = {r["Symbol"]: r for r in rows}
        self.assertLess(by["MSFT"]["NullP"], 0.02)
        self.assertEqual(summary["requested"], 3)
        self.assertEqual(summary["evaluated"], 3)
        self.assertEqual(summary["null_evaluated"], 3)
        self.assertGreaterEqual(summary["share_beating_null"], 1 / 3 - 1e-9)
        self.assertEqual(summary["draft_symbol"], "AAPL")
        self.assertAlmostEqual(summary["draft_sharpe"], by["AAPL"]["Sharpe"], places=1)
        basket_sharpes = [by[s]["Sharpe"] for s in ("MSFT", "NVDA", "AMD")]
        self.assertAlmostEqual(summary["median_sharpe"], float(np.median(basket_sharpes)), places=1)

    def test_each_symbol_gets_an_adapted_strategy(self):
        strategy = _strategy(confirm_symbols=["MSFT", "QQQ"], proxy_symbol="SOXX")
        bs.run_basket_sweep(strategy, ["MSFT", "NVDA"])
        seen = {v.symbol: v for v in self.requested}
        self.assertEqual(seen["MSFT"].confirm_symbols, ["QQQ"])
        self.assertEqual(seen["NVDA"].confirm_symbols, ["MSFT", "QQQ"])
        self.assertTrue(all(v.proxy_symbol is None for v in self.requested))
        self.assertEqual(seen["AAPL"].confirm_symbols, ["MSFT", "QQQ"])

    def test_without_numba_the_null_columns_are_empty(self):
        with patch.object(fs, "NUMBA_AVAILABLE", False):
            rows, summary = bs.run_basket_sweep(_strategy(), ["MSFT"])
        self.assertIsNone(rows[1]["NullP"])
        self.assertIsNone(rows[1]["NullSharpe"])
        self.assertIsNone(summary["share_beating_null"])


def _draft_request(**overrides) -> BuilderBacktestRequest:
    payload = dict(
        symbol="AAPL", years=25, direction="long", hold_days=3, profit=2, name="Draft",
        conditions=[BuilderCondition(left="RSI2", operator="<=", right="15", logic="AND")],
    )
    payload.update(overrides)
    return BuilderBacktestRequest(**payload)


class RefineModeTests(unittest.TestCase):
    def _reference(self):
        return _frame(50).assign(RSI2=np.linspace(10, 30, 1800))

    def test_mode_returns_rows_and_full_period_meta(self):
        request = BuilderRefineRequest(mode="basket-sweep", strategy=_draft_request(), basket_symbols=["MSFT", "NVDA"])
        summary = {"requested": 2, "evaluated": 2}
        rows = [{"Symbol": "AAPL", "Source": "Draft symbol"}, {"Symbol": "MSFT", "Source": "Basket"}]
        with patch("api.services.load_ticker_data", return_value=self._reference()):
            with patch("api.services.run_basket_sweep", return_value=(rows, summary)) as run:
                payload = run_builder_refine(request)
        self.assertEqual(payload["rows"], rows)
        self.assertEqual(payload["meta"]["sample"], "full")
        self.assertEqual(payload["meta"]["in_sample_fraction"], 1.0)
        self.assertEqual(payload["meta"]["basket"], summary)
        self.assertEqual(run.call_args.args[1], ["MSFT", "NVDA"])
        self.assertEqual(run.call_args.args[0].symbol, "AAPL")

    def test_custom_intraday_data_is_refused(self):
        request = BuilderRefineRequest(
            mode="basket-sweep", strategy=_draft_request(custom_dataset_id="d1"), basket_symbols=["MSFT"]
        )
        custom = SimpleNamespace(unavailable_indicator_ids=[], has_vwap=False)
        with patch("api.custom_data.custom_dataset_store") as store:
            store.require.return_value = custom
            store.load_backtest_frame.return_value = self._reference()
            with self.assertRaises(ValueError):
                run_builder_refine(request)

    def test_outcome_replays_one_symbol(self):
        request = BuilderRefineOutcomeRequest(
            mode="basket-sweep", strategy=_draft_request(), row={"Symbol": "MSFT"}, sample="full"
        )
        detail = {"PnL": 1e6, "MaxDD": 30.0, "Trades": 50, "%Pstv": 60.0, "CAGR": "12%", "Sharpe": 0.9, "Sortino": 1.1, "Yearly": [{"year": 2020}]}
        with patch("api.services.load_ticker_data", return_value=self._reference()):
            with patch("api.services.basket_symbol_metrics", return_value=detail) as metrics:
                outcome = run_builder_refine_outcome(request)
        self.assertEqual(metrics.call_args.args[1], "MSFT")
        self.assertEqual(outcome["metrics"]["Trades"], 50)
        self.assertEqual(outcome["meta"]["sample"], "full")


if __name__ == "__main__":
    unittest.main()
