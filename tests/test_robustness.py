import unittest

import numpy as np
import pandas as pd

import backtest as bt
import config
import stats
from api import fast_sweep as fs
from api.builder_strategy import draft_to_saved_strategy, prepare_builder_refine_frame
from api.robustness import _first_ready, _Run, robustness_report
from api.schemas import BuilderBacktestRequest, BuilderCondition

LABELS = {"RSI2": "RSI(2)", "IBR": "IBR", "ER": "Peek"}


def _data(rows: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, rows)))
    opens = close * (1 + rng.normal(0, 0.003, rows))
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2012-01-02", periods=rows, freq="B"),
            "Open": opens,
            "High": np.maximum(opens, close) * 1.004,
            "Low": np.minimum(opens, close) * 0.996,
            "Close": close,
            "RSI2": rng.uniform(0, 100, rows),
            "IBR": rng.uniform(0, 1, rows),
        }
    )
    # The ER column is replaced by a signal that looks one bar ahead: high when tomorrow closes up. A planted edge random timing cannot match.
    up_next = np.r_[frame["Close"].to_numpy()[1:] > frame["Close"].to_numpy()[:-1], False]
    frame["ER"] = np.where(up_next, rng.uniform(0.6, 1, rows), rng.uniform(0, 0.4, rows))
    frame["%Change"] = frame["Close"].pct_change().fillna(0) * config.Leverage
    return frame


def _strategy(*conditions, sell=(), hold_days=3, profit=1):
    request = BuilderBacktestRequest(
        symbol="SPY",
        hold_days=hold_days,
        profit=profit,
        conditions=[BuilderCondition(left=l, operator=o, right=r) for l, o, r in conditions],
        sell_conditions=[BuilderCondition(left=l, operator=o, right=r) for l, o, r in sell],
    )
    return draft_to_saved_strategy(request, labels={})


def _report(strategy, data):
    return robustness_report(strategy, data, years=25, labels=LABELS, strategy_resolver=None, shifts=300)


class SimulatorParityTests(unittest.TestCase):
    def test_close_fills_without_cost_match_the_sweep_loop(self):
        data = _data()
        strategy = _strategy(("RSI2", "<=", "30"), sell=(("IBR", ">=", "0.7"),))
        frame, days, profit, is_long, pnl_col = prepare_builder_refine_frame(strategy, data=data)
        run = _Run(frame, days, profit, is_long, pnl_col, 0)
        pnl, trades, _, _ = run.simulate()
        i = run.inputs
        expected = np.empty(len(frame))
        expected_trades = fs._simulate(
            i.base, i.sell, i.close, i.track, i.use_stream, i.v_ok, i.v_use, i.v_ret, i.s_ret,
            i.days, i.prof, i.is_long, i.hold_on_buy, expected,
        )
        np.testing.assert_allclose(pnl, expected)
        self.assertEqual(trades, expected_trades)

    def test_baseline_matches_the_backtest_engine(self):
        data = _data()
        strategy = _strategy(("RSI2", "<=", "30"), ("IBR", "<=", "0.5"))
        frame, days, profit, is_long, pnl_col = prepare_builder_refine_frame(strategy, data=data)
        exact = stats.compute_aggregate_metrics(bt.execute_strategy(frame.copy(), days, profit, is_long, pnl_column=pnl_col))
        base = _Run(frame, days, profit, is_long, pnl_col, _first_ready(frame)).score()
        self.assertAlmostEqual(base["sharpe"], exact["sharpe"], places=6)
        self.assertEqual(base["trades"], exact["trades"])

    def test_costs_and_next_open_fills_only_lower_a_long_edge(self):
        data = _data()
        strategy = _strategy(("ER", ">=", "0.6"))
        frame, days, profit, is_long, pnl_col = prepare_builder_refine_frame(strategy, data=data)
        run = _Run(frame, days, profit, is_long, pnl_col, 0)
        close = run.score()["sharpe"]
        costly = run.score(cost_leg=0.002)["sharpe"]
        self.assertLess(costly, close)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.data = _data()

    def test_a_planted_edge_beats_random_timing(self):
        report = _report(_strategy(("ER", ">=", "0.6"), ("RSI2", "<=", "90")), self.data)
        timing = report["checks"][0]
        self.assertEqual(timing["id"], "random-timing")
        self.assertEqual(timing["status"], "pass")
        self.assertLess(report["random_timing"][0]["p"], 0.01)

    def test_a_noise_strategy_does_not(self):
        report = _report(_strategy(("RSI2", "<=", "30"), ("IBR", "<=", "0.5")), self.data)
        self.assertNotEqual(report["checks"][0]["status"], "pass")
        self.assertNotEqual(report["verdict"], "robust")

    def test_an_idle_condition_is_flagged(self):
        # RSI2 <= 99 drops almost nothing: it cannot be doing the work.
        report = _report(_strategy(("ER", ">=", "0.6"), ("RSI2", "<=", "99")), self.data)
        rows = {r["condition"]: r for r in report["ablation"]}
        self.assertFalse(rows["RSI(2) <= 99"]["earns_place"])
        self.assertTrue(rows["Peek >= 0.6"]["earns_place"])

    def test_every_threshold_is_nudged_both_ways(self):
        report = _report(_strategy(("ER", ">=", "0.6"), sell=(("IBR", ">=", "0.7"),)), self.data)
        labels = {r["label"] for r in report["neighbourhood"]["rows"]}
        self.assertTrue({"ER >= 0.45", "ER >= 0.54", "ER >= 0.66", "ER >= 0.75"} <= labels)
        self.assertTrue({"IBR >= 0.525", "IBR >= 0.77"} <= labels)
        self.assertIn("hold 1", labels)
        # Exits are shifted with entries in a second null when the strategy has exit rules.
        self.assertEqual([n["shift_exits"] for n in report["random_timing"]], [False, True])

    def test_stability_covers_every_year_and_trade(self):
        report = _report(_strategy(("ER", ">=", "0.6")), self.data)
        stab = report["stability"]
        self.assertEqual([y["year"] for y in stab["years"]], sorted(self.data["Date"].dt.year.unique().tolist()))
        self.assertEqual(sum(y["trades"] for y in stab["years"]), report["baseline"]["trades"])
        self.assertEqual(stab["concentration"]["trades"], report["baseline"]["trades"])

    def test_a_strategy_that_never_trades_is_rejected(self):
        with self.assertRaises(ValueError):
            _report(_strategy(("RSI2", ">", "1000")), self.data)


if __name__ == "__main__":
    unittest.main()
