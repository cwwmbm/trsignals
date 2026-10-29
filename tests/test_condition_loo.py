import unittest

import numpy as np
import pandas as pd

import config
from api import fast_metrics
from api.builder_strategy import draft_to_saved_strategy, prepare_builder_refine_frame
from api.condition_loo import DRAFT_LABEL, _verdict, leave_one_out_rows
from api.schemas import BuilderBacktestRequest, BuilderCondition, BuilderRefineRequest


def _data(rows: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, rows)))
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2015-01-01", periods=rows, freq="B"),
            "Open": close,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "RSI2": rng.uniform(0, 100, rows),
            "IBR": rng.uniform(0, 1, rows),
        }
    )
    frame["%Change"] = frame["Close"].pct_change().fillna(0) * config.Leverage
    frame["SMA200"] = frame["Close"].rolling(200).mean()
    return frame


def _primary(*conditions):
    request = BuilderBacktestRequest(
        symbol="SPY",
        hold_days=2,
        profit=1,
        conditions=[BuilderCondition(left=l, operator=o, right=r) for l, o, r in conditions],
    )
    return draft_to_saved_strategy(request, labels={})


LABELS = {"RSI2": "RSI(2)", "IBR": "IBR", "Close": "Close", "SMA200": "SMA(200)"}
THREE = (("RSI2", "<=", "30"), ("IBR", "<=", "0.5"), ("Close", ">", "SMA200"))


class LeaveOneOutTests(unittest.TestCase):
    def setUp(self):
        self.data = _data()
        self.cut = pd.Timestamp(self.data["Date"].iloc[600])

    def rows(self, *conditions):
        return leave_one_out_rows(
            _primary(*conditions), self.data, years=25, labels=LABELS, in_sample_end=self.cut
        )

    def test_the_draft_comes_first_then_one_row_per_condition(self):
        rows = self.rows(*THREE)
        self.assertEqual([row["#"] for row in rows], [0, 1, 2, 3])
        self.assertEqual(rows[0]["Removed"], DRAFT_LABEL)
        self.assertEqual(rows[1]["Removed"], "RSI(2) <= 30")
        self.assertEqual(rows[3]["Removed"], "Close > SMA(200)")
        # Every row has the same columns, in the same order, so the table lines up.
        self.assertEqual({tuple(row) for row in rows}, {tuple(rows[0])})

    def test_the_draft_row_is_scored_like_the_sweeps(self):
        primary = _primary(*THREE)
        frame, days, profit, is_long, pnl = prepare_builder_refine_frame(primary, data=self.data)
        dates = pd.to_datetime(frame["Date"])
        expected = fast_metrics.run_metrics(
            frame.loc[dates <= self.cut].copy(), days, profit, is_long, pnl_column=pnl
        )
        draft = self.rows(*THREE)[0]
        self.assertEqual(draft["Sharpe"], round(expected["Sharpe"], 2))
        self.assertEqual(draft["Trades"], expected["Trades"])
        self.assertIsNone(draft["ΔSharpe"])

    def test_the_change_columns_are_against_the_draft(self):
        rows = self.rows(*THREE)
        for prefix in ("", "OOS "):
            for row in rows[1:]:
                self.assertAlmostEqual(
                    row[f"{prefix}ΔSharpe"], row[f"{prefix}Sharpe"] - rows[0][f"{prefix}Sharpe"], delta=0.011
                )

    def test_blocks_is_the_share_of_the_other_conditions_entries_it_removes(self):
        rows = self.rows(*THREE)
        draft = _primary(*THREE)
        without_rsi = _primary(*THREE[1:])
        count = lambda s: int(prepare_builder_refine_frame(s, data=self.data)[0]["Buy"].sum())
        # The RSI and IBR filters have no warm-up, so neither variant is clipped at the start.
        expected = round((1 - count(draft) / count(without_rsi)) * 100)
        self.assertEqual(rows[1]["Blocks"], f"{expected}%")

    def test_a_removed_warm_up_does_not_add_early_trades(self):
        # Without Close > SMA200 the variant could trade in the first 200 bars; it must not.
        from api.condition_loo import _start_at

        frame, days, profit, is_long, pnl = prepare_builder_refine_frame(_primary(*THREE[:2]), data=self.data)
        self.assertTrue(frame["Buy"].iloc[:199].any())
        clipped = _start_at(frame, pd.Timestamp(self.data["Date"].iloc[199]))
        self.assertFalse(clipped["Buy"].iloc[:199].any())
        in_sample = clipped.loc[pd.to_datetime(clipped["Date"]) <= self.cut].copy()
        expected = fast_metrics.run_metrics(in_sample, days, profit, is_long, pnl_column=pnl)
        self.assertEqual(self.rows(*THREE)[3]["Trades"], expected["Trades"])

    def test_rows_say_which_condition_they_removed(self):
        rows = self.rows(*THREE)
        self.assertIsNone(rows[0]["RemovedCondition"])
        self.assertEqual(
            rows[2]["RemovedCondition"], {"index": 1, "left": "IBR", "operator": "<=", "right": "0.5"}
        )

    def test_a_single_condition_gives_an_error_row_not_a_crash(self):
        rows = self.rows(("RSI2", "<=", "30"))
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0]["Error"])
        self.assertIn("only entry condition", rows[1]["Error"])
        # Nothing to offer removing: the draft would have no entry condition left.
        self.assertIsNone(rows[1]["RemovedCondition"])

    def test_the_draft_row_has_no_verdict(self):
        rows = self.rows(*THREE)
        self.assertIsNone(rows[0]["Verdict"])
        self.assertIn(rows[1]["Verdict"], {"Helps", "Hurts", "Mixed", "No effect"})


class VerdictTests(unittest.TestCase):
    def test_reads_the_sharpe_change_in_both_halves(self):
        self.assertEqual(_verdict(-0.3, -0.2, 0.05), "Helps")
        self.assertEqual(_verdict(0.3, 0.2, 0.05), "Hurts")
        self.assertEqual(_verdict(-0.3, 0.2, 0.05), "Mixed")
        self.assertEqual(_verdict(0.01, -0.02, 0.05), "No effect")
        self.assertEqual(_verdict(-0.3, 0.01, 0.05), "Mixed")
        self.assertEqual(_verdict(None, 0.2, 0.05), "-")


class WiringTests(unittest.TestCase):
    def test_refine_accepts_the_mode(self):
        request = BuilderRefineRequest(
            mode="entry-loo",
            strategy=BuilderBacktestRequest(conditions=[BuilderCondition(left="RSI2", operator="<=", right="30")]),
        )
        self.assertEqual(request.mode, "entry-loo")


if __name__ == "__main__":
    unittest.main()
