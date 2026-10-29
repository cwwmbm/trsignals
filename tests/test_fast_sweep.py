import itertools
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import backtest as bt
import config
from api import fast_sweep as fs
from api import vehicles
from api.fast_sweep import Filter
from api.schemas import BuilderBacktestRequest, BuilderCondition, BuilderRefineRequest
from api.services import run_builder_refine

BARS = 1800
INDICATORS = ("A", "B", "C", "D")


def _frame(seed: int = 0, columns=INDICATORS, bars: int = BARS) -> pd.DataFrame:
    """Random-walk price with sparse random entries and a handful of indicator columns."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, bars))
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2015-01-01", periods=bars, freq="B"),
            "Close": close,
            "PROXY": close * (1 + rng.normal(0, 0.002, bars)),
        }
    )
    frame["%Change"] = frame["Close"].pct_change().fillna(0) * config.Leverage
    for column in columns:
        frame[column] = rng.uniform(0, 1, bars)
    frame["Buy"] = rng.uniform(0, 1, bars) < 0.25
    frame["Sell"] = rng.uniform(0, 1, bars) < 0.10
    return frame


def _grid(thresholds=(0.2, 0.4, 0.6, 0.8), columns=INDICATORS) -> list[Filter]:
    return [
        Filter(column, condition, value)
        for column in columns
        for value in thresholds
        for condition in ("more", "less")
    ]


class ScreenMatchesEngineTests(unittest.TestCase):
    """The compiled screen must agree with the real engine on trades and Sharpe."""

    def _assert_parity(self, frame, *, is_long=True, hold_on_buy=False, pnl_column=None, days=4, profit=2):
        original = bt.HoldOnBuySignal
        bt.HoldOnBuySignal = hold_on_buy
        try:
            inputs = fs.build_sim_inputs(frame, days, profit, is_long, pnl_column=pnl_column)
            candidates = fs.build_candidates(frame, _grid(), inputs.base, 1)
            pairs = fs._all_pairs(candidates.compatible)
            rng = np.random.default_rng(3)
            sample = pairs[rng.choice(len(pairs), 25, replace=False)]
            sharpe, trades, _ = inputs.score(sample, candidates.masks, 1)
            checked = 0
            for row, combo in enumerate(sample):
                exact = fs.run_filters_exact(
                    frame, days, profit, is_long, [candidates.filters[i] for i in combo],
                    include_yearly=False, pnl_column=pnl_column,
                )
                self.assertEqual(int(trades[row]), int(exact["Trades"]))
                if exact["Sharpe"] is None or not np.isfinite(sharpe[row]):
                    continue
                self.assertAlmostEqual(float(sharpe[row]), float(exact["Sharpe"]), places=9)
                checked += 1
            self.assertGreater(checked, 10)
        finally:
            bt.HoldOnBuySignal = original

    def test_long(self):
        self._assert_parity(_frame())

    def test_short(self):
        self._assert_parity(_frame(1), is_long=False)

    def test_hold_on_buy(self):
        self._assert_parity(_frame(2), hold_on_buy=True)

    def test_proxy_pnl_column(self):
        self._assert_parity(_frame(3), pnl_column="PROXY")

    def test_single_day_hold_and_no_profit_target(self):
        self._assert_parity(_frame(4), days=1, profit=5)

    def test_leverage_vehicle_stream(self):
        frame = _frame(5)
        frame.attrs["symbol"] = "SPY"  # traded_stream only looks a stream up for a named symbol
        rng = np.random.default_rng(9)
        vehicle = rng.normal(0.001, 0.02, len(frame))
        vehicle[:300] = np.nan  # the vehicle does not exist yet
        stream = vehicles.TradedStream(
            symbol="SPY",
            vehicle_symbol="UPRO",
            leverage=3.0,
            synthetic=rng.normal(0.0008, 0.015, len(frame)),
            vehicle=vehicle,
            vehicle_ok=np.arange(len(frame)) >= 350,
            first_vehicle_date=frame["Date"].iloc[300],
        )
        with patch.object(vehicles, "stream_for", return_value=stream) as lookup:
            self._assert_parity(frame)
        self.assertTrue(lookup.called)
        inputs = None
        with patch.object(vehicles, "stream_for", return_value=stream):
            inputs = fs.build_sim_inputs(frame, 4, 2, True)
        self.assertTrue(inputs.use_stream)

    def test_without_best_year_exclusion(self):
        with patch.object(fs.stats, "ExcludeBestReturnYear", False):
            self._assert_parity(_frame(6))


class SingleFilterMatchesThresholdTests(unittest.TestCase):
    def test_one_filter_equals_run_indicator_threshold(self):
        frame = _frame(7)
        for condition, value in (("less", 0.3), ("more", 0.55)):
            exact = fs.run_filters_exact(
                frame, 4, 2, True, [Filter("A", condition, value)], include_yearly=True
            )
            reference = bt._run_indicator_threshold(
                frame, 4, 2, True, "A", "Buy", condition, value, include_yearly=True
            )
            for key in ("PnL", "MaxDD", "Trades", "%Pstv", "CAGR", "Sharpe", "Sortino", "Yearly"):
                self.assertEqual(exact[key], reference[key], key)


class CandidateTests(unittest.TestCase):
    def test_drops_no_op_duplicate_and_thin_filters(self):
        frame = _frame(8)
        frame["FLAG"] = 1.0
        frame["TWIN"] = frame["A"]
        inputs = fs.build_sim_inputs(frame, 4, 2, True)
        filters = [
            Filter("FLAG", "more", 0.0),  # keeps every entry: no-op
            Filter("A", "less", 0.5),
            Filter("TWIN", "less", 0.5),  # same entries as the line above
            Filter("A", "less", -1.0),  # keeps none
        ]
        candidates = fs.build_candidates(frame, filters, inputs.base, 5)
        self.assertEqual(candidates.filters, [Filter("A", "less", 0.5)])
        self.assertEqual(
            candidates.dropped, {"no_op": 1, "too_few_entries": 1, "duplicate": 1}
        )

    def test_same_column_only_combines_as_a_band(self):
        filters = [
            Filter("A", "more", 0.2),
            Filter("A", "less", 0.8),
            Filter("A", "less", 0.1),  # crossed band with more 0.2
            Filter("A", "more", 0.5),  # same direction as more 0.2
            Filter("B", "less", 0.5),
        ]
        compat = fs._compatibility(filters)
        self.assertTrue(compat[0, 1] and compat[1, 0])
        self.assertFalse(compat[0, 2])
        self.assertFalse(compat[0, 3])
        self.assertFalse(compat[1, 2])
        self.assertTrue(compat[0, 4] and compat[3, 4])
        self.assertFalse(compat.diagonal().any())

    def test_grid_values_are_clean_floats(self):
        frame = _frame(9, columns=("IBR",))
        grid = fs.threshold_grid(frame, check_breadth=False, check_both=False)
        values = {item.value for item in grid if item.column == "IBR"}
        self.assertIn(0.3, values)
        self.assertNotIn(0.30000000000000004, values)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.frame = _frame(10)
        self.inputs = fs.build_sim_inputs(self.frame, 4, 2, True)
        self.candidates = fs.build_candidates(self.frame, _grid(), self.inputs.base, 10)

    def test_pair_search_finds_the_brute_force_best(self):
        found = fs.search_combinations(
            self.inputs, self.candidates, size=2, top_n=5, min_trades=10, min_effect=0
        )
        best = None
        pairs = fs._all_pairs(self.candidates.compatible)
        for first, second in pairs:
            row = fs.run_filters_exact(
                self.frame, 4, 2, True,
                [self.candidates.filters[first], self.candidates.filters[second]],
                include_yearly=False,
            )
            if row["Trades"] < 10 or row["Sharpe"] is None:
                continue
            if best is None or row["Sharpe"] > best:
                best = row["Sharpe"]
        top = fs.run_filters_exact(
            self.frame, 4, 2, True, found.combos[0], include_yearly=False
        )
        self.assertEqual(found.tested, len(pairs))
        self.assertAlmostEqual(top["Sharpe"], best, places=9)

    def test_results_have_enough_trades_and_distinct_entries(self):
        found = fs.search_combinations(
            self.inputs, self.candidates, size=2, top_n=30, min_trades=25
        )
        self.assertTrue(found.combos)
        signatures = set()
        for combo in found.combos:
            row = fs.run_filters_exact(self.frame, 4, 2, True, combo, include_yearly=False)
            self.assertGreaterEqual(row["Trades"], 25)
            entries = self.frame["Buy"].to_numpy().copy()
            for item in combo:
                entries &= fs.filter_mask(self.frame, item)
            signatures.add(entries.tobytes())
        self.assertEqual(len(signatures), len(found.combos))

    def test_every_filter_in_a_result_cuts_enough_entries(self):
        found = fs.search_combinations(
            self.inputs, self.candidates, size=3, top_n=20, beam_width=20, min_trades=10
        )
        self.assertTrue(found.combos)
        base = self.inputs.base
        for combo in found.combos:
            masks = [fs.filter_mask(self.frame, item) for item in combo]
            together = (base & np.logical_and.reduce(masks)).sum()
            for k in range(len(masks)):
                others = (base & np.logical_and.reduce(masks[:k] + masks[k + 1:])).sum()
                self.assertLessEqual(together, 0.9 * others)

    def test_weak_pairs_are_counted_as_dropped(self):
        strict = fs.search_combinations(self.inputs, self.candidates, size=2, top_n=5, min_trades=10)
        loose = fs.search_combinations(self.inputs, self.candidates, size=2, top_n=5, min_trades=10, min_effect=0)
        self.assertEqual(loose.dropped.get("weak_filter"), 0)
        self.assertEqual(strict.tested + strict.dropped["weak_filter"], loose.tested)

    def test_trio_beam_extends_good_pairs(self):
        pairs = fs.search_combinations(
            self.inputs, self.candidates, size=2, top_n=10, min_trades=10
        )
        trios = fs.search_combinations(
            self.inputs, self.candidates, size=3, top_n=10, beam_width=20, min_trades=10
        )
        self.assertTrue(trios.combos)
        beam = {
            frozenset(combo)
            for combo in fs.search_combinations(
                self.inputs, self.candidates, size=2, top_n=20, min_trades=10
            ).combos
        }
        for combo in trios.combos:
            self.assertEqual(len(combo), 3)
            self.assertEqual(len(set(combo)), 3)
            # every trio contains a pair that made the beam
            self.assertTrue(
                any(frozenset(pair) in beam for pair in itertools.combinations(combo, 2))
            )
        self.assertGreater(trios.tested, pairs.tested)

    def test_too_few_candidates_returns_nothing(self):
        found = fs.search_combinations(
            self.inputs,
            fs.build_candidates(self.frame, [Filter("A", "less", 0.5)], self.inputs.base, 1),
            size=2,
        )
        self.assertEqual(found.combos, [])

    def test_rejects_unsupported_size_and_zero_days(self):
        with self.assertRaises(ValueError):
            fs.search_combinations(self.inputs, self.candidates, size=4)
        with self.assertRaises(ValueError):
            fs.build_sim_inputs(self.frame, 0, 0, True)


class CombinationSweepTests(unittest.TestCase):
    def test_rows_are_exact_ranked_and_round_trip(self):
        frame = _frame(11, columns=("IBR", "CCI", "Stoch", "ER", "ADX14"))
        rows, info = fs.combination_sweep(
            frame, 4, 2, True, size=2, top_n=8, min_trades=15
        )
        self.assertEqual(info["search"], "exhaustive")
        self.assertGreater(info["combinations_tested"], 0)
        self.assertTrue(0 < len(rows) <= 8)
        sharpes = rows["Sharpe"].tolist()
        self.assertEqual(sharpes, sorted(sharpes, reverse=True))
        first = rows.iloc[0]
        filters = fs.filters_from_row(first.to_dict())
        self.assertEqual(len(filters), 2)
        replay = fs.run_filters_exact(frame, 4, 2, True, filters, include_yearly=True)
        self.assertEqual(replay["Trades"], first["Trades"])
        self.assertEqual(replay["Yearly"], first["Yearly"])
        self.assertEqual(first["Indicator"], " + ".join(f.column for f in filters))

    def test_filters_from_row_rejects_bad_rows(self):
        with self.assertRaises(ValueError):
            fs.filters_from_row({})
        with self.assertRaises(ValueError):
            fs.filters_from_row({"Filters": [{"Indicator": "A", "Condition": "equal", "Value": 1}]})
        with self.assertRaises(ValueError):
            fs.filters_from_row({"Filters": [{"Indicator": "A", "Condition": "less", "Value": "x"}]})

    def test_negative_zero_is_not_shown(self):
        label = fs.combo_row_labels([Filter("A", "less", -0.0), Filter("B", "more", 0.25)])
        self.assertEqual(label["Value"], "0 + 0.25")


def _draft_request() -> BuilderBacktestRequest:
    return BuilderBacktestRequest(
        symbol="SPY",
        years=25,
        direction="long",
        hold_days=4,
        profit=2,
        name="Draft",
        conditions=[BuilderCondition(left="RSI2", operator="<=", right="20", logic="AND")],
    )


class RefineModeTests(unittest.TestCase):
    def _run(self, frame, **request_kwargs):
        request = BuilderRefineRequest(
            mode="indicator-combo-sweep", strategy=_draft_request(), **request_kwargs
        )
        reference = frame.assign(RSI2=np.linspace(10, 30, len(frame)))
        with patch("api.services.load_ticker_data", return_value=reference.copy()):
            with patch(
                "api.services.prepare_builder_refine_frame",
                return_value=(frame, 4, 2, True, None),
            ):
                return run_builder_refine(request)

    def test_returns_ranked_combos_with_full_period_and_search_meta(self):
        frame = _frame(12, columns=("IBR", "CCI", "Stoch", "ER", "ADX14"))
        with patch.object(config, "MIN_IN_SAMPLE_TRADES", 15):
            payload = self._run(frame, combo_top_n=6)
        rows = payload["rows"]
        self.assertTrue(rows)
        self.assertLessEqual(len(rows), 6)
        for row in rows:
            self.assertEqual(len(row["Filters"]), 2)
            self.assertIn("FullPeriod", row)
            self.assertGreaterEqual(row["Trades"], 15)
        meta = payload["meta"]
        self.assertEqual(meta["sample"], "in_sample")
        self.assertEqual(meta["combo_search"]["size"], 2)
        self.assertFalse(meta["combo_search"]["market_wide_only"])
        self.assertGreater(meta["combo_search"]["combinations_tested"], 0)

    def test_trio_mode(self):
        frame = _frame(13, columns=("IBR", "CCI", "Stoch", "ER", "ADX14"))
        with patch.object(config, "MIN_IN_SAMPLE_TRADES", 15):
            payload = self._run(frame, combo_size=3, combo_top_n=4, combo_beam_width=20)
        self.assertTrue(payload["rows"])
        self.assertTrue(all(len(row["Filters"]) == 3 for row in payload["rows"]))

    def test_sell_side_is_refused(self):
        with self.assertRaises(ValueError):
            self._run(_frame(14), is_sell=True)

    def test_outcome_replays_a_combo_row(self):
        from api.schemas import BuilderRefineOutcomeRequest
        from api.services import run_builder_refine_outcome

        frame = _frame(15, columns=("IBR", "CCI"))
        row = fs.combo_row_labels([Filter("IBR", "less", 0.6), Filter("CCI", "more", 0.2)])
        request = BuilderRefineOutcomeRequest(
            mode="indicator-combo-sweep", strategy=_draft_request(), row=row, sample="full"
        )
        reference = frame.assign(RSI2=np.linspace(10, 30, len(frame)))
        with patch("api.services.load_ticker_data", return_value=reference.copy()):
            with patch(
                "api.services.prepare_builder_refine_frame",
                return_value=(frame, 4, 2, True, None),
            ):
                outcome = run_builder_refine_outcome(request)
        expected = fs.run_filters_exact(
            frame, 4, 2, True, fs.filters_from_row(row), include_yearly=False
        )
        self.assertEqual(outcome["metrics"]["Trades"], expected["Trades"])


if __name__ == "__main__":
    unittest.main()
