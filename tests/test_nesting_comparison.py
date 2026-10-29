"""The allocation comparison with child-portfolio variants (``include_nesting``)."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from api import nesting_variants as nv
from api.allocation import AllInModel, proportional_model
from api.nested_portfolios import make_child_member
from api.portfolio_service import (
    START_CAPITAL,
    AllocationSettings,
    simulate_portfolio_allocation,
    simulate_portfolio_allocation_comparison,
)
from api.schemas import PortfolioAllocationComparisonRequest
from tests.test_allocation import DATES, _bind_neutral, _market, _signals, _strategy

# four strategies on one symbol, in and out at different times
EVENTS = {
    "a": ["", "buy", "", "", "sell", ""],
    "b": ["", "", "buy", "", "", "sell"],
    "c": ["buy", "", "", "sell", "", ""],
    "d": ["", "", "", "buy", "", "sell"],
}


def patched_portfolio(test, ids=("a", "b", "c", "d"), **request_kwargs):
    strategies = [_strategy(i) for i in ids]
    signals = {i: _signals(i, DATES, EVENTS[i]) for i in ids}
    track, close = _market()
    market = (pd.DataFrame(), {"AAA": pd.DataFrame({"Date": DATES, "Close": np.linspace(100, 110, 6)})}, track, close)
    by_id = {s.id: s for s in strategies}
    for item in (
        patch("api.portfolio_service.bind_regime_context", side_effect=_bind_neutral),
        patch("api.portfolio_service.get_strategy_by_id", side_effect=lambda sid: by_id[sid]),
        patch("api.portfolio_service._load_portfolio_market_data", return_value=market),
        patch("api.portfolio_service._load_strategy_signals", side_effect=lambda strategy, **_: signals[strategy.id]),
    ):
        item.start()
        test.addCleanup(item.stop)
    return SimpleNamespace(
        strategy_ids=list(ids), portfolio_ids=[], overlap_mode="hold_until_all_exit", proxy_symbol=None, years=25,
        **request_kwargs,
    )


class PlanTests(unittest.TestCase):
    def test_two_strategies_have_nothing_to_regroup(self):
        self.assertEqual(nv.balanced_splits(2), [])
        self.assertIsNone(nv.plan_nesting(2))
        self.assertIsNone(nv.plan_nesting(1))

    def test_four_strategies_pair_up_three_ways(self):
        labels = [s.label() for s in nv.balanced_splits(4)]
        self.assertEqual(labels, ["A+B | C+D", "A+C | B+D", "A+D | B+C"])

    def test_three_strategies_make_a_pair_and_a_single(self):
        labels = [s.label() for s in nv.balanced_splits(3)]
        self.assertEqual(labels, ["A+B | C", "A+C | B", "B+C | A"])

    def test_every_split_is_distinct_and_covers_every_strategy_once(self):
        for n in range(3, 9):
            seen = set()
            for structure in nv.balanced_splits(n):
                everyone = sorted(i for block in structure.blocks for i in block)
                self.assertEqual(everyone, list(range(n)))
                sizes = sorted(len(b) for b in structure.blocks)
                self.assertEqual(sizes, [n // 2, n - n // 2])
                canonical = frozenset(frozenset(b) for b in structure.blocks)
                self.assertNotIn(canonical, seen, f"{structure.label()} repeats a split")
                seen.add(canonical)

    def test_counts_match_the_binomials(self):
        self.assertEqual([len(nv.balanced_splits(n)) for n in range(3, 9)], [3, 3, 10, 10, 35, 35])

    def test_three_and_four_strategies_run_every_model_pair_in_full(self):
        for n in (3, 4):
            plan = nv.plan_nesting(n)
            self.assertEqual((len(plan.structures), len(plan.model_ids), plan.runs), (3, 5, 75))
            self.assertFalse(plan.truncated)
            self.assertEqual(len(plan.pairs), 25)

    def test_regime_adjusted_is_never_offered(self):
        for tier in nv.MODEL_TIERS:
            self.assertFalse([m for m in tier if "regime" in m])

    def test_bigger_portfolios_stay_inside_the_budget(self):
        for n in range(5, 12):
            plan = nv.plan_nesting(n)
            self.assertLessEqual(plan.runs, nv.MAX_RUNS)
            self.assertTrue(plan.truncated)
            self.assertEqual(plan.total_structures, len(nv.balanced_splits(n)))

    def test_over_budget_the_models_shrink_before_any_structure_is_dropped(self):
        plan = nv.plan_nesting(4, max_runs=50)         # 3 x 16 = 48 fits
        self.assertEqual((len(plan.structures), len(plan.model_ids)), (3, 4))
        plan = nv.plan_nesting(4, max_runs=30)         # 3 x 9 = 27 fits
        self.assertEqual((len(plan.structures), len(plan.model_ids)), (3, 3))
        plan = nv.plan_nesting(4, max_runs=18)         # only 2 structures of 3 models fit
        self.assertEqual((len(plan.structures), len(plan.model_ids)), (2, 3))

    def test_dropped_structures_are_spread_across_the_list(self):
        plan = nv.plan_nesting(7)
        everything = nv.balanced_splits(7)
        picked = [everything.index(s) for s in plan.structures]
        self.assertEqual(picked[0], 0)
        self.assertEqual(picked[-1], len(everything) - 1)
        self.assertEqual(picked, sorted(set(picked)))

    def test_the_keys_and_labels_are_stable(self):
        structure = nv.balanced_splits(4)[1]
        self.assertEqual((structure.key, structure.label()), ("AC-BD", "A+C | B+D"))

    def test_short_labels(self):
        self.assertEqual(nv.short_model_label(AllInModel()), "All-in")
        self.assertEqual(nv.short_model_label(proportional_model("early_conviction")), "Early")
        from api.allocation import tiered_model

        self.assertEqual(nv.short_model_label(tiered_model(55)), "Tiered 55")
        self.assertEqual(nv.short_model_label(tiered_model(37.5)), "Tiered 37.5")


class ComparisonTests(unittest.TestCase):
    def compare(self, nesting, **kwargs):
        request = patched_portfolio(self, include_nesting=nesting, **kwargs)
        return simulate_portfolio_allocation_comparison(request)

    def test_without_the_option_nothing_changes(self):
        payload = self.compare(False)
        self.assertEqual(len(payload["models"]), 9)
        self.assertNotIn("nesting", payload)
        self.assertEqual({row["group"] for row in payload["models"]}, {"flat"})

    def test_with_the_option_the_variants_follow_the_flat_rows(self):
        payload = self.compare(True)
        rows = payload["models"]
        self.assertEqual(len(rows), 9 + 75)
        self.assertEqual([r["group"] for r in rows[:9]], ["flat"] * 9)
        self.assertEqual({r["group"] for r in rows[9:]}, {"nested"})
        ids = [r["model"] for r in rows]
        self.assertEqual(len(set(ids)), len(ids))
        for key in ("model", "label", "metrics", "analytics", "attribution", "drawdown_risk", "structure"):
            self.assertIn(key, rows[-1])
        self.assertEqual(set(payload["equity_curves"][0]) - {"date"}, set(ids))

    def test_the_flat_rows_are_the_same_with_the_option_on(self):
        flat = {r["model"]: r["metrics"] for r in self.compare(False)["models"]}
        both = {r["model"]: r["metrics"] for r in self.compare(True)["models"]}
        for model_id, metrics in flat.items():
            self.assertEqual(both[model_id], metrics)

    def test_the_summary_says_what_ran(self):
        nesting = self.compare(True)["nesting"]
        self.assertIsNone(nesting["skipped"])
        self.assertEqual((nesting["structures"], nesting["total_structures"], nesting["variants"]), (3, 3, 75))
        self.assertEqual(nesting["models"], ["All-in", "Linear", "Early", "Tiered 55", "Confirm"])
        self.assertFalse(nesting["truncated"])
        self.assertEqual([s["letter"] for s in nesting["strategies"]], ["A", "B", "C", "D"])
        self.assertEqual(nesting["strategies"][0]["name"], "Strategy a")

    def test_labels_name_the_grouping_and_both_models(self):
        rows = {r["model"]: r for r in self.compare(True)["models"]}
        row = rows["nested:AB-CD:proportional_linear:all_in"]
        self.assertEqual(row["label"], "A+B | C+D · Linear / All-in")
        self.assertEqual(row["structure"], {"key": "AB-CD", "grouping": "A+B | C+D", "parent_model": "Linear", "child_model": "All-in"})

    def test_a_variant_is_a_real_run_of_the_nested_engine(self):
        request = patched_portfolio(self, include_nesting=True)
        payload = simulate_portfolio_allocation_comparison(request)
        row = {r["model"]: r for r in payload["models"]}["nested:AB-CD:proportional_linear:proportional_early_conviction"]

        strategies = [_strategy(i) for i in "abcd"]
        signals = {s.id: _signals(s.id, DATES, EVENTS[s.id]) for s in strategies}
        track, close = _market()
        members, member_signals = [], {}
        for name, group in (("A+B", strategies[:2]), ("C+D", strategies[2:])):
            child, state = make_child_member(
                member_id=f"portfolio:{name}", name=name, members=group,
                child_signals={m.id: signals[m.id] for m in group},
                allocation=AllocationSettings(model=proportional_model("early_conviction"), cost_bps=0.0),
                proxy=None, years=25,
            )
            members.append(child)
            member_signals[child.id] = state
        direct = simulate_portfolio_allocation(
            members, member_signals, model=proportional_model("linear"), cost_bps=0.0, global_proxy=None,
            track_changes=track, close_series=close, master_dates=DATES,
        )
        # the comparison rebases to START_CAPITAL from the first eligible bar
        start = direct.start_index
        expected = direct.frame["RollingPnL"].iloc[-1] / direct.frame["RollingPnL"].iloc[start] * START_CAPITAL
        self.assertAlmostEqual(row["metrics"]["rolling_pnl"], expected, places=6)

    def test_all_in_everywhere_on_one_symbol_is_the_flat_all_in(self):
        rows = {r["model"]: r for r in self.compare(True)["models"]}
        flat = rows["all_in"]["metrics"]["rolling_pnl"]
        for key in ("AB-CD", "AC-BD", "AD-BC"):
            nested = rows[f"nested:{key}:all_in:all_in"]["metrics"]["rolling_pnl"]
            self.assertAlmostEqual(nested, flat, places=6, msg=key)

    def test_three_strategies_leave_one_direct_member(self):
        payload = self.compare(True, ids=("a", "b", "c"))
        self.assertEqual(payload["nesting"]["variants"], 75)
        labels = {r["structure"]["grouping"] for r in payload["models"] if r["group"] == "nested"}
        self.assertEqual(labels, {"A+B | C", "A+C | B", "B+C | A"})

    def test_a_direct_member_is_not_wrapped_in_a_child(self):
        rows = {r["model"]: r for r in self.compare(True, ids=("a", "b", "c"))["models"]}
        # with the pair all-in and the parent all-in, the single is treated like any strategy: still 100% in the market
        self.assertAlmostEqual(
            rows["nested:AB-C:all_in:all_in"]["metrics"]["rolling_pnl"], rows["all_in"]["metrics"]["rolling_pnl"], places=6
        )

    def test_two_strategies_are_skipped_with_a_reason(self):
        payload = self.compare(True, ids=("a", "b"))
        self.assertEqual(len(payload["models"]), 9)
        self.assertIn("at least 3", payload["nesting"]["skipped"])
        self.assertEqual(payload["nesting"]["strategies"][1]["letter"], "B")

    def test_a_portfolio_that_already_has_children_is_skipped(self):
        from api.portfolio_service import _nested_comparison_rows

        strategies = [_strategy("a"), SimpleNamespace(id="portfolio:x", name="Child", nested_spec=object()), _strategy("c")]
        rows, curves, summary = _nested_comparison_rows(
            strategies, {}, cost_bps=0.0, tier_single_pct=None, global_proxy=None, track_changes={}, close_series={},
            master_dates=DATES, reference=None, start=0, years=25,
        )
        self.assertEqual((rows, curves), ([], []))
        self.assertIn("already has child portfolios", summary["skipped"])

    def test_trading_costs_apply_to_every_portfolio_in_a_variant(self):
        free = {r["model"]: r for r in self.compare(True)["models"]}
        paid = {r["model"]: r for r in self.compare(True, transaction_cost_bps=25.0)["models"]}
        key = "nested:AB-CD:proportional_linear:proportional_linear"
        self.assertEqual(free[key]["analytics"]["total_transaction_costs"], 0)
        self.assertGreater(paid[key]["analytics"]["total_transaction_costs"], 0)
        self.assertLess(paid[key]["metrics"]["rolling_pnl"], free[key]["metrics"]["rolling_pnl"])

    def test_every_curve_starts_at_the_starting_capital(self):
        payload = self.compare(True)
        first = payload["equity_curves"][0]
        for row in payload["models"]:
            self.assertAlmostEqual(first[row["model"]], START_CAPITAL, msg=row["model"])


class RequestTests(unittest.TestCase):
    def test_the_option_is_off_by_default(self):
        self.assertFalse(PortfolioAllocationComparisonRequest(strategy_ids=["a"]).include_nesting)
        self.assertTrue(PortfolioAllocationComparisonRequest(strategy_ids=["a"], include_nesting=True).include_nesting)


if __name__ == "__main__":
    unittest.main()
