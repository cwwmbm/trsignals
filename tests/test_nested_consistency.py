"""Nested portfolios report the same things a flat portfolio would.

Counts and names on a transaction describe the same strategies, signal-change ids are stable
and match flat ids for direct strategies, and scan rows name the real instruments.
"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from api.allocation import AllInModel, proportional_model
from api.portfolio_service import signal_symbols
from api import scan_service
from tests.test_allocation import _strategy
from tests.test_nested_portfolios import DATES, child, run, signal


def staggered_case():
    """Child A holds strategies on AAA and BBB (one of them delayed), next to a direct CCC strategy."""
    a1, a2, direct = _strategy("a1", "AAA"), _strategy("a2", "BBB"), _strategy("d", "CCC")
    states = {"a1": signal("a1", "AAA", 0, 8), "a2": signal("a2", "BBB", 3, 6)}
    return a1, a2, direct, states


class TransactionCountTests(unittest.TestCase):
    def check_counts_match_names(self, result):
        self.assertGreater(len(result.transactions), 0)
        for transaction in result.transactions:
            self.assertEqual(transaction.active_after, len(transaction.active_strategies))
            self.assertLessEqual(transaction.active_after, transaction.eligible_after)

    def test_counts_are_per_strategy_and_agree_with_the_names_listed(self):
        a1, a2, direct, states = staggered_case()
        wrapper, ws = child([a1, a2], states, proportional_model("linear"), name="A")
        result = run([wrapper, direct], {wrapper.id: ws, "d": signal("d", "CCC", 4, 7)},
                     proportional_model("linear"))
        self.check_counts_match_names(result)
        # Two strategies in A plus the direct one are eligible: three, not the two members.
        self.assertEqual({t.eligible_after for t in result.transactions}, {3})
        self.assertEqual(int(result.eligible_count[0]), 2)  # the frame keeps per-member counts

    def test_a_wrapped_flat_portfolio_reports_the_same_counts_as_the_flat_one(self):
        a1, a2, _, states = staggered_case()
        for model in (AllInModel(), proportional_model("early_conviction")):
            with self.subTest(model=model.id):
                wrapper, ws = child([a1, a2], states, model, name="A")
                nested = run([wrapper], {wrapper.id: ws})
                flat = run([a1, a2], states, model)
                self.assertEqual(len(nested.transactions), len(flat.transactions))
                for n, f in zip(nested.transactions, flat.transactions):
                    self.assertEqual((n.active_after, n.eligible_after), (f.active_after, f.eligible_after))
                    self.assertEqual(sorted(n.active_strategies), sorted(f"A / {name}" for name in f.active_strategies))


class SignalChangeIdTests(unittest.TestCase):
    def change_ids(self, result):
        return {c.strategy_id for t in result.transactions for c in t.changes}

    def test_direct_strategies_keep_their_plain_id_and_nested_ones_are_paths(self):
        a1, a2, direct, states = staggered_case()
        wrapper, ws = child([a1, a2], states, proportional_model("linear"), name="A")
        result = run([wrapper, direct], {wrapper.id: ws, "d": signal("d", "CCC", 4, 7)})
        ids = self.change_ids(result)
        self.assertIn("d", ids)
        self.assertIn(f"{wrapper.id}/a1", ids)
        self.assertIn(f"{wrapper.id}/a2", ids)
        self.assertFalse(any(i[:1].isdigit() and ":" in i[:3] for i in ids))  # no positional prefixes

    def test_ids_do_not_depend_on_member_order(self):
        a1, a2, direct, states = staggered_case()
        wrapper, ws = child([a1, a2], states, proportional_model("linear"), name="A")
        states_all = {wrapper.id: ws, "d": signal("d", "CCC", 4, 7)}
        forward = self.change_ids(run([wrapper, direct], states_all))
        backward = self.change_ids(run([direct, wrapper], states_all))
        self.assertEqual(forward, backward)

    def test_the_same_strategy_in_two_children_keeps_distinct_ids(self):
        # Same strategy id in both children, trading different instruments so each change fills.
        a, a_other = _strategy("a", "AAA"), _strategy("a", "BBB")
        c1, s1 = child([a], {"a": signal("a", "AAA", 0, 8)}, name="One")
        c2, s2 = child([a_other], {"a": signal("a", "BBB", 2, 6)}, name="Two")
        result = run([c1, c2], {c1.id: s1, c2.id: s2})
        self.assertEqual(self.change_ids(result), {f"{c1.id}/a", f"{c2.id}/a"})


class ScanSymbolTests(unittest.TestCase):
    def test_signal_symbols_look_through_children(self):
        a1, a2, direct, states = staggered_case()
        wrapper, _ = child([a1, a2], states, name="A")
        self.assertEqual(signal_symbols([_strategy("x", "spy"), _strategy("y", "SOXX")]), ["SOXX", "SPY"])
        self.assertEqual(signal_symbols([wrapper, direct]), ["AAA", "BBB", "CCC"])
        self.assertEqual(signal_symbols([wrapper]), ["AAA", "BBB"])

    def test_signal_symbols_through_two_levels(self):
        a1, a2, _, states = staggered_case()
        inner, inner_state = child([a1, a2], states, name="Inner")
        outer, _ = child([inner], {inner.id: inner_state}, name="Outer")
        self.assertEqual(signal_symbols([outer]), ["AAA", "BBB"])

    def test_scan_row_names_the_underlying_symbols_not_the_placeholder(self):
        a1, a2, direct, states = staggered_case()
        wrapper, ws = child([a1, a2], states, name="A")
        frame = pd.DataFrame({"Date": DATES, "LongTradeIn": False, "HoldLong": False, "LongTradeOut": False,
                              "RollingPnL": 15000.0, "TradePnL": 0.0, "Close": 100.0})
        portfolio = type("P", (), {"id": "p1", "name": "Parent"})()
        with patch("api.portfolio_service.build_portfolio_overlay_frame",
                   return_value=(frame, [wrapper, direct], "desc", 5, 1)):
            row = scan_service._portfolio_scan_row(portfolio)
        self.assertEqual(row["symbol"], "AAA+BBB+CCC")
        self.assertNotIn("PORTFOLIO", row["symbol"])

    def test_flat_scan_rows_are_unchanged(self):
        frame = pd.DataFrame({"Date": DATES, "LongTradeIn": False, "HoldLong": False, "LongTradeOut": False,
                              "RollingPnL": 15000.0, "TradePnL": 0.0, "Close": 100.0})
        portfolio = type("P", (), {"id": "p1", "name": "Flat"})()
        with patch("api.portfolio_service.build_portfolio_overlay_frame",
                   return_value=(frame, [_strategy("a", "SPY"), _strategy("b", "SOXX")], "desc", 5, 1)):
            row = scan_service._portfolio_scan_row(portfolio)
        self.assertEqual(row["symbol"], "SOXX+SPY")


if __name__ == "__main__":
    unittest.main()
