"""The position the Scan page reports for a portfolio: action, held vs target, orders, reason."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from api import scan_position as sp
from api.allocation import AllInModel, proportional_model, tiered_model
from api.portfolio_service import (
    APPROXIMATE_REGIME,
    build_portfolio_scan_book,
    scan_position,
    simulate_portfolio_allocation,
)
from api.position_episodes import ADD, ENTER, EXIT, REDUCE
from api.schemas import ScanPosition, ScanRowResponse
from tests.test_allocation import DATES, _market, _patched_portfolio, _signals, _strategy
from tests.test_nested_portfolios import child
from tests.test_nested_portfolios import run as run_nested
from tests.test_nested_portfolios import signal

CHANGES = [0.0, 0.10, 0.20, -0.10, 0.05, 0.02]


def two_symbol_market(changes_a=CHANGES, changes_b=CHANGES):
    track = {"AAA": pd.Series(changes_a, index=DATES), "BBB": pd.Series(changes_b, index=DATES)}
    close = {
        "AAA": pd.Series(np.linspace(100, 110, len(DATES)), index=DATES),
        "BBB": pd.Series(np.linspace(50, 60, len(DATES)), index=DATES),
    }
    return track, close


def run_until(bar, strategies, events, model, *, symbols=None):
    """The engine run on the first ``bar + 1`` bars, so its last bar is the day being scanned."""
    n = bar + 1
    dates = DATES[:n]
    signals = {s.id: _signals(s.id, dates, ev[:n], symbol=s.symbol) for s, ev in zip(strategies, events)}
    track, close = two_symbol_market()
    track = {k: v.iloc[:n] for k, v in track.items()}
    close = {k: v.iloc[:n] for k, v in close.items()}
    return simulate_portfolio_allocation(
        strategies, signals, model=model, cost_bps=0.0, global_proxy=None,
        track_changes=track, close_series=close,
    )


class ActionTests(unittest.TestCase):
    def test_every_transition(self):
        a = sp.action_for
        self.assertEqual(a(0.0, 0.55, True), ENTER)
        self.assertEqual(a(0.55, 1.0, True), ADD)
        self.assertEqual(a(1.0, 0.55, True), REDUCE)
        self.assertEqual(a(0.55, 0.0, True), EXIT)
        self.assertEqual(a(0.55, 0.55 * 1.2, False), sp.HOLD)    # drift is not a trade
        self.assertEqual(a(0.0, 0.0, False), sp.FLAT)
        self.assertEqual(a(1.0, 1.0, True), sp.REBALANCE)        # same total, different split

    def test_tiny_changes_still_count(self):
        self.assertEqual(sp.action_for(0.5, 0.5 + 1e-6, True), ADD)
        self.assertEqual(sp.action_for(0.5, 0.5 - 1e-6, True), REDUCE)

    def test_a_position_that_is_just_dust_is_flat(self):
        self.assertEqual(sp.action_for(0.0, 1e-12, False), sp.FLAT)

    def test_flags_keep_the_old_buy_hold_sell_meaning(self):
        self.assertEqual(sp.signal_flags(ENTER, 0.0), dict(buy_signal=True, hold_long=False, sell_signal=False))
        self.assertEqual(sp.signal_flags(ADD, 0.55), dict(buy_signal=True, hold_long=True, sell_signal=False))
        self.assertEqual(sp.signal_flags(REDUCE, 1.0), dict(buy_signal=False, hold_long=True, sell_signal=True))
        self.assertEqual(sp.signal_flags(EXIT, 0.55), dict(buy_signal=False, hold_long=True, sell_signal=True))
        self.assertEqual(sp.signal_flags(sp.HOLD, 0.55), dict(buy_signal=False, hold_long=True, sell_signal=False))
        self.assertEqual(sp.signal_flags(sp.FLAT, 0.0), dict(buy_signal=False, hold_long=False, sell_signal=False))

    def test_the_reason_names_who_changed_and_how_many_are_active(self):
        text = sp.reason_text(ADD, entered=["Sig 7"], exited=[], active=2, total=4)
        self.assertEqual(text, "Sig 7 entered (2 of 4 active)")
        text = sp.reason_text(REDUCE, entered=[], exited=["Sig 16"], active=1, total=4)
        self.assertEqual(text, "Sig 16 exited (1 of 4 active)")
        self.assertEqual(sp.reason_text(sp.HOLD, entered=[], exited=[], active=2, total=4), "No change (2 of 4 active)")
        self.assertEqual(sp.reason_text(sp.FLAT, entered=[], exited=[], active=0, total=4), "None active (0 of 4 active)")
        both = sp.reason_text(REDUCE, entered=["A"], exited=["B"], active=1, total=2)
        self.assertEqual(both, "A entered; B exited (1 of 2 active)")


class TieredBookTests(unittest.TestCase):
    """A flat tiered book: A holds bars 1-4, B holds bars 3-4 (see the events)."""

    events = [
        ["", "buy", "", "", "sell", ""],         # A: in from bar 1, exits on bar 4
        ["", "", "", "buy", "sell", ""],         # B: in from bar 3, exits on bar 4
    ]

    def position(self, bar, model=None):
        strategies = [_strategy("a", "AAA"), _strategy("b", "AAA")]
        sim = run_until(bar, strategies, self.events, model or tiered_model(55))
        return sp.position_from_simulation(sim, model_label=(model or tiered_model(55)).label), sim

    def test_flat_before_anything_happens(self):
        position, _ = self.position(0)
        self.assertEqual(position["action"], sp.FLAT)
        self.assertEqual((position["held_now"], position["target"], position["orders"]), (0.0, 0.0, []))
        self.assertEqual(position["reason"], "None active (0 of 2 active)")

    def test_the_first_signal_enters_at_the_lone_weight(self):
        position, _ = self.position(1)
        self.assertEqual(position["action"], ENTER)
        self.assertAlmostEqual(position["held_before"], 0.0)
        self.assertAlmostEqual(position["held_now"], 0.55)
        self.assertAlmostEqual(position["target"], 0.55)
        self.assertAlmostEqual(position["delta"], 0.55)
        self.assertEqual(position["entered"], ["Strategy a"])
        self.assertEqual(position["active_strategies"], ["Strategy a"])
        (order,) = position["orders"]
        self.assertEqual((order["side"], order["event"], order["instrument"]), ("buy", ENTER, "AAA"))
        self.assertAlmostEqual(order["weight_before"], 0.0)
        self.assertAlmostEqual(order["weight_after"], 0.55)
        self.assertAlmostEqual(order["price"], float(np.linspace(100, 110, len(DATES))[1]))
        self.assertEqual(order["kind"], "synthetic")

    def test_a_day_with_no_signal_change_holds_and_shows_the_drift(self):
        position, sim = self.position(2)
        self.assertEqual(position["action"], sp.HOLD)
        self.assertEqual(position["orders"], [])
        self.assertAlmostEqual(position["target"], 0.55)
        # the position gained 20% on the day, so it is now a bigger share of the book, with no trade
        self.assertGreater(position["held_now"], 0.55 + 0.02)
        self.assertAlmostEqual(position["held_before"], position["held_now"])
        self.assertEqual(position["delta"], 0.0)
        self.assertEqual(position["reason"], "No change (1 of 2 active)")
        np.testing.assert_allclose(position["held_now"], sim.post_weight[-1])

    def test_a_second_strategy_adds_up_to_the_full_book(self):
        position, sim = self.position(3)
        self.assertEqual(position["action"], ADD)
        txn = [t for t in sim.transactions if t.bar == 3][0]
        self.assertAlmostEqual(position["held_before"], txn.exposure_before)   # drifted, not the old 55%
        self.assertGreater(position["held_before"], 0.55)
        self.assertAlmostEqual(position["held_now"], 1.0)
        self.assertAlmostEqual(position["target"], 1.0)
        self.assertAlmostEqual(position["delta"], 1.0 - txn.exposure_before)
        self.assertEqual(position["entered"], ["Strategy b"])
        self.assertEqual(position["active"], 2)
        (order,) = position["orders"]
        self.assertEqual((order["side"], order["event"]), ("buy", ADD))
        self.assertGreater(order["weight_after"], order["weight_before"])

    def test_both_leaving_together_is_an_exit(self):
        position, _ = self.position(4)
        self.assertEqual(position["action"], EXIT)
        self.assertAlmostEqual(position["held_now"], 0.0)
        self.assertEqual(sorted(position["exited"]), ["Strategy a", "Strategy b"])
        (order,) = position["orders"]
        self.assertEqual((order["side"], order["event"], order["weight_after"]), ("sell", EXIT, 0.0))
        self.assertEqual(position["holdings"], [])

    def test_one_leaving_while_the_other_stays_is_a_reduce(self):
        events = [["", "buy", "", "", "sell", ""], ["", "", "", "buy", "", ""]]
        sim = run_until(4, [_strategy("a", "AAA"), _strategy("b", "AAA")], events, tiered_model(55))
        position = sp.position_from_simulation(sim, model_label="Tiered")
        self.assertEqual(position["action"], REDUCE)
        self.assertAlmostEqual(position["held_now"], 0.55)
        self.assertAlmostEqual(position["target"], 0.55)
        self.assertLess(position["delta"], 0)
        self.assertEqual(position["exited"], ["Strategy a"])
        (order,) = position["orders"]
        self.assertEqual((order["side"], order["event"]), ("sell", REDUCE))

    def test_holdings_list_what_is_held_after_the_decision(self):
        position, sim = self.position(3)
        (holding,) = position["holdings"]
        self.assertEqual(holding["instrument"], "AAA")
        self.assertAlmostEqual(holding["weight"], 1.0)
        self.assertAlmostEqual(holding["price"], float(np.linspace(100, 110, len(DATES))[3]))

    def test_proportional_and_all_in_use_the_same_vocabulary(self):
        strategies = [_strategy("a", "AAA"), _strategy("b", "AAA")]
        for model in (AllInModel(), proportional_model("linear")):
            sim = run_until(1, strategies, self.events, model)
            self.assertEqual(sp.position_from_simulation(sim, model_label=model.label)["action"], ENTER)
        sim = run_until(1, strategies, self.events, proportional_model("linear"))
        self.assertAlmostEqual(sp.position_from_simulation(sim, model_label="x")["held_now"], 0.5)

    def test_the_position_is_a_valid_scan_position(self):
        for bar in range(5):
            position, _ = self.position(bar)
            ScanPosition(**position)


class MixedSymbolTests(unittest.TestCase):
    def test_a_change_of_split_with_the_same_total_is_a_rebalance_with_two_orders(self):
        strategies = [_strategy("a", "AAA"), _strategy("b", "BBB")]
        events = [["", "buy", "", "", "", ""], ["", "", "", "buy", "", ""]]
        sim = run_until(3, strategies, events, AllInModel())
        position = sp.position_from_simulation(sim, model_label="All-in")
        self.assertEqual(position["action"], sp.REBALANCE)
        self.assertAlmostEqual(position["held_now"], 1.0)
        by_symbol = {o["instrument"]: o for o in position["orders"]}
        self.assertEqual(by_symbol["AAA"]["side"], "sell")
        self.assertEqual(by_symbol["BBB"]["side"], "buy")
        self.assertAlmostEqual(by_symbol["BBB"]["weight_after"], 0.5)
        self.assertEqual({h["instrument"] for h in position["holdings"]}, {"AAA", "BBB"})
        flags = sp.signal_flags(position["action"], position["held_before"])
        self.assertTrue(flags["buy_signal"] and flags["sell_signal"])


class NestedBookTests(unittest.TestCase):
    def test_a_nested_book_reports_leaf_names_and_the_final_weights(self):
        a1, a2 = _strategy("a1", "AAA"), _strategy("a2", "BBB")
        states = {"a1": signal("a1", "AAA", 0, 8), "a2": signal("a2", "BBB", 3, 6)}
        wrapper, ws = child([a1, a2], states, proportional_model("linear"), name="A")
        direct = _strategy("d", "CCC")
        sim = run_nested([wrapper, direct], {wrapper.id: ws, "d": signal("d", "CCC", 4, 7)}, proportional_model("linear"))
        position = sp.position_from_simulation(sim, model_label="Proportional")
        self.assertEqual(sim.strategy_count, 3)
        self.assertEqual(position["total"], 3)
        self.assertTrue(all(name in {"A / Strategy a1", "A / Strategy a2", "Strategy d"} for name in position["active_strategies"]))
        self.assertAlmostEqual(position["held_now"], float(sim.post_weight[-1]))
        ScanPosition(**position)


class OverlayAndSingleTests(unittest.TestCase):
    def frame(self, trade_in, hold, trade_out):
        return pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=2), "Close": [100.0, 101.5],
                             "LongTradeIn": [False, trade_in], "HoldLong": [False, hold], "LongTradeOut": [False, trade_out]})

    def overlay(self, *flags):
        return sp.position_from_overlay(self.frame(*flags), symbol="SOXX", leverage=3.0, names=["S1"], entered=[],
                                        exited=[], total=2, model_label="All-in")

    def test_each_day_of_a_plain_all_in_book(self):
        self.assertEqual(self.overlay(True, False, False)["action"], ENTER)
        self.assertEqual(self.overlay(False, True, False)["action"], sp.HOLD)
        self.assertEqual(self.overlay(False, True, True)["action"], EXIT)
        self.assertEqual(self.overlay(False, False, False)["action"], sp.FLAT)

    def test_an_overlay_entry_is_one_order_for_the_whole_book(self):
        position = self.overlay(True, False, False)
        (order,) = position["orders"]
        self.assertEqual((order["instrument"], order["side"], order["weight_after"], order["price"], order["leverage"]),
                         ("SOXX", "buy", 1.0, 101.5, 3.0))
        self.assertEqual(position["held_now"], 1.0)
        self.assertEqual(position["as_of"], "2024-01-02")
        self.assertEqual(self.overlay(False, True, False)["holdings"][0]["weight"], 1.0)

    def test_single_strategies_are_all_in(self):
        self.assertEqual(sp.position_from_flags(True, False, False)["action"], ENTER)
        self.assertEqual(sp.position_from_flags(False, True, False)["action"], sp.HOLD)
        self.assertEqual(sp.position_from_flags(False, True, True)["action"], EXIT)
        flat = sp.position_from_flags(False, False, False, as_of="2024-01-02")
        self.assertEqual((flat["action"], flat["held_now"], flat["reason"], flat["as_of"]), (sp.FLAT, 0.0, "No signal", "2024-01-02"))
        self.assertEqual(sp.position_from_flags(True, False, False)["held_now"], 1.0)
        for flags in ((True, False, False), (False, True, False), (False, True, True), (False, False, False)):
            ScanPosition(**sp.position_from_flags(*flags))


class ScanRowTests(unittest.TestCase):
    def test_a_regime_adjusted_portfolio_is_read_as_linear_and_flagged(self):
        request = _patched_portfolio(self, dict(allocation_model="regime_adjusted", regime_score="robustness", regime_penalty="medium"))
        book = build_portfolio_scan_book(request)
        self.assertEqual(book.approximate, APPROXIMATE_REGIME)
        self.assertEqual(book.allocation.model.id, "proportional_linear")
        position = scan_position(book)
        self.assertEqual(position["approximate"], APPROXIMATE_REGIME)

    def test_a_tiered_portfolio_is_exact(self):
        request = _patched_portfolio(self, dict(allocation_model="tiered", tier_single_pct=40))
        book = build_portfolio_scan_book(request)
        self.assertIsNone(book.approximate)
        position = scan_position(book)
        self.assertEqual(position["model"], "Tiered (40% / 100%)")
        self.assertIsNone(position["approximate"])
        ScanPosition(**position)

    def test_the_row_flags_follow_the_position(self):
        from api import scan_service

        request = _patched_portfolio(self, dict(allocation_model="tiered", tier_single_pct=40))
        portfolio = SimpleNamespace(id="p1", name="Tiered book", **vars(request))
        row = scan_service._portfolio_scan_row(portfolio)
        self.assertEqual(row["source"], "portfolio")
        action = row["position"]["action"]
        self.assertEqual(
            {k: row[k] for k in ("buy_signal", "hold_long", "sell_signal")},
            sp.signal_flags(action, row["position"]["held_before"]),
        )
        ScanRowResponse(**{**row, "position": row["position"]})

    def test_the_exact_position_of_an_unknown_portfolio_is_an_error(self):
        from api.scan_service import exact_portfolio_position

        with patch("api.portfolio_store.get_portfolio_by_id", return_value=None):
            with self.assertRaisesRegex(ValueError, "Unknown portfolio"):
                exact_portfolio_position("nope")

    def test_the_exact_position_runs_the_full_history_without_the_linear_stand_in(self):
        from api import scan_service

        request = _patched_portfolio(self, dict(allocation_model="regime_adjusted", regime_score="robustness", regime_penalty="medium"))
        portfolio = SimpleNamespace(id="p1", name="Regime", **vars(request))
        with patch("api.portfolio_store.get_portfolio_by_id", return_value=portfolio):
            position = scan_service.exact_portfolio_position("p1")
        self.assertIsNone(position["approximate"])
        self.assertEqual(position["model"], "Regime-adjusted")


class EngineFieldsTests(unittest.TestCase):
    def test_the_run_records_the_final_bar_for_the_scan(self):
        strategies = [_strategy("a", "AAA"), _strategy("b", "AAA")]
        events = [["", "buy", "", "", "sell", ""], ["", "", "", "buy", "sell", ""]]
        sim = run_until(4, strategies, events, tiered_model(55))
        self.assertEqual(sim.strategy_count, 2)
        self.assertEqual(sim.last_active_names, ())
        self.assertEqual({c.strategy_name for c in sim.last_signal_changes}, {"Strategy a", "Strategy b"})
        self.assertEqual(sim.last_close["AAA"], float(np.linspace(100, 110, len(DATES))[4]))
        sim = run_until(3, strategies, events, tiered_model(55))
        self.assertEqual(sorted(sim.last_active_names), ["Strategy a", "Strategy b"])


if __name__ == "__main__":
    unittest.main()
