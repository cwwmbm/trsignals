import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from api.allocation import AllInModel, proportional_model
from api.allocation_analytics import charge_overlay_costs
from api.allocation_engine import effective_state, simulate_allocation
from api.portfolio_service import (
    START_CAPITAL,
    StrategySignals,
    _positions_payload,
    replay_overlay_transactions,
    simulate_portfolio,
    simulate_portfolio_allocation,
    simulate_portfolio_overlay,
)
from api.position_episodes import (
    ADD,
    ENTER,
    EXIT,
    REDUCE,
    build_position_episodes,
    classify_event,
)
from api.rebalancer import PortfolioLedger

DATES = pd.bdate_range("2024-01-01", periods=6)
CHANGES = [0.0, 0.10, 0.20, -0.10, 0.05, 0.02]


def _strategy(strategy_id, symbol="AAA", name=None):
    return SimpleNamespace(
        id=strategy_id,
        name=name or strategy_id.upper(),
        symbol=symbol,
        direction="long",
        hold_days=2,
        profit=1,
        confirm_symbols=[],
        proxy_symbol=None,
    )


def _flags(events):
    n = len(events)
    trade_in = np.zeros(n, dtype=bool)
    trade_out = np.zeros(n, dtype=bool)
    hold = np.zeros(n, dtype=bool)
    for i, event in enumerate(events):
        if i > 0:
            hold[i] = (hold[i - 1] and not trade_out[i - 1]) or trade_in[i - 1]
        trade_in[i] = event == "buy" and not hold[i]
        trade_out[i] = event == "sell" and hold[i]
    return trade_in, hold, trade_out


def _signals(strategy_id, dates, events, symbol="AAA", entry_ready=None):
    trade_in, hold, trade_out = _flags(events)
    return StrategySignals(
        strategy_id=strategy_id,
        name=strategy_id.upper(),
        symbol=symbol,
        dates=pd.DatetimeIndex(dates),
        long_trade_in=trade_in,
        hold_long=hold,
        long_trade_out=trade_out,
        entry_ready=None if entry_ready is None else np.asarray(entry_ready, dtype=bool),
    )


def _market(changes=CHANGES, dates=DATES, symbol="AAA"):
    return (
        {symbol: pd.Series(changes, index=dates)},
        {symbol: pd.Series(np.linspace(100, 110, len(dates)), index=dates)},
    )


def _book(model, events_by_strategy, *, changes=CHANGES, dates=DATES, cost_bps=0.0):
    strategies = [_strategy(name) for name in events_by_strategy]
    signals = {name: _signals(name, dates, events) for name, events in events_by_strategy.items()}
    track, close = _market(changes, dates)
    return simulate_portfolio_allocation(
        strategies,
        signals,
        model=model,
        cost_bps=cost_bps,
        global_proxy=None,
        track_changes=track,
        close_series=close,
        master_dates=pd.DatetimeIndex(dates),
    )


def _episodes(book):
    return build_position_episodes(
        book.transactions,
        asset_weight_post=book.asset_weight_post,
        final_marks=book.final_marks,
        last_bar=len(book.dates) - 1,
    )


def _price_path(changes=CHANGES):
    prices = [100.0]
    for change in changes[1:]:
        prices.append(prices[-1] * (1 + change))
    return prices


LINEAR = proportional_model("linear")
NONE = [""] * 6


class ClassificationTests(unittest.TestCase):
    def test_events_come_from_quantity_changes_only(self):
        self.assertEqual(classify_event(0.0, 10.0), ENTER)
        self.assertEqual(classify_event(10.0, 15.0), ADD)
        self.assertEqual(classify_event(10.0, 4.0), REDUCE)
        self.assertEqual(classify_event(4.0, 0.0), EXIT)
        # Dust left over by float noise counts as flat.
        self.assertEqual(classify_event(5e-13, 10.0), ENTER)
        self.assertEqual(classify_event(10.0, 5e-13), EXIT)


class UnderlyingPriceTests(unittest.TestCase):
    """The prices a person reads are the underlying's closes, never simulator internals."""

    def _close_on(self, date):
        return float(_market()[1]["AAA"].loc[pd.Timestamp(date)])

    def test_proportional_events_and_episode_carry_the_underlying_close(self):
        book = _book(
            LINEAR,
            {"a": ["buy", "", "sell", "", "", ""], "b": ["buy", "", "", "sell", "", ""]},
        )
        (episode,) = _episodes(book)
        for event in episode["events"]:
            self.assertAlmostEqual(event["underlying_close"], self._close_on(event["date"]))
        self.assertAlmostEqual(episode["initial_underlying_close"], self._close_on(episode["start"]))
        self.assertAlmostEqual(episode["final_underlying_close"], self._close_on(episode["end"]))
        # The simulated price is a different number and is kept under its own name.
        last = episode["events"][-1]
        self.assertNotAlmostEqual(last["underlying_close"], last["simulated_price"])

    def test_open_episode_has_no_final_price(self):
        book = _book(LINEAR, {"a": ["buy", "", "", "", "", ""]})
        (episode,) = _episodes(book)
        self.assertIsNone(episode["final_underlying_close"])
        self.assertAlmostEqual(episode["initial_underlying_close"], self._close_on(episode["start"]))

    def test_all_in_episodes_use_the_same_closes_the_old_trade_table_did(self):
        strategies = [_strategy("a"), _strategy("b")]
        signals = {
            "a": _signals("a", DATES, ["buy", "", "", "sell", "", ""]),
            "b": _signals("b", DATES, ["", "buy", "", "", "sell", ""]),
        }
        track, close = _market()
        frame = simulate_portfolio_overlay(
            strategies, signals, overlap_mode="hold_until_all_exit", global_proxy=None,
            track_changes=track, close_series=close, strategy_order=["a", "b"],
        )
        replay = replay_overlay_transactions(frame, strategies, signals, global_proxy=None)
        (episode,) = _episodes(replay)
        # The old table priced a trade at the frame's Close on its entry and exit rows.
        entry_row = frame[frame["LongTradeIn"]].iloc[0]
        exit_row = frame[frame["LongTradeOut"]].iloc[0]
        self.assertAlmostEqual(episode["initial_underlying_close"], float(entry_row["Close"]))
        self.assertAlmostEqual(episode["final_underlying_close"], float(exit_row["Close"]))
        self.assertGreater(episode["initial_underlying_close"], 0)

    def test_every_all_in_episode_matches_the_legacy_trade_prices(self):
        for seed in range(4):
            strategies, signals, frame = AllInHistoryTests()._legacy(seed, "hold_until_all_exit")
            replay = replay_overlay_transactions(frame, strategies, signals, global_proxy=None)
            episodes = _episodes(replay)
            enters = frame[frame["LongTradeIn"]]["Close"].to_numpy()
            exits = frame[frame["LongTradeOut"]]["Close"].to_numpy()
            closed = [e for e in episodes if e["status"] == "Closed"]
            np.testing.assert_allclose([e["initial_underlying_close"] for e in episodes], enters)
            np.testing.assert_allclose([e["final_underlying_close"] for e in closed], exits)


class SingleFullEntryAndExitTests(unittest.TestCase):
    def test_one_enter_one_exit_all_in(self):
        book = _book(AllInModel(), {"a": ["buy", "", "", "sell", "", ""]})
        (episode,) = _episodes(book)
        prices = _price_path()
        self.assertEqual(episode["status"], "Closed")
        self.assertEqual([e["event"] for e in episode["events"]], [ENTER, EXIT])
        self.assertAlmostEqual(episode["initial_simulated_price"], prices[0])
        self.assertAlmostEqual(episode["final_simulated_price"], prices[3])
        self.assertEqual(episode["start"], DATES[0].strftime("%Y-%m-%d"))
        self.assertEqual(episode["end"], DATES[3].strftime("%Y-%m-%d"))
        self.assertEqual(episode["days"], 3)
        self.assertAlmostEqual(episode["max_exposure"], 1.0)
        self.assertAlmostEqual(episode["avg_exposure"], 1.0)
        expected = START_CAPITAL * (1.1 * 1.2 * 0.9) - START_CAPITAL
        self.assertAlmostEqual(episode["pnl"], expected, places=6)
        self.assertAlmostEqual(episode["return"], expected / START_CAPITAL, places=9)
        self.assertEqual(episode["unrealized_pnl"], 0.0)
        self.assertEqual(episode["rebalance_count"], 2)

    def test_event_fields_are_the_ledgers_numbers(self):
        book = _book(AllInModel(), {"a": ["buy", "", "", "sell", "", ""]})
        enter, exit_ = _episodes(book)[0]["events"]
        prices = _price_path()
        self.assertAlmostEqual(enter["units_delta"], START_CAPITAL / prices[0])
        self.assertAlmostEqual(enter["trade_value"], START_CAPITAL)
        self.assertAlmostEqual(enter["position_after"], START_CAPITAL / prices[0])
        self.assertAlmostEqual(enter["exposure_before"], 0.0)
        self.assertAlmostEqual(enter["exposure_after"], 1.0)
        self.assertAlmostEqual(exit_["units_delta"], -START_CAPITAL / prices[0])
        self.assertLess(exit_["trade_value"], 0)
        self.assertEqual(exit_["position_after"], 0.0)
        self.assertAlmostEqual(exit_["exposure_after"], 0.0)
        self.assertAlmostEqual(
            exit_["realized_pnl"], START_CAPITAL / prices[0] * (prices[3] - prices[0]), places=6
        )
        self.assertAlmostEqual(exit_["cash_after"], book.cash[3])


class AdditionsTests(unittest.TestCase):
    def setUp(self):
        self.book = _book(
            LINEAR,
            {
                "a": ["buy", "", "", "", "", ""],
                "b": ["", "buy", "", "", "", ""],
                "c": ["", "", "buy", "", "", ""],
            },
        )
        (self.episode,) = _episodes(self.book)

    def test_entry_then_additions_at_different_prices(self):
        events = self.episode["events"]
        prices = _price_path()
        self.assertEqual([e["event"] for e in events], [ENTER, ADD, ADD])
        self.assertEqual(
            [round(e["simulated_price"], 6) for e in events],
            [round(prices[0], 6), round(prices[1], 6), round(prices[2], 6)],
        )
        self.assertAlmostEqual(events[0]["units_delta"], 5000 / 100)
        self.assertAlmostEqual(events[0]["exposure_after"], 1 / 3)
        self.assertAlmostEqual(events[1]["exposure_before"], 5500 / 15500)
        self.assertAlmostEqual(events[1]["exposure_after"], 2 / 3)
        self.assertAlmostEqual(events[2]["exposure_after"], 1.0)
        for event in events:
            self.assertGreater(event["trade_value"], 0)
            self.assertGreater(event["units_delta"], 0)

    def test_average_cost_tracks_every_purchase(self):
        events = self.episode["events"]
        units = sum(e["units_delta"] for e in events)
        cost = sum(e["trade_value"] for e in events)
        self.assertAlmostEqual(events[-1]["position_after"], units)
        self.assertAlmostEqual(events[-1]["average_cost_after"], cost / units)

    def test_open_episode_at_end_carries_unrealized_pnl(self):
        episode = self.episode
        self.assertEqual(episode["status"], "Open")
        self.assertIsNone(episode["end"])
        self.assertIsNone(episode["final_simulated_price"])
        self.assertEqual(episode["days"], 5)  # carried into bars 1..5
        self.assertEqual(episode["realized_pnl"], 0.0)
        final_mark = self.book.final_marks["AAA"]
        expected = episode["units_end"] * (final_mark - episode["average_cost_end"])
        self.assertAlmostEqual(episode["unrealized_pnl"], expected)
        self.assertAlmostEqual(episode["pnl"], self.book.equity[-1] - START_CAPITAL, places=6)
        self.assertAlmostEqual(episode["return"], episode["pnl"] / START_CAPITAL)
        self.assertLess(episode["avg_exposure"], episode["max_exposure"])


class PartialExitTests(unittest.TestCase):
    def test_multiple_reductions_then_full_liquidation(self):
        book = _book(
            LINEAR,
            {
                "a": ["buy", "", "sell", "", "", ""],
                "b": ["buy", "", "", "sell", "", ""],
                "c": ["buy", "", "", "", "sell", ""],
            },
        )
        (episode,) = _episodes(book)
        events = episode["events"]
        self.assertEqual([e["event"] for e in events], [ENTER, REDUCE, REDUCE, EXIT])
        self.assertEqual(episode["status"], "Closed")
        self.assertEqual(episode["rebalance_count"], 4)
        self.assertAlmostEqual(events[0]["exposure_after"], 1.0)
        self.assertAlmostEqual(events[1]["exposure_after"], 2 / 3)
        self.assertAlmostEqual(events[2]["exposure_after"], 1 / 3)
        self.assertEqual(events[3]["position_after"], 0.0)
        for reduce in events[1:]:
            self.assertLess(reduce["trade_value"], 0)
            self.assertLess(reduce["units_delta"], 0)
        self.assertAlmostEqual(episode["pnl"], book.equity[4] - START_CAPITAL, places=6)
        self.assertAlmostEqual(
            episode["realized_pnl"], sum(e["realized_pnl"] for e in events), places=9
        )
        self.assertEqual(episode["unrealized_pnl"], 0.0)
        self.assertAlmostEqual(episode["final_simulated_price"], _price_path()[4])
        self.assertAlmostEqual(episode["max_exposure"], 1.0)
        self.assertLess(episode["avg_exposure"], 1.0 + 1e-9)

    def test_partial_reductions_then_additional_purchases(self):
        book = _book(
            LINEAR,
            {
                "a": ["buy", "sell", "buy", "", "", ""],
                "b": ["buy", "", "", "", "", ""],
            },
        )
        (episode,) = _episodes(book)
        events = episode["events"]
        self.assertEqual([e["event"] for e in events], [ENTER, REDUCE, ADD])
        self.assertEqual(episode["status"], "Open")
        reduce = events[1]
        sold = -reduce["units_delta"]
        self.assertAlmostEqual(
            reduce["realized_pnl"], sold * (reduce["simulated_price"] - events[0]["simulated_price"])
        )
        add = events[2]
        self.assertGreater(add["units_delta"], 0)
        # The add is bought at a new price, so the average cost moves off the first price.
        self.assertNotAlmostEqual(add["average_cost_after"], events[0]["simulated_price"])
        self.assertAlmostEqual(episode["pnl"], book.equity[-1] - START_CAPITAL, places=6)


class SimultaneousChangesTests(unittest.TestCase):
    def test_every_simultaneous_change_is_listed(self):
        book = _book(
            LINEAR,
            {
                "a": ["buy", "", "sell", "", "", ""],
                "b": ["", "", "buy", "", "", ""],
                "c": ["", "", "buy", "", "", ""],
            },
        )
        (episode,) = _episodes(book)
        events = episode["events"]
        self.assertEqual([e["event"] for e in events], [ENTER, ADD])
        simultaneous = events[1]
        self.assertEqual(
            [(c["strategy_name"], c["change"]) for c in simultaneous["signal_changes"]],
            [("A", "exited"), ("B", "entered"), ("C", "entered")],
        )
        self.assertEqual(simultaneous["active_signals"], 2)
        self.assertEqual(simultaneous["eligible_signals"], 3)
        self.assertEqual(simultaneous["active_strategies"], ["B", "C"])
        self.assertAlmostEqual(simultaneous["exposure_after"], 2 / 3)
        self.assertEqual(
            [(c["strategy_name"], c["change"]) for c in events[0]["signal_changes"]],
            [("A", "entered")],
        )

    def test_a_swap_that_keeps_the_count_still_rebalances_a_drifted_book(self):
        # A holds through a rally, then A exits and B enters on the same bar. The count is
        # unchanged, but the position has grown past its share, so it is trimmed back.
        book = _book(
            LINEAR,
            {
                "a": ["buy", "", "", "sell", "", ""],
                "b": ["", "", "", "buy", "", ""],
                "c": NONE,
            },
        )
        (episode,) = _episodes(book)
        events = episode["events"]
        self.assertEqual([e["event"] for e in events], [ENTER, REDUCE])
        trim = events[1]
        self.assertGreater(trim["exposure_before"], trim["exposure_after"])
        self.assertAlmostEqual(trim["exposure_after"], 1 / 3)
        self.assertEqual(
            [(c["strategy_name"], c["change"]) for c in trim["signal_changes"]],
            [("A", "exited"), ("B", "entered")],
        )
        self.assertEqual(trim["active_signals"], 1)


class NoTransactionTests(unittest.TestCase):
    def test_overlapping_signal_under_first_signal_only_is_not_a_trade(self):
        strategies = [_strategy("a"), _strategy("b")]
        signals = {
            "a": _signals("a", DATES, ["buy", "", "", "", "sell", ""]),
            "b": _signals("b", DATES, ["", "buy", "sell", "", "", ""]),
        }
        track, close = _market()
        frame = simulate_portfolio_overlay(
            strategies, signals, overlap_mode="first_signal_only", global_proxy=None,
            track_changes=track, close_series=close, strategy_order=["a", "b"],
        )
        replay = replay_overlay_transactions(
            frame, strategies, signals, global_proxy=None
        )
        (episode,) = _episodes(replay)
        self.assertEqual([e["event"] for e in episode["events"]], [ENTER, EXIT])
        self.assertEqual(
            [e["date"] for e in episode["events"]],
            [DATES[0].strftime("%Y-%m-%d"), DATES[4].strftime("%Y-%m-%d")],
        )
        # B entered and exited while A held the position: two signal changes, no trades.
        self.assertEqual(replay.signal_changes_without_transaction, 2)
        # The exit is attributed to A, whose exit closed the book.
        self.assertEqual(
            [(c["strategy_name"], c["change"]) for c in episode["events"][1]["signal_changes"]],
            [("A", "exited")],
        )

    def test_swap_that_leaves_the_book_on_target_trades_nothing(self):
        flat_prices = [0.0] * 6
        book = _book(
            LINEAR,
            {
                "a": ["buy", "sell", "", "", "", ""],
                "b": ["", "buy", "", "", "", ""],
                "c": ["", "", "", "", "", ""],
            },
            changes=flat_prices,
        )
        # Bar 0: A enters (1/3). Bar 1: A exits and B enters, same count, price unmoved.
        self.assertEqual(book.active_count.tolist()[:3], [1, 1, 1])
        (episode,) = _episodes(book)
        self.assertEqual([e["event"] for e in episode["events"]], [ENTER])
        self.assertEqual(book.signal_changes_without_transaction, 1)


class FeesAndSlippageTests(unittest.TestCase):
    def test_fees_are_charged_on_each_fill_and_reduce_episode_pnl(self):
        free = _book(LINEAR, {"a": ["buy", "", "sell", "", "", ""]})
        costly = _book(LINEAR, {"a": ["buy", "", "sell", "", "", ""]}, cost_bps=25)
        (free_episode,) = _episodes(free)
        (episode,) = _episodes(costly)
        fees = [e["fees"] for e in episode["events"]]
        self.assertTrue(all(fee > 0 for fee in fees))
        for event in episode["events"]:
            self.assertAlmostEqual(event["fees"], abs(event["trade_value"]) * 0.0025, places=9)
        self.assertAlmostEqual(episode["fees"], costly.total_costs[-1])
        self.assertLess(episode["pnl"], free_episode["pnl"])
        self.assertAlmostEqual(episode["pnl"], costly.equity[-1] - START_CAPITAL, places=6)
        self.assertAlmostEqual(
            episode["pnl"], episode["realized_pnl"] - episode["fees"], places=6
        )
        # The denominator is equity *before* the first entry, so entry fees do not shrink it.
        self.assertAlmostEqual(episode["equity_before_entry"], START_CAPITAL)
        self.assertAlmostEqual(episode["return"], episode["pnl"] / START_CAPITAL, places=12)
        self.assertLess(episode["events"][0]["equity_after"], episode["events"][0]["equity_before"])

    def test_slippage_moves_execution_price_against_the_trade(self):
        ledger = PortfolioLedger(cash=1000.0, cost_bps=10, slippage_bps=20)
        (buy,) = ledger.rebalance({"X": 1.0}, {"X": 10.0}).fills
        self.assertAlmostEqual(buy.mark_price, 10.0)
        self.assertAlmostEqual(buy.price, 10.0 * 1.002)
        self.assertAlmostEqual(buy.slippage_cost, buy.shares * (buy.price - 10.0))
        self.assertGreater(ledger.cash, -1e-9)
        (sell,) = ledger.rebalance({}, {"X": 10.0}).fills
        self.assertAlmostEqual(sell.price, 10.0 * 0.998)
        self.assertLess(sell.shares, 0)
        self.assertAlmostEqual(sell.realized_pnl, -sell.shares * (sell.price - buy.price))
        # Round trip at an unchanged mark loses slippage and fees, nothing else.
        self.assertAlmostEqual(
            1000.0 - ledger.cash,
            buy.slippage_cost + sell.slippage_cost + buy.cost + sell.cost,
            places=6,
        )

    def test_engine_episode_reconciles_under_slippage_and_fees(self):
        signal = _signals("a", DATES, ["buy", "", "sell", "", "", ""])
        active, eligible = effective_state(
            signal.long_trade_in, signal.hold_long, signal.long_trade_out,
            first_eligible=0, last_available=5,
        )
        track, close = _market()
        book = simulate_allocation(
            dates=DATES,
            active=active[None, :],
            eligible=eligible[None, :],
            pnl_symbols=["AAA"],
            track_changes=track,
            close_series=close,
            model=AllInModel(),
            start_capital=START_CAPITAL,
            cost_bps=10,
            slippage_bps=15,
        )
        (episode,) = _episodes(book)
        for event in episode["events"]:
            self.assertNotEqual(event["simulated_price"], event["mark_price"])
        self.assertGreater(episode["slippage_cost"], 0)
        self.assertAlmostEqual(episode["pnl"], book.equity[-1] - START_CAPITAL, places=6)
        # Equity is marked at the mark, so the cash left after exit explains it exactly.
        self.assertAlmostEqual(book.cash[-1], book.equity[-1], places=6)


class RollingEpisodePnlTests(unittest.TestCase):
    def test_second_line_already_has_a_running_pnl(self):
        book = _book(
            LINEAR,
            {
                "a": ["buy", "", "", "", "", ""],
                "b": ["", "buy", "", "", "", ""],
                "c": ["", "", "buy", "", "", ""],
            },
        )
        (episode,) = _episodes(book)
        first, second, third = episode["events"]
        prices = _price_path()
        # Enter at 100 with 1/3 of equity; the Add at 110 sees the first lot up 10%.
        self.assertAlmostEqual(first["episode_pnl_to_date"], 0.0)
        lot = START_CAPITAL / 3 / prices[0]
        self.assertAlmostEqual(second["episode_pnl_to_date"], lot * (prices[1] - prices[0]), places=6)
        self.assertAlmostEqual(
            second["episode_return_to_date"], second["episode_pnl_to_date"] / START_CAPITAL
        )
        self.assertNotEqual(second["episode_pnl_to_date"], 0.0)
        # Each line is the equity gained since before entry, as of that event.
        for event in (first, second, third):
            self.assertAlmostEqual(
                event["episode_pnl_to_date"], event["equity_after"] - START_CAPITAL, places=6
            )

    def test_closed_episode_ends_on_its_total(self):
        book = _book(
            LINEAR,
            {
                "a": ["buy", "", "sell", "", "", ""],
                "b": ["buy", "", "", "sell", "", ""],
            },
            cost_bps=10,
        )
        (episode,) = _episodes(book)
        self.assertEqual(episode["events"][-1]["event"], EXIT)
        self.assertAlmostEqual(episode["events"][-1]["episode_pnl_to_date"], episode["pnl"], places=6)
        self.assertAlmostEqual(
            episode["events"][-1]["episode_return_to_date"], episode["return"], places=12
        )
        # Fees are inside it: the Enter line is already slightly negative.
        self.assertLess(episode["events"][0]["episode_pnl_to_date"], 0.0)

    def test_every_line_ties_to_equity_across_random_books(self):
        for seed in range(6):
            book = ReconciliationTests()._random_book(seed, LINEAR, 8.0)
            for episode in _episodes(book):
                base = episode["equity_before_entry"]
                for event in episode["events"]:
                    self.assertAlmostEqual(
                        event["episode_pnl_to_date"],
                        event["equity_after"] - base,
                        delta=1e-9 * max(1.0, abs(base)),
                    )


class LiquidationDustTests(unittest.TestCase):
    """Regression: a full liquidation must leave exactly nothing, at any position size."""

    def test_full_liquidation_leaves_no_dust(self):
        rng = random.Random(7)
        for _ in range(3000):
            cash = rng.uniform(1e3, 5e8)
            buy_price = rng.uniform(1.0, 5e4)
            sell_price = buy_price * rng.uniform(0.5, 2.0)
            ledger = PortfolioLedger(cash=cash, cost_bps=rng.choice([0.0, 2.0]))
            ledger.rebalance({"X": rng.uniform(0.05, 1.0)}, {"X": buy_price})
            (fill,) = ledger.rebalance({}, {"X": sell_price}).fills
            self.assertEqual(ledger.positions, {})
            self.assertEqual(fill.units_after, 0.0)
            self.assertEqual(classify_event(fill.units_before, fill.units_after), EXIT)

    def test_a_reentry_after_liquidation_starts_a_new_episode(self):
        # Large position sizes are where the dust used to survive.
        dates = pd.bdate_range("2024-01-01", periods=6)
        book = _book(
            LINEAR,
            {"a": ["buy", "sell", "buy", "sell", "", ""]},
            changes=[0.0, 0.013, -0.007, 0.021, 0.0, 0.0],
            dates=dates,
        )
        episodes = _episodes(book)
        self.assertEqual(len(episodes), 2)
        self.assertEqual([e["status"] for e in episodes], ["Closed", "Closed"])
        for episode in episodes:
            self.assertEqual(
                [event["event"] for event in episode["events"]], [ENTER, EXIT]
            )


class CalendarTests(unittest.TestCase):
    def test_days_are_sessions_not_calendar_days(self):
        # Thursday close entry, then Independence Day (closed) and a weekend.
        dates = pd.to_datetime(
            ["2024-07-03", "2024-07-05", "2024-07-08", "2024-07-09", "2024-07-10"]
        )
        book = _book(
            AllInModel(),
            {"a": ["buy", "", "", "sell", ""]},
            changes=[0.0, 0.01, 0.01, 0.01, 0.01],
            dates=dates,
        )
        (episode,) = _episodes(book)
        self.assertEqual(episode["start"], "2024-07-03")
        self.assertEqual(episode["end"], "2024-07-09")
        calendar_days = (pd.Timestamp(episode["end"]) - pd.Timestamp(episode["start"])).days
        self.assertEqual(calendar_days, 6)
        self.assertEqual(episode["days"], 3)  # Fri, Mon, Tue


class ReconciliationTests(unittest.TestCase):
    def _random_book(self, seed, model, cost_bps):
        rng = random.Random(seed)
        n = 300
        dates = pd.bdate_range("2021-01-01", periods=n)
        returns = [0.0] + [rng.gauss(0.0005, 0.02) for _ in range(n - 1)]
        events = {}
        for name in ("a", "b", "c"):
            holding, row = False, []
            for _ in range(n):
                if holding:
                    event = "sell" if rng.random() < 0.25 else ""
                else:
                    event = "buy" if rng.random() < 0.12 else ""
                holding = (holding and event != "sell") or event == "buy"
                row.append(event)
            events[name] = row
        return _book(model, events, changes=returns, dates=dates, cost_bps=cost_bps)

    def test_episodes_events_and_equity_tie_out(self):
        for seed in range(8):
            for model in (LINEAR, proportional_model("early_conviction"), AllInModel()):
                for cost_bps in (0.0, 12.0):
                    with self.subTest(seed=seed, model=model.id, cost=cost_bps):
                        book = self._random_book(seed, model, cost_bps)
                        payload = _positions_payload(book)
                        rec = payload["reconciliation"]
                        scale = max(1.0, abs(rec["equity_change"]))
                        self.assertLess(abs(rec["episode_vs_equity_difference"]) / scale, 1e-9)
                        self.assertLess(abs(rec["episode_vs_event_difference"]) / scale, 1e-9)
                        self.assertLess(abs(rec["cash_difference"]) / scale, 1e-9)
                        for episode in payload["episodes"]:
                            events = episode["events"]
                            self.assertEqual(events[0]["event"], ENTER)
                            self.assertEqual(episode["rebalance_count"], len(events))
                            self.assertEqual(episode["equity_before_entry"], events[0]["equity_before"])
                            self.assertAlmostEqual(
                                episode["return"],
                                episode["pnl"] / events[0]["equity_before"],
                                places=12,
                            )
                            if episode["status"] == "Closed":
                                self.assertEqual(events[-1]["event"], EXIT)
                                self.assertEqual(episode["unrealized_pnl"], 0.0)
                                # Single instrument: the book is flat before entry and after
                                # exit, so episode P&L is exactly the equity change across it.
                                change = events[-1]["equity_after"] - events[0]["equity_before"]
                                self.assertAlmostEqual(
                                    episode["pnl"], change, delta=1e-9 * max(1.0, abs(change))
                                )
                            for before, after in zip(events, events[1:]):
                                self.assertAlmostEqual(
                                    before["position_after"], after["units_before"], places=9
                                )

    def test_units_and_cash_follow_the_transactions(self):
        book = self._random_book(3, LINEAR, 5.0)
        cash = START_CAPITAL
        units = 0.0
        for record in book.transactions:
            cash -= record.trade_value + record.fee
            units += record.units_delta
            self.assertAlmostEqual(record.cash_after, cash, delta=1e-6 * START_CAPITAL)
            self.assertAlmostEqual(record.units_after, units, delta=1e-6 * max(1.0, abs(units)))
        self.assertAlmostEqual(cash, book.cash[-1], delta=1e-6 * START_CAPITAL)


class AllInHistoryTests(unittest.TestCase):
    """The same episode model reproduces the legacy all-in trades."""

    def _legacy(self, seed, overlap_mode, mixed=False, cost_bps=0.0):
        rng = random.Random(seed)
        n = 400
        dates = pd.bdate_range("2020-01-01", periods=n)
        symbols = ["AAA", "BBB"] if mixed else ["AAA"]
        track = {
            sym: pd.Series([0.0] + [rng.gauss(0.0006, 0.02) for _ in range(n - 1)], index=dates)
            for sym in symbols
        }
        close = {
            sym: pd.Series(100 * np.cumprod(1 + track[sym].to_numpy() / 3), index=dates)
            for sym in symbols
        }
        strategies, signals = [], {}
        for k in range(3):
            symbol = symbols[k % len(symbols)]
            strategy = _strategy(f"s{k}", symbol)
            strategies.append(strategy)
            holding, row = False, []
            for _ in range(n):
                if holding:
                    event = "sell" if rng.random() < 0.3 else ""
                else:
                    event = "buy" if rng.random() < 0.1 else ""
                holding = (holding and event != "sell") or event == "buy"
                row.append(event)
            signals[strategy.id] = _signals(strategy.id, dates, row, symbol=symbol)
        frame = simulate_portfolio_overlay(
            strategies, signals, overlap_mode=overlap_mode, global_proxy=None,
            track_changes=track, close_series=close, strategy_order=[s.id for s in strategies],
        )
        frame = charge_overlay_costs(frame, cost_bps)
        return strategies, signals, frame

    def test_episodes_match_legacy_trades(self):
        for overlap in ("hold_until_all_exit", "first_signal_only"):
            for mixed in (False, True):
                for seed in range(6):
                    with self.subTest(overlap=overlap, mixed=mixed, seed=seed):
                        strategies, signals, frame = self._legacy(seed, overlap, mixed)
                        replay = replay_overlay_transactions(
                            frame, strategies, signals, global_proxy=None
                        )
                        episodes = _episodes(replay)

                        dates = pd.DatetimeIndex(frame["Date"])
                        enters = dates[frame["LongTradeIn"].to_numpy()]
                        exits = dates[frame["LongTradeOut"].to_numpy()]
                        self.assertEqual(len(episodes), len(enters))
                        self.assertEqual(
                            [e["start"] for e in episodes],
                            [d.strftime("%Y-%m-%d") for d in enters],
                        )
                        closed = [e for e in episodes if e["status"] == "Closed"]
                        self.assertEqual(
                            [e["end"] for e in closed], [d.strftime("%Y-%m-%d") for d in exits]
                        )
                        equity = frame.set_index("Date")["RollingPnL"]
                        for episode in closed:
                            self.assertEqual(
                                [e["event"] for e in episode["events"]], [ENTER, EXIT]
                            )
                            expected = (
                                equity[pd.Timestamp(episode["end"])]
                                / equity[pd.Timestamp(episode["start"])]
                                - 1
                            )
                            self.assertAlmostEqual(episode["return"], expected, places=9)
                        # Same equity curve, to numerical tolerance.
                        np.testing.assert_allclose(
                            replay.equity, frame["RollingPnL"].to_numpy(), rtol=1e-9
                        )

    def test_replay_matches_a_costed_overlay(self):
        for seed in range(4):
            strategies, signals, frame = self._legacy(seed, "hold_until_all_exit", cost_bps=15.0)
            replay = replay_overlay_transactions(
                frame, strategies, signals, global_proxy=None, cost_bps=15.0
            )
            np.testing.assert_allclose(replay.equity, frame["RollingPnL"].to_numpy(), rtol=1e-9)
            np.testing.assert_allclose(
                replay.cost, frame["TransactionCost"].to_numpy(), rtol=1e-9, atol=1e-9
            )
            payload = _positions_payload(replay, comparison_equity=frame["RollingPnL"].to_numpy())
            self.assertLess(
                payload["reconciliation"]["ledger_vs_simulation_max_relative_difference"], 1e-9
            )

    def test_triggers_name_the_strategies_that_moved(self):
        strategies, signals, frame = self._legacy(2, "hold_until_all_exit")
        replay = replay_overlay_transactions(frame, strategies, signals, global_proxy=None)
        episodes = _episodes(replay)
        self.assertTrue(episodes)
        first = episodes[0]["events"][0]
        self.assertTrue(first["signal_changes"])
        self.assertTrue(all(c["change"] == "entered" for c in first["signal_changes"]))


def _patched_request(test):
    strategies = [_strategy("a"), _strategy("b")]
    signals = {
        "a": _signals("a", DATES, ["buy", "", "", "sell", "", ""]),
        "b": _signals("b", DATES, ["", "buy", "", "", "sell", ""]),
    }
    track, close = _market()
    by_id = {s.id: s for s in strategies}
    market = (
        pd.DataFrame(),
        {"AAA": pd.DataFrame({"Date": DATES, "Close": np.linspace(100, 110, 6)})},
        track,
        close,
    )
    for item in (
        patch("api.portfolio_service.get_strategy_by_id", side_effect=lambda sid: by_id[sid]),
        patch("api.portfolio_service._load_portfolio_market_data", return_value=market),
        patch(
            "api.portfolio_service._load_strategy_signals",
            side_effect=lambda strategy, **_: signals[strategy.id],
        ),
    ):
        item.start()
        test.addCleanup(item.stop)
    return SimpleNamespace(
        strategy_ids=["a", "b"], overlap_mode="hold_until_all_exit", proxy_symbol=None, years=25
    )


class PositionsPayloadTests(unittest.TestCase):
    def test_all_in_payload_has_episodes_that_tie_to_the_summary(self):
        payload = simulate_portfolio(_patched_request(self))
        positions = payload["positions"]
        (episode,) = positions["episodes"]
        self.assertEqual([e["event"] for e in episode["events"]], [ENTER, EXIT])
        self.assertAlmostEqual(
            START_CAPITAL + episode["pnl"], payload["summary"]["rolling_pnl"], places=4
        )
        rec = positions["reconciliation"]
        self.assertLess(rec["ledger_vs_simulation_max_relative_difference"], 1e-9)
        self.assertAlmostEqual(rec["episode_vs_equity_difference"], 0.0, places=6)
        self.assertIn("underlying", positions["price_basis"]["description"])
        # The legacy trade list is still there for the panels that use it.
        self.assertEqual(len(payload["trades"]), 1)
        self.assertEqual(payload["trades"][0]["entry_date"], episode["start"])
        self.assertEqual(payload["trades"][0]["exit_date"], episode["end"])

    def test_all_in_episode_percent_is_exactly_the_old_trade_percent(self):
        payload = simulate_portfolio(_patched_request(self))
        (episode,) = payload["positions"]["episodes"]
        (trade,) = payload["trades"]
        self.assertEqual(episode["return"], trade["trade_pnl"])
        # The closing line agrees with it, so the table never contradicts itself.
        self.assertEqual(episode["events"][-1]["episode_return_to_date"], trade["trade_pnl"])
        # The ledger's own figure is still there, and is a different (compounded) number.
        self.assertAlmostEqual(
            episode["return_on_equity"], episode["pnl"] / episode["equity_before_entry"], places=12
        )
        self.assertNotAlmostEqual(episode["return"], episode["return_on_equity"], places=6)

    def test_proportional_episode_percent_is_the_ledger_figure(self):
        request = _patched_request(self)
        request.allocation_model = "proportional"
        request.proportional_mode = "linear"
        payload = simulate_portfolio(request)
        (episode,) = payload["positions"]["episodes"]
        self.assertEqual(episode["return"], episode["return_on_equity"])

    def test_proportional_payload_shows_partial_events(self):
        request = _patched_request(self)
        request.allocation_model = "proportional"
        request.proportional_mode = "linear"
        payload = simulate_portfolio(request)
        (episode,) = payload["positions"]["episodes"]
        self.assertEqual(
            [e["event"] for e in episode["events"]], [ENTER, ADD, REDUCE, EXIT]
        )
        self.assertAlmostEqual(
            START_CAPITAL + episode["pnl"], payload["summary"]["rolling_pnl"], places=4
        )
        add = episode["events"][1]
        self.assertAlmostEqual(add["exposure_after"], 1.0)
        self.assertEqual(add["active_signals"], 2)
        self.assertEqual(add["eligible_signals"], 2)

    def test_payload_is_valid_json(self):
        import json

        payload = simulate_portfolio(_patched_request(self))
        json.dumps(payload["positions"], allow_nan=False)


if __name__ == "__main__":
    unittest.main()
