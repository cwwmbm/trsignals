import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from api.allocation import (
    AllInModel,
    SignalState,
    clamp_exposure,
    comparison_models,
    proportional_model,
    resolve_allocation_model,
)
from api.allocation_analytics import (
    allocation_attribution,
    arrays_from_overlay,
    arrays_from_simulation,
    exposure_analytics,
)
from api.allocation_engine import effective_state, simulate_allocation
from api.portfolio_service import (
    START_CAPITAL,
    StrategySignals,
    allocation_settings,
    simulate_portfolio,
    simulate_portfolio_allocation,
    simulate_portfolio_allocation_comparison,
    simulate_portfolio_frame,
    simulate_portfolio_overlay,
)
from api.rebalancer import PortfolioLedger


def _strategy(strategy_id, symbol="AAA"):
    return SimpleNamespace(
        id=strategy_id,
        name=f"Strategy {strategy_id}",
        symbol=symbol,
        direction="long",
        hold_days=2,
        profit=1,
        confirm_symbols=[],
        proxy_symbol=None,
    )


def _flags(events):
    """Entry/hold/exit flags from 'buy' / 'sell' / '' events, as the backtester emits them."""
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
        name=f"Strategy {strategy_id}",
        symbol=symbol,
        dates=pd.DatetimeIndex(dates),
        long_trade_in=trade_in,
        hold_long=hold,
        long_trade_out=trade_out,
        entry_ready=None if entry_ready is None else np.asarray(entry_ready, dtype=bool),
    )


DATES = pd.bdate_range("2024-01-01", periods=6)
# Per-bar leveraged returns. The first bar never earns anything.
CHANGES = [0.0, 0.10, 0.20, -0.10, 0.05, 0.02]


def _market(changes=CHANGES, dates=DATES, symbol="AAA"):
    series = pd.Series(changes, index=dates)
    return {symbol: series}, {symbol: pd.Series(np.linspace(100, 110, len(dates)), index=dates)}


def _two_strategy_case():
    """A holds bars 1-3 (exits on 3), B holds bars 2-4 (exits on 4)."""
    a = _signals("a", DATES, ["buy", "", "", "sell", "", ""])
    b = _signals("b", DATES, ["", "buy", "", "", "sell", ""])
    return [_strategy("a"), _strategy("b")], {"a": a, "b": b}


def _run(model, strategies, signals, *, cost_bps=0.0, changes=CHANGES):
    track, close = _market(changes)
    return simulate_portfolio_allocation(
        strategies,
        signals,
        model=model,
        cost_bps=cost_bps,
        global_proxy=None,
        track_changes=track,
        close_series=close,
    )


class AllocationModelTests(unittest.TestCase):
    def test_all_in_is_binary(self):
        model = AllInModel()
        self.assertEqual(model.target_exposure(SignalState(0, 4)), 0.0)
        self.assertEqual(model.target_exposure(SignalState(1, 4)), 1.0)
        self.assertEqual(model.target_exposure(SignalState(4, 4)), 1.0)

    def test_proportional_curves(self):
        linear = proportional_model("linear")
        early = proportional_model("early_conviction")
        confirm = proportional_model("confirmation_required")
        state = SignalState(1, 4)
        self.assertAlmostEqual(linear.target_exposure(state), 0.25)
        self.assertAlmostEqual(early.target_exposure(state), 0.5)
        self.assertAlmostEqual(confirm.target_exposure(state), 0.0625)

    def test_zero_when_none_active_and_full_when_all_active(self):
        for mode in ("linear", "early_conviction", "confirmation_required"):
            model = proportional_model(mode)
            self.assertEqual(model.target_exposure(SignalState(0, 5)), 0.0)
            self.assertEqual(model.target_exposure(SignalState(5, 5)), 1.0)

    def test_exposure_is_bounded_and_monotonic(self):
        for mode in ("linear", "early_conviction", "confirmation_required"):
            model = proportional_model(mode)
            values = [model.target_exposure(SignalState(k, 7)) for k in range(8)]
            self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
            self.assertEqual(values, sorted(values))

    def test_no_eligible_strategies_means_no_exposure(self):
        self.assertEqual(proportional_model("linear").target_exposure(SignalState(0, 0)), 0.0)

    def test_state_rejects_impossible_counts(self):
        with self.assertRaises(ValueError):
            SignalState(3, 2)
        with self.assertRaises(ValueError):
            SignalState(-1, 2)

    def test_clamp_treats_non_finite_as_flat(self):
        self.assertEqual(clamp_exposure(float("nan")), 0.0)
        self.assertEqual(clamp_exposure(1.7), 1.0)
        self.assertEqual(clamp_exposure(-0.2), 0.0)

    def test_resolution_defaults_to_all_in(self):
        self.assertEqual(resolve_allocation_model().id, "all_in")
        self.assertEqual(resolve_allocation_model(None, "linear").id, "all_in")
        self.assertEqual(
            resolve_allocation_model("proportional", "early_conviction").id,
            "proportional_early_conviction",
        )
        with self.assertRaises(ValueError):
            resolve_allocation_model("kelly")
        with self.assertRaises(ValueError):
            resolve_allocation_model("proportional", "cubic")

    def test_comparison_lists_every_builtin_model(self):
        self.assertEqual(
            [model.id for model in comparison_models()],
            [
                "all_in",
                "proportional_linear",
                "proportional_early_conviction",
                "proportional_confirmation_required",
                "tiered",
            ],
        )

    def test_custom_model_plugs_in_without_engine_changes(self):
        class HalfWhenAny:
            id = "half"
            label = "Half"

            def target_exposure(self, state):
                return 0.5 if state.active_count else 0.0

        strategies, signals = _two_strategy_case()
        sim = _run(HalfWhenAny(), strategies, signals)
        self.assertAlmostEqual(sim.post_weight[1], 0.5, places=9)


class RebalancerTests(unittest.TestCase):
    def test_buys_to_target_weight_and_tracks_books(self):
        ledger = PortfolioLedger(cash=1000.0)
        result = ledger.rebalance({"X": 0.5}, {"X": 10.0})
        self.assertTrue(result.traded)
        self.assertAlmostEqual(ledger.shares("X"), 50.0)
        self.assertAlmostEqual(ledger.cash, 500.0)
        self.assertAlmostEqual(ledger.invested_capital({"X": 10.0}), 500.0)
        self.assertAlmostEqual(ledger.equity({"X": 12.0}), 1100.0)
        self.assertAlmostEqual(ledger.unrealized_pnl({"X": 12.0}), 100.0)

    def test_partial_exit_realizes_pnl_on_average_cost(self):
        ledger = PortfolioLedger(cash=1000.0)
        ledger.rebalance({"X": 1.0}, {"X": 10.0})  # 100 shares at 10
        ledger.rebalance({"X": 0.5}, {"X": 20.0})  # equity 2000, hold 1000 -> sell 50
        self.assertAlmostEqual(ledger.shares("X"), 50.0)
        self.assertAlmostEqual(ledger.realized_pnl, 500.0)
        self.assertAlmostEqual(ledger.cash, 1000.0)
        self.assertAlmostEqual(ledger.unrealized_pnl({"X": 20.0}), 500.0)

    def test_full_liquidation_when_symbol_absent_from_targets(self):
        ledger = PortfolioLedger(cash=1000.0)
        ledger.rebalance({"X": 1.0}, {"X": 10.0})
        ledger.rebalance({}, {"X": 15.0})
        self.assertEqual(ledger.positions, {})
        self.assertAlmostEqual(ledger.cash, 1500.0)
        self.assertAlmostEqual(ledger.realized_pnl, 500.0)

    def test_already_at_target_does_not_trade(self):
        ledger = PortfolioLedger(cash=1000.0)
        ledger.rebalance({"X": 1.0}, {"X": 10.0})
        result = ledger.rebalance({"X": 1.0}, {"X": 10.0})
        self.assertFalse(result.traded)
        self.assertEqual(ledger.rebalance_count, 1)

    def test_costs_are_charged_on_traded_notional_and_cash_stays_non_negative(self):
        ledger = PortfolioLedger(cash=10_000.0, cost_bps=10.0)
        result = ledger.rebalance({"X": 1.0}, {"X": 50.0})
        self.assertGreater(ledger.cash, -1e-9)
        self.assertAlmostEqual(result.cost, result.traded_notional * 0.001, places=9)
        self.assertAlmostEqual(ledger.equity({"X": 50.0}), 10_000.0 - result.cost, places=6)
        self.assertAlmostEqual(ledger.total_costs, result.cost)

    def test_splits_across_symbols(self):
        ledger = PortfolioLedger(cash=1000.0)
        ledger.rebalance({"X": 0.25, "Y": 0.25}, {"X": 10.0, "Y": 50.0})
        self.assertAlmostEqual(ledger.shares("X"), 25.0)
        self.assertAlmostEqual(ledger.shares("Y"), 5.0)
        weights = ledger.weights({"X": 10.0, "Y": 50.0})
        self.assertAlmostEqual(weights["X"], 0.25)

    def test_rejects_leverage_and_shorting(self):
        ledger = PortfolioLedger(cash=1000.0)
        with self.assertRaises(ValueError):
            ledger.rebalance({"X": 1.2}, {"X": 10.0})
        with self.assertRaises(ValueError):
            ledger.rebalance({"X": -0.1}, {"X": 10.0})
        with self.assertRaises(ValueError):
            PortfolioLedger(cash=1.0, cost_bps=-1.0)


class EffectiveStateTests(unittest.TestCase):
    def test_active_means_holding_into_next_bar(self):
        trade_in, hold, trade_out = _flags(["buy", "", "sell", ""])
        active, eligible = effective_state(
            trade_in, hold, trade_out, first_eligible=0, last_available=3
        )
        self.assertEqual(active.tolist(), [True, True, False, False])
        self.assertEqual(eligible.tolist(), [True] * 4)

    def test_ineligible_bars_are_never_active(self):
        trade_in, hold, trade_out = _flags(["buy", "", "", ""])
        active, eligible = effective_state(
            trade_in, hold, trade_out, first_eligible=2, last_available=3
        )
        self.assertEqual(eligible.tolist(), [False, False, True, True])
        self.assertEqual(active.tolist(), [False, False, True, True])

    def test_data_ending_early_stops_the_strategy(self):
        trade_in, hold, trade_out = _flags(["buy", "", "", ""])
        active, eligible = effective_state(
            trade_in, hold, trade_out, first_eligible=0, last_available=1
        )
        self.assertEqual(eligible.tolist(), [True, True, False, False])
        self.assertEqual(active.tolist(), [True, True, False, False])

    def test_never_eligible(self):
        trade_in, hold, trade_out = _flags(["buy", "", ""])
        active, eligible = effective_state(
            trade_in, hold, trade_out, first_eligible=None, last_available=None
        )
        self.assertFalse(active.any())
        self.assertFalse(eligible.any())


class ProportionalSimulationTests(unittest.TestCase):
    def setUp(self):
        self.strategies, self.signals = _two_strategy_case()

    def test_linear_matches_hand_computed_book(self):
        sim = _run(proportional_model("linear"), self.strategies, self.signals)
        # i0: A enters, 50% of 15000. i1: B enters, 100%. i3: A exits, back to 50%. i4: flat.
        self.assertEqual(sim.active_count.tolist(), [1, 2, 2, 1, 0, 0])
        np.testing.assert_allclose(sim.post_weight, [0.5, 1.0, 1.0, 0.5, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            sim.equity,
            [15000.0, 15750.0, 18900.0, 17010.0, 17435.25, 17435.25],
            rtol=1e-12,
        )
        np.testing.assert_allclose(sim.cash[-1], 17435.25, rtol=1e-12)
        self.assertEqual(sim.rebalanced.tolist(), [True, True, False, True, True, False])

    def test_confirmation_required_matches_hand_computed_book(self):
        sim = _run(proportional_model("confirmation_required"), self.strategies, self.signals)
        np.testing.assert_allclose(sim.post_weight, [0.25, 1.0, 1.0, 0.25, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(sim.equity[-1], 16812.5625, rtol=1e-12)

    def test_early_conviction_sits_between_linear_and_all_in(self):
        results = {
            mode: _run(proportional_model(mode), self.strategies, self.signals).equity[-1]
            for mode in ("confirmation_required", "linear", "early_conviction")
        }
        all_in = _run(AllInModel(), self.strategies, self.signals).equity[-1]
        self.assertLess(results["confirmation_required"], results["linear"])
        self.assertLess(results["linear"], results["early_conviction"])
        self.assertLess(results["early_conviction"], all_in)

    def test_rebalances_only_when_the_state_changes(self):
        sim = _run(proportional_model("linear"), self.strategies, self.signals)
        # Bar 2 has the same active set as bar 1, so positions drift untouched.
        self.assertFalse(sim.rebalanced[2])
        self.assertTrue(np.isnan(sim.target_exposure[2]))
        self.assertAlmostEqual(sim.target_exposure[3], 0.5)

    def test_ledger_identities_hold_every_bar(self):
        sim = _run(proportional_model("early_conviction"), self.strategies, self.signals, cost_bps=5)
        np.testing.assert_allclose(sim.cash + sim.invested, sim.equity, rtol=1e-12)
        self.assertTrue((sim.cash >= -1e-9).all())
        self.assertTrue(((sim.post_weight >= -1e-12) & (sim.post_weight <= 1 + 1e-9)).all())
        # Realized plus unrealized P&L explain equity change net of costs.
        explained = (
            START_CAPITAL
            + sim.realized_pnl
            + sim.unrealized_pnl
            - sim.total_costs
        )
        np.testing.assert_allclose(explained, sim.equity, rtol=1e-9)

    def test_transaction_costs_reduce_equity_and_are_recorded(self):
        free = _run(proportional_model("linear"), self.strategies, self.signals)
        costly = _run(proportional_model("linear"), self.strategies, self.signals, cost_bps=25)
        self.assertLess(costly.equity[-1], free.equity[-1])
        self.assertAlmostEqual(costly.total_costs[-1], costly.cost.sum())
        self.assertGreater(costly.total_costs[-1], 0)
        self.assertEqual(free.total_costs[-1], 0)

    def test_frame_is_compatible_with_overlay_consumers(self):
        sim = _run(proportional_model("linear"), self.strategies, self.signals)
        frame = sim.frame
        for column in (
            "Date", "Close", "TrackChange", "LongTradeIn", "LongTradeOut", "HoldLong",
            "RollingPnL", "TradePnL", "TradeEntry", "DaysInTrade", "RunningMax", "Drawdown",
        ):
            self.assertIn(column, frame.columns)
        # One portfolio trade from first exposure to flat; partial changes are not trades.
        self.assertEqual(frame["LongTradeIn"].tolist(), [True, False, False, False, False, False])
        self.assertEqual(frame["LongTradeOut"].tolist(), [False, False, False, False, True, False])
        self.assertEqual(frame["HoldLong"].tolist(), [False, True, True, True, True, False])
        exit_row = frame[frame["LongTradeOut"]].iloc[0]
        self.assertAlmostEqual(exit_row["TradePnL"], 17435.25 / 15000.0 - 1.0)

    def test_no_strategies_eligible_stays_in_cash(self):
        strategies = [_strategy("a")]
        signals = {
            "a": _signals("a", DATES, ["buy", "", "", "", "", ""], entry_ready=[False] * 6)
        }
        sim = _run(proportional_model("linear"), strategies, signals)
        np.testing.assert_allclose(sim.equity, START_CAPITAL)
        self.assertEqual(sim.eligible_count.tolist(), [0] * 6)
        self.assertIsNone(sim.eligibility[0]["first_eligible_date"])

    def test_same_active_count_swap_is_a_state_change(self):
        a = _signals("a", DATES, ["buy", "sell", "", "", "", ""])
        b = _signals("b", DATES, ["", "buy", "", "", "", ""])
        sim = _run(proportional_model("linear"), [_strategy("a"), _strategy("b")], {"a": a, "b": b})
        self.assertEqual(sim.active_count.tolist()[:3], [1, 1, 1])
        self.assertTrue(sim.rebalanced[1])  # bar 1 swaps A for B; drift is trued up to 50%

    def test_mixed_symbols_split_exposure_by_active_strategy(self):
        dates = DATES
        track = {
            "AAA": pd.Series([0, 0.1, 0.1, 0.1, 0.1, 0.1], index=dates),
            "BBB": pd.Series([0, -0.1, -0.1, -0.1, -0.1, -0.1], index=dates),
        }
        close = {s: pd.Series(np.linspace(100, 110, 6), index=dates) for s in track}
        a = _signals("a", dates, ["buy", "", "", "", "", ""], symbol="AAA")
        b = _signals("b", dates, ["buy", "", "", "", "", ""], symbol="BBB")
        sim = simulate_portfolio_allocation(
            [_strategy("a", "AAA"), _strategy("b", "BBB")],
            {"a": a, "b": b},
            model=proportional_model("linear"),
            global_proxy=None,
            track_changes=track,
            close_series=close,
        )
        symbols = {p["symbol"]: p for p in sim.final_positions}
        self.assertEqual(set(symbols), {"AAA", "BBB"})
        self.assertAlmostEqual(sim.post_weight[0], 1.0)
        self.assertAlmostEqual(
            symbols["AAA"]["market_value"] + symbols["BBB"]["market_value"],
            sim.equity[-1],
        )


class WarmupAndAvailabilityTests(unittest.TestCase):
    def test_strategy_joins_denominator_once_eligible_even_before_entering(self):
        a = _signals("a", DATES, ["buy", "", "", "", "", ""])
        b = _signals(
            "b", DATES, ["", "", "", "", "", ""], entry_ready=[False, False, False, True, True, True]
        )
        sim = _run(proportional_model("linear"), [_strategy("a"), _strategy("b")], {"a": a, "b": b})
        self.assertEqual(sim.eligible_count.tolist(), [1, 1, 1, 2, 2, 2])
        # Alone, A is 100%. B turns eligible on bar 3 without entering, so A drops to 50%.
        # Afterwards the position drifts with price until the next state change.
        np.testing.assert_allclose(sim.post_weight[:4], [1, 1, 1, 0.5], atol=1e-12)
        np.testing.assert_allclose(sim.frame["TargetExposure"], [1, 1, 1, 0.5, 0.5, 0.5])
        self.assertTrue(sim.rebalanced[3])
        self.assertFalse(sim.rebalanced[4])
        report = {item["strategy_id"]: item for item in sim.eligibility}
        self.assertEqual(report["b"]["warmup_bars"], 3)
        self.assertEqual(report["b"]["first_eligible_date"], DATES[3].strftime("%Y-%m-%d"))
        self.assertEqual(report["a"]["warmup_bars"], 0)

    def test_mid_history_readiness_gaps_are_counted_not_dropped(self):
        a = _signals("a", DATES, ["", "", "", "", "", ""], entry_ready=[True, True, False, False, True, True])
        sim = _run(proportional_model("linear"), [_strategy("a")], {"a": a})
        self.assertEqual(sim.eligible_count.tolist(), [1] * 6)
        self.assertEqual(sim.eligibility[0]["unavailable_bars"], 2)

    def test_strategy_without_entry_ready_is_eligible_from_its_first_bar(self):
        late_dates = DATES[2:]
        a = _signals("a", DATES, ["buy", "", "", "", "", ""])
        b = _signals("b", late_dates, ["", "", "", ""])
        sim = _run(proportional_model("linear"), [_strategy("a"), _strategy("b")], {"a": a, "b": b})
        self.assertEqual(sim.eligible_count.tolist(), [1, 1, 2, 2, 2, 2])

    def test_data_ending_early_removes_strategy_from_denominator_and_state(self):
        early_dates = DATES[:3]
        a = _signals("a", DATES, ["buy", "", "", "", "", ""])
        b = _signals("b", early_dates, ["buy", "", ""])
        sim = _run(proportional_model("linear"), [_strategy("a"), _strategy("b")], {"a": a, "b": b})
        self.assertEqual(sim.eligible_count.tolist(), [2, 2, 2, 1, 1, 1])
        self.assertEqual(sim.active_count.tolist(), [2, 2, 2, 1, 1, 1])
        np.testing.assert_allclose(sim.post_weight, [1, 1, 1, 1, 1, 1], atol=1e-12)

    def test_metrics_window_starts_at_first_eligible_bar(self):
        a = _signals(
            "a", DATES, ["", "", "buy", "", "", ""], entry_ready=[False, False, True, True, True, True]
        )
        sim = _run(proportional_model("linear"), [_strategy("a")], {"a": a})
        self.assertEqual(sim.start_index, 2)
        analytics = exposure_analytics(arrays_from_simulation(sim))
        self.assertEqual(analytics["bars"], 4)


def _random_flags(rng, n, enter=0.15, leave=0.3):
    events = []
    holding = False
    for _ in range(n):
        if holding:
            event = "sell" if rng.random() < leave else ""
        else:
            event = "buy" if rng.random() < enter else ""
        holding = (holding and event != "sell") or event == "buy"
        events.append(event)
    return events


class AllInReproductionTests(unittest.TestCase):
    """The all-in allocator on the new engine is the existing hold-until-all-exit overlay."""

    def _compare(self, seed, n_strategies, symbol_mode):
        rng = random.Random(seed)
        n = 400
        dates = pd.bdate_range("2020-01-01", periods=n)
        returns = np.array([rng.gauss(0.0005, 0.02) for _ in range(n)])
        track = {"AAA": pd.Series(returns, index=dates)}
        close = {"AAA": pd.Series(100 * np.cumprod(1 + returns / 3), index=dates)}
        strategies = [_strategy(f"s{k}") for k in range(n_strategies)]
        signals = {
            s.id: _signals(s.id, dates, _random_flags(rng, n)) for s in strategies
        }
        order = [s.id for s in strategies]
        legacy = simulate_portfolio_overlay(
            strategies, signals, overlap_mode="hold_until_all_exit", global_proxy=None,
            track_changes=track, close_series=close, strategy_order=order,
        )
        engine = simulate_portfolio_allocation(
            strategies, signals, model=AllInModel(), global_proxy=None,
            track_changes=track, close_series=close,
        )
        np.testing.assert_allclose(
            engine.frame["RollingPnL"].to_numpy(), legacy["RollingPnL"].to_numpy(), rtol=1e-9
        )
        np.testing.assert_array_equal(
            engine.frame["HoldLong"].to_numpy(), legacy["HoldLong"].to_numpy()
        )
        np.testing.assert_array_equal(
            engine.frame["LongTradeIn"].to_numpy(), legacy["LongTradeIn"].to_numpy()
        )
        np.testing.assert_array_equal(
            engine.frame["LongTradeOut"].to_numpy(), legacy["LongTradeOut"].to_numpy()
        )

    def test_matches_legacy_across_random_signal_sets(self):
        for seed in range(12):
            for n_strategies in (1, 2, 4):
                with self.subTest(seed=seed, strategies=n_strategies):
                    self._compare(seed, n_strategies, "single")

    def test_costed_all_in_overlay_matches_the_ledger(self):
        from api.allocation_analytics import charge_overlay_costs

        for seed in range(6):
            rng = random.Random(100 + seed)
            n = 300
            dates = pd.bdate_range("2021-01-01", periods=n)
            returns = np.array([rng.gauss(0.0005, 0.02) for _ in range(n)])
            track = {"AAA": pd.Series(returns, index=dates)}
            close = {"AAA": pd.Series(np.linspace(100, 200, n), index=dates)}
            strategies = [_strategy("s0"), _strategy("s1")]
            signals = {s.id: _signals(s.id, dates, _random_flags(rng, n)) for s in strategies}
            legacy = simulate_portfolio_overlay(
                strategies, signals, overlap_mode="hold_until_all_exit", global_proxy=None,
                track_changes=track, close_series=close, strategy_order=["s0", "s1"],
            )
            costed = charge_overlay_costs(legacy, 15.0)
            engine = simulate_portfolio_allocation(
                strategies, signals, model=AllInModel(), cost_bps=15.0, global_proxy=None,
                track_changes=track, close_series=close,
            )
            with self.subTest(seed=seed):
                np.testing.assert_allclose(
                    costed["RollingPnL"].to_numpy(), engine.equity, rtol=1e-9
                )
                np.testing.assert_allclose(
                    costed["TransactionCost"].to_numpy(), engine.cost, rtol=1e-9, atol=1e-9
                )
                self.assertLess(costed["RollingPnL"].iloc[-1], legacy["RollingPnL"].iloc[-1])

    def test_zero_cost_leaves_the_overlay_frame_identical(self):
        from api.allocation_analytics import charge_overlay_costs

        strategies, signals = _two_strategy_case()
        track, close = _market()
        legacy = simulate_portfolio_overlay(
            strategies, signals, overlap_mode="hold_until_all_exit", global_proxy=None,
            track_changes=track, close_series=close, strategy_order=["a", "b"],
        )
        self.assertIs(charge_overlay_costs(legacy, 0.0), legacy)

    def test_default_allocation_dispatches_to_the_untouched_overlay(self):
        strategies, signals = _two_strategy_case()
        track, close = _market()
        kwargs = dict(
            overlap_mode="hold_until_all_exit", global_proxy=None,
            track_changes=track, close_series=close, strategy_order=["a", "b"],
        )
        direct = simulate_portfolio_overlay(strategies, signals, **kwargs)
        via_default = simulate_portfolio_frame(strategies, signals, allocation=None, **kwargs)
        via_settings = simulate_portfolio_frame(
            strategies, signals, allocation=allocation_settings(SimpleNamespace()), **kwargs
        )
        pd.testing.assert_frame_equal(direct, via_default)
        pd.testing.assert_frame_equal(direct, via_settings)
        self.assertNotIn("Exposure", direct.columns)


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        strategies, signals = _two_strategy_case()
        self.sim = _run(proportional_model("linear"), strategies, signals)
        self.all_in = _run(AllInModel(), strategies, signals)

    def test_exposure_time_and_rebalance_counts(self):
        stats = exposure_analytics(arrays_from_simulation(self.sim))
        # Weights carried into each bar: 0, .5, 1, 1, .5, 0
        self.assertAlmostEqual(stats["average_exposure_pct"], 50.0)
        self.assertAlmostEqual(stats["time_invested_pct"], 4 / 6 * 100)
        self.assertAlmostEqual(stats["average_exposure_when_invested_pct"], 75.0)
        self.assertEqual(stats["rebalance_count"], 4)
        self.assertEqual(stats["entry_count"], 1)
        self.assertEqual(stats["exit_count"], 1)
        self.assertEqual(stats["adjustment_count"], 2)
        self.assertAlmostEqual(stats["max_exposure_pct"], 100.0)

    def test_turnover_uses_traded_notional(self):
        stats = exposure_analytics(arrays_from_simulation(self.sim))
        traded = 7500.0 + 7500.0 + 8505.0 + 8930.25
        self.assertAlmostEqual(stats["total_traded_notional"], traded, places=6)
        mean_equity = float(np.mean(self.sim.equity))
        self.assertAlmostEqual(stats["turnover_total_pct"], traded / 2 / mean_equity * 100)
        self.assertAlmostEqual(
            stats["turnover_annualized_pct"], stats["turnover_total_pct"] / (6 / 252)
        )

    def test_overlay_frame_adapter_reports_all_in_exposure(self):
        strategies, signals = _two_strategy_case()
        track, close = _market()
        legacy = simulate_portfolio_overlay(
            strategies, signals, overlap_mode="hold_until_all_exit", global_proxy=None,
            track_changes=track, close_series=close, strategy_order=["a", "b"],
        )
        stats = exposure_analytics(arrays_from_overlay(legacy))
        self.assertAlmostEqual(stats["average_exposure_when_invested_pct"], 100.0)
        self.assertAlmostEqual(stats["time_invested_pct"], 4 / 6 * 100)
        self.assertEqual(stats["entry_count"], 1)
        self.assertEqual(stats["exit_count"], 1)
        self.assertEqual(stats["adjustment_count"], 0)
        # The engine's all-in book agrees with the legacy adapter.
        engine_stats = exposure_analytics(arrays_from_simulation(self.all_in))
        self.assertAlmostEqual(engine_stats["average_exposure_pct"], stats["average_exposure_pct"])

    def test_attribution_isolates_the_allocation_effect(self):
        result = allocation_attribution(self.sim, self.all_in)
        # Half the book on bars 1 (+10%) and 4 (+5%) where all-in held everything.
        self.assertAlmostEqual(result["allocation_effect_pp"], -7.5, places=9)
        self.assertAlmostEqual(result["cost_effect_pp"], 0.0, places=9)
        self.assertAlmostEqual(result["other_pp"], 0.0, places=9)
        self.assertAlmostEqual(
            result["allocation_effect_pp"] + result["cost_effect_pp"] + result["other_pp"],
            result["total_return_gap_pp"],
            places=9,
        )
        self.assertAlmostEqual(
            result["final_equity_delta"], self.sim.equity[-1] - self.all_in.equity[-1]
        )
        by_level = {row["active_count"]: row for row in result["by_active_count"]}
        self.assertEqual(by_level[2]["allocation_effect_pp"], 0.0)
        self.assertAlmostEqual(by_level[1]["allocation_effect_pp"], -7.5)

    def test_attribution_against_itself_is_zero(self):
        result = allocation_attribution(self.all_in, self.all_in)
        self.assertEqual(result["allocation_effect_pp"], 0.0)
        self.assertEqual(result["final_equity_delta"], 0.0)

    def test_attribution_picks_up_cost_differences(self):
        strategies, signals = _two_strategy_case()
        model = _run(proportional_model("linear"), strategies, signals, cost_bps=50)
        reference = _run(AllInModel(), strategies, signals, cost_bps=50)
        result = allocation_attribution(model, reference)
        self.assertNotEqual(result["cost_effect_pp"], 0.0)
        self.assertAlmostEqual(
            result["allocation_effect_pp"] + result["cost_effect_pp"] + result["other_pp"],
            result["total_return_gap_pp"],
            places=9,
        )


class _NeutralRegimeContext:
    """No regime evidence: the regime-adjusted model then behaves like all-in."""

    def decisions(self, ids, active, dates=None, settings=None):
        return {}


def _bind_neutral(allocation, *args, **kwargs):
    from dataclasses import replace

    if not allocation.is_regime_adjusted:
        return allocation
    return replace(allocation, regime_context=_NeutralRegimeContext())


def _patched_portfolio(test, request_kwargs=None):
    strategies, signals = _two_strategy_case()
    track, close = _market()
    by_id = {s.id: s for s in strategies}
    market = (
        pd.DataFrame(),
        {"AAA": pd.DataFrame({"Date": DATES, "Close": np.linspace(100, 110, 6)})},
        track,
        close,
    )
    patches = [
        patch("api.portfolio_service.bind_regime_context", side_effect=_bind_neutral),
        patch("api.portfolio_service.get_strategy_by_id", side_effect=lambda sid: by_id[sid]),
        patch("api.portfolio_service._load_portfolio_market_data", return_value=market),
        patch(
            "api.portfolio_service._load_strategy_signals",
            side_effect=lambda strategy, **_: signals[strategy.id],
        ),
    ]
    for item in patches:
        item.start()
        test.addCleanup(item.stop)
    return SimpleNamespace(
        strategy_ids=["a", "b"],
        overlap_mode="hold_until_all_exit",
        proxy_symbol=None,
        years=25,
        **(request_kwargs or {}),
    )


class PortfolioPayloadTests(unittest.TestCase):
    def test_default_request_is_all_in_with_exposure_analytics(self):
        payload = simulate_portfolio(_patched_portfolio(self))
        allocation = payload["allocation"]
        self.assertEqual(allocation["model"], "all_in")
        self.assertIsNone(allocation["attribution"])
        self.assertAlmostEqual(allocation["analytics"]["average_exposure_when_invested_pct"], 100.0)
        self.assertEqual(allocation["exposure_steps"][0]["exposure"], 1.0)

    def test_proportional_payload(self):
        request = _patched_portfolio(
            self,
            dict(allocation_model="proportional", proportional_mode="linear", transaction_cost_bps=0.0),
        )
        payload = simulate_portfolio(request)
        allocation = payload["allocation"]
        self.assertEqual(allocation["model"], "proportional_linear")
        self.assertEqual(allocation["analytics"]["rebalance_count"], 4)
        self.assertAlmostEqual(allocation["attribution"]["allocation_effect_pp"], -7.5, places=6)
        self.assertEqual(len(allocation["eligibility"]), 2)
        self.assertAlmostEqual(allocation["ledger"]["equity"], 17435.25, places=4)
        self.assertEqual(allocation["ledger"]["positions"], [])
        self.assertAlmostEqual(payload["summary"]["rolling_pnl"], 17435.25, places=4)
        self.assertEqual(len(payload["contribution"]), 2)
        self.assertIn("proportional linear", payload["summary"]["description"])
        trade = payload["trades"][0]
        self.assertAlmostEqual(trade["trade_pnl"], 17435.25 / 15000 - 1, places=6)
        self.assertGreater(trade["entry_price"], 1000)  # portfolio equity, not an instrument price
        steps = allocation["exposure_steps"]
        self.assertEqual([s["exposure"] for s in steps], [0.5, 1.0, 0.5, 0.0])

    def test_comparison_charges_every_model_the_same_cost_rate(self):
        request = _patched_portfolio(self, dict(transaction_cost_bps=20.0))
        payload = simulate_portfolio_allocation_comparison(request)
        for row in payload["models"]:
            self.assertGreater(row["analytics"]["total_transaction_costs"], 0, row["model"])
        free = simulate_portfolio_allocation_comparison(_patched_portfolio(self))
        self.assertLess(
            payload["models"][0]["metrics"]["rolling_pnl"],
            free["models"][0]["metrics"]["rolling_pnl"],
        )

    def test_comparison_lists_all_models_on_one_window(self):
        payload = simulate_portfolio_allocation_comparison(_patched_portfolio(self))
        self.assertEqual(
            [row["model"] for row in payload["models"]],
            [
                "all_in",
                "regime_adjusted_harsh",
                "regime_adjusted_strict",
                "regime_adjusted_medium",
                "regime_adjusted_low",
                "proportional_linear",
                "proportional_early_conviction",
                "proportional_confirmation_required",
                "tiered",
            ],
        )
        curves = payload["equity_curves"]
        self.assertEqual(len(curves), 6)
        first = curves[0]
        for key in ("all_in", "proportional_linear"):
            self.assertAlmostEqual(first[key], START_CAPITAL)
        all_in = payload["models"][0]
        self.assertIsNone(all_in["attribution"])
        by_model = {row["model"]: row for row in payload["models"]}
        self.assertAlmostEqual(
            by_model["proportional_linear"]["metrics"]["rolling_pnl"], 17435.25, places=4
        )
        self.assertLess(
            by_model["proportional_confirmation_required"]["analytics"]["average_exposure_pct"],
            by_model["proportional_linear"]["analytics"]["average_exposure_pct"],
        )
        self.assertTrue(math.isfinite(all_in["metrics"]["rolling_pnl"]))


if __name__ == "__main__":
    unittest.main()
