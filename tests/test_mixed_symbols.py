"""Portfolios that trade more than one symbol.

All-in on one trading symbol is the original overlay and must not change. On several symbols
it runs through the engine: always 100% invested while anything is active, split across the
symbols of the active strategies by how many strategies are active in each.
"""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from api.allocation import AllInModel
from api.portfolio_service import (
    PORTFOLIO_OVERLAP_MODE,
    AllocationSettings,
    allocation_settings,
    bind_regime_context,
    build_portfolio_overlay_frame,
    is_single_symbol,
    simulate_portfolio,
    simulate_portfolio_allocation,
    simulate_portfolio_allocation_comparison,
    simulate_portfolio_frame,
    simulate_portfolio_overlay,
    trading_symbols,
)
from tests.test_allocation import _bind_neutral, _signals, _strategy

N = 8
DATES = pd.bdate_range("2024-01-01", periods=N)
SYMBOLS = ("AAA", "BBB")


def _market(returns):
    track = {s: pd.Series(returns[s], index=DATES) for s in SYMBOLS}
    close = {s: pd.Series(np.linspace(100, 110, N), index=DATES) for s in SYMBOLS}
    return track, close


# AAA gains, BBB loses; both strategies are in a position on the same bars.
RETURNS = {"AAA": [0.0] + [0.10] * (N - 1), "BBB": [0.0] + [-0.10] * (N - 1)}


def _mixed_case():
    events = ["buy", "", "", "", "", "sell", "", ""]
    strategies = [_strategy("a", "AAA"), _strategy("b", "BBB")]
    signals = {
        "a": _signals("a", DATES, events, symbol="AAA"),
        "b": _signals("b", DATES, events, symbol="BBB"),
    }
    return strategies, signals


def _kwargs(track, close):
    return dict(global_proxy=None, track_changes=track, close_series=close, master_dates=DATES)


class TradingSymbolTests(unittest.TestCase):
    def test_the_trading_symbol_is_the_proxy_not_the_signal_symbol(self):
        spy_signals_soxx_trades = _strategy("a", "SPY")
        spy_signals_soxx_trades.proxy_symbol = "SOXX"
        soxx = _strategy("b", "SOXX")
        self.assertEqual(trading_symbols([spy_signals_soxx_trades, soxx], None), ["SOXX"])
        self.assertTrue(is_single_symbol([spy_signals_soxx_trades, soxx], None))

    def test_without_proxies_the_signal_symbols_are_the_trading_symbols(self):
        strategies = [_strategy("a", "SOXX"), _strategy("b", "XBI")]
        self.assertEqual(trading_symbols(strategies, None), ["SOXX", "XBI"])
        self.assertFalse(is_single_symbol(strategies, None))

    def test_a_portfolio_proxy_makes_any_portfolio_one_symbol(self):
        strategies = [_strategy("a", "SOXX"), _strategy("b", "XBI")]
        self.assertTrue(is_single_symbol(strategies, "SOXX"))
        self.assertEqual(trading_symbols(strategies, "SOXX"), ["SOXX"])

    def test_a_proxy_equal_to_the_signal_symbol_is_no_proxy(self):
        strategy = _strategy("a", "XBI")
        strategy.proxy_symbol = "xbi"
        self.assertEqual(trading_symbols([strategy], None), ["XBI"])


class EngineAllInSplitTests(unittest.TestCase):
    """3 SOXX strategies and 1 XBI strategy joining and leaving one at a time."""

    def test_split_follows_the_number_of_active_strategies_per_symbol(self):
        n = 14
        dates = pd.bdate_range("2024-01-01", periods=n)

        def events(buy, sell):
            e = [""] * n
            e[buy], e[sell] = "buy", "sell"
            return e

        spec = {"s1": ("SOXX", 1, 12), "s2": ("SOXX", 5, 12), "s3": ("SOXX", 7, 12), "x": ("XBI", 3, 9)}
        strategies = [_strategy(k, v[0]) for k, v in spec.items()]
        signals = {k: _signals(k, dates, events(v[1], v[2]), symbol=v[0]) for k, v in spec.items()}
        track = {s: pd.Series(np.zeros(n), index=dates) for s in ("SOXX", "XBI")}  # flat: no drift
        close = {s: pd.Series(np.full(n, 100.0), index=dates) for s in ("SOXX", "XBI")}
        sim = simulate_portfolio_allocation(
            strategies, signals, model=AllInModel(), global_proxy=None,
            track_changes=track, close_series=close, master_dates=dates,
        )
        expected = {
            0: (0.0, 0.0),  # nothing active
            1: (1.0, 0.0),  # one SOXX strategy: all in SOXX
            3: (0.5, 0.5),  # XBI fires: sell SOXX down to an equal split
            5: (2 / 3, 1 / 3),
            7: (0.75, 0.25),  # 3 SOXX + 1 XBI
            9: (1.0, 0.0),  # XBI exits: back to all SOXX
            12: (0.0, 0.0),
        }
        for bar, (soxx, xbi) in expected.items():
            with self.subTest(bar=bar):
                self.assertAlmostEqual(sim.asset_weight_post["SOXX"][bar], soxx, places=9)
                self.assertAlmostEqual(sim.asset_weight_post["XBI"][bar], xbi, places=9)

    def test_only_the_xbi_strategy_active_is_all_in_xbi(self):
        events = ["buy", "", "", "sell", "", "", "", ""]
        strategies = [_strategy("s", "SOXX"), _strategy("x", "XBI")]
        signals = {
            "s": _signals("s", DATES, [""] * N, symbol="SOXX"),
            "x": _signals("x", DATES, events, symbol="XBI"),
        }
        track = {s: pd.Series(np.zeros(N), index=DATES) for s in ("SOXX", "XBI")}
        close = {s: pd.Series(np.full(N, 100.0), index=DATES) for s in ("SOXX", "XBI")}
        sim = simulate_portfolio_allocation(
            strategies, signals, model=AllInModel(), global_proxy=None,
            track_changes=track, close_series=close, master_dates=DATES,
        )
        self.assertAlmostEqual(sim.asset_weight_post["XBI"][1], 1.0)
        self.assertAlmostEqual(sim.asset_weight_post["SOXX"][1], 0.0)


class AllInDispatchTests(unittest.TestCase):
    def setUp(self):
        self.strategies, self.signals = _mixed_case()
        self.track, self.close = _market(RETURNS)

    def _frame(self, strategies, signals, track, close, allocation=None):
        return simulate_portfolio_frame(
            strategies, signals, allocation=allocation, overlap_mode=PORTFOLIO_OVERLAP_MODE,
            strategy_order=[s.id for s in strategies], **_kwargs(track, close),
        )

    def test_several_symbols_run_through_the_engine_not_the_first_holders_stream(self):
        overlay = simulate_portfolio_overlay(
            self.strategies, self.signals, overlap_mode=PORTFOLIO_OVERLAP_MODE, strategy_order=["a", "b"],
            **_kwargs(self.track, self.close),
        )
        engine = simulate_portfolio_allocation(
            self.strategies, self.signals, model=AllInModel(), **_kwargs(self.track, self.close)
        )
        frame = self._frame(self.strategies, self.signals, self.track, self.close)
        np.testing.assert_allclose(frame["RollingPnL"], engine.frame["RollingPnL"])
        # The overlay earned only AAA's gains and ignored BBB's losses; the split book did not.
        self.assertGreater(overlay["RollingPnL"].iloc[-1], 1.4 * frame["RollingPnL"].iloc[-1])

    def test_one_trading_symbol_is_still_the_untouched_overlay(self):
        strategies = [_strategy("a", "AAA"), _strategy("b", "AAA")]
        events = ["buy", "", "", "", "", "sell", "", ""]
        signals = {
            "a": _signals("a", DATES, events),
            "b": _signals("b", DATES, ["", "buy", "", "", "", "", "sell", ""]),
        }
        track, close = _market(RETURNS)
        legacy = simulate_portfolio_overlay(
            strategies, signals, overlap_mode=PORTFOLIO_OVERLAP_MODE, strategy_order=["a", "b"],
            **_kwargs(track, close),
        )
        frame = self._frame(strategies, signals, track, close, allocation=AllocationSettings(model=AllInModel()))
        pd.testing.assert_frame_equal(frame, legacy)

    def test_a_portfolio_proxy_keeps_a_mixed_signal_set_on_the_overlay(self):
        track, close = _market(RETURNS)
        legacy = simulate_portfolio_overlay(
            self.strategies, self.signals, overlap_mode=PORTFOLIO_OVERLAP_MODE, strategy_order=["a", "b"],
            global_proxy="AAA", track_changes=track, close_series=close, master_dates=DATES,
        )
        frame = simulate_portfolio_frame(
            self.strategies, self.signals, allocation=None, overlap_mode=PORTFOLIO_OVERLAP_MODE,
            strategy_order=["a", "b"], global_proxy="AAA", track_changes=track, close_series=close,
            master_dates=DATES,
        )
        pd.testing.assert_frame_equal(frame, legacy)

    def test_costs_apply_to_the_mixed_all_in_book(self):
        free = self._frame(self.strategies, self.signals, self.track, self.close)
        costly = self._frame(
            self.strategies, self.signals, self.track, self.close,
            allocation=AllocationSettings(model=AllInModel(), cost_bps=25.0),
        )
        self.assertLess(costly["RollingPnL"].iloc[-1], free["RollingPnL"].iloc[-1])
        self.assertGreater(costly["TotalTransactionCosts"].iloc[-1], 0)


def _patched_mixed_portfolio(test, request_kwargs=None):
    strategies, signals = _mixed_case()
    track, close = _market(RETURNS)
    by_id = {s.id: s for s in strategies}
    market = (
        pd.DataFrame(),
        {s: pd.DataFrame({"Date": DATES, "Close": np.linspace(100, 110, N)}) for s in SYMBOLS},
        track,
        close,
    )
    patches = [
        patch("api.portfolio_service.bind_regime_context", side_effect=_bind_neutral),
        patch("api.portfolio_service.get_strategy_by_id", side_effect=lambda sid: by_id[sid]),
        patch("api.portfolio_service._load_portfolio_market_data", return_value=market),
        patch("api.portfolio_service._load_strategy_signals", side_effect=lambda strategy, **_: signals[strategy.id]),
    ]
    for item in patches:
        item.start()
        test.addCleanup(item.stop)
    return SimpleNamespace(
        strategy_ids=["a", "b"], overlap_mode=PORTFOLIO_OVERLAP_MODE, proxy_symbol=None, years=25,
        **(request_kwargs or {}),
    )


class MixedPortfolioPayloadTests(unittest.TestCase):
    def test_all_in_payload_on_several_symbols(self):
        payload = simulate_portfolio(_patched_mixed_portfolio(self))
        allocation = payload["allocation"]
        self.assertEqual(allocation["model"], "all_in")
        self.assertIsNone(allocation["attribution"])  # all-in is its own reference
        self.assertAlmostEqual(allocation["analytics"]["average_exposure_when_invested_pct"], 100.0, places=4)
        symbols = {episode["symbol"] for episode in payload["positions"]["episodes"]}
        self.assertEqual(symbols, {"AAA", "BBB"})
        self.assertLess(abs(payload["positions"]["reconciliation"]["episode_vs_equity_difference"]), 1e-6)
        # Portfolio equity, not one instrument's price, in the trades table.
        self.assertGreater(payload["trades"][0]["entry_price"], 1000)

    def test_all_in_on_one_symbol_keeps_the_overlay_payload(self):
        request = _patched_mixed_portfolio(self)
        request.proxy_symbol = "AAA"
        payload = simulate_portfolio(request)
        self.assertEqual(payload["allocation"]["model"], "all_in")
        self.assertEqual({e["symbol"] for e in payload["positions"]["episodes"]}, {"AAA"})

    def test_contribution_runs_on_a_mixed_all_in_book(self):
        payload = simulate_portfolio(_patched_mixed_portfolio(self))
        self.assertEqual(len(payload["contribution"]), 2)


class MixedComparisonTests(unittest.TestCase):
    def test_all_in_row_is_the_split_book_and_the_response_says_which_symbols(self):
        payload = simulate_portfolio_allocation_comparison(_patched_mixed_portfolio(self))
        self.assertFalse(payload["single_symbol"])
        self.assertEqual(payload["trading_symbols"], ["AAA", "BBB"])
        rows = {row["model"]: row for row in payload["models"]}
        self.assertEqual(len(payload["models"]), 9)
        self.assertEqual(payload["models"][0]["model"], "all_in")
        self.assertAlmostEqual(rows["all_in"]["analytics"]["average_exposure_when_invested_pct"], 100.0, places=4)
        # The split book earned less than the first-holder-only overlay would have.
        strategies, signals = _mixed_case()
        track, close = _market(RETURNS)
        engine = simulate_portfolio_allocation(
            strategies, signals, model=AllInModel(), **_kwargs(track, close)
        )
        self.assertAlmostEqual(rows["all_in"]["metrics"]["rolling_pnl"], engine.equity[-1], places=4)

    def test_single_symbol_comparison_reports_it_and_keeps_the_overlay_row(self):
        request = _patched_mixed_portfolio(self)
        request.proxy_symbol = "AAA"
        payload = simulate_portfolio_allocation_comparison(request)
        self.assertTrue(payload["single_symbol"])
        self.assertEqual(payload["trading_symbols"], ["AAA"])


class MixedRegimeBaselineTests(unittest.TestCase):
    def test_the_consensus_baseline_of_a_mixed_book_is_the_split_all_in_book(self):
        strategies, signals = _mixed_case()
        track, close = _market(RETURNS)
        with patch("api.market_regimes.load_regime_calendar", return_value=None):
            bound = bind_regime_context(
                allocation_settings(SimpleNamespace(allocation_model="regime_adjusted")),
                strategies, signals, global_proxy=None, track_changes=track, close_series=close,
                years=25, master_dates=DATES,
            )
        history = bound.regime_context._baseline_history(("a", "b"))
        engine = simulate_portfolio_allocation(
            strategies, signals, model=AllInModel(), **_kwargs(track, close)
        )
        np.testing.assert_array_equal(history.hold, engine.frame["HoldLong"].to_numpy(bool))
        np.testing.assert_array_equal(history.long_out, engine.frame["LongTradeOut"].to_numpy(bool))
        # A single strategy of the book is one symbol, so it still uses the overlay.
        single = bound.regime_context._baseline_history(("a",))
        overlay = simulate_portfolio_overlay(
            strategies[:1], {"a": signals["a"]}, overlap_mode=PORTFOLIO_OVERLAP_MODE,
            strategy_order=["a"], **_kwargs(track, close),
        )
        np.testing.assert_array_equal(single.hold, overlay["HoldLong"].to_numpy(bool))


class MixedScanTests(unittest.TestCase):
    def test_a_mixed_all_in_portfolio_still_produces_scan_signals(self):
        request = _patched_mixed_portfolio(self, dict(allocation_model="all_in"))
        frame, strategies, description, _, _ = build_portfolio_overlay_frame(request, years=1)
        self.assertEqual(len(strategies), 2)
        self.assertFalse(frame.empty)
        self.assertTrue(frame["HoldLong"].any())
        self.assertIn("Portfolio (2 strategies", description)


if __name__ == "__main__":
    unittest.main()
