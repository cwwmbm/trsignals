"""The portfolio engines trade the leverage symbol once it exists, switching only between positions."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from api import leverage_settings as ls
from api import portfolio_service as ps
from api.allocation import AllInModel, proportional_model
from api.instruments import InstrumentBook
from api.position_episodes import ENTER, EXIT
from tests.test_allocation import _signals, _strategy
from tests.test_nested_portfolios import child
from tests.test_vehicles import settings

N = 30
LISTED = 12
DATES = pd.bdate_range("2020-01-01", periods=N)
ENTRY = [dict(symbol="AAA", leverage_symbol="AAAX", leverage=3)]


def returns(seed):
    return pd.Series(np.random.default_rng(seed).normal(0.001, 0.012, N), index=DATES)


R = {"AAA": returns(1), "BBB": returns(2), "CCC": returns(3)}
VEHICLE_RET = 2.9 * R["AAA"] - 0.0004


def dataset(symbols=("AAA", "BBB")):
    return {
        s: pd.DataFrame({"Date": DATES, "Close": (100 * (1 + R[s]).cumprod()).to_numpy()}) for s in symbols
    }


def vehicle_close(start=LISTED):
    return pd.Series((50 * (1 + VEHICLE_RET).cumprod()).to_numpy()[start:], index=DATES[start:])


def events(*trades):
    out = [""] * N
    for entry, exit_ in trades:
        out[entry], out[exit_] = "buy", "sell"
    return out


def signals_for(symbol, strategy_id, *trades):
    return _signals(strategy_id, DATES, events(*trades), symbol=symbol)


def market(symbols=("AAA", "BBB")):
    return ps._market_inputs_from_dataset(dataset(symbols))


def run(strategies, states, model, track, close, **kwargs):
    return ps.simulate_portfolio_allocation(
        strategies, states, model=model, global_proxy=None, track_changes=track, close_series=close,
        master_dates=DATES, **kwargs,
    )


class MarketInputTests(unittest.TestCase):
    def test_a_configured_symbol_also_gets_the_real_returns_and_closes(self):
        with settings(entries=ENTRY, vehicle_closes={"AAAX": vehicle_close()}):
            track, close = market()
        self.assertEqual(sorted(track), ["AAA", "AAA>AAAX", "BBB"])
        np.testing.assert_allclose(track["AAA"].iloc[1:], 3 * R["AAA"].iloc[1:].to_numpy(), atol=1e-12)
        self.assertTrue(track["AAA>AAAX"].iloc[:LISTED + 1].isna().all())
        np.testing.assert_allclose(
            track["AAA>AAAX"].iloc[LISTED + 1:], VEHICLE_RET.iloc[LISTED + 1:].to_numpy(), atol=1e-9
        )
        self.assertTrue(close["AAA>AAAX"].iloc[:LISTED].isna().all())
        self.assertTrue(close["AAA>AAAX"].iloc[LISTED:].notna().all())

    def test_without_settings_the_inputs_are_the_general_leverage_stand_in_only(self):
        with settings(general=2):
            track, close = market()
        self.assertEqual(sorted(track), ["AAA", "BBB"])
        np.testing.assert_allclose(track["BBB"].iloc[1:], 2 * R["BBB"].iloc[1:].to_numpy(), atol=1e-12)

    def test_a_leverage_symbol_without_data_leaves_only_the_entrys_stand_in(self):
        with settings(entries=ENTRY, vehicle_closes={}):
            track, _ = market()
        self.assertEqual(sorted(track), ["AAA", "BBB"])

    def test_timezone_aware_calendars_still_line_up(self):
        aware = dataset()
        for frame in aware.values():
            frame["Date"] = frame["Date"].dt.tz_localize("America/New_York")
        with settings(entries=ENTRY, vehicle_closes={"AAAX": vehicle_close()}):
            track, _ = ps._market_inputs_from_dataset(aware)
        self.assertTrue(track["AAA>AAAX"].notna().sum() > 10)


class RoutingTests(unittest.TestCase):
    def all_in(self, *trades, entries=ENTRY, vehicle=None):
        strategy = _strategy("a", "AAA")
        states = {"a": signals_for("AAA", "a", *trades)}
        with settings(entries=entries, vehicle_closes={"AAAX": vehicle if vehicle is not None else vehicle_close()}):
            track, close = market()
            simulation = run([strategy], states, AllInModel(), track, close)
        return simulation, track

    def test_each_position_opens_in_the_instrument_available_when_it_opens(self):
        simulation, _ = self.all_in((2, 6), (8, 16), (18, 25))
        by_trade = {}
        for transaction in simulation.transactions:
            by_trade.setdefault(transaction.bar, transaction.symbol)
        # opened before the listing, opened before it but closed after, opened after it
        self.assertEqual([by_trade[2], by_trade[8], by_trade[18]], ["AAA", "AAA", "AAAX"])
        exits = {t.bar: t.symbol for t in simulation.transactions if t.units_after == 0}
        self.assertEqual([exits[6], exits[16], exits[25]], ["AAA", "AAA", "AAAX"])

    def test_a_trade_open_at_the_listing_finishes_in_the_stand_in(self):
        simulation, _ = self.all_in((LISTED - 3, LISTED + 6))
        self.assertEqual({t.symbol for t in simulation.transactions}, {"AAA"})

    def test_a_position_opened_on_the_listing_day_uses_the_leverage_symbol(self):
        simulation, _ = self.all_in((LISTED, LISTED + 5))
        self.assertEqual({t.symbol for t in simulation.transactions}, {"AAAX"})

    def test_equity_compounds_exactly_the_returns_of_the_instrument_each_trade_held(self):
        simulation, track = self.all_in((2, 6), (8, 16), (18, 25))
        stand_in = track["AAA"].to_numpy()
        real = track["AAA>AAAX"].to_numpy()
        expected = 15000.0
        for entry, exit_ in ((2, 6), (8, 16), (18, 25)):
            use_real = entry >= LISTED
            for bar in range(entry + 1, exit_ + 1):
                expected *= 1 + (real[bar] if use_real and np.isfinite(real[bar]) else stand_in[bar])
        self.assertAlmostEqual(simulation.equity[-1], expected, places=6)

    def test_display_prices_are_the_instruments_own_closes(self):
        simulation, _ = self.all_in((2, 6), (18, 25))
        stand_in_fill = next(t for t in simulation.transactions if t.symbol == "AAA")
        vehicle_fill = next(t for t in simulation.transactions if t.symbol == "AAAX")
        self.assertAlmostEqual(stand_in_fill.underlying_close, dataset()["AAA"]["Close"].iloc[stand_in_fill.bar])
        self.assertAlmostEqual(vehicle_fill.underlying_close, vehicle_close().loc[DATES[vehicle_fill.bar]])

    def test_an_add_to_an_open_position_stays_in_the_instrument_it_is_in(self):
        a, b = _strategy("a", "AAA"), _strategy("b", "AAA")
        states = {"a": signals_for("AAA", "a", (LISTED + 2, 26)), "b": signals_for("AAA", "b", (LISTED + 6, 24))}
        with settings(entries=ENTRY, vehicle_closes={"AAAX": vehicle_close()}):
            track, close = market()
            simulation = run([a, b], states, proportional_model("linear"), track, close)
        self.assertEqual({t.symbol for t in simulation.transactions}, {"AAAX"})
        self.assertGreater(len(simulation.transactions), 2)  # enter, add, reduce, exit

    def test_adding_while_the_stand_in_is_open_does_not_hop_to_the_leverage_symbol(self):
        a, b = _strategy("a", "AAA"), _strategy("b", "AAA")
        states = {"a": signals_for("AAA", "a", (LISTED - 4, LISTED + 8)), "b": signals_for("AAA", "b", (LISTED + 2, LISTED + 6))}
        with settings(entries=ENTRY, vehicle_closes={"AAAX": vehicle_close()}):
            track, close = market()
            simulation = run([a, b], states, proportional_model("linear"), track, close)
        self.assertEqual({t.symbol for t in simulation.transactions}, {"AAA"})

    def test_after_going_flat_the_next_position_can_switch(self):
        simulation, _ = self.all_in((LISTED - 6, LISTED - 2), (LISTED + 4, LISTED + 9))
        symbols = [t.symbol for t in simulation.transactions]
        self.assertEqual(symbols, ["AAA", "AAA", "AAAX", "AAAX"])

    def test_other_symbols_are_untouched(self):
        a, b = _strategy("a", "AAA"), _strategy("b", "BBB")
        states = {"a": signals_for("AAA", "a", (LISTED + 2, 20)), "b": signals_for("BBB", "b", (LISTED + 2, 20))}
        with settings(entries=ENTRY, vehicle_closes={"AAAX": vehicle_close()}):
            track, close = market()
            simulation = run([a, b], states, AllInModel(), track, close)
        self.assertEqual({t.symbol for t in simulation.transactions}, {"AAAX", "BBB"})
        self.assertEqual(simulation.instruments["BBB"]["kind"], "synthetic")
        self.assertFalse(simulation.instruments["BBB"]["configured"])

    def test_instrument_metadata_and_report(self):
        simulation, _ = self.all_in((2, 6), (18, 25))
        self.assertEqual(simulation.instruments["AAA"], dict(symbol="AAA", underlying="AAA", kind="synthetic", leverage=3.0, configured=True))
        self.assertEqual(simulation.instruments["AAAX"]["kind"], "vehicle")
        self.assertEqual(simulation.instruments["AAAX"]["underlying"], "AAA")
        (row,) = simulation.vehicle_report
        self.assertEqual((row["symbol"], row["leverage_symbol"], row["available"]), ("AAA", "AAAX", True))
        self.assertEqual(row["first_date"], DATES[LISTED].strftime("%Y-%m-%d"))

    def test_the_ledger_ties_out_across_the_switch(self):
        simulation, _ = self.all_in((2, 6), (8, 16), (18, 25))
        np.testing.assert_allclose(simulation.equity, simulation.cash + simulation.invested)
        np.testing.assert_allclose(
            simulation.equity - 15000.0,
            simulation.realized_pnl + simulation.unrealized_pnl - simulation.total_costs, atol=1e-7,
        )

    def test_a_missing_vehicle_means_the_stand_in_for_everything(self):
        strategy = _strategy("a", "AAA")
        states = {"a": signals_for("AAA", "a", (18, 25))}
        with settings(entries=ENTRY, vehicle_closes={}):
            track, close = market()
            simulation = run([strategy], states, AllInModel(), track, close)
        self.assertEqual({t.symbol for t in simulation.transactions}, {"AAA"})
        self.assertFalse(simulation.vehicle_report[0]["available"])

    def test_a_leverage_symbol_that_is_also_a_symbol_in_the_book_is_refused(self):
        a, x = _strategy("a", "AAA"), _strategy("x", "AAAX")
        track_x = {"AAA": returns(1), "AAAX": returns(4), "AAA>AAAX": returns(4)}
        closes = {"AAA": 100 * (1 + R["AAA"]).cumprod(), "AAAX": 50 * (1 + returns(4)).cumprod(), "AAA>AAAX": 50 * (1 + returns(4)).cumprod()}
        states = {"a": signals_for("AAA", "a", (2, 6)), "x": signals_for("AAAX", "x", (2, 6))}
        with settings(entries=ENTRY, vehicle_closes={"AAAX": vehicle_close()}):
            with self.assertRaisesRegex(ValueError, "also traded as a symbol"):
                InstrumentBook(["AAA", "AAAX"], track_x, closes, DATES)


class EpisodeAndPayloadTests(unittest.TestCase):
    def request(self, **kwargs):
        return SimpleNamespace(
            strategy_ids=["a"], portfolio_ids=[], overlap_mode="hold_until_all_exit", proxy_symbol=None, years=25,
            allocation_model="all_in", proportional_mode="linear", transaction_cost_bps=0.0, **kwargs,
        )

    def payload(self, *trades, entries=ENTRY):
        strategy = _strategy("a", "AAA")
        states = {"a": signals_for("AAA", "a", *trades)}
        by_id = {"a": strategy}
        with settings(entries=entries, vehicle_closes={"AAAX": vehicle_close()}):
            track, close = market()
            market_tuple = (pd.DataFrame(), dataset(), track, close)
            with patch("api.portfolio_service.get_strategy_by_id", side_effect=lambda sid: by_id[sid]), \
                 patch("api.portfolio_service._load_portfolio_market_data", return_value=market_tuple), \
                 patch("api.portfolio_service._load_strategy_signals", side_effect=lambda s, **_: states[s.id]):
                return ps.simulate_portfolio(self.request())

    def test_all_in_on_a_configured_symbol_runs_on_the_engine_not_the_overlay(self):
        with settings(entries=ENTRY, vehicle_closes={"AAAX": vehicle_close()}):
            self.assertFalse(ps.uses_overlay([_strategy("a", "AAA")], None))
            self.assertTrue(ps.uses_overlay([_strategy("b", "BBB")], None))
            self.assertFalse(ps.uses_overlay([_strategy("a", "AAA"), _strategy("b", "BBB")], None))
        with settings(entries=[]):
            self.assertTrue(ps.uses_overlay([_strategy("a", "AAA")], None))

    def test_episodes_say_which_instrument_they_held(self):
        payload = self.payload((2, 6), (18, 25))
        kinds = [(e["symbol"], e["instrument"]["kind"]) for e in payload["positions"]["episodes"]]
        self.assertEqual(kinds, [("AAA", "synthetic"), ("AAAX", "vehicle")])
        self.assertEqual(payload["positions"]["episodes"][1]["instrument"]["underlying"], "AAA")
        self.assertEqual([e["events"][0]["event"] for e in payload["positions"]["episodes"]], [ENTER, ENTER])
        self.assertEqual([e["events"][-1]["event"] for e in payload["positions"]["episodes"]], [EXIT, EXIT])

    def test_episodes_reconcile_with_the_equity_change(self):
        payload = self.payload((2, 6), (8, 16), (18, 25))
        self.assertAlmostEqual(payload["positions"]["reconciliation"]["episode_vs_equity_difference"], 0.0, places=6)

    def test_the_allocation_block_reports_the_vehicles_and_the_general_leverage(self):
        payload = self.payload((2, 6), (18, 25))
        allocation = payload["allocation"]
        self.assertEqual(allocation["vehicles"][0]["leverage_symbol"], "AAAX")
        self.assertEqual(allocation["general_leverage"], 3.0)
        self.assertEqual(allocation["trading_symbols"], ["AAA"])

    def test_a_book_with_no_entry_reports_no_vehicles(self):
        payload = self.payload((2, 6), entries=[])
        self.assertEqual(payload["allocation"]["vehicles"], [])
        self.assertFalse(payload["positions"]["episodes"][0]["instrument"]["configured"])


class NestedAgreesWithFlatTests(unittest.TestCase):
    """Wrapping a portfolio as a child must not change what it trades or earns."""

    def compare(self, model, trades_a, trades_b, second_symbol="AAA"):
        a, b = _strategy("a", "AAA"), _strategy("b", second_symbol)
        states = {"a": signals_for("AAA", "a", *trades_a), "b": signals_for(second_symbol, "b", *trades_b)}
        with settings(entries=ENTRY, vehicle_closes={"AAAX": vehicle_close()}):
            track, close = market()
            flat = run([a, b], states, model, track, close)
            ca, sa = child_n([a], {"a": states["a"]}, AllInModel(), "A")
            cb, sb = child_n([b], {"b": states["b"]}, AllInModel(), "B")
            nested = run([ca, cb], {ca.id: sa, cb.id: sb}, model, track, close)
        return flat, nested

    def assert_same(self, flat, nested):
        for field in ("equity", "cash", "invested", "realized_pnl", "unrealized_pnl", "total_costs", "post_weight"):
            np.testing.assert_allclose(getattr(nested, field), getattr(flat, field), rtol=1e-9, atol=1e-7, err_msg=field)
        self.assertEqual([(t.bar, t.symbol) for t in nested.transactions], [(t.bar, t.symbol) for t in flat.transactions])

    def test_nested_trades_the_same_instruments_and_earns_the_same_as_flat(self):
        shapes = [
            (((2, 6), (18, 25)), ((4, 10), (20, 27))),                   # before and after the listing
            (((LISTED - 4, LISTED + 8),), ((LISTED + 2, LISTED + 6),)),  # one straddles, one joins inside it
            (((LISTED + 1, 20),), ((LISTED + 5, 24),)),                  # both after
        ]
        for model in (AllInModel(), proportional_model("linear"), proportional_model("early_conviction"),
                      proportional_model("confirmation_required")):
            for symbol in ("AAA", "BBB"):
                for trades_a, trades_b in shapes:
                    with self.subTest(model=model.id, second=symbol, a=trades_a, b=trades_b):
                        flat, nested = self.compare(model, trades_a, trades_b, symbol)
                        self.assert_same(flat, nested)
        self.assertIn("AAAX", {t.symbol for t in flat.transactions})

    def test_a_nested_book_uses_the_leverage_symbol_only_for_new_positions(self):
        flat, nested = self.compare(AllInModel(), ((LISTED - 4, LISTED + 8),), ((LISTED + 2, LISTED + 6),))
        self.assertEqual({t.symbol for t in nested.transactions}, {"AAA"})
        self.assertEqual(nested.instruments["AAAX"]["kind"], "vehicle")


def child_n(members, states, model, name):
    """tests.test_nested_portfolios.child, on this module's 30-bar calendar."""
    from api.portfolio_service import AllocationSettings, StrategySignals, _effective_signal_matrices

    active, eligible, _ = _effective_signal_matrices(members, states, DATES)
    active, eligible = active.any(axis=0), eligible.any(axis=0)
    previous = np.r_[False, active[:-1]]
    identity = f"portfolio:{name}"
    member = SimpleNamespace(
        id=identity, name=name, symbol="Portfolio", direction="long", proxy_symbol=None, hold_days=5, profit=1,
        nested_spec=SimpleNamespace(
            members=members, signals=states, proxy=None, years=1, allocation=AllocationSettings(model=model, cost_bps=0),
        ),
    )
    return member, StrategySignals(identity, name, "Portfolio", DATES, active & ~previous, previous, previous & ~active, entry_ready=eligible)


if __name__ == "__main__":
    unittest.main()
