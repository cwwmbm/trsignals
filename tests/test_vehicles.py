import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import backtest as bt
import config
from api import leverage_settings as ls
from api import vehicles
from api.leverage_settings import SymbolLeverage, normalize_settings

N = 120
DATES = pd.bdate_range("2018-01-01", periods=N)
LISTED = 60  # the leverage symbol's first close is bar 60


def underlying_close():
    rng = np.random.default_rng(1)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0008, 0.012, N)), index=DATES)


def vehicle_close_from(close, beta=2.9, daily_cost=0.0004, start=LISTED):
    """A leverage symbol that tracks beta * return - cost from bar `start`."""
    ret = close.pct_change().fillna(0.0)
    path = 100 * np.cumprod(1 + beta * ret - daily_cost)
    return path[start:]


@contextlib.contextmanager
def settings(general=3, entries=(), vehicle_closes=None, rates=None):
    """Leverage settings and leverage-symbol prices for the duration of a block."""
    original = ls.DEFAULT_STORE_PATH
    folder = tempfile.TemporaryDirectory()
    try:
        ls.DEFAULT_STORE_PATH = Path(folder.name) / "settings.json"
        ls.save_settings(normalize_settings(general, entries))
        closes = vehicle_closes or {}
        with patch("api.vehicle_data.load_close", side_effect=lambda symbol: closes.get(symbol)), \
             patch("api.vehicle_data.load_rates", return_value=rates):
            yield
    finally:
        ls.DEFAULT_STORE_PATH = original
        folder.cleanup()


def frame(close, buys=(5, 25, 80, 100), symbol="SOXX"):
    buy = np.zeros(N, dtype=bool)
    buy[list(buys)] = True
    data = pd.DataFrame({
        "Date": DATES,
        "Close": close.to_numpy(),
        "%Change": config.Leverage * close.pct_change().fillna(0.0).to_numpy(),
        "Buy": buy,
        "Sell": False,
    })
    data.attrs["symbol"] = symbol
    return data


def trades(executed):
    """(entry bar, exit bar) pairs."""
    starts = list(np.flatnonzero(executed["LongTradeIn"]))
    ends = list(np.flatnonzero(executed["LongTradeOut"]))
    return list(zip(starts, ends))


class SeriesMathTests(unittest.TestCase):
    def test_general_leverage_scales_the_symbols_return(self):
        close = underlying_close()
        out = vehicles.synthetic_returns(close, None, 2.5)
        np.testing.assert_allclose(out.iloc[1:], 2.5 * close.pct_change().iloc[1:])
        self.assertTrue(np.isnan(out.iloc[0]))

    def test_an_entry_uses_its_own_leverage_and_optional_costs(self):
        close = underlying_close()
        plain = SymbolLeverage("SOXX", "SOXL", 3.0)
        np.testing.assert_allclose(vehicles.synthetic_returns(close, plain, 9.0).iloc[1:], 3 * close.pct_change().iloc[1:])
        costed = SymbolLeverage("SOXX", "SOXL", 3.0, beta=2.96, annual_fee=0.0252, financing_multiple=2.0)
        rates = pd.Series(0.05, index=close.index)
        out = vehicles.synthetic_returns(close, costed, 3.0, rates)
        expected = 2.96 * close.pct_change() - (0.0252 + 2.0 * 0.05) / 252
        np.testing.assert_allclose(out.iloc[1:], expected.iloc[1:])

    def test_missing_rates_mean_no_financing_not_an_error(self):
        close = underlying_close()
        costed = SymbolLeverage("SOXX", "SOXL", 3.0, annual_fee=0.0252, financing_multiple=2.0)
        out = vehicles.synthetic_returns(close, costed, 3.0, None)
        np.testing.assert_allclose(out.iloc[1:], (3 * close.pct_change() - 0.0252 / 252).iloc[1:])

    def test_vehicle_returns_need_two_consecutive_closes(self):
        close = pd.Series([10.0, 11.0, 12.1, 13.31], index=DATES[:4])
        gap = close.drop(DATES[2])
        ret, ok = vehicles.vehicle_returns(gap, DATES[:5])
        self.assertEqual(list(ok), [True, True, False, True, False])
        self.assertAlmostEqual(ret.iloc[1], 0.1)
        self.assertTrue(np.isnan(ret.iloc[2]))
        self.assertTrue(np.isnan(ret.iloc[3]))  # its previous close is missing
        self.assertTrue(np.isnan(ret.iloc[0]))

    def test_the_stream_knows_when_the_leverage_symbol_starts(self):
        close = underlying_close()
        stream = vehicles.build_stream(
            pd.DatetimeIndex(DATES), close, SymbolLeverage("SOXX", "SOXL", 3.0),
            vehicle_close=pd.Series(vehicle_close_from(close).to_numpy(), index=DATES[LISTED:]), rates=None,
        )
        self.assertEqual(stream.first_vehicle_date, DATES[LISTED])
        self.assertFalse(stream.vehicle_ok[LISTED - 1])
        self.assertTrue(stream.vehicle_ok[LISTED])
        self.assertTrue(np.isnan(stream.vehicle[LISTED]))  # its first close has no return yet
        self.assertTrue(np.isfinite(stream.vehicle[LISTED + 1]))

    def test_a_missing_leverage_symbol_still_yields_the_entrys_stand_in(self):
        close = underlying_close()
        stream = vehicles.build_stream(
            pd.DatetimeIndex(DATES), close, SymbolLeverage("SOXX", "SOXL", 3.0), vehicle_close=None, rates=None
        )
        self.assertFalse(stream.vehicle_available)
        np.testing.assert_allclose(stream.synthetic[1:], 3 * close.pct_change().to_numpy()[1:])

    def test_the_cost_model_is_recovered_from_overlapping_history(self):
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2012-01-01", periods=1500)
        u = pd.Series(rng.normal(0.0006, 0.018, len(idx)), index=idx)
        rate = pd.Series(np.linspace(0.005, 0.05, len(idx)), index=idx)
        beta, fee, k = 2.95, 0.02, 2.5
        v = beta * u - (fee + k * rate) / 252 + rng.normal(0, 0.0004, len(idx))
        under = 100 * (1 + u).cumprod(); veh = 100 * (1 + v).cumprod()
        fit = vehicles.estimate_cost_model(under, veh, rate)
        self.assertAlmostEqual(fit["beta"], beta, delta=0.02)
        self.assertAlmostEqual(fit["fee"], fee, delta=0.01)
        self.assertAlmostEqual(fit["financing_multiple"], k, delta=0.5)
        self.assertGreater(fit["correlation"], 0.99)
        self.assertLess(fit["tracking_error"], 0.02)
        self.assertGreater(fit["days"], 1000)

    def test_estimating_needs_enough_history(self):
        close = underlying_close()
        with self.assertRaises(ValueError):
            vehicles.estimate_cost_model(close, close.iloc[:30], None)


class LongStratWithoutSettingsTests(unittest.TestCase):
    def test_an_unlisted_symbol_behaves_as_before_except_trades_now_compound(self):
        close = underlying_close()
        data = frame(close)
        executed = bt.execute_strategy(data.copy(), 4, 99, True)
        self.assertNotIn("VehicleHold", executed.columns)
        legacy_track = config.Leverage * close.pct_change().fillna(0.0).to_numpy()
        np.testing.assert_allclose(executed["TrackChange"].fillna(0.0).to_numpy(), legacy_track)
        # equity compounds the daily returns of the bars held
        held = executed["HoldLong"].to_numpy()
        expected = 15000 * np.cumprod(np.where(held, 1 + legacy_track, 1.0))
        np.testing.assert_allclose(executed["RollingPnL"].iloc[1:], expected[1:], rtol=1e-12)
        # and each trade reports the equity it earned
        for start, end in trades(executed):
            entry_equity = executed["RollingPnL"].iloc[start]
            self.assertAlmostEqual(executed["TradePnL"].iloc[end], executed["RollingPnL"].iloc[end] / entry_equity - 1, places=12)

    def test_general_leverage_rescales_the_returns(self):
        close = underlying_close()
        base = bt.execute_strategy(frame(close), 4, 99, True)["TrackChange"].fillna(0.0).to_numpy()
        with settings(general=2):
            scaled = bt.execute_strategy(frame(close), 4, 99, True)["TrackChange"].fillna(0.0).to_numpy()
        np.testing.assert_allclose(scaled, base * (2 / 3), atol=1e-12)

    def test_a_proxy_column_uses_general_leverage_too(self):
        close = underlying_close()
        data = frame(close, symbol="SPY")
        data["QQQ"] = (close * 0.5).to_numpy()
        with settings(general=2):
            executed = bt.execute_strategy(data, 4, 99, True, pnl_column="QQQ")
        np.testing.assert_allclose(executed["TrackChange"].iloc[1:], 2 * close.pct_change().iloc[1:].to_numpy(), atol=1e-12)


class LongStratWithAVehicleTests(unittest.TestCase):
    def setUp(self):
        self.close = underlying_close()
        self.veh = pd.Series(vehicle_close_from(self.close).to_numpy(), index=DATES[LISTED:])
        self.entry = [dict(symbol="SOXX", leverage_symbol="SOXL", leverage=3)]
        with settings(entries=[]):
            self.baseline = bt.execute_strategy(frame(self.close), 4, 99, True)

    def run_strategy(self, data=None, entries=None, **kwargs):
        with settings(entries=entries or self.entry, vehicle_closes={"SOXL": self.veh}):
            return bt.execute_strategy(data if data is not None else frame(self.close), 4, 99, True, **kwargs)

    def test_trades_are_identical_with_and_without_a_vehicle(self):
        executed = self.run_strategy()
        for column in ("LongTradeIn", "LongTradeOut", "HoldLong", "DaysInTrade", "ProfitableCloses", "TradeEntry"):
            np.testing.assert_array_equal(executed[column].to_numpy(), self.baseline[column].to_numpy(), column)

    def test_trades_opened_before_the_listing_earn_the_stand_in_and_later_ones_the_real_returns(self):
        executed = self.run_strategy()
        veh_ret = self.veh.pct_change().reindex(DATES).to_numpy()
        syn = 3 * self.close.pct_change().to_numpy()
        for start, end in trades(executed):
            held = np.arange(start + 1, end + 1)
            used = executed["TrackChange"].to_numpy()[held]
            if start >= LISTED:
                self.assertTrue(executed["VehicleHold"].to_numpy()[held].all(), (start, end))
                np.testing.assert_allclose(used, veh_ret[held])
            else:
                self.assertFalse(executed["VehicleHold"].to_numpy()[held].any(), (start, end))
                np.testing.assert_allclose(used, syn[held])
        self.assertTrue(executed["VehicleHold"].any() and (~executed["VehicleHold"]).any())

    def test_a_trade_open_when_the_leverage_symbol_appears_finishes_on_the_stand_in(self):
        straddling = frame(self.close, buys=(LISTED - 3, LISTED + 20))
        with settings(entries=self.entry, vehicle_closes={"SOXL": self.veh}):
            executed = bt.execute_strategy(straddling, 12, 99, True)
        (first_start, first_end), (second_start, second_end) = trades(executed)[:2]
        self.assertLess(first_start, LISTED)
        self.assertGreater(first_end, LISTED)  # still open after the listing
        held_first = np.arange(first_start + 1, first_end + 1)
        self.assertFalse(executed["VehicleHold"].to_numpy()[held_first].any())
        held_second = np.arange(second_start + 1, second_end + 1)
        self.assertTrue(executed["VehicleHold"].to_numpy()[held_second].all())

    def test_a_trade_entered_on_the_listing_day_uses_the_vehicle(self):
        data = frame(self.close, buys=(LISTED,))
        executed = self.run_strategy(data)
        start, end = trades(executed)[0]
        self.assertEqual(start, LISTED)
        self.assertTrue(executed["VehicleHold"].to_numpy()[start + 1:end + 1].all())

    def test_trade_returns_compound_the_returns_actually_earned(self):
        executed = self.run_strategy()
        for start, end in trades(executed):
            earned = np.prod(1 + executed["TrackChange"].to_numpy()[start + 1:end + 1]) - 1
            self.assertAlmostEqual(executed["TradePnL"].iloc[end], earned, places=12)
            self.assertAlmostEqual(
                executed["TradePnL"].iloc[end], executed["RollingPnL"].iloc[end] / executed["RollingPnL"].iloc[start] - 1, places=12
            )

    def test_a_missing_vehicle_day_falls_back_to_the_stand_in_for_that_day_only(self):
        gap_day = DATES[LISTED + 30]
        holed = self.veh.drop(gap_day)
        data = frame(self.close, buys=(LISTED + 28,))
        with settings(entries=self.entry, vehicle_closes={"SOXL": holed}):
            executed = bt.execute_strategy(data, 8, 99, True)
        start, end = trades(executed)[0]
        used = executed["TrackChange"].to_numpy()
        syn = 3 * self.close.pct_change().to_numpy()
        veh_ret = holed.pct_change().reindex(DATES).to_numpy()
        for bar in range(start + 1, end + 1):
            if DATES[bar] == gap_day or DATES[bar - 1] == gap_day:  # no usable return around the hole
                self.assertAlmostEqual(used[bar], syn[bar])
            else:
                self.assertAlmostEqual(used[bar], veh_ret[bar])

    def test_a_vehicle_that_cannot_be_loaded_leaves_the_entrys_stand_in(self):
        with settings(entries=self.entry, vehicle_closes={}):
            executed = bt.execute_strategy(frame(self.close), 4, 99, True)
        self.assertFalse(executed["VehicleHold"].any())
        np.testing.assert_allclose(executed["RollingPnL"].to_numpy(), self.baseline["RollingPnL"].to_numpy(), rtol=1e-12)

    def test_the_entrys_leverage_overrides_the_general_one_for_that_symbol_only(self):
        entries = [dict(symbol="SOXX", leverage_symbol="SOXL", leverage=2)]
        with settings(general=3, entries=entries, vehicle_closes={}):
            soxx = bt.execute_strategy(frame(self.close), 4, 99, True)
            spy = bt.execute_strategy(frame(self.close, symbol="SPY"), 4, 99, True)
        held = soxx["HoldLong"].to_numpy()
        np.testing.assert_allclose(soxx["TrackChange"].to_numpy()[held], (2 * self.close.pct_change()).to_numpy()[held])
        np.testing.assert_allclose(spy["TrackChange"].to_numpy()[held], (3 * self.close.pct_change()).to_numpy()[held])

    def test_the_trading_symbol_is_the_proxy_not_the_signal_symbol(self):
        data = frame(self.close, symbol="SPY")
        data["SOXX"] = self.close.to_numpy()
        executed = self.run_strategy(data, pnl_column="SOXX")
        self.assertTrue(executed["VehicleHold"].any())
        spy_only = self.run_strategy(frame(self.close, symbol="SPY"))
        self.assertNotIn("VehicleHold", spy_only.columns)  # SPY has no entry

    def test_an_explicit_trading_symbol_overrides_a_missing_stamp(self):
        data = frame(self.close)
        data.attrs.clear()
        self.assertNotIn("VehicleHold", self.run_strategy(data).columns)
        self.assertIn("VehicleHold", self.run_strategy(data, trading_symbol="SOXX").columns)

    def test_shorts_never_use_the_leverage_symbol(self):
        data = frame(self.close)
        with settings(entries=self.entry, vehicle_closes={"SOXL": self.veh}):
            short = bt.execute_strategy(data, 4, 99, False)
        self.assertNotIn("VehicleHold", short.columns)

    def test_the_cost_model_shapes_the_stand_in_only(self):
        entries = [dict(symbol="SOXX", leverage_symbol="SOXL", leverage=3, beta=2.9, annual_fee=0.02)]
        executed = self.run_strategy(entries=entries)
        pre = [(s, e) for s, e in trades(executed) if s < LISTED][0]
        held = np.arange(pre[0] + 1, pre[1] + 1)
        expected = 2.9 * self.close.pct_change().to_numpy()[held] - 0.02 / 252
        np.testing.assert_allclose(executed["TrackChange"].to_numpy()[held], expected)
        post = [(s, e) for s, e in trades(executed) if s >= LISTED][0]
        held = np.arange(post[0] + 1, post[1] + 1)
        np.testing.assert_allclose(executed["TrackChange"].to_numpy()[held], self.veh.pct_change().reindex(DATES).to_numpy()[held])


class StreamCacheTests(unittest.TestCase):
    def setUp(self):
        vehicles.clear_stream_cache()
        self.close = underlying_close()
        self.veh = pd.Series(vehicle_close_from(self.close).to_numpy(), index=DATES[LISTED:])

    def stream(self, entries, closes, close=None):
        with settings(entries=entries, vehicle_closes=closes):
            return vehicles.stream_for(DATES, (close if close is not None else self.close).to_numpy(), "SOXX")

    def test_the_same_inputs_share_one_stream(self):
        entry = [dict(symbol="SOXX", leverage_symbol="SOXL", leverage=3)]
        with settings(entries=entry, vehicle_closes={"SOXL": self.veh}):
            first = vehicles.stream_for(DATES, self.close.to_numpy(), "SOXX")
            second = vehicles.stream_for(DATES, self.close.to_numpy(), "SOXX")
        self.assertIs(first, second)

    def test_a_different_setting_prices_or_vehicle_data_gives_a_fresh_stream(self):
        entry = dict(symbol="SOXX", leverage_symbol="SOXL", leverage=3)
        base = self.stream([entry], {"SOXL": self.veh})
        self.assertIsNot(self.stream([dict(entry, leverage=2)], {"SOXL": self.veh}), base)
        self.assertIsNot(self.stream([dict(entry, annual_fee=0.02)], {"SOXL": self.veh}), base)
        self.assertIsNot(self.stream([entry], {"SOXL": self.veh * 1.01}), base)  # new vehicle data
        changed = self.close.copy(); changed.iloc[50] *= 1.02
        other = self.stream([entry], {"SOXL": self.veh}, changed)
        self.assertFalse(np.allclose(other.synthetic[50:52], base.synthetic[50:52]))

    def test_no_entry_means_no_stream_and_nothing_cached(self):
        with settings(entries=[]):
            self.assertIsNone(vehicles.stream_for(DATES, self.close.to_numpy(), "SOXX"))


class SingleBacktestPayloadTests(unittest.TestCase):
    def payload(self, entries, vehicle_closes):
        from api.serializers import detailed_backtest_payload

        close = underlying_close()
        with settings(entries=entries, vehicle_closes=vehicle_closes):
            executed = bt.execute_strategy(frame(close), 4, 99, True)
        return detailed_backtest_payload(executed, 4, 99, "test"), executed

    def test_the_payload_says_which_trades_used_which_instrument(self):
        close = underlying_close()
        veh = pd.Series(vehicle_close_from(close).to_numpy(), index=DATES[LISTED:])
        payload, executed = self.payload([dict(symbol="SOXX", leverage_symbol="SOXL", leverage=3)], {"SOXL": veh})
        info = payload["vehicle"]
        self.assertEqual((info["symbol"], info["leverage_symbol"], info["leverage"]), ("SOXX", "SOXL", 3.0))
        self.assertEqual(info["first_date"], DATES[LISTED].strftime("%Y-%m-%d"))
        self.assertTrue(info["available"])
        pairs = trades(executed)
        self.assertEqual(info["trades_on_leverage_symbol"], sum(1 for start, _ in pairs if start >= LISTED))
        self.assertEqual(info["trades_on_stand_in"], sum(1 for start, _ in pairs if start < LISTED))
        self.assertEqual(info["trades_on_leverage_symbol"] + info["trades_on_stand_in"], len(pairs))

    def test_a_leverage_symbol_without_history_is_reported_as_unavailable(self):
        payload, _ = self.payload([dict(symbol="SOXX", leverage_symbol="SOXL", leverage=3)], {})
        self.assertFalse(payload["vehicle"]["available"])
        self.assertEqual(payload["vehicle"]["trades_on_leverage_symbol"], 0)

    def test_a_symbol_without_an_entry_has_no_vehicle_block(self):
        payload, _ = self.payload([], {})
        self.assertNotIn("vehicle", payload)


if __name__ == "__main__":
    unittest.main()
