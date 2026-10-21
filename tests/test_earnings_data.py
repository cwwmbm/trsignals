"""Earnings dates and the DaysToEarnings / DaysSinceEarnings indicators."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import indicators as ind
from api import earnings_data as ed
from api.custom_data import stub_market_context_columns, unavailable_indicator_ids
from api.indicator_catalog import get_indicator_def, is_market_wide_indicator, list_indicators

ET = ZoneInfo("America/New_York")
NO = ed.NO_EARNINGS


def history(*rows):
    """rows of (date, session) -> a normalized history frame."""
    frame = pd.DataFrame(rows, columns=["date", "session"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["announced"] = frame["date"] + pd.to_timedelta(np.where(frame["session"] == ed.BEFORE_OPEN, 7, 16), unit="h")
    for column in ("eps_estimate", "eps_reported", "surprise_pct"):
        frame[column] = np.nan
    return frame


def raw_rows(*stamps, **columns):
    """What yfinance returns: a frame indexed by tz-aware report times."""
    index = pd.DatetimeIndex([pd.Timestamp(s, tz=ET) for s in stamps], name="Earnings Date")
    data = {"EPS Estimate": np.nan, "Reported EPS": np.nan, "Surprise(%)": np.nan}
    data.update(columns)
    return pd.DataFrame(data, index=index)


def brute_force(bar_days, rows):
    """Reference: for every bar, scan for the next and the previous reaction bar the slow, obvious way."""
    n = len(bar_days)
    reaction_bars = []
    for day, session in rows:
        day = pd.Timestamp(day)
        if session == ed.BEFORE_OPEN:
            candidates = [i for i in range(n) if bar_days[i] >= day]
            reaction_day = day
        else:
            candidates = [i for i in range(n) if bar_days[i] > day]
            reaction_day = day + pd.offsets.BDay(1)
        if reaction_day < bar_days[0]:
            continue  # not on this history
        if candidates:
            reaction_bars.append(candidates[0])
        else:
            extra = int(np.busday_count(bar_days[-1].date(), reaction_day.date()))
            reaction_bars.append(n - 1 + max(extra, 1))
    to, since = [], []
    for i in range(n):
        ahead = [p - i for p in reaction_bars if p >= i]
        behind = [i - p for p in reaction_bars if p <= i]
        to.append(min(ahead) if ahead else NO)
        since.append(min(behind) if behind else NO)
    return np.array(to), np.array(since)


class NormalizeTests(unittest.TestCase):
    def test_sorted_oldest_first_with_the_session_of_each_report(self):
        raw = raw_rows("2024-05-02 16:00", "2024-04-12 06:00", "2024-02-01 16:00", "2024-03-05 08:00")
        frame = ed.normalize(raw)
        self.assertEqual(frame["date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-02-01", "2024-03-05", "2024-04-12", "2024-05-02"])
        self.assertEqual(frame["session"].tolist(), [ed.AFTER_CLOSE, ed.BEFORE_OPEN, ed.BEFORE_OPEN, ed.AFTER_CLOSE])

    def test_the_cutoff_is_noon_eastern(self):
        raw = raw_rows("2024-01-02 11:59", "2024-01-03 12:00", "2024-01-04 15:00", "2024-01-05 09:00")
        self.assertEqual(ed.normalize(raw)["session"].tolist(), [ed.BEFORE_OPEN, ed.AFTER_CLOSE, ed.AFTER_CLOSE, ed.BEFORE_OPEN])

    def test_a_day_listed_twice_keeps_the_row_with_the_most_information(self):
        raw = raw_rows("2024-02-01 16:00", "2024-02-01 16:00", **{"EPS Estimate": [np.nan, 1.5], "Reported EPS": [np.nan, 1.7]})
        frame = ed.normalize(raw)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame["eps_reported"].iloc[0], 1.7)

    def test_other_time_zones_are_read_as_eastern_time(self):
        index = pd.DatetimeIndex([pd.Timestamp("2024-02-01 21:00", tz="UTC")])  # 16:00 in New York
        frame = ed.normalize(pd.DataFrame({"EPS Estimate": [1.0], "Reported EPS": [np.nan], "Surprise(%)": [np.nan]}, index=index))
        self.assertEqual(frame["date"].iloc[0], pd.Timestamp("2024-02-01"))
        self.assertEqual(frame["session"].iloc[0], ed.AFTER_CLOSE)

    def test_nothing_in_nothing_out(self):
        self.assertIsNone(ed.normalize(None))
        self.assertIsNone(ed.normalize(raw_rows()))


class IndicatorArithmeticTests(unittest.TestCase):
    bars = pd.bdate_range("2024-01-01", periods=30)

    def columns(self, *rows, bars=None):
        bars = self.bars if bars is None else bars
        return ed.earnings_columns(bars, history(*rows))

    def test_a_report_after_the_close_reacts_the_next_session(self):
        to, since = self.columns(("2024-01-10", ed.AFTER_CLOSE))   # reaction Thursday 11th
        day = {d.strftime("%m-%d"): i for i, d in enumerate(self.bars)}
        self.assertEqual(to[day["01-10"]], 1)     # buying that close is exposed to the report
        self.assertEqual(to[day["01-11"]], 0)
        self.assertEqual((since[day["01-11"]], since[day["01-12"]]), (0, 1))
        self.assertEqual(since[day["01-10"]], NO)

    def test_a_report_before_the_open_reacts_that_day(self):
        to, since = self.columns(("2024-01-10", ed.BEFORE_OPEN))
        day = {d.strftime("%m-%d"): i for i, d in enumerate(self.bars)}
        self.assertEqual((to[day["01-09"]], to[day["01-10"]], since[day["01-10"]]), (1, 0, 0))

    def test_friday_after_the_close_reacts_on_monday(self):
        to, _ = self.columns(("2024-01-12", ed.AFTER_CLOSE))
        day = {d.strftime("%m-%d"): i for i, d in enumerate(self.bars)}
        self.assertEqual((to[day["01-12"]], to[day["01-15"]]), (1, 0))

    def test_a_trading_holiday_in_the_calendar_is_skipped(self):
        bars = pd.bdate_range("2024-01-01", periods=10).delete(4)   # no bar on Friday the 5th
        to, _ = ed.earnings_columns(bars, history(("2024-01-04", ed.AFTER_CLOSE)))
        names = [d.strftime("%m-%d") for d in bars]
        self.assertEqual(to[names.index("01-04")], 1)
        self.assertEqual(to[names.index("01-08")], 0)   # the next bar after the 4th is Monday the 8th

    def test_where_nothing_is_known_both_read_999(self):
        for hist in (None, history()):
            to, since = ed.earnings_columns(self.bars, hist)
            self.assertTrue((to == NO).all() and (since == NO).all())

    def test_bars_after_the_last_known_report_have_no_next_one(self):
        to, since = self.columns(("2024-01-10", ed.AFTER_CLOSE))
        self.assertTrue((to[self.bars > "2024-01-11"] == NO).all())
        self.assertEqual(since[-1], len(self.bars) - 1 - 8)

    def test_a_reaction_after_the_last_bar_is_counted_in_weekdays(self):
        bars = pd.bdate_range("2024-01-01", periods=5)           # Mon 1st .. Fri 5th
        to, _ = ed.earnings_columns(bars, history(("2024-01-10", ed.BEFORE_OPEN)))   # Wed 10th
        self.assertEqual(to.tolist(), [7, 6, 5, 4, 3])
        to, _ = ed.earnings_columns(bars, history(("2024-01-05", ed.AFTER_CLOSE)))   # reaction Mon 8th
        self.assertEqual(to[-1], 1)

    def test_a_reaction_before_the_first_bar_is_not_on_this_history(self):
        to, since = self.columns(("2023-12-20", ed.AFTER_CLOSE), ("2024-01-10", ed.AFTER_CLOSE))
        self.assertEqual(since[0], NO)                      # nothing before the first bar is counted
        self.assertEqual(to[0], 8)

    def test_a_report_the_evening_before_the_first_bar_reacts_on_it(self):
        bars = pd.bdate_range("2024-01-02", periods=6)
        to, since = ed.earnings_columns(bars, history(("2024-01-01", ed.AFTER_CLOSE)))   # reaction Tue 2nd
        self.assertEqual((to[0], since[0]), (0, 0))

    def test_values_never_exceed_999(self):
        bars = pd.bdate_range("2000-01-03", periods=3000)
        to, since = ed.earnings_columns(bars, history(("2010-06-01", ed.BEFORE_OPEN)))
        self.assertEqual(max(to.max(), since.max()), NO)

    def test_matches_a_brute_force_reference_on_random_calendars(self):
        rng = np.random.default_rng(5)
        for trial in range(60):
            bars = pd.bdate_range("2023-01-02", periods=int(rng.integers(40, 220)))
            keep = rng.random(len(bars)) > 0.04                       # random holidays
            keep[0] = True
            bars = bars[keep]
            span = (bars[-1] - bars[0]).days
            offsets = rng.integers(-40, span + 60, size=int(rng.integers(0, 8)))
            rows = [((bars[0] + pd.Timedelta(days=int(o))).normalize(), ed.AFTER_CLOSE if rng.random() < 0.6 else ed.BEFORE_OPEN)
                    for o in offsets]
            # one report per day, like normalize() leaves
            rows = list({r[0]: r for r in rows}.values())
            # a report on a weekend is not something Yahoo lists; keep the reference and the code on weekdays
            rows = [r for r in rows if r[0].dayofweek < 5]
            frame = history(*rows) if rows else history()
            want_to, want_since = brute_force(bars, rows)
            got_to, got_since = ed.earnings_columns(bars, frame)
            np.testing.assert_array_equal(got_to, want_to, err_msg=f"trial {trial} to {rows}")
            np.testing.assert_array_equal(got_since, want_since, err_msg=f"trial {trial} since {rows}")

    def test_timezone_aware_bar_stamps_are_read_in_eastern_time(self):
        bars = pd.bdate_range("2024-01-01", periods=10, tz="UTC") + pd.Timedelta(hours=1)   # 20:00 the evening before, in New York
        to_aware, _ = ed.earnings_columns(bars, history(("2024-01-04", ed.AFTER_CLOSE)))
        naive_days = pd.DatetimeIndex([d.tz_convert(ET).tz_localize(None).normalize() for d in bars])
        to_naive, _ = ed.earnings_columns(naive_days, history(("2024-01-04", ed.AFTER_CLOSE)))
        np.testing.assert_array_equal(to_aware, to_naive)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        patcher = patch.object(ed, "CACHE_DIR", Path(self.folder.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        ed.clear_memory_cache()
        self.addCleanup(ed.clear_memory_cache)
        self.now = datetime(2026, 4, 13, 12, 0, tzinfo=ET)   # a month with no clock change in it

    def stub(self, *stamps):
        return patch.object(ed, "fetch_earnings", side_effect=lambda symbol: raw_rows(*stamps) if stamps else None)

    def test_first_load_downloads_and_caches_then_reads_from_memory(self):
        with self.stub("2026-01-29 16:00", "2025-10-30 16:00") as fetch:
            frame = ed.load_earnings("aapl", now=self.now)
            again = ed.load_earnings("AAPL", now=self.now + timedelta(hours=1))
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(frame["date"].dt.strftime("%Y-%m-%d").tolist(), ["2025-10-30", "2026-01-29"])
        pd.testing.assert_frame_equal(frame, again)
        self.assertTrue((Path(self.folder.name) / "AAPL.json").exists())

    def test_a_fresh_disk_cache_is_used_without_downloading_after_a_restart(self):
        with self.stub("2026-01-29 16:00") as fetch:
            ed.load_earnings("AAPL", now=self.now)
            ed.clear_memory_cache()
            frame = ed.load_earnings("AAPL", now=self.now + timedelta(days=3))
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(frame["session"].tolist(), [ed.AFTER_CLOSE])
        self.assertEqual(frame["date"].iloc[0], pd.Timestamp("2026-01-29"))

    def test_the_cache_survives_a_round_trip_through_the_file(self):
        raw = raw_rows("2026-01-29 16:00", "2025-10-30 07:00", **{"EPS Estimate": [1.9, 1.5], "Reported EPS": [2.0, np.nan]})
        with patch.object(ed, "fetch_earnings", return_value=raw):
            first = ed.load_earnings("AAPL", now=self.now)
        ed.clear_memory_cache()
        with patch.object(ed, "fetch_earnings", side_effect=AssertionError("cached")):
            second = ed.load_earnings("AAPL", now=self.now + timedelta(days=1))
        pd.testing.assert_frame_equal(first, second)

    def test_a_clock_change_does_not_shorten_the_refresh_interval(self):
        before_the_change = datetime(2026, 3, 2, 12, 0, tzinfo=ET)    # the US clocks moved on 2026-03-08
        with self.stub("2026-01-29 16:00") as fetch:
            ed.load_earnings("AAPL", now=before_the_change)
            ed.load_earnings("AAPL", now=before_the_change + timedelta(days=6, hours=23))   # 6 days 22 hours of real time
            self.assertEqual(fetch.call_count, 1)
            ed.clear_memory_cache()
            ed.load_earnings("AAPL", now=before_the_change + timedelta(days=7, hours=1, minutes=1))
        self.assertEqual(fetch.call_count, 2)

    def test_a_week_old_cache_is_refreshed(self):
        with self.stub("2026-01-29 16:00") as fetch:
            ed.load_earnings("AAPL", now=self.now)
            ed.load_earnings("AAPL", now=self.now + timedelta(days=ed.REFRESH_DAYS, minutes=1))
        self.assertEqual(fetch.call_count, 2)

    def test_a_failed_refresh_keeps_the_old_data(self):
        with self.stub("2026-01-29 16:00"):
            ed.load_earnings("AAPL", now=self.now)
        with patch.object(ed, "fetch_earnings", side_effect=RuntimeError("yahoo is down")):
            frame = ed.load_earnings("AAPL", now=self.now + timedelta(days=30))
        self.assertEqual(frame["date"].iloc[0], pd.Timestamp("2026-01-29"))

    def test_a_failed_first_download_is_none_and_not_remembered_as_no_earnings(self):
        with patch.object(ed, "fetch_earnings", side_effect=RuntimeError("yahoo is down")):
            self.assertIsNone(ed.load_earnings("AAPL", now=self.now))
        self.assertFalse((Path(self.folder.name) / "AAPL.json").exists())
        with self.stub("2026-01-29 16:00") as fetch:
            self.assertIsNotNone(ed.load_earnings("AAPL", now=self.now + timedelta(minutes=5)))
        self.assertEqual(fetch.call_count, 1)

    def test_a_symbol_with_no_earnings_is_rechecked_monthly_not_every_run(self):
        with self.stub() as fetch:
            self.assertIsNone(ed.load_earnings("SOXX", now=self.now))
            self.assertIsNone(ed.load_earnings("SOXX", now=self.now + timedelta(days=20)))
            self.assertEqual(fetch.call_count, 1)
            ed.clear_memory_cache()
            self.assertIsNone(ed.load_earnings("SOXX", now=self.now + timedelta(days=ed.EMPTY_REFRESH_DAYS, minutes=1)))
            self.assertEqual(fetch.call_count, 2)

    def test_indexes_futures_and_crypto_are_never_looked_up(self):
        with patch.object(ed, "fetch_earnings", side_effect=AssertionError("no lookup")):
            for symbol in ("^VIX", "GC=F", "BTC-USD", "", "  "):
                self.assertIsNone(ed.load_earnings(symbol, now=self.now), symbol)
        for symbol in ("AAPL", "BRK-B", "RIVN"):
            self.assertTrue(ed.can_have_earnings(symbol))

    def test_a_corrupt_cache_file_is_ignored_and_replaced(self):
        (Path(self.folder.name) / "AAPL.json").write_text("{not json")
        with self.stub("2026-01-29 16:00") as fetch:
            frame = ed.load_earnings("AAPL", now=self.now)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(len(frame), 1)
        json.loads((Path(self.folder.name) / "AAPL.json").read_text())   # valid again

    def test_prefetch_warms_every_symbol_once_and_skips_what_cannot_have_earnings(self):
        with patch.object(ed, "fetch_earnings", side_effect=lambda s: raw_rows("2026-01-29 16:00")) as fetch:
            ed.prefetch(["AAPL", "MSFT", "aapl", "^VIX", "GC=F"])
            self.assertEqual(sorted(call.args[0] for call in fetch.call_args_list), ["AAPL", "MSFT"])
            ed.prefetch(["AAPL", "MSFT"])
            self.assertEqual(fetch.call_count, 2)

    def test_fetching_pages_through_yahoo_until_a_short_page(self):
        first = raw_rows(*[(pd.Timestamp("2010-01-04") + pd.Timedelta(days=90 * k)).strftime("%Y-%m-%d") + " 16:00" for k in range(ed.PAGE_SIZE)])
        second = raw_rows("2009-10-01 16:00", "2009-07-01 16:00")
        calls = []

        class Ticker:
            def get_earnings_dates(self, limit, offset=0):
                calls.append((limit, offset))
                return {0: first, ed.PAGE_SIZE: second}.get(offset)

        with patch.object(ed, "_ticker", lambda symbol: Ticker()):
            frame = ed.fetch_earnings("AAPL")
        self.assertEqual(calls, [(ed.PAGE_SIZE, 0), (ed.PAGE_SIZE, ed.PAGE_SIZE)])
        self.assertEqual(len(frame), ed.PAGE_SIZE + 2)

    def test_an_older_yfinance_that_cannot_page_still_gives_the_first_hundred(self):
        first = raw_rows(*[(pd.Timestamp("2010-01-04") + pd.Timedelta(days=90 * k)).strftime("%Y-%m-%d") + " 16:00" for k in range(ed.PAGE_SIZE)])

        class OldTicker:
            def get_earnings_dates(self, limit=12):
                return first

        with patch.object(ed, "_ticker", lambda symbol: OldTicker()):
            self.assertEqual(len(ed.fetch_earnings("AAPL")), ed.PAGE_SIZE)

    def test_fetching_uses_the_yahoo_symbol_and_returns_none_when_yahoo_has_nothing(self):
        seen = []

        class Ticker:
            def get_earnings_dates(self, limit, offset=0):
                return None

        with patch.object(ed, "_ticker", lambda symbol: seen.append(symbol) or Ticker()):
            self.assertIsNone(ed.fetch_earnings("soxx"))
        self.assertEqual(seen, ["SOXX"])

    def test_a_type_error_on_the_first_page_is_a_real_error_not_an_old_library(self):
        class Ticker:
            def get_earnings_dates(self, limit, offset=0):
                raise TypeError("unexpected response")

        with patch.object(ed, "_ticker", lambda symbol: Ticker()), self.assertRaises(TypeError):
            ed.fetch_earnings("AAPL")

    def test_an_error_from_yahoo_is_raised_not_swallowed(self):
        class Ticker:
            def get_earnings_dates(self, limit, offset=0):
                raise RuntimeError("429 too many requests")

        with patch.object(ed, "_ticker", lambda symbol: Ticker()), self.assertRaises(RuntimeError):
            ed.fetch_earnings("AAPL")


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        ed.clear_memory_cache()
        self.addCleanup(ed.clear_memory_cache)

    def daily_frame(self, bars=300):
        dates = pd.bdate_range("2023-01-02", periods=bars)
        rng = np.random.default_rng(2)
        close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, bars))
        frame = pd.DataFrame({"Date": dates, "Open": close * 0.999, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1000.0})
        return stub_market_context_columns(frame)

    def test_add_indicators_fills_both_columns_for_a_stock(self):
        hist = history(("2023-03-02", ed.AFTER_CLOSE), ("2023-06-01", ed.BEFORE_OPEN))
        with patch.object(ed, "load_earnings", return_value=hist):
            frame = ind.add_indicators(self.daily_frame(), compute_hurst=False, symbol="XYZ")
        day = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(frame["Date"])}
        self.assertEqual(frame["DaysToEarnings"].iloc[day["2023-03-02"]], 1)
        self.assertEqual(frame["DaysToEarnings"].iloc[day["2023-03-03"]], 0)
        self.assertEqual(frame["DaysSinceEarnings"].iloc[day["2023-03-06"]], 1)
        self.assertEqual(frame["DaysToEarnings"].iloc[day["2023-05-31"]], 1)
        self.assertEqual(frame["DaysToEarnings"].iloc[-1], NO)
        self.assertEqual(frame["DaysSinceEarnings"].iloc[0], NO)
        self.assertFalse(frame[["DaysToEarnings", "DaysSinceEarnings"]].isna().any().any())

    def test_without_a_symbol_or_for_an_etf_both_columns_are_999(self):
        with patch.object(ed, "fetch_earnings", return_value=None):
            for symbol in (None, "SOXX"):
                frame = ind.add_indicators(self.daily_frame(), compute_hurst=False, symbol=symbol)
                self.assertTrue((frame["DaysToEarnings"] == NO).all(), symbol)
                self.assertTrue((frame["DaysSinceEarnings"] == NO).all(), symbol)

    def test_an_intraday_custom_dataset_never_looks_anything_up(self):
        frame = self.daily_frame()
        with patch.object(ed, "load_earnings", side_effect=AssertionError("no lookup")):
            out = ind.add_indicators(frame, compute_hurst=False, source_timezone="America/New_York", symbol="XYZ")
        self.assertTrue((out["DaysToEarnings"] == NO).all())

    def test_a_rule_on_an_etf_never_blocks_and_never_exits(self):
        with patch.object(ed, "fetch_earnings", return_value=None):
            frame = ind.add_indicators(self.daily_frame(), compute_hurst=False, symbol="SOXX")
        self.assertTrue((frame["DaysToEarnings"] > 3).all())
        self.assertFalse((frame["DaysToEarnings"] <= 1).any())

    def test_the_cache_version_was_bumped_for_the_new_columns(self):
        self.assertGreaterEqual(ind.INDICATOR_CACHE_VERSION, 4)

    def test_coverage_reports_gaps_and_the_upcoming_report(self):
        today = pd.Timestamp(datetime.now(ET).date())
        days = [today - pd.Timedelta(days=d) for d in (400, 310, 220, 40)] + [today + pd.Timedelta(days=30)]
        hist = history(*[(d, ed.AFTER_CLOSE) for d in days])
        with patch.object(ed, "load_earnings", return_value=hist):
            report = ed.coverage("XYZ")
        self.assertEqual(report["reports"], 4)
        self.assertEqual(len(report["gaps"]), 1)
        self.assertEqual(report["gaps"][0]["days"], 180)
        self.assertEqual(report["upcoming"], (today + pd.Timedelta(days=30)).strftime("%Y-%m-%d"))
        with patch.object(ed, "load_earnings", return_value=None):
            self.assertEqual(ed.coverage("SOXX")["reports"], 0)


class CatalogTests(unittest.TestCase):
    def test_both_indicators_are_in_the_builder_as_numbers_only(self):
        for indicator_id in ("DaysToEarnings", "DaysSinceEarnings"):
            item = get_indicator_def(indicator_id)
            self.assertEqual(item["category"], "Earnings")
            self.assertEqual(item["compareMode"], "number")
            self.assertIn(indicator_id, {i["id"] for i in list_indicators(builder_only=True, include_regimes=False)})
        self.assertEqual(get_indicator_def("DaysToEarnings")["defaultCompareNumber"], "5")

    def test_they_describe_the_traded_symbol_and_are_unavailable_for_custom_data(self):
        for indicator_id in ("DaysToEarnings", "DaysSinceEarnings"):
            self.assertTrue(is_market_wide_indicator(indicator_id))   # applied on the primary frame, not per confirm symbol
            self.assertIn(indicator_id, unavailable_indicator_ids())

    def test_they_are_comparable_to_a_number_and_not_to_another_indicator(self):
        from api.indicator_catalog import validate_compare_target

        validate_compare_target("DaysToEarnings", "3")
        with self.assertRaises(ValueError):
            validate_compare_target("DaysToEarnings", "SMA200")

    def test_the_sweep_lists_them(self):
        from indicator_sweep import PRICE_SWEEPS

        names = {spec[0] for spec in PRICE_SWEEPS}
        self.assertIn("DaysToEarnings", names)
        self.assertIn("DaysSinceEarnings", names)


if __name__ == "__main__":
    unittest.main()
