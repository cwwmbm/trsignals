"""Market regime bucket edges, breadth series, and trade-label attachment."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from api.market_regimes import (
    ATR_CONTRACTING,
    ATR_EXPANDING,
    SPY_BEAR,
    SPY_BULL,
    build_regime_calendar,
    classify_atr,
    classify_breadth,
    classify_sector_breadth,
    classify_spy,
    classify_vix,
    classify_vxn,
    clear_regime_calendar_cache,
    compute_market_regime_sharpe,
    current_market_regimes,
    load_regime_calendar,
    regime_coverage,
    market_regimes_for_timestamp,
    sharpe_ratio,
    sortino_ratio,
)
from api.serializers import detailed_backtest_payload, trade_payload


def _calendar_inputs(periods=220):
    idx = pd.bdate_range("2018-01-02", periods=periods)
    spy = pd.Series(100.0, index=idx)
    # Uneven swings so log(RSP/SPY) and the raw ratio do not share an RSI path.
    cycle = np.sin(np.linspace(0, 18, periods))
    rsp = pd.Series(50 + 15 * cycle + 8 * np.sin(np.linspace(0, 4, periods)), index=idx)
    vix = pd.Series(16.0, index=idx)
    vxn = pd.Series(22.0, index=idx)
    return vix, vxn, spy, rsp, idx


class BucketEdgeTests(unittest.TestCase):
    def test_vix_edges_belong_to_the_lower_bucket(self):
        self.assertIsNone(classify_vix(None))
        self.assertIsNone(classify_vix(float("nan")))
        self.assertEqual(classify_vix(15), "le_15")
        self.assertEqual(classify_vix(15.01), "15_20")
        self.assertEqual(classify_vix(20), "15_20")
        self.assertEqual(classify_vix(20.01), "20_30")
        self.assertEqual(classify_vix(30), "20_30")
        self.assertEqual(classify_vix(30.01), "gt_30")

    def test_vxn_uses_the_same_buckets_as_vix(self):
        self.assertEqual(classify_vxn(15), "le_15")
        self.assertEqual(classify_vxn(20), "15_20")
        self.assertEqual(classify_vxn(30), "20_30")
        self.assertEqual(classify_vxn(30.01), "gt_30")

    def test_breadth_edges_belong_to_the_lower_bucket(self):
        self.assertEqual(classify_breadth(39.99), "lt_40")
        self.assertEqual(classify_breadth(40), "40_50")
        self.assertEqual(classify_breadth(50), "40_50")
        self.assertEqual(classify_breadth(50.01), "50_60")
        self.assertEqual(classify_breadth(60), "50_60")
        self.assertEqual(classify_breadth(60.01), "gt_60")
        self.assertEqual(classify_breadth(70), "gt_60")
        self.assertEqual(classify_breadth(70.01), "gt_60")

    def test_sector_breadth_edges_belong_to_the_lower_bucket(self):
        self.assertIsNone(classify_sector_breadth(None))
        self.assertEqual(classify_sector_breadth(0.25), "le_25")
        self.assertEqual(classify_sector_breadth(0.2501), "25_50")
        self.assertEqual(classify_sector_breadth(0.50), "25_50")
        self.assertEqual(classify_sector_breadth(0.75), "50_75")
        self.assertEqual(classify_sector_breadth(0.7501), "gt_75")

    def test_spy_equality_is_bull(self):
        self.assertEqual(classify_spy(100, 100), SPY_BULL)
        self.assertEqual(classify_spy(100.01, 100), SPY_BULL)
        self.assertEqual(classify_spy(99.99, 100), SPY_BEAR)
        self.assertIsNone(classify_spy(float("nan"), 100))

    def test_atr_tie_is_contracting(self):
        self.assertEqual(classify_atr(1.5, 1.0), ATR_EXPANDING)
        self.assertEqual(classify_atr(1.0, 1.0), ATR_CONTRACTING)
        self.assertEqual(classify_atr(1.0, 1.5), ATR_CONTRACTING)
        self.assertIsNone(classify_atr(None, 1.0))


class CalendarTests(unittest.TestCase):
    def test_log_rsi_differs_from_ratio_rsi(self):
        vix, vxn, spy, rsp, _idx = _calendar_inputs()
        frame = build_regime_calendar(vix, vxn, spy, rsp)
        old = frame["breadth_old"].dropna()
        new = frame["breadth_new"].dropna()
        self.assertGreater(len(old), 0)
        self.assertGreater(len(new), 0)
        self.assertFalse(np.allclose(old.to_numpy(), new.reindex(old.index).to_numpy()))

    def test_flat_spy_sma_equality_is_bull(self):
        vix, vxn, spy, rsp, _idx = _calendar_inputs()
        frame = build_regime_calendar(vix, vxn, spy, rsp)
        last = frame.dropna(subset=["spy_sma50", "spy_sma200"]).iloc[-1]
        self.assertEqual(classify_spy(last["spy_sma50"], last["spy_sma200"]), SPY_BULL)

    def test_holiday_gap_does_not_blank_later_sma(self):
        vix, vxn, spy, rsp, _idx = _calendar_inputs()
        spy = spy.copy()
        spy.iloc[100] = float("nan")
        frame = build_regime_calendar(vix, vxn, spy, rsp)
        last = frame.dropna(subset=["spy_sma50", "spy_sma200"]).iloc[-1]
        self.assertEqual(classify_spy(last["spy_sma50"], last["spy_sma200"]), SPY_BULL)
        self.assertNotIn(spy.index[100], frame.index)

    def test_recent_drop_is_bear(self):
        vix, vxn, spy, rsp, _idx = _calendar_inputs()
        spy = spy.copy()
        spy.iloc[-30:] = 50.0
        frame = build_regime_calendar(vix, vxn, spy, rsp)
        last = frame.dropna(subset=["spy_sma50", "spy_sma200"]).iloc[-1]
        self.assertEqual(classify_spy(last["spy_sma50"], last["spy_sma200"]), SPY_BEAR)

    def test_sector_breadth_counts_closes_strictly_above_sma(self):
        idx = pd.bdate_range("2020-01-02", periods=60)
        rising = pd.Series(np.arange(1, 61, dtype=float), index=idx)
        flat = pd.Series(100.0, index=idx)
        gapped = rising.copy()
        gapped.iloc[30] = float("nan")
        sectors = [rising, rising, rising, gapped, flat, flat, flat, flat, flat]
        level = pd.Series(100.0, index=idx)
        frame = build_regime_calendar(level, level, level, level, sectors)
        # Four rising sectors finish above their SMA(50). A flat close equals its SMA
        # and does not count. The holiday gap in the fourth sector does not blank it.
        self.assertAlmostEqual(frame["sector_breadth_50"].iloc[-1], 4 / 9)
        self.assertEqual(classify_sector_breadth(frame["sector_breadth_50"].iloc[-1]), "25_50")
        self.assertTrue(pd.isna(frame["sector_breadth_200"].iloc[-1]))
        self.assertTrue(pd.isna(frame["sector_breadth_50"].iloc[48]))

    def test_semis_breadth_is_rsi_of_log_smh_over_spy(self):
        idx = pd.bdate_range("2020-01-02", periods=40)
        spy = pd.Series(100.0, index=idx)
        smh = pd.Series(np.linspace(40, 90, 40), index=idx)
        frame = build_regime_calendar(spy, spy, spy, spy, smh=smh)
        last = frame["breadth_semis"].dropna().iloc[-1]
        self.assertGreater(last, 70)
        self.assertEqual(classify_breadth(last), "gt_60")
        blank = build_regime_calendar(spy, spy, spy, spy)
        self.assertTrue(blank["breadth_semis"].isna().all())

    def test_equity_and_credit_risk_use_log_ratio_rsi(self):
        idx = pd.bdate_range("2020-01-02", periods=40)
        flat = pd.Series(100.0, index=idx)
        rising = pd.Series(np.linspace(40, 90, 40), index=idx)
        frame = build_regime_calendar(flat, flat, flat, flat, xly=rising, xlp=flat, hyg=rising, lqd=flat)
        equity = frame["breadth_equity_risk"].dropna().iloc[-1]
        credit = frame["breadth_credit_risk"].dropna().iloc[-1]
        self.assertGreater(equity, 70)
        self.assertGreater(credit, 70)
        self.assertEqual(classify_breadth(equity), "gt_60")
        self.assertEqual(classify_breadth(credit), "gt_60")

    def test_added_ratio_regimes_are_log_rsi(self):
        idx = pd.bdate_range("2020-01-02", periods=40)
        flat = pd.Series(100.0, index=idx)
        rising = pd.Series(np.linspace(40, 90, 40), index=idx)
        cases = (
            ("breadth_credit_risk_on", {"hyg": rising, "tlt": flat}),
            ("breadth_bond_duration", {"tlt": rising, "shy": flat}),
            ("breadth_copper_gold", {"copper": rising, "gold": flat}),
            ("breadth_materials", {"xlb": rising}),
        )
        for column, kwargs in cases:
            frame = build_regime_calendar(flat, flat, flat, flat, **kwargs)
            last = frame[column].dropna().iloc[-1]
            self.assertGreater(last, 70, column)
            self.assertEqual(classify_breadth(last), "gt_60", column)
        blank = build_regime_calendar(flat, flat, flat, flat)
        for column, _kwargs in cases:
            self.assertTrue(blank[column].isna().all(), column)

    def test_credit_breadth_is_blank_until_hyg_is_listed(self):
        idx = pd.bdate_range("2020-01-02", periods=40)
        flat = pd.Series(100.0, index=idx)
        hyg = pd.Series(np.linspace(40, 90, 20), index=idx[20:])
        frame = build_regime_calendar(flat, flat, flat, flat, hyg=hyg, lqd=flat)
        self.assertTrue(frame["breadth_credit_risk"].iloc[:33].isna().all())
        self.assertTrue(pd.notna(frame["breadth_credit_risk"].iloc[33]))
        before = market_regimes_for_timestamp(idx[10], frame)
        self.assertIsNone(before["credit_risk_breadth_regime"])
        after = market_regimes_for_timestamp(idx[33], frame)
        self.assertEqual(after["credit_risk_breadth_regime"], "gt_60")

    def test_sector_breadth_waits_until_every_sector_is_listed(self):
        idx = pd.bdate_range("2020-01-02", periods=100)
        early = pd.Series(np.arange(1, 101, dtype=float), index=idx)
        late = pd.Series(np.arange(1, 61, dtype=float), index=idx[40:])
        level = pd.Series(100.0, index=idx)
        frame = build_regime_calendar(level, level, level, level, [early] * 8 + [late])
        self.assertTrue(frame["sector_breadth_50"].iloc[:89].isna().all())
        self.assertTrue(pd.notna(frame["sector_breadth_50"].iloc[89]))

    def test_lookup_uses_the_entry_session_date(self):
        idx = pd.bdate_range("2024-01-02", periods=5)
        frame = pd.DataFrame(
            {
                "vix": [10.0, 16.0, 36.0, 14.0, 22.0],
                "vxn": [18.0, 22.0, 46.0, 28.0, 33.0],
                "spy_sma50": [10, 10, 9, 10, 10],
                "spy_sma200": [10, 10, 10, 10, 10],
                "breadth_old": [20, 40, 60, 80, 81],
                "breadth_new": [10, 25, 55, 70, 90],
            },
            index=idx,
        )
        labels = market_regimes_for_timestamp("2024-01-04 10:15", frame)
        self.assertEqual(labels["vix_regime"], "gt_30")
        self.assertEqual(labels["vxn_regime"], "gt_30")
        self.assertEqual(labels["spy_regime"], SPY_BEAR)
        self.assertEqual(labels["breadth_old_regime"], "50_60")
        self.assertEqual(labels["breadth_new_regime"], "50_60")
        missing = market_regimes_for_timestamp("2024-02-01", frame)
        self.assertTrue(all(value is None for value in missing.values()))

    def setUp(self):
        clear_regime_calendar_cache()

    def tearDown(self):
        clear_regime_calendar_cache()

    def test_successful_download_is_cached_for_the_day(self):
        vix, vxn, spy, rsp, _idx = _calendar_inputs()
        frame = build_regime_calendar(vix, vxn, spy, rsp)
        with patch("api.market_regimes._fetch_regime_calendar", return_value=frame) as fetch:
            first = load_regime_calendar(5)
            second = load_regime_calendar(5)
        self.assertIs(first, second)
        self.assertEqual(fetch.call_count, 1)

    def test_failed_download_is_not_cached(self):
        with patch(
            "api.market_regimes._fetch_regime_calendar",
            side_effect=RuntimeError("offline"),
        ) as fetch:
            self.assertIsNone(load_regime_calendar(5))
            self.assertIsNone(load_regime_calendar(5))
        self.assertEqual(fetch.call_count, 2)


def _executed_frame():
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "Close": [100.0, 101.0, 102.0],
            "RollingPnL": [1.0, 1.01, 1.02],
            "Drawdown": [0.0, 0.0, 0.01],
            "LongTradeIn": [True, False, False],
            "LongTradeOut": [False, True, False],
            "HoldLong": [True, False, False],
            "TradePnL": [0.0, 0.02, 0.0],
            "DaysInTrade": [0, 1, 0],
            "ATR20": [2.0, 1.0, 1.2],
            "ATR50": [1.0, 2.0, 1.2],
        }
    )


class RegimeSharpeTests(unittest.TestCase):
    def test_sharpe_matches_annualized_mean_over_std(self):
        returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
        mean = float(returns.mean())
        std = float(returns.std(ddof=1))
        expected = (252 ** 0.5) * mean / std
        self.assertAlmostEqual(sharpe_ratio(returns), expected)
        self.assertIsNone(sharpe_ratio(pd.Series([0.01])))
        self.assertEqual(sharpe_ratio(pd.Series([0.01, 0.01])), 0.0)

    def test_buckets_use_that_days_return_only(self):
        dates = pd.bdate_range("2024-01-02", periods=6)
        data = pd.DataFrame(
            {
                "Date": dates,
                "RollingPnL": [1.0, 1.01, 1.02, 1.00, 1.03, 1.01],
                "Drawdown": [0.01, 0.02, 0.05, 0.20, 0.10, 0.08],
                "ATR20": [2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
                "ATR50": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            }
        )
        calendar = pd.DataFrame(
            {
                "vix": [10.0, 10.0, 10.0, 40.0, 40.0, 40.0],
                "vxn": [18.0, 18.0, 18.0, 18.0, 18.0, 18.0],
                "spy_sma50": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                "spy_sma200": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "breadth_old": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                "breadth_new": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
            },
            index=dates,
        )
        result = compute_market_regime_sharpe(data, calendar)
        vix = {row["key"]: row for row in result["vix"]}
        # First bar has no return. The next two closes are still VIX <= 15.
        self.assertEqual(vix["le_15"]["days"], 2)
        self.assertEqual(vix["gt_30"]["days"], 3)
        self.assertIsNone(vix["le_15"]["max_drawdown"])
        self.assertIsNone(vix["gt_30"]["max_drawdown"])
        self.assertIsNone(vix["le_15"]["sharpe"])
        self.assertIsNone(vix["gt_30"]["sharpe"])

        atr = {row["key"]: row for row in result["atr"]}
        self.assertEqual(atr["expanding"]["days"], 2)
        self.assertEqual(atr["contracting"]["days"], 3)
        self.assertTrue(all(row["days"] == 0 for row in result["atr"] if row["key"] not in {"expanding", "contracting"}))

        breadth = {row["key"]: row for row in result["breadth_new"]}
        self.assertEqual(breadth["lt_40"]["days"], 2)
        self.assertEqual(breadth["gt_60"]["days"], 3)

    def test_max_drawdown_follows_the_entry_regime_through_the_trade(self):
        # Trade 1 is entered while VIX <= 15, then held on VIX > 30 days and loses 10%.
        # Trade 2 is entered while VIX > 30, then held on VIX <= 15 days and loses 20%.
        # The underwater days belong to the other regime; the drawdown stays with the entry.
        dates = pd.bdate_range("2024-01-02", periods=7)
        data = pd.DataFrame(
            {
                "Date": dates,
                "RollingPnL": [1.0, 1.0, 0.9, 0.945, 0.945, 0.756, 0.76356],
                "Drawdown": [0.0, 0.0, 0.10, 0.055, 0.055, 0.244, 0.23644],
                "LongTradeIn": [False, True, False, False, True, False, False],
                "HoldLong": [False, False, True, True, False, True, True],
                "LongTradeOut": [False, False, False, True, False, False, True],
                "ATR20": [1.0] * 7,
                "ATR50": [2.0] * 7,
            }
        )
        calendar = pd.DataFrame(
            {
                "vix": [10.0, 10.0, 40.0, 40.0, 40.0, 10.0, 10.0],
                "vxn": [18.0] * 7,
                "spy_sma50": [2.0] * 7,
                "spy_sma200": [1.0] * 7,
                "breadth_old": [50.0] * 7,
                "breadth_new": [50.0] * 7,
            },
            index=dates,
        )
        result = compute_market_regime_sharpe(data, calendar)
        vix = {row["key"]: row for row in result["vix"]}
        self.assertAlmostEqual(vix["le_15"]["max_drawdown"], 0.10)
        self.assertAlmostEqual(vix["gt_30"]["max_drawdown"], 0.20)
        self.assertIsNone(vix["15_20"]["max_drawdown"])
        # Holding-day returns only: trade 1 is -10% then +5%, trade 2 is -20% then +1%.
        # The VIX > 30 days inside trade 1 do not enter the VIX <= 15 Sharpe.
        self.assertAlmostEqual(vix["le_15"]["sharpe"], sharpe_ratio(pd.Series([-0.10, 0.05])))
        self.assertAlmostEqual(vix["gt_30"]["sharpe"], sharpe_ratio(pd.Series([-0.20, 0.01])))
        self.assertIsNone(vix["15_20"]["sharpe"])
        self.assertAlmostEqual(vix["le_15"]["sortino"], sortino_ratio(pd.Series([-0.10, 0.05])))
        self.assertAlmostEqual(vix["gt_30"]["sortino"], sortino_ratio(pd.Series([-0.20, 0.01])))
        self.assertIsNone(vix["15_20"]["sortino"])
        # CAGR compounds only that entry's holding bars, annualized over the full 7-bar sample.
        # Trade 1 ends at 0.945. Trade 2 ends at 0.808. Later bars do not leak across regimes.
        le_15_cagr = (0.945 ** (252 / 7) - 1) * 100
        gt_30_cagr = (0.808 ** (252 / 7) - 1) * 100
        self.assertAlmostEqual(vix["le_15"]["cagr"] * 100, le_15_cagr)
        self.assertAlmostEqual(vix["gt_30"]["cagr"] * 100, gt_30_cagr)
        self.assertIsNone(vix["15_20"]["cagr"])
        self.assertAlmostEqual(vix["le_15"]["calmar"], le_15_cagr / 10.0)
        self.assertAlmostEqual(vix["gt_30"]["calmar"], gt_30_cagr / 20.0)
        self.assertIsNone(vix["15_20"]["calmar"])

    def test_current_regime_uses_the_latest_labeled_session(self):
        dates = pd.bdate_range("2024-01-02", periods=3)
        data = pd.DataFrame(
            {
                "Date": dates,
                "RollingPnL": [1.0, 1.0, 1.0],
                "ATR20": [1.0, 1.0, 2.0],
                "ATR50": [2.0, 2.0, 1.0],
            }
        )
        calendar = pd.DataFrame(
            {
                "vix": [10.0, 10.0, 40.0],
                "vxn": [18.0, 18.0, 18.0],
                "spy_sma50": [2.0, 2.0, 1.0],
                "spy_sma200": [1.0, 1.0, 2.0],
                "breadth_old": [50.0, 50.0, 30.0],
                "breadth_new": [50.0, 50.0, 75.0],
            },
            index=dates,
        )
        current = current_market_regimes(data, calendar)
        self.assertEqual(current["as_of"], "2024-01-04")
        self.assertEqual(current["regimes"]["vix"], "gt_30")
        self.assertEqual(current["regimes"]["spy"], "bear")
        self.assertEqual(current["regimes"]["breadth_new"], "gt_60")
        self.assertEqual(current["regimes"]["atr"], "expanding")
        self.assertEqual(current["readings"]["vix"], 40.0)
        self.assertEqual(current["readings"]["breadth_new"], 75.0)
        self.assertEqual(current["readings"]["atr"], 2.0)
        self.assertIsNone(current["readings"]["spy"])

    def test_coverage_starts_on_the_first_labeled_session(self):
        dates = pd.bdate_range("2020-01-02", periods=6)
        data = pd.DataFrame(
            {
                "Date": dates,
                "RollingPnL": [1.0] * 6,
                "ATR20": [1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                "ATR50": [np.nan, np.nan, 1.0, 1.0, 1.0, 1.0],
            }
        )
        calendar = pd.DataFrame(
            {"breadth_credit_risk": [np.nan, np.nan, 55.0, 55.0, 80.0, 80.0]},
            index=dates,
        )
        coverage = regime_coverage(data, calendar)
        self.assertEqual(coverage["credit_risk_breadth"], {"start": "2020-01-06", "end": "2020-01-09"})
        self.assertEqual(coverage["atr"], {"start": "2020-01-06", "end": "2020-01-09"})
        self.assertIsNone(coverage["breadth_new"])


class TradeLabelTests(unittest.TestCase):
    def test_trades_without_years_omit_regime_keys(self):
        trades = trade_payload(_executed_frame())
        self.assertNotIn("vix_regime", trades[0])
        self.assertNotIn("atr_regime", trades[0])

    def test_entry_row_atr_and_calendar_labels(self):
        calendar = pd.DataFrame(
            {
                "vix": [12.0],
                "vxn": [45.0],
                "spy_sma50": [100.0],
                "spy_sma200": [100.0],
                "breadth_old": [20.0],
                "breadth_new": [80.01],
            },
            index=pd.to_datetime(["2024-01-02"]),
        )
        trades = trade_payload(_executed_frame(), regime_calendar=calendar, include_regimes=True)
        closed = trades[0]
        self.assertEqual(closed["status"], "Closed")
        self.assertEqual(closed["vix_regime"], "le_15")
        self.assertEqual(closed["vxn_regime"], "gt_30")
        self.assertEqual(closed["spy_regime"], SPY_BULL)
        self.assertEqual(closed["breadth_old_regime"], "lt_40")
        self.assertEqual(closed["breadth_new_regime"], "gt_60")
        self.assertEqual(closed["atr_regime"], ATR_EXPANDING)

    def test_missing_atr_columns_leave_atr_empty(self):
        frame = _executed_frame().drop(columns=["ATR20", "ATR50"])
        trades = trade_payload(frame, regime_calendar=None, include_regimes=True)
        self.assertIsNone(trades[0]["atr_regime"])
        self.assertIsNone(trades[0]["vix_regime"])

    def test_payload_loads_calendar_only_when_years_is_set(self):
        # Two calendar years so dropping the best year still leaves a metrics sample.
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-06-01", "2023-06-02", "2024-01-02", "2024-01-03"]),
                "Close": [100.0, 100.5, 101.0, 103.0],
                "RollingPnL": [1.0, 1.01, 1.05, 1.30],
                "Drawdown": [0.0, 0.0, 0.0, 0.01],
                "LongTradeIn": [False, False, True, False],
                "LongTradeOut": [False, False, False, True],
                "HoldLong": [False, False, True, False],
                "TradePnL": [0.0, 0.0, 0.0, 0.02],
                "DaysInTrade": [0, 0, 0, 1],
                "ATR20": [1.0, 1.0, 2.0, 1.0],
                "ATR50": [1.0, 1.0, 1.0, 2.0],
            }
        )
        with patch("api.serializers.load_regime_calendar") as load:
            detailed_backtest_payload(frame, 5, 1, "plain")
        load.assert_not_called()

        calendar = pd.DataFrame(
            {
                "vix": [10.0],
                "vxn": [18.0],
                "spy_sma50": [1.0],
                "spy_sma200": [2.0],
                "breadth_old": [50.0],
                "breadth_new": [50.0],
            },
            index=pd.to_datetime(["2024-01-02"]),
        )
        with patch("api.serializers.load_regime_calendar", return_value=calendar) as load:
            payload = detailed_backtest_payload(frame, 5, 1, "regimes", years=10)
        load.assert_called_once_with(10)
        self.assertEqual(payload["trades"][0]["vix_regime"], "le_15")
        self.assertEqual(payload["trades"][0]["spy_regime"], SPY_BEAR)
        self.assertEqual(payload["trades"][0]["atr_regime"], ATR_EXPANDING)
        self.assertIn("market_regime_sharpe", payload)
        self.assertNotIn(
            "market_regime_sharpe",
            detailed_backtest_payload(frame, 5, 1, "plain"),
        )


if __name__ == "__main__":
    unittest.main()
