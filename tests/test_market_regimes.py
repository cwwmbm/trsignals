"""Market regime bucket edges, breadth series, and trade-label attachment."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from api.indicator_catalog import list_indicators
from api.market_regimes import (
    ATR_CONTRACTING,
    ATR_EXPANDING,
    REGIME_INDICATOR_SPECS,
    SPY_BEAR,
    SPY_BULL,
    build_regime_calendar,
    classify_atr,
    classify_breadth,
    classify_curve_10y3m,
    classify_curve_change,
    classify_dollar_rates,
    classify_inflation,
    classify_inflation_yield,
    classify_rate_curve,
    classify_rate_shock,
    classify_sector_breadth,
    classify_sector_trend,
    classify_spy,
    classify_vix,
    classify_vxn,
    clear_regime_calendar_cache,
    compute_market_regime_sharpe,
    current_market_regimes,
    ensure_regime_indicator_columns,
    load_regime_calendar,
    regime_coverage,
    regime_score,
    market_regimes_for_timestamp,
    sharpe_ratio,
)
from api.serializers import detailed_backtest_payload, trade_payload
from api.strategy_compiler import compile_buy_mask
from stats import compute_aggregate_metrics


def _calendar_inputs(periods=220):
    idx = pd.bdate_range("2018-01-02", periods=periods)
    spy = pd.Series(100.0, index=idx)
    # Uneven swings so RSI of the raw ratio and RSI of its log do not match.
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

    def test_rate_shock_endpoints_stay_in_the_middle(self):
        self.assertIsNone(classify_rate_shock(None))
        self.assertEqual(classify_rate_shock(-1.01), "lt_neg_1")
        self.assertEqual(classify_rate_shock(-1), "neg_1_to_1")
        self.assertEqual(classify_rate_shock(0), "neg_1_to_1")
        self.assertEqual(classify_rate_shock(1), "neg_1_to_1")
        self.assertEqual(classify_rate_shock(1.01), "gt_1")

    def test_curve_level_zero_is_normal(self):
        self.assertIsNone(classify_curve_10y3m(float("nan")))
        self.assertEqual(classify_curve_10y3m(-0.01), "inverted")
        self.assertEqual(classify_curve_10y3m(0), "normal")
        self.assertEqual(classify_curve_10y3m(0.25), "normal")

    def test_curve_change_edges(self):
        self.assertEqual(classify_curve_change(-0.50), "le_neg_50")
        self.assertEqual(classify_curve_change(-0.499), "neg_50_neg_10")
        self.assertEqual(classify_curve_change(-0.10), "neg_50_neg_10")
        self.assertEqual(classify_curve_change(-0.099), "neg_10_pos_10")
        self.assertEqual(classify_curve_change(0.10), "neg_10_pos_10")
        self.assertEqual(classify_curve_change(0.101), "pos_10_pos_50")
        self.assertEqual(classify_curve_change(0.499), "pos_10_pos_50")
        self.assertEqual(classify_curve_change(0.50), "ge_pos_50")

    def test_rate_curve_crosses_the_two_signs(self):
        self.assertIsNone(classify_rate_curve(None, 0.2))
        self.assertIsNone(classify_rate_curve(0.2, None))
        self.assertIsNone(classify_rate_curve(0, 0.2))
        self.assertIsNone(classify_rate_curve(0, 0))
        self.assertEqual(classify_rate_curve(0.01, 0.01), "shock_pos_curve_pos")
        self.assertEqual(classify_rate_curve(0.01, 0), "shock_pos_curve_nonpos")
        self.assertEqual(classify_rate_curve(0.01, -0.2), "shock_pos_curve_nonpos")
        self.assertEqual(classify_rate_curve(-0.01, 0.2), "shock_neg_curve_pos")
        self.assertEqual(classify_rate_curve(-0.01, 0), "shock_neg_curve_nonpos")
        self.assertEqual(classify_rate_curve(-1.5, -0.6), "shock_neg_curve_nonpos")

    def test_sector_trend_edges_belong_to_the_lower_bucket(self):
        self.assertIsNone(classify_sector_trend(None))
        self.assertEqual(classify_sector_trend(-0.0501), "lt_neg_5")
        self.assertEqual(classify_sector_trend(-0.05), "neg_5_to_0")
        self.assertEqual(classify_sector_trend(0), "neg_5_to_0")
        self.assertEqual(classify_sector_trend(0.0001), "zero_to_pos_5")
        self.assertEqual(classify_sector_trend(0.05), "zero_to_pos_5")
        self.assertEqual(classify_sector_trend(0.0501), "gt_pos_5")

    def test_inflation_zero_is_the_lower_bucket(self):
        self.assertIsNone(classify_inflation(float("nan")))
        self.assertEqual(classify_inflation(-0.01), "nonpos")
        self.assertEqual(classify_inflation(0), "nonpos")
        self.assertEqual(classify_inflation(0.01), "pos")

    def test_dollar_rates_keeps_zero_with_the_nonpositive_side(self):
        self.assertIsNone(classify_dollar_rates(None, 0.2))
        self.assertIsNone(classify_dollar_rates(0.2, None))
        self.assertEqual(classify_dollar_rates(0, 0), "tnx_nonpos_dollar_nonpos")
        self.assertEqual(classify_dollar_rates(0, 0.2), "tnx_nonpos_dollar_pos")
        self.assertEqual(classify_dollar_rates(-0.4, 1.2), "tnx_nonpos_dollar_pos")
        self.assertEqual(classify_dollar_rates(0.4, 0), "tnx_pos_dollar_nonpos")
        self.assertEqual(classify_dollar_rates(0.4, -0.2), "tnx_pos_dollar_nonpos")
        self.assertEqual(classify_dollar_rates(0.4, 0.2), "tnx_pos_dollar_pos")

    def test_inflation_yield_crosses_rising_and_falling(self):
        self.assertIsNone(classify_inflation_yield(0.2, None))
        self.assertIsNone(classify_inflation_yield(None, 0.2))
        self.assertEqual(classify_inflation_yield(0.2, 0.01), "tnx_rising_inflation_rising")
        self.assertEqual(classify_inflation_yield(0.2, 0), "tnx_rising_inflation_falling")
        self.assertEqual(classify_inflation_yield(0.2, -0.01), "tnx_rising_inflation_falling")
        self.assertEqual(classify_inflation_yield(0, 0.01), "tnx_falling_inflation_rising")
        self.assertEqual(classify_inflation_yield(-0.2, -0.01), "tnx_falling_inflation_falling")

    def test_dollar_rates_and_inflation_yield_lookup_uses_both_series(self):
        frame = pd.DataFrame(
            {
                "rate_shock": [0.0, 0.4],
                "dollar_shock": [0.0, -0.2],
                "inflation_trend": [-0.01, 0.02],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )
        first = market_regimes_for_timestamp("2024-01-02", frame)
        self.assertEqual(first["dollar_rates_regime"], "tnx_nonpos_dollar_nonpos")
        self.assertEqual(first["inflation_yield_regime"], "tnx_falling_inflation_falling")
        second = market_regimes_for_timestamp("2024-01-03", frame)
        self.assertEqual(second["dollar_rates_regime"], "tnx_pos_dollar_nonpos")
        self.assertEqual(second["inflation_yield_regime"], "tnx_rising_inflation_rising")

    def test_rate_curve_lookup_uses_both_series(self):
        frame = pd.DataFrame(
            {"rate_shock": [0.4], "curve_change_20": [0.0]},
            index=pd.to_datetime(["2024-01-02"]),
        )
        labels = market_regimes_for_timestamp("2024-01-02", frame)
        self.assertEqual(labels["rate_curve_regime"], "shock_pos_curve_nonpos")


class CalendarTests(unittest.TestCase):
    def test_market_breadth_is_rsi_of_the_raw_rsp_spy_ratio(self):
        from api.market_regimes import _rsi

        vix, vxn, spy, rsp, _idx = _calendar_inputs()
        frame = build_regime_calendar(vix, vxn, spy, rsp)
        ratio = (rsp / spy).reindex(frame.index)
        expected = _rsi(ratio).dropna()
        breadth = frame["breadth"].dropna()
        self.assertGreater(len(breadth), 0)
        self.assertTrue(np.allclose(breadth.to_numpy(), expected.reindex(breadth.index).to_numpy()))
        logged = _rsi(pd.Series(np.log(ratio.where(ratio > 0)), index=frame.index)).reindex(breadth.index)
        self.assertFalse(np.allclose(breadth.to_numpy(), logged.to_numpy()))

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

    def test_rate_shock_scales_the_20_day_move_by_63_day_vol(self):
        idx = pd.bdate_range("2020-01-02", periods=90)
        rng = np.random.default_rng(7)
        steps = rng.normal(0.0, 0.04, len(idx) - 1)
        tnx = pd.Series(np.r_[4.0, 4.0 + np.cumsum(steps)], index=idx)
        flat = pd.Series(100.0, index=idx)
        frame = build_regime_calendar(flat, flat, flat, flat, tnx=tnx)
        daily = tnx.diff()
        vol = daily.rolling(63).std(ddof=1)
        expected = (tnx - tnx.shift(20)) / (vol * np.sqrt(20))
        both = pd.concat({"got": frame["rate_shock"], "expected": expected}, axis=1).dropna()
        self.assertGreater(len(both), 0)
        self.assertTrue(np.allclose(both["got"], both["expected"]))
        self.assertTrue(frame["rate_shock"].iloc[:63].isna().all())
        self.assertTrue(frame["curve_10y3m"].isna().all())
        self.assertTrue(frame["curve_change_20"].isna().all())

    def test_curve_is_tnx_minus_irx_and_its_20_day_change(self):
        idx = pd.bdate_range("2020-01-02", periods=25)
        tnx = pd.Series(5.0, index=idx)
        irx = pd.Series(4.0, index=idx)
        irx.iloc[-1] = 4.60
        flat = pd.Series(100.0, index=idx)
        frame = build_regime_calendar(flat, flat, flat, flat, tnx=tnx, irx=irx)
        self.assertAlmostEqual(frame["curve_10y3m"].iloc[-1], 0.40)
        self.assertEqual(classify_curve_10y3m(frame["curve_10y3m"].iloc[-1]), "normal")
        # Twenty sessions back the spread was 1.00, so the change is -0.60.
        self.assertAlmostEqual(frame["curve_change_20"].iloc[-1], -0.60)
        self.assertEqual(classify_curve_change(frame["curve_change_20"].iloc[-1]), "le_neg_50")
        self.assertTrue(frame["curve_change_20"].iloc[:20].isna().all())
        inverted = build_regime_calendar(flat, flat, flat, flat, tnx=irx, irx=tnx)
        self.assertLess(inverted["curve_10y3m"].iloc[0], 0)
        self.assertEqual(classify_curve_10y3m(inverted["curve_10y3m"].iloc[0]), "inverted")

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

    def test_sector_trend_is_the_mean_log_deviation_from_sma50(self):
        idx = pd.bdate_range("2020-01-02", periods=60)
        closes = pd.Series(np.linspace(100, 160, 60), index=idx)
        flat = pd.Series(100.0, index=idx)
        frame = build_regime_calendar(flat, flat, flat, flat, [closes] * 9)
        sma = closes.rolling(50).mean()
        expected = np.log(closes / sma)
        both = pd.concat({"got": frame["sector_trend_50"], "expected": expected}, axis=1).dropna()
        self.assertGreater(len(both), 0)
        self.assertTrue(np.allclose(both["got"], both["expected"]))
        self.assertTrue(frame["sector_trend_50"].iloc[:49].isna().all())
        self.assertEqual(classify_sector_trend(0.0), "neg_5_to_0")

    def test_sector_trend_waits_until_every_sector_sma_is_ready(self):
        idx = pd.bdate_range("2020-01-02", periods=100)
        early = pd.Series(np.arange(1, 101, dtype=float), index=idx)
        late = pd.Series(np.arange(1, 61, dtype=float), index=idx[40:])
        level = pd.Series(100.0, index=idx)
        frame = build_regime_calendar(level, level, level, level, [early] * 8 + [late])
        self.assertTrue(frame["sector_trend_50"].iloc[:89].isna().all())
        self.assertTrue(pd.notna(frame["sector_trend_50"].iloc[89]))

    def test_dollar_shock_scales_the_20_day_uup_move_by_63_day_vol(self):
        idx = pd.bdate_range("2020-01-02", periods=90)
        rng = np.random.default_rng(3)
        steps = rng.normal(0.0, 0.2, len(idx) - 1)
        uup = pd.Series(np.r_[25.0, 25.0 + np.cumsum(steps)], index=idx)
        flat = pd.Series(100.0, index=idx)
        frame = build_regime_calendar(flat, flat, flat, flat, uup=uup)
        daily = uup.diff()
        vol = daily.rolling(63).std(ddof=1)
        expected = (uup - uup.shift(20)) / (vol * np.sqrt(20))
        both = pd.concat({"got": frame["dollar_shock"], "expected": expected}, axis=1).dropna()
        self.assertGreater(len(both), 0)
        self.assertTrue(np.allclose(both["got"], both["expected"]))
        self.assertTrue(frame["dollar_shock"].iloc[:63].isna().all())
        quiet = build_regime_calendar(flat, flat, flat, flat, uup=pd.Series(25.0, index=idx))
        self.assertTrue(quiet["dollar_shock"].isna().all())

    def test_inflation_trend_is_the_sma_gap_of_log_tip_over_ief(self):
        idx = pd.bdate_range("2020-01-02", periods=120)
        step = 0.001
        logged = pd.Series(np.arange(len(idx), dtype=float) * step, index=idx)
        tip = pd.Series(np.exp(logged.to_numpy()), index=idx)
        ief = pd.Series(1.0, index=idx)
        flat = pd.Series(100.0, index=idx)
        frame = build_regime_calendar(flat, flat, flat, flat, tip=tip, ief=ief)
        expected = logged.rolling(20).mean() - logged.rolling(100).mean()
        both = pd.concat({"got": frame["inflation_trend"], "expected": expected}, axis=1).dropna()
        self.assertGreater(len(both), 0)
        self.assertTrue(np.allclose(both["got"], both["expected"]))
        self.assertTrue(frame["inflation_trend"].iloc[:99].isna().all())
        self.assertAlmostEqual(frame["inflation_trend"].iloc[-1], 40 * step)
        self.assertEqual(classify_inflation(frame["inflation_trend"].iloc[-1]), "pos")
        flat_ratio = build_regime_calendar(flat, flat, flat, flat, tip=flat, ief=flat)
        self.assertAlmostEqual(flat_ratio["inflation_trend"].iloc[-1], 0.0)
        self.assertEqual(classify_inflation(flat_ratio["inflation_trend"].iloc[-1]), "nonpos")

    def test_lookup_uses_the_entry_session_date(self):
        idx = pd.bdate_range("2024-01-02", periods=5)
        frame = pd.DataFrame(
            {
                "vix": [10.0, 16.0, 36.0, 14.0, 22.0],
                "vxn": [18.0, 22.0, 46.0, 28.0, 33.0],
                "spy_sma50": [10, 10, 9, 10, 10],
                "spy_sma200": [10, 10, 10, 10, 10],
                "breadth": [20, 40, 60, 80, 81],
            },
            index=idx,
        )
        labels = market_regimes_for_timestamp("2024-01-04 10:15", frame)
        self.assertEqual(labels["vix_regime"], "gt_30")
        self.assertEqual(labels["vxn_regime"], "gt_30")
        self.assertEqual(labels["spy_regime"], SPY_BEAR)
        self.assertEqual(labels["market_breadth_regime"], "50_60")
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

    def test_regime_score_blends_sortino_return_drawdown_and_robustness(self):
        perfect = regime_score(3.0, 0.03, 0.30, {2020: 0.10, 2021: 0.10})
        self.assertAlmostEqual(perfect["score"], 100.0)
        self.assertAlmostEqual(perfect["sortino"], 1.0)
        self.assertAlmostEqual(perfect["return"], 1.0)
        self.assertAlmostEqual(perfect["drawdown"], 1.0)
        self.assertAlmostEqual(perfect["robustness"], 1.0)

        # Each of the first three inputs is halfway. Equal years keep robustness at 1.
        halfway = regime_score(1.5, 0.015, 0.50, {2020: 0.10, 2021: 0.10})
        expected = 100.0 * (0.5**0.30) * (0.5**0.25) * (0.5**0.15) * (1.0**0.30)
        self.assertAlmostEqual(halfway["score"], expected)
        self.assertAlmostEqual(halfway["sortino"], 0.5)
        self.assertAlmostEqual(halfway["return"], 0.5)
        self.assertAlmostEqual(halfway["drawdown"], 0.5)
        self.assertAlmostEqual(halfway["robustness"], 1.0)

        # A zero part stays 0 in the breakdown, and the blend floors it at 0.05.
        wiped = regime_score(3.0, 0.03, 0.30, {2020: 0.20, 2021: 0.0})
        self.assertEqual(wiped["robustness"], 0.0)
        self.assertAlmostEqual(wiped["score"], 100.0 * (0.05**0.30))
        self.assertAlmostEqual(
            regime_score(-1.0, 0.03, 0.30, {2020: 0.10, 2021: 0.10})["score"],
            100.0 * (0.05**0.30),
        )
        self.assertAlmostEqual(
            regime_score(3.0, 0.03, 0.80, {2020: 0.10, 2021: 0.10})["score"],
            100.0 * (0.05**0.15),
        )
        self.assertIsNone(regime_score(None, 0.03, 0.30, {2020: 0.10, 2021: 0.10})["score"])

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
                "breadth": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
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

        breadth = {row["key"]: row for row in result["market_breadth"]}
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
                "breadth": [50.0] * 7,
            },
            index=dates,
        )
        result = compute_market_regime_sharpe(data, calendar)
        vix = {row["key"]: row for row in result["vix"]}
        self.assertAlmostEqual(vix["le_15"]["max_drawdown"], 0.10)
        self.assertAlmostEqual(vix["gt_30"]["max_drawdown"], 0.20)
        self.assertIsNone(vix["15_20"]["max_drawdown"])
        # Cash days stay in the path. Trade 1 is -10% then +5% on the VIX <= 15 entry.
        # Trade 2 is -20% then +1% on the VIX > 30 entry. Metrics match the summary cards.
        le_15_path = np.array([0.0, 0.0, -0.10, 0.05, 0.0, 0.0, 0.0])
        gt_30_path = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.20, 0.01])

        def summary_for(path):
            equity = np.cumprod(1.0 + path)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            frame = pd.DataFrame(
                {
                    "Date": dates,
                    "RollingPnL": equity,
                    "Drawdown": drawdown,
                    "LongTradeOut": False,
                    "TradePnL": 0.0,
                }
            )
            return compute_aggregate_metrics(frame)

        le_15_summary = summary_for(le_15_path)
        gt_30_summary = summary_for(gt_30_path)
        self.assertAlmostEqual(vix["le_15"]["sharpe"], le_15_summary["sharpe"])
        self.assertAlmostEqual(vix["gt_30"]["sharpe"], gt_30_summary["sharpe"])
        self.assertIsNone(vix["15_20"]["sharpe"])
        self.assertAlmostEqual(vix["le_15"]["sortino"], le_15_summary["sortino"])
        self.assertAlmostEqual(vix["gt_30"]["sortino"], gt_30_summary["sortino"])
        self.assertIsNone(vix["15_20"]["sortino"])
        self.assertAlmostEqual(vix["le_15"]["cagr"], le_15_summary["cagr_decimal"])
        self.assertAlmostEqual(vix["gt_30"]["cagr"], gt_30_summary["cagr_decimal"])
        self.assertIsNone(vix["15_20"]["cagr"])
        self.assertAlmostEqual(vix["le_15"]["calmar"], le_15_summary["calmar"])
        self.assertAlmostEqual(vix["gt_30"]["calmar"], gt_30_summary["calmar"])
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
                "breadth": [50.0, 50.0, 75.0],
            },
            index=dates,
        )
        current = current_market_regimes(data, calendar)
        self.assertEqual(current["as_of"], "2024-01-04")
        self.assertEqual(current["regimes"]["vix"], "gt_30")
        self.assertEqual(current["regimes"]["spy"], "bear")
        self.assertEqual(current["regimes"]["market_breadth"], "gt_60")
        self.assertEqual(current["regimes"]["atr"], "expanding")
        self.assertEqual(current["readings"]["vix"], 40.0)
        self.assertEqual(current["readings"]["market_breadth"], 75.0)
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
        self.assertIsNone(coverage["market_breadth"])


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
                "breadth": [20.0],
            },
            index=pd.to_datetime(["2024-01-02"]),
        )
        trades = trade_payload(_executed_frame(), regime_calendar=calendar, include_regimes=True)
        closed = trades[0]
        self.assertEqual(closed["status"], "Closed")
        self.assertEqual(closed["vix_regime"], "le_15")
        self.assertEqual(closed["vxn_regime"], "gt_30")
        self.assertEqual(closed["spy_regime"], SPY_BULL)
        self.assertEqual(closed["market_breadth_regime"], "lt_40")
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
                "breadth": [50.0],
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


class RegimeIndicatorTests(unittest.TestCase):
    def test_catalog_lists_every_bucket(self):
        listed = {
            item["id"]: item
            for item in list_indicators(builder_only=True)
            if item["category"] == "Market regimes"
        }
        self.assertEqual(set(listed), {spec.id for spec in REGIME_INDICATOR_SPECS})
        vix = listed["Regime_vix_15_20"]
        self.assertEqual(vix["label"], "VIX · 15–20")
        self.assertEqual(vix["valueType"], "flag")
        self.assertEqual(vix["compareMode"], "none")

    def test_flags_and_compiled_condition_follow_the_bucket(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        data = pd.DataFrame(
            {
                "Date": dates,
                "ATR20": [2.0, 1.0, np.nan, 1.0],
                "ATR50": [1.0, 2.0, 1.0, 1.0],
            }
        )
        calendar = pd.DataFrame({"vix": [18.0, 10.0, np.nan, 18.0]}, index=dates)
        condition = {"left": "Regime_vix_15_20", "operator": "is true", "right": "", "logic": "AND"}
        with patch("api.market_regimes.load_regime_calendar", return_value=calendar) as load:
            ensure_regime_indicator_columns(
                data,
                [{"left": "Regime_vix_15_20"}, {"left": "Regime_vix_le_15"}],
            )
            mask = compile_buy_mask(data, [condition])
        load.assert_called_once()
        self.assertEqual(data["Regime_vix_15_20"].tolist(), [1, -1, 0, 1])
        self.assertEqual(data["Regime_vix_le_15"].tolist(), [-1, 1, 0, -1])
        self.assertEqual(mask.tolist(), [True, False, False, True])

        outside = {"left": "Regime_vix_15_20", "operator": "is false", "right": "", "logic": "AND"}
        self.assertEqual(compile_buy_mask(data, [outside]).tolist(), [False, True, False, False])

    def test_unlabeled_day_matches_neither_flag(self):
        self.assertEqual(
            compile_buy_mask(
                pd.DataFrame(
                    {
                        "Date": pd.to_datetime(["2024-01-02"]),
                        "Regime_vix_15_20": [0],
                    }
                ),
                [{"left": "Regime_vix_15_20", "operator": "is true", "right": "", "logic": "AND"}],
            ).tolist(),
            [False],
        )

    def test_missing_calendar_raises(self):
        data = pd.DataFrame({"Date": pd.to_datetime(["2024-01-02"])})
        condition = {"left": "Regime_vix_le_15", "operator": "is true", "right": "", "logic": "AND"}
        with patch("api.market_regimes.load_regime_calendar", return_value=None):
            with self.assertRaises(ValueError):
                compile_buy_mask(data, [condition])

    def test_atr_flag_does_not_download_the_calendar(self):
        data = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "ATR20": [2.0, np.nan],
                "ATR50": [1.0, 1.0],
            }
        )
        condition = {"left": "Regime_atr_expanding", "operator": "is true", "right": "", "logic": "AND"}
        with patch("api.market_regimes.load_regime_calendar") as load:
            mask = compile_buy_mask(data, [condition])
        load.assert_not_called()
        self.assertEqual(data["Regime_atr_expanding"].tolist(), [1, 0])
        self.assertEqual(mask.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
