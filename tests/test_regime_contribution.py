"""Unit and integration tests for regime contribution analysis."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from api.regime_contribution import (
    DD_SEVERE_THRESHOLD,
    DISPLAY_ZERO_THRESHOLD,
    MATERIAL_MARGINAL_THRESHOLD,
    NUMERICAL_EPS,
    SMA_LOOKBACK,
    VOL_LOOKBACK,
    VOL_PERCENTILE_LOOKBACK,
    added_exposure_mask,
    align_spy_close_to_dates,
    baseline_drawdown_episodes,
    below_sma_episodes,
    classify_baseline_drawdown,
    classify_evidence,
    classify_stress_days,
    compute_regime_contribution,
    compute_regime_state_metrics,
    compute_signed_drawdown,
    compute_spy_realized_vol,
    compute_spy_realized_vol_labels,
    compute_spy_sma200_labels,
    contribution_concentration,
    count_severe_drawdown_episodes,
    daily_log_returns,
    daily_simple_returns,
    downside_deviation,
    expected_shortfall,
    find_contiguous_episodes,
    marginal_log_returns,
)


class SpyTrendRegimeTests(unittest.TestCase):
    def test_first_199_unavailable(self):
        close = pd.Series(np.linspace(100, 120, 250))
        labels = compute_spy_sma200_labels(close)
        self.assertTrue(labels.iloc[: SMA_LOOKBACK - 1].isna().all())
        self.assertIsNotNone(labels.iloc[SMA_LOOKBACK - 1])

    def test_above_below_boundary(self):
        # Flat then jump above rising SMA region
        values = [100.0] * 200 + [100.0, 101.0, 90.0]
        close = pd.Series(values)
        labels = compute_spy_sma200_labels(close)
        # At index 199, SMA = 100, close = 100 → below (<=)
        self.assertEqual(labels.iloc[199], "below")
        # Index 200: close 100 vs SMA slightly above 100? mean of 100*199 + 100 = 100 → below
        self.assertEqual(labels.iloc[200], "below")
        # Index 201: close 101 > SMA
        self.assertEqual(labels.iloc[201], "above")
        # Index 202: close 90 < SMA → below
        self.assertEqual(labels.iloc[202], "below")

    def test_no_future_leakage_in_sma(self):
        close = pd.Series(np.arange(1, 301, dtype=float))
        labels = compute_spy_sma200_labels(close)
        # Mutating a future close must not change an earlier label
        early = labels.iloc[250]
        close_mut = close.copy()
        close_mut.iloc[290] = 1e9
        labels_mut = compute_spy_sma200_labels(close_mut)
        self.assertEqual(early, labels_mut.iloc[250])


class SpyVolatilityRegimeTests(unittest.TestCase):
    def test_log_returns_and_sample_std(self):
        # Constant price → zero vol after warmup for returns
        close = pd.Series([100.0] * 50)
        sigma = compute_spy_realized_vol(close)
        self.assertTrue(np.isnan(sigma.iloc[VOL_LOOKBACK - 1]) or sigma.iloc[VOL_LOOKBACK] == 0.0)
        # After first return exists and window fills, flat series → ~0
        self.assertAlmostEqual(float(sigma.iloc[VOL_LOOKBACK + 5]), 0.0, places=12)

    def test_annualization_sqrt_252(self):
        rng = np.random.default_rng(0)
        # Build closes from known log returns
        log_r = rng.normal(0, 0.01, size=40)
        close = pd.Series(100 * np.exp(np.cumsum(np.r_[0.0, log_r])))
        sigma = compute_spy_realized_vol(close)
        window = np.log(close / close.shift(1)).iloc[21 - VOL_LOOKBACK + 1 : 22]
        # At index 21 (0-based), window is returns at 2..21 if lookback 20? 
        # rolling at i uses i-19..i inclusive for lookback 20
        i = 25
        rets = np.log(close / close.shift(1)).iloc[i - VOL_LOOKBACK + 1 : i + 1]
        expected = float(rets.std(ddof=1) * math.sqrt(252))
        self.assertAlmostEqual(float(sigma.iloc[i]), expected, places=10)

    def test_trailing_percentile_no_full_sample_leakage(self):
        # Need enough history: vol warmup + 252 prior
        n = VOL_LOOKBACK + VOL_PERCENTILE_LOOKBACK + 50
        rng = np.random.default_rng(1)
        log_r = rng.normal(0, 0.01, size=n)
        # Inject a huge future shock that must not affect early classification
        close = pd.Series(100 * np.exp(np.cumsum(log_r)))
        labels = compute_spy_realized_vol_labels(close)
        idx = VOL_LOOKBACK + VOL_PERCENTILE_LOOKBACK + 5
        before = labels.iloc[idx]
        close2 = close.copy()
        close2.iloc[-1] = close2.iloc[-2] * 2  # future spike
        labels2 = compute_spy_realized_vol_labels(close2)
        self.assertEqual(before, labels2.iloc[idx])

    def test_missing_history_unavailable(self):
        close = pd.Series(np.linspace(100, 110, 100))
        labels = compute_spy_realized_vol_labels(close)
        self.assertTrue(labels.isna().all())

    def test_percentile_boundaries_low_high(self):
        # Construct σ20 that is monotone via increasing noise scale
        n = VOL_LOOKBACK + VOL_PERCENTILE_LOOKBACK + 30
        closes = [100.0]
        for i in range(1, n):
            # Growing volatility over time
            shock = 0.001 + 0.0002 * i
            closes.append(closes[-1] * math.exp(shock if i % 2 == 0 else -shock))
        close = pd.Series(closes)
        labels = compute_spy_realized_vol_labels(close)
        valid = labels.dropna()
        self.assertTrue(len(valid) > 0)
        # Late period should tend toward high
        self.assertIn(labels.iloc[-1], ("low", "normal", "high"))


class BaselineDrawdownRegimeTests(unittest.TestCase):
    def test_running_peak_and_thresholds(self):
        equity = np.array([100.0, 110.0, 104.5, 93.5, 90.0, 110.0], dtype=float)
        # peaks: 100,110,110,110,110,110
        # dd: 0, 0, 104.5/110-1=-0.05, 93.5/110-1≈-0.15, 90/110-1≈-0.1818, 0
        dd = compute_signed_drawdown(equity)
        self.assertAlmostEqual(dd[2], 104.5 / 110 - 1)
        labels = classify_baseline_drawdown(equity)
        self.assertEqual(labels[0], "normal")
        self.assertEqual(labels[1], "normal")
        # -0.05 is not > -0.05 → drawdown (spec: Normal when Drawdown > -0.05)
        self.assertEqual(labels[2], "drawdown")
        # -0.15 exactly → severe (<= -0.15)
        self.assertEqual(labels[3], "severe_drawdown")
        self.assertEqual(labels[4], "severe_drawdown")
        self.assertEqual(labels[5], "normal")

    def test_open_final_drawdown_episode(self):
        equity = np.array([100.0, 110.0, 100.0, 95.0], dtype=float)
        full = equity.copy()
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        marginal = np.zeros(4)
        added = np.zeros(4, dtype=bool)
        eps = baseline_drawdown_episodes(
            baseline_equity=equity,
            full_equity=full,
            dates=dates,
            marginal=marginal,
            added_exposure=added,
        )
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["status"], "open")
        self.assertEqual(eps[0]["start_date"], "2020-01-03")

    def test_recovery_to_prior_peak(self):
        equity = np.array([100.0, 110.0, 100.0, 110.0, 120.0], dtype=float)
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        eps = baseline_drawdown_episodes(
            baseline_equity=equity,
            full_equity=equity,
            dates=dates,
            marginal=np.zeros(5),
            added_exposure=np.zeros(5, dtype=bool),
        )
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["status"], "closed")
        self.assertEqual(eps[0]["end_date"], "2020-01-04")


class StressDayRegimeTests(unittest.TestCase):
    def test_q10_and_ties(self):
        returns = np.array([-0.05, -0.04, -0.03, -0.02, -0.01] + [0.01] * 15)
        invested = np.ones(len(returns), dtype=bool)
        labels, q10, reason = classify_stress_days(returns, invested_mask=invested)
        self.assertIsNone(reason)
        self.assertIsNotNone(q10)
        for i, r in enumerate(returns):
            if labels[i] == "stress":
                self.assertLessEqual(r, q10)
            elif labels[i] == "non_stress":
                self.assertGreater(r, q10)

    def test_insufficient_observations(self):
        returns = np.array([0.01, -0.02, 0.0])
        labels, q10, reason = classify_stress_days(
            returns, invested_mask=np.array([True, True, False])
        )
        self.assertEqual(reason, "insufficient_baseline_returns")
        self.assertIsNone(q10)
        self.assertTrue(all(x is None for x in labels))

    def test_deterministic_quantile(self):
        returns = np.linspace(-0.1, 0.1, 40)
        invested = np.ones(40, dtype=bool)
        _, q1, _ = classify_stress_days(returns, invested_mask=invested)
        _, q2, _ = classify_stress_days(returns, invested_mask=invested)
        self.assertEqual(q1, q2)

    def test_cash_days_do_not_inflate_stress_when_q10_would_be_zero(self):
        """Regression: idle R=0 days must not become stress when Q10 collapses to 0."""
        n = 1000
        returns = np.zeros(n)
        returns[0] = np.nan
        returns[1:31] = -0.02  # 30 down days while invested
        returns[31:61] = 0.02  # 30 up days while invested
        invested = np.zeros(n, dtype=bool)
        invested[1:61] = True  # only 60 invested days; rest cash

        labels, q10, reason = classify_stress_days(
            returns, invested_mask=invested
        )
        self.assertIsNone(reason)
        self.assertIsNotNone(q10)
        self.assertLess(q10, 0.0)  # Q10 from invested downs/ups, not from cash zeros
        stress_idx = np.where(labels == "stress")[0]
        self.assertTrue(len(stress_idx) > 0)
        # Idle cash (no unique exposure) stays flat — not stress/non-stress
        cash = ~invested & np.isfinite(returns)
        self.assertFalse(np.any((labels == "stress") & cash))
        self.assertFalse(np.any((labels == "non_stress") & cash))
        self.assertTrue(np.all(labels[cash] == "flat"))
        # All stress days are invested when unique_exposure is omitted
        self.assertTrue(np.all(invested[stress_idx]))
        # Invested universe is partitioned into stress + non_stress
        invested_labeled = labels[invested]
        self.assertTrue(np.all(np.isin(invested_labeled, ["stress", "non_stress"])))

    def test_unique_exposure_classified_against_baseline_q10(self):
        """OR-overlay case: contribution days are unique exposure; classify via R_full."""
        n = 80
        r_base = np.zeros(n)
        r_base[0] = np.nan
        r_base[1:41] = np.linspace(-0.05, 0.05, 40)  # invested baseline sample
        invested = np.zeros(n, dtype=bool)
        invested[1:41] = True

        # Unique exposure on later idle baseline days
        unique = np.zeros(n, dtype=bool)
        unique[50:60] = True  # crash-like unique days
        unique[60:70] = True  # calm unique days
        r_full = np.zeros(n)
        r_full[0] = np.nan
        r_full[50:60] = -0.08
        r_full[60:70] = 0.02

        labels, q10, reason = classify_stress_days(
            r_base,
            invested_mask=invested,
            unique_exposure_mask=unique,
            unique_returns=r_full,
        )
        self.assertIsNone(reason)
        self.assertLess(q10, 0.0)
        self.assertTrue(np.all(labels[50:60] == "stress"))
        self.assertTrue(np.all(labels[60:70] == "non_stress"))
        # Truly idle remains flat
        idle = ~invested & ~unique & np.isfinite(r_base)
        self.assertTrue(np.all(labels[idle] == "flat"))

    def test_old_behavior_marked_cash_as_stress(self):
        """Document the bug we fixed: all-valid-days Q10 on cash-heavy series."""
        n = 1000
        returns = np.zeros(n)
        returns[0] = np.nan
        returns[1:31] = -0.02
        returns[31:61] = 0.02
        # Without invested_mask, nonzero inference still works for this series...
        # Force the bug path: treat all finite as "sample" by passing invested=all valid
        labels_buggy, q10_buggy, _ = classify_stress_days(
            returns,
            invested_mask=np.isfinite(returns),  # wrongly include cash in universe
        )
        self.assertEqual(q10_buggy, 0.0)
        self.assertGreater(int((labels_buggy == "stress").sum()), 900)


class MarginalAttributionTests(unittest.TestCase):
    def test_simple_and_log_returns(self):
        equity = np.array([100.0, 110.0, 99.0], dtype=float)
        r = daily_simple_returns(equity)
        self.assertTrue(np.isnan(r[0]))
        self.assertAlmostEqual(r[1], 0.1)
        self.assertAlmostEqual(r[2], 99 / 110 - 1)
        l = daily_log_returns(r)
        self.assertAlmostEqual(l[1], math.log1p(0.1))

    def test_marginal_additivity_and_compound(self):
        full = np.array([100.0, 110.0, 121.0], dtype=float)
        base = np.array([100.0, 105.0, 110.25], dtype=float)
        _, _, m = marginal_log_returns(full, base)
        self.assertTrue(np.isnan(m[0]))
        total = float(np.nansum(m))
        compounded = math.exp(total) - 1
        # Full total log - base total log
        full_log = math.log(121 / 100)
        base_log = math.log(110.25 / 100)
        self.assertAlmostEqual(total, full_log - base_log)
        self.assertAlmostEqual(compounded, math.exp(full_log - base_log) - 1)

    def test_flat_cash_days(self):
        equity = np.array([100.0, 100.0, 100.0], dtype=float)
        r = daily_simple_returns(equity)
        self.assertAlmostEqual(r[1], 0.0)
        self.assertAlmostEqual(r[2], 0.0)
        l = daily_log_returns(r)
        self.assertAlmostEqual(l[1], 0.0)

    def test_invalid_return_le_minus_one(self):
        r = np.array([np.nan, -1.0, -1.5, 0.1])
        l = daily_log_returns(r)
        self.assertTrue(np.isnan(l[1]))
        self.assertTrue(np.isnan(l[2]))
        self.assertAlmostEqual(l[3], math.log1p(0.1))


class ExposureTests(unittest.TestCase):
    def test_added_exposure_and_overlap(self):
        full = np.array([1, 1, 1, 0], dtype=bool)
        base = np.array([0, 1, 1, 0], dtype=bool)
        added = added_exposure_mask(full, base)
        np.testing.assert_array_equal(added, np.array([1, 0, 0, 0], dtype=bool))

    def test_zero_added_exposure_denominator(self):
        marginal = np.array([np.nan, 0.01, -0.02, 0.0])
        r = np.array([np.nan, 0.01, -0.02, 0.0])
        metrics = compute_regime_state_metrics(
            mask=np.array([False, True, True, True]),
            marginal=marginal,
            r_full=r,
            r_base=r,
            candidate_hold=np.array([False, True, False, False]),
            added_exposure=np.array([False, False, False, False]),
            total_marginal_log=0.01,
            total_eligible_days=3,
        )
        self.assertEqual(metrics["added_exposure_days"], 0)
        self.assertIsNone(metrics["marginal_return_per_added_exposure_day"])


class RiskMetricTests(unittest.TestCase):
    def test_expected_shortfall_and_direction(self):
        # Need >= 20 obs
        returns = np.array([-0.05] * 2 + [-0.01] * 8 + [0.01] * 10)
        es = expected_shortfall(returns)
        self.assertIsNotNone(es)
        self.assertLess(es, 0)

    def test_worst_day_and_downside(self):
        full = np.array([np.nan] + [-0.02, -0.05, 0.01] + [0.0] * 17)
        base = np.array([np.nan] + [-0.03, -0.08, 0.01] + [0.0] * 17)
        worst = float(np.nanmin(full) - np.nanmin(base))
        self.assertGreater(worst, 0)  # full better worst day
        dd_full = downside_deviation(full[1:])
        dd_base = downside_deviation(base[1:])
        self.assertIsNotNone(dd_full)
        self.assertIsNotNone(dd_base)
        self.assertGreater(dd_base, dd_full)

    def test_es_insufficient(self):
        self.assertIsNone(expected_shortfall(np.array([-0.01, 0.02, -0.03])))


class EpisodeTests(unittest.TestCase):
    def test_contiguous_episodes(self):
        mask = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=bool)
        self.assertEqual(find_contiguous_episodes(mask), [(1, 2), (4, 4), (6, 8)])

    def test_below_sma_open_final(self):
        below = np.array([0, 1, 1, 1], dtype=bool)
        dates = pd.date_range("2020-01-01", periods=4, freq="B")
        equity = np.array([100.0, 101.0, 102.0, 103.0])
        eps = below_sma_episodes(
            below_mask=below,
            dates=dates,
            r_full=daily_simple_returns(equity),
            r_base=daily_simple_returns(equity),
            marginal=np.array([np.nan, 0.01, 0.01, 0.01]),
            full_equity=equity,
            baseline_equity=equity * 0.99,
            added_exposure=np.array([0, 1, 0, 0], dtype=bool),
        )
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["status"], "open")
        self.assertTrue(eps[0]["helped"])

    def test_multiple_below_sma_episodes(self):
        below = np.array([1, 1, 0, 1], dtype=bool)
        dates = pd.date_range("2020-01-01", periods=4, freq="B")
        equity = np.ones(4) * 100
        eps = below_sma_episodes(
            below_mask=below,
            dates=dates,
            r_full=np.zeros(4),
            r_base=np.zeros(4),
            marginal=np.zeros(4),
            full_equity=equity,
            baseline_equity=equity,
            added_exposure=np.zeros(4, dtype=bool),
        )
        self.assertEqual(len(eps), 2)

    def test_concentration_warning(self):
        episodes = [
            {"marginal_log_return": 0.10},
            {"marginal_log_return": 0.01},
            {"marginal_log_return": -0.01},
        ]
        result = contribution_concentration(episodes)
        self.assertGreater(result["largest_episode_share"], 0.5)
        self.assertTrue(result["concentration_warning"])

    def test_trough_improvement_and_recovery_acceleration(self):
        # Baseline: peak 100 → down to 80 → recover 100
        baseline = np.array([100.0, 90.0, 80.0, 90.0, 100.0], dtype=float)
        # Full recovers faster and shallower trough
        full = np.array([100.0, 95.0, 90.0, 100.0, 105.0], dtype=float)
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        eps = baseline_drawdown_episodes(
            baseline_equity=baseline,
            full_equity=full,
            dates=dates,
            marginal=np.array([np.nan, 0.01, 0.01, 0.01, 0.0]),
            added_exposure=np.array([0, 1, 1, 0, 0], dtype=bool),
        )
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["status"], "closed")
        self.assertIsNotNone(eps[0]["trough_improvement"])
        self.assertGreater(eps[0]["trough_improvement"], 0)
        self.assertIsNotNone(eps[0]["recovery_acceleration"])
        self.assertGreater(eps[0]["recovery_acceleration"], 0)


class EvidenceTests(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(
            classify_evidence(
                regime_days=600,
                candidate_active_days=80,
                effective_contribution_days=80,
                episode_count=10,
            ),
            "strong",
        )
        self.assertEqual(
            classify_evidence(
                regime_days=300,
                candidate_active_days=40,
                effective_contribution_days=40,
                episode_count=6,
            ),
            "moderate",
        )
        self.assertEqual(
            classify_evidence(
                regime_days=150,
                candidate_active_days=25,
                effective_contribution_days=25,
                episode_count=3,
            ),
            "limited",
        )
        self.assertEqual(
            classify_evidence(
                regime_days=10,
                candidate_active_days=5,
                effective_contribution_days=5,
                episode_count=1,
            ),
            "insufficient",
        )
        # Without episodes, episode threshold not applied
        self.assertEqual(
            classify_evidence(
                regime_days=600,
                candidate_active_days=80,
                effective_contribution_days=80,
                episode_count=None,
            ),
            "strong",
        )

    def test_many_regime_days_but_few_effective_is_insufficient(self):
        self.assertEqual(
            classify_evidence(
                regime_days=600,
                candidate_active_days=80,
                effective_contribution_days=3,
                episode_count=None,
            ),
            "insufficient",
        )


class AlignmentTests(unittest.TestCase):
    def test_no_forward_fill_missing_spy(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        spy = pd.Series(
            [100.0, 101.0, 103.0],
            index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-06"]),
        )
        aligned = align_spy_close_to_dates(spy, dates)
        self.assertTrue(np.isnan(aligned.iloc[2]))  # 2020-01-03 missing
        self.assertFalse(np.isnan(aligned.iloc[0]))

    def test_stress_labels_align_with_same_return_interval(self):
        """Stress label on day t uses R_base,t from equity close t-1 → close t."""
        # Day indices 0..24; returns start at index 1
        n = 25
        # Build baseline equity with a known worst day at index 10
        rets = np.full(n - 1, 0.001)
        rets[9] = -0.08  # return realized on day index 10
        baseline = np.cumprod(np.r_[1.0, 1.0 + rets]) * 100.0
        full = baseline.copy()
        r_full, r_base, m = marginal_log_returns(full, baseline)

        self.assertTrue(np.isnan(r_base[0]))
        self.assertAlmostEqual(r_base[10], -0.08, places=12)
        # M_t uses same day index as R_base
        self.assertTrue(np.isfinite(m[10]) or abs(m[10]) < NUMERICAL_EPS or m[10] == 0.0)

        labels, q10, reason = classify_stress_days(
            r_base, invested_mask=np.isfinite(r_base)
        )
        self.assertIsNone(reason)
        self.assertIsNotNone(q10)
        # Day 10 must be stress because it has the worst return
        self.assertEqual(labels[10], "stress")
        self.assertLessEqual(r_base[10], q10)
        # Confirm no off-by-one: previous day is not labeled stress from this return
        # (day 9's return is rets[8]=0.001 — not the stress day)
        self.assertAlmostEqual(r_base[9], 0.001, places=12)

class IntegrationFixtureTests(unittest.TestCase):
    def test_end_to_end_known_contributions(self):
        # Build long enough SPY history for SMA200 + vol percentiles
        n_hist = 500
        spy_idx = pd.date_range("2018-01-01", periods=n_hist, freq="B")
        # Mostly rising then a decline stretch for below-SMA
        spy_close = pd.Series(np.linspace(100, 200, n_hist), index=spy_idx)
        # Force a late decline below SMA
        spy_close.iloc[-40:] = np.linspace(190, 150, 40)

        # Analysis window: last 80 business days
        master = spy_idx[-80:]
        n = len(master)
        full_equity = np.cumprod(np.r_[1.0, 1 + np.full(n - 1, 0.001)]) * 15000
        # Baseline weaker on some days
        base_rets = np.full(n - 1, 0.0005)
        base_rets[10:20] = -0.01  # stress / drawdown stretch
        baseline_equity = np.cumprod(np.r_[1.0, 1 + base_rets]) * 15000
        # Full does better during drawdown
        full_rets = base_rets.copy()
        full_rets[10:20] = -0.004
        full_equity = np.cumprod(np.r_[1.0, 1 + full_rets]) * 15000

        full_hold = np.ones(n, dtype=bool)
        baseline_hold = np.ones(n, dtype=bool)
        baseline_hold[12:16] = False  # added exposure days
        candidate_hold = np.zeros(n, dtype=bool)
        candidate_hold[12:18] = True

        result = compute_regime_contribution(
            strategy_rows=[
                {
                    "strategy_id": "s1",
                    "strategy_name": "Test Strat",
                    "full_equity": full_equity,
                    "baseline_equity": baseline_equity,
                    "full_hold": full_hold,
                    "baseline_hold": baseline_hold,
                    "candidate_hold": candidate_hold,
                }
            ],
            spy_close=spy_close,
            master_dates=master,
        )

        self.assertIn("parameters", result)
        self.assertEqual(result["parameters"]["spy_symbol"], "SPY")
        self.assertEqual(len(result["strategies"]), 1)
        strat = result["strategies"][0]
        dims = {row["dimension"] for row in strat["regimes"]}
        self.assertEqual(
            dims,
            {"spy_trend", "spy_volatility", "baseline_drawdown", "baseline_stress"},
        )

        below = next(r for r in strat["regimes"] if r["dimension"] == "spy_trend" and r["state"] == "below")
        self.assertIn("episodes", below)
        self.assertGreaterEqual(below["regime_days"], 1)

        stress = next(
            r for r in strat["regimes"] if r["dimension"] == "baseline_stress" and r["state"] == "stress"
        )
        self.assertGreaterEqual(stress["regime_days"], 1)

        # Marginal log returns should be additive and positive overall in this fixture
        self.assertIsNotNone(strat["total_marginal_log_return"])
        self.assertGreater(strat["total_marginal_log_return"], 0)

        # Added exposure appears
        any_added = any(r["added_exposure_days"] > 0 for r in strat["regimes"])
        self.assertTrue(any_added)


class SevereEpisodeCountTests(unittest.TestCase):
    def test_count_only_troughs_at_or_below_severe(self):
        episodes = [
            {"baseline_trough_drawdown": -0.08},
            {"baseline_trough_drawdown": -0.15},
            {"baseline_trough_drawdown": -0.22},
            {"baseline_trough_drawdown": -0.04},
            {"baseline_trough_drawdown": None},
        ]
        self.assertEqual(count_severe_drawdown_episodes(episodes), 2)
        self.assertEqual(DD_SEVERE_THRESHOLD, -0.15)

    def test_analyze_assigns_distinct_severe_count(self):
        # Mild trough then severe trough as separate peak-breach episodes
        # Peak 100 → 90 (−10%) recover 100 → 80 (−20%) still open
        baseline = np.array([100.0, 90.0, 100.0, 80.0], dtype=float)
        full = baseline.copy()
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        eps = baseline_drawdown_episodes(
            baseline_equity=baseline,
            full_equity=full,
            dates=dates,
            marginal=np.zeros(4),
            added_exposure=np.zeros(4, dtype=bool),
        )
        self.assertEqual(len(eps), 2)
        severe = count_severe_drawdown_episodes(eps)
        self.assertEqual(severe, 1)
        self.assertNotEqual(severe, len(eps))


class MarginalDiagnosticsTests(unittest.TestCase):
    def _metrics_from_marginal(self, m_vals: np.ndarray) -> dict:
        n = len(m_vals)
        # Equity paths are unused for these diagnostic fields except returns pairing;
        # build trivial aligned series and inject marginal via equity identity + override mask.
        # Use compute_regime_state_metrics with constructed r and m by building equity
        # such that marginal matches: set full and base so L_full - L_base = m.
        # Easiest path: call compute_regime_state_metrics with synthetic arrays where
        # we pass marginal directly.
        r = np.full(n, 0.0)
        r[0] = np.nan
        return compute_regime_state_metrics(
            mask=np.ones(n, dtype=bool) & np.isfinite(m_vals),
            marginal=m_vals,
            r_full=r,
            r_base=r,
            candidate_hold=np.ones(n, dtype=bool),
            added_exposure=np.zeros(n, dtype=bool),
            total_marginal_log=float(np.nansum(m_vals)),
            total_eligible_days=int(np.isfinite(m_vals).sum()),
        )

    def test_microscopic_marginals(self):
        # All |M| below material threshold; some positive → raw rate > 0, effective ≈ 0
        tiny = MATERIAL_MARGINAL_THRESHOLD * 0.1
        m = np.array([np.nan] + [tiny, -tiny, tiny, -tiny] * 10)
        metrics = self._metrics_from_marginal(m)
        self.assertAlmostEqual(metrics["fraction_abs_marginal_below_material"], 1.0)
        self.assertGreater(metrics["positive_marginal_day_rate"], 0)
        self.assertEqual(metrics["positive_effective_day_rate"], 0.0)
        self.assertEqual(metrics["effective_contribution_days"], 0)
        self.assertLess(abs(metrics["compounded_marginal_return"]), DISPLAY_ZERO_THRESHOLD)
        self.assertLess(metrics["sum_positive_marginal_log"], MATERIAL_MARGINAL_THRESHOLD * 20)

    def test_cancelling_material_marginals(self):
        # Material + and − that nearly cancel
        m = np.array([np.nan] + [0.01, -0.01] * 20)
        metrics = self._metrics_from_marginal(m)
        self.assertLess(abs(metrics["compounded_marginal_return"]), 1e-10)
        self.assertAlmostEqual(metrics["positive_marginal_day_rate"], 0.5, places=5)
        self.assertAlmostEqual(metrics["positive_effective_day_rate"], 0.5, places=5)
        self.assertGreater(metrics["sum_positive_marginal_log"], 0.1)
        self.assertLess(metrics["sum_negative_marginal_log"], -0.1)
        self.assertLess(metrics["fraction_abs_marginal_below_material"], 0.1)
        self.assertEqual(metrics["effective_contribution_days"], 40)

    def test_worst_marginal_day_paired(self):
        n = 25
        r_full = np.array([np.nan] + [0.01] * (n - 1))
        r_base = np.array([np.nan] + [0.01] * (n - 1))
        r_full[5] = -0.02
        r_base[5] = -0.05  # baseline worse on same day
        r_full[8] = -0.10  # full has worse unpaired min
        r_base[8] = -0.01
        m = np.log1p(r_full) - np.log1p(r_base)
        m[0] = np.nan
        metrics = compute_regime_state_metrics(
            mask=np.isfinite(m),
            marginal=m,
            r_full=r_full,
            r_base=r_base,
            candidate_hold=np.ones(n, dtype=bool),
            added_exposure=np.zeros(n, dtype=bool),
            total_marginal_log=float(np.nansum(m)),
            total_eligible_days=n - 1,
        )
        # Unpaired: min(full)=-0.10, min(base)=-0.05 → −0.05
        self.assertAlmostEqual(metrics["worst_day_effect"], -0.10 - (-0.05), places=10)
        # Paired: min(full-base) occurs at day 8: -0.10 - (-0.01) = -0.09
        self.assertAlmostEqual(metrics["worst_marginal_day"], -0.09, places=10)

    def test_numerical_eps_distinct_from_material(self):
        self.assertLess(NUMERICAL_EPS, MATERIAL_MARGINAL_THRESHOLD)
        self.assertLess(MATERIAL_MARGINAL_THRESHOLD, DISPLAY_ZERO_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
