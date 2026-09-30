"""Regime contribution analysis for portfolio leave-one-out baselines.

Conventions (also returned in response ``parameters``):
- Execution lag: same-bar / close-synchronized. Day-t SPY trend and volatility
  labels use closes through t (matches default close-execution engine).
- Volatility percentiles: classify σ20,t using the prior 252 valid σ20 values
  only (current observation excluded). Interpolation: numpy linear.
- Daily attribution uses the full aligned equity path (no best-return-year drop).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd

SPY_SYMBOL = "SPY"
SMA_LOOKBACK = 200
VOL_LOOKBACK = 20
ANNUALIZATION = 252
VOL_PERCENTILE_LOOKBACK = 252
VOL_P30 = 30.0
VOL_P70 = 70.0
DD_NORMAL_THRESHOLD = -0.05
DD_SEVERE_THRESHOLD = -0.15
STRESS_PERCENTILE = 10.0
CONTRIBUTION_SHARE_EPS = 1e-8
# Floating-point comparisons only — never for economic materiality / evidence.
NUMERICAL_EPS = 1e-15
# Economic materiality for effective contribution days and evidence (~0.01 bp log).
MATERIAL_MARGINAL_THRESHOLD = 1e-6
# |exp(ΣM)-1| or |M| below this rounds to 0.00% at two decimal places.
DISPLAY_ZERO_THRESHOLD = 5e-5
ES_MIN_OBSERVATIONS = 20
STRESS_MIN_OBSERVATIONS = 20
PERCENTILE_METHOD = "linear"
EXECUTION_LAG_CONVENTION = "same_bar_close_synchronized"

EvidenceLevel = Literal["strong", "moderate", "limited", "insufficient"]

EVIDENCE_THRESHOLDS = {
    "strong": {
        "regime_days": 504,
        "candidate_active_days": 60,
        "effective_contribution_days": 60,
        "episodes": 8,
    },
    "moderate": {
        "regime_days": 252,
        "candidate_active_days": 30,
        "effective_contribution_days": 30,
        "episodes": 5,
    },
    "limited": {
        "regime_days": 126,
        "candidate_active_days": 20,
        "effective_contribution_days": 20,
        "episodes": 3,
    },
}

REGIME_PARAMETERS = {
    "spy_symbol": SPY_SYMBOL,
    "sma_lookback": SMA_LOOKBACK,
    "vol_lookback": VOL_LOOKBACK,
    "annualization": ANNUALIZATION,
    "vol_percentile_lookback": VOL_PERCENTILE_LOOKBACK,
    "vol_low_percentile": VOL_P30,
    "vol_high_percentile": VOL_P70,
    "baseline_drawdown_normal_threshold": DD_NORMAL_THRESHOLD,
    "baseline_drawdown_severe_threshold": DD_SEVERE_THRESHOLD,
    "stress_day_percentile": STRESS_PERCENTILE,
    "stress_q10_universe": "baseline_invested_days_only",
    "stress_label_partition": "stress_non_stress_flat",
    "stress_unique_day_classifier": "full_portfolio_return_vs_baseline_q10",
    "evidence_thresholds": EVIDENCE_THRESHOLDS,
    "execution_lag_convention": EXECUTION_LAG_CONVENTION,
    "percentile_interpolation": PERCENTILE_METHOD,
    "vol_percentile_window": "prior_252_excluding_current",
    "contribution_share_eps": CONTRIBUTION_SHARE_EPS,
    "numerical_eps": NUMERICAL_EPS,
    "material_marginal_threshold": MATERIAL_MARGINAL_THRESHOLD,
    "display_zero_threshold": DISPLAY_ZERO_THRESHOLD,
    "es_min_observations": ES_MIN_OBSERVATIONS,
    "stress_min_observations": STRESS_MIN_OBSERVATIONS,
}


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def deterministic_percentile(values: np.ndarray, q: float) -> float | None:
    """Deterministic percentile with linear interpolation."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return None
    return float(np.percentile(clean, q, method=PERCENTILE_METHOD))


def classify_evidence(
    *,
    regime_days: int,
    candidate_active_days: int,
    effective_contribution_days: int,
    episode_count: int | None = None,
) -> EvidenceLevel:
    """Classify evidence strength.

    Episode thresholds apply only when episode_count is not None.
    Effective contribution days use MATERIAL_MARGINAL_THRESHOLD, not NUMERICAL_EPS.
    """

    def meets(level: str) -> bool:
        thresholds = EVIDENCE_THRESHOLDS[level]
        if regime_days < thresholds["regime_days"]:
            return False
        if candidate_active_days < thresholds["candidate_active_days"]:
            return False
        if effective_contribution_days < thresholds["effective_contribution_days"]:
            return False
        if episode_count is not None and episode_count < thresholds["episodes"]:
            return False
        return True

    if meets("strong"):
        return "strong"
    if meets("moderate"):
        return "moderate"
    if meets("limited"):
        return "limited"
    return "insufficient"


def count_severe_drawdown_episodes(episodes: list[dict[str, Any]]) -> int:
    """Episodes whose baseline trough reached severe threshold (<= -15%)."""
    count = 0
    for ep in episodes:
        trough = ep.get("baseline_trough_drawdown")
        if trough is not None and math.isfinite(float(trough)) and float(trough) <= DD_SEVERE_THRESHOLD:
            count += 1
    return count


def compute_spy_sma200_labels(close: pd.Series) -> pd.Series:
    """Return labels: 'above', 'below', or None (unavailable)."""
    close = pd.to_numeric(close, errors="coerce")
    sma = close.rolling(window=SMA_LOOKBACK, min_periods=SMA_LOOKBACK).mean()
    labels = pd.Series(index=close.index, dtype=object)
    valid = close.notna() & sma.notna()
    labels.loc[valid & (close > sma)] = "above"
    labels.loc[valid & (close <= sma)] = "below"
    return labels


def compute_spy_realized_vol(
    close: pd.Series,
    *,
    lookback: int = VOL_LOOKBACK,
    annualization: int = ANNUALIZATION,
) -> pd.Series:
    """Annualized 20-day sample realized volatility of log returns."""
    close = pd.to_numeric(close, errors="coerce")
    log_ret = np.log(close / close.shift(1))
    # Sample std over lookback (ddof=1); rolling.std uses sample std by default.
    sigma = log_ret.rolling(window=lookback, min_periods=lookback).std(ddof=1) * math.sqrt(
        annualization
    )
    return sigma


def compute_spy_realized_vol_labels(
    close: pd.Series,
    *,
    lookback: int = VOL_LOOKBACK,
    annualization: int = ANNUALIZATION,
    percentile_lookback: int = VOL_PERCENTILE_LOOKBACK,
) -> pd.Series:
    """Return labels: 'low', 'normal', 'high', or None.

    Thresholds for day t use the prior ``percentile_lookback`` valid σ20 values
    (excluding σ20,t).
    """
    sigma = compute_spy_realized_vol(close, lookback=lookback, annualization=annualization)
    labels = pd.Series(index=close.index, dtype=object)
    values = sigma.to_numpy(dtype=float)
    n = len(values)
    for i in range(n):
        current = values[i]
        if not np.isfinite(current):
            continue
        prior = values[:i]
        prior = prior[np.isfinite(prior)]
        if prior.size < percentile_lookback:
            continue
        window = prior[-percentile_lookback:]
        p30 = float(np.percentile(window, VOL_P30, method=PERCENTILE_METHOD))
        p70 = float(np.percentile(window, VOL_P70, method=PERCENTILE_METHOD))
        if current < p30:
            labels.iloc[i] = "low"
        elif current > p70:
            labels.iloc[i] = "high"
        else:
            labels.iloc[i] = "normal"
    return labels


def align_spy_close_to_dates(
    spy_close: pd.Series,
    master_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Align SPY closes to portfolio dates without forward-filling missing closes."""
    indexed = pd.to_numeric(spy_close, errors="coerce").copy()
    indexed.index = pd.to_datetime(indexed.index).normalize()
    target = pd.DatetimeIndex(pd.to_datetime(master_dates)).normalize()
    # Drop duplicate index keeping last
    if indexed.index.has_duplicates:
        indexed = indexed[~indexed.index.duplicated(keep="last")]
    return indexed.reindex(target)


def compute_signed_drawdown(equity: np.ndarray) -> np.ndarray:
    """Signed drawdown: Equity / Peak - 1 (negative or zero)."""
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, equity / peak - 1.0, 0.0)
    return dd


def classify_baseline_drawdown(equity: np.ndarray) -> np.ndarray:
    """Return object array of 'normal' | 'drawdown' | 'severe_drawdown'."""
    dd = compute_signed_drawdown(equity)
    labels = np.empty(len(dd), dtype=object)
    labels[:] = None
    for i, value in enumerate(dd):
        if not np.isfinite(value):
            continue
        if value > DD_NORMAL_THRESHOLD:
            labels[i] = "normal"
        elif value > DD_SEVERE_THRESHOLD:
            labels[i] = "drawdown"
        else:
            labels[i] = "severe_drawdown"
    return labels


def classify_stress_days(
    baseline_returns: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    invested_mask: np.ndarray | None = None,
    unique_exposure_mask: np.ndarray | None = None,
    unique_returns: np.ndarray | None = None,
) -> tuple[np.ndarray, float | None, str | None]:
    """Classify stress / non-stress / flat using Q10 on baseline invested days only.

    Q10 is estimated from baseline invested days only (cash R=0 must not enter the
    sample — otherwise Q10 collapses to 0 and cash is labeled stress).

    For OR / binary overlays, leave-one-out marginal returns are ~0 whenever the
    baseline is already invested. Real contribution lives on unique-exposure days
    (baseline flat, full invested). Those days are classified by comparing the
    full-portfolio return that day to the same baseline Q10, so stress/non-stress
    can carry non-zero contribution. Idle days (everyone flat) stay ``flat``.

    Labels (on valid days):
    - ``stress`` / ``non_stress``: baseline invested, via R_base vs Q10; or
      unique exposure, via R_full vs Q10
    - ``flat``: baseline flat and not unique exposure (idle)

    Returns (labels, q10, unavailable_reason).
    """
    returns = np.asarray(baseline_returns, dtype=float)
    n = len(returns)
    labels = np.empty(n, dtype=object)
    labels[:] = None

    if valid_mask is None:
        valid_mask = np.isfinite(returns)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(returns)

    if invested_mask is None:
        invested_mask = valid_mask & (np.abs(returns) > NUMERICAL_EPS)
    else:
        invested_mask = np.asarray(invested_mask, dtype=bool) & valid_mask

    if unique_exposure_mask is None:
        unique_exposure_mask = np.zeros(n, dtype=bool)
    else:
        unique_exposure_mask = np.asarray(unique_exposure_mask, dtype=bool) & valid_mask
        # Unique exposure cannot overlap baseline invested.
        unique_exposure_mask = unique_exposure_mask & ~invested_mask

    if unique_returns is None:
        unique_returns = np.full(n, np.nan, dtype=float)
    else:
        unique_returns = np.asarray(unique_returns, dtype=float)

    sample = returns[invested_mask]
    if sample.size < STRESS_MIN_OBSERVATIONS:
        return labels, None, "insufficient_baseline_returns"

    q10 = float(np.percentile(sample, STRESS_PERCENTILE, method=PERCENTILE_METHOD))
    for i in range(n):
        if not valid_mask[i]:
            continue
        if invested_mask[i]:
            ref = returns[i]
        elif unique_exposure_mask[i] and np.isfinite(unique_returns[i]):
            ref = unique_returns[i]
        else:
            labels[i] = "flat"
            continue
        labels[i] = "stress" if ref <= q10 else "non_stress"
    return labels, q10, None


def daily_simple_returns(equity: np.ndarray) -> np.ndarray:
    """Simple returns with NaN on first bar and when prior equity is non-positive."""
    equity = np.asarray(equity, dtype=float)
    out = np.full(len(equity), np.nan, dtype=float)
    if len(equity) < 2:
        return out
    prev = equity[:-1]
    curr = equity[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(prev > 0, curr / prev - 1.0, np.nan)
    out[1:] = ret
    return out


def daily_log_returns(simple_returns: np.ndarray) -> np.ndarray:
    """Log returns from simple returns; invalid when R <= -1 or non-finite."""
    r = np.asarray(simple_returns, dtype=float)
    out = np.full(len(r), np.nan, dtype=float)
    valid = np.isfinite(r) & (r > -1.0)
    out[valid] = np.log1p(r[valid])
    return out


def marginal_log_returns(
    full_equity: np.ndarray,
    baseline_equity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (R_full, R_base, M) where M = L_full - L_base."""
    r_full = daily_simple_returns(full_equity)
    r_base = daily_simple_returns(baseline_equity)
    l_full = daily_log_returns(r_full)
    l_base = daily_log_returns(r_base)
    m = np.full(len(r_full), np.nan, dtype=float)
    both = np.isfinite(l_full) & np.isfinite(l_base)
    m[both] = l_full[both] - l_base[both]
    return r_full, r_base, m


def added_exposure_mask(
    full_hold: np.ndarray,
    baseline_hold: np.ndarray,
) -> np.ndarray:
    return np.asarray(full_hold, dtype=bool) & ~np.asarray(baseline_hold, dtype=bool)


def expected_shortfall(returns: np.ndarray, *, tail_fraction: float = 0.10) -> float | None:
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < ES_MIN_OBSERVATIONS:
        return None
    k = max(1, int(math.ceil(clean.size * tail_fraction)))
    worst = np.sort(clean)[:k]
    return float(np.mean(worst))


def downside_deviation(returns: np.ndarray) -> float | None:
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return None
    downside = np.minimum(clean, 0.0)
    return float(math.sqrt(np.mean(np.square(downside))))


def max_drawdown_over_interval(equity: np.ndarray) -> float | None:
    """Maximum signed drawdown (most negative) within the interval."""
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0 or not np.isfinite(equity).any():
        return None
    dd = compute_signed_drawdown(equity)
    finite = dd[np.isfinite(dd)]
    if finite.size == 0:
        return None
    return float(np.min(finite))


def compute_regime_state_metrics(
    *,
    mask: np.ndarray,
    marginal: np.ndarray,
    r_full: np.ndarray,
    r_base: np.ndarray,
    candidate_hold: np.ndarray,
    added_exposure: np.ndarray,
    total_marginal_log: float | None,
    total_eligible_days: int,
    dates: pd.DatetimeIndex | None = None,
    episode_count: int | None = None,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    valid = mask & np.isfinite(marginal)
    regime_days = int(valid.sum())
    candidate_active_days = int((valid & np.asarray(candidate_hold, dtype=bool)).sum())
    added_days = int((valid & np.asarray(added_exposure, dtype=bool)).sum())

    m_vals = marginal[valid]
    sum_m = float(np.sum(m_vals)) if regime_days else None
    mean_m = float(np.mean(m_vals)) if regime_days else None
    compounded = float(math.exp(sum_m) - 1.0) if sum_m is not None else None
    annualized = float(ANNUALIZATION * mean_m) if mean_m is not None else None
    per_added = (sum_m / added_days) if sum_m is not None and added_days > 0 else None

    contribution_share = None
    if (
        sum_m is not None
        and total_marginal_log is not None
        and abs(total_marginal_log) >= CONTRIBUTION_SHARE_EPS
    ):
        contribution_share = float(sum_m / total_marginal_log)

    rf = r_full[valid & np.isfinite(r_full)]
    rb = r_base[valid & np.isfinite(r_base)]
    both = valid & np.isfinite(r_full) & np.isfinite(r_base)
    rf_both = r_full[both]
    rb_both = r_base[both]

    es_full = expected_shortfall(rf_both)
    es_base = expected_shortfall(rb_both)
    es_improvement = None
    if es_full is not None and es_base is not None:
        es_improvement = float(es_full - es_base)

    worst_day_effect = None
    worst_marginal_day = None
    if rf_both.size > 0 and rb_both.size > 0:
        worst_day_effect = float(np.min(rf_both) - np.min(rb_both))
        worst_marginal_day = float(np.min(rf_both - rb_both))

    dd_full = downside_deviation(rf)
    dd_base = downside_deviation(rb)
    downside_improvement = None
    if dd_full is not None and dd_base is not None:
        downside_improvement = float(dd_base - dd_full)

    positive_rate = None
    positive_effective_rate = None
    effective_days = 0
    sum_pos = None
    sum_neg = None
    max_abs_m = None
    mean_abs_m = None
    frac_below_material = None
    frac_below_display = None

    if regime_days > 0:
        positive_rate = float(np.mean(m_vals > 0))
        positive_effective_rate = float(np.mean(m_vals > MATERIAL_MARGINAL_THRESHOLD))
        effective_days = int(np.sum(np.abs(m_vals) > MATERIAL_MARGINAL_THRESHOLD))
        pos_mask = m_vals > 0
        neg_mask = m_vals < 0
        sum_pos = float(np.sum(m_vals[pos_mask])) if pos_mask.any() else 0.0
        sum_neg = float(np.sum(m_vals[neg_mask])) if neg_mask.any() else 0.0
        abs_m = np.abs(m_vals)
        max_abs_m = float(np.max(abs_m))
        mean_abs_m = float(np.mean(abs_m))
        frac_below_material = float(np.mean(abs_m < MATERIAL_MARGINAL_THRESHOLD))
        frac_below_display = float(np.mean(abs_m < DISPLAY_ZERO_THRESHOLD))

    earliest = None
    latest = None
    if dates is not None and regime_days > 0:
        regime_dates = pd.DatetimeIndex(dates)[valid]
        earliest = str(regime_dates.min().date())
        latest = str(regime_dates.max().date())

    evidence = classify_evidence(
        regime_days=regime_days,
        candidate_active_days=candidate_active_days,
        effective_contribution_days=effective_days,
        episode_count=episode_count,
    )

    return {
        "regime_days": regime_days,
        "regime_share_of_sample": (
            float(regime_days / total_eligible_days) if total_eligible_days > 0 else None
        ),
        "candidate_active_days": candidate_active_days,
        "effective_contribution_days": effective_days,
        "added_exposure_days": added_days,
        "added_exposure_percent": (
            float(added_days / regime_days) if regime_days > 0 else None
        ),
        "marginal_log_return": sum_m,
        "compounded_marginal_return": compounded,
        "mean_marginal_daily_return": mean_m,
        "annualized_conditional_contribution_rate": annualized,
        "marginal_return_per_added_exposure_day": per_added,
        "contribution_share": contribution_share,
        "expected_shortfall_effect": es_improvement,
        "worst_day_effect": worst_day_effect,
        "worst_marginal_day": worst_marginal_day,
        "downside_deviation_effect": downside_improvement,
        "positive_marginal_day_rate": positive_rate,
        "positive_effective_day_rate": positive_effective_rate,
        "sum_positive_marginal_log": sum_pos,
        "sum_negative_marginal_log": sum_neg,
        "max_abs_marginal": max_abs_m,
        "mean_abs_marginal": mean_abs_m,
        "fraction_abs_marginal_below_material": frac_below_material,
        "fraction_abs_marginal_below_display": frac_below_display,
        "episode_count": episode_count,
        "earliest_eligible_date": earliest,
        "latest_eligible_date": latest,
        "evidence": evidence,
        "data_availability": "ok" if regime_days > 0 else "no_regime_days",
    }


def find_contiguous_episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive (start, end) index pairs for True runs."""
    mask = np.asarray(mask, dtype=bool)
    episodes: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            episodes.append((start, i - 1))
            start = None
    if start is not None:
        episodes.append((start, len(mask) - 1))
    return episodes


def below_sma_episodes(
    *,
    below_mask: np.ndarray,
    dates: pd.DatetimeIndex,
    r_full: np.ndarray,
    r_base: np.ndarray,
    marginal: np.ndarray,
    full_equity: np.ndarray,
    baseline_equity: np.ndarray,
    added_exposure: np.ndarray,
) -> list[dict[str, Any]]:
    episodes = []
    for start, end in find_contiguous_episodes(below_mask):
        sl = slice(start, end + 1)
        m_vals = marginal[sl]
        valid_m = np.isfinite(m_vals)
        sum_m = float(np.nansum(m_vals)) if valid_m.any() else None
        # Prefer sum of finite values only
        if valid_m.any():
            sum_m = float(np.sum(m_vals[valid_m]))
        else:
            sum_m = None
        compounded = float(math.exp(sum_m) - 1.0) if sum_m is not None else None

        # Compounded portfolio returns over episode using equity endpoints
        full_ret = None
        base_ret = None
        if full_equity[start] > 0 and np.isfinite(full_equity[end]):
            full_ret = float(full_equity[end] / full_equity[start] - 1.0)
        if baseline_equity[start] > 0 and np.isfinite(baseline_equity[end]):
            base_ret = float(baseline_equity[end] / baseline_equity[start] - 1.0)

        max_dd_full = max_drawdown_over_interval(full_equity[sl])
        max_dd_base = max_drawdown_over_interval(baseline_equity[sl])
        dd_improvement = None
        if max_dd_full is not None and max_dd_base is not None:
            # Positive means full had less severe (less negative) max DD
            dd_improvement = float(max_dd_full - max_dd_base)

        added_days = int(np.asarray(added_exposure[sl], dtype=bool).sum())
        open_episode = end == len(below_mask) - 1 and below_mask[end]
        # Closed if the next day exists and is not below — already true unless open at end
        if end < len(below_mask) - 1:
            open_episode = False

        helped = compounded is not None and compounded > 0
        episodes.append(
            {
                "start_date": str(pd.Timestamp(dates[start]).date()),
                "end_date": str(pd.Timestamp(dates[end]).date()),
                "status": "open" if open_episode else "closed",
                "trading_days": int(end - start + 1),
                "full_compounded_return": full_ret,
                "baseline_compounded_return": base_ret,
                "marginal_log_return": sum_m,
                "compounded_marginal_return": compounded,
                "max_drawdown_full": max_dd_full,
                "max_drawdown_baseline": max_dd_base,
                "max_drawdown_improvement": dd_improvement,
                "added_exposure_days": added_days,
                "helped": helped,
            }
        )
    return episodes


def baseline_drawdown_episodes(
    *,
    baseline_equity: np.ndarray,
    full_equity: np.ndarray,
    dates: pd.DatetimeIndex,
    marginal: np.ndarray,
    added_exposure: np.ndarray,
) -> list[dict[str, Any]]:
    """Episodes from peak breach until recovery to prior peak (or open)."""
    equity = np.asarray(baseline_equity, dtype=float)
    full = np.asarray(full_equity, dtype=float)
    n = len(equity)
    if n == 0:
        return []

    episodes: list[dict[str, Any]] = []
    peak = equity[0]
    peak_idx = 0
    in_dd = False
    start = 0
    trough_idx = 0
    trough_equity = equity[0]
    episode_peak = peak

    def recovery_days_from(start_i: int, eq: np.ndarray, peak_value: float) -> int | None:
        for j in range(start_i, len(eq)):
            if eq[j] >= peak_value:
                return int(j - start_i)
        return None

    i = 1
    while i < n:
        if not in_dd:
            if equity[i] >= peak:
                peak = equity[i]
                peak_idx = i
            elif equity[i] < peak:
                in_dd = True
                start = i
                episode_peak = peak
                trough_idx = i
                trough_equity = equity[i]
        else:
            if equity[i] < trough_equity:
                trough_idx = i
                trough_equity = equity[i]
            if equity[i] >= episode_peak:
                # Closed episode
                end = i
                _append_dd_episode(
                    episodes,
                    dates=dates,
                    start=start,
                    trough_idx=trough_idx,
                    end=end,
                    open_episode=False,
                    episode_peak=episode_peak,
                    trough_equity=trough_equity,
                    baseline_equity=equity,
                    full_equity=full,
                    marginal=marginal,
                    added_exposure=added_exposure,
                    recovery_days_from=recovery_days_from,
                    peak_idx=peak_idx,
                )
                in_dd = False
                peak = equity[i]
                peak_idx = i
        i += 1

    if in_dd:
        _append_dd_episode(
            episodes,
            dates=dates,
            start=start,
            trough_idx=trough_idx,
            end=n - 1,
            open_episode=True,
            episode_peak=episode_peak,
            trough_equity=trough_equity,
            baseline_equity=equity,
            full_equity=full,
            marginal=marginal,
            added_exposure=added_exposure,
            recovery_days_from=recovery_days_from,
            peak_idx=peak_idx,
        )
    return episodes


def _append_dd_episode(
    episodes: list[dict[str, Any]],
    *,
    dates: pd.DatetimeIndex,
    start: int,
    trough_idx: int,
    end: int,
    open_episode: bool,
    episode_peak: float,
    trough_equity: float,
    baseline_equity: np.ndarray,
    full_equity: np.ndarray,
    marginal: np.ndarray,
    added_exposure: np.ndarray,
    recovery_days_from,
    peak_idx: int,
) -> None:
    sl = slice(start, end + 1)
    baseline_trough_dd = (
        float(trough_equity / episode_peak - 1.0) if episode_peak > 0 else None
    )
    # Full drawdown over same interval relative to full equity at episode start peak proxy:
    # use full equity running peak from the bar before start if available, else start.
    pre = max(0, start - 1)
    full_ref_peak = float(np.max(full_equity[peak_idx : start + 1])) if start >= peak_idx else float(
        full_equity[pre]
    )
    if not np.isfinite(full_ref_peak) or full_ref_peak <= 0:
        full_ref_peak = float(full_equity[start]) if full_equity[start] > 0 else None
    full_trough = float(np.min(full_equity[sl]))
    full_trough_dd = (
        float(full_trough / full_ref_peak - 1.0)
        if full_ref_peak is not None and full_ref_peak > 0
        else None
    )
    trough_improvement = None
    if baseline_trough_dd is not None and full_trough_dd is not None:
        # Positive means full trough less severe than baseline
        trough_improvement = float(full_trough_dd - baseline_trough_dd)

    baseline_recovery = None if open_episode else int(end - start)
    full_recovery = None
    if not open_episode and full_ref_peak is not None and full_ref_peak > 0:
        full_recovery = recovery_days_from(start, full_equity, full_ref_peak)

    recovery_acceleration = None
    if baseline_recovery is not None and full_recovery is not None:
        recovery_acceleration = int(baseline_recovery - full_recovery)

    m_vals = marginal[sl]
    valid_m = np.isfinite(m_vals)
    sum_m = float(np.sum(m_vals[valid_m])) if valid_m.any() else None
    compounded = float(math.exp(sum_m) - 1.0) if sum_m is not None else None
    added_days = int(np.asarray(added_exposure[sl], dtype=bool).sum())

    improved_trough = trough_improvement is not None and trough_improvement > 0
    shortened_recovery = (
        recovery_acceleration is not None and recovery_acceleration > 0
    )
    helped = compounded is not None and compounded > 0

    episodes.append(
        {
            "start_date": str(pd.Timestamp(dates[start]).date()),
            "trough_date": str(pd.Timestamp(dates[trough_idx]).date()),
            "end_date": str(pd.Timestamp(dates[end]).date()),
            "status": "open" if open_episode else "closed",
            "baseline_peak_equity": _finite(episode_peak),
            "baseline_trough_drawdown": baseline_trough_dd,
            "full_trough_drawdown": full_trough_dd,
            "trough_improvement": trough_improvement,
            "baseline_recovery_days": baseline_recovery,
            "full_recovery_days": full_recovery,
            "recovery_acceleration": recovery_acceleration,
            "marginal_log_return": sum_m,
            "compounded_marginal_return": compounded,
            "added_exposure_days": added_days,
            "improved_trough": improved_trough,
            "shortened_recovery": shortened_recovery,
            "helped": helped,
            "trading_days": int(end - start + 1),
        }
    )


def contribution_concentration(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    abs_vals = [
        abs(float(ep["marginal_log_return"]))
        for ep in episodes
        if ep.get("marginal_log_return") is not None
        and math.isfinite(float(ep["marginal_log_return"]))
    ]
    denom = sum(abs_vals)
    if denom <= 0:
        return {
            "largest_episode_share": None,
            "concentration_warning": False,
            "concentration_message": None,
        }
    largest = max(abs_vals) / denom
    warning = largest > 0.5
    return {
        "largest_episode_share": float(largest),
        "concentration_warning": warning,
        "concentration_message": (
            "Most observed contribution came from one episode" if warning else None
        ),
    }


def _mask_for_state(labels: np.ndarray | pd.Series, state: str) -> np.ndarray:
    if isinstance(labels, pd.Series):
        arr = labels.to_numpy()
    else:
        arr = np.asarray(labels, dtype=object)
    return arr == state


def analyze_strategy_regimes(
    *,
    strategy_id: str,
    strategy_name: str,
    dates: pd.DatetimeIndex,
    full_equity: np.ndarray,
    baseline_equity: np.ndarray,
    full_hold: np.ndarray,
    baseline_hold: np.ndarray,
    candidate_hold: np.ndarray,
    spy_trend_labels: pd.Series,
    spy_vol_labels: pd.Series,
) -> dict[str, Any]:
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    r_full, r_base, marginal = marginal_log_returns(full_equity, baseline_equity)
    eligible = np.isfinite(marginal)
    total_eligible_days = int(eligible.sum())
    total_marginal_log = float(np.sum(marginal[eligible])) if total_eligible_days else 0.0

    added = added_exposure_mask(full_hold, baseline_hold)
    dd_labels = classify_baseline_drawdown(baseline_equity)
    stress_labels, stress_q10, stress_unavailable = classify_stress_days(
        r_base,
        valid_mask=eligible,
        invested_mask=np.asarray(baseline_hold, dtype=bool),
        unique_exposure_mask=added,
        unique_returns=r_full,
    )

    errors: list[dict[str, str]] = []
    regimes: list[dict[str, Any]] = []

    # --- SPY trend ---
    try:
        trend_arr = np.asarray(spy_trend_labels, dtype=object)
        if len(trend_arr) != len(dates):
            raise ValueError("SPY trend labels must align with master dates")
        below_raw = _mask_for_state(trend_arr, "below")
        below_mask = below_raw & eligible
        above_mask = _mask_for_state(trend_arr, "above") & eligible
        below_eps = below_sma_episodes(
            below_mask=below_raw,
            dates=dates,
            r_full=r_full,
            r_base=r_base,
            marginal=marginal,
            full_equity=full_equity,
            baseline_equity=baseline_equity,
            added_exposure=added,
        )
        concentration = contribution_concentration(below_eps)

        for state, mask, eps in (
            ("above", above_mask, None),
            ("below", below_mask, below_eps),
        ):
            metrics = compute_regime_state_metrics(
                mask=mask,
                marginal=marginal,
                r_full=r_full,
                r_base=r_base,
                candidate_hold=candidate_hold,
                added_exposure=added,
                total_marginal_log=total_marginal_log,
                total_eligible_days=total_eligible_days,
                dates=dates,
                episode_count=len(eps) if eps is not None else None,
            )
            row: dict[str, Any] = {
                "dimension": "spy_trend",
                "state": state,
                **metrics,
            }
            if eps is not None:
                row["episodes"] = eps
                row.update(concentration)
            regimes.append(row)
    except Exception as exc:  # noqa: BLE001 — isolate dimension failures
        errors.append({"dimension": "spy_trend", "message": str(exc)})

    # --- SPY volatility ---
    try:
        vol_arr = np.asarray(spy_vol_labels, dtype=object)
        if len(vol_arr) != len(dates):
            raise ValueError("SPY volatility labels must align with master dates")
        for state in ("low", "normal", "high"):
            mask = _mask_for_state(vol_arr, state) & eligible
            metrics = compute_regime_state_metrics(
                mask=mask,
                marginal=marginal,
                r_full=r_full,
                r_base=r_base,
                candidate_hold=candidate_hold,
                added_exposure=added,
                total_marginal_log=total_marginal_log,
                total_eligible_days=total_eligible_days,
                dates=dates,
            )
            regimes.append({"dimension": "spy_volatility", "state": state, **metrics})
    except Exception as exc:  # noqa: BLE001
        errors.append({"dimension": "spy_volatility", "message": str(exc)})

    # --- Baseline drawdown ---
    try:
        dd_eps = baseline_drawdown_episodes(
            baseline_equity=baseline_equity,
            full_equity=full_equity,
            dates=dates,
            marginal=marginal,
            added_exposure=added,
        )
        concentration = contribution_concentration(dd_eps)
        severe_episode_count = count_severe_drawdown_episodes(dd_eps)
        for state in ("normal", "drawdown", "severe_drawdown"):
            mask = (dd_labels == state) & eligible
            if state == "normal":
                ep_count = None
            elif state == "drawdown":
                ep_count = len(dd_eps)
            else:
                ep_count = severe_episode_count
            metrics = compute_regime_state_metrics(
                mask=mask,
                marginal=marginal,
                r_full=r_full,
                r_base=r_base,
                candidate_hold=candidate_hold,
                added_exposure=added,
                total_marginal_log=total_marginal_log,
                total_eligible_days=total_eligible_days,
                dates=dates,
                episode_count=ep_count,
            )
            row = {"dimension": "baseline_drawdown", "state": state, **metrics}
            if state == "drawdown":
                row["episodes"] = dd_eps
                row.update(concentration)
            regimes.append(row)
    except Exception as exc:  # noqa: BLE001
        errors.append({"dimension": "baseline_drawdown", "message": str(exc)})

    # --- Stress days ---
    try:
        for state in ("stress", "non_stress", "flat"):
            if stress_unavailable:
                regimes.append(
                    {
                        "dimension": "baseline_stress",
                        "state": state,
                        "regime_days": 0,
                        "regime_share_of_sample": None,
                        "candidate_active_days": 0,
                        "effective_contribution_days": 0,
                        "added_exposure_days": 0,
                        "added_exposure_percent": None,
                        "marginal_log_return": None,
                        "compounded_marginal_return": None,
                        "mean_marginal_daily_return": None,
                        "annualized_conditional_contribution_rate": None,
                        "marginal_return_per_added_exposure_day": None,
                        "contribution_share": None,
                        "expected_shortfall_effect": None,
                        "worst_day_effect": None,
                        "worst_marginal_day": None,
                        "downside_deviation_effect": None,
                        "positive_marginal_day_rate": None,
                        "positive_effective_day_rate": None,
                        "sum_positive_marginal_log": None,
                        "sum_negative_marginal_log": None,
                        "max_abs_marginal": None,
                        "mean_abs_marginal": None,
                        "fraction_abs_marginal_below_material": None,
                        "fraction_abs_marginal_below_display": None,
                        "episode_count": None,
                        "earliest_eligible_date": None,
                        "latest_eligible_date": None,
                        "evidence": "insufficient",
                        "data_availability": stress_unavailable,
                        "unavailable_reason": stress_unavailable,
                        "stress_q10": stress_q10,
                    }
                )
                continue
            mask = (stress_labels == state) & eligible
            metrics = compute_regime_state_metrics(
                mask=mask,
                marginal=marginal,
                r_full=r_full,
                r_base=r_base,
                candidate_hold=candidate_hold,
                added_exposure=added,
                total_marginal_log=total_marginal_log,
                total_eligible_days=total_eligible_days,
                dates=dates,
            )
            regimes.append(
                {
                    "dimension": "baseline_stress",
                    "state": state,
                    **metrics,
                    "stress_q10": stress_q10,
                }
            )
    except Exception as exc:  # noqa: BLE001
        errors.append({"dimension": "baseline_stress", "message": str(exc)})

    result: dict[str, Any] = {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "total_eligible_days": total_eligible_days,
        "total_marginal_log_return": total_marginal_log if total_eligible_days else None,
        "regimes": regimes,
    }
    if errors:
        result["errors"] = errors
    return result


def build_spy_regime_labels(
    spy_close: pd.Series,
    master_dates: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Align SPY and compute trend/vol labels indexed like master_dates."""
    aligned = align_spy_close_to_dates(spy_close, master_dates)
    # Compute indicators on the full SPY history when available for proper warmup,
    # then reindex to master dates without ffill.
    full_close = pd.to_numeric(spy_close, errors="coerce").copy()
    full_close.index = pd.to_datetime(full_close.index).normalize()
    if full_close.index.has_duplicates:
        full_close = full_close[~full_close.index.duplicated(keep="last")]
    full_close = full_close.sort_index()

    trend_full = compute_spy_sma200_labels(full_close)
    vol_full = compute_spy_realized_vol_labels(full_close)

    target = pd.DatetimeIndex(pd.to_datetime(master_dates)).normalize()
    trend = trend_full.reindex(target)
    vol = vol_full.reindex(target)
    return aligned, trend, vol


def compute_regime_contribution(
    *,
    strategy_rows: list[dict[str, Any]],
    spy_close: pd.Series,
    master_dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Compute regime analysis for each strategy row.

    Each ``strategy_rows`` item must include:
    strategy_id, strategy_name, full_equity, baseline_equity,
    full_hold, baseline_hold, candidate_hold
    (arrays aligned to master_dates).
    """
    _, trend, vol = build_spy_regime_labels(spy_close, master_dates)
    strategies = []
    for row in strategy_rows:
        strategies.append(
            analyze_strategy_regimes(
                strategy_id=row["strategy_id"],
                strategy_name=row["strategy_name"],
                dates=master_dates,
                full_equity=np.asarray(row["full_equity"], dtype=float),
                baseline_equity=np.asarray(row["baseline_equity"], dtype=float),
                full_hold=np.asarray(row["full_hold"], dtype=bool),
                baseline_hold=np.asarray(row["baseline_hold"], dtype=bool),
                candidate_hold=np.asarray(row["candidate_hold"], dtype=bool),
                spy_trend_labels=trend,
                spy_vol_labels=vol,
            )
        )
    return {
        "parameters": dict(REGIME_PARAMETERS),
        "strategies": strategies,
    }
