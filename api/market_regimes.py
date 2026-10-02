"""Entry-date market regime labels for backtest analysis and builder filters.

Regime flags are a separate indicator category. They are attached only when a
compiled strategy references them. Existing Breadth and SPYBull columns are unchanged.
Market breadth is RSI(14) of RSP/SPY. Semis breadth is RSI(14) of log(SMH/SPY).
Equity risk breadth is RSI(14) of log(XLY/XLP). Credit quality breadth is RSI(14) of log(HYG/LQD).
Credit risk on is RSI(14) of log(HYG/TLT). Bond duration is RSI(14) of log(TLT/SHY).
Copper/gold is RSI(14) of log(HG=F/GC=F). Sensitive materials breadth is RSI(14) of log(XLB/SPY).
Rate shock is the 20-day ^TNX change divided by 63-day daily-change volatility times √20.
The 10Y–3M curve is ^TNX minus ^IRX, and curve change is that spread's 20-day difference.
Rate shock × curve crosses the sign of rate shock with the sign of that 20-day curve change.
Sector breadth is the share of nine
sector ETFs closing strictly above their own SMA(50) or SMA(200). Sector deviation
from SMA50 is the mean of log(close / SMA50) across those same nine ETFs.
Dollar shock applies the rate-shock formula to UUP. Dollar + rates crosses the sign
of the rate shock with the sign of the dollar shock, and zero counts as ≤ 0.
Inflation trend is SMA(log(TIP/IEF), 20) minus SMA(log(TIP/IEF), 100). Inflation and
yield crosses the sign of the rate shock with the sign of that trend.
SPY bull/bear uses SMA(50) vs SMA(200) and is separate from the existing SPYBull flag.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import indicators as ind
import numpy as np
import pandas as pd
import ta

from stats import compute_aggregate_metrics, yearly_returns

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
BREADTH_RSI_WINDOW = 14
SPY_SMA_FAST = 50
SPY_SMA_SLOW = 200
SECTOR_SYMBOLS = ("XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLK", "XLU")
CREDIT_SYMBOLS = ("HYG", "LQD")
RATIO_SYMBOLS = ("TLT", "SHY", "HG=F", "GC=F")
YIELD_SYMBOLS = ("^TNX", "^IRX")
DOLLAR_INFLATION_SYMBOLS = ("UUP", "TIP", "IEF")
REGIME_SYMBOLS = (
    "^VIX",
    "^VXN",
    "SPY",
    "RSP",
    "SMH",
    *SECTOR_SYMBOLS,
    *CREDIT_SYMBOLS,
    *RATIO_SYMBOLS,
    *YIELD_SYMBOLS,
    *DOLLAR_INFLATION_SYMBOLS,
)
RATE_SHOCK_HORIZON = 20
RATE_SHOCK_VOL_WINDOW = 63
CURVE_CHANGE_HORIZON = 20

VIX_KEYS = ("le_15", "15_20", "20_30", "gt_30")
VXN_KEYS = VIX_KEYS
BREADTH_KEYS = ("lt_40", "40_50", "50_60", "gt_60")
SECTOR_BREADTH_KEYS = ("le_25", "25_50", "50_75", "gt_75")
ATR_EXPANDING = "expanding"
ATR_CONTRACTING = "contracting"
SPY_BULL = "bull"
SPY_BEAR = "bear"
RATE_SHOCK_KEYS = ("lt_neg_1", "neg_1_to_1", "gt_1")
CURVE_LEVEL_KEYS = ("inverted", "normal")
CURVE_CHANGE_KEYS = ("le_neg_50", "neg_50_neg_10", "neg_10_pos_10", "pos_10_pos_50", "ge_pos_50")
RATE_CURVE_KEYS = (
    "shock_pos_curve_pos",
    "shock_pos_curve_nonpos",
    "shock_neg_curve_pos",
    "shock_neg_curve_nonpos",
)
SECTOR_TREND_KEYS = ("lt_neg_5", "neg_5_to_0", "zero_to_pos_5", "gt_pos_5")
DOLLAR_RATES_KEYS = (
    "tnx_nonpos_dollar_nonpos",
    "tnx_nonpos_dollar_pos",
    "tnx_pos_dollar_nonpos",
    "tnx_pos_dollar_pos",
)
INFLATION_KEYS = ("nonpos", "pos")
INFLATION_YIELD_KEYS = (
    "tnx_rising_inflation_rising",
    "tnx_rising_inflation_falling",
    "tnx_falling_inflation_rising",
    "tnx_falling_inflation_falling",
)
INFLATION_FAST = 20
INFLATION_SLOW = 100

MARKET_REGIME_KEYS = (
    "vix_regime",
    "vxn_regime",
    "spy_regime",
    "market_breadth_regime",
    "semis_breadth_regime",
    "equity_risk_breadth_regime",
    "credit_risk_breadth_regime",
    "credit_risk_on_regime",
    "bond_duration_regime",
    "copper_gold_regime",
    "materials_breadth_regime",
    "sector_breadth_50_regime",
    "sector_breadth_200_regime",
    "sector_trend_50_regime",
    "rate_shock_regime",
    "curve_10y3m_regime",
    "curve_change_20_regime",
    "rate_curve_regime",
    "dollar_shock_regime",
    "dollar_rates_regime",
    "inflation_regime",
    "inflation_yield_regime",
)

_CALENDAR_CACHE: dict[tuple[int, object], pd.DataFrame] = {}


def clear_regime_calendar_cache() -> None:
    _CALENDAR_CACHE.clear()


def _finite(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _classify_upper_inclusive(value, cuts: tuple[float, ...], keys: tuple[str, ...], *, first_inclusive: bool) -> str | None:
    """Partition a number into buckets that share edges.

    first_inclusive False: value < cuts[0], then value <= each later cut, else the last key.
    first_inclusive True: value <= each cut in order, else the last key.
    """
    number = _finite(value)
    if number is None:
        return None
    if not first_inclusive:
        if number < cuts[0]:
            return keys[0]
        for cut, key in zip(cuts[1:], keys[1:-1]):
            if number <= cut:
                return key
        return keys[-1]
    for cut, key in zip(cuts, keys[:-1]):
        if number <= cut:
            return key
    return keys[-1]


def classify_vix(value) -> str | None:
    """<= 15, 15–20, 20–30, > 30. Shared edges belong to the lower bucket."""
    return _classify_upper_inclusive(value, (15, 20, 30), VIX_KEYS, first_inclusive=True)


def classify_vxn(value) -> str | None:
    """Same buckets as VIX: <= 15, 15–20, 20–30, > 30."""
    return classify_vix(value)


def classify_breadth(value) -> str | None:
    """< 40, 40–50, 50–60, > 60. Shared edges belong to the lower bucket."""
    return _classify_upper_inclusive(value, (40, 50, 60), BREADTH_KEYS, first_inclusive=False)


def classify_sector_breadth(value) -> str | None:
    """<= 25%, 25–50%, 50–75%, > 75%. Shared edges belong to the lower bucket."""
    return _classify_upper_inclusive(value, (0.25, 0.50, 0.75), SECTOR_BREADTH_KEYS, first_inclusive=True)


def classify_atr(atr20, atr50) -> str | None:
    fast = _finite(atr20)
    slow = _finite(atr50)
    if fast is None or slow is None:
        return None
    if fast > slow:
        return ATR_EXPANDING
    return ATR_CONTRACTING


def classify_rate_shock(value) -> str | None:
    """< -1, -1 through 1, > 1. The endpoints stay in the middle bucket."""
    number = _finite(value)
    if number is None:
        return None
    if number < -1:
        return RATE_SHOCK_KEYS[0]
    if number > 1:
        return RATE_SHOCK_KEYS[2]
    return RATE_SHOCK_KEYS[1]


def classify_sector_trend(value) -> str | None:
    """< -5%, -5% through 0%, above 0% through +5%, > +5%.

    The value is a log deviation, so -5% is -0.05. Shared edges belong to the
    lower bucket, which puts -5% and 0% in the second bucket and +5% in the third.
    """
    return _classify_upper_inclusive(value, (-0.05, 0.0, 0.05), SECTOR_TREND_KEYS, first_inclusive=False)


def classify_inflation(value) -> str | None:
    """<= 0, then > 0. Zero stays in the lower bucket."""
    number = _finite(value)
    if number is None:
        return None
    if number <= 0:
        return INFLATION_KEYS[0]
    return INFLATION_KEYS[1]


def classify_dollar_rates(rate_shock, dollar_shock) -> str | None:
    """Sign of the rate shock crossed with the sign of the dollar shock.

    Zero on either side stays with ≤ 0. A missing value is unlabeled.
    """
    rates = _finite(rate_shock)
    dollar = _finite(dollar_shock)
    if rates is None or dollar is None:
        return None
    if rates <= 0 and dollar <= 0:
        return DOLLAR_RATES_KEYS[0]
    if rates <= 0:
        return DOLLAR_RATES_KEYS[1]
    if dollar <= 0:
        return DOLLAR_RATES_KEYS[2]
    return DOLLAR_RATES_KEYS[3]


def classify_inflation_yield(rate_shock, inflation_trend) -> str | None:
    """Sign of the rate shock crossed with the sign of the inflation trend.

    Above 0 is rising. Zero on either side stays with falling. A missing value
    is unlabeled.
    """
    rates = _finite(rate_shock)
    inflation = _finite(inflation_trend)
    if rates is None or inflation is None:
        return None
    if rates > 0 and inflation > 0:
        return INFLATION_YIELD_KEYS[0]
    if rates > 0:
        return INFLATION_YIELD_KEYS[1]
    if inflation > 0:
        return INFLATION_YIELD_KEYS[2]
    return INFLATION_YIELD_KEYS[3]


def classify_rate_curve(rate_shock, curve_change) -> str | None:
    """Sign of rate shock crossed with the sign of the 20-day curve change.

    Rate shock must be strictly above or below 0, so 0 is unlabeled. A curve
    change of 0 stays with the non-positive side.
    """
    shock = _finite(rate_shock)
    change = _finite(curve_change)
    if shock is None or change is None or shock == 0:
        return None
    if shock > 0 and change > 0:
        return RATE_CURVE_KEYS[0]
    if shock > 0:
        return RATE_CURVE_KEYS[1]
    if change > 0:
        return RATE_CURVE_KEYS[2]
    return RATE_CURVE_KEYS[3]


def classify_curve_10y3m(value) -> str | None:
    """< 0 inverted, >= 0 normal."""
    number = _finite(value)
    if number is None:
        return None
    if number < 0:
        return CURVE_LEVEL_KEYS[0]
    return CURVE_LEVEL_KEYS[1]


def classify_curve_change(value) -> str | None:
    """<= -0.50, then through -0.10, +0.10, and up to but not including +0.50, else >= +0.50.

    Inner edges belong to the lower bucket. The ends follow the stated inequalities,
    so -0.50 is in the bottom bucket and +0.50 is in the top bucket. Values are
    percentage points of yield: 0.10 is 10 bp.
    """
    number = _finite(value)
    if number is None:
        return None
    if number <= -0.50:
        return CURVE_CHANGE_KEYS[0]
    if number <= -0.10:
        return CURVE_CHANGE_KEYS[1]
    if number <= 0.10:
        return CURVE_CHANGE_KEYS[2]
    if number < 0.50:
        return CURVE_CHANGE_KEYS[3]
    return CURVE_CHANGE_KEYS[4]


def classify_spy(sma50, sma200) -> str | None:
    fast = _finite(sma50)
    slow = _finite(sma200)
    if fast is None or slow is None:
        return None
    if fast >= slow:
        return SPY_BULL
    return SPY_BEAR


def atr_regime_from_row(row: pd.Series) -> str | None:
    if "ATR20" not in row.index or "ATR50" not in row.index:
        return None
    return classify_atr(row["ATR20"], row["ATR50"])


def _normalize_timestamp(value) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _as_day_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    idx = pd.DatetimeIndex(pd.to_datetime(series.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out = pd.Series(values.to_numpy(), index=idx.normalize(), dtype=float)
    return out[~out.index.duplicated(keep="last")].sort_index()


def _rsi(series: pd.Series, window: int = BREADTH_RSI_WINDOW) -> pd.Series:
    """RSI on the series' own real observations.

    Leading gaps (a symbol that lists later) are dropped before the window, so
    they stay blank instead of coming back as 100. The first ``window - 1``
    real observations are warmup and stay blank too.
    """
    values = pd.to_numeric(series, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    clean = values.dropna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if len(clean) < window:
        return out
    rsi = ta.momentum.RSIIndicator(clean, window=window).rsi().copy()
    rsi.iloc[: window - 1] = np.nan
    return rsi.reindex(series.index)


def _sector_breadth(closes: list[pd.Series], index: pd.Index, window: int) -> pd.Series:
    """Share of sectors with close > SMA(close, window). Denominator is always 9.

    Each SMA uses that sector's own non-null closes, so one holiday NaN does not
    blank the following window. A day is blank until every sector has a completed
    SMA, which is also when a later-listed sector joins the count.
    """
    if len(closes) < len(SECTOR_SYMBOLS):
        return pd.Series(np.nan, index=index, dtype=float)
    above = np.zeros(len(index), dtype=float)
    ready = np.ones(len(index), dtype=bool)
    for series in closes:
        clean = _as_day_series(series).dropna()
        if clean.empty:
            return pd.Series(np.nan, index=index, dtype=float)
        sma = clean.rolling(window).mean()
        aligned_close = clean.reindex(index).to_numpy(dtype=float)
        aligned_sma = sma.reindex(index).to_numpy(dtype=float)
        finite = np.isfinite(aligned_close) & np.isfinite(aligned_sma)
        ready &= finite
        above += (finite & (aligned_close > aligned_sma)).astype(float)
    breadth = above / len(SECTOR_SYMBOLS)
    breadth[~ready] = np.nan
    return pd.Series(breadth, index=index, dtype=float)


def _sector_sma_deviation(closes: list[pd.Series], index: pd.Index, window: int) -> pd.Series:
    """Mean of log(close / SMA(close, window)) across the nine sectors.

    Blank until every sector has a completed SMA and a positive close. Each SMA
    uses that sector's own non-null closes, so one holiday NaN does not blank
    the following window.
    """
    blank = pd.Series(np.nan, index=index, dtype=float)
    if len(closes) < len(SECTOR_SYMBOLS):
        return blank
    total = np.zeros(len(index), dtype=float)
    ready = np.ones(len(index), dtype=bool)
    for series in closes:
        clean = _as_day_series(series).dropna()
        clean = clean[clean > 0]
        if clean.empty:
            return blank
        sma = clean.rolling(window).mean()
        aligned_close = clean.reindex(index).to_numpy(dtype=float)
        aligned_sma = sma.reindex(index).to_numpy(dtype=float)
        finite = np.isfinite(aligned_close) & np.isfinite(aligned_sma) & (aligned_sma > 0)
        ready &= finite
        deviation = np.zeros(len(index), dtype=float)
        deviation[finite] = np.log(aligned_close[finite] / aligned_sma[finite])
        total += deviation
    mean = total / len(SECTOR_SYMBOLS)
    mean[~ready] = np.nan
    return pd.Series(mean, index=index, dtype=float)


def _log_ratio_rsi(
    numerator: pd.Series | None,
    denominator: pd.Series | None,
    index: pd.Index,
) -> pd.Series:
    """RSI(14) of log(numerator / denominator), aligned to the regime calendar."""
    if numerator is None or denominator is None:
        return pd.Series(np.nan, index=index, dtype=float)
    num = _as_day_series(numerator).reindex(index)
    den = _as_day_series(denominator).reindex(index)
    ratio = num / den
    return _rsi(pd.Series(np.log(ratio.where(ratio > 0)), index=index))


def _own_closes(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return _as_day_series(series).dropna()


def _paired_yield_curve(tnx: pd.Series | None, irx: pd.Series | None) -> pd.Series | None:
    """^TNX minus ^IRX on sessions where both yields printed."""
    left = _own_closes(tnx)
    right = _own_closes(irx)
    if left.empty or right.empty:
        return None
    paired = pd.concat({"tnx": left, "irx": right}, axis=1).dropna()
    if paired.empty:
        return None
    return paired["tnx"] - paired["irx"]


def _rate_shock(tnx: pd.Series | None, index: pd.Index) -> pd.Series:
    """20-day ^TNX change divided by 63-day daily-change volatility scaled by √20.

    Computed on ^TNX's own closes, then aligned to the regime calendar. A zero
    volatility window stays blank.
    """
    blank = pd.Series(np.nan, index=index, dtype=float)
    closes = _own_closes(tnx)
    if len(closes) <= RATE_SHOCK_VOL_WINDOW:
        return blank
    daily = closes.diff()
    vol = daily.rolling(RATE_SHOCK_VOL_WINDOW).std(ddof=1)
    shock = (closes - closes.shift(RATE_SHOCK_HORIZON)) / (vol * math.sqrt(RATE_SHOCK_HORIZON))
    return shock.replace([np.inf, -np.inf], np.nan).reindex(index)


def _curve_10y3m(tnx: pd.Series | None, irx: pd.Series | None, index: pd.Index) -> pd.Series:
    blank = pd.Series(np.nan, index=index, dtype=float)
    curve = _paired_yield_curve(tnx, irx)
    if curve is None:
        return blank
    return curve.reindex(index)


def _curve_change_20(tnx: pd.Series | None, irx: pd.Series | None, index: pd.Index) -> pd.Series:
    blank = pd.Series(np.nan, index=index, dtype=float)
    curve = _paired_yield_curve(tnx, irx)
    if curve is None:
        return blank
    return (curve - curve.shift(CURVE_CHANGE_HORIZON)).reindex(index)


def _inflation_trend(tip: pd.Series | None, ief: pd.Series | None, index: pd.Index) -> pd.Series:
    """SMA(log(TIP/IEF), 20) minus SMA(log(TIP/IEF), 100) on paired sessions."""
    blank = pd.Series(np.nan, index=index, dtype=float)
    left = _own_closes(tip)
    right = _own_closes(ief)
    if left.empty or right.empty:
        return blank
    paired = pd.concat({"tip": left, "ief": right}, axis=1).dropna()
    if paired.empty:
        return blank
    ratio = paired["tip"] / paired["ief"]
    logged = pd.Series(np.log(ratio.where(ratio > 0)), index=paired.index)
    logged = logged.replace([np.inf, -np.inf], np.nan).dropna()
    if len(logged) < INFLATION_SLOW:
        return blank
    trend = logged.rolling(INFLATION_FAST).mean() - logged.rolling(INFLATION_SLOW).mean()
    return trend.replace([np.inf, -np.inf], np.nan).reindex(index)


def build_regime_calendar(
    vix: pd.Series,
    vxn: pd.Series,
    spy: pd.Series,
    rsp: pd.Series,
    sector_closes: list[pd.Series] | None = None,
    smh: pd.Series | None = None,
    xly: pd.Series | None = None,
    xlp: pd.Series | None = None,
    hyg: pd.Series | None = None,
    lqd: pd.Series | None = None,
    tlt: pd.Series | None = None,
    shy: pd.Series | None = None,
    copper: pd.Series | None = None,
    gold: pd.Series | None = None,
    xlb: pd.Series | None = None,
    tnx: pd.Series | None = None,
    irx: pd.Series | None = None,
    uup: pd.Series | None = None,
    tip: pd.Series | None = None,
    ief: pd.Series | None = None,
) -> pd.DataFrame:
    """Daily VIX, VXN, SPY SMA(50/200), and market-breadth RSI, indexed by session date."""
    # Drop sessions with no SPY close (Yahoo inserts holiday rows as NaN in a
    # multi-ticker download). A NaN inside the rolling window would blank SMA
    # for the next 200 sessions.
    spy_close = _as_day_series(spy).dropna()
    vix_close = _as_day_series(vix).reindex(spy_close.index)
    vxn_close = _as_day_series(vxn).reindex(spy_close.index)
    rsp_close = _as_day_series(rsp).reindex(spy_close.index)
    ratio = rsp_close / spy_close
    frame = pd.DataFrame(
        {
            "vix": vix_close,
            "vxn": vxn_close,
            "spy_sma50": spy_close.rolling(SPY_SMA_FAST).mean(),
            "spy_sma200": spy_close.rolling(SPY_SMA_SLOW).mean(),
            "breadth": _rsi(ratio),
            "breadth_semis": _log_ratio_rsi(smh, spy_close, spy_close.index),
            "breadth_equity_risk": _log_ratio_rsi(xly, xlp, spy_close.index),
            "breadth_credit_risk": _log_ratio_rsi(hyg, lqd, spy_close.index),
            "breadth_credit_risk_on": _log_ratio_rsi(hyg, tlt, spy_close.index),
            "breadth_bond_duration": _log_ratio_rsi(tlt, shy, spy_close.index),
            "breadth_copper_gold": _log_ratio_rsi(copper, gold, spy_close.index),
            "breadth_materials": _log_ratio_rsi(xlb, spy_close, spy_close.index),
            "sector_breadth_50": _sector_breadth(sector_closes or [], spy_close.index, SPY_SMA_FAST),
            "sector_breadth_200": _sector_breadth(sector_closes or [], spy_close.index, SPY_SMA_SLOW),
            "sector_trend_50": _sector_sma_deviation(sector_closes or [], spy_close.index, SPY_SMA_FAST),
            "rate_shock": _rate_shock(tnx, spy_close.index),
            "curve_10y3m": _curve_10y3m(tnx, irx, spy_close.index),
            "curve_change_20": _curve_change_20(tnx, irx, spy_close.index),
            "dollar_shock": _rate_shock(uup, spy_close.index),
            "inflation_trend": _inflation_trend(tip, ief, spy_close.index),
        },
        index=spy_close.index,
    )
    frame.index.name = "Date"
    return frame


def _empty_market_regimes() -> dict[str, None]:
    return {key: None for key in MARKET_REGIME_KEYS}


def market_regimes_for_timestamp(value, calendar: pd.DataFrame | None) -> dict[str, str | None]:
    empty = _empty_market_regimes()
    if calendar is None or calendar.empty:
        return empty
    day = _normalize_timestamp(value)
    if day is None or day not in calendar.index:
        return empty
    row = calendar.loc[day]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return {
        "vix_regime": classify_vix(row.get("vix")),
        "vxn_regime": classify_vxn(row.get("vxn")),
        "spy_regime": classify_spy(row.get("spy_sma50"), row.get("spy_sma200")),
        "market_breadth_regime": classify_breadth(row.get("breadth")),
        "semis_breadth_regime": classify_breadth(row.get("breadth_semis")),
        "equity_risk_breadth_regime": classify_breadth(row.get("breadth_equity_risk")),
        "credit_risk_breadth_regime": classify_breadth(row.get("breadth_credit_risk")),
        "credit_risk_on_regime": classify_breadth(row.get("breadth_credit_risk_on")),
        "bond_duration_regime": classify_breadth(row.get("breadth_bond_duration")),
        "copper_gold_regime": classify_breadth(row.get("breadth_copper_gold")),
        "materials_breadth_regime": classify_breadth(row.get("breadth_materials")),
        "sector_breadth_50_regime": classify_sector_breadth(row.get("sector_breadth_50")),
        "sector_breadth_200_regime": classify_sector_breadth(row.get("sector_breadth_200")),
        "sector_trend_50_regime": classify_sector_trend(row.get("sector_trend_50")),
        "rate_shock_regime": classify_rate_shock(row.get("rate_shock")),
        "curve_10y3m_regime": classify_curve_10y3m(row.get("curve_10y3m")),
        "curve_change_20_regime": classify_curve_change(row.get("curve_change_20")),
        "rate_curve_regime": classify_rate_curve(row.get("rate_shock"), row.get("curve_change_20")),
        "dollar_shock_regime": classify_rate_shock(row.get("dollar_shock")),
        "dollar_rates_regime": classify_dollar_rates(row.get("rate_shock"), row.get("dollar_shock")),
        "inflation_regime": classify_inflation(row.get("inflation_trend")),
        "inflation_yield_regime": classify_inflation_yield(row.get("rate_shock"), row.get("inflation_trend")),
    }


_SHARPE_BUCKETS: dict[str, tuple[str, ...]] = {
    "vix": VIX_KEYS,
    "vxn": VXN_KEYS,
    "atr": (ATR_EXPANDING, ATR_CONTRACTING),
    "spy": (SPY_BULL, SPY_BEAR),
    "market_breadth": BREADTH_KEYS,
    "semis_breadth": BREADTH_KEYS,
    "equity_risk_breadth": BREADTH_KEYS,
    "credit_risk_breadth": BREADTH_KEYS,
    "credit_risk_on": BREADTH_KEYS,
    "bond_duration": BREADTH_KEYS,
    "copper_gold": BREADTH_KEYS,
    "materials_breadth": BREADTH_KEYS,
    "sector_breadth_50": SECTOR_BREADTH_KEYS,
    "sector_breadth_200": SECTOR_BREADTH_KEYS,
    "sector_trend_50": SECTOR_TREND_KEYS,
    "rate_shock": RATE_SHOCK_KEYS,
    "curve_10y3m": CURVE_LEVEL_KEYS,
    "curve_change_20": CURVE_CHANGE_KEYS,
    "rate_curve": RATE_CURVE_KEYS,
    "dollar_shock": RATE_SHOCK_KEYS,
    "dollar_rates": DOLLAR_RATES_KEYS,
    "inflation": INFLATION_KEYS,
    "inflation_yield": INFLATION_YIELD_KEYS,
}

_VIX_BUCKET_LABELS = {"le_15": "≤ 15", "15_20": "15–20", "20_30": "20–30", "gt_30": "> 30"}
_BREADTH_BUCKET_LABELS = {"lt_40": "< 40", "40_50": "40–50", "50_60": "50–60", "gt_60": "> 60"}
_SECTOR_BUCKET_LABELS = {"le_25": "≤ 25%", "25_50": "25–50%", "50_75": "50–75%", "gt_75": "> 75%"}
_RATE_CURVE_LABELS = {
    "shock_pos_curve_pos": "> 0, > 0",
    "shock_pos_curve_nonpos": "> 0, ≤ 0",
    "shock_neg_curve_pos": "< 0, > 0",
    "shock_neg_curve_nonpos": "< 0, ≤ 0",
}
_SECTOR_TREND_LABELS = {
    "lt_neg_5": "< -5%",
    "neg_5_to_0": "-5% to 0%",
    "zero_to_pos_5": "0% to +5%",
    "gt_pos_5": "> +5%",
}
_DOLLAR_RATES_LABELS = {
    "tnx_nonpos_dollar_nonpos": "≤ 0, ≤ 0",
    "tnx_nonpos_dollar_pos": "≤ 0, > 0",
    "tnx_pos_dollar_nonpos": "> 0, ≤ 0",
    "tnx_pos_dollar_pos": "> 0, > 0",
}
_INFLATION_YIELD_LABELS = {
    "tnx_rising_inflation_rising": "> 0, > 0",
    "tnx_rising_inflation_falling": "> 0, ≤ 0",
    "tnx_falling_inflation_rising": "≤ 0, > 0",
    "tnx_falling_inflation_falling": "≤ 0, ≤ 0",
}
_CURVE_CHANGE_LABELS = {
    "le_neg_50": "≤ -50 bp",
    "neg_50_neg_10": "-50 to -10",
    "neg_10_pos_10": "-10 to +10",
    "pos_10_pos_50": "+10 to +50",
    "ge_pos_50": "≥ +50 bp",
}

# Titles and hints match the regime charts. market_wide flags use the shared calendar.
# ATR is the traded symbol, so it stays on each confirm symbol.
_REGIME_DIMENSION_META: tuple[dict[str, Any], ...] = (
    {"id": "vix", "title": "VIX", "hint": "VIX close that day", "labels": _VIX_BUCKET_LABELS, "market_wide": True},
    {"id": "vxn", "title": "VXN", "hint": "VXN close that day", "labels": _VIX_BUCKET_LABELS, "market_wide": True},
    {
        "id": "atr",
        "title": "ATR",
        "hint": "Traded symbol ATR(20) vs ATR(50) that day",
        "labels": {ATR_EXPANDING: "ATR(20) > ATR(50)", ATR_CONTRACTING: "ATR(20) ≤ ATR(50)"},
        "market_wide": False,
    },
    {
        "id": "spy",
        "title": "SPY regime",
        "hint": "Bull: SMA50 ≥ SMA200 · Bear: SMA50 < SMA200",
        "labels": {SPY_BULL: "Bull", SPY_BEAR: "Bear"},
        "market_wide": True,
    },
    {"id": "market_breadth", "title": "Market breadth", "hint": "RSI(14) of RSP / SPY", "labels": _BREADTH_BUCKET_LABELS, "market_wide": True},
    {"id": "semis_breadth", "title": "Semis breadth", "hint": "RSI(14) of log(SMH / SPY)", "labels": _BREADTH_BUCKET_LABELS, "market_wide": True},
    {"id": "equity_risk_breadth", "title": "Equity Risk Breadth", "hint": "RSI(14) of log(XLY / XLP)", "labels": _BREADTH_BUCKET_LABELS, "market_wide": True},
    {"id": "credit_risk_breadth", "title": "Credit Quality Breadth", "hint": "RSI(14) of log(HYG / LQD)", "labels": _BREADTH_BUCKET_LABELS, "market_wide": True},
    {"id": "credit_risk_on", "title": "Credit Risk On", "hint": "RSI(14) of log(HYG / TLT)", "labels": _BREADTH_BUCKET_LABELS, "market_wide": True},
    {"id": "bond_duration", "title": "Bond Duration Regime", "hint": "RSI(14) of log(TLT / SHY)", "labels": _BREADTH_BUCKET_LABELS, "market_wide": True},
    {"id": "copper_gold", "title": "Copper/Gold", "hint": "RSI(14) of log(HG=F / GC=F)", "labels": _BREADTH_BUCKET_LABELS, "market_wide": True},
    {"id": "materials_breadth", "title": "Sensitive Materials Breadth", "hint": "RSI(14) of log(XLB / SPY)", "labels": _BREADTH_BUCKET_LABELS, "market_wide": True},
    {
        "id": "sector_breadth_50",
        "title": "Sector breadth 50",
        "hint": "Share of XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLK, XLU above their 50-day SMA",
        "labels": _SECTOR_BUCKET_LABELS,
        "market_wide": True,
    },
    {
        "id": "sector_breadth_200",
        "title": "Sector breadth 200",
        "hint": "Share of XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLK, XLU above their 200-day SMA",
        "labels": _SECTOR_BUCKET_LABELS,
        "market_wide": True,
    },
    {
        "id": "sector_trend_50",
        "title": "Sector deviation from SMA50",
        "hint": "Mean of log(close / SMA50) across XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLK, XLU. −0.05 is −5%",
        "labels": _SECTOR_TREND_LABELS,
        "market_wide": True,
    },
    {
        "id": "rate_shock",
        "title": "Rate shock",
        "hint": "20-day ^TNX change divided by the 63-day standard deviation of daily changes, times √20",
        "labels": {"lt_neg_1": "< -1", "neg_1_to_1": "-1 to 1", "gt_1": "> 1"},
        "market_wide": True,
    },
    {
        "id": "curve_10y3m",
        "title": "10Y–3M curve",
        "hint": "^TNX minus ^IRX. Below 0 is inverted",
        "labels": {"inverted": "Inverted", "normal": "Normal"},
        "market_wide": True,
    },
    {
        "id": "curve_change_20",
        "title": "10Y–3M curve change",
        "hint": "20-day change in ^TNX minus ^IRX. 0.10 is 10 bp",
        "labels": _CURVE_CHANGE_LABELS,
        "market_wide": True,
    },
    {
        "id": "rate_curve",
        "title": "Rate shock × curve",
        "hint": "Rate shock, then the 20-day curve change. A curve change of 0 counts as ≤ 0. A rate shock of 0 is left out",
        "labels": _RATE_CURVE_LABELS,
        "market_wide": True,
    },
    {
        "id": "dollar_shock",
        "title": "Dollar shock",
        "hint": "20-day UUP change divided by the 63-day standard deviation of daily changes, times √20",
        "labels": {"lt_neg_1": "< -1", "neg_1_to_1": "-1 to 1", "gt_1": "> 1"},
        "market_wide": True,
    },
    {
        "id": "dollar_rates",
        "title": "Dollar + rates",
        "hint": "Rate shock, then the dollar shock. A value of 0 counts as ≤ 0",
        "labels": _DOLLAR_RATES_LABELS,
        "market_wide": True,
    },
    {
        "id": "inflation",
        "title": "Inflation trend",
        "hint": "SMA(log(TIP / IEF), 20) minus SMA(log(TIP / IEF), 100). At or below 0 is the lower bucket",
        "labels": {"nonpos": "≤ 0", "pos": "> 0"},
        "market_wide": True,
    },
    {
        "id": "inflation_yield",
        "title": "Inflation and yield",
        "hint": "Rate shock, then the inflation trend. Above 0 is rising. Zero counts as falling",
        "labels": _INFLATION_YIELD_LABELS,
        "market_wide": True,
    },
)


def regime_indicator_id(dimension: str, key: str) -> str:
    return f"Regime_{dimension}_{key}"


class RegimeIndicatorSpec:
    def __init__(self, dimension: str, key: str, title: str, bucket_label: str, hint: str, market_wide: bool):
        self.dimension = dimension
        self.key = key
        self.market_wide = market_wide
        self.id = regime_indicator_id(dimension, key)
        self.label = f"{title} · {bucket_label}"
        self.description = (
            f"{hint}. Flag (1) on days in this bucket, −1 on other defined days, "
            "and 0 before the series exists."
        )


REGIME_INDICATOR_SPECS: tuple[RegimeIndicatorSpec, ...] = tuple(
    RegimeIndicatorSpec(
        dimension=item["id"],
        key=key,
        title=item["title"],
        bucket_label=label,
        hint=item["hint"],
        market_wide=item["market_wide"],
    )
    for item in _REGIME_DIMENSION_META
    for key, label in item["labels"].items()
)
REGIME_INDICATOR_IDS = frozenset(spec.id for spec in REGIME_INDICATOR_SPECS)
_REGIME_SPEC_BY_ID = {spec.id: spec for spec in REGIME_INDICATOR_SPECS}


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float | None:
    """Annualized Sharpe of simple returns. Same mean/std construction as the summary Sharpe."""
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return None
    std = float(clean.std(ddof=1))
    if not math.isfinite(std) or std == 0:
        return 0.0
    value = math.sqrt(periods_per_year) * float(clean.mean()) / std
    if not math.isfinite(value):
        return None
    return value


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float | None:
    """Annualized Sortino of simple returns. Downside matches the summary Sortino.

    Each return is capped at zero, then downside risk is the root-mean-square of that
    series, so up days contribute nothing to the denominator but stay in the count.
    """
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return None
    downside = np.minimum(clean.to_numpy(dtype=float), 0.0)
    downside_risk = float(np.sqrt(np.mean(downside**2)))
    if not math.isfinite(downside_risk) or downside_risk == 0:
        return 0.0
    value = math.sqrt(periods_per_year) * float(clean.mean()) / downside_risk
    if not math.isfinite(value):
        return None
    return value


def _blank_labels(index) -> pd.Series:
    return pd.Series([None] * len(index), index=index, dtype="object")


def _atr_labels(data: pd.DataFrame) -> pd.Series:
    if "ATR20" not in data.columns or "ATR50" not in data.columns:
        return _blank_labels(data.index)
    fast = pd.to_numeric(data["ATR20"], errors="coerce")
    slow = pd.to_numeric(data["ATR50"], errors="coerce")
    valid = fast.notna() & slow.notna()
    labels = _blank_labels(data.index)
    labels.loc[valid & (fast > slow)] = ATR_EXPANDING
    labels.loc[valid & (fast <= slow)] = ATR_CONTRACTING
    return labels


def _pair_labels(dates: pd.Series, calendar: pd.DataFrame | None, left_column: str, right_column: str, classify) -> pd.Series:
    needed = {left_column, right_column}
    if calendar is None or calendar.empty or not needed <= set(calendar.columns):
        return _blank_labels(dates.index)
    days = pd.DatetimeIndex([_normalize_timestamp(value) for value in dates])
    left = calendar[left_column].reindex(days)
    right = calendar[right_column].reindex(days)
    labels = [
        classify(left_value, right_value)
        for left_value, right_value in zip(left.to_numpy(), right.to_numpy())
    ]
    return pd.Series(labels, index=dates.index, dtype="object")


def _rate_curve_labels(dates: pd.Series, calendar: pd.DataFrame | None) -> pd.Series:
    return _pair_labels(dates, calendar, "rate_shock", "curve_change_20", classify_rate_curve)


def _spy_labels(dates: pd.Series, calendar: pd.DataFrame | None) -> pd.Series:
    if calendar is None or calendar.empty or not {"spy_sma50", "spy_sma200"} <= set(calendar.columns):
        return _blank_labels(dates.index)
    days = pd.DatetimeIndex([_normalize_timestamp(value) for value in dates])
    sma50 = calendar["spy_sma50"].reindex(days)
    sma200 = calendar["spy_sma200"].reindex(days)
    labels = [classify_spy(fast, slow) for fast, slow in zip(sma50.to_numpy(), sma200.to_numpy())]
    return pd.Series(labels, index=dates.index, dtype="object")


def _calendar_label_series(dates: pd.Series, calendar: pd.DataFrame | None, column: str, classify) -> pd.Series:
    if calendar is None or calendar.empty or column not in calendar.columns:
        return _blank_labels(dates.index)
    days = pd.DatetimeIndex([_normalize_timestamp(value) for value in dates])
    aligned = calendar[column].reindex(days)
    aligned.index = dates.index
    return aligned.map(classify)


def _bool_column(data: pd.DataFrame, name: str) -> np.ndarray:
    if name not in data.columns:
        return np.zeros(len(data), dtype=bool)
    return data[name].fillna(False).astype(bool).to_numpy()


def _strategy_bar_returns(equity: pd.Series) -> np.ndarray:
    """Simple return of RollingPnL. Flat bars are 0. This is the return the strategy booked."""
    values = pd.to_numeric(equity, errors="coerce").to_numpy(dtype=float)
    returns = np.zeros(len(values), dtype=float)
    if len(values) < 2:
        return returns
    prev = values[:-1]
    curr = values[1:]
    ok = np.isfinite(prev) & np.isfinite(curr) & (prev != 0)
    step = np.zeros(len(values) - 1, dtype=float)
    step[ok] = curr[ok] / prev[ok] - 1.0
    returns[1:] = step
    return returns


def _hold_entry_labels(
    labels: np.ndarray,
    long_in: np.ndarray,
    hold: np.ndarray,
    long_out: np.ndarray,
) -> np.ndarray:
    """Regime on the entry bar, carried across that trade's holding bars.

    The entry bar itself is not a holding bar. The exit bar is, and it keeps the
    entry regime. Later bars are flat until the next entry.
    """
    active = np.empty(len(labels), dtype=object)
    current = None
    for i in range(len(labels)):
        if long_in[i]:
            current = labels[i]
        active[i] = current if hold[i] else None
        if long_out[i]:
            current = None
    return active


_REGIME_SCORE_COMPONENT_FLOOR = 0.05


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _robustness_score(year_returns: dict) -> float:
    """Share of the average log year-return that remains after dropping the best year."""
    logs: list[float] = []
    for year_return in year_returns.values():
        try:
            returned = float(year_return)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(returned) or 1.0 + returned <= 0:
            continue
        logs.append(math.log(1.0 + returned))
    if len(logs) < 2:
        return 0.0
    all_years = sum(logs) / len(logs)
    if all_years <= 0:
        return 0.0
    remaining = list(logs)
    remaining.pop(remaining.index(max(remaining)))
    without_best = sum(remaining) / len(remaining)
    return _clamp(without_best / all_years, 0.0, 1.0)


def _empty_regime_score() -> dict[str, float | None]:
    return {
        "score": None,
        "sortino": None,
        "return": None,
        "drawdown": None,
        "robustness": None,
    }


def regime_score(
    sortino: float | None,
    avg_trade_return: float | None,
    max_drawdown: float | None,
    year_returns: dict,
) -> dict[str, float | None]:
    """0–100 blend of full-path Sortino, average trade return, max drawdown, and robustness.

    Each part is 0–1. Sortino of 3, a 3% average trade, and a 30% max drawdown
    each score 1. A 70% max drawdown scores 0. Robustness is the average log
    year-return without the best year, divided by the average that includes it.
    The blend floors every part at 0.05 so one zero does not wipe the score.
    The returned parts stay unfloored.
    """
    try:
        sortino_value = float(sortino) if sortino is not None else None
        trade_value = float(avg_trade_return) if avg_trade_return is not None else None
        drawdown_value = float(max_drawdown) if max_drawdown is not None else None
    except (TypeError, ValueError):
        return _empty_regime_score()
    if (
        sortino_value is None
        or trade_value is None
        or drawdown_value is None
        or not math.isfinite(sortino_value)
        or not math.isfinite(trade_value)
        or not math.isfinite(drawdown_value)
    ):
        return _empty_regime_score()
    sortino_score = _clamp(sortino_value / 3.0, 0.0, 1.0)
    return_score = _clamp(trade_value / 0.03, 0.0, 1.0)
    drawdown_score = _clamp((0.70 - drawdown_value) / (0.70 - 0.30), 0.0, 1.0)
    robustness = _robustness_score(year_returns)
    floor = _REGIME_SCORE_COMPONENT_FLOOR
    score = (
        100.0
        * max(sortino_score, floor) ** 0.30
        * max(return_score, floor) ** 0.25
        * max(drawdown_score, floor) ** 0.15
        * max(robustness, floor) ** 0.30
    )
    if not math.isfinite(score):
        return _empty_regime_score()
    return {
        "score": score,
        "sortino": sortino_score,
        "return": return_score,
        "drawdown": drawdown_score,
        "robustness": robustness,
    }


def _finite_metric(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _summary_metrics_of_trades(
    dates: pd.Series,
    returns: np.ndarray,
    mask: np.ndarray,
    periods_per_year: int,
) -> dict[str, float | None]:
    """Summary-card metrics for the equity curve of trades entered in one regime.

    Unselected bars stay flat, the same path as re-running the strategy with that
    regime as an entry filter. Sharpe, Sortino, CAGR, Calmar, and max drawdown then
    go through the summary formulas, including dropping the best return year.
    """
    empty = {
        "sharpe": None,
        "sortino": None,
        "max_drawdown": None,
        "cagr": None,
        "calmar": None,
        "score_sortino": None,
        "score_max_drawdown": None,
        "year_returns": {},
    }
    if not np.any(mask) or len(returns) == 0:
        return empty
    selected = np.where(mask, returns, 0.0)
    selected = np.where(np.isfinite(selected), selected, 0.0)
    if np.any(1.0 + selected <= 0):
        return empty
    equity = np.cumprod(1.0 + selected)
    peak = np.maximum.accumulate(equity)
    drawdown = np.zeros_like(equity)
    positive_peak = peak > 0
    drawdown[positive_peak] = (peak[positive_peak] - equity[positive_peak]) / peak[positive_peak]
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates).to_numpy(),
            "RollingPnL": equity,
            "Drawdown": drawdown,
            "LongTradeOut": False,
            "TradePnL": 0.0,
        }
    )
    metrics = compute_aggregate_metrics(frame, periods_per_year=periods_per_year)
    # Score inputs keep the best year. Robustness is what measures dependence on it.
    score_frame = frame.copy()
    return {
        "sharpe": _finite_metric(metrics["sharpe"]),
        "sortino": _finite_metric(metrics["sortino"]),
        "max_drawdown": _finite_metric(metrics["max_drawdown"]),
        "cagr": _finite_metric(metrics["cagr_decimal"]),
        "calmar": _finite_metric(metrics["calmar"]),
        "score_sortino": _finite_metric(ind.sortino_ratio(score_frame, periods_per_year=periods_per_year)),
        "score_max_drawdown": _finite_metric(frame["Drawdown"].max()),
        "year_returns": yearly_returns(frame),
    }


def _regime_label_sets(data: pd.DataFrame, calendar: pd.DataFrame | None) -> dict[str, pd.Series]:
    dates = data["Date"]
    return {
        "vix": _calendar_label_series(dates, calendar, "vix", classify_vix),
        "vxn": _calendar_label_series(dates, calendar, "vxn", classify_vxn),
        "spy": _spy_labels(dates, calendar),
        "market_breadth": _calendar_label_series(dates, calendar, "breadth", classify_breadth),
        "semis_breadth": _calendar_label_series(dates, calendar, "breadth_semis", classify_breadth),
        "equity_risk_breadth": _calendar_label_series(dates, calendar, "breadth_equity_risk", classify_breadth),
        "credit_risk_breadth": _calendar_label_series(dates, calendar, "breadth_credit_risk", classify_breadth),
        "credit_risk_on": _calendar_label_series(dates, calendar, "breadth_credit_risk_on", classify_breadth),
        "bond_duration": _calendar_label_series(dates, calendar, "breadth_bond_duration", classify_breadth),
        "copper_gold": _calendar_label_series(dates, calendar, "breadth_copper_gold", classify_breadth),
        "materials_breadth": _calendar_label_series(dates, calendar, "breadth_materials", classify_breadth),
        "sector_breadth_50": _calendar_label_series(dates, calendar, "sector_breadth_50", classify_sector_breadth),
        "sector_breadth_200": _calendar_label_series(dates, calendar, "sector_breadth_200", classify_sector_breadth),
        "sector_trend_50": _calendar_label_series(dates, calendar, "sector_trend_50", classify_sector_trend),
        "atr": _atr_labels(data),
        "rate_shock": _calendar_label_series(dates, calendar, "rate_shock", classify_rate_shock),
        "curve_10y3m": _calendar_label_series(dates, calendar, "curve_10y3m", classify_curve_10y3m),
        "curve_change_20": _calendar_label_series(dates, calendar, "curve_change_20", classify_curve_change),
        "rate_curve": _rate_curve_labels(dates, calendar),
        "dollar_shock": _calendar_label_series(dates, calendar, "dollar_shock", classify_rate_shock),
        "dollar_rates": _pair_labels(dates, calendar, "rate_shock", "dollar_shock", classify_dollar_rates),
        "inflation": _calendar_label_series(dates, calendar, "inflation_trend", classify_inflation),
        "inflation_yield": _pair_labels(dates, calendar, "rate_shock", "inflation_trend", classify_inflation_yield),
    }


def regime_coverage(data: pd.DataFrame, calendar: pd.DataFrame | None) -> dict[str, dict[str, str] | None]:
    """First and last session each regime can actually be labeled.

    Sessions before every input is listed (and its window is complete) are unlabeled,
    so this is the period the chart is calculated from.
    """
    labels = _regime_label_sets(data, calendar)
    dates = pd.to_datetime(data["Date"], errors="coerce")
    coverage: dict[str, dict[str, str] | None] = {}
    for dimension, series in labels.items():
        first = None
        last = None
        for i, value in enumerate(series.to_numpy()):
            if not isinstance(value, str) or not value:
                continue
            stamp = dates.iloc[i]
            if pd.isna(stamp):
                continue
            day = pd.Timestamp(stamp).strftime("%Y-%m-%d")
            if first is None:
                first = day
            last = day
        coverage[dimension] = None if first is None or last is None else {"start": first, "end": last}
    return coverage


_READING_COLUMN = {
    "vix": "vix",
    "vxn": "vxn",
    "market_breadth": "breadth",
    "semis_breadth": "breadth_semis",
    "equity_risk_breadth": "breadth_equity_risk",
    "credit_risk_breadth": "breadth_credit_risk",
    "credit_risk_on": "breadth_credit_risk_on",
    "bond_duration": "breadth_bond_duration",
    "copper_gold": "breadth_copper_gold",
    "materials_breadth": "breadth_materials",
    "sector_breadth_50": "sector_breadth_50",
    "sector_breadth_200": "sector_breadth_200",
    "sector_trend_50": "sector_trend_50",
    "rate_shock": "rate_shock",
    "curve_10y3m": "curve_10y3m",
    "curve_change_20": "curve_change_20",
    "dollar_shock": "dollar_shock",
    "inflation": "inflation_trend",
}


def _aligned_numeric(dates: pd.Series, calendar: pd.DataFrame | None, column: str) -> pd.Series:
    blank = pd.Series(np.nan, index=dates.index, dtype=float)
    if calendar is None or calendar.empty or column not in calendar.columns:
        return blank
    days = pd.DatetimeIndex([_normalize_timestamp(value) for value in dates])
    aligned = pd.to_numeric(calendar[column], errors="coerce").reindex(days)
    aligned.index = dates.index
    return aligned


def _atr_ratio(data: pd.DataFrame) -> pd.Series:
    blank = pd.Series(np.nan, index=data.index, dtype=float)
    if "ATR20" not in data.columns or "ATR50" not in data.columns:
        return blank
    fast = pd.to_numeric(data["ATR20"], errors="coerce")
    slow = pd.to_numeric(data["ATR50"], errors="coerce")
    ratio = fast / slow.where(slow != 0)
    return ratio.replace([np.inf, -np.inf], np.nan)


def _reading_at(
    data: pd.DataFrame,
    calendar: pd.DataFrame | None,
    dates: pd.Series,
    dimension: str,
    index: int | None,
) -> float | None:
    if index is None or dimension == "spy":
        return None
    if dimension == "atr":
        series = _atr_ratio(data)
    else:
        column = _READING_COLUMN.get(dimension)
        if column is None:
            return None
        series = _aligned_numeric(dates, calendar, column)
    number = series.iloc[index]
    if number is None or not np.isfinite(number):
        return None
    return float(number)


def current_market_regimes(data: pd.DataFrame, calendar: pd.DataFrame | None) -> dict[str, Any]:
    """Latest labeled session for each regime, plus the raw reading on that session."""
    labels = _regime_label_sets(data, calendar)
    dates = data["Date"]
    regimes: dict[str, str | None] = {}
    readings: dict[str, float | None] = {}
    latest = None
    for dimension, series in labels.items():
        key = None
        found_at = None
        found_i = None
        values = series.to_numpy()
        stamped = pd.to_datetime(dates, errors="coerce")
        for i in range(len(values) - 1, -1, -1):
            value = values[i]
            if isinstance(value, str) and value:
                key = value
                found_i = i
                found_at = stamped.iloc[i]
                break
        regimes[dimension] = key
        readings[dimension] = _reading_at(data, calendar, dates, dimension, found_i)
        if found_at is not None and not pd.isna(found_at) and (latest is None or found_at > latest):
            latest = found_at
    as_of = None if latest is None else pd.Timestamp(latest).strftime("%Y-%m-%d")
    return {"as_of": as_of, "regimes": regimes, "readings": readings}


def compute_market_regime_sharpe(
    data: pd.DataFrame,
    calendar: pd.DataFrame | None,
    *,
    periods_per_year: int = 252,
) -> dict[str, list[dict[str, Any]]]:
    """Entry-regime Sharpe, Sortino, max drawdown, CAGR, and Calmar.

    A trade belongs to the regime on its entry bar for the whole hold. The metrics are
    the summary-card figures for an equity curve that compounds only those holding bars
    and stays flat otherwise, including the best-return-year exclusion.
    `days` is still the number of bars whose own label is the regime, used to tell a
    missing calendar from an empty bucket.
    """
    equity = pd.to_numeric(data["RollingPnL"], errors="coerce")
    daily = equity.pct_change().replace([np.inf, -np.inf], np.nan)
    bar_returns = _strategy_bar_returns(equity)
    long_in = _bool_column(data, "LongTradeIn")
    hold = _bool_column(data, "HoldLong")
    long_out = _bool_column(data, "LongTradeOut")
    trade_pnl = (
        pd.to_numeric(data["TradePnL"], errors="coerce").to_numpy(dtype=float)
        if "TradePnL" in data.columns
        else np.full(len(data), np.nan)
    )
    label_sets = _regime_label_sets(data, calendar)

    payload: dict[str, list[dict[str, Any]]] = {}
    for dimension, keys in _SHARPE_BUCKETS.items():
        labels = label_sets[dimension]
        held_as = _hold_entry_labels(labels.to_numpy(dtype=object), long_in, hold, long_out)
        rows: list[dict[str, Any]] = []
        for key in keys:
            in_bucket = labels == key
            finite_days = daily[in_bucket]
            finite_days = finite_days[np.isfinite(finite_days.to_numpy(dtype=float))]
            trade_bars = held_as == key
            metrics = _summary_metrics_of_trades(data["Date"], bar_returns, trade_bars, periods_per_year)
            closed = trade_bars & long_out & np.isfinite(trade_pnl)
            avg_trade = float(np.mean(trade_pnl[closed])) if np.any(closed) else None
            scored = regime_score(
                metrics["score_sortino"],
                avg_trade,
                metrics["score_max_drawdown"],
                metrics["year_returns"],
            )
            rows.append(
                {
                    "key": key,
                    "sharpe": metrics["sharpe"],
                    "sortino": metrics["sortino"],
                    "days": int(len(finite_days)),
                    "max_drawdown": metrics["max_drawdown"],
                    "cagr": metrics["cagr"],
                    "calmar": metrics["calmar"],
                    "regime_score": scored["score"],
                    "regime_score_sortino": scored["sortino"],
                    "regime_score_return": scored["return"],
                    "regime_score_drawdown": scored["drawdown"],
                    "regime_score_robustness": scored["robustness"],
                }
            )
        payload[dimension] = rows
    return payload


_BOOK_METRIC_FIELDS = (
    "regime_score",
    "sortino",
    "avg_trade_return",
    "max_drawdown",
    "robustness",
    "exposure",
    "cagr",
)


def _empty_book_metrics() -> dict[str, float | int | None]:
    return {
        "regime_score": None,
        "sortino": None,
        "avg_trade_return": None,
        "max_drawdown": None,
        "robustness": None,
        "exposure": None,
        "cagr": None,
        "trades": 0,
    }


def _exposure_per_day(trade_pnl: np.ndarray, days: np.ndarray, closed: np.ndarray) -> float | None:
    """Average closed-trade return per day held. Trades with no positive day count are left out."""
    if closed.size == 0 or not np.any(closed):
        return None
    pnl = trade_pnl[closed]
    held = days[closed]
    ok = np.isfinite(pnl) & np.isfinite(held) & (held > 0)
    if not np.any(ok):
        return None
    total_days = float(np.sum(held[ok]))
    if total_days <= 0:
        return None
    return float(np.sum(pnl[ok]) / total_days)


def _book_metrics_for_mask(
    dates: pd.Series,
    bar_returns: np.ndarray,
    equity_mask: np.ndarray,
    trade_pnl: np.ndarray,
    days_in_trade: np.ndarray,
    closed: np.ndarray,
    periods_per_year: int,
) -> dict[str, float | int | None]:
    """Regime-score inputs for one book.

    ``equity_mask`` selects the holding bars that compound. ``closed`` selects
    the exit bars whose trade return and days held belong to this book.
    Sortino and max drawdown are the score inputs (best year kept), not the
    summary-card figures that drop it.
    """
    summary = _summary_metrics_of_trades(dates, bar_returns, equity_mask, periods_per_year)
    finite_closed = closed & np.isfinite(trade_pnl)
    count = int(np.count_nonzero(finite_closed))
    avg_trade = float(np.mean(trade_pnl[finite_closed])) if count else None
    scored = regime_score(
        summary["score_sortino"],
        avg_trade,
        summary["score_max_drawdown"],
        summary["year_returns"],
    )
    return {
        "regime_score": scored["score"],
        "sortino": summary["score_sortino"],
        "avg_trade_return": avg_trade,
        "max_drawdown": summary["score_max_drawdown"],
        "robustness": scored["robustness"],
        "exposure": _exposure_per_day(trade_pnl, days_in_trade, finite_closed),
        "cagr": summary["cagr"],
        "trades": count,
    }


def score_regime_book(
    data: pd.DataFrame,
    calendar: pd.DataFrame | None,
    *,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """BASE and every regime bucket, using the same score as the overview charts.

    BASE compounds the whole equity path. A bucket compounds only bars whose
    trade was entered in that regime, matching ``compute_market_regime_sharpe``.
    """
    blank = {
        "base": _empty_book_metrics(),
        "regimes": {
            dimension: {key: _empty_book_metrics() for key in keys}
            for dimension, keys in _SHARPE_BUCKETS.items()
        },
    }
    if data is None or data.empty or "RollingPnL" not in data.columns or "Date" not in data.columns:
        return blank

    equity = pd.to_numeric(data["RollingPnL"], errors="coerce")
    bar_returns = _strategy_bar_returns(equity)
    long_in = _bool_column(data, "LongTradeIn")
    hold = _bool_column(data, "HoldLong")
    long_out = _bool_column(data, "LongTradeOut")
    trade_pnl = (
        pd.to_numeric(data["TradePnL"], errors="coerce").to_numpy(dtype=float)
        if "TradePnL" in data.columns
        else np.full(len(data), np.nan)
    )
    if "DaysInTrade" in data.columns:
        days_in_trade = pd.to_numeric(data["DaysInTrade"], errors="coerce").to_numpy(dtype=float)
    else:
        days_in_trade = np.full(len(data), np.nan)
    dates = data["Date"]
    all_bars = np.ones(len(data), dtype=bool)
    all_closed = long_out & np.isfinite(trade_pnl)
    base = _book_metrics_for_mask(
        dates,
        bar_returns,
        all_bars,
        trade_pnl,
        days_in_trade,
        all_closed,
        periods_per_year,
    )

    label_sets = _regime_label_sets(data, calendar)
    regimes: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for dimension, keys in _SHARPE_BUCKETS.items():
        labels = label_sets[dimension]
        held_as = _hold_entry_labels(labels.to_numpy(dtype=object), long_in, hold, long_out)
        buckets: dict[str, dict[str, float | int | None]] = {}
        for key in keys:
            trade_bars = held_as == key
            closed = trade_bars & long_out & np.isfinite(trade_pnl)
            buckets[key] = _book_metrics_for_mask(
                dates,
                bar_returns,
                trade_bars,
                trade_pnl,
                days_in_trade,
                closed,
                periods_per_year,
            )
        regimes[dimension] = buckets
    return {"base": base, "regimes": regimes}


def regime_metric_delta(
    full: dict[str, float | int | None],
    reduced: dict[str, float | int | None],
) -> dict[str, float | None]:
    """Full-book metric minus the book with one strategy removed."""
    delta: dict[str, float | None] = {}
    for key in _BOOK_METRIC_FIELDS:
        left = full.get(key)
        right = reduced.get(key)
        if left is None or right is None:
            delta[key] = None
            continue
        try:
            number = float(left) - float(right)
        except (TypeError, ValueError):
            delta[key] = None
            continue
        delta[key] = number if math.isfinite(number) else None
    return delta


def build_regime_comparison(full_book: dict[str, Any], reduced_book: dict[str, Any]) -> dict[str, Any]:
    """Per-cell contribution: delta, the full book, and the book without the strategy."""

    def cell(full_metrics: dict, reduced_metrics: dict) -> dict[str, Any]:
        return {
            "delta": regime_metric_delta(full_metrics, reduced_metrics),
            "with": full_metrics,
            "without": reduced_metrics,
        }

    regimes: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension, buckets in full_book["regimes"].items():
        reduced_buckets = reduced_book["regimes"][dimension]
        regimes[dimension] = {
            key: cell(metrics, reduced_buckets[key]) for key, metrics in buckets.items()
        }
    return {"base": cell(full_book["base"], reduced_book["base"]), "regimes": regimes}


def _cache_day():
    return datetime.now(EASTERN).date()


def _flag_series(labels: pd.Series, key: str) -> pd.Series:
    """1 in the bucket, -1 in another defined bucket, 0 when the day is unlabeled."""
    flags = np.zeros(len(labels), dtype=int)
    for index, value in enumerate(labels.to_numpy(dtype=object)):
        if isinstance(value, str) and value:
            flags[index] = 1 if value == key else -1
    return pd.Series(flags, index=labels.index)


def _years_covering_dates(dates: pd.Series) -> int:
    stamps = pd.to_datetime(dates, errors="coerce").dropna()
    if stamps.empty:
        return 2
    span_years = (pd.Timestamp(stamps.max()) - pd.Timestamp(stamps.min())).days / 365.25
    # SMA(200) and listing warmup need history before the first bar.
    return max(2, int(math.ceil(span_years)) + 2)


def _referenced_regime_ids(conditions: list[dict]) -> list[str]:
    found: list[str] = []
    for condition in conditions:
        for field in ("left", "right"):
            value = condition.get(field) or ""
            if value in REGIME_INDICATOR_IDS and value not in found:
                found.append(value)
    return found


def ensure_regime_indicator_columns(data: pd.DataFrame, conditions: list[dict]) -> None:
    """Add regime flag columns referenced by conditions.

    Unreferenced flags are left off the frame, so a built-in signal backtest is unchanged.
    A missing calendar raises when a non-ATR regime is required.
    """
    needed = [
        _REGIME_SPEC_BY_ID[indicator_id]
        for indicator_id in _referenced_regime_ids(conditions)
        if indicator_id not in data.columns
    ]
    if not needed:
        return
    if "Date" not in data.columns:
        raise ValueError("Regime conditions need a Date column")
    calendar = None
    if any(spec.dimension != "atr" for spec in needed):
        calendar = load_regime_calendar(_years_covering_dates(data["Date"]))
        if calendar is None:
            raise ValueError(
                "Market regime calendar is unavailable, so regime conditions cannot be evaluated"
            )
    labels = _regime_label_sets(data, calendar)
    for spec in needed:
        data[spec.id] = _flag_series(labels[spec.dimension], spec.key)


def _fetch_regime_calendar(years: int) -> pd.DataFrame:
    import getdata as dt

    bulk = dt.get_bulk_data(list(REGIME_SYMBOLS), years=years)
    return build_regime_calendar(
        dt._bulk_close(bulk, "^VIX"),
        dt._bulk_close(bulk, "^VXN"),
        dt._bulk_close(bulk, "SPY"),
        dt._bulk_close(bulk, "RSP"),
        [dt._bulk_close(bulk, symbol) for symbol in SECTOR_SYMBOLS],
        dt._bulk_close(bulk, "SMH"),
        dt._bulk_close(bulk, "XLY"),
        dt._bulk_close(bulk, "XLP"),
        dt._bulk_close(bulk, "HYG"),
        dt._bulk_close(bulk, "LQD"),
        dt._bulk_close(bulk, "TLT"),
        dt._bulk_close(bulk, "SHY"),
        dt._bulk_close(bulk, "HG=F"),
        dt._bulk_close(bulk, "GC=F"),
        dt._bulk_close(bulk, "XLB"),
        dt._bulk_close(bulk, "^TNX"),
        dt._bulk_close(bulk, "^IRX"),
        dt._bulk_close(bulk, "UUP"),
        dt._bulk_close(bulk, "TIP"),
        dt._bulk_close(bulk, "IEF"),
    )


def load_regime_calendar(years: int) -> pd.DataFrame | None:
    """Download and cache the regime calendar for this Eastern calendar day.

    A failed download returns None and is not cached, so a later backtest can retry.
    Callers should still return the backtest when this is None.
    """
    key = (int(years), _cache_day())
    cached = _CALENDAR_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        frame = _fetch_regime_calendar(int(years))
    except Exception:
        logger.warning("Market regime calendar unavailable", exc_info=True)
        return None
    _CALENDAR_CACHE[key] = frame
    return frame
