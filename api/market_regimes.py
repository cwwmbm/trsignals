"""Entry-date market regime labels for backtest analysis.

Strategy indicators are unchanged. Old breadth is RSI(14) of RSP/SPY.
New breadth is RSI(14) of log(RSP/SPY). Semis breadth is RSI(14) of log(SMH/SPY).
Equity risk breadth is RSI(14) of log(XLY/XLP). Credit quality breadth is RSI(14) of log(HYG/LQD).
Credit risk on is RSI(14) of log(HYG/TLT). Bond duration is RSI(14) of log(TLT/SHY).
Copper/gold is RSI(14) of log(HG=F/GC=F). Sensitive materials breadth is RSI(14) of log(XLB/SPY).
Sector breadth is the share of nine
sector ETFs closing strictly above their own SMA(50) or SMA(200). SPY bull/bear
uses SMA(50) vs SMA(200) and is separate from the existing SPYBull flag.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import ta

from stats import calmar_ratio

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
BREADTH_RSI_WINDOW = 14
SPY_SMA_FAST = 50
SPY_SMA_SLOW = 200
SECTOR_SYMBOLS = ("XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLK", "XLU")
CREDIT_SYMBOLS = ("HYG", "LQD")
RATIO_SYMBOLS = ("TLT", "SHY", "HG=F", "GC=F")
REGIME_SYMBOLS = ("^VIX", "^VXN", "SPY", "RSP", "SMH", *SECTOR_SYMBOLS, *CREDIT_SYMBOLS, *RATIO_SYMBOLS)

VIX_KEYS = ("le_15", "15_20", "20_30", "gt_30")
VXN_KEYS = VIX_KEYS
BREADTH_KEYS = ("lt_40", "40_50", "50_60", "gt_60")
SECTOR_BREADTH_KEYS = ("le_25", "25_50", "50_75", "gt_75")
ATR_EXPANDING = "expanding"
ATR_CONTRACTING = "contracting"
SPY_BULL = "bull"
SPY_BEAR = "bear"

MARKET_REGIME_KEYS = (
    "vix_regime",
    "vxn_regime",
    "spy_regime",
    "breadth_new_regime",
    "breadth_old_regime",
    "semis_breadth_regime",
    "equity_risk_breadth_regime",
    "credit_risk_breadth_regime",
    "credit_risk_on_regime",
    "bond_duration_regime",
    "copper_gold_regime",
    "materials_breadth_regime",
    "sector_breadth_50_regime",
    "sector_breadth_200_regime",
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
) -> pd.DataFrame:
    """Daily VIX, VXN, SPY SMA(50/200), and both breadth RSIs, indexed by session date."""
    # Drop sessions with no SPY close (Yahoo inserts holiday rows as NaN in a
    # multi-ticker download). A NaN inside the rolling window would blank SMA
    # for the next 200 sessions.
    spy_close = _as_day_series(spy).dropna()
    vix_close = _as_day_series(vix).reindex(spy_close.index)
    vxn_close = _as_day_series(vxn).reindex(spy_close.index)
    rsp_close = _as_day_series(rsp).reindex(spy_close.index)
    ratio = rsp_close / spy_close
    log_ratio = np.log(ratio.where(ratio > 0))
    frame = pd.DataFrame(
        {
            "vix": vix_close,
            "vxn": vxn_close,
            "spy_sma50": spy_close.rolling(SPY_SMA_FAST).mean(),
            "spy_sma200": spy_close.rolling(SPY_SMA_SLOW).mean(),
            "breadth_old": _rsi(ratio),
            "breadth_new": _rsi(pd.Series(log_ratio, index=ratio.index)),
            "breadth_semis": _log_ratio_rsi(smh, spy_close, spy_close.index),
            "breadth_equity_risk": _log_ratio_rsi(xly, xlp, spy_close.index),
            "breadth_credit_risk": _log_ratio_rsi(hyg, lqd, spy_close.index),
            "breadth_credit_risk_on": _log_ratio_rsi(hyg, tlt, spy_close.index),
            "breadth_bond_duration": _log_ratio_rsi(tlt, shy, spy_close.index),
            "breadth_copper_gold": _log_ratio_rsi(copper, gold, spy_close.index),
            "breadth_materials": _log_ratio_rsi(xlb, spy_close, spy_close.index),
            "sector_breadth_50": _sector_breadth(sector_closes or [], spy_close.index, SPY_SMA_FAST),
            "sector_breadth_200": _sector_breadth(sector_closes or [], spy_close.index, SPY_SMA_SLOW),
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
        "breadth_new_regime": classify_breadth(row.get("breadth_new")),
        "breadth_old_regime": classify_breadth(row.get("breadth_old")),
        "semis_breadth_regime": classify_breadth(row.get("breadth_semis")),
        "equity_risk_breadth_regime": classify_breadth(row.get("breadth_equity_risk")),
        "credit_risk_breadth_regime": classify_breadth(row.get("breadth_credit_risk")),
        "credit_risk_on_regime": classify_breadth(row.get("breadth_credit_risk_on")),
        "bond_duration_regime": classify_breadth(row.get("breadth_bond_duration")),
        "copper_gold_regime": classify_breadth(row.get("breadth_copper_gold")),
        "materials_breadth_regime": classify_breadth(row.get("breadth_materials")),
        "sector_breadth_50_regime": classify_sector_breadth(row.get("sector_breadth_50")),
        "sector_breadth_200_regime": classify_sector_breadth(row.get("sector_breadth_200")),
    }


_SHARPE_BUCKETS: dict[str, tuple[str, ...]] = {
    "vix": VIX_KEYS,
    "vxn": VXN_KEYS,
    "atr": (ATR_EXPANDING, ATR_CONTRACTING),
    "spy": (SPY_BULL, SPY_BEAR),
    "breadth_new": BREADTH_KEYS,
    "breadth_old": BREADTH_KEYS,
    "semis_breadth": BREADTH_KEYS,
    "equity_risk_breadth": BREADTH_KEYS,
    "credit_risk_breadth": BREADTH_KEYS,
    "credit_risk_on": BREADTH_KEYS,
    "bond_duration": BREADTH_KEYS,
    "copper_gold": BREADTH_KEYS,
    "materials_breadth": BREADTH_KEYS,
    "sector_breadth_50": SECTOR_BREADTH_KEYS,
    "sector_breadth_200": SECTOR_BREADTH_KEYS,
}


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


def _max_drawdown_of_trades(returns: np.ndarray, mask: np.ndarray) -> float | None:
    """Peak-to-trough drawdown of an equity curve that compounds only the selected bars.

    Unselected bars stay flat, so a loss booked in another regime cannot deepen this one.
    """
    if not np.any(mask):
        return None
    selected = np.where(mask, returns, 0.0)
    selected = np.where(np.isfinite(selected), selected, 0.0)
    curve = np.cumprod(1.0 + selected)
    peak = np.maximum.accumulate(curve)
    valid = np.isfinite(peak) & (peak > 0) & np.isfinite(curve)
    if not np.any(valid):
        return None
    drawdown = np.zeros_like(curve)
    drawdown[valid] = (peak[valid] - curve[valid]) / peak[valid]
    value = float(np.nanmax(drawdown))
    if not math.isfinite(value):
        return None
    return value


def _cagr_of_trades(returns: np.ndarray, mask: np.ndarray, periods_per_year: int) -> float | None:
    """CAGR of an equity curve that compounds only the selected holding bars.

    Unselected bars stay flat. The rate is annualized over the full sample, so cash
    time is in the calendar and only trades entered in the regime move the equity.
    """
    if not np.any(mask) or periods_per_year <= 0 or len(returns) == 0:
        return None
    selected = np.where(mask, returns, 0.0)
    selected = np.where(np.isfinite(selected), selected, 0.0)
    steps = 1.0 + selected
    if np.any(steps <= 0):
        return None
    ending = float(np.prod(steps))
    years = len(returns) / periods_per_year
    if not math.isfinite(ending) or ending <= 0 or years <= 0:
        return None
    value = ending ** (1.0 / years) - 1.0
    if not math.isfinite(value):
        return None
    return value


def _calmar_of_trades(returns: np.ndarray, mask: np.ndarray, periods_per_year: int) -> float | None:
    """Calmar of the entry-regime path: its CAGR divided by its max drawdown."""
    cagr = _cagr_of_trades(returns, mask, periods_per_year)
    drawdown = _max_drawdown_of_trades(returns, mask)
    if cagr is None or drawdown is None:
        return None
    return calmar_ratio(cagr * 100.0, drawdown)


def _sharpe_of_trades(returns: np.ndarray, mask: np.ndarray, periods_per_year: int) -> float | None:
    """Sharpe of the daily returns booked while holding trades selected by mask.

    Cash days outside those trades are left out, so the ratio describes that trade
    path rather than how often the regime occurred. Annualization matches the summary Sharpe.
    """
    if not np.any(mask):
        return None
    sample = returns[mask]
    sample = sample[np.isfinite(sample)]
    return sharpe_ratio(pd.Series(sample), periods_per_year)


def _sortino_of_trades(returns: np.ndarray, mask: np.ndarray, periods_per_year: int) -> float | None:
    """Sortino of the daily returns booked while holding trades selected by mask.

    Cash days outside those trades are left out, the same population as regime Sharpe.
    """
    if not np.any(mask):
        return None
    sample = returns[mask]
    sample = sample[np.isfinite(sample)]
    return sortino_ratio(pd.Series(sample), periods_per_year)


def _regime_label_sets(data: pd.DataFrame, calendar: pd.DataFrame | None) -> dict[str, pd.Series]:
    dates = data["Date"]
    return {
        "vix": _calendar_label_series(dates, calendar, "vix", classify_vix),
        "vxn": _calendar_label_series(dates, calendar, "vxn", classify_vxn),
        "spy": _spy_labels(dates, calendar),
        "breadth_new": _calendar_label_series(dates, calendar, "breadth_new", classify_breadth),
        "breadth_old": _calendar_label_series(dates, calendar, "breadth_old", classify_breadth),
        "semis_breadth": _calendar_label_series(dates, calendar, "breadth_semis", classify_breadth),
        "equity_risk_breadth": _calendar_label_series(dates, calendar, "breadth_equity_risk", classify_breadth),
        "credit_risk_breadth": _calendar_label_series(dates, calendar, "breadth_credit_risk", classify_breadth),
        "credit_risk_on": _calendar_label_series(dates, calendar, "breadth_credit_risk_on", classify_breadth),
        "bond_duration": _calendar_label_series(dates, calendar, "breadth_bond_duration", classify_breadth),
        "copper_gold": _calendar_label_series(dates, calendar, "breadth_copper_gold", classify_breadth),
        "materials_breadth": _calendar_label_series(dates, calendar, "breadth_materials", classify_breadth),
        "sector_breadth_50": _calendar_label_series(dates, calendar, "sector_breadth_50", classify_sector_breadth),
        "sector_breadth_200": _calendar_label_series(dates, calendar, "sector_breadth_200", classify_sector_breadth),
        "atr": _atr_labels(data),
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
    "breadth_new": "breadth_new",
    "breadth_old": "breadth_old",
    "semis_breadth": "breadth_semis",
    "equity_risk_breadth": "breadth_equity_risk",
    "credit_risk_breadth": "breadth_credit_risk",
    "credit_risk_on": "breadth_credit_risk_on",
    "bond_duration": "breadth_bond_duration",
    "copper_gold": "breadth_copper_gold",
    "materials_breadth": "breadth_materials",
    "sector_breadth_50": "sector_breadth_50",
    "sector_breadth_200": "sector_breadth_200",
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

    A trade belongs to the regime on its entry bar for the whole hold. Sharpe and Sortino
    use the daily strategy returns on those holding bars. Max drawdown is the
    peak-to-trough of an equity curve that compounds only those bars. CAGR is that
    curve's growth, annualized over the full sample. Calmar is that CAGR divided by
    that max drawdown.
    `days` is still the number of bars whose own label is the regime, used to tell a
    missing calendar from an empty bucket.
    """
    equity = pd.to_numeric(data["RollingPnL"], errors="coerce")
    daily = equity.pct_change().replace([np.inf, -np.inf], np.nan)
    bar_returns = _strategy_bar_returns(equity)
    long_in = _bool_column(data, "LongTradeIn")
    hold = _bool_column(data, "HoldLong")
    long_out = _bool_column(data, "LongTradeOut")
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
            rows.append(
                {
                    "key": key,
                    "sharpe": _sharpe_of_trades(bar_returns, trade_bars, periods_per_year),
                    "sortino": _sortino_of_trades(bar_returns, trade_bars, periods_per_year),
                    "days": int(len(finite_days)),
                    "max_drawdown": _max_drawdown_of_trades(bar_returns, trade_bars),
                    "cagr": _cagr_of_trades(bar_returns, trade_bars, periods_per_year),
                    "calmar": _calmar_of_trades(bar_returns, trade_bars, periods_per_year),
                }
            )
        payload[dimension] = rows
    return payload


def _cache_day():
    return datetime.now(EASTERN).date()


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
