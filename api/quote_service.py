from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import ta

from api.serializers import _clean_number
from indicators import internal_bar_range

EASTERN = ZoneInfo("America/New_York")
CASH_CLOSE = time(16, 0)
LOOKBACK_CALENDAR_DAYS = 5

QUOTE_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("SPY", "SPY"),
    ("QQQ", "QQQ"),
    ("SOXX", "SOXX"),
    ("VIX", "^VIX"),
)

YF_SYMBOLS = [yf_symbol for _, yf_symbol in QUOTE_SYMBOLS]


def eastern_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(EASTERN)
    if now.tzinfo is None:
        return now.replace(tzinfo=EASTERN)
    return now.astimezone(EASTERN)


def _index_date(value) -> date:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(EASTERN)
    return ts.tz_localize(None).normalize().date()


def dates_with_valid_close(close: pd.Series) -> set[date]:
    present: set[date] = set()
    if close is None or close.empty:
        return present
    for index_value, value in close.items():
        if pd.isna(value):
            continue
        present.add(_index_date(index_value))
    return present


def missing_weekdays(
    close: pd.Series,
    *,
    now: datetime | None = None,
    lookback_days: int = LOOKBACK_CALENDAR_DAYS,
) -> list[str]:
    """Weekdays in the last `lookback_days` calendar days with no Close.

    Saturdays and Sundays are never flagged. Today is skipped until the
    regular cash session has closed (16:00 America/New_York). Exchange
    holidays are left flagged so the caller can judge them.
    """
    moment = eastern_now(now)
    today = moment.date()
    include_today = moment.time() >= CASH_CLOSE
    present = dates_with_valid_close(close)

    missing: list[str] = []
    for offset in range(lookback_days):
        day = today - timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        if day == today and not include_today:
            continue
        if day not in present:
            missing.append(day.isoformat())
    missing.sort()
    return missing


def metrics_from_ohlcv(frame: pd.DataFrame) -> dict:
    empty = {
        "as_of": None,
        "close": None,
        "pct_change": None,
        "ibr": None,
        "rsi2": None,
        "rsi5": None,
        "stoch": None,
    }
    if frame is None or frame.empty or "Close" not in frame.columns:
        return empty

    valid = frame.copy()
    valid["Close"] = pd.to_numeric(valid["Close"], errors="coerce")
    valid = valid[valid["Close"].notna()]
    if valid.empty:
        return empty

    last = valid.iloc[-1]
    close = float(last["Close"])
    as_of = _index_date(valid.index[-1]).isoformat()

    pct_change = None
    if len(valid) >= 2:
        previous = float(valid["Close"].iloc[-2])
        if previous != 0:
            pct_change = close / previous - 1.0

    ibr = None
    if {"High", "Low"}.issubset(valid.columns):
        high = pd.to_numeric(valid["High"], errors="coerce")
        low = pd.to_numeric(valid["Low"], errors="coerce")
        ibr_values = internal_bar_range(high, low, valid["Close"], period=1)
        ibr_last = ibr_values[-1] if len(ibr_values) else None
        if ibr_last is not None and pd.notna(ibr_last):
            ibr = float(ibr_last)

    rsi2 = None
    rsi5 = None
    stoch = None
    close_series = valid["Close"]
    if len(close_series) >= 3:
        rsi2_last = ta.momentum.RSIIndicator(close_series, window=2).rsi().iloc[-1]
        if pd.notna(rsi2_last):
            rsi2 = float(rsi2_last)
    if len(close_series) >= 6:
        rsi5_last = ta.momentum.RSIIndicator(close_series, window=5).rsi().iloc[-1]
        if pd.notna(rsi5_last):
            rsi5 = float(rsi5_last)
    if {"High", "Low"}.issubset(valid.columns) and len(valid) >= 14:
        high = pd.to_numeric(valid["High"], errors="coerce")
        low = pd.to_numeric(valid["Low"], errors="coerce")
        stoch_last = ta.momentum.stoch(high, low, close_series, window=14, smooth_window=3).iloc[-1]
        if pd.notna(stoch_last):
            stoch = float(stoch_last)

    return {
        "as_of": as_of,
        "close": _clean_number(close),
        "pct_change": _clean_number(pct_change),
        "ibr": _clean_number(ibr),
        "rsi2": _clean_number(rsi2),
        "rsi5": _clean_number(rsi5),
        "stoch": _clean_number(stoch),
    }


def ohlcv_from_bulk(full_data: pd.DataFrame, yf_symbol: str) -> pd.DataFrame:
    columns = ["Open", "High", "Low", "Close", "Volume"]
    empty = pd.DataFrame(columns=columns)
    if full_data is None or full_data.empty:
        return empty
    if not isinstance(full_data.columns, pd.MultiIndex):
        return empty
    tickers = set(full_data.columns.get_level_values(1))
    if yf_symbol not in tickers:
        return empty
    frame = full_data.xs(yf_symbol, axis=1, level=1, drop_level=True).copy()
    keep = [column for column in columns if column in frame.columns]
    frame = frame[keep]
    index = pd.to_datetime(frame.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(EASTERN).tz_localize(None)
    frame.index = index.normalize()
    frame.index.name = "Date"
    return frame


def quote_for_symbol(
    full_data: pd.DataFrame,
    display_symbol: str,
    yf_symbol: str,
    *,
    now: datetime | None = None,
) -> dict:
    frame = ohlcv_from_bulk(full_data, yf_symbol)
    metrics = metrics_from_ohlcv(frame)
    close = frame["Close"] if "Close" in frame.columns else pd.Series(dtype=float)
    return {
        "symbol": display_symbol,
        **metrics,
        "missing_days": missing_weekdays(close, now=now),
    }


def build_quote_snapshot(
    *,
    full_data: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> dict:
    if full_data is None:
        import getdata as dt

        full_data = dt.get_bulk_data(YF_SYMBOLS, years=1)

    quotes = [
        quote_for_symbol(full_data, display, yf_symbol, now=now)
        for display, yf_symbol in QUOTE_SYMBOLS
    ]
    as_ofs = [row["as_of"] for row in quotes if row.get("as_of")]
    return {
        "as_of": max(as_ofs) if as_ofs else None,
        "quotes": quotes,
    }
