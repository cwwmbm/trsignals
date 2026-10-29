"""Earnings dates for stocks, and the two indicators built from them.

``DaysToEarnings`` and ``DaysSinceEarnings`` count **trading bars** to the next, and since the last,
*reaction day*: the first session whose price reflects a report. A company that reports before the
open (a time before noon, Eastern) is reacted to the same day; one that reports after the close
reacts the next session. So on the bar before the reaction day ``DaysToEarnings`` is 1: a position
opened at that close is exposed to the report, and one opened a bar earlier is exposed to it for
two bars.

Where nothing is known (an ETF or index, a symbol Yahoo has no dates for, a stretch before the first
or after the last date it lists) both are ``NO_EARNINGS`` (999). That keeps a rule such as
``DaysToEarnings > 3`` true on an ETF, instead of silently blocking every trade, and keeps an exit
rule such as ``DaysToEarnings <= 1`` from ever firing there.

The dates come from Yahoo through yfinance, which scrapes them and returns at most 100 rows a call.
They are cached per symbol on disk and refreshed weekly (a symbol Yahoo has nothing for is rechecked
monthly). A failed download never overwrites a good cache and is never cached as "no earnings".
History reaches back to the late 1990s or early 2000s for large companies. A quarter is occasionally missing, and the
last row is usually a projection until the company confirms its date; ``coverage`` reports both.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "market_cache" / "earnings"
EASTERN = ZoneInfo("America/New_York")

NO_EARNINGS = 999
REFRESH_DAYS = 7
EMPTY_REFRESH_DAYS = 30
PAGE_SIZE = 100  # yfinance refuses more per call
MAX_PAGES = 5
# A report before this hour (Eastern) is before the open; anything later is treated as after the close.
BEFORE_OPEN_HOUR = 12
# More than this between consecutive known reports means a quarter is probably missing.
GAP_DAYS = 130

BEFORE_OPEN = "before_open"
AFTER_CLOSE = "after_close"

_lock = threading.Lock()
_memory: dict[str, tuple[datetime, pd.DataFrame | None]] = {}


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _path(symbol: str) -> Path:
    return CACHE_DIR / f"{re.sub(r'[^A-Za-z0-9._=-]+', '_', _normalize_symbol(symbol))}.json"


def _now() -> datetime:
    return datetime.now(EASTERN)


def can_have_earnings(symbol: str) -> bool:
    """False for indexes, futures, currencies and crypto, which have no earnings to look up."""
    text = _normalize_symbol(symbol)
    return bool(text) and not text.startswith("^") and "=" not in text and not text.endswith("-USD")


# --- fetching and normalizing ------------------------------------------------------------------


def _ticker(yf_symbol: str):
    import yfinance as yf

    return yf.Ticker(yf_symbol)


def fetch_earnings(symbol: str) -> pd.DataFrame | None:
    """Every earnings row Yahoo lists for ``symbol``, newest first, or None when it lists none.

    Raises on a network or parsing failure, so the caller can tell "no earnings" from "could not ask".
    """
    from getdata import to_yf_symbol

    ticker = _ticker(to_yf_symbol(_normalize_symbol(symbol)))
    frames: list[pd.DataFrame] = []
    for page in range(MAX_PAGES):
        if page == 0:
            frame = ticker.get_earnings_dates(limit=PAGE_SIZE)
        else:
            try:
                frame = ticker.get_earnings_dates(limit=PAGE_SIZE, offset=page * PAGE_SIZE)
            except TypeError:  # a yfinance too old to page: the first 100 rows are what there is
                break
        if frame is None or frame.empty:
            break
        frames.append(frame)
        if len(frame) < PAGE_SIZE:
            break
    if not frames:
        return None
    return pd.concat(frames)


def normalize(raw: pd.DataFrame | None) -> pd.DataFrame | None:
    """One row per report day, oldest first: ``announced`` (Eastern wall time), ``date``, ``session``,
    and the EPS columns. Duplicate days keep the row that carries the most information."""
    if raw is None or raw.empty:
        return None
    index = pd.DatetimeIndex(raw.index)
    index = index.tz_localize(EASTERN) if index.tz is None else index.tz_convert(EASTERN)
    frame = pd.DataFrame(
        {
            "announced": index.tz_localize(None),
            "eps_estimate": pd.to_numeric(raw.get("EPS Estimate"), errors="coerce").to_numpy() if "EPS Estimate" in raw else np.nan,
            "eps_reported": pd.to_numeric(raw.get("Reported EPS"), errors="coerce").to_numpy() if "Reported EPS" in raw else np.nan,
            "surprise_pct": pd.to_numeric(raw.get("Surprise(%)"), errors="coerce").to_numpy() if "Surprise(%)" in raw else np.nan,
        }
    )
    frame["date"] = frame["announced"].dt.normalize()
    frame["info"] = frame[["eps_estimate", "eps_reported", "surprise_pct"]].notna().sum(axis=1)
    frame = frame.sort_values(["date", "info"]).drop_duplicates("date", keep="last").drop(columns="info")
    frame["session"] = np.where(frame["announced"].dt.hour < BEFORE_OPEN_HOUR, BEFORE_OPEN, AFTER_CLOSE)
    return frame.sort_values("date").reset_index(drop=True)[
        ["date", "announced", "session", "eps_estimate", "eps_reported", "surprise_pct"]
    ]


# --- cache -------------------------------------------------------------------------------------


def _read_cache(path: Path) -> tuple[datetime, pd.DataFrame | None] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(payload["fetched"])
        rows = payload.get("rows") or []
    except (OSError, ValueError, KeyError):
        return None
    if not rows:
        return fetched, None
    frame = pd.DataFrame(rows)
    frame["announced"] = pd.to_datetime(frame["announced"])
    for column in ("eps_estimate", "eps_reported", "surprise_pct"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")  # null in the file means missing
    frame["date"] = frame["announced"].dt.normalize()
    return fetched, frame[["date", "announced", "session", "eps_estimate", "eps_reported", "surprise_pct"]]


def _write_cache(path: Path, fetched: datetime, frame: pd.DataFrame | None) -> None:
    rows: list[dict[str, Any]] = []
    if frame is not None:
        for row in frame.itertuples(index=False):
            rows.append(
                {
                    "announced": row.announced.isoformat(),
                    "session": row.session,
                    "eps_estimate": None if pd.isna(row.eps_estimate) else float(row.eps_estimate),
                    "eps_reported": None if pd.isna(row.eps_reported) else float(row.eps_reported),
                    "surprise_pct": None if pd.isna(row.surprise_pct) else float(row.surprise_pct),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps({"fetched": fetched.isoformat(), "rows": rows}), encoding="utf-8")
    temp.replace(path)


def _is_fresh(fetched: datetime, frame: pd.DataFrame | None, now: datetime) -> bool:
    limit = EMPTY_REFRESH_DAYS if frame is None else REFRESH_DAYS
    # in UTC, so a clock change between the two moments cannot make a week 167 hours
    return now.astimezone(timezone.utc) - fetched.astimezone(timezone.utc) < timedelta(days=limit)


def load_earnings(symbol: str, *, now: datetime | None = None) -> pd.DataFrame | None:
    """The cached earnings history for ``symbol``, refreshed when stale. None when there is none.

    A download that fails falls back to whatever is cached, however old, and to None without
    caching anything when there is nothing.
    """
    key = _normalize_symbol(symbol)
    if not can_have_earnings(key):
        return None
    now = now or _now()
    with _lock:
        hit = _memory.get(key)
        if hit is not None and _is_fresh(hit[0], hit[1], now):
            return hit[1]
    path = _path(key)
    cached = _read_cache(path)
    if cached is not None and _is_fresh(cached[0], cached[1], now):
        with _lock:
            _memory[key] = cached
        return cached[1]
    try:
        frame = normalize(fetch_earnings(key))
    except Exception as exc:  # network, rate limit, a changed page: keep what we have
        LOGGER.warning("Could not download earnings dates for %s: %s", key, exc)
        return cached[1] if cached is not None else None
    try:
        _write_cache(path, now, frame)
    except OSError as exc:
        LOGGER.warning("Could not cache earnings dates for %s: %s", key, exc)
    with _lock:
        _memory[key] = (now, frame)
    return frame


def prefetch(symbols, *, workers: int = 8) -> None:
    """Warm the cache for several symbols at once, so building their frames does not wait on Yahoo
    one symbol at a time."""
    wanted = [s for s in dict.fromkeys(_normalize_symbol(s) for s in symbols) if can_have_earnings(s)]
    if len(wanted) < 2:
        return
    with ThreadPoolExecutor(max_workers=min(workers, len(wanted))) as pool:
        list(pool.map(load_earnings, wanted))


def clear_memory_cache() -> None:
    with _lock:
        _memory.clear()


# --- the indicators ----------------------------------------------------------------------------


def _bar_dates(dates) -> np.ndarray:
    """The calendar day of each bar as ``datetime64[D]``, in Eastern time when the stamps carry a zone."""
    series = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    if getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_convert(EASTERN).dt.tz_localize(None)
    return series.dt.normalize().to_numpy(dtype="datetime64[D]")


def reaction_positions(bar_days: np.ndarray, history: pd.DataFrame) -> np.ndarray:
    """Where each report's reaction day falls on the bar axis, as bar positions (sorted, unique).

    A reaction day beyond the last bar is placed after it by counting weekdays, which is exact
    except for a market holiday in between. A report whose reaction day is before the first bar
    is not on the axis and is dropped.
    """
    n = len(bar_days)
    if n == 0 or history is None or history.empty:
        return np.empty(0, dtype=np.int64)
    days = history["date"].to_numpy(dtype="datetime64[D]")
    after_close = history["session"].to_numpy() == AFTER_CLOSE
    # the weekday the market reacts on: the same day for a morning report, the next for an evening one
    reaction = np.where(
        after_close, np.busday_offset(days, 1, roll="forward"), np.busday_offset(days, 0, roll="forward")
    ).astype("datetime64[D]")
    # first bar on or after the day for a morning report, first bar after it for an evening one
    positions = np.where(
        after_close,
        np.searchsorted(bar_days, days, side="right"),
        np.searchsorted(bar_days, days, side="left"),
    ).astype(np.int64)
    beyond = positions >= n
    if beyond.any():
        extra = np.busday_count(bar_days[-1], reaction)
        positions = np.where(beyond, n - 1 + np.maximum(extra, 1), positions)
    return np.unique(positions[reaction >= bar_days[0]])


def days_to_and_since(n: int, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(to, since)`` per bar, in bars, from sorted reaction positions. ``NO_EARNINGS`` where unknown."""
    bars = np.arange(n)
    to = np.full(n, NO_EARNINGS, dtype=np.int64)
    since = np.full(n, NO_EARNINGS, dtype=np.int64)
    if len(positions):
        nxt = np.searchsorted(positions, bars, side="left")
        known = nxt < len(positions)
        to[known] = positions[nxt[known]] - bars[known]
        prev = np.searchsorted(positions, bars, side="right") - 1
        seen = prev >= 0
        since[seen] = bars[seen] - positions[prev[seen]]
    return np.minimum(to, NO_EARNINGS), np.minimum(since, NO_EARNINGS)


def earnings_columns(dates, history: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    bar_days = _bar_dates(dates)
    n = len(bar_days)
    if history is None or n == 0:
        return days_to_and_since(n, np.empty(0, dtype=np.int64))
    return days_to_and_since(n, reaction_positions(bar_days, history))


def add_earnings_columns(frame: pd.DataFrame, symbol: str | None) -> pd.DataFrame:
    """Add ``DaysToEarnings`` and ``DaysSinceEarnings`` to a daily frame for ``symbol``.

    Without a symbol (custom data, tests) or without known dates both columns are ``NO_EARNINGS``.
    """
    history = load_earnings(symbol) if symbol else None
    to, since = earnings_columns(frame["Date"], history)
    frame["DaysToEarnings"] = to.astype(float)
    frame["DaysSinceEarnings"] = since.astype(float)
    return frame


# --- diagnostics -------------------------------------------------------------------------------


def coverage(symbol: str) -> dict[str, Any]:
    """What is known for ``symbol``: how many reports, from when, and where a quarter looks missing."""
    history = load_earnings(symbol)
    if history is None:
        return {"symbol": _normalize_symbol(symbol), "reports": 0, "first": None, "last": None, "gaps": [], "upcoming": None}
    today = pd.Timestamp(_now().date())
    past = history[history["date"] <= today]
    days = pd.DatetimeIndex(past["date"])
    gaps = [
        {"after": days[i].strftime("%Y-%m-%d"), "before": days[i + 1].strftime("%Y-%m-%d"), "days": int((days[i + 1] - days[i]).days)}
        for i in range(len(days) - 1)
        if (days[i + 1] - days[i]).days > GAP_DAYS
    ]
    upcoming = history[history["date"] > today]
    return {
        "symbol": _normalize_symbol(symbol),
        "reports": int(len(past)),
        "first": None if past.empty else past["date"].iloc[0].strftime("%Y-%m-%d"),
        "last": None if past.empty else past["date"].iloc[-1].strftime("%Y-%m-%d"),
        "gaps": gaps,
        "upcoming": None if upcoming.empty else upcoming["date"].iloc[0].strftime("%Y-%m-%d"),
    }
