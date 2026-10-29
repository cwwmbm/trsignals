"""Full-history closes for leverage symbols (SOXL) and the T-bill rate, cached per day.

These are kept apart from the strategy frames: they are loaded once per symbol regardless of how
many years a backtest asks for, so a sweep of thousands of backtests never re-downloads them.
"""

from __future__ import annotations

import logging
import pickle
import re
import threading
import time
from datetime import date, datetime, time as clock
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

LOGGER = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "market_cache" / "vehicle"
EASTERN = ZoneInfo("America/New_York")
HISTORY_START = "1990-01-01"
RATE_SYMBOL = "^IRX"  # 13-week T-bill yield, in percent

# While the market is open today's bar keeps moving, but the history is cached for the whole trading
# day. A frozen bar makes any position priced through a leverage symbol (the live scan's PnL) stand
# still all day, so during the session the last few bars are re-fetched at most this often.
INTRADAY_REFRESH_SECONDS = 60
TAIL_PERIOD = "5d"
SESSION_START = clock(9, 25)
SESSION_END = clock(16, 35)  # the close plus a little for Yahoo to publish the final bar

_lock = threading.Lock()
_memory: dict[str, tuple[date, pd.Series | None]] = {}
_tail_fetched: dict[str, float] = {}
_rates_cache: dict[str, object] = {"source": None, "rates": None}


def _normalize(symbol: str) -> str:
    return symbol.strip().upper()


def _path(symbol: str) -> Path:
    return CACHE_DIR / f"{re.sub(r'[^A-Za-z0-9._=-]+', '_', _normalize(symbol))}.pkl"


def _today() -> date:
    return datetime.now(EASTERN).date()


def _fresh(path: Path) -> bool:
    if not path.exists():
        return False
    return datetime.fromtimestamp(path.stat().st_mtime, EASTERN).date() == _today()


def _download(symbol: str, **kwargs) -> pd.Series | None:
    import yfinance as yf

    from getdata import to_yf_symbol

    frame = yf.download([to_yf_symbol(_normalize(symbol))], progress=False, auto_adjust=True, **kwargs)
    if frame is None or frame.empty:
        return None
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 0:
            return None
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close if len(close) else None


def download_close(symbol: str) -> pd.Series | None:
    """Adjusted daily closes from Yahoo, oldest first. None when nothing came back."""
    return _download(symbol, start=HISTORY_START)


def download_tail(symbol: str) -> pd.Series | None:
    """The last few daily closes, including today's bar so far."""
    return _download(symbol, period=TAIL_PERIOD)


def session_open(now: datetime | None = None) -> bool:
    """Whether today's bar can still be moving: a weekday between the open and shortly after the close."""
    now = now or datetime.now(EASTERN)
    return now.weekday() < 5 and SESSION_START <= now.time() <= SESSION_END


def _tail_is_stale(key: str, series: pd.Series | None) -> bool:
    if key == RATE_SYMBOL or series is None or not len(series) or not session_open():
        return False
    if series.index[-1].date() > _today():
        return False
    return time.monotonic() - _tail_fetched.get(key, float("-inf")) >= INTRADAY_REFRESH_SECONDS


def _refresh_tail(key: str, series: pd.Series) -> pd.Series:
    """``series`` with its last bars replaced by fresh ones. A failed fetch keeps the old series."""
    _tail_fetched[key] = time.monotonic()  # also on failure, so an outage is not retried every call
    try:
        tail = download_tail(key)
    except Exception as error:  # network or vendor trouble
        LOGGER.warning("Could not refresh %s: %s", key, error)
        return series
    if tail is None:
        return series
    merged = pd.concat([series[~series.index.isin(tail.index)], tail]).sort_index()
    return merged[~merged.index.duplicated(keep="last")]


def load_close(symbol: str) -> pd.Series | None:
    """Cached full-history closes, refreshed once per Eastern trading day. None if unavailable."""
    key = _normalize(symbol)
    today = _today()
    with _lock:
        hit = _memory.get(key)
    if hit is not None and hit[0] == today:
        series = hit[1]
        if _tail_is_stale(key, series):
            series = _refresh_tail(key, series)
            with _lock:
                _memory[key] = (today, series)
        return series
    path = _path(key)
    series: pd.Series | None = None
    if _fresh(path):
        try:
            with path.open("rb") as handle:
                loaded = pickle.load(handle)
            if isinstance(loaded, pd.Series) and len(loaded):
                series = loaded
        except Exception:  # a damaged cache file is just a miss
            series = None
    if series is None:
        try:
            series = download_close(key)
        except Exception as error:  # network or vendor trouble
            LOGGER.warning("Could not download %s: %s", key, error)
            series = None
        if series is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".tmp.pkl")
                with temp.open("wb") as handle:
                    pickle.dump(series, handle, protocol=pickle.HIGHEST_PROTOCOL)
                temp.replace(path)
            except OSError as error:
                LOGGER.warning("Could not cache %s: %s", key, error)
        elif path.exists():  # stale beats nothing
            try:
                with path.open("rb") as handle:
                    stale = pickle.load(handle)
                if isinstance(stale, pd.Series) and len(stale):
                    series = stale
            except Exception:
                series = None
    if _tail_is_stale(key, series):
        series = _refresh_tail(key, series)
    with _lock:
        _memory[key] = (today, series)
    return series


def load_rates() -> pd.Series | None:
    """The 3-month T-bill rate as a decimal (0.045 = 4.5%), or None."""
    close = load_close(RATE_SYMBOL)
    if close is None:
        return None
    # The same object every call while the data is unchanged, so callers can tell it is unchanged.
    with _lock:
        if _rates_cache["source"] is not close:
            _rates_cache["source"], _rates_cache["rates"] = close, close / 100.0
        return _rates_cache["rates"]  # type: ignore[return-value]


def clear_memory() -> None:
    with _lock:
        _memory.clear()
        _tail_fetched.clear()
        _rates_cache["source"] = _rates_cache["rates"] = None
