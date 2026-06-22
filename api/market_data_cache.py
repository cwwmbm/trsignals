from __future__ import annotations

import json
import pickle
import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PROFILE_SINGLE = "single"
PROFILE_BULK = "bulk"
ALL_PROFILES = (PROFILE_SINGLE, PROFILE_BULK)

CACHE_ROOT = Path(__file__).resolve().parent.parent / "data" / "market_cache"
EASTERN = ZoneInfo("America/New_York")


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _filename_symbol(symbol: str) -> str:
    cleaned = _normalize_symbol(symbol)
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", cleaned)


def _cache_path(symbol: str, years: int, profile: str) -> Path:
    safe = _filename_symbol(symbol)
    return CACHE_ROOT / profile / f"{safe}_{years}y.pkl"


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".meta.json")


def _eastern_today() -> date:
    return datetime.now(EASTERN).date()


def _is_fresh(cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    cached_date = datetime.fromtimestamp(cache_path.stat().st_mtime, EASTERN).date()
    return cached_date == _eastern_today()


def _meta_compatible(meta_path: Path) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    cached_version = meta.get("pandas_version")
    if cached_version is None:
        return False
    return cached_version == pd.__version__


def _invalidate_cache(cache_path: Path) -> None:
    meta_path = _meta_path(cache_path)
    for path in (cache_path, meta_path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def load(symbol: str, years: int, profile: str) -> pd.DataFrame | None:
    """Return a cached prepared frame copy, or None if missing/stale."""
    cache_path = _cache_path(symbol, years, profile)
    meta_path = _meta_path(cache_path)
    if not _is_fresh(cache_path) or not _meta_compatible(meta_path):
        if cache_path.exists() and not _meta_compatible(meta_path):
            _invalidate_cache(cache_path)
        return None
    try:
        with cache_path.open("rb") as handle:
            frame = pickle.load(handle)
    except Exception:
        _invalidate_cache(cache_path)
        return None
    if not isinstance(frame, pd.DataFrame):
        _invalidate_cache(cache_path)
        return None
    return frame.copy()


def save(symbol: str, years: int, profile: str, frame: pd.DataFrame) -> None:
    cache_path = _cache_path(symbol, years, profile)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".tmp.pkl")
    with temp_path.open("wb") as handle:
        pickle.dump(frame, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(cache_path)

    meta = {
        "symbol": _normalize_symbol(symbol),
        "years": years,
        "profile": profile,
        "pandas_version": pd.__version__,
        "cached_date": _eastern_today().isoformat(),
        "row_count": int(frame.shape[0]),
        "last_date": pd.to_datetime(frame["Date"]).iloc[-1].isoformat()
        if "Date" in frame.columns and not frame.empty
        else None,
    }
    _meta_path(cache_path).write_text(json.dumps(meta, indent=2))


def load_close_column(symbol: str, years: int) -> pd.Series | None:
    """Load proxy close prices from any fresh cached profile."""
    for profile in ALL_PROFILES:
        frame = load(symbol, years, profile)
        if frame is None or "Close" not in frame.columns or "Date" not in frame.columns:
            continue
        indexed = frame.set_index(pd.to_datetime(frame["Date"]))
        return indexed["Close"].copy()
    return None
