"""Saved Quote-screen symbols.

Missing file means the built-in SPY, QQQ, SOXX, and VIX list. The first add or
delete writes data/quote_symbols.json, which then stays the source of truth.
"""

from __future__ import annotations

import fcntl
import json
import re
from contextlib import contextmanager
from pathlib import Path

from api.quote_service import QUOTE_SYMBOLS

DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "quote_symbols.json"

_TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")
_FUTURES = frozenset({"NQ", "ES", "RTY", "CL", "GC", "SI", "HG", "NG"})


def normalize_quote_symbol(raw: str) -> tuple[str, str]:
    """Return (display symbol, Yahoo symbol)."""
    value = str(raw).strip().upper()
    if not value:
        raise ValueError("Enter a symbol")
    if value in {"VIX", "^VIX"}:
        return "VIX", "^VIX"

    yf_symbol = value
    if value.endswith("=F") or value.endswith("=X"):
        base = value[:-2]
        display = base
    elif value.startswith("^"):
        base = value[1:]
        display = value
    else:
        base = value
        display = value
        if value in _FUTURES:
            yf_symbol = f"{value}=F"
    if not base or _TICKER.fullmatch(base) is None:
        raise ValueError("Use a ticker like IWM, BRK-B, or ^TNX")
    return display, yf_symbol


@contextmanager
def _store_lock(path: Path):
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_raw(path: Path) -> list:
    with path.open(encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Quote symbol store JSON is invalid: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("Quote symbol store must contain a JSON array")
    return payload


def _pairs_from_raw(items: list) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Quote symbol store entries must be objects")
        symbol = str(item.get("symbol", "")).strip()
        yf_symbol = str(item.get("yf_symbol", "")).strip()
        if not symbol or not yf_symbol:
            raise ValueError("Quote symbol entry needs symbol and yf_symbol")
        pairs.append((symbol, yf_symbol))
    return pairs


def _write_raw(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"symbol": symbol, "yf_symbol": yf_symbol} for symbol, yf_symbol in pairs]
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temp_path.replace(path)


def list_quote_symbols(store_path: Path | None = None) -> list[tuple[str, str]]:
    path = store_path or DEFAULT_STORE_PATH
    if not path.exists():
        return list(QUOTE_SYMBOLS)
    return _pairs_from_raw(_load_raw(path))


def add_quote_symbol(raw: str, store_path: Path | None = None) -> list[tuple[str, str]]:
    path = store_path or DEFAULT_STORE_PATH
    display, yf_symbol = normalize_quote_symbol(raw)
    with _store_lock(path):
        pairs = list_quote_symbols(path)
        if any(symbol == display for symbol, _yf in pairs):
            raise ValueError(f"{display} is already on the quote list")
        pairs.append((display, yf_symbol))
        _write_raw(path, pairs)
    return pairs


def remove_quote_symbol(raw: str, store_path: Path | None = None) -> list[tuple[str, str]]:
    path = store_path or DEFAULT_STORE_PATH
    display, _yf_symbol = normalize_quote_symbol(raw)
    with _store_lock(path):
        pairs = list_quote_symbols(path)
        kept = [pair for pair in pairs if pair[0] != display]
        if len(kept) == len(pairs):
            raise ValueError(f"{display} is not on the quote list")
        _write_raw(path, kept)
    return kept
