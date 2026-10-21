"""Leverage settings: what a trading symbol is traded as.

- **General leverage** applies to every symbol without an entry: the symbol's own daily return
  times the factor, compounded (a daily-reset leveraged product).
- A **symbol entry** (for example SOXX with leverage symbol SOXL, leverage 3) trades the leverage
  symbol's real returns once it exists. Before that, and for any trade opened before it existed,
  the stand-in is the symbol's own return times the entry's leverage, with an optional cost model
  fitted from the period both exist. Signals always come from the original symbol.
- An entry **without a leverage symbol** (for example RIVN at leverage 1) has no product to trade:
  the stand-in carries every trade for the whole history. This is how a single stock is traded
  unleveraged while the general leverage stays 3 for everything else. A leverage symbol left blank,
  or equal to the symbol itself, means exactly this.

The settings live in ``data/settings.json``. A missing file means "general leverage from
``config.Leverage``, no entries", which is the behavior before this feature existed.
"""

from __future__ import annotations

import fcntl
import json
import math
import re
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import config

DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "settings.json"

MAX_LEVERAGE = 10.0
MAX_ANNUAL_FEE = 0.20
MAX_FINANCING_MULTIPLE = 10.0
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.^=_-]{0,15}$")


@dataclass(frozen=True)
class SymbolLeverage:
    """One trading symbol and the leveraged product traded in its place.

    ``leverage_symbol`` is None when there is no product to trade (see the module docstring).

    ``beta``, ``annual_fee`` and ``financing_multiple`` shape only the stand-in used where the
    leverage symbol has no history (everywhere, when there is none): ``leverage_symbol_return ~
    beta * return - (annual_fee + financing_multiple * T-bill rate) / 252``. Left at their
    defaults the stand-in is an ideal ``leverage * return``.
    """

    symbol: str
    leverage_symbol: str | None
    leverage: float
    beta: float | None = None
    annual_fee: float = 0.0
    financing_multiple: float = 0.0

    @property
    def has_vehicle(self) -> bool:
        return self.leverage_symbol is not None

    @property
    def effective_beta(self) -> float:
        return self.leverage if self.beta is None else self.beta

    @property
    def has_cost_model(self) -> bool:
        return self.beta is not None or self.annual_fee > 0 or self.financing_multiple > 0


@dataclass(frozen=True)
class LeverageSettings:
    general_leverage: float = float(config.Leverage)
    symbols: tuple[SymbolLeverage, ...] = field(default_factory=tuple)

    def for_symbol(self, symbol: str | None) -> SymbolLeverage | None:
        if not symbol:
            return None
        key = str(symbol).strip().upper()
        for entry in self.symbols:
            if entry.symbol == key:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "general_leverage": self.general_leverage,
            "symbols": [asdict(entry) for entry in self.symbols],
        }


def default_settings() -> LeverageSettings:
    return LeverageSettings(general_leverage=float(config.Leverage), symbols=())


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _symbol(value: Any, name: str) -> str:
    text = str(value or "").strip().upper()
    if not _SYMBOL.match(text):
        raise ValueError(f"{name} must be a ticker symbol such as SOXX (got {value!r})")
    return text


def normalize_entry(raw: Any) -> SymbolLeverage:
    get = raw.get if isinstance(raw, dict) else lambda key, default=None: getattr(raw, key, default)
    leverage = _finite(get("leverage"), "Leverage")
    if not 0 < leverage <= MAX_LEVERAGE:
        raise ValueError(f"Leverage must be above 0 and at most {MAX_LEVERAGE:g}")
    beta_raw = get("beta")
    beta = None if beta_raw in (None, "") else _finite(beta_raw, "Beta")
    if beta is not None and not 0 < beta <= MAX_LEVERAGE:
        raise ValueError(f"Beta must be above 0 and at most {MAX_LEVERAGE:g}")
    fee = _finite(get("annual_fee") or 0.0, "Annual fee")
    if not 0 <= fee <= MAX_ANNUAL_FEE:
        raise ValueError(f"Annual fee must be between 0% and {MAX_ANNUAL_FEE:.0%}")
    financing = _finite(get("financing_multiple") or 0.0, "Financing multiple")
    if not 0 <= financing <= MAX_FINANCING_MULTIPLE:
        raise ValueError(f"Financing multiple must be between 0 and {MAX_FINANCING_MULTIPLE:g}")
    symbol = _symbol(get("symbol"), "Symbol")
    raw_vehicle = get("leverage_symbol")
    vehicle = None if raw_vehicle is None or not str(raw_vehicle).strip() else _symbol(raw_vehicle, "Leverage symbol")
    if vehicle == symbol:
        vehicle = None  # trading the symbol itself: there is no separate product
    return SymbolLeverage(
        symbol=symbol,
        leverage_symbol=vehicle,
        leverage=leverage,
        beta=beta,
        annual_fee=fee,
        financing_multiple=financing,
    )


def normalize_settings(general_leverage: Any, entries: Iterable[Any]) -> LeverageSettings:
    """Validate and normalize user input. Raises ValueError with a message fit to show."""
    general = _finite(general_leverage, "General leverage")
    if not 0 < general <= MAX_LEVERAGE:
        raise ValueError(f"General leverage must be above 0 and at most {MAX_LEVERAGE:g}")
    normalized = tuple(normalize_entry(entry) for entry in entries)
    symbols = [entry.symbol for entry in normalized]
    vehicles = [entry.leverage_symbol for entry in normalized if entry.has_vehicle]
    if len(set(symbols)) != len(symbols):
        duplicate = next(s for s in symbols if symbols.count(s) > 1)
        raise ValueError(f"{duplicate} is listed more than once")
    if len(set(vehicles)) != len(vehicles):
        duplicate = next(v for v in vehicles if vehicles.count(v) > 1)
        raise ValueError(f"{duplicate} is the leverage symbol of more than one symbol")
    for entry in normalized:
        if entry.has_vehicle and entry.leverage_symbol in symbols:
            raise ValueError(
                f"{entry.leverage_symbol} is itself listed as a symbol, so {entry.symbol} would "
                "trade a chain of leverage symbols"
            )
    return LeverageSettings(general_leverage=general, symbols=normalized)


def settings_from_dict(data: dict[str, Any]) -> LeverageSettings:
    return normalize_settings(
        data.get("general_leverage", config.Leverage), data.get("symbols") or []
    )


# --- storage ---------------------------------------------------------------

_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"key": None, "settings": None}


def _store_path(path: Path | None) -> Path:
    return Path(path) if path is not None else Path(DEFAULT_STORE_PATH)


@contextmanager
def _file_lock(path: Path):
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _stat_key(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path), stat.st_mtime_ns, stat.st_size)


def load_settings(path: Path | None = None) -> LeverageSettings:
    """The saved settings, re-read only when the file changes. Missing or unreadable means defaults."""
    target = _store_path(path)
    key = _stat_key(target)
    with _cache_lock:
        if key is not None and _cache["key"] == key:
            return _cache["settings"]
        if key is None:
            settings = default_settings()
        else:
            try:
                settings = settings_from_dict(json.loads(target.read_text(encoding="utf-8")))
            except (OSError, ValueError, json.JSONDecodeError):
                settings = default_settings()
        _cache["key"], _cache["settings"] = key, settings
        return settings


def save_settings(settings: LeverageSettings, path: Path | None = None) -> LeverageSettings:
    """Validate, then write atomically. Returns what was stored."""
    clean = normalize_settings(settings.general_leverage, settings.symbols)
    target = _store_path(path)
    with _file_lock(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(f"{target.suffix}.tmp")
        temp.write_text(json.dumps(clean.to_dict(), indent=2) + "\n", encoding="utf-8")
        temp.replace(target)
    with _cache_lock:
        _cache["key"] = None
    return clean


def general_leverage() -> float:
    return load_settings().general_leverage


def symbol_setting(symbol: str | None) -> SymbolLeverage | None:
    return load_settings().for_symbol(symbol)
