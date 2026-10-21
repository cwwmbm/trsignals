"""The instruments a book can hold, and which one a new position opens in.

Strategies and allocation models speak in *underlyings* (SOXX). The ledger holds *instruments*:
the leveraged stand-in (keyed by the underlying's symbol) and, when the leverage settings name
one, the real leverage symbol (SOXL). The rule is the one in ``api.vehicles``: while the book
holds nothing in an underlying, its next position opens in the leverage symbol if that has a
close that day, otherwise in the stand-in; a position that is open stays in the instrument it
opened in until the book is flat in that underlying again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from api import leverage_settings, vehicles

# Quantities at or below this are flat. Matches the ledger's dust threshold.
FLAT_UNITS = 1e-12
SYNTHETIC_START_PRICE = 100.0


@dataclass(frozen=True)
class Instrument:
    key: str  # what the ledger and the transactions call it
    underlying: str
    kind: str  # "synthetic" (stand-in) or "vehicle" (the real leverage symbol)
    leverage: float | None
    configured: bool  # the underlying has a leverage-settings entry

    def as_dict(self) -> dict:
        return {
            "symbol": self.key,
            "underlying": self.underlying,
            "kind": self.kind,
            "leverage": self.leverage,
            "configured": self.configured,
        }


def _returns(symbol: str, track_changes: Mapping[str, pd.Series], dates: pd.DatetimeIndex) -> np.ndarray:
    series = track_changes.get(symbol)
    if series is None:
        return np.zeros(len(dates), dtype=float)
    series = series[~series.index.duplicated(keep="last")]
    values = series.reindex(dates).to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    # An instrument cannot lose more than everything in a day.
    return np.maximum(values, -1.0)


def _prices(returns: np.ndarray) -> np.ndarray:
    prices = np.empty(len(returns), dtype=float)
    prices[0] = SYNTHETIC_START_PRICE
    for i in range(1, len(returns)):
        prices[i] = prices[i - 1] * (1.0 + returns[i])
    return prices


def _display_close(symbol: str, close_series: Mapping[str, pd.Series], dates: pd.DatetimeIndex) -> np.ndarray:
    series = close_series.get(symbol)
    if series is None:
        return np.full(len(dates), np.nan)
    series = series[~series.index.duplicated(keep="last")]
    return series.reindex(dates).ffill().to_numpy(dtype=float)


class InstrumentBook:
    """Price paths, display closes and routing for every instrument of a set of underlyings."""

    def __init__(
        self,
        underlyings: Sequence[str],
        track_changes: Mapping[str, pd.Series],
        close_series: Mapping[str, pd.Series],
        dates: pd.DatetimeIndex,
    ) -> None:
        settings = leverage_settings.load_settings()
        self.dates = pd.DatetimeIndex(dates)
        self.underlyings = sorted(set(underlyings))
        self.instruments: dict[str, Instrument] = {}
        self.paths: dict[str, np.ndarray] = {}
        self.display_close: dict[str, np.ndarray] = {}
        self._entries: dict[str, leverage_settings.SymbolLeverage] = {}
        self._vehicle: dict[str, str] = {}
        self._vehicle_ok: dict[str, np.ndarray] = {}
        self.current: dict[str, str] = {}

        for underlying in self.underlyings:
            entry = settings.for_symbol(underlying)
            self.instruments[underlying] = Instrument(
                underlying, underlying, "synthetic",
                entry.leverage if entry else settings.general_leverage, entry is not None,
            )
            self.paths[underlying] = _prices(_returns(underlying, track_changes, self.dates))
            self.display_close[underlying] = _display_close(underlying, close_series, self.dates)
            self.current[underlying] = underlying
            if entry is None:
                continue
            self._entries[underlying] = entry
            if not entry.has_vehicle:
                continue  # no product to trade: the stand-in carries the whole history
            key = vehicles.vehicle_key(underlying, entry.leverage_symbol)
            if key not in track_changes or key not in close_series:
                continue  # the leverage symbol has no data: the stand-in carries the whole history
            if entry.leverage_symbol in self.underlyings or entry.leverage_symbol in self.instruments:
                raise ValueError(
                    f"{entry.leverage_symbol} is the leverage symbol of {underlying} and is also traded "
                    "as a symbol of its own in this portfolio"
                )
            vehicle_close = close_series[key]
            vehicle_close = vehicle_close[~vehicle_close.index.duplicated(keep="last")]
            ok = vehicle_close.reindex(self.dates).notna().to_numpy()
            returns = _returns(key, track_changes, self.dates)
            # Where the leverage symbol has no usable return, the stand-in covers that day so the
            # price path never jumps.
            synthetic = _returns(underlying, track_changes, self.dates)
            raw_series = track_changes[key]
            raw = raw_series[~raw_series.index.duplicated(keep="last")].reindex(self.dates).to_numpy(dtype=float)
            returns = np.where(np.isfinite(raw), returns, synthetic)
            self.instruments[entry.leverage_symbol] = Instrument(
                entry.leverage_symbol, underlying, "vehicle", 1.0, True
            )
            self.paths[entry.leverage_symbol] = _prices(returns)
            self.display_close[entry.leverage_symbol] = _display_close(key, close_series, self.dates)
            self._vehicle[underlying] = entry.leverage_symbol
            self._vehicle_ok[underlying] = ok

    @property
    def keys(self) -> list[str]:
        return sorted(self.instruments)

    def prices_at(self, bar: int) -> dict[str, float]:
        return {key: self.paths[key][bar] for key in self.instruments}

    def route(self, bar: int, shares: Callable[[str], float]) -> dict[str, str]:
        """Choose each underlying's instrument for new positions on ``bar``.

        An underlying the book holds nothing in may switch; one it holds stays where it is.
        """
        for underlying in self.underlyings:
            vehicle = self._vehicle.get(underlying)
            if vehicle is None:
                continue
            if shares(self.current[underlying]) > FLAT_UNITS:
                continue
            self.current[underlying] = vehicle if self._vehicle_ok[underlying][bar] else underlying
        return dict(self.current)

    def metadata(self) -> dict[str, dict]:
        return {key: instrument.as_dict() for key, instrument in self.instruments.items()}

    def report(self) -> list[dict]:
        """One row per underlying that has a leverage-settings entry, for the payload."""
        rows = []
        for underlying, entry in self._entries.items():
            ok = self._vehicle_ok.get(underlying)
            first = self.dates[int(np.argmax(ok))] if ok is not None and ok.any() else None
            rows.append(
                {
                    "symbol": underlying,
                    "leverage_symbol": entry.leverage_symbol,
                    "leverage": entry.leverage,
                    "first_date": None if first is None else pd.Timestamp(first).strftime("%Y-%m-%d"),
                    # False: no price history for the leverage symbol, so the stand-in carried every trade
                    "available": ok is not None and bool(ok.any()),
                }
            )
        return rows
