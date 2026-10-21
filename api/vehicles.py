"""What a trading symbol earns: the leveraged stand-in, the real leverage symbol, and when each applies.

For a symbol with a leverage-settings entry there are two daily return streams on the symbol's
own calendar:

- ``synthetic``: the stand-in, ``beta * return - costs`` (an ideal ``leverage * return`` unless the
  entry carries a cost model). Always available.
- ``vehicle``: the leverage symbol's real return, NaN where it has no history.

``vehicle_ok`` marks the bars where the leverage symbol has a close, so a position can be opened
in it. Engines choose the stream when a position is opened and keep it until the position is
closed; a trade already open when the leverage symbol appears finishes on the stand-in.
A symbol without an entry has no vehicle: its stand-in is ``general_leverage * return``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from api import leverage_settings, vehicle_data
from api.leverage_settings import LeverageSettings, SymbolLeverage

TRADING_DAYS = 252
# Engines keep the leverage symbol's returns and closes beside the stand-in under this key.
VEHICLE_SEP = ">"


def stamp(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Record which symbol a frame belongs to, so a strategy trading its own symbol knows it."""
    frame.attrs["symbol"] = str(symbol).strip().upper()
    return frame


def vehicle_key(symbol: str, vehicle_symbol: str) -> str:
    return f"{symbol}{VEHICLE_SEP}{vehicle_symbol}"


def split_vehicle_key(key: str) -> tuple[str, str] | None:
    if VEHICLE_SEP not in key:
        return None
    symbol, vehicle = key.split(VEHICLE_SEP, 1)
    return symbol, vehicle


@dataclass
class TradedStream:
    symbol: str
    vehicle_symbol: str | None  # None: no product to trade, the stand-in carries every trade
    leverage: float
    synthetic: np.ndarray
    vehicle: np.ndarray
    vehicle_ok: np.ndarray
    first_vehicle_date: pd.Timestamp | None

    @property
    def vehicle_available(self) -> bool:
        return bool(self.vehicle_ok.any())

    # Plain lists read faster than numpy scalars inside a per-bar Python loop. Computed once.
    @cached_property
    def synthetic_list(self) -> list[float]:
        return self.synthetic.tolist()

    @cached_property
    def vehicle_list(self) -> list[float]:
        return self.vehicle.tolist()

    @cached_property
    def vehicle_usable_list(self) -> list[bool]:
        return np.isfinite(self.vehicle).tolist()

    @cached_property
    def vehicle_ok_list(self) -> list[bool]:
        return self.vehicle_ok.tolist()


def synthetic_returns(
    close: pd.Series,
    entry: SymbolLeverage | None,
    general: float,
    rates: pd.Series | None = None,
) -> pd.Series:
    """Daily returns of the leveraged stand-in, indexed like ``close`` (NaN where unknown)."""
    returns = close.astype(float).pct_change()
    if entry is None:
        return returns * general
    out = entry.effective_beta * returns
    if entry.annual_fee or entry.financing_multiple:
        rate = (
            rates.reindex(close.index, method="ffill").fillna(0.0)
            if rates is not None and len(rates)
            else pd.Series(0.0, index=close.index)
        )
        out = out - (entry.annual_fee + entry.financing_multiple * rate) / TRADING_DAYS
    return out


def align_close(vehicle_close: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """The leverage symbol's closes on ``index`` (NaN where it has none).

    Matching is by calendar day, so a timezone or a time of day on either side cannot make
    every date miss.
    """
    index = pd.DatetimeIndex(index)
    day = index.tz_localize(None) if index.tz is not None else index
    source = vehicle_close.copy()
    source.index = pd.DatetimeIndex(source.index).normalize()
    aligned = source[~source.index.duplicated(keep="last")].reindex(day.normalize())
    aligned.index = index
    return aligned


def vehicle_returns(vehicle_close: pd.Series, index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """``(returns, ok)`` of the leverage symbol on ``index``.

    A return needs a close on both the day and the day before; otherwise it is NaN and the
    stand-in covers that day.
    """
    closes = align_close(vehicle_close, index)
    ok = closes.notna()
    returns = closes / closes.shift(1) - 1.0
    return returns.where(closes.notna() & closes.shift(1).notna()), ok


def build_stream(
    index: pd.DatetimeIndex,
    close: pd.Series,
    entry: SymbolLeverage,
    *,
    vehicle_close: pd.Series | None,
    rates: pd.Series | None,
) -> TradedStream:
    synthetic = synthetic_returns(close, entry, entry.leverage, rates)
    if vehicle_close is not None and len(vehicle_close):
        veh, ok = vehicle_returns(vehicle_close, index)
        first = vehicle_close.index.min()
    else:
        veh = pd.Series(np.nan, index=index)
        ok = pd.Series(False, index=index)
        first = None
    return TradedStream(
        symbol=entry.symbol,
        vehicle_symbol=entry.leverage_symbol,
        leverage=entry.leverage,
        synthetic=synthetic.to_numpy(dtype=float),
        vehicle=veh.to_numpy(dtype=float),
        vehicle_ok=ok.to_numpy(dtype=bool),
        first_vehicle_date=None if first is None else pd.Timestamp(first),
    )


def _as_index(dates) -> pd.DatetimeIndex:
    """A DatetimeIndex without re-parsing dates that already are datetimes (that costs milliseconds)."""
    if isinstance(dates, pd.DatetimeIndex):
        return dates
    series = dates if isinstance(dates, pd.Series) else pd.Series(dates)
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.DatetimeIndex(series)
    return pd.DatetimeIndex(pd.to_datetime(series))


# A sweep runs thousands of backtests on the same frame; the stream is the same each time.
_STREAM_CACHE_SIZE = 32
_stream_cache: dict[tuple, tuple[object, object, TradedStream]] = {}


def stream_for(
    dates: pd.Series | pd.DatetimeIndex,
    close_values: np.ndarray | pd.Series,
    symbol: str | None,
    settings: LeverageSettings | None = None,
) -> TradedStream | None:
    """The traded stream for ``symbol`` on a frame's calendar, or None without a settings entry.

    The returned arrays are shared between calls and must not be modified.
    """
    settings = settings or leverage_settings.load_settings()
    entry = settings.for_symbol(symbol)
    if entry is None:
        return None
    values = np.ascontiguousarray(close_values, dtype=float)
    index = _as_index(dates)
    vehicle_close = vehicle_data.load_close(entry.leverage_symbol) if entry.has_vehicle else None
    rates = vehicle_data.load_rates() if entry.financing_multiple else None
    key = (entry, len(index), index[0] if len(index) else None, index[-1] if len(index) else None, hash(values.tobytes()))
    hit = _stream_cache.get(key)
    # The cached stream is only valid for the very same price series objects it was built from.
    if hit is not None and hit[0] is vehicle_close and hit[1] is rates:
        return hit[2]
    close = pd.Series(values, index=index)
    stream = build_stream(index, close, entry, vehicle_close=vehicle_close, rates=rates)
    if len(_stream_cache) >= _STREAM_CACHE_SIZE:
        _stream_cache.pop(next(iter(_stream_cache)))
    _stream_cache[key] = (vehicle_close, rates, stream)
    return stream


def clear_stream_cache() -> None:
    _stream_cache.clear()


def estimate_cost_model(
    underlying_close: pd.Series,
    vehicle_close: pd.Series,
    rates: pd.Series | None,
) -> dict:
    """Fit ``vehicle_return = beta * return - (fee + k * T-bill) / 252`` on the days both exist.

    Returns the fitted parameters and how well the fit tracks, so a person can judge them.
    """
    joined = pd.DataFrame(
        {"u": underlying_close.pct_change(), "v": vehicle_close.pct_change()}
    ).dropna()
    if len(joined) < 60:
        raise ValueError("Not enough overlapping history to estimate a cost model")
    columns = [joined["u"].to_numpy()]
    names = ["beta"]
    rate = (
        rates.reindex(joined.index, method="ffill").fillna(0.0).to_numpy()
        if rates is not None and len(rates)
        else None
    )
    # A rate that never moves cannot be told apart from the fee, so it is left out of the fit.
    if rate is not None and rate.std() > 1e-4:
        columns += [-np.ones(len(joined)) / TRADING_DAYS, -rate / TRADING_DAYS]
        names += ["fee", "financing_multiple"]
    else:
        columns += [-np.ones(len(joined)) / TRADING_DAYS]
        names += ["fee"]
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, joined["v"].to_numpy(), rcond=None)
    fit = dict(zip(names, (float(value) for value in coefficients)))
    fit.setdefault("financing_multiple", 0.0)
    fitted = design @ coefficients
    residual = joined["v"].to_numpy() - fitted
    return {
        **fit,
        "days": int(len(joined)),
        "first_date": joined.index[0].strftime("%Y-%m-%d"),
        "last_date": joined.index[-1].strftime("%Y-%m-%d"),
        "correlation": float(np.corrcoef(joined["u"], joined["v"])[0, 1]),
        "tracking_error": float(residual.std() * np.sqrt(TRADING_DAYS)),
    }
