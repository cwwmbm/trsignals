"""Buy-and-hold benchmarks for a backtest result.

A strategy that is invested 45% of the time can look good or bad only against what holding
the instrument would have done. Two benchmarks are built from the same inputs the engine
trades on, over the same bars as the result's metrics:

- **underlying**: the traded symbol itself, unleveraged.
- **vehicle**: what the engine would have earned holding the traded product all the time: the
  leverage symbol's real daily returns where it exists, the leveraged stand-in before that.
  A symbol without a leverage-settings entry earns the general leverage times its return.

Both start at the result's first scored equity value, so the curves overlay directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from api import leverage_settings, vehicles
from stats import compute_aggregate_metrics

BENCHMARK_COLUMN_PREFIX = "Benchmark_"


@dataclass
class BenchmarkInputs:
    """Daily inputs for one symbol, each indexed by calendar date."""

    symbol: str
    close: pd.Series
    synthetic: pd.Series  # leveraged stand-in returns
    vehicle: pd.Series | None  # leverage symbol's real returns, NaN where it has no close
    leverage: float
    leverage_symbol: str | None = None
    first_vehicle_date: pd.Timestamp | None = None


def _calendar_index(dates) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(dates))).normalize()


def inputs_from_frame(data: pd.DataFrame, symbol: str | None = None) -> BenchmarkInputs | None:
    """Inputs for the symbol a single-strategy frame trades, or None when it cannot be told.

    ``symbol`` is the trading symbol (the proxy when there is one). It defaults to the symbol
    the frame was loaded for, whose prices are the frame's own ``Close``.
    """
    if data is None or data.empty or "Date" not in data.columns:
        return None
    own = getattr(data, "attrs", {}).get("symbol")
    symbol = (symbol or own or "").strip().upper()
    if not symbol:
        return None
    if own and symbol != own:
        if symbol not in data.columns:
            return None
        values = data[symbol]
    elif "Close" in data.columns:
        values = data["Close"]
    else:
        return None
    index = _calendar_index(data["Date"])
    close = pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float), index=index)
    close = close[~close.index.duplicated(keep="last")]

    general = leverage_settings.general_leverage()
    stream = vehicles.stream_for(data["Date"], close.reindex(index).to_numpy(dtype=float), symbol)
    if stream is None:
        return BenchmarkInputs(
            symbol=symbol,
            close=close,
            synthetic=close.pct_change() * general,
            vehicle=None,
            leverage=general,
        )
    vehicle = None
    if stream.vehicle_symbol is not None and stream.vehicle_available:
        vehicle = pd.Series(stream.vehicle, index=index)
        vehicle = vehicle[~vehicle.index.duplicated(keep="last")]
    synthetic = pd.Series(stream.synthetic, index=index)
    return BenchmarkInputs(
        symbol=symbol,
        close=close,
        synthetic=synthetic[~synthetic.index.duplicated(keep="last")],
        vehicle=vehicle,
        leverage=stream.leverage,
        leverage_symbol=stream.vehicle_symbol if vehicle is not None else None,
        first_vehicle_date=stream.first_vehicle_date if vehicle is not None else None,
    )


def inputs_from_series(
    symbol: str,
    close_series: dict[str, pd.Series],
    track_changes: dict[str, pd.Series],
) -> BenchmarkInputs | None:
    """Inputs for a portfolio, from the market series its simulation already loaded."""
    close = close_series.get(symbol)
    synthetic = track_changes.get(symbol)
    if close is None or synthetic is None:
        return None
    entry = leverage_settings.load_settings().for_symbol(symbol)
    vehicle = None
    leverage_symbol = None
    first = None
    if entry is not None and entry.has_vehicle:
        key = vehicles.vehicle_key(symbol, entry.leverage_symbol)
        vehicle = track_changes.get(key)
        if vehicle is not None and vehicle.notna().any():
            leverage_symbol = entry.leverage_symbol
            first = pd.Timestamp(vehicle.first_valid_index())
        else:
            vehicle = None
    leverage = entry.leverage if entry is not None else leverage_settings.general_leverage()
    return BenchmarkInputs(
        symbol=symbol,
        close=close,
        synthetic=synthetic,
        vehicle=vehicle,
        leverage=leverage,
        leverage_symbol=leverage_symbol,
        first_vehicle_date=first,
    )


def _held_returns(inputs: BenchmarkInputs) -> pd.Series:
    """Per-bar return of holding the traded product: the real one where it exists, else the stand-in."""
    returns = inputs.synthetic
    if inputs.vehicle is not None:
        real = inputs.vehicle.reindex(returns.index)
        returns = real.where(real.notna(), returns)
    return returns


def _equity(returns: pd.Series, dates: pd.DatetimeIndex, start: float) -> np.ndarray:
    values = returns.reindex(dates).fillna(0.0).to_numpy(dtype=float).copy()
    if len(values):
        values[0] = 0.0  # the first scored bar is the starting value, as in the strategy's own curve
    values = np.where(np.isfinite(values), values, 0.0)
    return start * np.cumprod(1.0 + values)


def _metrics(dates: pd.Series, equity: np.ndarray) -> dict:
    frame = pd.DataFrame({"Date": pd.to_datetime(dates).to_numpy(), "RollingPnL": equity})
    peak = frame["RollingPnL"].cummax()
    frame["Drawdown"] = (peak - frame["RollingPnL"]) / peak
    frame["LongTradeOut"] = False
    frame["TradePnL"] = 0.0
    m = compute_aggregate_metrics(frame, include_full=True)
    full = m.get("full") or {}
    return {
        "total_return": float(equity[-1] / equity[0] - 1.0) if len(equity) and equity[0] else None,
        # Same basis as the result cards: ratios leave out the series' own best year,
        # drawdown covers every year.
        "cagr_percent": m["cagr_percent"],
        "sharpe": m["sharpe"],
        "sortino": m["sortino"],
        "max_drawdown": full.get("max_drawdown"),
        "full_cagr_percent": full.get("cagr_percent"),
        "full_sharpe": full.get("sharpe"),
        "full_sortino": full.get("sortino"),
        "excluded_year": m["excluded_year"],
    }


def _finite(value):
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _series_label(inputs: BenchmarkInputs) -> tuple[str, str | None]:
    """Labels for the underlying and the vehicle benchmark."""
    underlying = f"{inputs.symbol} buy & hold"
    if inputs.leverage_symbol:
        since = (
            f" (as {inputs.symbol} {inputs.leverage:g}x before {inputs.first_vehicle_date:%Y-%m-%d})"
            if inputs.first_vehicle_date is not None
            else ""
        )
        return underlying, f"{inputs.leverage_symbol} buy & hold{since}"
    return underlying, f"{inputs.symbol} {inputs.leverage:g}x buy & hold"


def benchmark_for(scored: pd.DataFrame, inputs: BenchmarkInputs | None) -> tuple[dict, dict] | None:
    """``(columns, meta)`` for a scored frame, or None when there is nothing to compare.

    ``columns`` maps ``Benchmark_<id>`` to an equity array on the frame's rows. ``meta`` is the
    payload block: labels, metrics and the strategy's time in the market.
    """
    if inputs is None or scored is None or len(scored) < 2:
        return None
    dates = _calendar_index(scored["Date"])
    start = float(scored["RollingPnL"].iloc[0])
    if not np.isfinite(start) or start <= 0:
        return None
    close = inputs.close[~inputs.close.index.duplicated(keep="last")]
    underlying_returns = close.pct_change()
    # A frame with no price at all for the symbol cannot be benchmarked.
    if not underlying_returns.reindex(dates).notna().any():
        return None

    underlying_label, vehicle_label = _series_label(inputs)
    columns: dict[str, np.ndarray] = {}
    series: list[dict] = []

    equity = _equity(underlying_returns, dates, start)
    columns[f"{BENCHMARK_COLUMN_PREFIX}underlying"] = equity
    series.append(
        {
            "id": "underlying",
            "label": underlying_label,
            "symbol": inputs.symbol,
            "leverage": 1.0,
            "metrics": {key: _finite(value) for key, value in _metrics(scored["Date"], equity).items()},
        }
    )

    # An unleveraged symbol with no product is the same series twice: show it once.
    if inputs.leverage != 1.0 or inputs.leverage_symbol:
        equity = _equity(_held_returns(inputs), dates, start)
        columns[f"{BENCHMARK_COLUMN_PREFIX}vehicle"] = equity
        series.append(
            {
                "id": "vehicle",
                "label": vehicle_label,
                "symbol": inputs.leverage_symbol or inputs.symbol,
                "leverage": float(inputs.leverage),
                "metrics": {key: _finite(value) for key, value in _metrics(scored["Date"], equity).items()},
            }
        )

    in_market = None
    if "HoldLong" in scored.columns:
        in_market = float(scored["HoldLong"].astype(bool).mean() * 100.0)
    return columns, {"symbol": inputs.symbol, "series": series, "time_in_market_percent": in_market}
