from __future__ import annotations

import config
import pandas as pd


def in_sample_fraction(fraction: float | None = None) -> float:
    value = config.IN_SAMPLE_FRACTION if fraction is None else fraction
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError("in-sample fraction must be in (0, 1]")
    return value


def in_sample_end_timestamp(
    frame: pd.DataFrame,
    fraction: float | None = None,
) -> pd.Timestamp:
    """Return the Date cutoff for the first `fraction` of rows (by timeline order)."""
    if frame is None or len(frame) == 0:
        raise ValueError("Cannot compute in-sample window on an empty frame")
    if "Date" not in frame.columns:
        raise ValueError("Frame must include a Date column")

    ordered = frame.sort_values("Date")
    dates = pd.to_datetime(ordered["Date"])
    frac = in_sample_fraction(fraction)
    if frac >= 1.0:
        return pd.Timestamp(dates.iloc[-1])

    # At least one bar; index of last in-sample row.
    end_index = max(0, min(len(dates) - 1, int(len(dates) * frac) - 1))
    return pd.Timestamp(dates.iloc[end_index])


def slice_frame_to_end(frame: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    if frame is None or len(frame) == 0:
        return frame
    dates = pd.to_datetime(frame["Date"])
    return frame.loc[dates <= pd.Timestamp(end)].copy()


def slice_symbol_data_to_end(symbol_data: dict, end: pd.Timestamp) -> dict:
    return {
        symbol: slice_frame_to_end(frame, end)
        for symbol, frame in symbol_data.items()
    }


def period_bounds(frame: pd.DataFrame) -> tuple[str, str]:
    dates = pd.to_datetime(frame["Date"])
    start = pd.Timestamp(dates.min()).strftime("%Y-%m-%d")
    end = pd.Timestamp(dates.max()).strftime("%Y-%m-%d")
    return start, end


def sample_window_meta(
    full_frame: pd.DataFrame,
    *,
    sample: str,
    fraction: float | None = None,
    in_sample_end: pd.Timestamp | None = None,
) -> dict:
    frac = in_sample_fraction(fraction)
    period_start, period_end = period_bounds(full_frame)
    end = in_sample_end
    if end is None:
        end = in_sample_end_timestamp(full_frame, frac)
    return {
        "sample": sample,
        "in_sample_fraction": frac,
        "in_sample_end": pd.Timestamp(end).strftime("%Y-%m-%d"),
        "period_start": period_start,
        "period_end": period_end,
    }
