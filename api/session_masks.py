from __future__ import annotations

import pandas as pd

RTH_OPEN_MINUTES = 9 * 60 + 30
RTH_CLOSE_MINUTES = 16 * 60
MARKET_TIMEZONE = "America/New_York"


def infer_market_timestamp_timezone(timestamps: pd.Series) -> str:
    """
    Infer how bar timestamps should be interpreted for US session rules.

    Returns ``UTC`` for standard exchange timestamps (UTC instants converted to ET),
    or ``America/New_York`` when wall-clock Eastern times are stored without a proper
  offset (naive timestamps or UTC labels on Eastern clock times).
    """
    ts = pd.to_datetime(timestamps)
    if ts.empty:
        return "UTC"

    if ts.dt.tz is None:
        hours = ts.dt.hour
        if hours.between(8, 17).mean() >= 0.45 and hours.between(13, 21).mean() < 0.35:
            return MARKET_TIMEZONE
        return "UTC"

    tz_name = str(ts.dt.tz)
    if tz_name not in ("UTC", "Etc/UTC"):
        return tz_name

    utc_clock_hours = ts.dt.hour
    eastern = ts.dt.tz_convert(MARKET_TIMEZONE)
    eastern_minutes = eastern.dt.hour * 60 + eastern.dt.minute
    canonical_rth = (
        (eastern_minutes >= RTH_OPEN_MINUTES)
        & (eastern_minutes < RTH_CLOSE_MINUTES)
        & (eastern.dt.weekday < 5)
    ).mean()
    wall_clock_rth = utc_clock_hours.between(9, 16).mean()

    if wall_clock_rth >= 0.35 and wall_clock_rth > canonical_rth + 0.25:
        return MARKET_TIMEZONE
    return "UTC"


def _to_eastern_timestamps(
    dates: pd.Series,
    source_timezone: str | None = None,
) -> pd.Series:
    ts = pd.to_datetime(dates)
    source = source_timezone or "UTC"

    if source == MARKET_TIMEZONE:
        if ts.dt.tz is None:
            return ts.dt.tz_localize(
                MARKET_TIMEZONE,
                ambiguous="infer",
                nonexistent="shift_forward",
            )
        if str(ts.dt.tz) in ("UTC", "Etc/UTC"):
            return ts.dt.tz_localize(None).dt.tz_localize(
                MARKET_TIMEZONE,
                ambiguous="infer",
                nonexistent="shift_forward",
            )
        return ts.dt.tz_convert(MARKET_TIMEZONE)

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert(MARKET_TIMEZONE)


def regular_trading_hours_mask(
    dates: pd.Series,
    *,
    source_timezone: str | None = None,
) -> pd.Series:
    """True for bars whose Eastern timestamp falls in regular session (9:30–16:00 ET)."""
    eastern = _to_eastern_timestamps(dates, source_timezone)
    minutes = eastern.dt.hour * 60 + eastern.dt.minute
    weekday = eastern.dt.weekday < 5
    in_session = (minutes >= RTH_OPEN_MINUTES) & (minutes < RTH_CLOSE_MINUTES)
    return (weekday & in_session).reindex(dates.index, fill_value=False)


def last_rth_bar_mask(
    dates: pd.Series,
    *,
    source_timezone: str | None = None,
) -> pd.Series:
    """True for the last regular-hours bar of each Eastern calendar date."""
    timestamps = _to_eastern_timestamps(dates, source_timezone)
    rth = regular_trading_hours_mask(dates, source_timezone=source_timezone)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session": timestamps.dt.date,
            "rth": rth,
        },
        index=dates.index,
    )
    mask = pd.Series(False, index=dates.index)
    rth_rows = frame[frame["rth"]]
    if rth_rows.empty:
        return mask

    for _, group in rth_rows.groupby("session", sort=False):
        last_index = group["timestamp"].idxmax()
        mask[last_index] = True
    return mask
