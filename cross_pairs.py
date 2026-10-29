"""Crosses between two series, generated from one list of pairs.

Each pair (A, B) yields four true/false flags (1 / -1, like the other flags), so the sweeps, the
builder and the scan read them like any other indicator:

* ``CrossUp_A_B``: A crossed above B on this bar.
* ``CrossDown_A_B``: A crossed below B on this bar.
* ``CrossUpPrev_A_B`` / ``CrossDownPrev_A_B``: the cross happened on the previous bar, whatever
  A has done since.

Adding a pair is one line in ``CROSS_PAIRS``; the indicator cache key includes the pair list, so
cached frames rebuild on their own.
"""
from __future__ import annotations

import hashlib
from typing import NamedTuple

import numpy as np
import pandas as pd

# Part of the cache key: bump when the columns made per pair change.
_COLUMN_SCHEME = "flags-today-prev"


class CrossPair(NamedTuple):
    left: str
    right: str


CROSS_PAIRS: list[CrossPair] = [
    CrossPair("Close", "EMA8"),
    CrossPair("Close", "EMA20"),
    CrossPair("Close", "SMA20"),
    CrossPair("Close", "KAMA10"),
    CrossPair("Close", "SMA50"),
    CrossPair("Close", "SMA200"),
    CrossPair("EMA8", "EMA20"),
    CrossPair("EMA20", "SMA20"),
    CrossPair("EMA20", "EMA50"),
    CrossPair("SMA20", "SMA50"),
    CrossPair("SMA50", "SMA200"),
]


def up_column(pair: CrossPair) -> str:
    return f"CrossUp_{pair.left}_{pair.right}"


def down_column(pair: CrossPair) -> str:
    return f"CrossDown_{pair.left}_{pair.right}"


def up_prev_column(pair: CrossPair) -> str:
    return f"CrossUpPrev_{pair.left}_{pair.right}"


def down_prev_column(pair: CrossPair) -> str:
    return f"CrossDownPrev_{pair.left}_{pair.right}"


def pair_columns(pair: CrossPair) -> list[str]:
    return [up_column(pair), down_column(pair), up_prev_column(pair), down_prev_column(pair)]


def cross_pairs_signature() -> str:
    """Short hash of the pair list and column scheme, part of the indicator cache key."""
    text = _COLUMN_SCHEME + ";" + ";".join(f"{p.left}>{p.right}" for p in CROSS_PAIRS)
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def signed_cross_age(spread: pd.Series) -> pd.Series:
    """Days since `spread` last changed sign: +N while it has been positive N days, -N while negative.

    +1 is the day of an upward cross, -1 a downward one; a 0 keeps the side that led before it.
    """
    side = np.sign(spread).replace(0, np.nan).ffill()
    run = (side != side.shift()).cumsum()
    age = side.groupby(run).cumcount() + 1
    cross_age = (age * side).astype(float)
    # The first run started before the data did, so its age is unknown.
    first_run = run == run[side.notna()].min()
    return cross_age.where(side.notna() & ~first_run)


def _flag(event: pd.Series) -> pd.Series:
    return pd.Series(np.where(event.fillna(False).astype(bool), 1, -1), index=event.index)


def cross_columns(data: pd.DataFrame) -> dict[str, pd.Series]:
    """The four flags of every pair whose inputs are in `data`."""
    out: dict[str, pd.Series] = {}
    for pair in CROSS_PAIRS:
        if pair.left not in data.columns or pair.right not in data.columns:
            continue
        age = signed_cross_age(data[pair.left] - data[pair.right])
        up, down = age == 1, age == -1
        out[up_column(pair)] = _flag(up)
        out[down_column(pair)] = _flag(down)
        out[up_prev_column(pair)] = _flag(up.shift(1, fill_value=False))
        out[down_prev_column(pair)] = _flag(down.shift(1, fill_value=False))
    return out


def sweep_specs() -> list[tuple]:
    """Sweep specs in indicator_sweep's (buy_col, sell_col, condition, min, max, step) form.

    Only "is true" is swept: a cross is rare, so "no cross today" keeps nearly every bar as a buy
    filter and sells on nearly every bar as an exit.
    """
    return [(column, column, "more", 0, 0, 1) for pair in CROSS_PAIRS for column in pair_columns(pair)]
