"""Fast screening engine and combination search for indicator filters.

A sweep grid point only changes the Buy mask; everything else about the backtest is fixed.
``backtest.execute_strategy`` still re-simulates the whole frame in pandas/Python for each point
(about 22 ms for 25 years), which makes searching pairs or trios of filters impractical.

This module keeps each filter's mask precomputed, ANDs masks together as a combination, and runs a
compiled copy of the ``backtest.long_strat`` loop plus a Sharpe that mirrors ``stats``
(about 0.2 ms per combination, in parallel across candidates). It is a *screen*: it ranks and prunes.
The winners are re-run through the real engine (``run_filters_exact``), so every number shown to the
user comes from ``execute_strategy`` and a drift in the screen can only change which rows surface.

Supported: hold-days strategies (``days > 0``), long or short, with or without a proxy or leverage
vehicle stream, with or without hold-on-buy. Zero-day "original" strategies are not.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import backtest as bt
import config
import stats

try:  # numba is what makes the screen fast; without it the kernels run as plain Python.
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where numba is not installed
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn


START_CAPITAL = 15000.0
PERIODS_PER_YEAR = 252


# ---------------------------------------------------------------------------------------------
# Compiled kernels
# ---------------------------------------------------------------------------------------------


@njit(cache=True, error_model="numpy")
def _simulate(
    buy_of,
    sell,
    close,
    track,
    use_stream,
    v_ok,
    v_use,
    v_ret,
    s_ret,
    days,
    prof,
    is_long,
    hold_on_buy,
    pnl,
):
    """``backtest.long_strat`` reduced to what ranking needs. Fills ``pnl``; returns the trade count.

    ``buy_of`` is the combined entry mask. Trade entry prices and per-trade PnL are not tracked:
    they feed neither Sharpe nor the trade count.
    """
    n = buy_of.shape[0]
    trades = 0
    prev_hold = False
    prev_out = False
    prev_in = False
    prev_dfe = 0
    prev_pc = 0
    in_vehicle = False
    for i in range(n):
        hold = False
        if i > 0:
            hold = (prev_hold and not prev_out) or prev_in
        b = buy_of[i]
        lin = b and not hold
        if lin:
            in_vehicle = use_stream and v_ok[i]
        dfe = 0
        pc = 0
        if hold and i > 0:
            dfe = prev_dfe + 1
            if is_long:
                up = close[i] > close[i - 1]
            else:
                up = close[i] < close[i - 1]
            pc = prev_pc + 1 if up else prev_pc
        would_exit = (sell[i] and hold) or (dfe >= days) or (pc >= prof)
        out = would_exit
        if hold_on_buy and b and hold and would_exit:
            out = False
            dfe = 0
            pc = 0
        if i == 0:
            pnl[i] = START_CAPITAL
        elif hold:
            if use_stream:
                if in_vehicle and v_use[i]:
                    change = v_ret[i]
                else:
                    change = s_ret[i]
            else:
                change = track[i]
            if np.isnan(change):
                change = 0.0
            if is_long:
                pnl[i] = (1.0 + change) * pnl[i - 1]
            else:
                pnl[i] = (1.0 - change) * pnl[i - 1]
        else:
            pnl[i] = pnl[i - 1]
        if out:
            trades += 1
        prev_hold = hold
        prev_out = out
        prev_in = lin
        prev_dfe = dfe
        prev_pc = pc
    return trades


@njit(cache=True, error_model="numpy")
def _sharpe(pnl, ystart, yend, exclude_best):
    """``stats.compute_aggregate_metrics`` Sharpe: the best positive year's returns are left out."""
    best = -1
    best_r = 0.0
    if exclude_best:
        for y in range(ystart.shape[0]):
            s = ystart[y]
            e = yend[y]
            if e - s + 1 < 2 or pnl[s] <= 0:
                continue
            r = pnl[e] / pnl[s] - 1.0
            if r > best_r:
                best = y
                best_r = r
    count = 0
    total = 0.0
    last = -1
    for y in range(ystart.shape[0]):
        if y == best:
            continue
        for i in range(max(ystart[y], 1), yend[y] + 1):
            if pnl[i] == 0.0 or pnl[i - 1] == 0.0:
                continue
            total += pnl[i] / pnl[i - 1] - 1.0
            count += 1
            last = i
    if count < 2:
        return np.nan
    mean = total / count
    sq = 0.0
    for y in range(ystart.shape[0]):
        if y == best:
            continue
        for i in range(max(ystart[y], 1), yend[y] + 1):
            if pnl[i] == 0.0 or pnl[i - 1] == 0.0:
                continue
            d = pnl[i] / pnl[i - 1] - 1.0 - mean
            sq += d * d
    std = math.sqrt(sq / (count - 1))
    if pnl[last] < 0:
        return -100.0
    if std == 0.0:
        return 0.0
    return math.sqrt(PERIODS_PER_YEAR) * mean / std


@njit(cache=True, error_model="numpy")
def _hold_flags(buy, sell, close, days, prof, is_long, hold_on_buy):
    """The bars a position is held, by the same state machine as ``long_strat`` (no equity needed)."""
    n = buy.shape[0]
    hold_out = np.zeros(n, dtype=np.bool_)
    prev_hold = False
    prev_out = False
    prev_in = False
    prev_dfe = 0
    prev_pc = 0
    for i in range(n):
        hold = False
        if i > 0:
            hold = (prev_hold and not prev_out) or prev_in
        b = buy[i]
        lin = b and not hold
        dfe = 0
        pc = 0
        if hold and i > 0:
            dfe = prev_dfe + 1
            if is_long:
                up = close[i] > close[i - 1]
            else:
                up = close[i] < close[i - 1]
            pc = prev_pc + 1 if up else prev_pc
        would_exit = (sell[i] and hold) or (dfe >= days) or (pc >= prof)
        out = would_exit
        if hold_on_buy and b and hold and would_exit:
            out = False
            dfe = 0
            pc = 0
        hold_out[i] = hold
        prev_hold = hold
        prev_out = out
        prev_in = lin
        prev_dfe = dfe
        prev_pc = pc
    return hold_out


@njit(cache=True, parallel=True, error_model="numpy")
def _score_batch(
    combos,
    masks,
    base,
    base_idx,
    min_signals,
    sell,
    close,
    track,
    use_stream,
    v_ok,
    v_use,
    v_ret,
    s_ret,
    days,
    prof,
    is_long,
    hold_on_buy,
    ystart,
    yend,
    exclude_best,
    out_sharpe,
    out_trades,
    out_final,
):
    """Score each row of ``combos`` (indices into ``masks``) as ``base`` AND every listed mask."""
    rows, k = combos.shape
    n = base.shape[0]
    for r in prange(rows):
        signals = 0
        for t in range(base_idx.shape[0]):
            i = base_idx[t]
            ok = True
            for q in range(k):
                if not masks[combos[r, q], i]:
                    ok = False
                    break
            if ok:
                signals += 1
        if signals < min_signals:
            out_sharpe[r] = np.nan
            out_trades[r] = 0
            out_final[r] = np.nan
            continue
        buy = np.zeros(n, dtype=np.bool_)
        for t in range(base_idx.shape[0]):
            i = base_idx[t]
            ok = True
            for q in range(k):
                if not masks[combos[r, q], i]:
                    ok = False
                    break
            buy[i] = ok
        pnl = np.empty(n)
        out_trades[r] = _simulate(
            buy, sell, close, track, use_stream, v_ok, v_use, v_ret, s_ret,
            days, prof, is_long, hold_on_buy, pnl,
        )
        out_sharpe[r] = _sharpe(pnl, ystart, yend, exclude_best)
        out_final[r] = pnl[n - 1]


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Filter:
    """One entry filter: ``column >= value`` ("more") or ``column <= value`` ("less")."""

    column: str
    condition: str
    value: float

    def as_dict(self) -> dict:
        return {"Indicator": self.column, "Condition": self.condition, "Value": self.value}


@dataclass
class SimInputs:
    base: np.ndarray
    base_idx: np.ndarray
    sell: np.ndarray
    close: np.ndarray
    track: np.ndarray
    use_stream: bool
    v_ok: np.ndarray
    v_use: np.ndarray
    v_ret: np.ndarray
    s_ret: np.ndarray
    days: int
    prof: int
    is_long: bool
    hold_on_buy: bool
    ystart: np.ndarray
    yend: np.ndarray
    exclude_best: bool

    def score(self, combos: np.ndarray, masks: np.ndarray, min_signals: int):
        """Sharpe, trade count and final equity for each combination row."""
        rows = combos.shape[0]
        sharpe = np.full(rows, np.nan)
        trades = np.zeros(rows, dtype=np.int64)
        final = np.full(rows, np.nan)
        if rows:
            _score_batch(
                np.ascontiguousarray(combos, dtype=np.int64),
                masks,
                self.base,
                self.base_idx,
                int(min_signals),
                self.sell,
                self.close,
                self.track,
                self.use_stream,
                self.v_ok,
                self.v_use,
                self.v_ret,
                self.s_ret,
                int(self.days),
                int(self.prof),
                bool(self.is_long),
                bool(self.hold_on_buy),
                self.ystart,
                self.yend,
                bool(self.exclude_best),
                sharpe,
                trades,
                final,
            )
        return sharpe, trades, final


def build_sim_inputs(
    frame: pd.DataFrame,
    days: int,
    profit: int,
    is_long: bool,
    *,
    pnl_column: str | None = None,
) -> SimInputs:
    if days <= 0:
        raise ValueError("Combination sweeps need a hold-days limit above zero")
    n = len(frame)
    if n < 3:
        raise ValueError("Not enough bars to sweep")

    base = np.ascontiguousarray(frame["Buy"].values, dtype=np.bool_)
    sell = np.ascontiguousarray(frame["Sell"].values, dtype=np.bool_)
    price_col = pnl_column or "Close"
    close = np.ascontiguousarray(frame[price_col].values, dtype=np.float64)

    tracked = bt.track_change_series(frame, pnl_column)
    if isinstance(tracked, pd.Series):
        track = tracked.fillna(0).to_numpy(dtype=np.float64)
    else:
        track = np.full(n, float(tracked), dtype=np.float64)

    stream = bt.traded_stream(frame, is_long, pnl_column)
    if stream is not None:
        use_stream = True
        v_ok = np.asarray(stream.vehicle_ok, dtype=np.bool_)
        v_ret = np.asarray(stream.vehicle, dtype=np.float64)
        v_use = np.isfinite(v_ret)
        s_ret = np.asarray(stream.synthetic, dtype=np.float64)
    else:
        use_stream = False
        v_ok = np.zeros(n, dtype=np.bool_)
        v_use = np.zeros(n, dtype=np.bool_)
        v_ret = np.zeros(n, dtype=np.float64)
        s_ret = np.zeros(n, dtype=np.float64)

    years = pd.to_datetime(frame["Date"]).dt.year.to_numpy()
    if np.any(np.diff(years) < 0):
        raise ValueError("Frame dates must be in ascending order")
    starts = np.flatnonzero(np.r_[True, np.diff(years) != 0])
    ends = np.r_[starts[1:] - 1, n - 1]

    return SimInputs(
        base=base,
        base_idx=np.flatnonzero(base).astype(np.int64),
        sell=sell,
        close=close,
        track=track,
        use_stream=use_stream,
        v_ok=v_ok,
        v_use=v_use,
        v_ret=v_ret,
        s_ret=s_ret,
        days=int(days),
        prof=int(profit),
        is_long=bool(is_long),
        hold_on_buy=bool(bt.HoldOnBuySignal),
        ystart=starts.astype(np.int64),
        yend=ends.astype(np.int64),
        exclude_best=bool(stats.ExcludeBestReturnYear),
    )


# ---------------------------------------------------------------------------------------------
# Candidate filters
# ---------------------------------------------------------------------------------------------


def threshold_grid(
    frame: pd.DataFrame,
    *,
    check_breadth: bool,
    check_both: bool,
    include_vwap_sweeps: bool = False,
    exclude_columns=None,
    column_ok=None,
) -> list[Filter]:
    """Every buy-side threshold ``indicator_tryout`` would try, as individual filters."""
    from config import ticker
    from indicator_sweep import (
        BREADTH_SWEEPS,
        POST_2003_BREADTH_SWEEPS,
        PRICE_SWEEPS,
        VFI_EXCLUDED_TICKERS,
        VFI_SWEEPS,
        VWAP_SWEEPS,
    )

    groups = []
    if check_breadth:
        groups.append(BREADTH_SWEEPS)
        if frame["Date"].dt.year.iloc[0] >= 2003:
            groups.append(POST_2003_BREADTH_SWEEPS)
    if check_both or not check_breadth:
        groups.append(PRICE_SWEEPS)
        if ticker not in VFI_EXCLUDED_TICKERS:
            groups.append(VFI_SWEEPS)
        if include_vwap_sweeps and "VWAP" in frame.columns and frame["VWAP"].notna().any():
            groups.append(VWAP_SWEEPS)

    excluded = set(exclude_columns or [])
    seen: set[Filter] = set()
    filters: list[Filter] = []
    for specs in groups:
        for spec in specs:
            column, condition = spec[0], spec[2]
            lo, hi, step = spec[3], spec[4], spec[5]
            if column in excluded or spec[1] in excluded or column not in frame.columns:
                continue
            if column_ok is not None and not column_ok(column):
                continue
            for value in np.arange(lo, hi + step, step):
                conditions = ("more", "less") if condition == "both" else (condition,)
                for cond in conditions:
                    # arange leaves float noise (0.30000000000000004, -5.5e-17); a clean value is
                    # what the user sees and what "Add to entry" will round to.
                    item = Filter(column, cond, round(float(value), 10) + 0.0)
                    if item not in seen:
                        seen.add(item)
                        filters.append(item)
    return filters


def filter_mask(frame: pd.DataFrame, item: Filter) -> np.ndarray:
    """The same comparison ``backtest._run_indicator_threshold`` applies (NaN compares False)."""
    values = frame[item.column].to_numpy(dtype=np.float64)
    return (values <= item.value) if item.condition == "less" else (values >= item.value)


@dataclass
class Candidates:
    filters: list[Filter]
    masks: np.ndarray  # (len(filters), n) bool
    compatible: np.ndarray  # (len, len) bool: may these two sit in one combination?
    dropped: dict = field(default_factory=dict)


def build_candidates(
    frame: pd.DataFrame,
    filters: list[Filter],
    base: np.ndarray,
    min_signals: int,
) -> Candidates:
    """Masks for the usable filters. Drops ones that cannot help a combination.

    * a filter that removes none of the base entries is a no-op, adding it changes nothing;
    * one that leaves fewer than ``min_signals`` entries can never reach the trade minimum,
      and adding more filters only removes entries;
    * one that leaves exactly the entries of an earlier filter is a duplicate.
    """
    kept: list[Filter] = []
    rows: list[np.ndarray] = []
    dropped = {"no_op": 0, "too_few_entries": 0, "duplicate": 0}
    base_total = int(base.sum())
    seen: set[bytes] = set()
    for item in filters:
        mask = filter_mask(frame, item)
        entries = mask & base
        count = int(entries.sum())
        if count == base_total:
            dropped["no_op"] += 1
            continue
        if count < min_signals:
            dropped["too_few_entries"] += 1
            continue
        digest = hashlib.md5(np.packbits(entries).tobytes()).digest()
        if digest in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(digest)
        kept.append(item)
        rows.append(mask)

    n = len(frame)
    masks = np.ascontiguousarray(np.vstack(rows)) if rows else np.zeros((0, n), dtype=np.bool_)
    return Candidates(kept, masks, _compatibility(kept), dropped)


def _compatibility(filters: list[Filter]) -> np.ndarray:
    """Two filters on different columns always combine. On one column only a band does.

    A band is a ``more`` and a ``less`` filter with ``more.value < less.value``. Anything else on
    the same column is redundant (same direction) or empty (a crossed band).
    """
    count = len(filters)
    columns = np.array([item.column for item in filters], dtype=object)
    is_more = np.array([item.condition == "more" for item in filters], dtype=bool)
    values = np.array([item.value for item in filters], dtype=np.float64)
    same = columns[:, None] == columns[None, :]
    band = (
        same
        & is_more[:, None]
        & ~is_more[None, :]
        & (values[:, None] < values[None, :])
    )
    band = band | band.T
    compat = ~same | band
    compat[np.arange(count), np.arange(count)] = False
    return compat


# ---------------------------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------------------------


@dataclass
class ComboSearchResult:
    combos: list[tuple[Filter, ...]]
    screen_sharpe: list[float]
    tested: int
    candidates: int
    dropped: dict
    pairs_possible: int = 0


def _signature(masks: np.ndarray, base: np.ndarray, combo: np.ndarray) -> bytes:
    entries = base.copy()
    for index in combo:
        entries &= masks[index]
    return hashlib.md5(np.packbits(entries).tobytes()).digest()


def _top_distinct(
    combos: np.ndarray,
    sharpe: np.ndarray,
    trades: np.ndarray,
    masks: np.ndarray,
    base: np.ndarray,
    min_trades: int,
    limit: int,
) -> list[int]:
    """Row numbers of the best ``limit`` combinations, skipping ones with identical entry days."""
    ok = np.flatnonzero((trades >= min_trades) & np.isfinite(sharpe))
    if ok.size == 0:
        return []
    order = ok[np.argsort(-sharpe[ok], kind="stable")]
    chosen: list[int] = []
    seen: set[bytes] = set()
    for row in order:
        digest = _signature(masks, base, combos[row])
        if digest in seen:
            continue
        seen.add(digest)
        chosen.append(int(row))
        if len(chosen) >= limit:
            break
    return chosen


# Each filter in a combination must remove at least this share of the entries the other filters
# leave. A filter that trims a handful of crash-day entries makes a "pair" that is really the other
# filter alone, already in the single sweep, and such pairs crowded the top of the list.
MIN_FILTER_EFFECT = 0.10


def _effective(entries: np.ndarray, combos: np.ndarray, min_effect: float, chunk: int = 4096) -> np.ndarray:
    """True for combinations where dropping any one filter adds back >= ``min_effect`` more entries.

    ``entries`` is the (filters, base entries) mask restricted to base entry bars.
    """
    ok = np.ones(len(combos), dtype=bool)
    if min_effect <= 0 or len(combos) == 0:
        return ok
    size = combos.shape[1]
    for start in range(0, len(combos), chunk):
        part = combos[start:start + chunk]
        rows = entries[part]  # (chunk, size, base entries)
        together = rows.all(axis=1).sum(axis=1)
        for k in range(size):
            others = np.delete(rows, k, axis=1).all(axis=1).sum(axis=1)
            # Filter k must cut at least min_effect of what the others let through.
            ok[start:start + chunk] &= together <= (1.0 - min_effect) * others
    return ok


def _all_pairs(compatible: np.ndarray) -> np.ndarray:
    first, second = np.nonzero(np.triu(compatible, k=1))
    return np.column_stack([first, second]).astype(np.int64)


def _score_chunked(inputs: SimInputs, combos: np.ndarray, masks, min_signals, chunk=32768):
    sharpe = np.full(len(combos), np.nan)
    trades = np.zeros(len(combos), dtype=np.int64)
    for start in range(0, len(combos), chunk):
        part = slice(start, start + chunk)
        s, t, _ = inputs.score(combos[part], masks, min_signals)
        sharpe[part] = s
        trades[part] = t
    return sharpe, trades


def search_combinations(
    inputs: SimInputs,
    candidates: Candidates,
    *,
    size: int = 2,
    top_n: int = 50,
    beam_width: int = 150,
    min_trades: int = 10,
    min_effect: float = MIN_FILTER_EFFECT,
) -> ComboSearchResult:
    """Best ``size``-filter combinations by screened Sharpe.

    Pairs are searched exhaustively. Trios extend the ``beam_width`` best pairs by every
    compatible third filter, so a trio is only found when some pair inside it already ranks well.
    Combinations where some filter removes less than ``min_effect`` of the entries the others
    leave are skipped (counted in ``dropped["weak_filter"]``).
    """
    if size not in (2, 3):
        raise ValueError("Combination size must be 2 or 3")
    count = len(candidates.filters)
    masks = candidates.masks
    if count < size:
        return ComboSearchResult([], [], 0, count, candidates.dropped)
    min_signals = max(1, int(min_trades))

    entries = np.ascontiguousarray(masks[:, inputs.base])
    dropped = dict(candidates.dropped)
    pairs = _all_pairs(candidates.compatible)
    pairs_possible = len(pairs)
    effective = _effective(entries, pairs, min_effect)
    dropped["weak_filter"] = int((~effective).sum())
    pairs = pairs[effective]
    sharpe, trades = _score_chunked(inputs, pairs, masks, min_signals)
    tested = len(pairs)
    keep_pairs = _top_distinct(pairs, sharpe, trades, masks, inputs.base, min_trades, top_n if size == 2 else beam_width)

    if size == 2:
        picked = keep_pairs[:top_n]
        final_combos, final_sharpe = pairs[picked], sharpe[picked]
    else:
        beam = pairs[keep_pairs]
        expanded: set[tuple[int, int, int]] = set()
        compat = candidates.compatible
        for a, b in beam:
            for c in np.flatnonzero(compat[a] & compat[b]):
                expanded.add(tuple(sorted((int(a), int(b), int(c)))))
        trios = np.array(sorted(expanded), dtype=np.int64).reshape(-1, 3)
        effective = _effective(entries, trios, min_effect)
        dropped["weak_filter"] += int((~effective).sum())
        trios = trios[effective]
        t_sharpe, t_trades = _score_chunked(inputs, trios, masks, min_signals)
        tested += len(trios)
        picked = _top_distinct(trios, t_sharpe, t_trades, masks, inputs.base, min_trades, top_n)
        final_combos, final_sharpe = trios[picked], t_sharpe[picked]

    combos = [tuple(candidates.filters[int(i)] for i in row) for row in final_combos]
    return ComboSearchResult(
        combos=combos,
        screen_sharpe=[float(value) for value in final_sharpe],
        tested=tested,
        candidates=count,
        dropped=dropped,
        pairs_possible=pairs_possible,
    )


# ---------------------------------------------------------------------------------------------
# One-mask scoring
# ---------------------------------------------------------------------------------------------


class Scorer:
    """Score entry (and exit) masks on one frame with the compiled loop.

    The Sharpe and trade count match ``stats.compute_aggregate_metrics`` for the same masks. ``first``
    is the first bar counted: bars before it are dropped, as ``stats.performance_frame`` does when the
    frame carries an ``EntryReady`` column. ``backtest._run_indicator_threshold`` slices that column
    away for hold-day strategies, so the single-indicator sweep scores from bar 0.
    """

    def __init__(self, frame: pd.DataFrame, days: int, profit: int, is_long: bool, *, pnl_column: str | None = None):
        self.inputs = build_sim_inputs(frame, days, profit, is_long, pnl_column=pnl_column)
        self.n = len(frame)
        self._years = pd.to_datetime(frame["Date"]).dt.year.to_numpy()
        self._bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._pnl = np.empty(self.n)
        self.exclude_best = bool(stats.ExcludeBestReturnYear)

    @property
    def base_buy(self) -> np.ndarray:
        return self.inputs.base

    @property
    def base_sell(self) -> np.ndarray:
        return self.inputs.sell

    def _year_bounds(self, first: int):
        hit = self._bounds.get(first)
        if hit is None:
            years = self._years[first:]
            starts = np.flatnonzero(np.r_[True, np.diff(years) != 0]).astype(np.int64)
            ends = np.r_[starts[1:] - 1, len(years) - 1].astype(np.int64)
            hit = self._bounds[first] = (starts, ends)
        return hit

    def score(self, buy: np.ndarray, sell: np.ndarray | None = None, *, first: int = 0) -> tuple[float, int]:
        """(Sharpe, trades) for ``buy`` and ``sell`` (the frame's own exits when omitted)."""
        i = self.inputs
        trades = _simulate(
            np.ascontiguousarray(buy, dtype=np.bool_),
            i.sell if sell is None else np.ascontiguousarray(sell, dtype=np.bool_),
            i.close, i.track, i.use_stream, i.v_ok, i.v_use, i.v_ret, i.s_ret,
            i.days, i.prof, i.is_long, i.hold_on_buy, self._pnl,
        )
        starts, ends = self._year_bounds(first)
        return float(_sharpe(self._pnl[first:], starts, ends, self.exclude_best)), int(trades)


def hold_flags(buy: np.ndarray, sell: np.ndarray, close: np.ndarray, days: int, profit: int, is_long: bool) -> np.ndarray:
    """Held bars for one symbol's own strategy run, honoring the hold-on-buy setting."""
    return _hold_flags(
        np.ascontiguousarray(buy, dtype=np.bool_), np.ascontiguousarray(sell, dtype=np.bool_),
        np.ascontiguousarray(close, dtype=np.float64), int(days), int(profit), bool(is_long), bool(bt.HoldOnBuySignal),
    )


def best_candidates(sharpe: np.ndarray, trades: np.ndarray, *, limit: int = 3, min_trades: int = 0) -> list[int]:
    """Indices of the best ``limit`` scores (highest Sharpe first, ties in input order).

    Candidates with an undefined Sharpe, or fewer than ``min_trades`` trades, are skipped.
    """
    sharpe = np.asarray(sharpe, dtype=float)
    ok = np.flatnonzero(np.isfinite(sharpe) & (np.asarray(trades) >= min_trades))
    order = ok[np.argsort(-sharpe[ok], kind="stable")]
    return [int(i) for i in order[:limit]]


# ---------------------------------------------------------------------------------------------
# Shifted-signal null
# ---------------------------------------------------------------------------------------------


def shift_null(
    frame: pd.DataFrame,
    days: int,
    profit: int,
    is_long: bool,
    *,
    pnl_column: str | None = None,
    shifts: int = 300,
    seed: int = 5,
    min_shift: int = 60,
) -> dict | None:
    """Rank a strategy's Sharpe against the same entry signals placed at the wrong time.

    The entry mask is circularly shifted against the price series: its frequency and clustering
    survive, its link to what the market does next does not. Exits keep their rules. The strategy's
    Sharpe (the app's: best year left out, bars before the first ready entry dropped) is ranked
    against the shifted ones. ``p`` is the share of shifts at least as good, counting the real run, so
    it can never be 0. Returns None when the frame is too short to shift or the strategy never trades.
    """
    inputs = build_sim_inputs(frame, days, profit, is_long, pnl_column=pnl_column)
    n = len(frame)
    if "EntryReady" in frame.columns:
        ready = frame["EntryReady"].fillna(False).to_numpy(bool)
        first = int(np.argmax(ready)) if ready.any() else 0
    else:
        first = 0
    length = n - first
    if length <= 2 * min_shift + 2 or not inputs.base[first:].any():
        return None

    years = pd.to_datetime(frame["Date"]).dt.year.to_numpy()[first:]
    starts = np.flatnonzero(np.r_[True, np.diff(years) != 0]).astype(np.int64)
    ends = np.r_[starts[1:] - 1, length - 1].astype(np.int64)
    exclude_best = bool(stats.ExcludeBestReturnYear)
    pnl = np.empty(n)

    def sharpe_of(buy: np.ndarray) -> float:
        _simulate(
            buy, inputs.sell, inputs.close, inputs.track, inputs.use_stream, inputs.v_ok, inputs.v_use,
            inputs.v_ret, inputs.s_ret, inputs.days, inputs.prof, inputs.is_long, inputs.hold_on_buy, pnl,
        )
        return float(_sharpe(pnl[first:], starts, ends, exclude_best))

    actual = sharpe_of(inputs.base)
    rng = np.random.default_rng(seed)
    null = np.empty(int(shifts))
    for i in range(int(shifts)):
        buy = inputs.base.copy()
        buy[first:] = np.roll(inputs.base[first:], int(rng.integers(min_shift, length - min_shift)))
        null[i] = sharpe_of(buy)
    null = null[np.isfinite(null)]
    if not np.isfinite(actual) or null.size == 0:
        return None
    return {
        "sharpe": actual,
        "null_median": float(np.median(null)),
        "null_p95": float(np.quantile(null, 0.95)),
        "p": float((1 + (null >= actual).sum()) / (1 + null.size)),
        "shifts": int(null.size),
    }


# ---------------------------------------------------------------------------------------------
# Exact rescoring through the real engine
# ---------------------------------------------------------------------------------------------


def _apply_filters(data: pd.DataFrame, filters) -> None:
    for item in filters:
        if item.condition == "less":
            data["Buy"] = data["Buy"] & (data[item.column] <= item.value)
        else:
            data["Buy"] = data["Buy"] & (data[item.column] >= item.value)


def run_filters_exact(
    frame: pd.DataFrame,
    days: int,
    profit: int,
    is_long: bool,
    filters,
    *,
    include_yearly: bool = True,
    pnl_column: str | None = None,
) -> dict:
    """Backtest ``frame`` with every filter ANDed onto Buy, through ``execute_strategy``.

    Mirrors ``backtest._run_indicator_threshold`` for a single filter.
    """
    filters = tuple(filters)
    if days > 0:
        columns = ["Date", "Close", "%Change", "Buy", "Sell", *[item.column for item in filters]]
        if config.UseProxyUnderlying:
            columns.append(config.ProxySymbol)
        if pnl_column:
            columns.append(pnl_column)
        columns = list(dict.fromkeys(columns))
        data = frame.loc[:, [c for c in columns if c in frame.columns]].copy()
    else:
        data = frame.copy()
    _apply_filters(data, filters)
    from api import fast_metrics

    m = fast_metrics.run_metrics(data, days, profit, is_long, pnl_column=pnl_column, include_yearly=include_yearly)
    row = combo_row_labels(filters)
    row.update(
        {
            "PnL": m["PnL"],
            "MaxDD": m["MaxDD"],
            "Trades": m["Trades"],
            "%Pstv": m["%Pstv"],
            "CAGR": m["CAGR"],
            "Sharpe": m["Sharpe"],
            "Sortino": m["Sortino"],
        }
    )
    if include_yearly:
        row["Yearly"] = m["Yearly"]
    return row


def _format_value(value: float) -> str:
    return f"{round(float(value), 4) + 0.0:g}"


def combo_row_labels(filters) -> dict:
    """Display columns for a combination row, plus the machine-readable ``Filters`` list."""
    filters = tuple(filters)
    return {
        "Buysell": "Buy",
        "Indicator": " + ".join(item.column for item in filters),
        "Condition": " + ".join(item.condition for item in filters),
        "Value": " + ".join(_format_value(item.value) for item in filters),
        "Filters": [
            {"Indicator": item.column, "Condition": item.condition, "Value": float(item.value)}
            for item in filters
        ],
    }


def filters_from_row(row: dict) -> tuple[Filter, ...]:
    """Parse the ``Filters`` list of a combination row back into filters."""
    raw = row.get("Filters")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Combination row requires Filters")
    parsed = []
    for entry in raw:
        column = str(entry.get("Indicator") or "")
        condition = str(entry.get("Condition") or "").lower()
        try:
            value = float(entry.get("Value"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid filter Value: {entry.get('Value')}") from exc
        if not column or condition not in {"more", "less"}:
            raise ValueError("Each filter requires Indicator and a more/less Condition")
        parsed.append(Filter(column, condition, value))
    return tuple(parsed)


def combination_sweep(
    frame: pd.DataFrame,
    days: int,
    profit: int,
    is_long: bool,
    *,
    size: int = 2,
    top_n: int = 50,
    beam_width: int = 150,
    min_trades: int | None = None,
    check_breadth: bool = False,
    check_both: bool = False,
    include_vwap_sweeps: bool = False,
    exclude_columns=None,
    column_ok=None,
    pnl_column: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Screen all filter pairs (or trios) on ``frame`` and rescore the best through the real engine.

    Returns ``(rows, info)``: rows ranked by exact Sharpe, with ``Yearly`` attached, and a summary of
    how much was searched.
    """
    min_trades = config.MIN_IN_SAMPLE_TRADES if min_trades is None else int(min_trades)
    inputs = build_sim_inputs(frame, days, profit, is_long, pnl_column=pnl_column)
    grid = threshold_grid(
        frame,
        check_breadth=check_breadth,
        check_both=check_both,
        include_vwap_sweeps=include_vwap_sweeps,
        exclude_columns=exclude_columns,
        column_ok=column_ok,
    )
    candidates = build_candidates(frame, grid, inputs.base, max(1, min_trades))
    found = search_combinations(
        inputs,
        candidates,
        size=size,
        top_n=top_n,
        beam_width=beam_width,
        min_trades=min_trades,
    )
    rows = [
        run_filters_exact(
            frame, days, profit, is_long, combo, include_yearly=True, pnl_column=pnl_column
        )
        for combo in found.combos
    ]
    results = pd.DataFrame(rows)
    if not results.empty:
        yearly = results["Yearly"].tolist()
        labels = results["Filters"].tolist()
        results = bt._format_ranking_results(results.drop(columns=["Yearly", "Filters"]))
        order = results.index
        results["Yearly"] = [yearly[i] for i in order]
        results["Filters"] = [labels[i] for i in order]
        results = results.reset_index(drop=True)
    info = {
        "size": size,
        "grid_points": len(grid),
        "candidates": found.candidates,
        "combinations_tested": found.tested,
        "dropped": found.dropped,
        "search": "exhaustive" if size == 2 else f"beam({beam_width}) over exhaustive pairs",
        "engine": "numba" if NUMBA_AVAILABLE else "python",
    }
    return results, info
