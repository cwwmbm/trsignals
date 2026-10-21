"""Ranking metrics for one backtest, without pandas in the loop.

``backtest._ranking_metrics(execute_strategy(frame, ...))`` is what every sweep row, refine outcome and
basket row is made of. It costs about 50 ms: ``long_strat`` runs a Python loop, then
``compute_aggregate_metrics`` and ``yearly_performance`` run several pandas groupby/apply passes.

``ranking_metrics`` returns the same dictionary from arrays: a compiled copy of the ``long_strat`` loop
that also keeps each trade's return, then the metrics in numpy with the same definitions (leading bars
before the first ready entry dropped, best positive year left out, drawdown measured from the full
run's running peak). It takes a few milliseconds.

It returns None whenever it is not certain to agree with the real engine, and callers then run the real
path: no numba, a zero-day ("og") strategy, non-finite or non-positive equity (a blown-up account has
sentinel values), dates out of order, or an input the metric code treats specially. The tests compare it
with the real engine across long/short, hold-on-buy, proxy, leverage vehicle, entry-ready and
best-year-exclusion variants.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import backtest as bt
import stats
from api import fast_sweep, leverage_settings

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = fast_sweep.njit

START_CAPITAL = 15000.0
PERIODS_PER_YEAR = 252


@njit(cache=True, error_model="numpy")
def _simulate_full(
    buy, sell, close, mark, track, use_stream, v_ok, v_use, v_ret, s_ret,
    days, prof, is_long, hold_on_buy, pnl, out, compounded, raw_trade,
):
    """``long_strat``'s loop, keeping equity, exits and each trade's return."""
    n = buy.shape[0]
    prev_hold = False
    prev_out = False
    prev_in = False
    prev_dfe = 0
    prev_pc = 0
    in_vehicle = False
    trade_return = 0.0
    entry = 0.0
    for i in range(n):
        hold = False
        if i > 0:
            hold = (prev_hold and not prev_out) or prev_in
        b = buy[i]
        lin = b and not hold
        if lin:
            in_vehicle = use_stream and v_ok[i]
            trade_return = 0.0
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
        o = would_exit
        if hold_on_buy and b and hold and would_exit:
            o = False
            dfe = 0
            pc = 0
        if lin:
            entry = mark[i]
        elif not hold:
            entry = 0.0
        price = mark[i]
        raw_trade[i] = (price - entry) / entry if hold and entry != 0.0 and not np.isnan(price) else 0.0
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
                trade_return = (1.0 + trade_return) * (1.0 + change) - 1.0
                compounded[i] = trade_return
            else:
                pnl[i] = (1.0 - change) * pnl[i - 1]
        else:
            pnl[i] = pnl[i - 1]
        out[i] = o
        prev_hold = hold
        prev_out = o
        prev_in = lin
        prev_dfe = dfe
        prev_pc = pc


def _mark_prices(close: np.ndarray) -> np.ndarray:
    """``long_strat.mark_price`` for every bar: the close, or the last usable earlier close."""
    valid = ~np.isnan(close) & (close != 0)
    filled = pd.Series(np.where(valid, close, np.nan)).ffill().to_numpy()
    mark = np.where(valid, close, np.r_[np.nan, filled[:-1]])
    mark[0] = close[0]
    return mark


def _year_groups(years: np.ndarray):
    starts = np.flatnonzero(np.r_[True, np.diff(years) != 0])
    ends = np.r_[starts[1:] - 1, len(years) - 1]
    return starts, ends


def ranking_metrics(
    frame: pd.DataFrame,
    days: int,
    profit: int,
    is_long: bool,
    *,
    pnl_column: str | None = None,
    include_yearly: bool = True,
) -> dict | None:
    """``backtest._ranking_metrics`` of the backtest of ``frame``, or None to use the real path."""
    if not fast_sweep.NUMBA_AVAILABLE or days <= 0 or len(frame) < 3:
        return None
    try:
        inputs = fast_sweep.build_sim_inputs(frame, days, profit, is_long, pnl_column=pnl_column)
    except ValueError:
        return None
    n = len(frame)
    close = inputs.close
    pnl = np.empty(n)
    out = np.zeros(n, dtype=np.bool_)
    compounded = np.zeros(n)
    raw_trade = np.zeros(n)
    _simulate_full(
        inputs.base, inputs.sell, close, _mark_prices(close), inputs.track, inputs.use_stream, inputs.v_ok, inputs.v_use,
        inputs.v_ret, inputs.s_ret, inputs.days, inputs.prof, inputs.is_long, inputs.hold_on_buy, pnl, out, compounded, raw_trade,
    )
    if not np.all(np.isfinite(pnl)) or (pnl <= 0).any():
        return None
    general = leverage_settings.general_leverage()
    trade_pnl = compounded if is_long else (-1 * general) * raw_trade

    running_max = np.maximum.accumulate(pnl)
    drawdown = (running_max - pnl) / running_max

    first = 0
    if "EntryReady" in frame.columns:
        ready = frame["EntryReady"].fillna(False).to_numpy(bool)
        if ready.any():
            first = int(np.argmax(ready))
    years_all = pd.to_datetime(frame["Date"]).dt.year.to_numpy()
    if np.any(np.diff(years_all) < 0):
        return None
    years = years_all[first:]
    pnl_p, dd_p = pnl[first:], drawdown[first:]
    out_p, tp_p = out[first:], trade_pnl[first:]
    starts, ends = _year_groups(years)

    exclude_best = bool(stats.ExcludeBestReturnYear)
    returns: dict[int, float] = {}
    for s, e in zip(starts, ends):
        if e - s + 1 < 2 or pnl_p[s] <= 0:
            continue
        returns[int(years[s])] = (pnl_p[e] / pnl_p[s]) - 1
    excluded = None
    if exclude_best and returns:
        best = max(returns, key=returns.get)
        excluded = best if returns[best] > 0 else None
    keep = np.ones(len(pnl_p), dtype=bool) if excluded is None else years != excluded
    if not keep.any():
        return None  # the real code has no Sharpe/MaxDD to give here

    def _cagr() -> float:
        whole = (pnl_p[-1] / pnl_p[0]) ** (1 / (len(pnl_p) / PERIODS_PER_YEAR)) - 1
        if not exclude_best or excluded is None or len(returns) <= 1:
            return whole
        remaining = [r for y, r in returns.items() if y != excluded]
        cumulative = 1.0
        for r in remaining:
            cumulative *= (1 + r)
        return cumulative ** (1 / len(remaining)) - 1

    cagr_percent = round(_cagr() * 100, 2)
    trades = int(out.sum())
    positive = int((trade_pnl[out] > 0).sum())
    pct_positive = (positive / trades * 100) if trades else 0
    max_drawdown = float(dd_p[keep].max())

    sharpe = fast_sweep._sharpe(pnl_p, starts.astype(np.int64), ends.astype(np.int64), exclude_best)
    daily = pnl_p[1:] / pnl_p[:-1] - 1.0
    usable = keep[1:]
    selected = daily[usable]
    if selected.size:
        average = selected.mean()
        downside = math.sqrt(float((np.minimum(selected, 0.0) ** 2).mean()))
        sortino = math.sqrt(PERIODS_PER_YEAR) * (average - 0) / downside if downside != 0 else 0
    else:
        return None

    metrics = {
        "PnL": pnl_p[-1],
        "MaxDD": max_drawdown * 100,
        "Trades": trades,
        "%Pstv": pct_positive,
        "CAGR": str(cagr_percent) + "%",
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
    }
    if include_yearly:
        records = []
        for s, e in zip(starts, ends):
            sl = slice(s, e + 1)
            pnl_percent = ((pnl_p[e] - pnl_p[s]) / pnl_p[s]) * 100
            records.append(
                {
                    "year": int(years[s]),
                    "pnl_percent": float(f"{pnl_percent:.2f}"),
                    "drawdown_percent": float(f"{dd_p[sl].max() * 100:.2f}"),
                    "num_trades": int(out_p[sl].sum()),
                    "positive_trades": int(((tp_p[sl] > 0) & out_p[sl]).sum()),
                }
            )
        metrics["Yearly"] = records
    return metrics


def run_metrics(
    frame: pd.DataFrame,
    days: int,
    profit: int,
    is_long: bool,
    *,
    pnl_column: str | None = None,
    include_yearly: bool = True,
) -> dict:
    """``_ranking_metrics`` of the backtest of ``frame``: the fast version, or the real engine's."""
    fast = ranking_metrics(frame, days, profit, is_long, pnl_column=pnl_column, include_yearly=include_yearly)
    if fast is not None:
        return fast
    executed = bt.execute_strategy(frame, days, profit, is_long, pnl_column=pnl_column)
    return bt._ranking_metrics(executed, include_yearly=include_yearly)
