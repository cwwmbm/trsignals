"""Allocation analytics: exposure, turnover, attribution, and model comparison.

Everything here works on plain arrays so it applies equally to a proportional run
(``AllocationSimulation``) and to the legacy all-in overlay frame (``arrays_from_overlay``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from api.allocation_engine import FLAT_WEIGHT, AllocationSimulation

PERIODS_PER_YEAR = 252
MAX_COMPARISON_POINTS = 1000


@dataclass
class BookArrays:
    """The per-bar facts every analytic needs, independent of how the book was built."""

    held_weight: np.ndarray
    post_weight: np.ndarray
    equity: np.ndarray
    traded_notional: np.ndarray
    cost: np.ndarray
    rebalanced: np.ndarray
    start: int = 0


def arrays_from_simulation(sim: AllocationSimulation) -> BookArrays:
    return BookArrays(
        held_weight=sim.held_weight,
        post_weight=sim.post_weight,
        equity=sim.equity,
        traded_notional=sim.traded_notional,
        cost=sim.cost,
        rebalanced=sim.rebalanced,
        start=sim.start_index,
    )


def arrays_from_overlay(frame: pd.DataFrame, *, start: int = 0) -> BookArrays:
    """Book facts for an all-in overlay frame, which is always 0% or 100% invested.

    The weight carried into a bar is ``HoldLong``. The weight after a bar's close is the
    next bar's ``HoldLong``; on the last bar it is what the entry/exit flags imply.
    """
    hold = frame["HoldLong"].to_numpy(dtype=bool)
    trade_in = frame["LongTradeIn"].to_numpy(dtype=bool)
    trade_out = frame["LongTradeOut"].to_numpy(dtype=bool)
    equity = frame["RollingPnL"].to_numpy(dtype=float)
    post = np.empty(len(hold), dtype=bool)
    post[:-1] = hold[1:]
    post[-1] = (hold[-1] and not trade_out[-1]) or trade_in[-1]
    post_weight = post.astype(float)
    previous = np.concatenate(([0.0], post_weight[:-1]))
    traded = np.abs(post_weight - previous) * equity
    cost = (
        frame["TransactionCost"].to_numpy(dtype=float)
        if "TransactionCost" in frame.columns
        else np.zeros(len(hold))
    )
    return BookArrays(
        held_weight=hold.astype(float),
        post_weight=post_weight,
        equity=equity,
        traded_notional=traded,
        cost=cost,
        rebalanced=traded > 0,
        start=start,
    )


def charge_overlay_costs(frame: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Charge trading costs on an all-in overlay frame, matching the rebalancing ledger.

    The overlay is fully invested or flat, so a cost is paid only when it flips. Entering
    puts ``equity / (1 + c)`` to work (the ledger sizes the buy net of its own cost);
    exiting pays ``c`` on the proceeds. Equity after each flip is scaled accordingly. The
    frame is returned untouched when ``cost_bps`` is zero, so the default path is exact.
    """
    if cost_bps <= 0:
        return frame
    rate = cost_bps / 10_000.0
    book = arrays_from_overlay(frame)
    post = book.post_weight
    previous = np.concatenate(([0.0], post[:-1]))
    n = len(post)
    factor = np.ones(n)
    cost = np.zeros(n)
    running = 1.0
    gross = frame["RollingPnL"].to_numpy(dtype=float)
    for i in range(n):
        pre_trade = gross[i] * running
        if post[i] > previous[i]:
            cost[i] = pre_trade * rate / (1.0 + rate)
            running /= 1.0 + rate
        elif post[i] < previous[i]:
            cost[i] = pre_trade * rate
            running *= 1.0 - rate
        factor[i] = running
    out = frame.copy()
    out["RollingPnL"] = gross * factor
    out["RunningMax"] = out["RollingPnL"].cummax()
    out["Drawdown"] = (out["RunningMax"] - out["RollingPnL"]) / out["RunningMax"]
    out["TransactionCost"] = cost
    return out


def exposure_analytics(book: BookArrays, *, periods_per_year: int = PERIODS_PER_YEAR) -> dict[str, Any]:
    """Average exposure, time invested, turnover, and rebalance counts over the live window."""
    s = book.start
    held = book.held_weight[s:]
    post = book.post_weight[s:]
    equity = book.equity[s:]
    traded = book.traded_notional[s:]
    cost = book.cost[s:]
    rebalanced = book.rebalanced[s:]
    n = len(held)
    if n == 0:
        return _empty_exposure_analytics()

    invested = held > FLAT_WEIGHT
    before = np.concatenate(([0.0], post[:-1]))
    entries = (post > FLAT_WEIGHT) & ~(before > FLAT_WEIGHT)
    exits = (before > FLAT_WEIGHT) & ~(post > FLAT_WEIGHT)
    adjustments = rebalanced & ~entries & ~exits

    years = n / periods_per_year
    mean_equity = float(np.mean(equity)) if equity.size else 0.0
    one_way = float(traded.sum()) / 2.0
    turnover_total = one_way / mean_equity if mean_equity > 0 else None
    return {
        "bars": int(n),
        "average_exposure_pct": float(held.mean() * 100.0),
        "average_exposure_when_invested_pct": (
            float(held[invested].mean() * 100.0) if invested.any() else None
        ),
        "max_exposure_pct": float(max(held.max(), post.max()) * 100.0),
        "time_invested_pct": float(invested.mean() * 100.0),
        "rebalance_count": int(rebalanced.sum()),
        "entry_count": int(entries.sum()),
        "exit_count": int(exits.sum()),
        "adjustment_count": int(adjustments.sum()),
        "total_traded_notional": float(traded.sum()),
        "turnover_total_pct": None if turnover_total is None else turnover_total * 100.0,
        "turnover_annualized_pct": (
            None if turnover_total is None or years <= 0 else turnover_total / years * 100.0
        ),
        "total_transaction_costs": float(cost.sum()),
        "transaction_costs_pct_of_avg_equity": (
            float(cost.sum()) / mean_equity * 100.0 if mean_equity > 0 else None
        ),
    }


def _empty_exposure_analytics() -> dict[str, Any]:
    return {
        "bars": 0,
        "average_exposure_pct": 0.0,
        "average_exposure_when_invested_pct": None,
        "max_exposure_pct": 0.0,
        "time_invested_pct": 0.0,
        "rebalance_count": 0,
        "entry_count": 0,
        "exit_count": 0,
        "adjustment_count": 0,
        "total_traded_notional": 0.0,
        "turnover_total_pct": None,
        "turnover_annualized_pct": None,
        "total_transaction_costs": 0.0,
        "transaction_costs_pct_of_avg_equity": None,
    }


def allocation_attribution(
    model: AllocationSimulation,
    reference: AllocationSimulation,
) -> dict[str, Any]:
    """What the allocation model changed, measured against an all-in book on the same signals.

    Per bar the model earns ``held_weight * basket_return`` where the basket is the same
    instruments the all-in reference holds. So the return gap splits cleanly:

    - allocation effect: ``(w_model - w_reference) * basket_return``, the cost (or gain)
      of holding less than everything on bars the reference is fully in;
    - cost effect: difference in transaction costs as a fraction of equity;
    - other: the remainder, which is basket drift when positions in several instruments
      have drifted apart between rebalances.

    Contributions are sums of daily returns (arithmetic), so they add up and can be bucketed.
    Final equity and CAGR are compounded and reported separately.
    """
    if model.dates.shape != reference.dates.shape:
        raise ValueError("Model and reference must cover the same bars")
    s = model.start_index
    sl = slice(s, None)
    w_model = model.held_weight[sl]
    w_ref = reference.held_weight[sl]
    basket = reference.basket_return[sl]
    allocation = (w_model - w_ref) * basket

    def cost_fraction(sim: AllocationSimulation) -> np.ndarray:
        pre_trade_equity = sim.equity + sim.cost  # equity before this bar's costs
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(pre_trade_equity > 0, sim.cost / pre_trade_equity, 0.0)
        return fraction[sl]

    cost_effect = -(cost_fraction(model) - cost_fraction(reference))
    total_gap = model.gross_return[sl] - reference.gross_return[sl] + cost_effect
    other = total_gap - allocation - cost_effect

    # A bar's return is earned under the state at the previous close, so bucket by that.
    held_under = np.concatenate(([0], model.active_count[:-1]))
    active = held_under[sl]
    eligible = model.eligible_count[sl]
    buckets: list[dict[str, Any]] = []
    for level in sorted(set(active.tolist())):
        mask = active == level
        buckets.append(
            {
                "active_count": int(level),
                "days": int(mask.sum()),
                "average_exposure_pct": float(w_model[mask].mean() * 100.0),
                "allocation_effect_pp": float(allocation[mask].sum() * 100.0),
            }
        )

    start_equity = float(model.equity[s]) if model.equity.size else 0.0
    final_delta = float(model.equity[-1] - reference.equity[-1])
    return {
        "reference": "all_in_same_signals",
        "allocation_effect_pp": float(allocation.sum() * 100.0),
        "cost_effect_pp": float(cost_effect.sum() * 100.0),
        "other_pp": float(other.sum() * 100.0),
        "total_return_gap_pp": float(total_gap.sum() * 100.0),
        "final_equity_delta": final_delta,
        "final_equity_delta_pct": (
            None if start_equity <= 0 else final_delta / start_equity * 100.0
        ),
        "days_below_reference": int((w_model < w_ref - FLAT_WEIGHT).sum()),
        "days_partially_invested": int(
            ((w_model > FLAT_WEIGHT) & (w_model < 1.0 - FLAT_WEIGHT)).sum()
        ),
        "average_eligible_strategies": float(eligible.mean()) if eligible.size else 0.0,
        "by_active_count": buckets,
    }


def rebased_comparison_frame(frame: pd.DataFrame, start: int, start_capital: float) -> pd.DataFrame:
    """Slice from ``start`` and rebase equity so every model begins at the same capital."""
    sub = frame.iloc[start:].copy()
    if "EntryReady" in sub.columns:
        sub = sub.drop(columns=["EntryReady"])
    if sub.empty:
        return sub
    first = float(sub["RollingPnL"].iloc[0])
    if first > 0:
        sub["RollingPnL"] = sub["RollingPnL"] * (start_capital / first)
    sub["RunningMax"] = sub["RollingPnL"].cummax()
    sub["Drawdown"] = (sub["RunningMax"] - sub["RollingPnL"]) / sub["RunningMax"]
    return sub.reset_index(drop=True)


def merged_equity_curves(
    curves: Sequence[tuple[str, pd.DataFrame]],
    *,
    max_points: int = MAX_COMPARISON_POINTS,
) -> list[dict[str, Any]]:
    """One row per date with each model's rebased equity, thinned for charting."""
    if not curves:
        return []
    base = curves[0][1][["Date"]].copy()
    for key, frame in curves:
        base[key] = frame["RollingPnL"].to_numpy(dtype=float)
    if len(base) > max_points:
        base = base.iloc[np.linspace(0, len(base) - 1, max_points, dtype=int)]
    rows = []
    for record in base.to_dict(orient="records"):
        row = {"date": pd.Timestamp(record.pop("Date")).strftime("%Y-%m-%d")}
        row.update({key: float(value) for key, value in record.items()})
        rows.append(row)
    return rows
