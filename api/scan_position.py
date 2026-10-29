"""What the Scan page says to do with a portfolio today.

The scan used to read three flags (buy, hold, sell) from whether a portfolio's exposure crossed
zero, so moving from 55% to 100%, or from 100% to 55%, looked like "hold". This reads the last bar
of the same engine run instead, and reports the **position**: how much of the book is held going
in, how much after today's decision, what the model is asking for, and the orders that get there.

Words
- ``held_before``: the share of the book held going into today's decision, drifted to today's close
  (the engine rebalances only when a signal changes, so a position moves with price in between).
- ``held_now``: the share held after today's decision. On a day with no trade it is the drifted share.
- ``target``: the share the model asked for at its last change. Between changes it differs from
  ``held_now`` by drift only, and no trade is needed.
- ``action``: Enter, Add, Reduce, Exit, Hold or Flat, from the engine's own trades. A change of
  split between instruments with the same total is a Rebalance. Any change counts, however small.

All weights are fractions of the portfolio's own book (1.0 is the whole of that portfolio), never of
an account, which the app does not know about.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from api.position_episodes import (
    ADD,
    ENTER,
    ENTERED,
    EXIT,
    EXITED,
    FLAT_WEIGHT,
    REDUCE,
    classify_event,
)

HOLD = "Hold"
FLAT = "Flat"
REBALANCE = "Rebalance"
ACTIONS = (ENTER, ADD, REDUCE, EXIT, REBALANCE, HOLD, FLAT)

# Any change in weight above this is a change: the scan keeps tiny ones.
EPSILON = 1e-9


def action_for(before: float, after: float, traded: bool) -> str:
    """The action that takes a book from ``before`` to ``after`` (shares of the book)."""
    if not traded:
        return HOLD if after > FLAT_WEIGHT else FLAT
    if before <= FLAT_WEIGHT and after > FLAT_WEIGHT:
        return ENTER
    if before > FLAT_WEIGHT and after <= FLAT_WEIGHT:
        return EXIT
    if after > before + EPSILON:
        return ADD
    if after < before - EPSILON:
        return REDUCE
    return REBALANCE


def signal_flags(action: str, held_before: float) -> dict[str, bool]:
    """The buy / hold / sell flags the Scan page and older callers still read."""
    return {
        "buy_signal": action in (ENTER, ADD, REBALANCE),
        "hold_long": held_before > FLAT_WEIGHT,
        "sell_signal": action in (EXIT, REDUCE, REBALANCE),
    }


def _number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _names(items: Sequence[str]) -> str:
    return ", ".join(items)


def reason_text(
    action: str,
    *,
    entered: Sequence[str],
    exited: Sequence[str],
    active: int,
    total: int,
) -> str:
    """One line on why the position is what it is today."""
    counts = f"{active} of {total} active" if total else f"{active} active"
    parts = []
    if entered:
        parts.append(f"{_names(entered)} entered")
    if exited:
        parts.append(f"{_names(exited)} exited")
    if parts:
        return f"{'; '.join(parts)} ({counts})"
    if action == FLAT:
        return "None active" if not total else f"None active ({counts})"
    return f"No change ({counts})"


def _position(
    *,
    action: str,
    held_before: float,
    held_now: float,
    target: float,
    names: Sequence[str],
    total: int,
    entered: Sequence[str],
    exited: Sequence[str],
    orders: list[dict],
    holdings: list[dict],
    model_label: str,
    as_of: str | None,
    approximate: str | None,
) -> dict[str, Any]:
    return {
        "action": action,
        "held_before": held_before,
        "held_now": held_now,
        "target": target,
        # Zero on a day with no trade, whatever the price did to the weight.
        "delta": held_now - held_before if action not in (HOLD, FLAT) else 0.0,
        "active": len(names),
        "total": total,
        "active_strategies": list(names),
        "entered": list(entered),
        "exited": list(exited),
        "reason": reason_text(action, entered=entered, exited=exited, active=len(names), total=total),
        "orders": orders,
        "holdings": holdings,
        "model": model_label,
        "as_of": as_of,
        "approximate": approximate,
    }


def _day(value: Any) -> str | None:
    return None if value is None else pd.Timestamp(value).strftime("%Y-%m-%d")


def position_from_simulation(
    simulation,
    *,
    model_label: str,
    approximate: str | None = None,
) -> dict[str, Any]:
    """The position of a book that ran through the allocation engine, from its final bar."""
    last = len(simulation.dates) - 1
    transactions = [t for t in simulation.transactions if t.bar == last]
    held_now = float(simulation.post_weight[last])
    held_before = float(transactions[0].exposure_before) if transactions else held_now
    rebalanced = np.flatnonzero(simulation.rebalanced)
    target = float(simulation.post_weight[rebalanced[-1]]) if len(rebalanced) else held_now
    action = action_for(held_before, held_now, bool(transactions))

    def instrument(key: str) -> dict[str, Any]:
        meta = simulation.instruments.get(key, {})
        return {
            "instrument": key,
            "underlying": meta.get("underlying", key),
            "kind": meta.get("kind", "synthetic"),
            "leverage": meta.get("leverage"),
        }

    orders = []
    for fill in transactions:
        orders.append(
            {
                **instrument(fill.symbol),
                "side": "buy" if fill.units_delta > 0 else "sell",
                "event": classify_event(fill.units_before, fill.units_after),
                "weight_before": float(fill.asset_exposure_before),
                "weight_after": float(fill.asset_exposure_after),
                "price": _number(fill.underlying_close) or _number(fill.mark_price),
            }
        )
    holdings = []
    for key, weights in simulation.asset_weight_post.items():
        weight = float(weights[last])
        if weight > FLAT_WEIGHT:
            holdings.append({**instrument(key), "weight": weight, "price": _number(simulation.last_close.get(key))})
    entered = [c.strategy_name for c in simulation.last_signal_changes if c.change == ENTERED]
    exited = [c.strategy_name for c in simulation.last_signal_changes if c.change == EXITED]
    return _position(
        action=action,
        held_before=held_before,
        held_now=held_now,
        target=target,
        names=simulation.last_active_names,
        total=simulation.strategy_count,
        entered=entered,
        exited=exited,
        orders=orders,
        holdings=holdings,
        model_label=model_label,
        as_of=_day(simulation.dates[last]),
        approximate=approximate,
    )


def position_from_overlay(
    frame: pd.DataFrame,
    *,
    symbol: str,
    leverage: float,
    names: Sequence[str],
    entered: Sequence[str],
    exited: Sequence[str],
    total: int,
    model_label: str,
    approximate: str | None = None,
) -> dict[str, Any]:
    """The position of an all-in book on one plain symbol (the original overlay): 0% or 100%."""
    last = frame.iloc[-1]
    trade_in, trade_out, hold = bool(last["LongTradeIn"]), bool(last["LongTradeOut"]), bool(last["HoldLong"])
    held_before = 1.0 if hold else 0.0
    held_now = 1.0 if (trade_in or (hold and not trade_out)) else 0.0
    traded = trade_in or trade_out
    action = action_for(held_before, held_now, traded)
    price = _number(last.get("Close"))
    meta = {"instrument": symbol, "underlying": symbol, "kind": "synthetic", "leverage": leverage}
    orders = []
    if traded:
        orders.append(
            {
                **meta,
                "side": "buy" if held_now > held_before else "sell",
                "event": ENTER if held_now > held_before else EXIT,
                "weight_before": held_before,
                "weight_after": held_now,
                "price": price,
            }
        )
    holdings = [{**meta, "weight": held_now, "price": price}] if held_now > FLAT_WEIGHT else []
    return _position(
        action=action,
        held_before=held_before,
        held_now=held_now,
        target=held_now,
        names=names,
        total=total,
        entered=entered,
        exited=exited,
        orders=orders,
        holdings=holdings,
        model_label=model_label,
        as_of=_day(last.get("Date")),
        approximate=approximate,
    )


def position_from_flags(buy: bool, hold: bool, sell: bool, *, as_of: str | None = None) -> dict[str, Any]:
    """A single strategy is all-in: 0% or 100%, from its own buy / hold / sell flags."""
    held_before = 1.0 if hold else 0.0
    held_now = 1.0 if (buy or (hold and not sell)) else 0.0
    traded = bool(buy or sell)
    action = action_for(held_before, held_now, traded)
    reason = {
        ENTER: "Entry signal",
        EXIT: "Exit signal",
        HOLD: "In a trade",
        FLAT: "No signal",
    }.get(action, "Entry and exit on the same bar")
    result = _position(
        action=action,
        held_before=held_before,
        held_now=held_now,
        target=held_now,
        names=[],
        total=0,
        entered=[],
        exited=[],
        orders=[],
        holdings=[],
        model_label="Single strategy",
        as_of=as_of,
        approximate=None,
    )
    result["reason"] = reason
    return result
