"""Position episodes and rebalance events, derived from executed transactions.

Three layers, kept apart on purpose:

1. ``TransactionRecord``: one executed fill, written by the simulator (``api.allocation_engine``)
   from the ledger's authoritative ``Fill``. It carries the accounting facts (units, execution
   price, fee, realized P&L, cash, equity) and which strategies entered or exited on that bar.
2. ``classify_event``: Enter / Add / Reduce / Exit, from the position's quantity change alone.
3. ``build_position_episodes``: groups one instrument's transactions into continuous episodes,
   from first entry to full liquidation, and totals them.

Nothing here looks at strategy signals to decide what was traded. Signals only explain *why* a
transaction happened. Quantities, prices, fees, and realized P&L are the ledger's, never
reconstructed from target percentages.

Conventions
- An episode is per instrument: it opens when that instrument's position goes from zero to
  positive and closes when it returns to zero. Partial entries and exits stay inside it.
- Episode P&L = realized P&L (gross, average cost) - fees + unrealized P&L on any remaining
  units at the final mark. For a closed episode the unrealized part is zero. With nothing but
  trading in the account this equals the change in portfolio equity across the episode.
- Episode return = episode P&L / portfolio equity immediately before the first entry. It is a
  contribution to portfolio equity, not the return on deployed capital and not the
  instrument's price return.
- Days counts sessions the position was carried into (so a buy on Monday's close and a sell on
  Wednesday's close is 2), using the simulation calendar. Weekends and holidays are not
  sessions. Average exposure is the session-weighted mean of actual exposure over those
  sessions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ENTER = "Enter"
ADD = "Add"
REDUCE = "Reduce"
EXIT = "Exit"

# Position quantities at or below this are flat. Matches the ledger's dust threshold.
FLAT_UNITS = 1e-12
FLAT_WEIGHT = 1e-9

ENTERED = "entered"
EXITED = "exited"


@dataclass(frozen=True)
class SignalChange:
    """One strategy's state change on the bar a transaction executed."""

    strategy_id: str
    strategy_name: str
    change: str  # ENTERED | EXITED


@dataclass(frozen=True)
class TransactionRecord:
    """One executed fill plus the portfolio state around it.

    Exposure and cash fields are portfolio-level and shared by every fill of the same
    rebalance. Unit, price, and P&L fields are this fill's.
    """

    bar: int
    date: pd.Timestamp
    symbol: str
    units_delta: float  # signed: positive buys
    units_before: float
    units_after: float
    price: float  # execution price
    mark_price: float
    underlying_close: float | None
    fee: float
    slippage_cost: float
    realized_pnl: float  # gross of fee
    avg_cost_after: float
    cash_before: float
    cash_after: float
    equity_before: float
    equity_after: float
    exposure_before: float  # portfolio invested / equity
    exposure_after: float
    asset_exposure_before: float  # this instrument only
    asset_exposure_after: float
    active_after: int
    eligible_after: int
    changes: tuple[SignalChange, ...] = ()
    active_strategies: tuple[str, ...] = ()

    @property
    def trade_value(self) -> float:
        """Signed money moved: positive for buys."""
        return self.units_delta * self.price


def classify_event(units_before: float, units_after: float) -> str:
    """Enter / Add / Reduce / Exit from the position's quantity change alone."""
    was_flat = units_before <= FLAT_UNITS
    is_flat = units_after <= FLAT_UNITS
    if was_flat and not is_flat:
        return ENTER
    if not was_flat and is_flat:
        return EXIT
    if units_after > units_before:
        return ADD
    return REDUCE


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _day(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def event_payload(record: TransactionRecord, event: str) -> dict[str, Any]:
    """JSON-ready rebalance event. Full precision; the UI rounds for display."""
    return {
        "date": _day(record.date),
        "event": event,
        "symbol": record.symbol,
        "signal_changes": [
            {
                "strategy_id": change.strategy_id,
                "strategy_name": change.strategy_name,
                "change": change.change,
            }
            for change in record.changes
        ],
        "active_strategies": list(record.active_strategies),
        "active_signals": record.active_after,
        "eligible_signals": record.eligible_after,
        "exposure_before": record.exposure_before,
        "exposure_after": record.exposure_after,
        "asset_exposure_before": record.asset_exposure_before,
        "asset_exposure_after": record.asset_exposure_after,
        "trade_value": record.trade_value,
        "units_delta": record.units_delta,
        # The price a person reads: the underlying's close on the trade date.
        "underlying_close": _finite(record.underlying_close),
        # The ledger's own price for the simulated leveraged instrument. An accounting input,
        # not a market quote; keep it out of anything meant to be read as a price.
        "simulated_price": record.price,
        "mark_price": record.mark_price,
        "units_before": record.units_before,
        "position_after": record.units_after,
        "position_value_after": record.units_after * record.mark_price,
        "average_cost_after": record.avg_cost_after,
        "unrealized_pnl_after": record.units_after * (record.mark_price - record.avg_cost_after),
        "realized_pnl": record.realized_pnl,
        "fees": record.fee,
        "slippage_cost": record.slippage_cost,
        "cash_before": record.cash_before,
        "cash_after": record.cash_after,
        "equity_before": record.equity_before,
        "equity_after": record.equity_after,
    }


def build_position_episodes(
    transactions: Sequence[TransactionRecord],
    *,
    asset_weight_post: Mapping[str, np.ndarray],
    final_marks: Mapping[str, float],
    last_bar: int,
) -> list[dict[str, Any]]:
    """Group executed transactions into position episodes, oldest first.

    ``asset_weight_post[symbol][i]`` is the instrument's share of portfolio equity after bar
    ``i``'s trading; exposure while holding is read from it. ``final_marks`` prices any units
    still held at the end, and ``last_bar`` is the final bar index of the run.
    """
    episodes: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    open_records: dict[str, list[TransactionRecord]] = {}

    def close(symbol: str, records: list[TransactionRecord], closed: bool) -> None:
        counters[symbol] = counters.get(symbol, 0) + 1
        episodes.append(
            _episode(
                symbol,
                counters[symbol],
                records,
                closed=closed,
                weights=asset_weight_post[symbol],
                final_mark=final_marks.get(symbol, 0.0),
                last_bar=last_bar,
            )
        )

    for record in sorted(transactions, key=lambda item: item.bar):
        symbol = record.symbol
        event = classify_event(record.units_before, record.units_after)
        if event == ENTER:
            if symbol in open_records:
                raise ValueError(f"{symbol}: entry while an episode is already open")
            open_records[symbol] = [record]
        else:
            if symbol not in open_records:
                raise ValueError(f"{symbol}: {event.lower()} with no open episode")
            open_records[symbol].append(record)
        if event == EXIT:
            close(symbol, open_records.pop(symbol), closed=True)

    for symbol, records in open_records.items():
        close(symbol, records, closed=False)

    episodes.sort(key=lambda item: (item["start_bar"], item["symbol"]))
    return episodes


def _episode(
    symbol: str,
    number: int,
    records: list[TransactionRecord],
    *,
    closed: bool,
    weights: np.ndarray,
    final_mark: float,
    last_bar: int,
) -> dict[str, Any]:
    first, last = records[0], records[-1]
    start_bar = first.bar
    end_bar = last.bar if closed else last_bar

    held_bars = range(start_bar + 1, end_bar + 1)
    held_weights = np.array([weights[bar - 1] for bar in held_bars], dtype=float)
    held_weights = held_weights[held_weights > FLAT_WEIGHT]
    span = np.asarray(weights[start_bar : end_bar + 1], dtype=float)
    avg_exposure = float(held_weights.mean()) if held_weights.size else float(weights[start_bar])

    realized = sum(record.realized_pnl for record in records)
    fees = sum(record.fee for record in records)
    slippage = sum(record.slippage_cost for record in records)
    unrealized = (
        0.0 if closed else last.units_after * (final_mark - last.avg_cost_after)
    )
    pnl = realized - fees + unrealized
    equity_before = first.equity_before

    # Running episode P&L at each event: everything the ledger has recorded so far (realized
    # minus fees) plus what the units still held are worth at that event's mark. The last
    # event of a closed episode therefore equals the episode's P&L.
    events = []
    realized_to_date = fees_to_date = 0.0
    for record in records:
        realized_to_date += record.realized_pnl
        fees_to_date += record.fee
        to_date = (
            realized_to_date
            - fees_to_date
            + record.units_after * (record.mark_price - record.avg_cost_after)
        )
        event = event_payload(record, classify_event(record.units_before, record.units_after))
        event["episode_pnl_to_date"] = to_date
        event["episode_return_to_date"] = (
            to_date / first.equity_before if first.equity_before else None
        )
        events.append(event)
    return {
        "id": f"{symbol}-{number}",
        "symbol": symbol,
        "status": "Closed" if closed else "Open",
        "start": _day(first.date),
        "end": _day(last.date) if closed else None,
        "start_bar": start_bar,
        "end_bar": end_bar,
        "days": int(held_weights.size),
        "initial_simulated_price": first.price,
        "final_simulated_price": last.price if closed else None,
        "max_exposure": float(span.max()) if span.size else float(first.asset_exposure_after),
        "avg_exposure": avg_exposure,
        "pnl": pnl,
        # What the table shows as Episode %. The ledger's own figure unless a caller replaces
        # it with a legacy one (all-in keeps the old Trades table's number).
        "return": pnl / equity_before if equity_before else None,
        # Always the ledger's figure: episode P&L / equity just before the first entry.
        "return_on_equity": pnl / equity_before if equity_before else None,
        "equity_before_entry": equity_before,
        "rebalance_count": len(records),
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "fees": fees,
        "slippage_cost": slippage,
        "peak_units": max(record.units_after for record in records),
        "units_end": last.units_after,
        "average_cost_end": last.avg_cost_after,
        "initial_underlying_close": _finite(first.underlying_close),
        "final_underlying_close": _finite(last.underlying_close) if closed else None,
        "events": events,
    }


def reconcile_episodes(
    episodes: Sequence[Mapping[str, Any]],
    *,
    start_equity: float,
    final_equity: float,
    start_cash: float,
    final_cash: float,
    transactions: Sequence[TransactionRecord],
) -> dict[str, Any]:
    """Cross-checks that totals tie out. Differences should be float noise.

    All P&L comes from trading (cash earns nothing and there are no external flows), so every
    episode's P&L added together is the change in portfolio equity, and the money moved by
    the transactions explains the change in cash.
    """
    total_pnl = float(sum(episode["pnl"] for episode in episodes))
    equity_change = final_equity - start_equity
    cash_from_trades = -sum(record.trade_value for record in transactions) - sum(
        record.fee for record in transactions
    )
    event_pnl = sum(
        event["realized_pnl"] - event["fees"]
        for episode in episodes
        for event in episode["events"]
    ) + sum(episode["unrealized_pnl"] for episode in episodes)
    return {
        "episode_pnl_total": total_pnl,
        "equity_change": equity_change,
        "episode_vs_equity_difference": total_pnl - equity_change,
        "event_pnl_total": float(event_pnl),
        "episode_vs_event_difference": total_pnl - float(event_pnl),
        "cash_change": final_cash - start_cash,
        "cash_from_transactions": float(cash_from_trades),
        "cash_difference": (final_cash - start_cash) - float(cash_from_trades),
        "episode_count": len(episodes),
        "transaction_count": len(transactions),
    }
