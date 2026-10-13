"""Portfolio simulation driven by an allocation model and a rebalancing ledger.

Three separate pieces cooperate here and none knows the others' internals:

1. ``effective_state``: turns one strategy's entry/hold/exit flags into per-bar
   *active* and *eligible* flags (the effective signal state).
2. ``api.allocation``: a pure model maps ``SignalState`` to a target exposure.
3. ``api.rebalancer``: a ledger trades the book to those target weights.

Timing and price conventions follow the single-strategy backtester exactly. A signal seen on
the close of bar ``i`` is acted on at that close; the position earns bar ``i+1``'s
close-to-close move. Returns are the leveraged ``config.Leverage * pct_change`` series the
backtester already uses, so each symbol trades as a synthetic leveraged instrument whose
price is the cumulative product of those returns. A fully invested book therefore compounds
exactly like the all-in overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from api.allocation import AllocationModel, SignalState
from api.position_episodes import ENTERED, EXITED, SignalChange, TransactionRecord
from api.rebalancer import PortfolioLedger

SYNTHETIC_START_PRICE = 100.0
# A book weight below this is "flat" for flagging entries, exits, and time invested.
FLAT_WEIGHT = 1e-9


def effective_state(
    trade_in: np.ndarray,
    hold: np.ndarray,
    trade_out: np.ndarray,
    *,
    first_eligible: int | None,
    last_available: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar ``(active, eligible)`` for one strategy.

    A strategy is *eligible* from the first bar its entry could be evaluated (after
    warmup) until its data ends. Once eligible it stays in the denominator even if it has
    never entered. It is *active* after bar ``i``'s close when it holds into bar ``i+1``:
    held and not exiting, or entering. A strategy outside its eligible window is never
    active, so missing data cannot hold a position open.
    """
    n = len(hold)
    eligible = np.zeros(n, dtype=bool)
    if first_eligible is not None and last_available is not None and first_eligible <= last_available:
        eligible[max(first_eligible, 0) : min(last_available, n - 1) + 1] = True
    held_next = (np.asarray(hold, dtype=bool) & ~np.asarray(trade_out, dtype=bool)) | np.asarray(
        trade_in, dtype=bool
    )
    return held_next & eligible, eligible


@dataclass
class AllocationSimulation:
    """Everything one run produced, as aligned per-bar arrays plus the model frame."""

    frame: pd.DataFrame
    model: AllocationModel
    dates: pd.DatetimeIndex
    active_count: np.ndarray
    eligible_count: np.ndarray
    target_exposure: np.ndarray  # set only on bars where the state changed, else NaN
    held_weight: np.ndarray  # book weight carried into the bar (earns that bar's move)
    post_weight: np.ndarray  # book weight after the bar's rebalance
    basket_return: np.ndarray  # return per unit invested on the held positions
    gross_return: np.ndarray  # equity change over the bar, before the bar's trading costs
    equity: np.ndarray
    cash: np.ndarray
    invested: np.ndarray
    traded_notional: np.ndarray
    cost: np.ndarray
    rebalanced: np.ndarray
    realized_pnl: np.ndarray
    unrealized_pnl: np.ndarray
    total_costs: np.ndarray
    final_positions: list[dict]
    eligibility: list[dict] = field(default_factory=list)
    # Authoritative executed transactions, oldest first, and what episodes are derived from.
    transactions: list[TransactionRecord] = field(default_factory=list)
    # Per instrument: share of portfolio equity after each bar's trading.
    asset_weight_post: dict[str, np.ndarray] = field(default_factory=dict)
    final_marks: dict[str, float] = field(default_factory=dict)
    start_capital: float = 0.0
    # Bars where a strategy's state changed but nothing was traded.
    signal_changes_without_transaction: int = 0

    @property
    def start_index(self) -> int:
        """First bar any strategy was eligible; metrics before it are not meaningful."""
        ready = self.eligible_count > 0
        return int(np.argmax(ready)) if ready.any() else 0


def _symbol_returns(
    symbol: str,
    track_changes: Mapping[str, pd.Series],
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    series = track_changes.get(symbol)
    if series is None:
        return np.zeros(len(dates), dtype=float)
    series = series[~series.index.duplicated(keep="last")]
    values = series.reindex(dates).to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    # An instrument cannot lose more than everything in a day.
    return np.maximum(values, -1.0)


def _synthetic_prices(returns: np.ndarray) -> np.ndarray:
    prices = np.empty(len(returns), dtype=float)
    prices[0] = SYNTHETIC_START_PRICE
    for i in range(1, len(returns)):
        prices[i] = prices[i - 1] * (1.0 + returns[i])
    return prices


def _signal_changes(
    labels: Sequence[tuple[str, str]],
    prev_active: np.ndarray,
    active: np.ndarray,
) -> list[SignalChange]:
    """Strategies that entered or exited since the previous bar, in strategy order."""
    changes: list[SignalChange] = []
    for k, (strategy_id, name) in enumerate(labels):
        if active[k] and not prev_active[k]:
            changes.append(SignalChange(strategy_id, name, ENTERED))
        elif prev_active[k] and not active[k]:
            changes.append(SignalChange(strategy_id, name, EXITED))
    return changes


def _target_weights(
    exposure: float,
    active_symbols: Sequence[str],
) -> dict[str, float]:
    """Split the exposure across active strategies' instruments, one share per strategy."""
    if exposure <= 0 or not active_symbols:
        return {}
    weights: dict[str, float] = {}
    share = exposure / len(active_symbols)
    for symbol in active_symbols:
        weights[symbol] = weights.get(symbol, 0.0) + share
    return weights


def simulate_allocation(
    *,
    dates: pd.DatetimeIndex,
    active: np.ndarray,
    eligible: np.ndarray,
    pnl_symbols: Sequence[str],
    track_changes: Mapping[str, pd.Series],
    close_series: Mapping[str, pd.Series],
    model: AllocationModel,
    start_capital: float,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    strategy_labels: Sequence[tuple[str, str]] | None = None,
    signal_context: tuple[np.ndarray, np.ndarray] | None = None,
) -> AllocationSimulation:
    """Run the book bar by bar.

    ``strategy_labels`` are ``(id, name)`` per strategy row. ``signal_context`` is an
    ``(active, eligible)`` pair used only to explain transactions (who entered, who exited);
    it defaults to the sizing matrices. All-in replay passes the real strategies here while
    sizing from a single book-level flag.

    ``active`` and ``eligible`` are ``(strategies, bars)`` boolean matrices from
    ``effective_state``; ``pnl_symbols[k]`` is the instrument strategy ``k`` books P&L on.
    The model is consulted, and the ledger rebalanced, only on bars where the effective
    state (who is active, who is eligible) changed. Between changes positions drift.
    """
    dates = pd.DatetimeIndex(dates)
    n = len(dates)
    if n == 0:
        raise ValueError("No overlapping market data for selected strategies")
    active = np.asarray(active, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    if active.shape != eligible.shape or active.shape[1] != n or active.shape[0] != len(pnl_symbols):
        raise ValueError("Signal state must be (strategies, bars) and match pnl_symbols and dates")

    symbols = sorted(set(pnl_symbols))
    price_paths = {
        symbol: _synthetic_prices(_symbol_returns(symbol, track_changes, dates))
        for symbol in symbols
    }
    raw_close = {
        symbol: (
            close_series[symbol][~close_series[symbol].index.duplicated(keep="last")]
            .reindex(dates)
            .ffill()
            .to_numpy(dtype=float)
            if symbol in close_series
            else np.full(n, np.nan)
        )
        for symbol in symbols
    }

    ledger = PortfolioLedger(
        cash=float(start_capital), cost_bps=cost_bps, slippage_bps=slippage_bps
    )
    ctx_active, ctx_eligible = (
        (np.asarray(signal_context[0], dtype=bool), np.asarray(signal_context[1], dtype=bool))
        if signal_context is not None
        else (active, eligible)
    )
    labels = (
        list(strategy_labels)
        if strategy_labels is not None
        else [(f"s{k}", f"Strategy {k + 1}") for k in range(ctx_active.shape[0])]
    )
    if len(labels) != ctx_active.shape[0] or ctx_active.shape[1] != n:
        raise ValueError("Signal context must be (strategies, bars) and match strategy_labels")
    ctx_active_count = ctx_active.sum(axis=0).astype(int)
    ctx_eligible_count = ctx_eligible.sum(axis=0).astype(int)
    transactions: list[TransactionRecord] = []
    asset_weight_post = {symbol: np.zeros(n) for symbol in symbols}
    prev_ctx_active = np.zeros(ctx_active.shape[0], dtype=bool)
    without_transaction = 0

    active_count = active.sum(axis=0).astype(int)
    eligible_count = eligible.sum(axis=0).astype(int)
    target_exposure = np.full(n, np.nan)
    held_weight = np.zeros(n)
    post_weight = np.zeros(n)
    basket_return = np.zeros(n)
    gross_return = np.zeros(n)
    equity = np.zeros(n)
    cash = np.zeros(n)
    invested = np.zeros(n)
    traded_notional = np.zeros(n)
    cost = np.zeros(n)
    rebalanced = np.zeros(n, dtype=bool)
    realized = np.zeros(n)
    unrealized = np.zeros(n)
    total_costs = np.zeros(n)
    close_out = np.zeros(n)

    display_symbol = symbols[0]
    prev_key: bytes | None = None
    prev_invested = 0.0
    prev_equity = float(start_capital)

    for i in range(n):
        prices = {symbol: price_paths[symbol][i] for symbol in symbols}
        # Mark the book to this bar's close. Positions were set at the prior close.
        pre_invested = ledger.invested_capital(prices)
        pre_equity = ledger.cash + pre_invested
        if i > 0:
            held_weight[i] = post_weight[i - 1]
            if prev_invested > 0:
                basket_return[i] = pre_invested / prev_invested - 1.0
            if prev_equity > 0:
                gross_return[i] = pre_equity / prev_equity - 1.0

        changes = _signal_changes(labels, prev_ctx_active, ctx_active[:, i])
        shares_before = {symbol: ledger.shares(symbol) for symbol in symbols}
        cash_before = ledger.cash

        key = active[:, i].tobytes() + eligible[:, i].tobytes()
        if key != prev_key:
            prev_key = key
            state = SignalState(
                active_count=int(active_count[i]),
                eligible_count=int(eligible_count[i]),
            )
            exposure = model.target_exposure(state)
            target_exposure[i] = exposure
            active_symbols = [
                pnl_symbols[k] for k in range(len(pnl_symbols)) if active[k, i]
            ]
            result = ledger.rebalance(_target_weights(exposure, active_symbols), prices)
            if result.traded:
                rebalanced[i] = True
                traded_notional[i] = result.traded_notional
                cost[i] = result.cost
                equity_after = ledger.equity(prices)
                invested_after = ledger.invested_capital(prices)
                for fill in result.fills:
                    symbol = fill.symbol
                    close_now = raw_close[symbol][i]
                    transactions.append(
                        TransactionRecord(
                            bar=i,
                            date=dates[i],
                            symbol=symbol,
                            units_delta=fill.shares,
                            units_before=fill.units_before,
                            units_after=fill.units_after,
                            price=fill.price,
                            mark_price=fill.mark_price,
                            underlying_close=float(close_now) if np.isfinite(close_now) else None,
                            fee=fill.cost,
                            slippage_cost=fill.slippage_cost,
                            realized_pnl=fill.realized_pnl,
                            avg_cost_after=fill.avg_cost_after,
                            cash_before=cash_before,
                            cash_after=ledger.cash,
                            equity_before=pre_equity,
                            equity_after=equity_after,
                            exposure_before=pre_invested / pre_equity if pre_equity > 0 else 0.0,
                            exposure_after=(
                                invested_after / equity_after if equity_after > 0 else 0.0
                            ),
                            asset_exposure_before=(
                                shares_before[symbol] * fill.mark_price / pre_equity
                                if pre_equity > 0
                                else 0.0
                            ),
                            asset_exposure_after=(
                                fill.units_after * fill.mark_price / equity_after
                                if equity_after > 0
                                else 0.0
                            ),
                            active_after=int(ctx_active_count[i]),
                            eligible_after=int(ctx_eligible_count[i]),
                            changes=tuple(changes),
                            active_strategies=tuple(
                                labels[k][1] for k in range(len(labels)) if ctx_active[k, i]
                            ),
                        )
                    )
                if ledger.positions:
                    display_symbol = max(
                        ledger.positions,
                        key=lambda s: ledger.position_value(s, prices[s]),
                    )

        # A strategy flipped but the book did not trade, e.g. the target was already met.
        if changes and not rebalanced[i]:
            without_transaction += 1
        prev_ctx_active = ctx_active[:, i].copy()

        eq = ledger.equity(prices)
        inv = ledger.invested_capital(prices)
        for symbol in symbols:
            asset_weight_post[symbol][i] = (
                ledger.shares(symbol) * prices[symbol] / eq if eq > 0 else 0.0
            )
        equity[i] = eq
        cash[i] = ledger.cash
        invested[i] = inv
        post_weight[i] = inv / eq if eq > 0 else 0.0
        realized[i] = ledger.realized_pnl
        unrealized[i] = ledger.unrealized_pnl(prices)
        total_costs[i] = ledger.total_costs
        shown = raw_close[display_symbol][i]
        close_out[i] = shown if np.isfinite(shown) else (close_out[i - 1] if i > 0 else 0.0)
        prev_invested = inv
        prev_equity = eq

    frame = _build_frame(
        dates=dates,
        close=close_out,
        held_weight=held_weight,
        post_weight=post_weight,
        equity=equity,
        cash=cash,
        invested=invested,
        active_count=active_count,
        eligible_count=eligible_count,
        rebalanced=rebalanced,
        traded_notional=traded_notional,
        cost=cost,
        realized=realized,
        unrealized=unrealized,
        total_costs=total_costs,
        gross_return=gross_return,
        target_exposure=target_exposure,
    )
    last_prices = {symbol: price_paths[symbol][-1] for symbol in symbols}
    final_positions = [
        {
            "symbol": symbol,
            "quantity": position.shares,
            "price": last_prices[symbol],
            "market_value": position.shares * last_prices[symbol],
            "average_cost": position.avg_cost,
            "unrealized_pnl": position.shares * (last_prices[symbol] - position.avg_cost),
        }
        for symbol, position in sorted(ledger.positions.items())
    ]
    return AllocationSimulation(
        frame=frame,
        model=model,
        dates=dates,
        active_count=active_count,
        eligible_count=eligible_count,
        target_exposure=target_exposure,
        held_weight=held_weight,
        post_weight=post_weight,
        basket_return=basket_return,
        gross_return=gross_return,
        equity=equity,
        cash=cash,
        invested=invested,
        traded_notional=traded_notional,
        cost=cost,
        rebalanced=rebalanced,
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        total_costs=total_costs,
        final_positions=final_positions,
        transactions=transactions,
        asset_weight_post=asset_weight_post,
        final_marks=last_prices,
        start_capital=float(start_capital),
        signal_changes_without_transaction=without_transaction,
    )


def _build_frame(
    *,
    dates: pd.DatetimeIndex,
    close: np.ndarray,
    held_weight: np.ndarray,
    post_weight: np.ndarray,
    equity: np.ndarray,
    cash: np.ndarray,
    invested: np.ndarray,
    active_count: np.ndarray,
    eligible_count: np.ndarray,
    rebalanced: np.ndarray,
    traded_notional: np.ndarray,
    cost: np.ndarray,
    realized: np.ndarray,
    unrealized: np.ndarray,
    total_costs: np.ndarray,
    gross_return: np.ndarray,
    target_exposure: np.ndarray,
) -> pd.DataFrame:
    """Overlay-compatible frame: one portfolio "trade" runs from first exposure to flat.

    Partial entries and exits adjust the position inside that trade; they are rebalances,
    not trades. Column names match ``simulate_portfolio_overlay`` so metrics, payloads,
    contribution, and regime scoring work unchanged.
    """
    n = len(dates)
    held = held_weight > FLAT_WEIGHT
    now = post_weight > FLAT_WEIGHT
    before = np.concatenate(([False], now[:-1]))
    trade_in = now & ~before
    trade_out = before & ~now

    trade_entry = np.zeros(n)
    trade_pnl = np.zeros(n)
    days_in_trade = np.zeros(n, dtype=int)
    entry_equity = 0.0
    for i in range(n):
        if trade_in[i]:
            entry_equity = equity[i]
            trade_entry[i] = entry_equity
        elif i > 0:
            trade_entry[i] = trade_entry[i - 1]
        if (held[i] or trade_out[i]) and entry_equity:
            trade_pnl[i] = equity[i] / entry_equity - 1.0
        days_in_trade[i] = days_in_trade[i - 1] + 1 if held[i] and i > 0 else 0

    frame = pd.DataFrame(
        {
            "Date": dates,
            "Close": close,
            "TrackChange": gross_return,
            "LongTradeIn": trade_in,
            "LongTradeOut": trade_out,
            "HoldLong": held,
            "RollingPnL": equity,
            "TradePnL": trade_pnl,
            "TradeEntry": trade_entry,
            "DaysInTrade": days_in_trade,
            "EntryReady": eligible_count > 0,
            # Weight actually held after the bar's rebalance; drifts until the next one.
            "Exposure": post_weight,
            # Weight the model asked for, carried forward from the last state change.
            "TargetExposure": pd.Series(target_exposure).ffill().fillna(0.0).to_numpy(),
            "HeldExposure": held_weight,
            "ActiveCount": active_count,
            "EligibleCount": eligible_count,
            "Cash": cash,
            "InvestedCapital": invested,
            "Rebalanced": rebalanced,
            "TradedNotional": traded_notional,
            "TransactionCost": cost,
            "RealizedPnL": realized,
            "UnrealizedPnL": unrealized,
            "TotalTransactionCosts": total_costs,
        }
    )
    frame["RunningMax"] = frame["RollingPnL"].cummax()
    frame["Drawdown"] = (frame["RunningMax"] - frame["RollingPnL"]) / frame["RunningMax"]
    return frame
