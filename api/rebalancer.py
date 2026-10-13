"""Reusable rebalancing ledger.

Given target weights (fractions of current equity per symbol) and prices, ``PortfolioLedger``
trades the existing positions to those weights and keeps the books: cash, position
quantity, invested capital, realized and unrealized P&L, and transaction costs. It has no
idea why a weight was chosen, so any allocation model can drive it.

Accounting conventions
- Average-cost basis. ``realized_pnl`` is gross of costs; ``total_costs`` is tracked
  separately, so net realized P&L is ``realized_pnl - total_costs``.
- Cost is ``cost_bps`` basis points of traded notional, charged on buys and sells, and paid
  from cash. Targets are sized on equity net of that cost, so cash cannot go negative.
- Optional slippage: ``slippage_bps`` moves the execution price against the trade (up for
  buys, down for sells) away from the mark. Quantities are sized at the mark, cash moves at
  the execution price, and equity is always marked at the mark, so slippage shows up as an
  immediate loss. Every ``Fill`` records both prices.
- Cash earns nothing, matching the single-strategy backtester.
- Long only. Weights must be non-negative and sum to at most 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

# A trade smaller than this fraction of equity is noise, not a rebalance.
MIN_TRADE_FRACTION = 1e-9
_COST_ITERATIONS = 6


@dataclass
class Position:
    shares: float = 0.0
    avg_cost: float = 0.0


@dataclass(frozen=True)
class Fill:
    """One executed transaction in one instrument. The authoritative transaction record."""

    symbol: str
    shares: float  # signed: positive buys, negative sells
    price: float  # execution price
    cost: float  # explicit fee
    mark_price: float = 0.0  # market price the order was sized at
    realized_pnl: float = 0.0  # gross of fees, average-cost basis
    units_before: float = 0.0
    units_after: float = 0.0
    avg_cost_after: float = 0.0  # average cost per unit of what remains, 0 when flat
    slippage_cost: float = 0.0  # |shares| * |execution - mark|

    @property
    def notional(self) -> float:
        return abs(self.shares) * self.price


@dataclass(frozen=True)
class RebalanceResult:
    fills: tuple[Fill, ...] = ()

    @property
    def traded_notional(self) -> float:
        return sum(fill.notional for fill in self.fills)

    @property
    def cost(self) -> float:
        return sum(fill.cost for fill in self.fills)

    @property
    def traded(self) -> bool:
        return bool(self.fills)


@dataclass
class PortfolioLedger:
    cash: float
    cost_bps: float = 0.0
    slippage_bps: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    total_costs: float = 0.0
    traded_notional: float = 0.0
    rebalance_count: int = 0

    def __post_init__(self) -> None:
        if self.cost_bps < 0 or not math.isfinite(self.cost_bps):
            raise ValueError("cost_bps must be a non-negative number")
        if self.slippage_bps < 0 or not math.isfinite(self.slippage_bps):
            raise ValueError("slippage_bps must be a non-negative number")

    @property
    def cost_rate(self) -> float:
        return self.cost_bps / 10_000.0

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000.0

    def shares(self, symbol: str) -> float:
        position = self.positions.get(symbol)
        return position.shares if position else 0.0

    def position_value(self, symbol: str, price: float) -> float:
        return self.shares(symbol) * price

    def invested_capital(self, prices: Mapping[str, float]) -> float:
        """Market value of all positions."""
        return sum(
            position.shares * prices[symbol]
            for symbol, position in self.positions.items()
        )

    def equity(self, prices: Mapping[str, float]) -> float:
        return self.cash + self.invested_capital(prices)

    def unrealized_pnl(self, prices: Mapping[str, float]) -> float:
        return sum(
            position.shares * (prices[symbol] - position.avg_cost)
            for symbol, position in self.positions.items()
        )

    def weights(self, prices: Mapping[str, float]) -> dict[str, float]:
        equity = self.equity(prices)
        if equity <= 0:
            return {}
        return {
            symbol: position.shares * prices[symbol] / equity
            for symbol, position in self.positions.items()
        }

    def rebalance(
        self,
        target_weights: Mapping[str, float],
        prices: Mapping[str, float],
    ) -> RebalanceResult:
        """Trade existing positions to ``target_weights`` at ``prices``.

        Symbols that are held but absent from ``target_weights`` are liquidated. Returns the
        fills, which are empty when every position is already at target.
        """
        if any(weight < -1e-12 for weight in target_weights.values()):
            raise ValueError("Target weights must be non-negative")
        if sum(target_weights.values()) > 1.0 + 1e-9:
            raise ValueError("Target weights must sum to at most 1")

        equity = self.equity(prices)
        if equity <= 0:
            return RebalanceResult()

        symbols = sorted(set(self.positions) | set(target_weights))
        current = {symbol: self.position_value(symbol, prices[symbol]) for symbol in symbols}
        threshold = MIN_TRADE_FRACTION * equity

        # Size targets on equity net of the cost of getting there. The cost depends on the
        # trades, which depend on the targets, so settle it by iteration.
        cost = 0.0
        deltas: dict[str, float] = {}
        for _ in range(_COST_ITERATIONS):
            investable = equity - cost
            deltas = {
                symbol: target_weights.get(symbol, 0.0) * investable - current[symbol]
                for symbol in symbols
            }
            deltas = {s: d for s, d in deltas.items() if abs(d) > threshold}
            # Fee on the executed notional plus slippage, both estimated from mark value.
            all_in_rate = self.cost_rate * (1.0 + self.slippage_rate) + self.slippage_rate
            next_cost = all_in_rate * sum(abs(d) for d in deltas.values())
            if abs(next_cost - cost) <= 1e-12 * equity:
                cost = next_cost
                break
            cost = next_cost
        if not deltas:
            return RebalanceResult()

        fills: list[Fill] = []
        # Sells first so the cash they raise is on the books before any buy.
        for symbol, delta in sorted(deltas.items(), key=lambda item: item[1]):
            price = prices[symbol]
            if not math.isfinite(price) or price <= 0:
                continue
            if target_weights.get(symbol, 0.0) <= 0.0:
                # Liquidation sells exactly what is held. Going through delta / price can
                # miss by an ulp and leave dust that looks like an open position.
                units = -self.shares(symbol)
            else:
                units = delta / price
            fills.append(self._fill(symbol, units, price))
        result = RebalanceResult(tuple(fills))
        if result.traded:
            self.rebalance_count += 1
            self.traded_notional += result.traded_notional
        return result

    def _fill(self, symbol: str, shares: float, mark: float) -> Fill:
        slip = self.slippage_rate
        position = self.positions.setdefault(symbol, Position())
        units_before = position.shares
        if shares > 0:
            execution = mark * (1.0 + slip)
            total = position.shares + shares
            position.avg_cost = (position.shares * position.avg_cost + shares * execution) / total
            position.shares = total
            notional = shares * execution
            self.cash -= notional
            realized = 0.0
            signed = shares
        else:
            execution = mark * (1.0 - slip)
            sold = min(-shares, position.shares)
            realized = sold * (execution - position.avg_cost)
            position.shares -= sold
            notional = sold * execution
            self.cash += notional
            self.realized_pnl += realized
            signed = -sold
        fee = notional * self.cost_rate
        units_after = position.shares
        avg_cost_after = position.avg_cost
        # Dust is relative to the size traded: 1 ulp of 100,000 units is far above 1e-12.
        dust = 1e-12 * max(1.0, units_before, abs(shares))
        if units_after <= dust:
            del self.positions[symbol]
            units_after = 0.0
            avg_cost_after = 0.0
        self.cash -= fee
        self.total_costs += fee
        return Fill(
            symbol=symbol,
            shares=signed,
            price=execution,
            cost=fee,
            mark_price=mark,
            realized_pnl=realized,
            units_before=units_before,
            units_after=units_after,
            avg_cost_after=avg_cost_after,
            slippage_cost=abs(signed) * abs(execution - mark),
        )
