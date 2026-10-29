"""Funded portfolio sleeves. Signal state and allocated capital are independent.

Each branch keeps its cash, model and member budgets. A rejected child remains active
for its parent. Transfers resize holdings at current marks; they never reset signals.
Sleeves plan ownership without executing account trades. Their targets are aggregated
by symbol and executed once through the same ledger used by flat portfolios.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from api.allocation import SignalState, RegimeAdjustedModel
from api.allocation_engine import AllocationSimulation, _build_frame, _signal_changes
from api.instruments import InstrumentBook
from api.position_episodes import TransactionRecord
from api.rebalancer import PortfolioLedger, Position


@dataclass
class Sleeve:
    name: str
    active: np.ndarray
    eligible: np.ndarray
    children: list = field(default_factory=list)
    model: object = None
    decisions: dict = field(default_factory=dict)
    symbol: str | None = None
    ledger: PortfolioLedger | None = None
    cash: float = 0.0
    key: bytes | None = None
    changed: bool = False
    fractions: list = field(default_factory=list)
    target: float = 0.0
    member_active: object = None
    member_eligible: object = None
    symbols: list = field(default_factory=list)
    targets: dict = field(default_factory=dict)
    execution_cost_bps: float = 0.0

    def equity(self, prices):
        if self.ledger is not None:
            return self.ledger.equity(prices)
        return self.cash + sum(child.equity(prices) for child in self.children)

    def evaluate(self, bar):
        for child in self.children:
            child.evaluate(bar)
        if self.ledger is not None and self.model is None:
            key = bytes([bool(self.active[bar])])
            self.target = float(self.active[bar])
            self.targets = {self.symbol: self.target} if self.target else {}
        else:
            active = (self.member_active[:, bar] if self.ledger is not None else
                      np.array([child.active[bar] for child in self.children], dtype=bool))
            eligible = (self.member_eligible[:, bar] if self.ledger is not None else
                        np.array([child.eligible[bar] for child in self.children], dtype=bool))
            key = active.tobytes()
            if getattr(self.model, 'reacts_to_eligibility', True):
                key += eligible.tobytes()
            if key != self.key:
                self.target = self.model.target_exposure(SignalState(
                    int(active.sum()), int(eligible.sum()), self.decisions.get(bar, {}),
                ))
                self.fractions = [self.target / active.sum() if value else 0.0 for value in active]
                if self.ledger is not None:
                    self.targets = {}
                    for symbol, fraction in zip(self.symbols, self.fractions):
                        self.targets[symbol] = self.targets.get(symbol, 0.0) + fraction
        self.changed = key != self.key
        self.key = key

    def withdraw(self, fraction, prices, execute):
        """Redeem the same fraction of virtual cash and holdings at current marks."""
        fraction = min(1.0, max(0.0, fraction))
        if self.ledger is not None:
            weights = self.ledger.weights(prices)
            execute(self, {symbol: weight * (1 - fraction) for symbol, weight in weights.items()})
            amount = fraction * self.ledger.equity(prices)
            self.ledger.cash -= amount
            if abs(self.ledger.cash) < 1e-8:
                self.ledger.cash = 0.0
            return amount
        amount = self.cash * fraction
        for child in self.children:
            redeemed = child.withdraw(fraction, prices, execute)
            self.cash += redeemed
            amount += redeemed
        self.cash -= amount
        return amount

    def apply(self, prices, execute, transfer=0.0):
        before = self.equity(prices)
        resized = transfer > 1e-10 * max(1.0, before)
        if self.ledger is not None:
            targets = self.ledger.weights(prices) if resized and not self.changed and before > 0 else self.targets
            self.ledger.cash += transfer
            if resized or self.changed:
                execute(self, targets)
            return
        fractions = self.fractions
        if resized and not self.changed and before > 0:
            fractions = [child.equity(prices) / before for child in self.children]
        self.cash += transfer
        if self.changed or resized:
            budget = self.equity(prices)
            desired = [budget * fraction for fraction in fractions]
            # Redeem virtual budgets first, then fund additions. Execution happens after planning.
            for child, target in zip(self.children, desired):
                current = child.equity(prices)
                if current > target + 1e-9:
                    self.cash += child.withdraw((current - target) / current, prices, execute)
            increases = [max(0.0, target - child.equity(prices))
                         for child, target in zip(self.children, desired)]
            needed = sum(increases)
            available = min(1.0, max(0.0, self.cash) / needed) if needed else 0.0
            for child, increase in zip(self.children, increases):
                amount = increase * available
                self.cash -= amount
                child.apply(prices, execute, amount)
        else:
            for child in self.children:
                child.apply(prices, execute)


def simulate_nested(strategies, signals_by_id, *, model, cost_bps, global_proxy,
                    track_changes, close_series, dates, regime_context=None, regime_settings=None):
    from api.portfolio_service import _effective_signal_matrices, _strategy_pnl_symbol, START_CAPITAL

    leaves = []
    labels = []
    leaf_active = []
    leaf_eligible = []

    def build(members, signals, allocation_model, costs, proxy, context, settings, name, prefix=''):
        active, eligible, _ = _effective_signal_matrices(members, signals, dates)
        decisions = {}
        if isinstance(allocation_model, RegimeAdjustedModel):
            if context is None:
                raise ValueError('Regime-adjusted portfolio is missing its historical context')
            decisions = context.decisions([member.id for member in members], active, dates, settings)
        if not any(getattr(member, 'nested_spec', None) for member in members):
            sleeve = Sleeve(name, active.any(axis=0), eligible.any(axis=0),
                model=allocation_model, decisions=decisions, ledger=PortfolioLedger(0.0), execution_cost_bps=costs,
                member_active=active, member_eligible=eligible,
                symbols=[_strategy_pnl_symbol(member, proxy) for member in members])
            leaves.append(sleeve)
            for index, member in enumerate(members):
                # Ids follow the member's position in the hierarchy: stable, unique, and a
                # direct strategy of the root keeps its plain id, as in a flat portfolio.
                labels.append((f'{prefix}{member.id}', f'{name} / {member.name}'))
                leaf_active.append(active[index])
                leaf_eligible.append(eligible[index])
            return sleeve
        children = []
        for index, member in enumerate(members):
            spec = getattr(member, 'nested_spec', None)
            path = f'{name} / {member.name}' if name else member.name
            if spec is not None:
                if spec.allocation.is_regime_adjusted and not spec.allocation.regime_context.master_dates.equals(dates):
                    from api.portfolio_service import bind_regime_context
                    spec.allocation = bind_regime_context(spec.allocation, spec.members, spec.signals,
                        global_proxy=spec.proxy, track_changes=track_changes, close_series=close_series,
                        years=spec.years, master_dates=dates)
                child = build(spec.members, spec.signals, spec.allocation.model,
                              spec.allocation.cost_bps, spec.proxy, spec.allocation.regime_context,
                              spec.allocation.regime, path, f'{prefix}{member.id}/')
                # Parent eligibility and activation are based on descendant signals,
                # even if the child's model rejects every one of them.
                child.active = active[index]
                child.eligible = eligible[index]
            else:
                child = Sleeve(path, active[index], eligible[index],
                               symbol=_strategy_pnl_symbol(member, proxy),
                               ledger=PortfolioLedger(0.0), execution_cost_bps=costs,
                               symbols=[_strategy_pnl_symbol(member, proxy)])
                leaves.append(child)
                labels.append((f'{prefix}{member.id}', path))
                leaf_active.append(active[index])
                leaf_eligible.append(eligible[index])
            children.append(child)
        return Sleeve(name, active.any(axis=0), eligible.any(axis=0), children,
                      allocation_model, decisions)

    root = build(strategies, signals_by_id, model, cost_bps, global_proxy,
                 regime_context, regime_settings, '')
    root.cash = float(START_CAPITAL)
    ledger = PortfolioLedger(float(START_CAPITAL), cost_bps=cost_bps)
    n = len(dates)
    # Sleeves and strategies speak in underlyings; the account holds instruments (the leveraged
    # stand-in and, when the leverage settings name one, the real leverage symbol).
    symbols = sorted({symbol for leaf in leaves for symbol in leaf.symbols})
    book = InstrumentBook(symbols, track_changes, close_series, dates)
    instrument_keys = book.keys
    closes = book.display_close
    active_count = np.sum([child.active for child in root.children], axis=0).astype(int)
    eligible_count = np.sum([child.eligible for child in root.children], axis=0).astype(int)
    arrays = {key: np.zeros(n) for key in (
        'held_weight', 'post_weight', 'basket_return', 'gross_return', 'equity', 'cash',
        'invested', 'traded_notional', 'cost', 'realized_pnl', 'unrealized_pnl', 'total_costs')}
    target = np.full(n, np.nan)
    rebalanced = np.zeros(n, dtype=bool)
    weights = {key: np.zeros(n) for key in instrument_keys}
    member_weights = {member.id: np.zeros(n) for member in strategies}
    transactions = []
    previous_active = np.zeros(len(labels), dtype=bool)
    active_matrix = np.asarray(leaf_active)
    # Counts on a transaction describe the strategies named beside them, so they are counted
    # at the same level. The frame's ActiveCount/EligibleCount stay per immediate member.
    leaf_active_count = active_matrix.sum(axis=0).astype(int)
    leaf_eligible_count = np.asarray(leaf_eligible).sum(axis=0).astype(int)
    previous_equity = float(START_CAPITAL)
    previous_invested = 0.0
    without_transaction = 0
    changes = ()

    def cash_of(node):
        return node.ledger.cash if node.ledger is not None else node.cash + sum(cash_of(c) for c in node.children)

    def quantity(symbol):
        return ledger.shares(symbol)

    def invested():
        return ledger.invested_capital(instrument_prices)

    def average_cost(symbol):
        position = ledger.positions.get(symbol)
        return position.avg_cost if position else 0.0

    def scale_cash(node, factor):
        if node.ledger is not None:
            node.ledger.cash *= factor
        else:
            node.cash *= factor
            for child in node.children:
                scale_cash(child, factor)

    for bar in range(n):
        instrument_prices = book.prices_at(bar)
        # A symbol the account holds nothing in may switch instrument; one it holds stays put.
        routed = book.route(bar, quantity)
        # Sleeves value each underlying at the price of the instrument it is held in.
        prices = {symbol: instrument_prices[routed[symbol]] for symbol in symbols}
        before_equity = ledger.equity(instrument_prices)
        before_invested = invested()
        if bar:
            arrays['held_weight'][bar] = arrays['post_weight'][bar - 1]
            arrays['gross_return'][bar] = before_equity / previous_equity - 1 if previous_equity > 0 else 0
            arrays['basket_return'][bar] = before_invested / previous_invested - 1 if previous_invested > 0 else 0
        changes = tuple(_signal_changes(labels, previous_active, active_matrix[:, bar]))

        old_units = [{symbol: leaf.ledger.shares(symbol) for symbol in symbols} for leaf in leaves]
        cash_before = ledger.cash
        quantities_before = {key: quantity(key) for key in instrument_keys}

        root.evaluate(bar)
        if root.changed:
            target[bar] = root.target
        # Virtual reassignments do not realize P&L or incur costs in the account.
        root.apply(prices, lambda leaf, targets: leaf.ledger.rebalance(targets, prices))
        planned = {symbol: sum(leaf.ledger.shares(symbol) for leaf in leaves) for symbol in symbols}
        targets = {routed[symbol]: units * prices[symbol] / before_equity
                   for symbol, units in planned.items()} if before_equity > 0 else {}
        schedules = {}
        for direction in (1, -1):
            rates = {}
            for symbol in symbols:
                amounts = [max(0.0, direction * (leaf.ledger.shares(symbol) - old[symbol]))
                           for leaf, old in zip(leaves, old_units)]
                if not sum(amounts):
                    # Fees can induce a small sale of an otherwise unchanged holding.
                    amounts = [(leaf.ledger.shares(symbol) if direction > 0 else old[symbol])
                               for leaf, old in zip(leaves, old_units)]
                rates[routed[symbol]] = (sum(amount * leaf.execution_cost_bps
                                    for amount, leaf in zip(amounts, leaves)) / sum(amounts)
                                 if sum(amounts) else cost_bps)
            schedules[direction] = rates
        execution = ledger.rebalance(targets, instrument_prices,
            buy_cost_bps=schedules[1], sell_cost_bps=schedules[-1])
        eq = ledger.equity(instrument_prices)
        inv = invested()
        for fill in execution.fills:
            symbol = fill.symbol
            units_before = quantities_before[symbol]
            units_after = quantity(symbol)
            transactions.append(TransactionRecord(
                bar=bar, date=dates[bar], symbol=symbol,
                units_delta=fill.shares, units_before=units_before, units_after=units_after,
                price=fill.price, mark_price=fill.mark_price,
                underlying_close=float(closes[symbol][bar]) if np.isfinite(closes[symbol][bar]) else None,
                fee=fill.cost, slippage_cost=fill.slippage_cost, realized_pnl=fill.realized_pnl,
                avg_cost_after=fill.avg_cost_after, cash_before=cash_before, cash_after=ledger.cash,
                equity_before=before_equity, equity_after=eq,
                exposure_before=before_invested / before_equity if before_equity > 0 else 0,
                exposure_after=inv / eq if eq > 0 else 0,
                asset_exposure_before=units_before * instrument_prices[symbol] / before_equity if before_equity > 0 else 0,
                asset_exposure_after=units_after * instrument_prices[symbol] / eq if eq > 0 else 0,
                active_after=int(leaf_active_count[bar]), eligible_after=int(leaf_eligible_count[bar]),
                changes=changes,
                active_strategies=tuple(labels[k][1] for k in range(len(labels)) if active_matrix[k, bar]),
            ))
        arrays['traded_notional'][bar] = execution.traded_notional
        arrays['cost'][bar] = execution.cost
        rebalanced[bar] = execution.traded

        # Preserve planned ownership and reserved cash, scaled to net-of-cost equity.
        # Only the account ledger owns cost basis and realized P&L. Internal transfers
        # carry that basis instead of resetting it to the current market price.
        scale_cash(root, eq / before_equity if before_equity > 0 else 1.0)
        for symbol in symbols:
            held = routed[symbol]
            for leaf in leaves:
                share = leaf.ledger.shares(symbol) / planned[symbol] if planned[symbol] > 0 else 0.0
                units = quantity(held) * share
                if units > 0:
                    leaf.ledger.positions[symbol] = Position(units, average_cost(held))
                else:
                    leaf.ledger.positions.pop(symbol, None)
        # Absorb floating-point / minimum-order rounding in unassigned parent cash.
        root.cash += ledger.cash - cash_of(root)
        arrays['equity'][bar] = eq
        arrays['cash'][bar] = ledger.cash
        arrays['invested'][bar] = inv
        arrays['post_weight'][bar] = inv / eq if eq > 0 else 0
        arrays['realized_pnl'][bar] = ledger.realized_pnl
        arrays['unrealized_pnl'][bar] = ledger.unrealized_pnl(instrument_prices)
        arrays['total_costs'][bar] = ledger.total_costs
        for key in instrument_keys:
            weights[key][bar] = quantity(key) * instrument_prices[key] / eq if eq > 0 else 0
        for member, child in zip(strategies, root.children):
            member_weights[member.id][bar] = child.equity(prices) / eq if eq > 0 else 0
        if changes and not rebalanced[bar]:
            without_transaction += 1
        previous_active = active_matrix[:, bar].copy()
        previous_equity, previous_invested = eq, inv

    frame = _build_frame(
        dates=dates, close=closes[symbols[0]], active_count=active_count, eligible_count=eligible_count,
        rebalanced=rebalanced, target_exposure=target,
        **{key: arrays[key] for key in ('held_weight', 'post_weight', 'equity', 'cash', 'invested',
                                       'traded_notional', 'cost', 'total_costs', 'gross_return')},
        realized=arrays['realized_pnl'], unrealized=arrays['unrealized_pnl'],
    )
    final_positions = [dict(symbol=key, quantity=quantity(key), price=instrument_prices[key],
                            market_value=quantity(key) * instrument_prices[key], average_cost=average_cost(key),
                            unrealized_pnl=quantity(key) * (instrument_prices[key] - average_cost(key)))
                       for key in instrument_keys if quantity(key) > 0]
    result = AllocationSimulation(
        frame=frame, model=model, dates=dates, active_count=active_count, eligible_count=eligible_count,
        target_exposure=target, rebalanced=rebalanced, final_positions=final_positions,
        transactions=transactions, asset_weight_post=weights, final_marks=instrument_prices,
        start_capital=float(START_CAPITAL), signal_changes_without_transaction=without_transaction,
        instruments=book.metadata(), vehicle_report=book.report(), **arrays,
        last_signal_changes=tuple(changes),
        last_active_names=tuple(labels[k][1] for k in range(len(labels)) if active_matrix[k, -1]),
        strategy_count=len(labels),
        last_close={key: float(closes[key][-1]) for key in instrument_keys if np.isfinite(closes[key][-1])},
    )
    result.eligibility = _effective_signal_matrices(strategies, signals_by_id, dates)[2]
    result.member_weight_post = member_weights
    result.member_details = [
        dict(id=member.id, name=member.name, budget_weight=float(member_weights[member.id][-1]),
             invested_weight=(sleeve.equity(prices) - cash_of(sleeve)) / result.equity[-1]
                 if result.equity[-1] > 0 else 0.0,
             cash_weight=cash_of(sleeve) / result.equity[-1] if result.equity[-1] > 0 else 0.0)
        for member, sleeve in zip(strategies, root.children)
    ]
    return result
