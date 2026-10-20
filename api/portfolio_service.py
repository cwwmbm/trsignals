from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

import backtest as bt
import config
from api.allocation import (
    ALL_IN,
    AllInModel,
    AllocationModel,
    RegimeAdjustedModel,
    PROPORTIONAL_LINEAR,
    TIERED,
    comparison_models,
    proportional_model,
    resolve_allocation_model,
)
from api.allocation_analytics import (
    allocation_attribution,
    arrays_from_overlay,
    arrays_from_simulation,
    charge_overlay_costs,
    exposure_analytics,
    merged_equity_curves,
    rebased_comparison_frame,
)
from api.allocation_engine import AllocationSimulation, effective_state, simulate_allocation
from api.drawdown_risk import COMPARISON_PATHS, drawdown_risk_for_frame
from api.position_episodes import build_position_episodes, reconcile_episodes
from api import leverage_settings, vehicle_data, vehicles
from api.proxy_symbol import proxy_column
from api.regime_allocation import (
    PENALTY_PRESETS,
    RegimeAllocationContext,
    RegimePenaltySettings,
    history_from_frame,
    regime_labels,
)
from api.scan_service import execute_saved_strategy
from api.serializers import detailed_backtest_payload, regime_request_payload
from api.strategy_store import get_strategy_by_id
from stats import (
    calmar_ratio,
    compute_aggregate_metrics,
    time_under_water_percent as frame_time_under_water_percent,
    trades_per_year,
    ulcer_index,
    yearly_returns,
)

PortfolioOverlapMode = str
START_CAPITAL = 15000
# Every portfolio runs with every strategy holding until it exits. Saved portfolios and
# requests may still carry an ``overlap_mode``; it is read and ignored.
PORTFOLIO_OVERLAP_MODE = "hold_until_all_exit"


@dataclass
class StrategySignals:
    strategy_id: str
    name: str
    symbol: str
    dates: pd.DatetimeIndex
    long_trade_in: np.ndarray
    hold_long: np.ndarray
    long_trade_out: np.ndarray
    executed: pd.DataFrame | None = None
    # Bars where an entry could be evaluated. None means the strategy reports no warmup.
    entry_ready: np.ndarray | None = None


def _normalize_proxy(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = str(value).strip().upper()
    return cleaned or None


def _track_change_from_close(close: pd.Series) -> pd.Series:
    return leverage_settings.general_leverage() * close.pct_change()


def _collect_symbols(strategies, global_proxy: str | None) -> set[str]:
    symbols: set[str] = set()
    for strategy in strategies:
        symbols.add(strategy.symbol.strip().upper())
        for item in getattr(strategy, "confirm_symbols", None) or []:
            value = item.strip().upper()
            if value:
                symbols.add(value)
        if not global_proxy:
            saved_proxy = proxy_column(strategy)
            if saved_proxy:
                symbols.add(saved_proxy)
    if global_proxy:
        symbols.add(global_proxy)
    return symbols


def _market_inputs_from_dataset(
    symbol_dataset: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Per symbol: the leveraged stand-in's daily returns and the close.

    A symbol with a leverage-settings entry also gets its leverage symbol's real daily returns and
    closes, under ``vehicles.vehicle_key(symbol, leverage_symbol)``. Where that history is not
    available the entry is simply absent and the stand-in carries every trade.
    """
    settings = leverage_settings.load_settings()
    rates = None
    track_changes: dict[str, pd.Series] = {}
    close_series: dict[str, pd.Series] = {}
    for symbol, data in symbol_dataset.items():
        indexed = data.set_index(pd.to_datetime(data["Date"]))
        close = indexed["Close"]
        close_series[symbol] = close
        entry = settings.for_symbol(symbol)
        if entry is not None and entry.financing_multiple and rates is None:
            rates = vehicle_data.load_rates()
        track_changes[symbol] = vehicles.synthetic_returns(close, entry, settings.general_leverage, rates)
        if entry is None:
            continue
        vehicle_close = vehicle_data.load_close(entry.leverage_symbol)
        if vehicle_close is None or not len(vehicle_close):
            continue
        key = vehicles.vehicle_key(symbol, entry.leverage_symbol)
        returns, _ = vehicles.vehicle_returns(vehicle_close, pd.DatetimeIndex(close.index))
        track_changes[key] = returns
        close_series[key] = vehicles.align_close(vehicle_close, pd.DatetimeIndex(close.index))
    return track_changes, close_series


def _load_portfolio_market_data(
    symbols: set[str],
    *,
    years: int,
    use_cache: bool = True,
    symbol_dataset: dict[str, pd.DataFrame] | None = None,
    bulk_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, dict[str, pd.DataFrame], dict[str, pd.Series], dict[str, pd.Series]]:
    """Load required symbols once and derive PnL inputs from cached or fresh data."""
    if symbol_dataset is not None:
        track_changes, close_series = _market_inputs_from_dataset(symbol_dataset)
        return bulk_data, symbol_dataset, track_changes, close_series

    symbol_list = sorted(symbols)
    symbol_dataset = bt.load_symbol_dataset(symbol_list, years=years, use_cache=use_cache)
    track_changes, close_series = _market_inputs_from_dataset(symbol_dataset)
    return None, symbol_dataset, track_changes, close_series


def _align_strategy_signals(frame: pd.DataFrame, master_dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indexed = frame.set_index(pd.to_datetime(frame["Date"]))
    hold = indexed["HoldLong"].reindex(master_dates).ffill().fillna(False).astype(bool)
    trade_in = indexed["LongTradeIn"].reindex(master_dates).fillna(False).astype(bool)
    trade_out = indexed["LongTradeOut"].reindex(master_dates).fillna(False).astype(bool)
    return (
        trade_in.to_numpy(dtype=bool),
        hold.to_numpy(dtype=bool),
        trade_out.to_numpy(dtype=bool),
    )


def _strategy_pnl_symbol(strategy, global_proxy: str | None) -> str:
    if global_proxy:
        return global_proxy
    saved_proxy = proxy_column(strategy)
    if saved_proxy:
        return saved_proxy
    return strategy.symbol.strip().upper()


def _first_in_list(items: set[str], order: list[str]) -> str | None:
    for strategy_id in order:
        if strategy_id in items:
            return strategy_id
    return next(iter(items), None)


def simulate_portfolio_overlay(
    strategies: list,
    signals_by_id: dict[str, StrategySignals],
    *,
    overlap_mode: PortfolioOverlapMode,
    global_proxy: str | None,
    track_changes: dict[str, pd.Series],
    close_series: dict[str, pd.Series],
    strategy_order: list[str],
    master_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    if not strategies:
        raise ValueError("At least one strategy is required")

    if master_dates is None:
        master_dates = pd.DatetimeIndex(
            sorted(set().union(*(signals.dates for signals in signals_by_id.values())))
        )
    else:
        master_dates = pd.DatetimeIndex(master_dates)
    n = len(master_dates)
    if n == 0:
        raise ValueError("No overlapping market data for selected strategies")

    aligned: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for strategy in strategies:
        signals = signals_by_id[strategy.id]
        aligned[strategy.id] = _align_strategy_signals(
            pd.DataFrame(
                {
                    "Date": signals.dates,
                    "LongTradeIn": signals.long_trade_in,
                    "HoldLong": signals.hold_long,
                    "LongTradeOut": signals.long_trade_out,
                }
            ),
            master_dates,
        )

    pnl_symbol_by_strategy = {
        strategy.id: _strategy_pnl_symbol(strategy, global_proxy)
        for strategy in strategies
    }
    mixed_pnl_symbols = len(set(pnl_symbol_by_strategy.values())) > 1

    leading_strategy_id: str | None = None
    active_strategy_ids: set[str] = set()

    long_in = np.zeros(n, dtype=bool)
    long_out = np.zeros(n, dtype=bool)
    hold_long = np.zeros(n, dtype=bool)
    rolling_pnl = np.zeros(n, dtype=float)
    track_change_out = np.zeros(n, dtype=float)
    close_out = np.zeros(n, dtype=float)
    trade_entry = np.zeros(n, dtype=float)
    trade_entry_rolling = np.zeros(n, dtype=float)
    trade_pnl = np.zeros(n, dtype=float)
    days_in_trade = np.zeros(n, dtype=int)

    for i in range(n):
        date = master_dates[i]
        entries = {
            strategy_id
            for strategy_id in strategy_order
            if aligned[strategy_id][0][i]
        }
        exits = {
            strategy_id
            for strategy_id in strategy_order
            if aligned[strategy_id][2][i]
        }

        if overlap_mode == "first_signal_only":
            if leading_strategy_id:
                trade_in = bool(aligned[leading_strategy_id][0][i])
                trade_out = bool(aligned[leading_strategy_id][2][i])
                returns_hold = bool(aligned[leading_strategy_id][1][i])
            else:
                trade_in = bool(entries)
                trade_out = False
                returns_hold = False
            new_active: set[str] = set()
            contributing: set[str] = set()
        elif overlap_mode == "hold_until_all_exit":
            prev_active = set(active_strategy_ids)
            if not active_strategy_ids:
                new_active = set(entries) if entries else set()
            else:
                new_active = (active_strategy_ids - exits) | entries
            trade_in = bool(entries) and not prev_active
            trade_out = bool(prev_active) and not new_active
            contributing = prev_active | new_active
            returns_hold = bool(contributing) and any(
                aligned[strategy_id][1][i] for strategy_id in contributing
            )
        else:
            raise ValueError(f"Unsupported overlap mode: {overlap_mode}")

        leader_for_bar = leading_strategy_id or (
            _first_in_list(entries, strategy_order) if entries else None
        )

        if global_proxy:
            price_symbol = (
                global_proxy
                if (returns_hold or trade_in or trade_out or leader_for_bar or contributing)
                else None
            )
        elif overlap_mode == "first_signal_only" and leader_for_bar:
            price_symbol = pnl_symbol_by_strategy[leader_for_bar]
        elif overlap_mode == "hold_until_all_exit" and contributing:
            hold_ids = [
                strategy_id
                for strategy_id in strategy_order
                if strategy_id in contributing and aligned[strategy_id][1][i]
            ]
            pick = hold_ids[0] if hold_ids else _first_in_list(contributing, strategy_order)
            price_symbol = pnl_symbol_by_strategy[pick] if pick else None
        else:
            price_symbol = None

        current_pnl_symbol = price_symbol if returns_hold else None

        hold_long[i] = returns_hold
        long_in[i] = trade_in
        long_out[i] = trade_out

        if price_symbol:
            close_series_for_symbol = close_series.get(price_symbol)
            price = (
                float(close_series_for_symbol.get(date, np.nan))
                if close_series_for_symbol is not None
                else np.nan
            )
            if not np.isnan(price):
                close_out[i] = price
            elif i > 0:
                close_out[i] = close_out[i - 1]
        elif i > 0:
            close_out[i] = close_out[i - 1]

        if returns_hold and current_pnl_symbol:
            series = track_changes.get(current_pnl_symbol)
            change = float(series.get(date, 0.0)) if series is not None and date in series.index else 0.0
            if np.isnan(change):
                change = 0.0
            track_change_out[i] = change
        elif i > 0:
            track_change_out[i] = 0.0

        if i == 0:
            rolling_pnl[i] = START_CAPITAL
        elif returns_hold:
            rolling_pnl[i] = (1 + track_change_out[i]) * rolling_pnl[i - 1]
        else:
            rolling_pnl[i] = rolling_pnl[i - 1]

        # A trade reports the equity it earned (its leveraged daily returns compounded). The
        # entry column shows the instrument's price when there is one instrument, else equity.
        if long_in[i]:
            trade_entry_rolling[i] = rolling_pnl[i]
            trade_entry[i] = rolling_pnl[i] if mixed_pnl_symbols else close_out[i]
        elif i > 0:
            trade_entry[i] = trade_entry[i - 1]
            trade_entry_rolling[i] = trade_entry_rolling[i - 1]

        mark_to_market = returns_hold or trade_out
        entry_basis = trade_entry_rolling[i]
        if mark_to_market and entry_basis:
            trade_pnl[i] = rolling_pnl[i] / entry_basis - 1
        else:
            trade_pnl[i] = 0.0

        days_in_trade[i] = days_in_trade[i - 1] + 1 if returns_hold and i > 0 else 0

        if overlap_mode == "first_signal_only":
            if leading_strategy_id is None:
                if entries:
                    leading_strategy_id = _first_in_list(entries, strategy_order)
            elif leading_strategy_id in exits:
                leading_strategy_id = None
        elif overlap_mode == "hold_until_all_exit":
            active_strategy_ids = new_active

    frame = pd.DataFrame(
        {
            "Date": master_dates,
            "Close": close_out,
            "TrackChange": track_change_out,
            "LongTradeIn": long_in,
            "LongTradeOut": long_out,
            "HoldLong": hold_long,
            "RollingPnL": rolling_pnl,
            "TradePnL": trade_pnl,
            "TradeEntry": trade_entry,
            "DaysInTrade": days_in_trade,
        }
    )
    frame["RunningMax"] = frame["RollingPnL"].cummax()
    frame["Drawdown"] = (frame["RunningMax"] - frame["RollingPnL"]) / frame["RunningMax"]
    return frame


@dataclass(frozen=True)
class AllocationSettings:
    """Which allocation model sizes the book, and what trading it costs.

    ``regime`` holds the user's penalty choices when the model is regime-adjusted.
    ``regime_context`` is built from the loaded strategies (``bind_regime_context``) and is what
    actually scores the regimes; a settings object without one cannot run that model.
    """

    model: AllocationModel
    cost_bps: float = 0.0
    regime: RegimePenaltySettings | None = None
    regime_context: RegimeAllocationContext | None = field(default=None, compare=False, repr=False)

    @property
    def is_all_in(self) -> bool:
        return self.model.id == ALL_IN

    @property
    def is_regime_adjusted(self) -> bool:
        return isinstance(self.model, RegimeAdjustedModel)


def allocation_settings(source) -> AllocationSettings:
    """Read allocation fields off a request or saved portfolio. Missing means legacy all-in."""
    model = resolve_allocation_model(
        getattr(source, "allocation_model", None),
        getattr(source, "proportional_mode", None),
        getattr(source, "tier_single_pct", None),
    )
    cost = getattr(source, "transaction_cost_bps", 0.0)
    regime = RegimePenaltySettings.from_source(source) if isinstance(model, RegimeAdjustedModel) else None
    return AllocationSettings(model=model, cost_bps=float(cost or 0.0), regime=regime)


def bind_regime_context(
    allocation: AllocationSettings,
    strategies: list,
    signals_by_id: dict[str, "StrategySignals"],
    *,
    global_proxy: str | None,
    track_changes: dict[str, pd.Series],
    close_series: dict[str, pd.Series],
    years: int,
    master_dates: pd.DatetimeIndex | None = None,
) -> AllocationSettings:
    """Give regime-adjusted settings the scorer they need. Other models pass through.

    Needs each strategy's own backtest (``signals.executed``), so load them with
    ``keep_executed=True``. The context is tied to the full portfolio's calendar; every
    smaller portfolio built from it (leave-one-out, Shapley coalitions) reuses its scores.
    """
    if not allocation.is_regime_adjusted or allocation.regime is None:
        return allocation
    from api.market_regimes import load_regime_calendar

    master_dates = _master_dates_for(signals_by_id, master_dates)
    ids = [strategy.id for strategy in strategies]
    active, _, _ = _effective_signal_matrices(strategies, signals_by_id, master_dates)
    calendar = load_regime_calendar(years)
    current = regime_labels(master_dates, calendar)
    by_id = {strategy.id: strategy for strategy in strategies}

    def strategy_history(strategy_id: str):
        executed = signals_by_id[strategy_id].executed
        if executed is None:
            raise ValueError(f"Missing backtest for strategy: {by_id[strategy_id].name}")
        return history_from_frame(executed, regime_labels(executed["Date"], calendar))

    def baseline_history(member_ids: tuple[str, ...]):
        # The unadjusted all-in portfolio of these strategies: it must not depend on the
        # allocation being decided, or the decision would feed on itself.
        frame = simulate_portfolio_frame(
            [by_id[item] for item in member_ids],
            {item: signals_by_id[item] for item in member_ids},
            allocation=None,
            overlap_mode=PORTFOLIO_OVERLAP_MODE,
            global_proxy=global_proxy,
            track_changes=track_changes,
            close_series=close_series,
            strategy_order=list(member_ids),
            master_dates=master_dates,
        )
        return history_from_frame(frame, current)

    context = RegimeAllocationContext(
        allocation.regime,
        strategy_ids=ids,
        master_dates=master_dates,
        full_active=active,
        strategy_history=strategy_history,
        baseline_history=baseline_history,
        current_labels=current,
    )
    return replace(allocation, regime_context=context)


def _master_dates_for(
    signals_by_id: dict[str, StrategySignals],
    master_dates: pd.DatetimeIndex | None,
) -> pd.DatetimeIndex:
    if master_dates is not None:
        return pd.DatetimeIndex(master_dates)
    return pd.DatetimeIndex(
        sorted(set().union(*(signals.dates for signals in signals_by_id.values())))
    )


def _strategy_availability(
    strategy,
    signals: StrategySignals,
    master_dates: pd.DatetimeIndex,
) -> tuple[int | None, int | None, dict[str, Any]]:
    """Where on the master calendar a strategy's signals can be evaluated.

    Eligible from the first bar its entry conditions had every input (after indicator
    warmup, instrument listing, regime labels), until its own data ends. Strategies with
    no ``EntryReady`` column are eligible from their first bar. Mid-history gaps in
    readiness are not removed from the denominator; they are counted and reported.
    """
    n = len(master_dates)
    dates = pd.DatetimeIndex(signals.dates)
    report: dict[str, Any] = {
        "strategy_id": strategy.id,
        "strategy_name": strategy.name,
        "first_eligible_date": None,
        "last_available_date": None,
        "warmup_bars": n,
        "eligible_bars": 0,
        "unavailable_bars": 0,
    }
    if len(dates) == 0 or n == 0:
        return None, None, report

    ready_date = dates.min()
    unavailable = 0
    if signals.entry_ready is not None:
        ready = np.asarray(signals.entry_ready, dtype=bool)
        if not ready.any():
            return None, None, report
        first_ready = int(np.argmax(ready))
        ready_date = dates[first_ready]
        unavailable = int((~ready[first_ready:]).sum())

    first_idx = int(master_dates.searchsorted(ready_date, side="left"))
    last_idx = int(master_dates.searchsorted(dates.max(), side="right")) - 1
    if first_idx >= n or last_idx < first_idx:
        return None, None, report

    report.update(
        first_eligible_date=master_dates[first_idx].strftime("%Y-%m-%d"),
        last_available_date=master_dates[last_idx].strftime("%Y-%m-%d"),
        warmup_bars=first_idx,
        eligible_bars=last_idx - first_idx + 1,
        unavailable_bars=unavailable,
    )
    return first_idx, last_idx, report


def simulate_portfolio_allocation(
    strategies: list,
    signals_by_id: dict[str, StrategySignals],
    *,
    model: AllocationModel,
    cost_bps: float = 0.0,
    global_proxy: str | None,
    track_changes: dict[str, pd.Series],
    close_series: dict[str, pd.Series],
    master_dates: pd.DatetimeIndex | None = None,
    regime_context: RegimeAllocationContext | None = None,
    regime_settings: RegimePenaltySettings | None = None,
) -> AllocationSimulation:
    """Size the book from how many strategies are active, using ``model``.

    Each strategy keeps its own entry and exit timing; overlap modes do not apply because
    every strategy's state counts toward the exposure. The regime-adjusted model also needs
    ``regime_context`` (see ``bind_regime_context``), and the penalties to apply as
    ``regime_settings`` (default: the context's own).
    """
    if not strategies:
        raise ValueError("At least one strategy is required")
    master_dates = _master_dates_for(signals_by_id, master_dates)
    if len(master_dates) == 0:
        raise ValueError("No overlapping market data for selected strategies")
    if any(getattr(member, "nested_spec", None) is not None for member in strategies):
        from api.nested_allocation import simulate_nested
        return simulate_nested(strategies, signals_by_id, model=model, cost_bps=cost_bps,
            global_proxy=global_proxy, track_changes=track_changes, close_series=close_series,
            dates=master_dates, regime_context=regime_context, regime_settings=regime_settings)
    active, eligible, reports = _effective_signal_matrices(strategies, signals_by_id, master_dates)

    decisions = None
    if isinstance(model, RegimeAdjustedModel):
        if regime_context is None:
            raise ValueError("Regime-adjusted allocation needs its regime context")
        decisions = regime_context.decisions(
            [strategy.id for strategy in strategies], active, master_dates, regime_settings
        )

    simulation = simulate_allocation(
        dates=master_dates,
        active=active,
        eligible=eligible,
        pnl_symbols=[_strategy_pnl_symbol(strategy, global_proxy) for strategy in strategies],
        track_changes=track_changes,
        close_series=close_series,
        model=model,
        start_capital=START_CAPITAL,
        cost_bps=cost_bps,
        strategy_labels=[(strategy.id, strategy.name) for strategy in strategies],
        decisions=decisions,
    )
    simulation.eligibility = reports
    return simulation


def _effective_signal_matrices(
    strategies: list,
    signals_by_id: dict[str, StrategySignals],
    master_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """``(active, eligible, reports)``: per-strategy effective state on the master calendar."""
    n = len(master_dates)
    active = np.zeros((len(strategies), n), dtype=bool)
    eligible = np.zeros((len(strategies), n), dtype=bool)
    reports: list[dict[str, Any]] = []
    for k, strategy in enumerate(strategies):
        signals = signals_by_id[strategy.id]
        trade_in, hold, trade_out = _aligned_signal_arrays(signals, master_dates)
        first, last, report = _strategy_availability(strategy, signals, master_dates)
        active[k], eligible[k] = effective_state(
            trade_in,
            hold,
            trade_out,
            first_eligible=first,
            last_available=last,
        )
        reports.append(report)
    return active, eligible, reports


def trading_symbols(strategies: list, global_proxy: str | None) -> list[str]:
    """The instruments the portfolio trades: each strategy's proxy (or the portfolio's), else
    its own symbol. This, not the signal symbol, is what decides whether a book is one symbol."""
    symbols = set()
    for strategy in strategies:
        spec = getattr(strategy, "nested_spec", None)
        if spec is not None:
            symbols.update(trading_symbols(spec.members, spec.proxy))
        else:
            symbols.add(_strategy_pnl_symbol(strategy, global_proxy))
    return sorted(symbols)


def signal_symbols(strategies: list) -> list[str]:
    """Symbols the strategies' signals are read from, looking through child portfolios.

    A child portfolio contributes its own members' symbols, never its placeholder name.
    """
    symbols: set[str] = set()
    for strategy in strategies:
        spec = getattr(strategy, "nested_spec", None)
        if spec is not None:
            symbols.update(signal_symbols(spec.members))
        else:
            symbols.add(strategy.symbol.strip().upper())
    return sorted(symbols)


def is_single_symbol(strategies: list, global_proxy: str | None) -> bool:
    return not any(getattr(s, "nested_spec", None) for s in strategies) and len(trading_symbols(strategies, global_proxy)) == 1


def uses_overlay(strategies: list, global_proxy: str | None) -> bool:
    """Whether All-in can run on the original overlay.

    The overlay follows one plain return stream, so it only fits a book that trades a single
    symbol with no leverage-settings entry. A symbol with a leverage symbol (SOXX/SOXL) needs
    the engine, which holds the right instrument and switches only between positions.
    """
    symbols = trading_symbols(strategies, global_proxy)
    return (
        is_single_symbol(strategies, global_proxy)
        and leverage_settings.symbol_setting(symbols[0]) is None
    )


def _overlay_book_label(strategies: list, global_proxy: str | None) -> str:
    """Name of the instrument an all-in book holds: its one symbol, else the whole portfolio."""
    symbols = trading_symbols(strategies, global_proxy)
    return symbols[0] if len(symbols) == 1 else "Portfolio"


def replay_overlay_transactions(
    frame: pd.DataFrame,
    strategies: list,
    signals_by_id: dict[str, StrategySignals],
    *,
    global_proxy: str | None,
    cost_bps: float = 0.0,
) -> AllocationSimulation:
    """Run an all-in overlay frame's positions through the same engine and ledger.

    The overlay stays the source of the all-in equity curve, so its results are unchanged.
    Replaying its in/out flags through ``simulate_allocation`` gives all-in the same
    transaction records and episodes as every other allocation model, from one accounting
    implementation. The book trades as one instrument whose returns are the overlay's own
    ``TrackChange``, which is the held symbol's leveraged return on every bar it is held.
    Strategies enter only as context, so rows can say who entered and who exited.
    """
    dates = pd.DatetimeIndex(frame["Date"])
    active_ctx, eligible_ctx, _ = _effective_signal_matrices(strategies, signals_by_id, dates)
    in_position = (arrays_from_overlay(frame).post_weight > 0.5)[None, :]
    label = _overlay_book_label(strategies, global_proxy)
    close = pd.Series(frame["Close"].to_numpy(dtype=float), index=dates)
    return simulate_allocation(
        dates=dates,
        active=in_position,
        eligible=np.ones_like(in_position),
        pnl_symbols=[label],
        track_changes={label: pd.Series(frame["TrackChange"].to_numpy(dtype=float), index=dates)},
        close_series={label: close.where(close > 0)},
        model=AllInModel(),
        start_capital=START_CAPITAL,
        cost_bps=cost_bps,
        strategy_labels=[(strategy.id, strategy.name) for strategy in strategies],
        signal_context=(active_ctx, eligible_ctx),
    )


def _costed_overlay(
    strategies: list,
    signals_by_id: dict[str, StrategySignals],
    *,
    cost_bps: float,
    **kwargs,
) -> pd.DataFrame:
    """The legacy overlay, with trading costs charged only when a cost is set."""
    frame = simulate_portfolio_overlay(strategies, signals_by_id, **kwargs)
    return charge_overlay_costs(frame, cost_bps)


def simulate_portfolio_frame(
    strategies: list,
    signals_by_id: dict[str, StrategySignals],
    *,
    allocation: AllocationSettings | None,
    overlap_mode: PortfolioOverlapMode,
    global_proxy: str | None,
    track_changes: dict[str, pd.Series],
    close_series: dict[str, pd.Series],
    strategy_order: list[str],
    master_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """The portfolio frame for either allocation path.

    All-in on one plain trading symbol is the original overlay. All-in on several symbols, or
    on a symbol with a leverage symbol, runs through the engine (100% invested, split by active
    strategies).
    """
    if allocation is None or allocation.is_all_in:
        cost_bps = 0.0 if allocation is None else allocation.cost_bps
        if not uses_overlay(strategies, global_proxy):
            return simulate_portfolio_allocation(
                strategies,
                signals_by_id,
                model=AllInModel(),
                cost_bps=cost_bps,
                global_proxy=global_proxy,
                track_changes=track_changes,
                close_series=close_series,
                master_dates=master_dates,
            ).frame
        return _costed_overlay(
            strategies,
            signals_by_id,
            cost_bps=cost_bps,
            overlap_mode=overlap_mode,
            global_proxy=global_proxy,
            track_changes=track_changes,
            close_series=close_series,
            strategy_order=strategy_order,
            master_dates=master_dates,
        )
    return simulate_portfolio_allocation(
        strategies,
        signals_by_id,
        model=allocation.model,
        cost_bps=allocation.cost_bps,
        global_proxy=global_proxy,
        track_changes=track_changes,
        close_series=close_series,
        master_dates=master_dates,
        regime_context=allocation.regime_context,
    ).frame


def _aligned_signal_arrays(
    signals: StrategySignals,
    master_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _align_strategy_signals(
        pd.DataFrame(
            {
                "Date": signals.dates,
                "LongTradeIn": signals.long_trade_in,
                "HoldLong": signals.hold_long,
                "LongTradeOut": signals.long_trade_out,
            }
        ),
        master_dates,
    )


def _load_strategy_signals(
    strategy,
    *,
    years: int,
    bulk_data: pd.DataFrame | None,
    symbol_dataset: dict[str, pd.DataFrame],
    use_cache: bool = True,
    keep_executed: bool = False,
) -> StrategySignals:
    symbol = strategy.symbol.strip().upper()
    if symbol not in symbol_dataset:
        raise ValueError(f"No market data for symbol: {symbol}")
    data = symbol_dataset[symbol].copy()
    executed = execute_saved_strategy(
        data,
        strategy,
        years=years,
        bulk_data=bulk_data,
        symbol_data=symbol_dataset,
        use_cache=use_cache,
    )
    dates = pd.to_datetime(executed["Date"])
    return StrategySignals(
        strategy_id=strategy.id,
        name=strategy.name,
        symbol=symbol,
        dates=dates,
        long_trade_in=executed["LongTradeIn"].to_numpy(dtype=bool),
        hold_long=executed["HoldLong"].to_numpy(dtype=bool),
        long_trade_out=executed["LongTradeOut"].to_numpy(dtype=bool),
        executed=executed if keep_executed else None,
        entry_ready=(
            executed["EntryReady"].fillna(False).to_numpy(dtype=bool)
            if "EntryReady" in executed.columns
            else None
        ),
    )


def _resolve_portfolio_strategies(strategy_ids: list[str]) -> list:
    if not strategy_ids:
        raise ValueError("At least one strategy is required")

    strategies = []
    for strategy_id in strategy_ids:
        strategy = get_strategy_by_id(strategy_id)
        if strategy is None:
            raise ValueError(f"Unknown strategy: {strategy_id}")
        if strategy.direction != "long":
            raise ValueError(
                f"Short strategies are not supported in portfolio simulation: {strategy.name}"
            )
        strategies.append(strategy)
    return strategies


def _portfolio_description(
    strategies: list,
    overlap_mode: str,
    global_proxy: str | None,
    allocation: AllocationSettings | None = None,
) -> str:
    if allocation is not None and not allocation.is_all_in:
        mode_label = (
            allocation.model.label.lower()
            if allocation.model.id == TIERED
            else allocation.model.id.replace("_", " ")
        )
        if allocation.regime is not None:
            mode_label += f", {allocation.regime.preset} penalties, {allocation.regime.score} score"
    else:
        mode_label = (
            "first signal only"
            if overlap_mode == "first_signal_only"
            else "hold until all exit"
        )
    names = ", ".join(strategy.name for strategy in strategies)
    proxy_note = f", proxy {global_proxy}" if global_proxy else ""
    member_label = "members" if any(getattr(s, "nested_spec", None) for s in strategies) else "strategies"
    return f"Portfolio ({len(strategies)} {member_label}, {mode_label}{proxy_note}): {names}"


def _prepare_portfolio_simulation(
    portfolio,
    *,
    years: int = 1,
    use_cache: bool = True,
    symbol_dataset: dict[str, pd.DataFrame] | None = None,
    bulk_data: pd.DataFrame | None = None,
    keep_executed: bool = False,
) -> tuple[list, list[str], dict[str, StrategySignals], str, str | None, dict[str, pd.Series], dict[str, pd.Series]]:
    if getattr(portfolio, "portfolio_ids", None):
        from api.nested_portfolios import prepare_nested
        return prepare_nested(portfolio, years=years, use_cache=use_cache,
                              symbol_dataset=symbol_dataset, bulk_data=bulk_data)
    # Regime-adjusted sizing scores each strategy's own backtest, so it has to be kept.
    keep_executed = keep_executed or allocation_settings(portfolio).is_regime_adjusted
    strategy_ids = list(portfolio.strategy_ids)
    strategies = _resolve_portfolio_strategies(strategy_ids)
    overlap_mode = PORTFOLIO_OVERLAP_MODE
    global_proxy = _normalize_proxy(getattr(portfolio, "proxy_symbol", None))

    symbols = _collect_symbols(strategies, global_proxy)
    bulk_data, symbol_dataset, track_changes, close_series = _load_portfolio_market_data(
        symbols,
        years=years,
        use_cache=use_cache,
        symbol_dataset=symbol_dataset,
        bulk_data=bulk_data,
    )

    signals_by_id: dict[str, StrategySignals] = {}
    for strategy in strategies:
        signals_by_id[strategy.id] = _load_strategy_signals(
            strategy,
            years=years,
            bulk_data=bulk_data,
            symbol_dataset=symbol_dataset,
            use_cache=use_cache,
            keep_executed=keep_executed,
        )

    return (
        strategies,
        strategy_ids,
        signals_by_id,
        overlap_mode,
        global_proxy,
        track_changes,
        close_series,
    )


def build_portfolio_overlay_frame(
    portfolio,
    *,
    years: int = 1,
    use_cache: bool = True,
    symbol_dataset: dict[str, pd.DataFrame] | None = None,
    bulk_data: pd.DataFrame | None = None,
    signals_only: bool = False,
) -> tuple[pd.DataFrame, list, str, int, int]:
    """The portfolio's frame, strategies, description, and its largest hold and profit settings.

    ``signals_only`` is for callers that read nothing but today's buy/hold/sell flags (the
    scan). A regime-adjusted portfolio is then sized like a linear one, which is in the market
    exactly when any strategy is; its regime scores need history the scan does not load.
    """
    (
        strategies,
        strategy_ids,
        signals_by_id,
        overlap_mode,
        global_proxy,
        track_changes,
        close_series,
    ) = _prepare_portfolio_simulation(
        portfolio,
        years=years,
        use_cache=use_cache,
        symbol_dataset=symbol_dataset,
        bulk_data=bulk_data,
    )

    allocation = allocation_settings(portfolio)
    if signals_only and allocation.is_regime_adjusted:
        allocation = AllocationSettings(
            model=proportional_model(PROPORTIONAL_LINEAR), cost_bps=allocation.cost_bps
        )
    allocation = bind_regime_context(
        allocation,
        strategies,
        signals_by_id,
        global_proxy=global_proxy,
        track_changes=track_changes,
        close_series=close_series,
        years=years,
    )
    frame = simulate_portfolio_frame(
        strategies,
        signals_by_id,
        allocation=allocation,
        overlap_mode=overlap_mode,
        global_proxy=global_proxy,
        track_changes=track_changes,
        close_series=close_series,
        strategy_order=strategy_ids,
    )

    if signals_only and any(getattr(member, "nested_spec", None) for member in strategies):
        active, _, _ = _effective_signal_matrices(strategies, signals_by_id, pd.DatetimeIndex(frame["Date"]))
        now = active.any(axis=0)
        before = np.concatenate(([False], now[:-1]))
        frame = frame.assign(LongTradeIn=now & ~before, LongTradeOut=before & ~now, HoldLong=before)

    description = _portfolio_description(strategies, overlap_mode, global_proxy, allocation)
    max_hold = max(strategy.hold_days for strategy in strategies)
    max_profit = max(strategy.profit for strategy in strategies)
    return frame, strategies, description, max_hold, max_profit


def flat_cash_frame(master_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Cash / flat portfolio on an aligned calendar (single-strategy leave-one-out)."""
    n = len(master_dates)
    rolling = np.full(n, START_CAPITAL, dtype=float)
    frame = pd.DataFrame(
        {
            "Date": master_dates,
            "Close": np.zeros(n, dtype=float),
            "TrackChange": np.zeros(n, dtype=float),
            "LongTradeIn": np.zeros(n, dtype=bool),
            "LongTradeOut": np.zeros(n, dtype=bool),
            "HoldLong": np.zeros(n, dtype=bool),
            "RollingPnL": rolling,
            "TradePnL": np.zeros(n, dtype=float),
            "TradeEntry": np.zeros(n, dtype=float),
            "DaysInTrade": np.zeros(n, dtype=int),
        }
    )
    frame["RunningMax"] = frame["RollingPnL"].cummax()
    frame["Drawdown"] = (frame["RunningMax"] - frame["RollingPnL"]) / frame["RunningMax"]
    return frame


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def time_in_market_percent(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return float(frame["HoldLong"].astype(bool).mean() * 100.0)


def portfolio_utility(
    *,
    cagr_percent: float | None,
    sharpe: float | None,
    max_drawdown: float | None,
    time_in_market_percent: float | None,
    trades_per_year: float | None,
) -> float | None:
    """
    Explicit portfolio scoring utility.

    Units: CAGR %, Sharpe ratio, MaxDrawdown %, TimeInMarket %, TradesPerYear.
    utility = 1.0*CAGR + 4.0*Sharpe - 0.5*MaxDD - 0.1*TIM - 0.02*TPY
    """
    cagr = _finite_or_none(cagr_percent)
    sharpe_value = _finite_or_none(sharpe)
    max_dd_fraction = _finite_or_none(max_drawdown)
    time_in_market = _finite_or_none(time_in_market_percent)
    tpy = _finite_or_none(trades_per_year)
    if (
        cagr is None
        or sharpe_value is None
        or max_dd_fraction is None
        or time_in_market is None
        or tpy is None
    ):
        return None
    max_dd_percent = max_dd_fraction * 100.0
    return (
        1.0 * cagr
        + 4.0 * sharpe_value
        - 0.5 * max_dd_percent
        - 0.1 * time_in_market
        - 0.02 * tpy
    )


def exposure_adjusted_return(
    cagr_percent: float | None,
    time_in_market_percent: float | None,
) -> float | None:
    """CAGR% / time_in_market%. None when exposure is missing or zero."""
    cagr = _finite_or_none(cagr_percent)
    time_in_market = _finite_or_none(time_in_market_percent)
    if cagr is None or time_in_market is None or time_in_market == 0:
        return None
    return cagr / time_in_market


def marginal_cagr_per_10pp_exposure(
    incremental_cagr_pp: float | None,
    incremental_exposure_pp: float | None,
) -> float | None:
    """
    incremental CAGR / incremental exposure * 10.

    Returns CAGR percentage points gained per +10pp of time-in-market.
    """
    cagr_pp = _finite_or_none(incremental_cagr_pp)
    exposure_pp = _finite_or_none(incremental_exposure_pp)
    if cagr_pp is None or exposure_pp is None or exposure_pp == 0:
        return None
    return (cagr_pp / exposure_pp) * 10.0


def worst_calendar_year_pnl_percent(frame: pd.DataFrame) -> float | None:
    returns = yearly_returns(frame)
    if not returns:
        return None
    return float(min(returns.values()) * 100.0)


def avg_median_days_in_trade(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    closed = frame.loc[frame["LongTradeOut"].astype(bool), "DaysInTrade"]
    if closed.empty:
        return None, None
    values = closed.astype(float)
    return float(values.mean()), float(values.median())


def _empty_contribution_metrics(
    *,
    rolling_pnl: float | None,
    time_in_market: float | None,
    trades: int,
    worst_year: float | None,
    avg_days: float | None,
    median_days: float | None,
) -> dict[str, float | int | None]:
    return {
        "cagr_percent": None,
        "sharpe": None,
        "sortino": None,
        "max_drawdown": None,
        "rolling_pnl": rolling_pnl,
        "time_in_market_percent": time_in_market,
        "trades": trades,
        "worst_calendar_year_pnl_percent": worst_year,
        "avg_days_in_trade": avg_days,
        "median_days_in_trade": median_days,
        "calmar": None,
        "ulcer_index": None,
        "time_under_water_percent": None,
        "trades_per_year": None,
        "utility": None,
        "exposure_adjusted_return": None,
    }


def extract_contribution_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    avg_days, median_days = avg_median_days_in_trade(frame)
    time_in_market = _finite_or_none(time_in_market_percent(frame))
    worst_year = worst_calendar_year_pnl_percent(frame)
    try:
        metrics = compute_aggregate_metrics(frame)
    except (IndexError, ValueError, ZeroDivisionError):
        # Short single-year frames can become empty after best-year exclusion.
        rolling = _finite_or_none(frame["RollingPnL"].iloc[-1]) if not frame.empty else START_CAPITAL
        trades = int(frame["LongTradeOut"].astype(bool).sum()) if not frame.empty else 0
        return _empty_contribution_metrics(
            rolling_pnl=rolling,
            time_in_market=time_in_market,
            trades=trades,
            worst_year=worst_year,
            avg_days=avg_days,
            median_days=median_days,
        )

    cagr_percent_value = _finite_or_none(metrics["cagr_percent"])
    sharpe = _finite_or_none(metrics["sharpe"])
    sortino = _finite_or_none(metrics["sortino"])
    max_drawdown = _finite_or_none(metrics["max_drawdown"])
    trades = int(metrics["trades"])
    tpy = _finite_or_none(metrics.get("trades_per_year"))
    if tpy is None:
        tpy = _finite_or_none(trades_per_year(trades, len(frame)))
    calmar = _finite_or_none(metrics.get("calmar"))
    if calmar is None:
        calmar = _finite_or_none(calmar_ratio(cagr_percent_value, max_drawdown))
    ulcer = _finite_or_none(metrics.get("ulcer_index"))
    if ulcer is None and not frame.empty:
        ulcer = _finite_or_none(ulcer_index(frame))
    tuw = _finite_or_none(metrics.get("time_under_water_percent"))
    if tuw is None and not frame.empty:
        tuw = _finite_or_none(frame_time_under_water_percent(frame))

    result = {
        "cagr_percent": cagr_percent_value,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "rolling_pnl": _finite_or_none(metrics["rolling_pnl"]),
        "time_in_market_percent": time_in_market,
        "trades": trades,
        "worst_calendar_year_pnl_percent": worst_year,
        "avg_days_in_trade": avg_days,
        "median_days_in_trade": median_days,
        "calmar": calmar,
        "ulcer_index": ulcer,
        "time_under_water_percent": tuw,
        "trades_per_year": tpy,
    }
    result["utility"] = portfolio_utility(
        cagr_percent=cagr_percent_value,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        time_in_market_percent=time_in_market,
        trades_per_year=tpy,
    )
    result["exposure_adjusted_return"] = exposure_adjusted_return(
        cagr_percent_value,
        time_in_market,
    )
    return result


def holding_overlap(
    candidate_hold: np.ndarray,
    reduced_hold: np.ndarray,
) -> dict[str, int | float | None]:
    candidate = np.asarray(candidate_hold, dtype=bool)
    reduced = np.asarray(reduced_hold, dtype=bool)
    if candidate.shape != reduced.shape:
        raise ValueError("Candidate and reduced hold masks must align")

    candidate_holding_days = int(candidate.sum())
    overlapping_holding_days = int((candidate & reduced).sum())
    unique_holding_days = int((candidate & ~reduced).sum())

    if candidate_holding_days == 0:
        unique_holding_percent = None
        redundant_holding_percent = None
    else:
        unique_holding_percent = unique_holding_days / candidate_holding_days
        redundant_holding_percent = overlapping_holding_days / candidate_holding_days

    return {
        "candidate_holding_days": candidate_holding_days,
        "overlapping_holding_days": overlapping_holding_days,
        "unique_holding_days": unique_holding_days,
        "unique_holding_percent": unique_holding_percent,
        "redundant_holding_percent": redundant_holding_percent,
    }


def _delta(full: float | int | None, reduced: float | int | None) -> float | int | None:
    if full is None or reduced is None:
        return None
    return full - reduced


def _pp_delta_from_fraction(full: float | None, reduced: float | None) -> float | None:
    delta = _delta(full, reduced)
    if delta is None:
        return None
    return float(delta) * 100.0


def build_strategy_contribution(
    *,
    strategy_id: str,
    strategy_name: str,
    full_metrics: dict[str, float | int | None],
    reduced_metrics: dict[str, float | int | None],
    overlap: dict[str, int | float | None],
) -> dict[str, Any]:
    cagr_contribution_pp = _delta(full_metrics.get("cagr_percent"), reduced_metrics.get("cagr_percent"))
    added_exposure_pp = _delta(
        full_metrics.get("time_in_market_percent"),
        reduced_metrics.get("time_in_market_percent"),
    )
    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "full": full_metrics,
        "without_strategy": reduced_metrics,
        "cagr_contribution_pp": cagr_contribution_pp,
        "sharpe_delta": _delta(full_metrics.get("sharpe"), reduced_metrics.get("sharpe")),
        "sortino_delta": _delta(full_metrics.get("sortino"), reduced_metrics.get("sortino")),
        "calmar_delta": _delta(full_metrics.get("calmar"), reduced_metrics.get("calmar")),
        "ulcer_index_delta": _delta(full_metrics.get("ulcer_index"), reduced_metrics.get("ulcer_index")),
        "time_under_water_delta_pp": _delta(
            full_metrics.get("time_under_water_percent"),
            reduced_metrics.get("time_under_water_percent"),
        ),
        "max_drawdown_effect_pp": _pp_delta_from_fraction(
            full_metrics.get("max_drawdown"),
            reduced_metrics.get("max_drawdown"),
        ),
        "final_equity_delta": _delta(full_metrics.get("rolling_pnl"), reduced_metrics.get("rolling_pnl")),
        "added_exposure_pp": added_exposure_pp,
        "added_portfolio_trades": _delta(full_metrics.get("trades"), reduced_metrics.get("trades")),
        "worst_calendar_year_pnl_delta_pp": _delta(
            full_metrics.get("worst_calendar_year_pnl_percent"),
            reduced_metrics.get("worst_calendar_year_pnl_percent"),
        ),
        "avg_days_in_trade_delta": _delta(
            full_metrics.get("avg_days_in_trade"),
            reduced_metrics.get("avg_days_in_trade"),
        ),
        "median_days_in_trade_delta": _delta(
            full_metrics.get("median_days_in_trade"),
            reduced_metrics.get("median_days_in_trade"),
        ),
        "marginal_utility": _delta(full_metrics.get("utility"), reduced_metrics.get("utility")),
        "exposure_adjusted_return_delta": _delta(
            full_metrics.get("exposure_adjusted_return"),
            reduced_metrics.get("exposure_adjusted_return"),
        ),
        "marginal_cagr_per_10pp_exposure": marginal_cagr_per_10pp_exposure(
            cagr_contribution_pp,
            added_exposure_pp,
        ),
        **overlap,
    }


def _aligned_candidate_hold(
    signals: StrategySignals,
    master_dates: pd.DatetimeIndex,
) -> np.ndarray:
    _, hold, _ = _align_strategy_signals(
        pd.DataFrame(
            {
                "Date": signals.dates,
                "LongTradeIn": signals.long_trade_in,
                "HoldLong": signals.hold_long,
                "LongTradeOut": signals.long_trade_out,
            }
        ),
        master_dates,
    )
    return hold


def compute_portfolio_contribution(
    strategies: list,
    signals_by_id: dict[str, StrategySignals],
    *,
    full_frame: pd.DataFrame,
    overlap_mode: PortfolioOverlapMode,
    global_proxy: str | None,
    track_changes: dict[str, pd.Series],
    close_series: dict[str, pd.Series],
    strategy_order: list[str],
    allocation: AllocationSettings | None = None,
) -> list[dict[str, Any]]:
    master_dates = pd.DatetimeIndex(full_frame["Date"])
    full_metrics = extract_contribution_metrics(full_frame)
    strategies_by_id = {strategy.id: strategy for strategy in strategies}
    contribution: list[dict[str, Any]] = []

    for strategy_id in strategy_order:
        strategy = strategies_by_id[strategy_id]
        remaining = [item for item in strategies if item.id != strategy_id]
        remaining_order = [item_id for item_id in strategy_order if item_id != strategy_id]

        if remaining:
            reduced_frame = simulate_portfolio_frame(
                remaining,
                {item.id: signals_by_id[item.id] for item in remaining},
                allocation=allocation,
                overlap_mode=overlap_mode,
                global_proxy=global_proxy,
                track_changes=track_changes,
                close_series=close_series,
                strategy_order=remaining_order,
                master_dates=master_dates,
            )
        else:
            reduced_frame = flat_cash_frame(master_dates)

        reduced_hold = reduced_frame["HoldLong"].astype(bool).to_numpy()
        candidate_hold = _aligned_candidate_hold(signals_by_id[strategy_id], master_dates)
        overlap = holding_overlap(candidate_hold, reduced_hold)
        reduced_metrics = extract_contribution_metrics(reduced_frame)
        contribution.append(
            build_strategy_contribution(
                strategy_id=strategy_id,
                strategy_name=strategy.name,
                full_metrics=full_metrics,
                reduced_metrics=reduced_metrics,
                overlap=overlap,
            )
        )

    return contribution


SHAPLEY_METRIC_KEYS = (
    "cagr_percent",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "ulcer_index",
    "time_under_water_percent",
    "time_in_market_percent",
    "trades",
    "rolling_pnl",
    "utility",
    "exposure_adjusted_return",
    "worst_calendar_year_pnl_percent",
    "avg_days_in_trade",
    "median_days_in_trade",
)

SHAPLEY_SAMPLES_DEFAULT = 64
SHAPLEY_SAMPLES_MIN = 8
SHAPLEY_SAMPLES_MAX = 512


def clamp_shapley_samples(samples: int | None) -> int:
    value = SHAPLEY_SAMPLES_DEFAULT if samples is None else int(samples)
    return max(SHAPLEY_SAMPLES_MIN, min(SHAPLEY_SAMPLES_MAX, value))


def _empty_contribution_metrics() -> dict[str, float | int | None]:
    return {
        "cagr_percent": None,
        "sharpe": None,
        "sortino": None,
        "max_drawdown": None,
        "rolling_pnl": None,
        "time_in_market_percent": None,
        "trades": 0,
        "worst_calendar_year_pnl_percent": None,
        "avg_days_in_trade": None,
        "median_days_in_trade": None,
        "calmar": None,
        "ulcer_index": None,
        "time_under_water_percent": None,
        "trades_per_year": None,
        "utility": None,
        "exposure_adjusted_return": None,
    }


def _metric_margin(
    after: dict[str, float | int | None],
    before: dict[str, float | int | None],
    key: str,
) -> float | None:
    after_value = _finite_or_none(after.get(key))
    before_value = _finite_or_none(before.get(key))
    if after_value is None or before_value is None:
        return None
    return after_value - before_value


def _average_or_none(total: float, count: int) -> float | None:
    if count <= 0:
        return None
    return total / count


def _shapley_orderings(
    strategy_ids: list[str],
    samples: int,
    rng: random.Random,
) -> tuple[list[tuple[str, ...]], bool, int]:
    n = len(strategy_ids)
    if n == 0:
        return [], True, 0
    n_fact = math.factorial(n)
    if n <= 6 and n_fact <= samples:
        return [tuple(order) for order in itertools.permutations(strategy_ids)], True, n_fact

    orderings: list[tuple[str, ...]] = []
    for _ in range(samples):
        order = list(strategy_ids)
        rng.shuffle(order)
        orderings.append(tuple(order))
    return orderings, False, samples


def build_shapley_contribution_row(
    *,
    strategy_id: str,
    strategy_name: str,
    averages: dict[str, float | None],
) -> dict[str, Any]:
    cagr_contribution_pp = averages.get("cagr_percent")
    added_exposure_pp = averages.get("time_in_market_percent")
    max_drawdown_avg = averages.get("max_drawdown")
    empty = _empty_contribution_metrics()
    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "full": empty,
        "without_strategy": empty,
        "cagr_contribution_pp": cagr_contribution_pp,
        "sharpe_delta": averages.get("sharpe"),
        "sortino_delta": averages.get("sortino"),
        "calmar_delta": averages.get("calmar"),
        "ulcer_index_delta": averages.get("ulcer_index"),
        "time_under_water_delta_pp": averages.get("time_under_water_percent"),
        "max_drawdown_effect_pp": (
            None if max_drawdown_avg is None else float(max_drawdown_avg) * 100.0
        ),
        "final_equity_delta": averages.get("rolling_pnl"),
        "added_exposure_pp": added_exposure_pp,
        "added_portfolio_trades": averages.get("trades"),
        "worst_calendar_year_pnl_delta_pp": averages.get("worst_calendar_year_pnl_percent"),
        "avg_days_in_trade_delta": averages.get("avg_days_in_trade"),
        "median_days_in_trade_delta": averages.get("median_days_in_trade"),
        "marginal_utility": averages.get("utility"),
        "exposure_adjusted_return_delta": averages.get("exposure_adjusted_return"),
        "marginal_cagr_per_10pp_exposure": marginal_cagr_per_10pp_exposure(
            cagr_contribution_pp,
            added_exposure_pp,
        ),
        "candidate_holding_days": 0,
        "overlapping_holding_days": 0,
        "unique_holding_days": 0,
        "unique_holding_percent": None,
        "redundant_holding_percent": None,
    }


def compute_portfolio_shapley(
    strategies: list,
    signals_by_id: dict[str, StrategySignals],
    *,
    master_dates: pd.DatetimeIndex,
    overlap_mode: PortfolioOverlapMode,
    global_proxy: str | None,
    track_changes: dict[str, pd.Series],
    close_series: dict[str, pd.Series],
    strategy_order: list[str],
    samples: int = SHAPLEY_SAMPLES_DEFAULT,
    seed: int | None = None,
    allocation: AllocationSettings | None = None,
) -> dict[str, Any]:
    samples_clamped = clamp_shapley_samples(samples)
    strategy_ids = list(strategy_order)
    strategies_by_id = {strategy.id: strategy for strategy in strategies}
    rng = random.Random(seed)
    orderings, exact, samples_used = _shapley_orderings(strategy_ids, samples_clamped, rng)

    metrics_cache: dict[frozenset[str], dict[str, float | int | None]] = {}

    def coalition_metrics(member_ids: frozenset[str]) -> dict[str, float | int | None]:
        cached = metrics_cache.get(member_ids)
        if cached is not None:
            return cached
        if not member_ids:
            metrics = extract_contribution_metrics(flat_cash_frame(master_dates))
        else:
            subset = [strategies_by_id[item_id] for item_id in strategy_ids if item_id in member_ids]
            subset_order = [item_id for item_id in strategy_ids if item_id in member_ids]
            frame = simulate_portfolio_frame(
                subset,
                {item.id: signals_by_id[item.id] for item in subset},
                allocation=allocation,
                overlap_mode=overlap_mode,
                global_proxy=global_proxy,
                track_changes=track_changes,
                close_series=close_series,
                strategy_order=subset_order,
                master_dates=master_dates,
            )
            metrics = extract_contribution_metrics(frame)
        metrics_cache[member_ids] = metrics
        return metrics

    sums: dict[str, dict[str, float]] = {
        strategy_id: {key: 0.0 for key in SHAPLEY_METRIC_KEYS} for strategy_id in strategy_ids
    }
    counts: dict[str, dict[str, int]] = {
        strategy_id: {key: 0 for key in SHAPLEY_METRIC_KEYS} for strategy_id in strategy_ids
    }

    for ordering in orderings:
        members: set[str] = set()
        before_metrics = coalition_metrics(frozenset())
        for strategy_id in ordering:
            members.add(strategy_id)
            after_metrics = coalition_metrics(frozenset(members))
            for key in SHAPLEY_METRIC_KEYS:
                margin = _metric_margin(after_metrics, before_metrics, key)
                if margin is None:
                    continue
                sums[strategy_id][key] += margin
                counts[strategy_id][key] += 1
            before_metrics = after_metrics

    shapley_rows: list[dict[str, Any]] = []
    for strategy_id in strategy_ids:
        averages = {
            key: _average_or_none(sums[strategy_id][key], counts[strategy_id][key])
            for key in SHAPLEY_METRIC_KEYS
        }
        strategy = strategies_by_id[strategy_id]
        shapley_rows.append(
            build_shapley_contribution_row(
                strategy_id=strategy_id,
                strategy_name=strategy.name,
                averages=averages,
            )
        )

    return {
        "samples_used": samples_used,
        "exact": exact,
        "shapley": shapley_rows,
    }


def simulate_portfolio_shapley(request) -> dict:
    strategy_ids = list(request.strategy_ids)
    if not strategy_ids and not getattr(request, "portfolio_ids", None):
        raise ValueError("At least one strategy is required")

    (
        strategies,
        strategy_ids,
        signals_by_id,
        overlap_mode,
        global_proxy,
        track_changes,
        close_series,
    ) = _prepare_portfolio_simulation(
        request,
        years=request.years,
        use_cache=True,
    )

    master_dates = pd.DatetimeIndex(
        sorted(set().union(*(signals.dates for signals in signals_by_id.values())))
    )
    if len(master_dates) == 0:
        raise ValueError("No overlapping market data for selected strategies")

    return compute_portfolio_shapley(
        strategies,
        signals_by_id,
        master_dates=master_dates,
        overlap_mode=overlap_mode,
        global_proxy=global_proxy,
        track_changes=track_changes,
        close_series=close_series,
        strategy_order=strategy_ids,
        samples=getattr(request, "samples", SHAPLEY_SAMPLES_DEFAULT),
        seed=getattr(request, "seed", None),
        allocation=bind_regime_context(
            allocation_settings(request),
            strategies,
            signals_by_id,
            global_proxy=global_proxy,
            track_changes=track_changes,
            close_series=close_series,
            years=request.years,
            master_dates=master_dates,
        ),
    )


EXPOSURE_STEP_LIMIT = 4000


def _exposure_steps(frame: pd.DataFrame, *, limit: int | None = EXPOSURE_STEP_LIMIT) -> list[dict[str, Any]]:
    """Exposure as a step function: one row per bar where it or the signal counts changed."""
    columns = ["TargetExposure" if "TargetExposure" in frame.columns else "Exposure"]
    if "ActiveCount" in frame.columns:
        columns += ["ActiveCount", "EligibleCount"]
    changed = frame[columns].ne(frame[columns].shift()).any(axis=1)
    changed.iloc[0] = True
    rows = frame.loc[changed]
    if limit is not None and len(rows) > limit:
        rows = rows.iloc[-limit:]
    steps = []
    for _, row in rows.iterrows():
        step = {
            "date": pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"),
            "exposure": float(row[columns[0]]),
        }
        if "ActiveCount" in rows.columns:
            step["active_count"] = int(row["ActiveCount"])
            step["eligible_count"] = int(row["EligibleCount"])
        steps.append(step)
    return steps


def _overlay_exposure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """All-in frames are 0%/100%; give them an Exposure column so steps work the same way."""
    out = frame[["Date"]].copy()
    book = arrays_from_overlay(frame)
    out["Exposure"] = book.post_weight
    return out


def _allocation_payload(
    allocation: AllocationSettings,
    frame: pd.DataFrame,
    simulation: AllocationSimulation | None,
    reference: AllocationSimulation | None,
) -> dict[str, Any]:
    """Allocation analytics shipped with a portfolio backtest."""
    model = allocation.model
    payload: dict[str, Any] = {
        "model": model.id,
        "label": model.label,
        "transaction_cost_bps": allocation.cost_bps,
    }
    if allocation.regime is not None:
        payload["regime"] = allocation.regime.as_dict()
    if simulation is None:
        payload.update(
            analytics=exposure_analytics(arrays_from_overlay(frame)),
            attribution=None,
            eligibility=None,
            ledger=None,
            exposure_steps=_exposure_steps(_overlay_exposure_frame(frame)),
        )
        return payload

    last = len(frame) - 1
    payload.update(
        analytics=exposure_analytics(arrays_from_simulation(simulation)),
        attribution=(
            allocation_attribution(simulation, reference) if reference is not None else None
        ),
        eligibility=simulation.eligibility,
        ledger={
            "cash": float(simulation.cash[last]),
            "invested_capital": float(simulation.invested[last]),
            "equity": float(simulation.equity[last]),
            "realized_pnl": float(simulation.realized_pnl[last]),
            "unrealized_pnl": float(simulation.unrealized_pnl[last]),
            "transaction_costs": float(simulation.total_costs[last]),
            "positions": simulation.final_positions,
        },
        exposure_steps=_exposure_steps(frame),
    )
    return payload


def _json_safe(value: Any) -> Any:
    """Recursively turn non-finite floats into None so the response is valid JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _positions_payload(
    book: AllocationSimulation,
    *,
    comparison_equity: np.ndarray | None = None,
    legacy_returns: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Position episodes and rebalance events from the book's executed transactions.

    ``comparison_equity`` is the equity curve the book is supposed to reproduce (the all-in
    overlay's). The largest relative gap is reported so a mismatch is visible, not hidden.

    ``legacy_returns`` maps an episode's start date to the return the old all-in Trades table
    showed for that trade. When given, it is what Episode % displays, so all-in numbers match
    the old table exactly. The last line of a closed episode follows it, so the two never
    disagree on screen. The ledger's own figure stays available as ``return_on_equity``.
    """
    episodes = build_position_episodes(
        book.transactions,
        asset_weight_post=book.asset_weight_post,
        final_marks=book.final_marks,
        last_bar=len(book.dates) - 1,
    )
    for episode in episodes:
        # Which instrument the episode held: the leveraged stand-in or the real leverage symbol.
        episode["instrument"] = book.instruments.get(episode["symbol"])
    reconciliation = reconcile_episodes(
        episodes,
        start_equity=book.start_capital,
        final_equity=float(book.equity[-1]),
        start_cash=book.start_capital,
        final_cash=float(book.cash[-1]),
        transactions=book.transactions,
    )
    if legacy_returns:
        for episode in episodes:
            legacy = legacy_returns.get(episode["start"])
            if legacy is None:
                continue
            episode["return"] = legacy
            if episode["status"] == "Closed":
                episode["events"][-1]["episode_return_to_date"] = legacy
    if comparison_equity is not None:
        reference = np.asarray(comparison_equity, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.abs(book.equity - reference) / np.abs(reference)
        reconciliation["ledger_vs_simulation_max_relative_difference"] = float(
            np.nanmax(gap) if gap.size else 0.0
        )
    return _json_safe(
        {
            "episodes": episodes,
            "reconciliation": reconciliation,
            "signal_changes_without_transaction": book.signal_changes_without_transaction,
            "price_basis": {
                "leverage": leverage_settings.general_leverage(),
                "description": (
                    "Prices are the closing price on the trade date of the instrument traded: the "
                    "underlying's own price, or its leverage symbol's. A symbol without a "
                    "leverage-settings entry is simulated at "
                    f"{leverage_settings.general_leverage():g}x its daily return; one with an "
                    "entry trades its leverage symbol once that exists. P&L and exposure come "
                    "from the ledger and are not the change between two prices."
                ),
            },
        }
    )


def simulate_portfolio(request) -> dict:
    strategy_ids = list(request.strategy_ids)
    if not strategy_ids and not getattr(request, "portfolio_ids", None):
        raise ValueError("At least one strategy is required")

    (
        strategies,
        strategy_ids,
        signals_by_id,
        overlap_mode,
        global_proxy,
        track_changes,
        close_series,
    ) = _prepare_portfolio_simulation(
        request,
        years=request.years,
        use_cache=True,
    )

    allocation = bind_regime_context(
        allocation_settings(request),
        strategies,
        signals_by_id,
        global_proxy=global_proxy,
        track_changes=track_changes,
        close_series=close_series,
        years=request.years,
    )
    simulation: AllocationSimulation | None = None
    reference: AllocationSimulation | None = None
    # All-in on one trading symbol is the original overlay, untouched. On several symbols it
    # runs through the engine like every other model.
    if allocation.is_all_in and uses_overlay(strategies, global_proxy):
        frame = _costed_overlay(
            strategies,
            signals_by_id,
            cost_bps=allocation.cost_bps,
            overlap_mode=overlap_mode,
            global_proxy=global_proxy,
            track_changes=track_changes,
            close_series=close_series,
            strategy_order=strategy_ids,
        )
        mixed_pnl_symbols = False
    else:
        simulation = simulate_portfolio_allocation(
            strategies,
            signals_by_id,
            model=allocation.model,
            cost_bps=allocation.cost_bps,
            global_proxy=global_proxy,
            track_changes=track_changes,
            close_series=close_series,
            regime_context=allocation.regime_context,
        )
        frame = simulation.frame
        # Reference book for attribution: the same signals held all-in, at the same cost.
        # All-in is its own reference.
        if not allocation.is_all_in:
            reference = simulate_portfolio_allocation(
                strategies,
                signals_by_id,
                model=resolve_allocation_model(ALL_IN),
                cost_bps=allocation.cost_bps,
                global_proxy=global_proxy,
                track_changes=track_changes,
                close_series=close_series,
                master_dates=simulation.dates,
            )
        # Equity is book value rather than one instrument's price.
        mixed_pnl_symbols = True

    description = _portfolio_description(strategies, overlap_mode, global_proxy, allocation)
    max_hold = max(strategy.hold_days for strategy in strategies)
    max_profit = max(strategy.profit for strategy in strategies)

    regime_mode = getattr(request, "regime_mode", None)
    if regime_mode:
        return regime_request_payload(
            frame,
            mode=regime_mode,
            regime_id=getattr(request, "regime_id", None),
            years=request.years,
            portfolio_equity=mixed_pnl_symbols,
        )

    payload = detailed_backtest_payload(
        frame,
        max_hold,
        max_profit,
        description,
        portfolio_equity=mixed_pnl_symbols,
    )
    payload["allocation"] = _allocation_payload(allocation, frame, simulation, reference)
    payload["allocation"]["trading_symbols"] = trading_symbols(strategies, global_proxy)
    payload["allocation"]["general_leverage"] = leverage_settings.general_leverage()
    payload["allocation"]["vehicles"] = [] if simulation is None else simulation.vehicle_report
    if simulation is not None and simulation.member_details:
        payload["allocation"]["members"] = simulation.member_details
        payload["allocation"]["exposure_steps"] = _exposure_steps(frame.assign(TargetExposure=frame["Exposure"]), limit=None)
        payload["allocation"]["nested_cash_policy"] = "respect_child_rejection"
    if simulation is not None:
        payload["positions"] = _positions_payload(simulation)
    else:
        replay = replay_overlay_transactions(
            frame,
            strategies,
            signals_by_id,
            global_proxy=global_proxy,
            cost_bps=allocation.cost_bps,
        )
        payload["positions"] = _positions_payload(
            replay,
            comparison_equity=frame["RollingPnL"].to_numpy(dtype=float),
            # The old Trades table's own numbers, taken from the unchanged overlay code.
            legacy_returns={
                trade["entry_date"]: float(trade["trade_pnl"]) for trade in payload["trades"]
            },
        )
    payload["contribution"] = compute_portfolio_contribution(
        strategies,
        signals_by_id,
        full_frame=frame,
        overlap_mode=overlap_mode,
        global_proxy=global_proxy,
        track_changes=track_changes,
        close_series=close_series,
        strategy_order=strategy_ids,
        allocation=allocation,
    )
    return payload


def simulate_portfolio_allocation_comparison(request) -> dict:
    """Every allocation model on the same signals. Computed only when asked.

    Rows: All-in, regime-adjusted at each penalty preset (Harsh, Strict, Medium, Low), then the
    proportional modes. Every model is cut to the same window (first bar any strategy is
    eligible) and rebased to the same starting capital, so equity curves and ratios compare
    directly.

    The All-in row is always the real thing: 100% while any strategy is active, with every
    strategy holding until it exits. On one trading symbol that is the original overlay. On
    several, the book stays fully invested and splits across the active strategies' symbols in
    proportion to how many strategies are active in each (linear proportional, scaled to 100%).
    The regime rows share one scoring pass and differ only in their penalties; the score they
    read is the request's ``regime_score``.
    """
    strategy_ids = list(request.strategy_ids)
    if not strategy_ids and not getattr(request, "portfolio_ids", None):
        raise ValueError("At least one strategy is required")

    (
        strategies,
        strategy_ids,
        signals_by_id,
        _,
        global_proxy,
        track_changes,
        close_series,
    ) = _prepare_portfolio_simulation(request, years=request.years, use_cache=True, keep_executed=True)

    cost_bps = allocation_settings(request).cost_bps
    master_dates = _master_dates_for(signals_by_id, None)
    if len(master_dates) == 0:
        raise ValueError("No overlapping market data for selected strategies")
    score = getattr(request, "regime_score", None)
    regime_context = bind_regime_context(
        AllocationSettings(
            model=RegimeAdjustedModel(),
            regime=RegimePenaltySettings.from_preset(None, score),
        ),
        strategies,
        signals_by_id,
        global_proxy=global_proxy,
        track_changes=track_changes,
        close_series=close_series,
        years=request.years,
        master_dates=master_dates,
    ).regime_context

    def run(model: AllocationModel) -> AllocationSimulation:
        preset = getattr(model, "preset", None)
        return simulate_portfolio_allocation(
            strategies,
            signals_by_id,
            model=model,
            cost_bps=cost_bps,
            global_proxy=global_proxy,
            track_changes=track_changes,
            close_series=close_series,
            master_dates=master_dates,
            regime_context=regime_context,
            regime_settings=RegimePenaltySettings.from_preset(preset, score) if preset else None,
        )

    reference = run(resolve_allocation_model(ALL_IN))
    start = reference.start_index
    symbols = trading_symbols(strategies, global_proxy)
    all_in = (
        _costed_overlay(
            strategies,
            signals_by_id,
            cost_bps=cost_bps,
            overlap_mode=PORTFOLIO_OVERLAP_MODE,
            global_proxy=global_proxy,
            track_changes=track_changes,
            close_series=close_series,
            strategy_order=strategy_ids,
            master_dates=master_dates,
        )
        if uses_overlay(strategies, global_proxy)
        else None
    )

    models = [
        AllInModel(),
        *(RegimeAdjustedModel(preset=preset) for preset in PENALTY_PRESETS),
        *(model for model in comparison_models(getattr(request, "tier_single_pct", None)) if model.id != ALL_IN),
    ]
    rows: list[dict[str, Any]] = []
    curves: list[tuple[str, pd.DataFrame]] = []
    for model in models:
        if model.id == ALL_IN:
            if all_in is not None:
                frame = all_in
                book = arrays_from_overlay(all_in, start=start)
            else:
                frame = reference.frame
                book = arrays_from_simulation(reference)
            attribution = None
        else:
            simulation = run(model)
            frame = simulation.frame
            book = arrays_from_simulation(simulation)
            attribution = allocation_attribution(simulation, reference)
        comparable = rebased_comparison_frame(frame, start, START_CAPITAL)
        curves.append((model.id, comparable))
        rows.append(
            {
                "model": model.id,
                "label": model.label,
                "metrics": extract_contribution_metrics(comparable),
                "analytics": exposure_analytics(book),
                "attribution": attribution,
                "drawdown_risk": drawdown_risk_for_frame(comparable, paths=COMPARISON_PATHS),
            }
        )

    return {
        "start_date": pd.Timestamp(master_dates[start]).strftime("%Y-%m-%d"),
        "start_capital": START_CAPITAL,
        "transaction_cost_bps": cost_bps,
        "regime_score": score or "robustness",
        "trading_symbols": symbols,
        "single_symbol": len(symbols) == 1,
        "models": rows,
        "equity_curves": merged_equity_curves(curves),
    }


def _portfolio_atr_symbol(strategies: list, global_proxy: str | None) -> str | None:
    """Symbol whose ATR labels the portfolio book.

    A global proxy is that symbol. Otherwise every strategy must book PnL on
    the same symbol. Mixed symbols have no single ATR.
    """
    symbols = set(trading_symbols(strategies, global_proxy))
    if len(symbols) == 1:
        return next(iter(symbols))
    return None


def _atr_by_day(symbol: str, *, years: int) -> pd.DataFrame | None:
    dataset = bt.load_symbol_dataset([symbol], years=years, use_cache=True)
    data = dataset.get(symbol)
    if data is None or "ATR20" not in data.columns or "ATR50" not in data.columns or "Date" not in data.columns:
        return None
    frame = data.copy()
    frame["_day"] = pd.to_datetime(frame["Date"], errors="coerce")
    if getattr(frame["_day"].dt, "tz", None) is not None:
        frame["_day"] = frame["_day"].dt.tz_localize(None)
    frame["_day"] = frame["_day"].dt.normalize()
    frame = frame.dropna(subset=["_day"]).drop_duplicates("_day").set_index("_day")
    return pd.DataFrame(
        {
            "ATR20": pd.to_numeric(frame["ATR20"], errors="coerce"),
            "ATR50": pd.to_numeric(frame["ATR50"], errors="coerce"),
        }
    )


def _with_symbol_atr(frame: pd.DataFrame, atr: pd.DataFrame | None) -> pd.DataFrame:
    if atr is None:
        return frame
    out = frame.copy()
    days = pd.to_datetime(out["Date"], errors="coerce")
    if getattr(days.dt, "tz", None) is not None:
        days = days.dt.tz_localize(None)
    days = days.dt.normalize()
    aligned = atr.reindex(pd.DatetimeIndex(days))
    out["ATR20"] = aligned["ATR20"].to_numpy(dtype=float)
    out["ATR50"] = aligned["ATR50"].to_numpy(dtype=float)
    return out


def simulate_portfolio_regimes(request) -> dict:
    """Market-regime quality and leave-one-out contribution. Computed only when asked."""
    from api.market_regimes import build_regime_comparison, load_regime_calendar, score_regime_book

    strategy_ids = list(request.strategy_ids)
    if not strategy_ids and not getattr(request, "portfolio_ids", None):
        raise ValueError("At least one strategy is required")

    (
        strategies,
        strategy_ids,
        signals_by_id,
        overlap_mode,
        global_proxy,
        track_changes,
        close_series,
    ) = _prepare_portfolio_simulation(
        request,
        years=request.years,
        use_cache=True,
        keep_executed=True,
    )

    allocation = bind_regime_context(
        allocation_settings(request),
        strategies,
        signals_by_id,
        global_proxy=global_proxy,
        track_changes=track_changes,
        close_series=close_series,
        years=request.years,
    )
    full_frame = simulate_portfolio_frame(
        strategies,
        signals_by_id,
        allocation=allocation,
        overlap_mode=overlap_mode,
        global_proxy=global_proxy,
        track_changes=track_changes,
        close_series=close_series,
        strategy_order=strategy_ids,
    )
    master_dates = pd.DatetimeIndex(full_frame["Date"])
    if len(master_dates) == 0:
        raise ValueError("No overlapping market data for selected strategies")

    calendar = load_regime_calendar(request.years)
    atr_symbol = _portfolio_atr_symbol(strategies, global_proxy)
    atr = _atr_by_day(atr_symbol, years=request.years) if atr_symbol else None
    full_frame = _with_symbol_atr(full_frame, atr)
    portfolio_book = score_regime_book(full_frame, calendar)
    strategies_by_id = {strategy.id: strategy for strategy in strategies}

    strategy_rows: list[dict[str, Any]] = []
    for strategy_id in strategy_ids:
        strategy = strategies_by_id[strategy_id]
        executed = signals_by_id[strategy_id].executed
        if executed is None:
            raise ValueError(f"Missing backtest for strategy: {strategy.name}")
        remaining = [item for item in strategies if item.id != strategy_id]
        remaining_order = [item_id for item_id in strategy_ids if item_id != strategy_id]
        if remaining:
            reduced_frame = simulate_portfolio_frame(
                remaining,
                {item.id: signals_by_id[item.id] for item in remaining},
                allocation=allocation,
                overlap_mode=overlap_mode,
                global_proxy=global_proxy,
                track_changes=track_changes,
                close_series=close_series,
                strategy_order=remaining_order,
                master_dates=master_dates,
            )
        else:
            reduced_frame = flat_cash_frame(master_dates)
        reduced_frame = _with_symbol_atr(reduced_frame, atr)
        strategy_rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": strategy.name,
                "quality": score_regime_book(executed, calendar),
                "contribution": build_regime_comparison(
                    portfolio_book,
                    score_regime_book(reduced_frame, calendar),
                ),
            }
        )

    return {"portfolio": portfolio_book, "strategies": strategy_rows}
