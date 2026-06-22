from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as bt
import config
from api.proxy_symbol import proxy_column
from api.scan_service import execute_saved_strategy
from api.serializers import detailed_backtest_payload
from api.strategy_store import get_strategy_by_id

PortfolioOverlapMode = str
START_CAPITAL = 15000


@dataclass
class StrategySignals:
    strategy_id: str
    name: str
    symbol: str
    dates: pd.DatetimeIndex
    long_trade_in: np.ndarray
    hold_long: np.ndarray
    long_trade_out: np.ndarray


def _normalize_proxy(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = str(value).strip().upper()
    return cleaned or None


def _track_change_from_close(close: pd.Series) -> pd.Series:
    return config.Leverage * close.pct_change()


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


def _load_portfolio_market_data(
    symbols: set[str],
    *,
    years: int,
    use_cache: bool = True,
) -> tuple[pd.DataFrame | None, dict[str, pd.DataFrame], dict[str, pd.Series], dict[str, pd.Series]]:
    """Load required symbols once and derive PnL inputs from cached or fresh data."""
    symbol_list = sorted(symbols)
    symbol_dataset = bt.load_symbol_dataset(symbol_list, years=years, use_cache=use_cache)

    track_changes: dict[str, pd.Series] = {}
    close_series: dict[str, pd.Series] = {}
    for symbol in symbol_list:
        data = symbol_dataset[symbol]
        indexed = data.set_index(pd.to_datetime(data["Date"]))
        close_series[symbol] = indexed["Close"]
        track_changes[symbol] = _track_change_from_close(indexed["Close"])

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
) -> pd.DataFrame:
    if not strategies:
        raise ValueError("At least one strategy is required")

    master_dates = pd.DatetimeIndex(sorted(set().union(*(signals.dates for signals in signals_by_id.values()))))
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

        if long_in[i]:
            if mixed_pnl_symbols:
                trade_entry[i] = rolling_pnl[i]
                trade_entry_rolling[i] = rolling_pnl[i]
            else:
                trade_entry[i] = close_out[i]
        elif returns_hold and i > 0:
            trade_entry[i] = trade_entry[i - 1]
            if mixed_pnl_symbols:
                trade_entry_rolling[i] = trade_entry_rolling[i - 1]
        elif i > 0:
            trade_entry[i] = trade_entry[i - 1]
            if mixed_pnl_symbols:
                trade_entry_rolling[i] = trade_entry_rolling[i - 1]

        mark_to_market = returns_hold or trade_out
        if mixed_pnl_symbols:
            entry_basis = trade_entry_rolling[i]
            if mark_to_market and entry_basis:
                trade_pnl[i] = rolling_pnl[i] / entry_basis - 1
            else:
                trade_pnl[i] = 0.0
        else:
            entry = trade_entry[i]
            price = close_out[i]
            if mark_to_market and entry and not np.isnan(price):
                trade_pnl[i] = config.Leverage * ((price - entry) / entry)
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


def _load_strategy_signals(
    strategy,
    *,
    years: int,
    bulk_data: pd.DataFrame | None,
    symbol_dataset: dict[str, pd.DataFrame],
    use_cache: bool = True,
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


def build_portfolio_overlay_frame(
    portfolio,
    *,
    years: int = 1,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, list, str, int, int]:
    strategy_ids = list(portfolio.strategy_ids)
    strategies = _resolve_portfolio_strategies(strategy_ids)
    overlap_mode = portfolio.overlap_mode
    global_proxy = _normalize_proxy(getattr(portfolio, "proxy_symbol", None))

    symbols = _collect_symbols(strategies, global_proxy)
    bulk_data, symbol_dataset, track_changes, close_series = _load_portfolio_market_data(
        symbols,
        years=years,
        use_cache=use_cache,
    )

    signals_by_id: dict[str, StrategySignals] = {}
    for strategy in strategies:
        signals_by_id[strategy.id] = _load_strategy_signals(
            strategy,
            years=years,
            bulk_data=bulk_data,
            symbol_dataset=symbol_dataset,
            use_cache=use_cache,
        )

    frame = simulate_portfolio_overlay(
        strategies,
        signals_by_id,
        overlap_mode=overlap_mode,
        global_proxy=global_proxy,
        track_changes=track_changes,
        close_series=close_series,
        strategy_order=strategy_ids,
    )

    mode_label = (
        "first signal only"
        if overlap_mode == "first_signal_only"
        else "hold until all exit"
    )
    names = ", ".join(strategy.name for strategy in strategies)
    proxy_note = f", proxy {global_proxy}" if global_proxy else ""
    description = f"Portfolio ({len(strategies)} strategies, {mode_label}{proxy_note}): {names}"
    max_hold = max(strategy.hold_days for strategy in strategies)
    max_profit = max(strategy.profit for strategy in strategies)
    return frame, strategies, description, max_hold, max_profit


def simulate_portfolio(request) -> dict:
    strategy_ids = list(request.strategy_ids)
    if not strategy_ids:
        raise ValueError("At least one strategy is required")

    frame, strategies, description, max_hold, max_profit = build_portfolio_overlay_frame(
        request,
        years=request.years,
        use_cache=True,
    )
    mixed_pnl_symbols = len(
        {_strategy_pnl_symbol(strategy, _normalize_proxy(getattr(request, "proxy_symbol", None))) for strategy in strategies}
    ) > 1
    return detailed_backtest_payload(
        frame,
        max_hold,
        max_profit,
        description,
        portfolio_equity=mixed_pnl_symbols,
    )
