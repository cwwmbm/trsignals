from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import backtest as bt
import config
from api.proxy_symbol import proxy_column
from api.scan_service import execute_saved_strategy
from api.serializers import detailed_backtest_payload
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


def _market_inputs_from_dataset(
    symbol_dataset: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    track_changes: dict[str, pd.Series] = {}
    close_series: dict[str, pd.Series] = {}
    for symbol, data in symbol_dataset.items():
        indexed = data.set_index(pd.to_datetime(data["Date"]))
        close_series[symbol] = indexed["Close"]
        track_changes[symbol] = _track_change_from_close(indexed["Close"])
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


def _portfolio_description(strategies: list, overlap_mode: str, global_proxy: str | None) -> str:
    mode_label = (
        "first signal only"
        if overlap_mode == "first_signal_only"
        else "hold until all exit"
    )
    names = ", ".join(strategy.name for strategy in strategies)
    proxy_note = f", proxy {global_proxy}" if global_proxy else ""
    return f"Portfolio ({len(strategies)} strategies, {mode_label}{proxy_note}): {names}"


def _prepare_portfolio_simulation(
    portfolio,
    *,
    years: int = 1,
    use_cache: bool = True,
    symbol_dataset: dict[str, pd.DataFrame] | None = None,
    bulk_data: pd.DataFrame | None = None,
) -> tuple[list, list[str], dict[str, StrategySignals], str, str | None, dict[str, pd.Series], dict[str, pd.Series]]:
    strategy_ids = list(portfolio.strategy_ids)
    strategies = _resolve_portfolio_strategies(strategy_ids)
    overlap_mode = portfolio.overlap_mode
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
) -> tuple[pd.DataFrame, list, str, int, int]:
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

    frame = simulate_portfolio_overlay(
        strategies,
        signals_by_id,
        overlap_mode=overlap_mode,
        global_proxy=global_proxy,
        track_changes=track_changes,
        close_series=close_series,
        strategy_order=strategy_ids,
    )

    description = _portfolio_description(strategies, overlap_mode, global_proxy)
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
            reduced_frame = simulate_portfolio_overlay(
                remaining,
                {item.id: signals_by_id[item.id] for item in remaining},
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
            frame = simulate_portfolio_overlay(
                subset,
                {item.id: signals_by_id[item.id] for item in subset},
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
    if not strategy_ids:
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
    )


def simulate_portfolio(request) -> dict:
    strategy_ids = list(request.strategy_ids)
    if not strategy_ids:
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

    frame = simulate_portfolio_overlay(
        strategies,
        signals_by_id,
        overlap_mode=overlap_mode,
        global_proxy=global_proxy,
        track_changes=track_changes,
        close_series=close_series,
        strategy_order=strategy_ids,
    )

    description = _portfolio_description(strategies, overlap_mode, global_proxy)
    max_hold = max(strategy.hold_days for strategy in strategies)
    max_profit = max(strategy.profit for strategy in strategies)
    mixed_pnl_symbols = len(
        {_strategy_pnl_symbol(strategy, global_proxy) for strategy in strategies}
    ) > 1

    payload = detailed_backtest_payload(
        frame,
        max_hold,
        max_profit,
        description,
        portfolio_equity=mixed_pnl_symbols,
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
    )
    return payload
