"""Fast single-indicator trials for strategies with confirm symbols.

A symbol-specific threshold filter ("RSI14 <= 50") on a confirm strategy is added to the conditions of
every symbol: the primary must pass it, and each confirm symbol must pass it before it counts as
active. ``builder_strategy._run_confirm_embedded_buy_threshold`` therefore rebuilds the whole
cross-symbol frame for each grid point (about 0.7 s), which makes a sweep take minutes.

Almost all of that work does not depend on the threshold. This context does it once:

* each symbol's own buy and sell masks for the strategy without the new filter;
* the date alignment of every confirm symbol onto the primary's rows;
* the market-wide (primary) filters, which gate the merged entry.

A candidate is then: for each symbol, ``base_buy & (column <op> value)``; for each confirm symbol
the hold state of that symbol's own strategy run (``fast_sweep.hold_flags``) gives "buy today, or
still holding without a sell"; the active masks are ANDed onto the primary's entry and the primary
filters. That is exactly what ``bt.apply_cross_symbol_signal`` does, minus rebuilding frames.

``build`` returns None, so callers fall back to the slow path, whenever it cannot be sure:
no numba, a zero-day (``og``) strategy, duplicated dates, or if the composition without any new
filter does not reproduce the real frame's entry mask.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as bt
from api import fast_sweep
from api.builder_strategy import (
    _confirm_strategy_for_cross_symbol,
    _format_threshold_right,
    _ranking_row_from_frame,
    builder_signal_callable,
)
from api.strategy_compiler import compile_buy_mask, compile_entry_ready_mask


def _bool_array(value, index: pd.Index) -> np.ndarray:
    """A signal's buy/sell output as a plain bool array, like ``backtest._buy_series_by_date``."""
    if isinstance(value, pd.Series):
        return value.reindex(index).fillna(False).astype(bool).to_numpy()
    if isinstance(value, (bool, np.bool_)):
        return np.full(len(index), bool(value))
    return np.asarray(value, dtype=bool)


class ConfirmSweepContext:
    def __init__(self, strategy, frame, days, profit, is_long, pnl_column, symbols, symbol_data, strategy_resolver, labels):
        self.strategy = strategy
        self.frame = frame
        self.days, self.profit, self.is_long, self.pnl_column = days, profit, is_long, pnl_column
        self.primary = symbols[0]
        self.confirm = symbols[1:]
        self.resolver = strategy_resolver
        self._symbol_data = symbol_data
        self._columns: dict[str, list[np.ndarray]] = {}
        self._ready: dict[str, np.ndarray] = {}
        self._frame_ready = frame["EntryReady"].fillna(False).to_numpy(bool) if "EntryReady" in frame.columns else np.ones(len(frame), bool)
        self.scorer = fast_sweep.Scorer(frame, days, profit, is_long, pnl_column=pnl_column)

        confirm_strategy, primary_filters = _confirm_strategy_for_cross_symbol(strategy)
        self._confirm_strategy = confirm_strategy
        signal = builder_signal_callable(confirm_strategy, strategy_resolver=strategy_resolver, labels=labels)
        dates = pd.to_datetime(frame["Date"])
        self._buy: list[np.ndarray] = []
        self._sell: list[np.ndarray] = []
        self._close: list[np.ndarray] = []
        self._align: list[np.ndarray | None] = []
        for position, symbol in enumerate(symbols):
            data = symbol_data[symbol]
            buy, sell, *_rest = signal(data.copy() if position == 0 else data, symbol)
            self._buy.append(_bool_array(buy, data.index))
            self._sell.append(_bool_array(sell, data.index))
            self._close.append(data["Close"].to_numpy(dtype=np.float64))
            if position == 0:
                self._align.append(None)
                continue
            symbol_dates = pd.Index(pd.to_datetime(data["Date"]))
            if not symbol_dates.is_unique:
                raise ValueError("duplicated dates")
            self._align.append(symbol_dates.get_indexer(dates))
        if primary_filters:
            self._primary_filter = compile_buy_mask(frame.copy(), primary_filters, strategy_resolver=strategy_resolver).fillna(False).to_numpy(bool)
        else:
            self._primary_filter = np.ones(len(frame), dtype=bool)

    # ---- composition
    def _active_on_primary(self, position: int, buy: np.ndarray) -> np.ndarray:
        """A confirm symbol's "buy today or still holding without a sell", on the primary's rows."""
        sell = self._sell[position]
        hold = fast_sweep.hold_flags(buy, sell, self._close[position], self.days, self.profit, self.is_long)
        active = buy | (hold & ~sell)
        index = self._align[position]
        out = np.zeros(len(index), dtype=bool)
        found = index >= 0
        out[found] = active[index[found]]
        return out

    def _combine(self, masks: list[np.ndarray | None]) -> np.ndarray:
        """Entry mask given each symbol's extra threshold mask (None means no extra filter)."""
        primary = self._buy[0] if masks[0] is None else self._buy[0] & masks[0]
        combined = primary.copy()
        for position in range(1, len(self._buy)):
            buy = self._buy[position] if masks[position] is None else self._buy[position] & masks[position]
            combined &= self._active_on_primary(position, buy)
        return combined & self._primary_filter

    def base_matches_frame(self) -> bool:
        composed = self._combine([None] * len(self._buy))
        return bool(np.array_equal(composed, np.asarray(self.frame["Buy"].to_numpy(), dtype=bool)))

    def _column_arrays(self, column: str) -> list[np.ndarray]:
        hit = self._columns.get(column)
        if hit is None:
            hit = self._columns[column] = [
                self._symbol_data[symbol][column].to_numpy(dtype=np.float64) for symbol in [self.primary, *self.confirm]
            ]
        return hit

    def _column_ready(self, column: str) -> np.ndarray:
        """Rows where ``column`` exists on every symbol (the filter's own input readiness)."""
        hit = self._ready.get(column)
        if hit is None:
            probe = [{"left": column, "operator": ">=", "right": "0", "logic": "AND"}]
            ready = compile_entry_ready_mask(self.frame, probe, strategy_resolver=self.resolver).fillna(False).to_numpy(bool)
            for position, symbol in enumerate(self.confirm, start=1):
                data = self._symbol_data[symbol]
                sec = compile_entry_ready_mask(data, probe, strategy_resolver=self.resolver).fillna(False).to_numpy(bool)
                index = self._align[position]
                aligned = np.zeros(len(index), dtype=bool)
                found = index >= 0
                aligned[found] = sec[index[found]]
                ready &= aligned
            hit = self._ready[column] = ready
        return hit

    def candidate(self, column: str, condition: str, value) -> tuple[np.ndarray, np.ndarray]:
        """(entry mask, EntryReady) for the strategy plus ``column`` ``>=``/``<=`` ``value``."""
        # The slow path writes the threshold into a condition as text rounded to 4 decimals.
        threshold = float(_format_threshold_right(value))
        arrays = self._column_arrays(column)
        masks = [(a <= threshold) if condition == "less" else (a >= threshold) for a in arrays]
        return self._combine(masks), self._frame_ready & self._column_ready(column)

    # ---- screening and exact rows
    def top_candidates(self, column: str, candidates: list[tuple[str, float]], *, limit: int = 3, min_trades: int = 0) -> list[int]:
        sharpe = np.full(len(candidates), np.nan)
        trades = np.zeros(len(candidates), dtype=np.int64)
        for k, (condition, value) in enumerate(candidates):
            buy, ready = self.candidate(column, condition, value)
            first = int(np.argmax(ready)) if ready.any() else 0
            sharpe[k], trades[k] = self.scorer.score(buy, first=first)
        return fast_sweep.best_candidates(sharpe, trades, limit=limit, min_trades=min_trades)

    def exact_row(self, column: str, condition: str, value, *, include_yearly: bool = True) -> dict:
        """The ranking row the slow path would give, run through the real engine."""
        buy, ready = self.candidate(column, condition, value)
        frame = self.frame.copy()
        frame["Buy"] = buy
        frame["EntryReady"] = ready
        return _ranking_row_from_frame(
            frame,
            days=self.days,
            profit=self.profit,
            is_long=self.is_long,
            column_name=column,
            buy_sell="Buy",
            condition=condition,
            value=value,
            include_yearly=include_yearly,
            pnl_column=self.pnl_column,
        )


def build_confirm_context(
    strategy,
    frame: pd.DataFrame,
    days: int,
    profit: int,
    is_long: bool,
    pnl_column,
    symbol_data: dict,
    *,
    strategy_resolver=None,
    labels=None,
) -> ConfirmSweepContext | None:
    """A context for ``strategy`` on its prepared refine ``frame``, or None to use the slow path."""
    if not fast_sweep.NUMBA_AVAILABLE or days <= 0 or not strategy.confirm_symbols or symbol_data is None:
        return None
    primary = strategy.symbol.strip().upper()
    symbols = [primary, *[s for s in strategy.confirm_symbols if s != primary]]
    if any(symbol not in symbol_data for symbol in symbols) or len(symbols) < 2:
        return None
    try:
        if len(symbol_data[primary]) != len(frame):
            return None
        context = ConfirmSweepContext(
            strategy, frame, days, profit, is_long, pnl_column, symbols, symbol_data, strategy_resolver, labels
        )
        return context if context.base_matches_frame() else None
    except (ValueError, KeyError):
        return None
