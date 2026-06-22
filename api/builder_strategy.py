from __future__ import annotations

from collections.abc import Callable
from typing import Any

import backtest as bt
import indicators as ind
import pandas as pd

from api.indicator_catalog import list_indicators
from api.proxy_symbol import execute_with_proxy
from api.schemas import BuilderBacktestRequest, SavedStrategy
from api.session_masks import last_rth_bar_mask, regular_trading_hours_mask
from api.strategy_compiler import compile_buy_mask, compile_sell_mask, format_condition_preview
from api.strategy_store import _normalize_confirm_symbols, _normalize_proxy_symbol

StrategyResolver = Callable[[str], Any | None]


def _model_dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def draft_to_saved_strategy(
    request: BuilderBacktestRequest,
    *,
    labels: dict[str, str] | None = None,
) -> SavedStrategy:
    conditions = request.conditions
    labels = labels or {item["id"]: item["label"] for item in list_indicators(builder_only=True)}
    rule_preview = format_condition_preview(
        [_model_dump(condition) for condition in conditions],
        labels,
    )
    description = request.description.strip() or rule_preview
    if request.name.strip():
        description = f"{request.name.strip()}: {description}"

    symbol = request.symbol.strip().upper()
    intraday_session = bool(request.custom_dataset_id)
    return SavedStrategy(
        id="draft",
        name=request.name.strip() or "Untitled draft",
        symbol=symbol,
        direction=request.direction,
        hold_days=request.hold_days,
        profit=request.profit,
        description=description,
        conditions=conditions,
        sell_conditions=request.sell_conditions,
        confirm_symbols=_normalize_confirm_symbols(symbol, request.confirm_symbols),
        proxy_symbol=_normalize_proxy_symbol(symbol, request.proxy_symbol),
        hold_on_buy_signal=request.hold_on_buy_signal,
        rth_entries_only=request.rth_entries_only if intraday_session else False,
        eod_exit=request.eod_exit if intraday_session else False,
        created_at="",
        updated_at="",
    )


def _strategy_description(
    strategy: SavedStrategy,
    *,
    labels: dict[str, str] | None = None,
) -> str:
    if strategy.description.strip():
        return strategy.description.strip()
    labels = labels or {item["id"]: item["label"] for item in list_indicators(builder_only=True)}
    return format_condition_preview(
        [_model_dump(condition) for condition in strategy.conditions],
        labels,
    )


def _compile_strategy_masks(
    data: pd.DataFrame,
    strategy: SavedStrategy,
    *,
    strategy_resolver: StrategyResolver | None = None,
):
    conditions = [_model_dump(condition) for condition in strategy.conditions]
    sell_conditions = [_model_dump(condition) for condition in strategy.sell_conditions]
    buy = compile_buy_mask(data, conditions, strategy_resolver=strategy_resolver)
    sell = (
        compile_sell_mask(data, sell_conditions, strategy_resolver=strategy_resolver)
        if sell_conditions
        else False
    )
    if getattr(strategy, "rth_entries_only", False):
        source_timezone = data.attrs.get("timezone")
        buy = buy & regular_trading_hours_mask(
            data["Date"],
            source_timezone=source_timezone,
        )
    if getattr(strategy, "eod_exit", False):
        source_timezone = data.attrs.get("timezone")
        eod = last_rth_bar_mask(data["Date"], source_timezone=source_timezone)
        sell = eod if sell is False else sell | eod
    return buy, sell


def combine_builder_buy_masks(
    primary: SavedStrategy,
    secondary: SavedStrategy,
    data: pd.DataFrame,
    mode: str,
    *,
    strategy_resolver: StrategyResolver | None = None,
    labels: dict[str, str] | None = None,
):
    mode = mode.lower()
    if mode not in ("and", "or"):
        raise ValueError("mode must be 'and' or 'or'")

    p_buy, p_sell = _compile_strategy_masks(data, primary, strategy_resolver=strategy_resolver)
    s_buy, _ = _compile_strategy_masks(data, secondary, strategy_resolver=strategy_resolver)

    if mode == "and":
        buy = p_buy & s_buy
        join = " AND "
    else:
        buy = p_buy | s_buy
        join = " OR "

    p_desc = _strategy_description(primary, labels=labels)
    s_desc = _strategy_description(secondary, labels=labels)
    description = f"({p_desc}){join}({s_desc})"
    is_long = primary.direction == "long"
    return buy, p_sell, primary.hold_days, primary.profit, description, None, is_long, False


def builder_signal_callable(
    strategy: SavedStrategy,
    *,
    strategy_resolver: StrategyResolver | None = None,
    labels: dict[str, str] | None = None,
) -> Callable:
    def _signal(data, symbol):
        buy, sell = _compile_strategy_masks(
            data,
            strategy,
            strategy_resolver=strategy_resolver,
        )
        description = _strategy_description(strategy, labels=labels)
        is_long = strategy.direction == "long"
        return buy, sell, strategy.hold_days, strategy.profit, description, None, is_long, False

    _signal.__name__ = strategy.name or "draft"
    return _signal


def prepare_builder_refine_frame(
    strategy: SavedStrategy,
    *,
    years: int = 25,
    data: pd.DataFrame | None = None,
    strategy_resolver: StrategyResolver | None = None,
    labels: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, int, int, bool, str | None]:
    """
    Build a Buy/Sell frame for refine sweeps, honoring confirm symbols and proxy.
    Returns (data, hold_days, profit, is_long, pnl_column).
    """
    from api.proxy_symbol import proxy_column
    from backtest_runners import attach_proxy_column

    pnl_col = proxy_column(strategy)
    confirm_symbols = list(strategy.confirm_symbols or [])

    if confirm_symbols:
        signal = builder_signal_callable(
            strategy,
            strategy_resolver=strategy_resolver,
            labels=labels,
        )
        primary_symbol = strategy.symbol.strip().upper()
        needed = list(dict.fromkeys([primary_symbol, *confirm_symbols]))
        symbol_data = bt.load_symbol_dataset(needed, years=years)
        data, days, profit, _, _, is_long, _ = bt.apply_cross_symbol_signal(
            signal,
            primary_symbol,
            confirm_symbols,
            symbol_data,
        )
    else:
        if data is None:
            raise ValueError("Primary market data is required when strategy has no confirm symbols")
        buy, sell = _compile_strategy_masks(
            data,
            strategy,
            strategy_resolver=strategy_resolver,
        )
        data = data.copy()
        data["Buy"] = buy
        data["Sell"] = sell
        days = strategy.hold_days
        profit = strategy.profit
        is_long = strategy.direction == "long"

    if pnl_col:
        data = attach_proxy_column(data, pnl_col, years=years)

    return data, days, profit, is_long, pnl_col


def backtest_builder_signal_combinations(
    primary: SavedStrategy,
    secondary: SavedStrategy,
    data: pd.DataFrame,
    symbol: str,
    *,
    strategy_resolver: StrategyResolver | None = None,
    labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    signal_a = builder_signal_callable(
        primary,
        strategy_resolver=strategy_resolver,
        labels=labels,
    )
    signal_b = builder_signal_callable(
        secondary,
        strategy_resolver=strategy_resolver,
        labels=labels,
    )
    results = pd.DataFrame(
        columns=[
            "Primary",
            "Secondary",
            "Mode",
            "Days",
            "Profit",
            "PnL",
            "MaxDD",
            "Trades",
            "%Pstv",
            "CAGR",
            "Sharpe",
            "Sortino",
        ]
    )

    for p_strategy, s_strategy, mode in (
        (primary, secondary, "and"),
        (primary, secondary, "or"),
        (secondary, primary, "and"),
        (secondary, primary, "or"),
    ):
        data_copy = data.copy()
        buy, sell, days, profit, _, _, is_long, _ = combine_builder_buy_masks(
            p_strategy,
            s_strategy,
            data_copy,
            mode,
            strategy_resolver=strategy_resolver,
            labels=labels,
        )
        data_copy["Buy"] = buy
        data_copy["Sell"] = sell
        data_copy = execute_with_proxy(data_copy, primary, days, profit, is_long)
        m = bt._ranking_metrics(data_copy)

        results = results._append(
            {
                "Primary": p_strategy.name,
                "Secondary": s_strategy.name,
                    "SecondaryId": s_strategy.id,
                "Mode": mode.upper(),
                "Days": days,
                "Profit": profit,
                "PnL": m["PnL"],
                "MaxDD": m["MaxDD"],
                "Trades": m["Trades"],
                "%Pstv": m["%Pstv"],
                "CAGR": m["CAGR"],
                "Sharpe": m["Sharpe"],
                "Sortino": m["Sortino"],
                "Yearly": m["Yearly"],
            },
            ignore_index=True,
        )

    results = results.sort_values(by=["Sharpe"], ascending=False)
    results["PnL"] = results["PnL"].astype(int)
    results["MaxDD"] = results["MaxDD"].round(2)
    results["PnL"] = results["PnL"].apply(ind.format_dollar_value)
    results["MaxDD"] = results["MaxDD"].astype(str) + "%"
    results["Sharpe"] = pd.to_numeric(results["Sharpe"], errors="coerce")
    if not (results["Sharpe"].isnull().values.any() or pd.isna(results["Sharpe"]).any()):
        results["Sharpe"] = results["Sharpe"].round(2)
    results["Sortino"] = pd.to_numeric(results["Sortino"], errors="coerce")
    if not (results["Sortino"].isnull().values.any() or pd.isna(results["Sortino"]).any()):
        results["Sortino"] = results["Sortino"].round(2)
    results["%Pstv"] = pd.to_numeric(results["%Pstv"], errors="coerce")
    if not (results["%Pstv"].isnull().values.any() or pd.isna(results["%Pstv"]).any()):
        results["%Pstv"] = results["%Pstv"].round(1)
    results["Trades"] = results["Trades"].astype(int)
    results["Days"] = results["Days"].astype(int)
    results["Profit"] = results["Profit"].astype(int)
    return results


def backtest_builder_signal_sweep(
    primary: SavedStrategy,
    secondary_strategies: list[SavedStrategy],
    data: pd.DataFrame,
    symbol: str,
    *,
    strategy_resolver: StrategyResolver | None = None,
    labels: dict[str, str] | None = None,
    years: int = 25,
) -> pd.DataFrame:
    from api.proxy_symbol import proxy_column

    pnl_col = proxy_column(primary)
    if primary.confirm_symbols:
        base_data, days, profit, is_long, pnl_col = prepare_builder_refine_frame(
            primary,
            years=years,
            strategy_resolver=strategy_resolver,
            labels=labels,
        )
    else:
        base_data = data.copy()
        days = primary.hold_days
        profit = primary.profit
        is_long = primary.direction == "long"
        if pnl_col:
            from backtest_runners import attach_proxy_column

            base_data = attach_proxy_column(base_data, pnl_col, years=years)

    results = pd.DataFrame(
        columns=[
            "Primary",
            "Secondary",
            "Mode",
            "Days",
            "Profit",
            "PnL",
            "MaxDD",
            "Trades",
            "%Pstv",
            "CAGR",
            "Sharpe",
            "Sortino",
        ]
    )

    for secondary in secondary_strategies:
        for mode in ("and", "or"):
            data_copy = base_data.copy()
            if primary.confirm_symbols:
                s_buy, s_sell = _compile_strategy_masks(
                    data_copy,
                    secondary,
                    strategy_resolver=strategy_resolver,
                )
                p_buy = data_copy["Buy"]
                p_sell = data_copy["Sell"]
                buy = p_buy & s_buy if mode == "and" else p_buy | s_buy
                data_copy["Buy"] = buy
                data_copy["Sell"] = p_sell
            else:
                buy, sell, days, profit, _, _, is_long, _ = combine_builder_buy_masks(
                    primary,
                    secondary,
                    data_copy,
                    mode,
                    strategy_resolver=strategy_resolver,
                    labels=labels,
                )
                data_copy["Buy"] = buy
                data_copy["Sell"] = sell
            if pnl_col and pnl_col not in data_copy.columns:
                from backtest_runners import attach_proxy_column

                data_copy = attach_proxy_column(data_copy, pnl_col, years=years)
            data_copy = bt.execute_strategy(
                data_copy,
                days,
                profit,
                is_long,
                pnl_column=pnl_col,
            )
            m = bt._ranking_metrics(data_copy)

            results = results._append(
                {
                    "Primary": primary.name,
                    "Secondary": secondary.name,
                    "SecondaryId": secondary.id,
                    "Mode": mode.upper(),
                    "Days": days,
                    "Profit": profit,
                    "PnL": m["PnL"],
                    "MaxDD": m["MaxDD"],
                    "Trades": m["Trades"],
                    "%Pstv": m["%Pstv"],
                    "CAGR": m["CAGR"],
                    "Sharpe": m["Sharpe"],
                    "Sortino": m["Sortino"],
                    "Yearly": m["Yearly"],
                },
                ignore_index=True,
            )

    if results.empty:
        return results

    results = results.sort_values(by=["Sharpe"], ascending=False)
    results["PnL"] = results["PnL"].astype(int)
    results["MaxDD"] = results["MaxDD"].round(2)
    results["PnL"] = results["PnL"].apply(ind.format_dollar_value)
    results["MaxDD"] = results["MaxDD"].astype(str) + "%"
    results["Sharpe"] = pd.to_numeric(results["Sharpe"], errors="coerce")
    if not (results["Sharpe"].isnull().values.any() or pd.isna(results["Sharpe"]).any()):
        results["Sharpe"] = results["Sharpe"].round(2)
    results["Sortino"] = pd.to_numeric(results["Sortino"], errors="coerce")
    if not (results["Sortino"].isnull().values.any() or pd.isna(results["Sortino"]).any()):
        results["Sortino"] = results["Sortino"].round(2)
    results["%Pstv"] = pd.to_numeric(results["%Pstv"], errors="coerce")
    if not (results["%Pstv"].isnull().values.any() or pd.isna(results["%Pstv"]).any()):
        results["%Pstv"] = results["%Pstv"].round(1)
    results["Trades"] = results["Trades"].astype(int)
    results["Days"] = results["Days"].astype(int)
    results["Profit"] = results["Profit"].astype(int)
    return results
