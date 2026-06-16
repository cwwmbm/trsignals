from __future__ import annotations

from collections.abc import Callable
from typing import Any

import backtest as bt
import indicators as ind
import pandas as pd

from api.indicator_catalog import list_indicators
from api.schemas import BuilderBacktestRequest, SavedStrategy
from api.strategy_compiler import compile_buy_mask, compile_sell_mask, format_condition_preview

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

    return SavedStrategy(
        id="draft",
        name=request.name.strip() or "Untitled draft",
        symbol=request.symbol.strip().upper(),
        direction=request.direction,
        hold_days=request.hold_days,
        profit=request.profit,
        description=description,
        conditions=conditions,
        sell_conditions=request.sell_conditions,
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
        data_copy = bt.execute_strategy(data_copy, days, profit, is_long)
        m = bt._ranking_metrics(data_copy)

        results = results._append(
            {
                "Primary": p_strategy.name,
                "Secondary": s_strategy.name,
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
) -> pd.DataFrame:
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
            data_copy = data.copy()
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
            data_copy = bt.execute_strategy(data_copy, days, profit, is_long)
            m = bt._ranking_metrics(data_copy)

            results = results._append(
                {
                    "Primary": primary.name,
                    "Secondary": secondary.name,
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
