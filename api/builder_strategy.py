from __future__ import annotations

from collections.abc import Callable
from typing import Any

import backtest as bt
import indicators as ind
import numpy as np
import pandas as pd

from api.indicator_catalog import list_indicators
from api.proxy_symbol import execute_with_proxy
from api.schemas import BuilderBacktestRequest, BuilderCondition, SavedStrategy
from api.session_masks import last_rth_bar_mask, regular_trading_hours_mask
from api.strategy_compiler import (
    compile_buy_mask,
    compile_sell_mask,
    format_condition_preview,
    split_entry_conditions_for_confirm,
    uses_split_confirm_entry_filters,
)
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


def _clone_strategy_with_conditions(strategy: SavedStrategy, conditions) -> SavedStrategy:
    normalized = [
        condition if isinstance(condition, BuilderCondition) else BuilderCondition(**condition)
        for condition in conditions
    ]
    return strategy.copy(update={"conditions": normalized})


def apply_primary_entry_filters(
    data: pd.DataFrame,
    conditions: list[dict],
    *,
    strategy_resolver: StrategyResolver | None = None,
) -> pd.DataFrame:
    if not conditions:
        return data
    buy_filter = compile_buy_mask(data, conditions, strategy_resolver=strategy_resolver)
    data = data.copy()
    data["Buy"] = data["Buy"] & buy_filter
    return data


def _confirm_strategy_for_cross_symbol(
    strategy: SavedStrategy,
) -> tuple[SavedStrategy, list[dict]]:
    confirm_conditions, primary_filters = split_entry_conditions_for_confirm(strategy.conditions)
    if uses_split_confirm_entry_filters(strategy.conditions, strategy.confirm_symbols):
        return _clone_strategy_with_conditions(strategy, confirm_conditions), primary_filters
    return strategy, []


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
    symbol_data: dict | None = None,
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
        primary_symbol = strategy.symbol.strip().upper()
        needed = list(dict.fromkeys([primary_symbol, *confirm_symbols]))
        if symbol_data is None:
            symbol_data = bt.load_symbol_dataset(needed, years=years)
        confirm_strategy, primary_filters = _confirm_strategy_for_cross_symbol(strategy)
        signal = builder_signal_callable(
            confirm_strategy,
            strategy_resolver=strategy_resolver,
            labels=labels,
        )
        data, days, profit, _, _, is_long, _ = bt.apply_cross_symbol_signal(
            signal,
            primary_symbol,
            confirm_symbols,
            symbol_data,
        )
        data = apply_primary_entry_filters(
            data,
            primary_filters,
            strategy_resolver=strategy_resolver,
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


def _format_threshold_right(value) -> str:
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    return str(value).strip()


def _threshold_entry_condition(column_name: str, condition: str, value) -> BuilderCondition:
    operator = ">=" if condition == "more" else "<="
    return BuilderCondition(
        left=column_name,
        operator=operator,
        right=_format_threshold_right(value),
        logic="AND",
    )


def _ranking_row_from_frame(
    frame: pd.DataFrame,
    *,
    days: int,
    profit: int,
    is_long: bool,
    column_name: str,
    buy_sell: str,
    condition: str,
    value,
    include_yearly: bool,
    pnl_column: str | None,
) -> dict:
    executed = bt.execute_strategy(frame, days, profit, is_long, pnl_column=pnl_column)
    m = bt._ranking_metrics(executed, include_yearly=include_yearly)
    row = {
        "Buysell": buy_sell,
        "Indicator": column_name,
        "Condition": condition,
        "Value": value,
        "PnL": m["PnL"],
        "MaxDD": m["MaxDD"],
        "Trades": m["Trades"],
        "%Pstv": m["%Pstv"],
        "CAGR": m["CAGR"],
        "Sharpe": m["Sharpe"],
        "Sortino": m["Sortino"],
    }
    if include_yearly:
        row["Yearly"] = m["Yearly"]
    return row


def _run_confirm_embedded_buy_threshold(
    strategy: SavedStrategy,
    *,
    years: int,
    symbol_data: dict,
    column_name: str,
    condition: str,
    value,
    include_yearly: bool = False,
    strategy_resolver: StrategyResolver | None = None,
    labels: dict[str, str] | None = None,
    pnl_column: str | None = None,
) -> dict:
    """Evaluate a symbol-specific buy filter via full cross-symbol confirm path."""
    trial_conditions = [
        *strategy.conditions,
        _threshold_entry_condition(column_name, condition, value),
    ]
    trial = _clone_strategy_with_conditions(strategy, trial_conditions)
    frame, days, profit, is_long, pnl_col = prepare_builder_refine_frame(
        trial,
        years=years,
        symbol_data=symbol_data,
        strategy_resolver=strategy_resolver,
        labels=labels,
    )
    return _ranking_row_from_frame(
        frame,
        days=days,
        profit=profit,
        is_long=is_long,
        column_name=column_name,
        buy_sell="Buy",
        condition=condition,
        value=value,
        include_yearly=include_yearly,
        pnl_column=pnl_col or pnl_column,
    )


def builder_indicator_tryout(
    strategy: SavedStrategy,
    refine_frame: pd.DataFrame,
    days: int,
    profit: int,
    is_long: bool,
    *,
    years: int = 25,
    is_sell: bool = False,
    check_breadth: bool = True,
    check_both: bool = True,
    exclude_columns=None,
    include_vwap_sweeps: bool = False,
    pnl_column: str | None = None,
    strategy_resolver: StrategyResolver | None = None,
    labels: dict[str, str] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Builder refine indicator sweep.

    Market-wide buy filters are applied on the confirmed primary frame (post-merge).
    Symbol-specific buy filters are re-tested through the full confirm path so
    "Add to entry" matches backtest. Sell sweeps stay on the primary frame.
    """
    from api.indicator_catalog import is_market_wide_indicator
    from indicator_sweep import (
        BREADTH_SWEEPS,
        POST_2003_BREADTH_SWEEPS,
        PRICE_SWEEPS,
        VFI_EXCLUDED_TICKERS,
        VFI_SWEEPS,
        VWAP_SWEEPS,
        _collect_sweeps,
    )
    from config import ticker

    exclude_columns = set(exclude_columns or [])
    confirm_symbols = list(strategy.confirm_symbols or [])
    symbol_data = None
    if confirm_symbols and not is_sell:
        primary_symbol = strategy.symbol.strip().upper()
        needed = list(dict.fromkeys([primary_symbol, *confirm_symbols]))
        symbol_data = bt.load_symbol_dataset(needed, years=years)

    def _filter_specs(specs):
        if not exclude_columns:
            return specs
        return [
            spec
            for spec in specs
            if spec[0] not in exclude_columns and spec[1] not in exclude_columns
        ]

    def _split_specs(specs):
        market_wide = []
        embedded = []
        for spec in _filter_specs(specs):
            column = spec[1] if is_sell else spec[0]
            if (
                confirm_symbols
                and not is_sell
                and not is_market_wide_indicator(column)
            ):
                embedded.append(spec)
            else:
                market_wide.append(spec)
        return market_wide, embedded

    def _run_embedded_spec(spec) -> pd.DataFrame:
        buy_col, _sell_col, condition = spec[0], spec[1], spec[2]
        buy_min, buy_max, buy_step = spec[3], spec[4], spec[5]
        rows = []
        for value in np.arange(buy_min, buy_max + buy_step, buy_step):
            if condition == "both":
                rows.append(
                    _run_confirm_embedded_buy_threshold(
                        strategy,
                        years=years,
                        symbol_data=symbol_data,
                        column_name=buy_col,
                        condition="more",
                        value=value,
                        strategy_resolver=strategy_resolver,
                        labels=labels,
                        pnl_column=pnl_column,
                    )
                )
                cond = "less"
            else:
                cond = condition
            rows.append(
                _run_confirm_embedded_buy_threshold(
                    strategy,
                    years=years,
                    symbol_data=symbol_data,
                    column_name=buy_col,
                    condition=cond,
                    value=value,
                    strategy_resolver=strategy_resolver,
                    labels=labels,
                    pnl_column=pnl_column,
                )
            )
        return bt._format_ranking_results(pd.DataFrame(rows))

    def _add_yearly_embedded(results: pd.DataFrame) -> pd.DataFrame:
        if results.empty:
            return results
        results = results.copy()
        yearly = []
        for _, row in results.iterrows():
            detail = _run_confirm_embedded_buy_threshold(
                strategy,
                years=years,
                symbol_data=symbol_data,
                column_name=row["Indicator"],
                condition=row["Condition"],
                value=row["Value"],
                include_yearly=True,
                strategy_resolver=strategy_resolver,
                labels=labels,
                pnl_column=pnl_column,
            )
            yearly.append(detail["Yearly"])
        results["Yearly"] = yearly
        return results

    running_rows: list[dict] = []
    og = days == 0
    all_spec_groups = []
    if check_breadth:
        all_spec_groups.append(BREADTH_SWEEPS)
        if refine_frame["Date"].dt.year.iloc[0] >= 2003:
            all_spec_groups.append(POST_2003_BREADTH_SWEEPS)
    if check_both or not check_breadth:
        all_spec_groups.append(PRICE_SWEEPS)
        if ticker not in VFI_EXCLUDED_TICKERS:
            all_spec_groups.append(VFI_SWEEPS)
        if include_vwap_sweeps and "VWAP" in refine_frame.columns and refine_frame["VWAP"].notna().any():
            all_spec_groups.append(VWAP_SWEEPS)

    for specs in all_spec_groups:
        market_wide_specs, embedded_specs = _split_specs(specs)
        if market_wide_specs:
            running_rows = _collect_sweeps(
                running_rows,
                refine_frame,
                days,
                profit,
                is_long,
                is_sell,
                og,
                market_wide_specs,
                verbose,
                False,
                pnl_column=pnl_column,
            )
        for spec in embedded_specs:
            results = _run_embedded_spec(spec)
            top_results = _add_yearly_embedded(results.head(3))
            if verbose:
                print(results.head(5))
            running_rows.extend(top_results.to_dict(orient="records"))

    running_results = pd.DataFrame(running_rows)
    if not running_results.empty:
        running_results = running_results.sort_values(by=["Sharpe"], ascending=False)
    if verbose:
        print(running_results)
    return running_results


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
