import backtest as bt
import config
from backtest_runners import load_ticker_data
from indicator_sweep import indicator_tryout

from api.builder_strategy import (
    _compile_strategy_masks,
    _run_confirm_embedded_buy_threshold,
    backtest_builder_signal_sweep,
    builder_indicator_tryout,
    builder_signal_callable,
    combine_builder_buy_masks,
    draft_to_saved_strategy,
    prepare_builder_refine_frame,
)
from api.indicator_catalog import is_market_wide_indicator, list_indicators
from api.proxy_symbol import proxy_column, with_proxy_description
from api.sample_window import (
    in_sample_end_timestamp,
    sample_window_meta,
    slice_frame_to_end,
    slice_symbol_data_to_end,
)
from api.serializers import dataframe_records, detailed_backtest_payload
from api.signal_registry import get_signal, resolve_signal
from api.strategy_compiler import compile_buy_mask, compile_sell_mask, format_condition_preview
from api.scan_service import execute_saved_strategy
from api.strategy_store import (
    create_strategy,
    delete_strategy,
    get_strategy_by_id,
    list_strategies,
    update_strategy,
)
from api.portfolio_store import (
    create_portfolio,
    delete_portfolio,
    list_portfolios,
    update_portfolio,
)
from api.scan_service import run_scan


def _builder_condition_labels() -> dict[str, str]:
    labels = {item["id"]: item["label"] for item in list_indicators(builder_only=True)}
    labels.update(
        {
            f"strategy:{strategy.id}": strategy.name
            for strategy in list_strategies()
        }
    )
    return labels


def _model_dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _apply_runtime_options(request):
    options = getattr(request, "runtime_options", None)
    original = {
        "MondayBuy": bt.MondayBuy,
        "LowVolumeBuy": bt.LowVolumeBuy,
        "HoldOnBuySignal": bt.HoldOnBuySignal,
    }
    if options is not None:
        if options.monday_buy is not None:
            bt.MondayBuy = options.monday_buy
        if options.low_volume_buy is not None:
            bt.LowVolumeBuy = options.low_volume_buy
        if options.hold_on_buy_signal is not None:
            bt.HoldOnBuySignal = options.hold_on_buy_signal
    return original


def _restore_runtime_options(original):
    bt.MondayBuy = original["MondayBuy"]
    bt.LowVolumeBuy = original["LowVolumeBuy"]
    bt.HoldOnBuySignal = original["HoldOnBuySignal"]


def _with_builder_hold_on_buy(request, custom_dataset, fn):
    if custom_dataset is not None:
        return fn()
    original = bt.HoldOnBuySignal
    bt.HoldOnBuySignal = bool(getattr(request, "hold_on_buy_signal", False))
    try:
        return fn()
    finally:
        bt.HoldOnBuySignal = original


def _with_runtime_options(request, fn):
    original = _apply_runtime_options(request)
    try:
        return fn()
    finally:
        _restore_runtime_options(original)


def run_single_backtest(request) -> dict:
    def _run():
        signal = resolve_signal(request.signal.model_dump())
        data = load_ticker_data(request.symbol, years=request.years)
        data["Buy"], data["Sell"], days, profit, description, _, is_long, _ = signal(
            data, request.symbol
        )
        executed = bt.execute_strategy(data, days, profit, is_long)
        return detailed_backtest_payload(executed, days, profit, description)

    return _with_runtime_options(request, _run)


def run_signal_combo_sweep(request) -> list[dict]:
    def _run():
        signal_a = get_signal(request.signal_a)
        signal_b = get_signal(request.signal_b)
        data = load_ticker_data(request.symbol, years=request.years)
        results = bt.backtest_signal_combinations(signal_a, signal_b, data, request.symbol)
        return dataframe_records(results)

    return _with_runtime_options(request, _run)


def run_symbol_confirm_sweep(request) -> list[dict]:
    def _run():
        signal = resolve_signal(request.signal.model_dump())
        results = bt.backtest_symbol_confirmation_sweep(
            signal,
            request.primary_symbol,
            request.symbol_pool,
            years=request.years,
        )
        return dataframe_records(results)

    return _with_runtime_options(request, _run)


def run_symbol_confirm_detail(request) -> dict:
    def _run():
        signal = resolve_signal(request.signal.model_dump())
        data, days, profit, description, _ = bt.backtest_cross_symbol(
            signal,
            request.primary_symbol,
            request.confirm_symbols,
            years=request.years,
        )
        return detailed_backtest_payload(data, days, profit, description)

    return _with_runtime_options(request, _run)


def run_hold_days_sweep(request) -> list[dict]:
    def _run():
        signal = resolve_signal(request.signal.model_dump())
        data = load_ticker_data(request.symbol, years=request.years)
        data["Buy"], data["Sell"], _, _, _, _, is_long, _ = signal(data, request.symbol)
        results = bt.backtest_days(data, request.max_days, is_long)
        return dataframe_records(results)

    return _with_runtime_options(request, _run)


def run_indicator_sweep(request) -> list[dict]:
    def _run():
        signal = resolve_signal(request.signal.model_dump())
        data = load_ticker_data(request.symbol, years=request.years)
        data["Buy"], data["Sell"], days, profit, _, _, is_long, _ = signal(data, request.symbol)
        results = indicator_tryout(
            data,
            days,
            profit,
            is_long,
            is_sell=request.is_sell,
            check_breadth=request.check_breadth,
            check_both=request.check_both,
            verbose=False,
        )
        return dataframe_records(results)

    return _with_runtime_options(request, _run)


def _load_builder_data(strategy_request):
    from api.custom_data import custom_dataset_store

    custom_dataset_id = getattr(strategy_request, "custom_dataset_id", None)
    if custom_dataset_id:
        custom_dataset = custom_dataset_store.require(custom_dataset_id)
        backtest_all_data = bool(getattr(strategy_request, "backtest_all_data", False))
        data = custom_dataset_store.load_backtest_frame(
            custom_dataset_id,
            backtest_all_data=backtest_all_data,
        )
        return data, custom_dataset
    symbol = strategy_request.symbol.strip().upper()
    years = strategy_request.years
    return load_ticker_data(symbol, years=years), None


def run_builder_backtest(request) -> dict:
    from api.schemas import SavedStrategy
    from api.strategy_store import _normalize_confirm_symbols, _normalize_proxy_symbol

    custom_dataset = None
    if request.custom_dataset_id:
        if request.confirm_symbols:
            raise ValueError("Confirm symbols are not supported with custom intraday data")
        if request.proxy_symbol and str(request.proxy_symbol).strip():
            raise ValueError("Proxy symbol is not supported with custom intraday data")
        data, custom_dataset = _load_builder_data(request)
    else:
        data, _ = _load_builder_data(request)

    conditions = [_model_dump(condition) for condition in request.conditions]
    sell_conditions = [_model_dump(condition) for condition in request.sell_conditions]
    compile_buy_mask(data, conditions, strategy_resolver=get_strategy_by_id)
    if sell_conditions:
        compile_sell_mask(data, sell_conditions, strategy_resolver=get_strategy_by_id)

    labels = _builder_condition_labels()
    rule_preview = format_condition_preview(conditions, labels)
    description = request.description.strip() or rule_preview
    if request.name.strip():
        description = f"{request.name.strip()}: {description}"

    symbol = request.symbol.strip().upper()
    if custom_dataset is not None:
        symbol = custom_dataset.symbol
    confirm_symbols = _normalize_confirm_symbols(symbol, request.confirm_symbols)
    session_flags = (
        {
            "rth_entries_only": request.rth_entries_only,
            "eod_exit": request.eod_exit,
        }
        if custom_dataset is not None
        else {
            "rth_entries_only": False,
            "eod_exit": False,
        }
    )
    strategy = SavedStrategy(
        id="preview",
        name=request.name,
        symbol=symbol,
        direction=request.direction,
        hold_days=request.hold_days,
        profit=request.profit,
        description=description,
        conditions=request.conditions,
        sell_conditions=request.sell_conditions,
        confirm_symbols=confirm_symbols,
        proxy_symbol=_normalize_proxy_symbol(symbol, request.proxy_symbol),
        hold_on_buy_signal=request.hold_on_buy_signal,
        **session_flags,
        created_at="",
        updated_at="",
    )

    if custom_dataset is not None:
        executed = execute_saved_strategy(data, strategy, years=request.years)
        final_description = description
        return detailed_backtest_payload(
            executed,
            strategy.hold_days,
            strategy.profit,
            final_description,
            periods_per_year=custom_dataset.periods_per_year,
            is_intraday=True,
        )

    def _run():
        executed = execute_saved_strategy(data, strategy, years=request.years)
        final_description = with_proxy_description(description, strategy)
        return detailed_backtest_payload(
            executed,
            strategy.hold_days,
            strategy.profit,
            final_description,
        )

    return _with_builder_hold_on_buy(request, custom_dataset, _run)


def _refine_reference_frame(primary, data, *, years: int, labels: dict[str, str]):
    """Full-period frame used to compute the in-sample cutoff."""
    if primary.confirm_symbols:
        return prepare_builder_refine_frame(
            primary,
            years=years,
            strategy_resolver=get_strategy_by_id,
            labels=labels,
        )[0]
    if data is None:
        raise ValueError("Primary market data is required when strategy has no confirm symbols")
    return data


def _format_outcome_metrics(metrics: dict) -> dict:
    import indicators as ind

    pnl = metrics["PnL"]
    max_dd = metrics["MaxDD"]
    return {
        "PnL": ind.format_dollar_value(int(pnl)),
        "MaxDD": f"{round(float(max_dd), 2)}%",
        "Trades": int(metrics["Trades"]),
        "%Pstv": round(float(metrics["%Pstv"]), 1),
        "CAGR": metrics["CAGR"],
        "Sharpe": round(float(metrics["Sharpe"]), 2)
        if metrics["Sharpe"] == metrics["Sharpe"]
        else metrics["Sharpe"],
        "Sortino": round(float(metrics["Sortino"]), 2)
        if metrics["Sortino"] == metrics["Sortino"]
        else metrics["Sortino"],
        "Yearly": metrics.get("Yearly") or [],
    }


def _full_period_summary(metrics: dict) -> dict:
    formatted = _format_outcome_metrics(metrics)
    return {
        "PnL": formatted["PnL"],
        "MaxDD": formatted["MaxDD"],
        "Trades": formatted["Trades"],
        "Sharpe": formatted["Sharpe"],
        "Yearly": formatted["Yearly"],
    }


def _evaluate_refine_row_metrics(
    *,
    mode: str,
    primary,
    row: dict,
    data,
    years: int,
    labels: dict[str, str],
    use_end,
    primary_symbol: str | None = None,
    cache: dict | None = None,
) -> dict:
    """Return raw ranking metrics (including Yearly) for one refine row."""
    cache = cache if cache is not None else {}
    pnl_col = proxy_column(primary)
    row = dict(row or {})

    if mode == "indicator-sweep":
        column = str(row.get("Indicator") or "")
        condition = str(row.get("Condition") or "").lower()
        buy_sell = str(row.get("Buysell") or row.get("BuySell") or "Buy")
        value = row.get("Value")
        if not column or condition not in {"more", "less"} or value is None:
            raise ValueError("Indicator sweep outcome requires Indicator, Condition, and Value")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid indicator Value: {value}") from exc

        if (
            buy_sell == "Buy"
            and primary.confirm_symbols
            and not is_market_wide_indicator(column)
        ):
            cache_key = ("confirm_symbol_data", use_end)
            symbol_data = cache.get(cache_key)
            if symbol_data is None:
                needed = list(
                    dict.fromkeys(
                        [primary.symbol.strip().upper(), *primary.confirm_symbols]
                    )
                )
                symbol_data = bt.load_symbol_dataset(needed, years=years)
                if use_end is not None:
                    symbol_data = slice_symbol_data_to_end(symbol_data, use_end)
                cache[cache_key] = symbol_data
            detail = _run_confirm_embedded_buy_threshold(
                primary,
                years=years,
                symbol_data=symbol_data,
                column_name=column,
                condition=condition,
                value=value,
                include_yearly=True,
                strategy_resolver=get_strategy_by_id,
                labels=labels,
                pnl_column=pnl_col,
            )
            return {
                "PnL": detail["PnL"],
                "MaxDD": detail["MaxDD"],
                "Trades": detail["Trades"],
                "%Pstv": detail["%Pstv"],
                "CAGR": detail["CAGR"],
                "Sharpe": detail["Sharpe"],
                "Sortino": detail["Sortino"],
                "Yearly": detail.get("Yearly") or [],
            }

        cache_key = ("refine_frame", use_end)
        packed = cache.get(cache_key)
        if packed is None:
            packed = prepare_builder_refine_frame(
                primary,
                years=years,
                data=data,
                in_sample_end=use_end,
                strategy_resolver=get_strategy_by_id,
                labels=labels,
            )
            cache[cache_key] = packed
        refine_frame, days, profit, is_long, frame_pnl = packed
        detail = bt._run_indicator_threshold(
            refine_frame,
            days,
            profit,
            is_long,
            column,
            buy_sell,
            condition,
            value,
            include_yearly=True,
            pnl_column=frame_pnl or pnl_col,
        )
        return {
            "PnL": detail["PnL"],
            "MaxDD": detail["MaxDD"],
            "Trades": detail["Trades"],
            "%Pstv": detail["%Pstv"],
            "CAGR": detail["CAGR"],
            "Sharpe": detail["Sharpe"],
            "Sortino": detail["Sortino"],
            "Yearly": detail.get("Yearly") or [],
        }

    if mode == "hold-days-sweep":
        hold_days = int(row.get("Days") or primary.hold_days)
        profit = int(row.get("Prf") if row.get("Prf") is not None else primary.profit)
        cache_key = ("refine_frame", use_end)
        packed = cache.get(cache_key)
        if packed is None:
            packed = prepare_builder_refine_frame(
                primary,
                years=years,
                data=data,
                in_sample_end=use_end,
                strategy_resolver=get_strategy_by_id,
                labels=labels,
            )
            cache[cache_key] = packed
        refine_frame, _, _, is_long, frame_pnl = packed
        executed = bt.execute_strategy(
            refine_frame,
            hold_days,
            profit,
            is_long,
            pnl_column=frame_pnl or pnl_col,
        )
        return bt._ranking_metrics(executed, include_yearly=True)

    if mode == "symbol-confirm-sweep":
        symbol = (primary_symbol or primary.symbol).strip().upper()
        confirm_raw = str(row.get("Confirm") or "").strip()
        if not confirm_raw or confirm_raw == "(none)":
            confirm_symbols = []
        else:
            confirm_symbols = [
                part.strip().upper()
                for part in confirm_raw.split("+")
                if part.strip()
            ]
        signal = builder_signal_callable(
            primary,
            strategy_resolver=get_strategy_by_id,
            labels=labels,
        )
        needed = list(dict.fromkeys([symbol, *confirm_symbols]))
        cache_key = ("symbol_data", tuple(needed), use_end)
        symbol_data = cache.get(cache_key)
        if symbol_data is None:
            symbol_data = bt.load_symbol_dataset(needed, years=years)
            if use_end is not None:
                symbol_data = slice_symbol_data_to_end(symbol_data, use_end)
            cache[cache_key] = symbol_data
        frame, days, profit, _, _, is_long, _ = bt.apply_cross_symbol_signal(
            signal,
            symbol,
            confirm_symbols,
            symbol_data,
        )
        if pnl_col:
            from backtest_runners import attach_proxy_column

            frame = attach_proxy_column(frame, pnl_col, years=years)
        executed = bt.execute_strategy(
            frame, days, profit, is_long, pnl_column=pnl_col
        )
        return bt._ranking_metrics(executed, include_yearly=True)

    if mode == "signal-combo-sweep":
        secondary_id = str(row.get("SecondaryId") or "").strip()
        combo_mode = str(row.get("Mode") or "AND").lower()
        if combo_mode not in {"and", "or"}:
            raise ValueError("Combo outcome requires Mode AND or OR")
        secondary = get_strategy_by_id(secondary_id) if secondary_id else None
        if secondary is None:
            raise ValueError(f"Unknown secondary strategy: {secondary_id}")

        if primary.confirm_symbols:
            cache_key = ("refine_frame", use_end)
            packed = cache.get(cache_key)
            if packed is None:
                packed = prepare_builder_refine_frame(
                    primary,
                    years=years,
                    in_sample_end=use_end,
                    strategy_resolver=get_strategy_by_id,
                    labels=labels,
                )
                cache[cache_key] = packed
            refine_frame, days, profit, is_long, frame_pnl = packed
            data_copy = refine_frame.copy()
            s_buy, _s_sell = _compile_strategy_masks(
                data_copy,
                secondary,
                strategy_resolver=get_strategy_by_id,
            )
            buy = data_copy["Buy"] & s_buy if combo_mode == "and" else data_copy["Buy"] | s_buy
            data_copy["Buy"] = buy
            executed = bt.execute_strategy(
                data_copy,
                days,
                profit,
                is_long,
                pnl_column=frame_pnl or pnl_col,
            )
        else:
            frame_data = data if use_end is None else slice_frame_to_end(data, use_end)
            data_copy = frame_data.copy()
            buy, sell, days, profit, _, _, is_long, _ = combine_builder_buy_masks(
                primary,
                secondary,
                data_copy,
                combo_mode,
                strategy_resolver=get_strategy_by_id,
                labels=labels,
            )
            data_copy["Buy"] = buy
            data_copy["Sell"] = sell
            if pnl_col:
                from backtest_runners import attach_proxy_column

                data_copy = attach_proxy_column(data_copy, pnl_col, years=years)
            executed = bt.execute_strategy(
                data_copy, days, profit, is_long, pnl_column=pnl_col
            )
        return bt._ranking_metrics(executed, include_yearly=True)

    raise ValueError(f"Unsupported refine mode: {mode}")


def _row_trade_count(row: dict) -> int:
    value = row.get("Trades")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _filter_rows_by_min_in_sample_trades(rows, minimum: int | None = None) -> list[dict]:
    min_trades = config.MIN_IN_SAMPLE_TRADES if minimum is None else int(minimum)
    row_list = rows if isinstance(rows, list) else dataframe_records(rows)
    return [row for row in row_list if _row_trade_count(row) >= min_trades]


def _attach_full_period_to_rows(
    rows,
    *,
    mode: str,
    primary,
    data,
    years: int,
    labels: dict[str, str],
    primary_symbol: str | None = None,
) -> list[dict]:
    row_list = _filter_rows_by_min_in_sample_trades(rows)
    cache: dict = {}
    for row in row_list:
        if row.get("Yearly") is None:
            continue
        metrics = _evaluate_refine_row_metrics(
            mode=mode,
            primary=primary,
            row=row,
            data=data,
            years=years,
            labels=labels,
            use_end=None,
            primary_symbol=primary_symbol,
            cache=cache,
        )
        row["FullPeriod"] = _full_period_summary(metrics)
    return row_list


def _wrap_refine_rows(rows, full_frame, *, in_sample_end) -> dict:
    meta = sample_window_meta(
        full_frame,
        sample="in_sample",
        in_sample_end=in_sample_end,
    )
    meta["min_in_sample_trades"] = int(config.MIN_IN_SAMPLE_TRADES)
    return {
        "rows": rows if isinstance(rows, list) else dataframe_records(rows),
        "meta": meta,
    }

def run_builder_refine(request) -> dict:
    if not request.strategy.conditions:
        raise ValueError("Draft strategy must have at least one entry condition")

    labels = _builder_condition_labels()
    primary = draft_to_saved_strategy(request.strategy, labels=labels)
    symbol = primary.symbol
    years = request.strategy.years

    data, custom_dataset = _load_builder_data(request.strategy)
    if custom_dataset is not None and request.mode == "symbol-confirm-sweep":
        raise ValueError("Symbol confirmation sweep is not supported with custom intraday data")

    def _run():
        full_frame = _refine_reference_frame(primary, data, years=years, labels=labels)
        in_sample_end = in_sample_end_timestamp(full_frame)
        pnl_col = proxy_column(primary)

        if request.mode == "signal-combo-sweep":
            secondary_strategies = [
                strategy
                for strategy in list_strategies()
                if strategy.symbol.strip().upper() == symbol
            ]
            if not secondary_strategies:
                raise ValueError(f"No saved strategies found for {symbol}")
            results = backtest_builder_signal_sweep(
                primary,
                secondary_strategies,
                data,
                symbol,
                strategy_resolver=get_strategy_by_id,
                labels=labels,
                years=years,
                in_sample_end=in_sample_end,
            )
            rows = _attach_full_period_to_rows(
                results,
                mode=request.mode,
                primary=primary,
                data=data,
                years=years,
                labels=labels,
            )
            return _wrap_refine_rows(rows, full_frame, in_sample_end=in_sample_end)

        refine_frame, days, profit, is_long, pnl_col = prepare_builder_refine_frame(
            primary,
            years=years,
            data=data,
            in_sample_end=in_sample_end,
            strategy_resolver=get_strategy_by_id,
            labels=labels,
        )

        if request.mode == "symbol-confirm-sweep":
            primary_symbol = (request.primary_symbol or symbol).strip().upper()
            symbol_pool = request.symbol_pool or [primary_symbol, "SMH", "QQQ"]
            if primary.confirm_symbols:
                symbol_pool = list(
                    dict.fromkeys([*symbol_pool, primary_symbol, *primary.confirm_symbols])
                )
            signal = builder_signal_callable(
                primary,
                strategy_resolver=get_strategy_by_id,
                labels=labels,
            )
            results = bt.backtest_symbol_confirmation_sweep(
                signal,
                primary_symbol,
                symbol_pool,
                years=years,
                pnl_column=pnl_col,
                in_sample_end=in_sample_end,
            )
            rows = _attach_full_period_to_rows(
                results,
                mode=request.mode,
                primary=primary,
                data=data,
                years=years,
                labels=labels,
                primary_symbol=primary_symbol,
            )
            return _wrap_refine_rows(rows, full_frame, in_sample_end=in_sample_end)

        if request.mode == "hold-days-sweep":
            results = bt.backtest_days(
                refine_frame,
                request.max_days,
                is_long,
                pnl_column=pnl_col,
            )
            rows = _attach_full_period_to_rows(
                results,
                mode=request.mode,
                primary=primary,
                data=data,
                years=years,
                labels=labels,
            )
            return _wrap_refine_rows(rows, full_frame, in_sample_end=in_sample_end)

        if request.mode == "indicator-sweep":
            check_breadth = request.check_breadth if custom_dataset is None else False
            exclude_columns = set(custom_dataset.unavailable_indicator_ids) if custom_dataset else None
            if primary.confirm_symbols:
                results = builder_indicator_tryout(
                    primary,
                    refine_frame,
                    days,
                    profit,
                    is_long,
                    years=years,
                    is_sell=request.is_sell,
                    check_breadth=check_breadth,
                    check_both=request.check_both,
                    exclude_columns=exclude_columns,
                    include_vwap_sweeps=custom_dataset is not None and custom_dataset.has_vwap,
                    pnl_column=pnl_col,
                    in_sample_end=in_sample_end,
                    strategy_resolver=get_strategy_by_id,
                    labels=labels,
                    verbose=False,
                )
            else:
                results = indicator_tryout(
                    refine_frame,
                    days,
                    profit,
                    is_long,
                    is_sell=request.is_sell,
                    check_breadth=check_breadth,
                    check_both=request.check_both,
                    verbose=False,
                    exclude_columns=exclude_columns,
                    include_vwap_sweeps=custom_dataset is not None and custom_dataset.has_vwap,
                    pnl_column=pnl_col,
                )
            rows = _attach_full_period_to_rows(
                results,
                mode=request.mode,
                primary=primary,
                data=data,
                years=years,
                labels=labels,
            )
            return _wrap_refine_rows(rows, full_frame, in_sample_end=in_sample_end)

        raise ValueError(f"Unsupported refine mode: {request.mode}")

    return _with_builder_hold_on_buy(request.strategy, custom_dataset, _run)


def run_builder_refine_outcome(request) -> dict:
    """Evaluate one refine sweep row on in-sample or full period."""
    if not request.strategy.conditions:
        raise ValueError("Draft strategy must have at least one entry condition")

    labels = _builder_condition_labels()
    primary = draft_to_saved_strategy(request.strategy, labels=labels)
    years = request.strategy.years
    row = dict(request.row or {})
    data, custom_dataset = _load_builder_data(request.strategy)
    if custom_dataset is not None and request.mode == "symbol-confirm-sweep":
        raise ValueError("Symbol confirmation sweep is not supported with custom intraday data")

    def _run():
        full_frame = _refine_reference_frame(primary, data, years=years, labels=labels)
        in_sample_end = in_sample_end_timestamp(full_frame)
        use_end = in_sample_end if request.sample == "in_sample" else None
        meta = sample_window_meta(
            full_frame,
            sample=request.sample,
            in_sample_end=in_sample_end,
        )
        metrics = _evaluate_refine_row_metrics(
            mode=request.mode,
            primary=primary,
            row=row,
            data=data,
            years=years,
            labels=labels,
            use_end=use_end,
            primary_symbol=request.primary_symbol,
        )
        return {"metrics": _format_outcome_metrics(metrics), "meta": meta}

    return _with_builder_hold_on_buy(request.strategy, custom_dataset, _run)


def save_strategy(request) -> dict:
    conditions = [_model_dump(condition) for condition in request.conditions]
    sell_conditions = [_model_dump(condition) for condition in request.sell_conditions]
    data = load_ticker_data(request.symbol.strip().upper(), years=1)
    compile_buy_mask(data, conditions, strategy_resolver=get_strategy_by_id)
    if sell_conditions:
        compile_sell_mask(data, sell_conditions, strategy_resolver=get_strategy_by_id)
    saved = create_strategy(request)
    return _model_dump(saved)


def update_saved_strategy(strategy_id: str, request) -> dict:
    if hasattr(request, "model_dump"):
        payload = request.model_dump(exclude_unset=True)
    else:
        payload = request.dict(exclude_unset=True)
    trading_keys = {
        "name",
        "symbol",
        "direction",
        "hold_days",
        "profit",
        "conditions",
        "sell_conditions",
        "confirm_symbols",
        "proxy_symbol",
        "hold_on_buy_signal",
        "rth_entries_only",
        "eod_exit",
    }
    if trading_keys & set(payload.keys()):
        existing = get_strategy_by_id(strategy_id)
        if existing is None:
            raise ValueError(f"Unknown strategy: {strategy_id}")
        symbol = str(payload.get("symbol") or existing.symbol).strip().upper()
        conditions = payload.get("conditions")
        if conditions is None:
            conditions = [_model_dump(condition) for condition in existing.conditions]
        sell_conditions = payload.get("sell_conditions")
        if sell_conditions is None:
            sell_conditions = [_model_dump(condition) for condition in existing.sell_conditions]
        data = load_ticker_data(symbol, years=1)
        compile_buy_mask(data, conditions, strategy_resolver=get_strategy_by_id)
        if sell_conditions:
            compile_sell_mask(data, sell_conditions, strategy_resolver=get_strategy_by_id)

    updated = update_strategy(strategy_id, request)
    if updated is None:
        raise ValueError(f"Unknown strategy: {strategy_id}")
    return _model_dump(updated)


def delete_saved_strategy(strategy_id: str) -> dict:
    deleted = delete_strategy(strategy_id)
    if not deleted:
        raise ValueError(f"Unknown strategy: {strategy_id}")
    return {"deleted": True}


def list_saved_strategies() -> list[dict]:
    return [_model_dump(strategy) for strategy in list_strategies()]


def run_live_scan() -> list[dict]:
    return run_scan()


def run_portfolio_simulation(request) -> dict:
    from api.portfolio_service import simulate_portfolio

    return simulate_portfolio(request)


def run_portfolio_shapley(request) -> dict:
    from api.portfolio_service import simulate_portfolio_shapley

    return simulate_portfolio_shapley(request)


def save_portfolio(request) -> dict:
    saved = create_portfolio(request)
    return _model_dump(saved)


def update_saved_portfolio(portfolio_id: str, request) -> dict:
    updated = update_portfolio(portfolio_id, request)
    if updated is None:
        raise ValueError(f"Unknown portfolio: {portfolio_id}")
    return _model_dump(updated)


def delete_saved_portfolio(portfolio_id: str) -> dict:
    deleted = delete_portfolio(portfolio_id)
    if not deleted:
        raise ValueError(f"Unknown portfolio: {portfolio_id}")
    return {"deleted": True}


def list_saved_portfolios() -> list[dict]:
    return [_model_dump(portfolio) for portfolio in list_portfolios()]


def upload_custom_dataset(content: bytes, filename: str | None = None) -> dict:
    from api.custom_data import custom_dataset_store

    dataset = custom_dataset_store.add_from_csv(content, filename=filename)
    return custom_dataset_store.metadata(dataset)


def run_monte_carlo_simulation(request) -> dict:
    from api.monte_carlo import run_monte_carlo

    return run_monte_carlo(
        request.trade_returns,
        method=request.method,
        n_sims=request.n_sims,
        start_capital=request.start_capital,
    )


def get_custom_dataset_metadata(dataset_id: str) -> dict:
    from api.custom_data import custom_dataset_store

    dataset = custom_dataset_store.require(dataset_id)
    return custom_dataset_store.metadata(dataset)


def delete_custom_dataset(dataset_id: str) -> dict:
    from api.custom_data import custom_dataset_store

    deleted = custom_dataset_store.delete(dataset_id)
    if not deleted:
        raise ValueError(f"Custom dataset not found: {dataset_id}")
    return {"deleted": True}
