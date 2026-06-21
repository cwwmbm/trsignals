import backtest as bt
from backtest_runners import load_ticker_data
from indicator_sweep import indicator_tryout

from api.builder_strategy import (
    backtest_builder_signal_sweep,
    builder_signal_callable,
    draft_to_saved_strategy,
)
from api.serializers import dataframe_records, detailed_backtest_payload
from api.signal_registry import get_signal, resolve_signal
from api.strategy_compiler import compile_buy_mask, compile_sell_mask, format_condition_preview
from api.proxy_symbol import with_proxy_description
from api.scan_service import execute_saved_strategy
from api.indicator_catalog import list_indicators
from api.strategy_store import (
    create_strategy,
    delete_strategy,
    get_strategy_by_id,
    list_strategies,
    update_strategy,
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


def run_builder_refine(request) -> dict | list[dict]:
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
            )
            return dataframe_records(results)

        signal = builder_signal_callable(
            primary,
            strategy_resolver=get_strategy_by_id,
            labels=labels,
        )

        if request.mode == "symbol-confirm-sweep":
            primary_symbol = (request.primary_symbol or symbol).strip().upper()
            symbol_pool = request.symbol_pool or [primary_symbol, "SMH", "QQQ"]
            results = bt.backtest_symbol_confirmation_sweep(
                signal,
                primary_symbol,
                symbol_pool,
                years=years,
            )
            return dataframe_records(results)

        if request.mode == "hold-days-sweep":
            data["Buy"], data["Sell"], _, _, _, _, is_long, _ = signal(data, symbol)
            results = bt.backtest_days(data, request.max_days, is_long)
            return dataframe_records(results)

        if request.mode == "indicator-sweep":
            data["Buy"], data["Sell"], days, profit, _, _, is_long, _ = signal(data, symbol)
            check_breadth = request.check_breadth if custom_dataset is None else False
            exclude_columns = set(custom_dataset.unavailable_indicator_ids) if custom_dataset else None
            results = indicator_tryout(
                data,
                days,
                profit,
                is_long,
                is_sell=request.is_sell,
                check_breadth=check_breadth,
                check_both=request.check_both,
                verbose=False,
                exclude_columns=exclude_columns,
                include_vwap_sweeps=custom_dataset is not None and custom_dataset.has_vwap,
            )
            return dataframe_records(results)

        raise ValueError(f"Unsupported refine mode: {request.mode}")

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


def upload_custom_dataset(content: bytes, filename: str | None = None) -> dict:
    from api.custom_data import custom_dataset_store

    dataset = custom_dataset_store.add_from_csv(content, filename=filename)
    return custom_dataset_store.metadata(dataset)


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
