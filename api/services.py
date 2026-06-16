import backtest as bt
from backtest_runners import load_ticker_data
from indicator_sweep import indicator_tryout

from api.builder_strategy import (
    backtest_builder_signal_combinations,
    builder_signal_callable,
    draft_to_saved_strategy,
)
from api.serializers import dataframe_records, detailed_backtest_payload
from api.signal_registry import get_signal, resolve_signal
from api.strategy_compiler import compile_buy_mask, compile_sell_mask, format_condition_preview
from api.scan_service import execute_saved_strategy
from api.indicator_catalog import list_indicators
from api.strategy_store import create_strategy, get_strategy_by_id, list_strategies
from api.scan_service import run_scan


def _apply_runtime_options(request):
    options = getattr(request, "runtime_options", None)
    original = {
        "MondayBuy": bt.MondayBuy,
        "LowVolumeBuy": bt.LowVolumeBuy,
    }
    if options is not None:
        if options.monday_buy is not None:
            bt.MondayBuy = options.monday_buy
        if options.low_volume_buy is not None:
            bt.LowVolumeBuy = options.low_volume_buy
    return original


def _restore_runtime_options(original):
    bt.MondayBuy = original["MondayBuy"]
    bt.LowVolumeBuy = original["LowVolumeBuy"]


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


def run_builder_backtest(request) -> dict:
    data = load_ticker_data(request.symbol, years=request.years)
    conditions = [condition.model_dump() for condition in request.conditions]
    sell_conditions = [condition.model_dump() for condition in request.sell_conditions]
    compile_buy_mask(data, conditions)
    if sell_conditions:
        compile_sell_mask(data, sell_conditions)

    labels = {item["id"]: item["label"] for item in list_indicators(builder_only=True)}
    rule_preview = format_condition_preview(conditions, labels)
    description = request.description.strip() or rule_preview
    if request.name.strip():
        description = f"{request.name.strip()}: {description}"

    from api.schemas import SavedStrategy

    strategy = SavedStrategy(
        id="preview",
        name=request.name,
        symbol=request.symbol,
        direction=request.direction,
        hold_days=request.hold_days,
        profit=request.profit,
        description=description,
        conditions=request.conditions,
        sell_conditions=request.sell_conditions,
        created_at="",
        updated_at="",
    )
    executed = execute_saved_strategy(data, strategy)
    return detailed_backtest_payload(executed, request.hold_days, request.profit, description)


def run_builder_refine(request) -> dict | list[dict]:
    if not request.strategy.conditions:
        raise ValueError("Draft strategy must have at least one entry condition")

    primary = draft_to_saved_strategy(request.strategy)
    symbol = primary.symbol
    years = request.strategy.years

    if request.mode == "signal-combo-sweep":
        if not request.secondary_strategy_id:
            raise ValueError("Secondary strategy is required for signal combo sweep")
        secondary = get_strategy_by_id(request.secondary_strategy_id)
        if secondary is None:
            raise ValueError(f"Unknown strategy: {request.secondary_strategy_id}")
        if secondary.symbol.strip().upper() != symbol:
            raise ValueError("Secondary strategy must use the same symbol as the draft")
        data = load_ticker_data(symbol, years=years)
        results = backtest_builder_signal_combinations(primary, secondary, data, symbol)
        return dataframe_records(results)

    signal = builder_signal_callable(primary)

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
        data = load_ticker_data(symbol, years=years)
        data["Buy"], data["Sell"], _, _, _, _, is_long, _ = signal(data, symbol)
        results = bt.backtest_days(data, request.max_days, is_long)
        return dataframe_records(results)

    if request.mode == "indicator-sweep":
        data = load_ticker_data(symbol, years=years)
        data["Buy"], data["Sell"], days, profit, _, _, is_long, _ = signal(data, symbol)
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

    raise ValueError(f"Unsupported refine mode: {request.mode}")


def save_strategy(request) -> dict:
    conditions = [condition.model_dump() for condition in request.conditions]
    sell_conditions = [condition.model_dump() for condition in request.sell_conditions]
    data = load_ticker_data(request.symbol.strip().upper(), years=1)
    compile_buy_mask(data, conditions)
    if sell_conditions:
        compile_sell_mask(data, sell_conditions)
    saved = create_strategy(request)
    return saved.model_dump()


def list_saved_strategies() -> list[dict]:
    return [strategy.model_dump() for strategy in list_strategies()]


def run_live_scan() -> list[dict]:
    return run_scan()
