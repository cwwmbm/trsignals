import backtest as bt
from backtest_runners import load_ticker_data
from indicator_sweep import indicator_tryout

from api.serializers import dataframe_records, detailed_backtest_payload
from api.signal_registry import get_signal, resolve_signal


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
