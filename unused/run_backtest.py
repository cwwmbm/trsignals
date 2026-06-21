"""
Backtest entry point. Configure RUN_MODE and settings below, then:

    python run_backtest.py
"""
import _bootstrap  # noqa: F401
import warn_config  # noqa: F401 — suppress third-party FutureWarnings
import time
import indicators as ind
from config import ticker
from backtest_runners import (
    load_ticker_data,
    run_single_symbol_backtest,
    signal_combination_tryout,
    symbol_confirmation_tryout,
    symbol_confirmation_detail,
)
from indicator_sweep import indicator_tryout
import backtest as bt

# ---------------------------------------------------------------------------
# Configuration — edit this block for each run
# ---------------------------------------------------------------------------

# RUN_MODE = 'single'  # one signal, one symbol, full stats + CSV
# RUN_MODE = 'indicator_sweep'  # grid-search indicator filters on a signal
RUN_MODE = 'signal_combo_sweep'  # compare AND/OR between two signals (same symbol)
# RUN_MODE = 'symbol_confirm_sweep'   # sweep confirm subsets from SYMBOL_POOL (start here)
# RUN_MODE = 'symbol_confirm_detail'  # one primary + CONFIRM_SYMBOLS, yearly breakdown
# RUN_MODE = 'hold_days_sweep'  # search days-in-trade × profitable closes
# Modes:
#   single                 — one signal, one symbol, full stats + CSV
#   indicator_sweep        — grid-search indicator filters on a signal
#   signal_combo_sweep     — compare AND/OR between two signals (same symbol)
#   symbol_confirm_sweep   — sweep confirm subsets from SYMBOL_POOL (start here)
#   symbol_confirm_detail  — drill into one PRIMARY + CONFIRM_SYMBOLS (yearly stats)
#   hold_days_sweep        — search days-in-trade × profitable closes

YEARS = 25
SYMBOL = ticker  # from config.py, e.g. 'SPY'

# Signal to backtest — pick ONE line:
SIGNAL_A = ind.buy_signal16
SIGNAL_B = ind.buy_signal7
# SIGNAL = ind.buy_signal16
SIGNAL = ind.combined_signal(SIGNAL_A, SIGNAL_B, 'or')



PRIMARY_SYMBOL = 'SOXX'
SYMBOL_POOL = ['SOXX', 'SMH', 'QQQ', 'SPY']
CONFIRM_SYMBOLS = ['SMH', 'QQQ']  # e.g. ['SMH', 'QQQ']

INDICATOR_SWEEP = dict(is_sell=False, check_breadth=False, check_both=False)
HOLD_DAYS_MAX = 7


def main():
    start = time.perf_counter()

    if RUN_MODE == 'single':
        run_single_symbol_backtest(SIGNAL, SYMBOL, years=YEARS)

    elif RUN_MODE == 'indicator_sweep':
        data = load_ticker_data(SYMBOL, years=YEARS)
        data['Buy'], data['Sell'], days, profit, _, _, is_long, _ = SIGNAL(data, SYMBOL)
        indicator_tryout(data, days, profit, is_long, **INDICATOR_SWEEP)

    elif RUN_MODE == 'signal_combo_sweep':
        data = load_ticker_data(SYMBOL, years=YEARS)
        signal_combination_tryout(SIGNAL_A, SIGNAL_B, data, SYMBOL)

    elif RUN_MODE == 'symbol_confirm_sweep':
        symbol_confirmation_tryout(SIGNAL, PRIMARY_SYMBOL, SYMBOL_POOL, years=YEARS)

    elif RUN_MODE == 'symbol_confirm_detail':
        symbol_confirmation_detail(SIGNAL, PRIMARY_SYMBOL, CONFIRM_SYMBOLS, years=YEARS)

    elif RUN_MODE == 'hold_days_sweep':
        data = load_ticker_data(SYMBOL, years=YEARS)
        data['Buy'], data['Sell'], days, profit, _, _, is_long, _ = SIGNAL(data, SYMBOL)
        results = bt.backtest_days(data, HOLD_DAYS_MAX, is_long)
        print(results.head(20))

    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")

    print(f"Execution time: {time.perf_counter() - start:.2f} seconds")


if __name__ == '__main__':
    main()
