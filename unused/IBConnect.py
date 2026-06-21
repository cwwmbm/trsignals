"""
Legacy entry point — delegates to run_backtest.py.

Re-exports are kept so existing imports continue to work:
    from IBConnect import symbol_confirmation_detail, print_stats
"""
import _bootstrap  # noqa: F401
import warn_config  # noqa: F401
from run_backtest import main
from stats import print_stats
from indicator_sweep import indicator_tryout
from backtest_runners import (
    load_ticker_data,
    run_single_symbol_backtest,
    signal_combination_tryout,
    symbol_confirmation_tryout,
    symbol_confirmation_detail,
)

if __name__ == '__main__':
    main()
