import pandas as pd
import backtest as bt
from config import ticker
from time import perf_counter

# (buy_column, sell_column, condition, min, max, step)
BREADTH_SWEEPS = [
    ('RSI2BondBreadth', 'RSI2GoldBreadth', 'both', 10, 90, 10),
    ('RSI5BondBreadth', 'RSI2GoldBreadth', 'both', 10, 90, 10),
    ('RSI14BondBreadth', 'RSI2GoldBreadth', 'both', 10, 90, 10),
    ('RSI14FinancialsBreadth', 'RSI14FinancialsBreadth', 'both', 10, 90, 10),
    ('RSI14EnergyBreadth', 'RSI14EnergyBreadth', 'both', 10, 90, 10),
    ('RSI14UtilitiesBreadth', 'RSI14UtilitiesBreadth', 'both', 10, 90, 10),
    ('RSI14IndustrialsBreadth', 'RSI14IndustrialsBreadth', 'both', 10, 90, 10),
    ('RSI5FinancialsBreadth', 'RSI5FinancialsBreadth', 'both', 10, 90, 10),
    ('RSI5EnergyBreadth', 'RSI5EnergyBreadth', 'both', 10, 90, 10),
    ('RSI5UtilitiesBreadth', 'RSI5UtilitiesBreadth', 'both', 10, 90, 10),
    ('RSI5IndustrialsBreadth', 'RSI5IndustrialsBreadth', 'both', 10, 90, 10),
    ('RSI2GoldBreadth', 'RSI2GoldBreadth', 'both', 10, 90, 10),
    ('RSI5GoldBreadth', 'RSI2GoldBreadth', 'both', 10, 90, 10),
    ('RSI14GoldBreadth', 'RSI2GoldBreadth', 'both', 10, 90, 10),
    ('RSI14RiskBreadth', 'RSI14RiskBreadth', 'both', 20, 80, 10),
    ('RSI5RiskBreadth', 'RSI5RiskBreadth', 'both', 20, 80, 10),
    ('RSI2RiskBreadth', 'RSI2RiskBreadth', 'both', 10, 90, 10),
    ('RSI5SemisBreadth', 'RSI5SemisBreadth', 'both', 20, 80, 10),
    ('RSI14SemisBreadth', 'RSI14SemisBreadth', 'both', 20, 80, 10),
    ('RSI2SemisBreadth', 'RSI2SemisBreadth', 'both', 10, 90, 10),
]

POST_2003_BREADTH_SWEEPS = [
    ('RSI14Breadth', 'RSI14Breadth', 'both', 20, 80, 10),
    ('RSI2Breadth', 'RSI2Breadth', 'both', 10, 90, 10),
    ('RSI5Breadth', 'RSI5Breadth', 'both', 20, 80, 10),
]

PRICE_SWEEPS = [
    ('LowerCloses2', 'LowestClose2', 'both', 0, 0, 1),
    ('LowerCloses3', 'LowestClose3', 'both', 0, 0, 1),
    ('HigherCloses2', 'HighestClose2', 'both', 0, 0, 1),
    ('HigherCloses3', 'HighestClose3', 'both', 0, 0, 1),
    ('%Change', '%Change', 'both', -0.06, 0.06, 0.01),
    ('RSIBuy', 'RSISell', 'both', 0, 0, 1),
    ('Close_EMA8', 'Close_EMA8', 'both', -10, 10, 1),
    ('EMA8CrossUp', 'EMA8CrossUp', 'both', 0, 0, 1),
    ('EMA8CrossDown', 'EMA8CrossDown', 'both', 0, 0, 1),
    ('ATR20_ATR50', 'ATR20_ATR50', 'both', 0, 0, 1),
    ('ChangeVelocity', 'ChangeVelocity', 'both', -2, 2, 0.5),
    ('EMA20_EMA100', 'EMA20_EMA100', 'both', 0, 0, 1),
    ('LowestClose2', 'LowestClose2', 'both', 0, 0, 1),
    ('LowestClose3', 'LowestClose3', 'both', 0, 0, 1),
    ('HighestClose2', 'HighestClose2', 'both', 0, 0, 1),
    ('HighestClose3', 'HighestClose3', 'both', 0, 0, 1),
    ('SMA50_SMA200', 'SMA50_SMA200', 'both', 0, 0, 1),
    ('SMA20_SMA50', 'SMA20_SMA50', 'both', 0, 0, 1),
    ('Close_SMA200', 'Close_SMA200', 'both', 0, 0, 1),
    ('Close_SMA50', 'Close_SMA50', 'both', 0, 0, 1),
    ('Close_SMA20', 'Close_SMA20', 'both', 0, 0, 1),
    ('Vix', 'Vix', 'both', 10, 50, 5),
    ('Stoch', 'Stoch', 'both', 10, 90, 10),
    ('RSI14', 'RSI14', 'both', 20, 90, 10),
    ('RSI5', 'RSI5', 'both', 20, 90, 10),
    ('RSI2', 'RSI2', 'both', 10, 50, 5, 50, 99, 5),
    ('ER', 'ER', 'both', 0.1, 0.9, 0.1),
    ('CCI', 'CCI', 'both', -150, 150, 50),
    ('IBR', 'IBR', 'both', 0.1, 0.9, 0.1),
    ('IBR2', 'IBR2', 'both', 0.1, 0.9, 0.1),
    ('IBR3', 'IBR3', 'both', 0.1, 0.9, 0.1),
    ('ValueCharts', 'ValueCharts', 'both', -12, 12, 2),
    ('MACDHist', 'MACDHist', 'both', 0, 0, 1),
    ('StochOscilator', 'StochOscilator', 'both', 0, 0, 1),
    ('SPYBull', 'SPYBull', 'both', 0, 0, 1),
    ('HigherCloses2', 'HigherCloses2', 'both', 0, 0, 1),
    ('HigherCloses3', 'HigherCloses3', 'both', 0, 0, 1),
    ('LowerCloses2', 'LowerCloses2', 'both', 0, 0, 1),
    ('LowerCloses3', 'LowerCloses3', 'both', 0, 0, 1),
    ('DownMonday', 'DownMonday', 'both', 0, 0, 1),
    ('ADX14', 'ADX14', 'both', 10, 50, 5),
    ('WilliamsR14', 'WilliamsR14', 'both', -90, -10, 10),
    ('ROC20', 'ROC20', 'both', -10, 10, 2),
    ('TRIX', 'TRIX', 'both', -0.5, 0.5, 0.1),
    ('CMF20', 'CMF20', 'both', -0.4, 0.4, 0.1),
    ('OBVSlope20', 'OBVSlope20', 'both', -5, 5, 1),
    ('BBWidth', 'BBWidth', 'both', 0, 0.2, 0.02),
    ('BBPercentB', 'BBPercentB', 'both', 0, 1, 0.1),
    ('LinRegSlope20', 'LinRegSlope20', 'both', -2, 2, 0.5),
    ('VolatilityPercentile', 'VolatilityPercentile', 'both', 0, 100, 10),
    ('CloseAboveDonchianUpper20', 'CloseAboveDonchianUpper20', 'both', 0, 0, 1),
    ('CloseBelowDonchianLower20', 'CloseBelowDonchianLower20', 'both', 0, 0, 1),
    ('CloseAboveDonchianUpper55', 'CloseAboveDonchianUpper55', 'both', 0, 0, 1),
    ('CloseBelowDonchianLower55', 'CloseBelowDonchianLower55', 'both', 0, 0, 1),
    ('CloseAbovePSAR', 'CloseAbovePSAR', 'both', 0, 0, 1),
    ('CloseAboveKCUpper20', 'CloseAboveKCUpper20', 'both', 0, 0, 1),
    ('CloseBelowKCLower20', 'CloseBelowKCLower20', 'both', 0, 0, 1),
    ('BBSqueeze', 'BBSqueeze', 'both', 0, 0, 1),
]

VFI_SWEEPS = [
    ('VFI80', 'VFI80', 'both', -8, 8, 2),
    ('VFI40', 'VFI40', 'both', -8, 8, 2),
    ('VFI10', 'VFI10', 'both', -8, 8, 2),
    ('VFI20', 'VFI20', 'both', -8, 8, 2),
]

VWAP_SWEEPS = [
    ('Close_VWAP', 'Close_VWAP', 'both', -2, 2, 0.25),
    ('VWAPCrossUp', 'VWAPCrossUp', 'both', 0, 0, 1),
    ('VWAPCrossDown', 'VWAPCrossDown', 'both', 0, 0, 1),
    ('VWAPSlope8', 'VWAPSlope8', 'both', -2, 2, 0.5),
    ('VWAPSlope20', 'VWAPSlope20', 'both', -2, 2, 0.5),
    ('VWAPPercentB', 'VWAPPercentB', 'both', 0, 1, 0.1),
    ('VWAPWidth', 'VWAPWidth', 'both', 0, 0.1, 0.01),
    ('CloseAboveVWAPUpper1', 'CloseAboveVWAPUpper1', 'both', 0, 0, 1),
    ('CloseBelowVWAPLower1', 'CloseBelowVWAPLower1', 'both', 0, 0, 1),
    ('CloseAboveVWAPUpper2', 'CloseAboveVWAPUpper2', 'both', 0, 0, 1),
    ('CloseBelowVWAPLower2', 'CloseBelowVWAPLower2', 'both', 0, 0, 1),
]

VFI_EXCLUDED_TICKERS = {'NQ', 'ES', 'GC', 'SI', 'HG', 'RTY', 'YM', 'CL', 'SOXX', 'FXI'}


def _run_one_sweep(data, days, profit, is_long, is_sell, og, spec, *, pnl_column=None):
    buy_col, sell_col, condition = spec[0], spec[1], spec[2]
    buy_min, buy_max, buy_step = spec[3], spec[4], spec[5]
    sell_min, sell_max, sell_step = (spec[6], spec[7], spec[8]) if len(spec) > 6 else (buy_min, buy_max, buy_step)
    column = sell_col if is_sell else buy_col
    min_val, max_val, step = (sell_min, sell_max, sell_step) if is_sell else (buy_min, buy_max, buy_step)
    fn = bt.backtest_sell_ind if is_sell else bt.backtest_ind
    return fn(data, days, profit, is_long, column, condition, min_val, max_val, step, og, include_yearly=False, pnl_column=pnl_column)


def _collect_sweeps(running_rows, data, days, profit, is_long, is_sell, og, specs, verbose=True, timing=False, *, pnl_column=None):
    for spec in specs:
        started = perf_counter()
        results = _run_one_sweep(data, days, profit, is_long, is_sell, og, spec, pnl_column=pnl_column)
        top_results = bt.add_yearly_to_indicator_rows(results.head(3), data, days, profit, is_long)
        if timing:
            buy_col, sell_col = spec[0], spec[1]
            label = sell_col if is_sell else buy_col
            print(f"[timing] sweep {label}: {perf_counter() - started:.3f}s")
        if verbose:
            print(results.head(5))
        running_rows.extend(top_results.to_dict(orient='records'))
    return running_rows


def indicator_tryout(data, days, profit, is_long, is_sell=False, check_breadth=True, check_both=True, verbose=True, timing=False, exclude_columns=None, include_vwap_sweeps=False, *, pnl_column=None):
    """Grid-search indicator filters layered on the current buy/sell signal."""
    exclude_columns = set(exclude_columns or [])
    total_started = perf_counter()
    previous_timing = bt.TIMING_ENABLED
    bt.set_timing_enabled(timing)
    running_rows = []
    og = days == 0

    def _filter_specs(specs):
        if not exclude_columns:
            return specs
        return [
            spec
            for spec in specs
            if spec[0] not in exclude_columns and spec[1] not in exclude_columns
        ]

    try:
        if check_breadth:
            running_rows = _collect_sweeps(
                running_rows, data, days, profit, is_long, is_sell, og,
                _filter_specs(BREADTH_SWEEPS), verbose, timing, pnl_column=pnl_column,
            )
            if data['Date'].dt.year.iloc[0] >= 2003:
                running_rows = _collect_sweeps(
                    running_rows, data, days, profit, is_long, is_sell, og,
                    _filter_specs(POST_2003_BREADTH_SWEEPS), verbose, timing, pnl_column=pnl_column,
                )

        if check_both or not check_breadth:
            running_rows = _collect_sweeps(
                running_rows, data, days, profit, is_long, is_sell, og,
                _filter_specs(PRICE_SWEEPS), verbose, timing, pnl_column=pnl_column,
            )
            if ticker not in VFI_EXCLUDED_TICKERS:
                running_rows = _collect_sweeps(
                    running_rows, data, days, profit, is_long, is_sell, og,
                    _filter_specs(VFI_SWEEPS), verbose, timing, pnl_column=pnl_column,
                )
            if include_vwap_sweeps and 'VWAP' in data.columns and data['VWAP'].notna().any():
                running_rows = _collect_sweeps(
                    running_rows, data, days, profit, is_long, is_sell, og,
                    _filter_specs(VWAP_SWEEPS), verbose, timing, pnl_column=pnl_column,
                )
    finally:
        bt.set_timing_enabled(previous_timing)

    running_results = pd.DataFrame(running_rows)
    running_results = running_results.sort_values(by=['Sharpe'], ascending=False)
    if verbose:
        print(running_results)
    if timing:
        print(f"[timing] indicator_tryout total: {perf_counter() - total_started:.3f}s")
    return running_results
