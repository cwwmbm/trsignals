import pandas as pd
import indicators as ind
import numpy as np
from config import *
import getdata as dt
from itertools import combinations
from stats import compute_aggregate_metrics, yearly_performance
from contextlib import contextmanager
from time import perf_counter
import os


TIMING_ENABLED = os.getenv("BACKTEST_TIMING", "").lower() in {"1", "true", "yes"}
HoldOnBuySignal = False


def set_timing_enabled(enabled):
    global TIMING_ENABLED
    TIMING_ENABLED = enabled


@contextmanager
def timed(label):
    if not TIMING_ENABLED:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - started
        print(f"[timing] {label}: {elapsed:.3f}s")


def _yearly_records(data):
    yearly = yearly_performance(data).reset_index()
    yearly = yearly.rename(
        columns={
            'Date': 'year',
            'PnL%': 'pnl_percent',
            'Drawdown%': 'drawdown_percent',
            'Num_Trades': 'num_trades',
            'Positive_Trades': 'positive_trades',
        }
    )
    records = []
    for row in yearly.to_dict(orient='records'):
        records.append(
            {
                'year': int(row['year']),
                'pnl_percent': float(str(row['pnl_percent']).replace('%', '')),
                'drawdown_percent': float(str(row['drawdown_percent']).replace('%', '')),
                'num_trades': int(row['num_trades']),
                'positive_trades': int(row['positive_trades']),
            }
        )
    return records


def _ranking_metrics(data, include_yearly=True):
    with timed("_ranking_metrics"):
        m = compute_aggregate_metrics(data)
    metrics = {
        'PnL': m['rolling_pnl'],
        'MaxDD': m['max_drawdown'] * 100,
        'Trades': m['trades'],
        '%Pstv': m['pct_positive'],
        'CAGR': str(m['cagr_percent']) + '%',
        'Sharpe': m['sharpe'],
        'Sortino': m['sortino'],
    }
    if include_yearly:
        metrics['Yearly'] = _yearly_records(data)
    return metrics


def _format_ranking_results(results, include_value=True):
    if results.empty:
        return results
    results = results.sort_values(by=['Sharpe'], ascending=False)
    results['PnL'] = results['PnL'].astype(int)
    results['MaxDD'] = results['MaxDD'].round(2)
    results['PnL'] = results['PnL'].apply(ind.format_dollar_value)
    results['MaxDD'] = results['MaxDD'].astype(str)+'%'
    results['Sharpe'] = pd.to_numeric(results['Sharpe'], errors='coerce')
    results['Sharpe'] = results['Sharpe'].round(2) if not (results['Sharpe'].isnull().values.any() or np.isinf(results['Sharpe']).any()) else results['Sharpe']
    results['Sortino'] = pd.to_numeric(results['Sortino'], errors='coerce')
    results['Sortino'] = results['Sortino'].round(2) if not (results['Sortino'].isnull().values.any() or np.isinf(results['Sortino']).any()) else results['Sortino']
    results['%Pstv'] = pd.to_numeric(results['%Pstv'], errors='coerce')
    results['%Pstv'] = results['%Pstv'].round(1) if not (results['%Pstv'].isnull().values.any() or np.isinf(results['%Pstv']).any()) else results['%Pstv']
    results['Trades'] = results['Trades'].astype(int)
    return results


def _run_indicator_threshold(
    data,
    days_in_trade,
    profitable_close,
    is_long,
    column_name,
    buy_sell,
    condition,
    value,
    include_yearly=True,
    *,
    pnl_column=None,
):
    if days_in_trade > 0:
        columns = ['Date', 'Close', '%Change', 'Buy', 'Sell', column_name]
        if UseProxyUnderlying:
            columns.append(ProxySymbol)
        columns = list(dict.fromkeys(columns))
        data_copy = data.loc[:, columns].copy()
    else:
        data_copy = data.copy()
    if buy_sell == 'Buy':
        if condition == 'less':
            data_copy['Buy'] = data_copy['Buy'] & (data_copy[column_name] <= value)
        else:
            data_copy['Buy'] = data_copy['Buy'] & (data_copy[column_name] >= value)
    else:
        if condition == 'less':
            data_copy['Sell'] = data_copy['Sell'] | (data_copy[column_name] <= value)
        else:
            data_copy['Sell'] = data_copy['Sell'] | (data_copy[column_name] >= value)

    data_copy = execute_strategy(
        data_copy,
        days_in_trade,
        profitable_close,
        is_long,
        pnl_column=pnl_column,
    )
    m = _ranking_metrics(data_copy, include_yearly=include_yearly)
    row = {
        'Buysell': buy_sell,
        'Indicator': column_name,
        'Condition': condition,
        'Value': value,
        'PnL': m['PnL'],
        'MaxDD': m['MaxDD'],
        'Trades': m['Trades'],
        '%Pstv': m['%Pstv'],
        'CAGR': m['CAGR'],
        'Sharpe': m['Sharpe'],
        'Sortino': m['Sortino'],
    }
    if include_yearly:
        row['Yearly'] = m['Yearly']
    return row


def add_yearly_to_indicator_rows(results, data, days_in_trade, profitable_close, is_long):
    if results.empty:
        return results
    results = results.copy()
    yearly = []
    for _, row in results.iterrows():
        detail = _run_indicator_threshold(
            data,
            days_in_trade,
            profitable_close,
            is_long,
            row['Indicator'],
            row['Buysell'],
            row['Condition'],
            row['Value'],
            include_yearly=True,
        )
        yearly.append(detail['Yearly'])
    results['Yearly'] = yearly
    return results


#Backtest function that iterates over number of days in trade / profitable days in trade
def backtest_days(data, max_days = 10, is_long = True, og = False, *, pnl_column=None):
    results = pd.DataFrame(columns=['Days', 'Prf', 'PnL', 'MaxDD', 'Trades', '%Pstv', 'Sharpe', 'Sortino', 'Yearly'])
    for i in range(1, max_days+1):
        for k in range(1, i+1):
            signals = execute_strategy(data.copy(), i, k, is_long, pnl_column=pnl_column)
            m = _ranking_metrics(signals)
            results.loc[i*10+k] = {
                'Days': i,
                'Prf': k,
                'PnL': m['PnL'],
                'MaxDD': m['MaxDD'] / 100,
                'Trades': m['Trades'],
                '%Pstv': m['%Pstv'],
                'Sharpe': m['Sharpe'],
                'Sortino': m['Sortino'],
                'Yearly': m['Yearly'],
            }

    
    #sort by Sharpe ratio
    results = results.sort_values(by=['Sharpe'], ascending=False)
    #apply formating to PnL and DD
    results['PnL'] = results['PnL'].astype(int)
    #results['MaxDD'] = results['MaxDD'].astype(int)
    results['PnL'] = results['PnL'].apply(ind.format_dollar_value)
    results['MaxDD'] = results['MaxDD'].round(2)*100#.apply(ind.format_dollar_value)
    # Round the 'Sharpes' column to 2 decimal places
    results['Sharpe'] = results['Sharpe'].round(2)
    results['Sortino'] = results['Sortino'].round(2)
    results['%Pstv'] = results['%Pstv'].round(1)

    # Convert the 'Trades' column to integers
    results['Trades'] = results['Trades'].astype(int)
    results['Days'] = results['Days'].astype(int)
    results['Prf'] = results['Prf'].astype(int)

    return results

#Backtest function that iterates over input indicator and its value
def backtest_ind(data, days_in_trade, profitable_close, is_long, column_name, condition, min_value, max_value, step=0.1, og = False, include_yearly=True, *, pnl_column=None):
    rows = []
    for value in np.arange(min_value, max_value + step, step):
        if condition == 'both':
            rows.append(_run_indicator_threshold(data, days_in_trade, profitable_close, is_long, column_name, 'Buy', 'more', value, include_yearly, pnl_column=pnl_column))
            cond = 'less'
        else:
            cond = condition
        rows.append(_run_indicator_threshold(data, days_in_trade, profitable_close, is_long, column_name, 'Buy', cond, value, include_yearly, pnl_column=pnl_column))

    return _format_ranking_results(pd.DataFrame(rows))

def backtest_sell_ind(data, days_in_trade, profitable_close, is_long, column_name, condition, min_value, max_value, step=0.1, og = False, include_yearly=True, *, pnl_column=None):
    rows = []
    for value in np.arange(min_value, max_value + step, step):
        if condition == 'both':
            rows.append(_run_indicator_threshold(data, days_in_trade, profitable_close, is_long, column_name, 'Sell', 'more', value, include_yearly, pnl_column=pnl_column))
            cond = 'less'
        else:
            cond = condition
        rows.append(_run_indicator_threshold(data, days_in_trade, profitable_close, is_long, column_name, 'Sell', cond, value, include_yearly, pnl_column=pnl_column))

    return _format_ranking_results(pd.DataFrame(rows))

def backtest_signal_combinations(signal_a, signal_b, data, symbol=ticker):
    """
    Backtest all four primary/secondary AND/OR combinations of two signals.
    days/profit/sell/is_long come from whichever signal is primary for that row.
    """
    results = pd.DataFrame(columns=['Primary', 'Secondary', 'Mode', 'Days', 'Profit', 'PnL', 'MaxDD', 'Trades', '%Pstv', 'CAGR', 'Sharpe', 'Sortino'])

    for primary, secondary, mode in (
        (signal_a, signal_b, 'and'),
        (signal_a, signal_b, 'or'),
        (signal_b, signal_a, 'and'),
        (signal_b, signal_a, 'or'),
    ):
        data_copy = data.copy()
        buy, sell, days, profit, _, _, is_long, _ = ind.combine_buy_signals(
            primary, secondary, data_copy, symbol, mode
        )
        data_copy['Buy'] = buy
        data_copy['Sell'] = sell
        data_copy = execute_strategy(data_copy, days, profit, is_long)
        m = _ranking_metrics(data_copy)

        results = results._append({
            'Primary': primary.__name__,
            'Secondary': secondary.__name__,
            'Mode': mode.upper(),
            'Days': days,
            'Profit': profit,
            'PnL': m['PnL'],
            'MaxDD': m['MaxDD'],
            'Trades': m['Trades'],
            '%Pstv': m['%Pstv'],
            'CAGR': m['CAGR'],
            'Sharpe': m['Sharpe'],
            'Sortino': m['Sortino'],
            'Yearly': m['Yearly'],
        }, ignore_index=True)

    results = results.sort_values(by=['Sharpe'], ascending=False)
    results['PnL'] = results['PnL'].astype(int)
    results['MaxDD'] = results['MaxDD'].round(2)
    results['PnL'] = results['PnL'].apply(ind.format_dollar_value)
    results['MaxDD'] = results['MaxDD'].astype(str) + '%'
    results['Sharpe'] = pd.to_numeric(results['Sharpe'], errors='coerce')
    results['Sharpe'] = results['Sharpe'].round(2) if not (results['Sharpe'].isnull().values.any() or np.isinf(results['Sharpe']).any()) else results['Sharpe']
    results['Sortino'] = pd.to_numeric(results['Sortino'], errors='coerce')
    results['Sortino'] = results['Sortino'].round(2) if not (results['Sortino'].isnull().values.any() or np.isinf(results['Sortino']).any()) else results['Sortino']
    results['%Pstv'] = pd.to_numeric(results['%Pstv'], errors='coerce')
    results['%Pstv'] = results['%Pstv'].round(1) if not (results['%Pstv'].isnull().values.any() or np.isinf(results['%Pstv']).any()) else results['%Pstv']
    results['Trades'] = results['Trades'].astype(int)
    results['Days'] = results['Days'].astype(int)
    results['Profit'] = results['Profit'].astype(int)

    return results

def build_symbol_dataset(full_data, symbols, symbol_to_yf=None):
    """Build enriched, indicator-ready data from an existing bulk download."""
    if symbol_to_yf is None:
        context_symbols = [s for s in dt.MARKET_CONTEXT_SYMBOLS if s not in symbols]
        all_symbols = list(dict.fromkeys(list(symbols) + context_symbols))
        symbol_to_yf = {symbol: dt.to_yf_symbol(symbol) for symbol in all_symbols}
    market_context = dt.extract_market_context(full_data, symbol_to_yf)

    dataset = {}
    for symbol in symbols:
        data = dt.symbol_frame_from_bulk(full_data, symbol_to_yf[symbol], market_context)
        dataset[symbol] = ind.add_indicators(data)
    return dataset


def load_symbol_dataset(symbols, years=25, *, use_cache=True):
    """Load enriched, indicator-ready data for each symbol."""
    from api.market_data_cache import PROFILE_BULK, load as load_cached_frame, save as save_cached_frame

    normalized = list(dict.fromkeys(symbol.strip().upper() for symbol in symbols))
    dataset = {}
    missing = []

    if use_cache:
        for symbol in normalized:
            cached = load_cached_frame(symbol, years, PROFILE_BULK)
            if cached is not None:
                dataset[symbol] = cached
            else:
                missing.append(symbol)
    else:
        missing = normalized

    if not missing:
        return dataset

    context_symbols = [s for s in dt.MARKET_CONTEXT_SYMBOLS if s not in missing]
    all_symbols = list(dict.fromkeys(list(missing) + context_symbols))
    symbol_to_yf = {symbol: dt.to_yf_symbol(symbol) for symbol in all_symbols}
    yf_symbols = list(symbol_to_yf.values())
    full_data = dt.get_bulk_data(yf_symbols, years=years)
    fresh = build_symbol_dataset(full_data, missing, symbol_to_yf)
    for symbol in missing:
        dataset[symbol] = fresh[symbol]
        if use_cache:
            save_cached_frame(symbol, years, PROFILE_BULK, fresh[symbol])
    return dataset

def _buy_series_by_date(data, buy):
    return pd.Series(buy.values, index=pd.to_datetime(data['Date']))

def apply_cross_symbol_signal(buy_signal, primary_symbol, confirm_symbols, symbol_data):
    """
    Apply a signal across symbols. Buy fires only when every symbol confirms;
    sell and hold rules come from the primary symbol only.
    """
    confirm_symbols = [s for s in confirm_symbols if s != primary_symbol]
    primary = symbol_data[primary_symbol].copy()
    p_buy, p_sell, days, profit, description, verdict, is_long, ignore = buy_signal(primary, primary_symbol)

    combined_buy = _buy_series_by_date(primary, p_buy)
    for symbol in confirm_symbols:
        sec_data = symbol_data[symbol]
        s_buy, _, _, _, _, _, _, _ = buy_signal(sec_data, symbol)
        combined_buy = combined_buy & _buy_series_by_date(sec_data, s_buy).reindex(combined_buy.index, fill_value=False)

    primary['Buy'] = combined_buy.values
    primary['Sell'] = p_sell
    if confirm_symbols:
        confirm_label = '+'.join(confirm_symbols)
        description = f"[{primary_symbol} trade, confirm: {confirm_label}] {description}"
    return primary, days, profit, description, verdict, is_long, ignore

def _backtest_result_row(buy_signal, primary_symbol, confirm_symbols, data_copy, days, profit):
    m = _ranking_metrics(data_copy)
    confirm_symbols = [s for s in confirm_symbols if s != primary_symbol]
    return {
        'Signal': buy_signal.__name__,
        'Primary': primary_symbol,
        'Confirm': '+'.join(confirm_symbols) if confirm_symbols else '(none)',
        'Days': days,
        'Profit': profit,
        'PnL': m['PnL'],
        'MaxDD': m['MaxDD'],
        'Trades': m['Trades'],
        '%Pstv': m['%Pstv'],
        'CAGR': m['CAGR'],
        'Sharpe': m['Sharpe'],
        'Sortino': m['Sortino'],
        'Yearly': m['Yearly'],
    }

def backtest_cross_symbol(buy_signal, primary_symbol, confirm_symbols=None, years=25, symbol_data=None):
    """Backtest a signal on the primary symbol with optional cross-symbol buy confirmation."""
    confirm_symbols = confirm_symbols or []
    needed = list(dict.fromkeys([primary_symbol] + confirm_symbols))
    if symbol_data is None:
        symbol_data = load_symbol_dataset(needed, years=years)
    data, days, profit, description, _, is_long, _ = apply_cross_symbol_signal(
        buy_signal, primary_symbol, confirm_symbols, symbol_data
    )
    data = execute_strategy(data, days, profit, is_long)
    return data, days, profit, description, is_long

def backtest_symbol_confirmation_sweep(buy_signal, primary_symbol, symbol_pool, years=25, confirm_sets=None, *, pnl_column=None):
    """
    Sweep all confirmation subsets from symbol_pool (excluding primary).
    Includes a primary-only row with no confirmation symbols.
    """
    candidates = [s for s in symbol_pool if s != primary_symbol]
    if confirm_sets is None:
        confirm_sets = [[]]
        for r in range(1, len(candidates) + 1):
            confirm_sets.extend(list(combinations(candidates, r)))

    needed = list(dict.fromkeys([primary_symbol] + candidates))
    symbol_data = load_symbol_dataset(needed, years=years)
    results = pd.DataFrame(columns=['Signal', 'Primary', 'Confirm', 'Days', 'Profit', 'PnL', 'MaxDD', 'Trades', '%Pstv', 'CAGR', 'Sharpe', 'Sortino'])

    for confirm_symbols in confirm_sets:
        confirm_symbols = list(confirm_symbols)
        data, days, profit, _, _, is_long, _ = apply_cross_symbol_signal(
            buy_signal, primary_symbol, confirm_symbols, symbol_data
        )
        data = execute_strategy(data, days, profit, is_long, pnl_column=pnl_column)
        results = results._append(
            _backtest_result_row(buy_signal, primary_symbol, confirm_symbols, data, days, profit),
            ignore_index=True,
        )

    results = results.sort_values(by=['Sharpe'], ascending=False)
    results['PnL'] = results['PnL'].astype(int)
    results['MaxDD'] = results['MaxDD'].round(2)
    results['PnL'] = results['PnL'].apply(ind.format_dollar_value)
    results['MaxDD'] = results['MaxDD'].astype(str) + '%'
    results['Sharpe'] = pd.to_numeric(results['Sharpe'], errors='coerce')
    results['Sharpe'] = results['Sharpe'].round(2) if not (results['Sharpe'].isnull().values.any() or np.isinf(results['Sharpe']).any()) else results['Sharpe']
    results['Sortino'] = pd.to_numeric(results['Sortino'], errors='coerce')
    results['Sortino'] = results['Sortino'].round(2) if not (results['Sortino'].isnull().values.any() or np.isinf(results['Sortino']).any()) else results['Sortino']
    results['%Pstv'] = pd.to_numeric(results['%Pstv'], errors='coerce')
    results['%Pstv'] = results['%Pstv'].round(1) if not (results['%Pstv'].isnull().values.any() or np.isinf(results['%Pstv']).any()) else results['%Pstv']
    results['Trades'] = results['Trades'].astype(int)
    results['Days'] = results['Days'].astype(int)
    results['Profit'] = results['Profit'].astype(int)
    return results

def execute_strategy (data, days, profit, is_long = True, *, pnl_column=None):
    with timed("execute_strategy"):
        if days > 0:
            results = long_strat(data, days, profit, is_long, pnl_column=pnl_column)
        else:
            if UseProxyUnderlying:
                results = long_og_strat_proxy(data = data, days = days, profit = profit)
            else:
                results = og_strat(data)
                #print('og_strat'+str(is_long)+str(days)+str(profit))    
    return results

#Original Main Strategy
def og_strat(data, days = 0, profit = 0, external_count = 0, start_capital = 15000):
    # Main Trading signals
    #data['MainBuy'] = False
    #if set_sell: data['Sell'] = False
    data['OneDayBuy'] = False
    data['HoldLong'] = False
    data['LongTradeIn'] = False
    data['LongTradeOut'] = False
    data['DaysInTrade'] = 0
    data['ProfitableCloses'] = 0
    data['RollingPnL'] = 0.0
    data['TradePnL'] = 0.0
    data['TradeEntry'] = 0.0
    ExternalBuy = False
    if external_count > 0:
        for i in range (0, external_count):
            buy_column = 'Buy' + str(i)
            hold_column = 'HoldLong' + str(i)
            ExternalBuy = ExternalBuy | data[buy_column] | data[hold_column]
    else:
        ExternalBuy = True
    """
    if set_sell:
        data['Sell'] = ((data['RSI2'] > RSI2Sell) & (data['RSI5'] > RSI5Sell)) | (                            #RSI2 and RSI5 above threashold
            (data['Close'].shift(1) > data['EMA8'].shift(1)) & (data['Close'] < data['EMA8'])) | (            #Crossing EMA8 down
            (ExitOnVolatility) & ((data['VolumeEMADiff'] >= VolumeEMAThreashold))) | (                        #Volume more than EMAThreashold !!!!!!!!!!!DOESNT WORK - INVESTIGATE!!!!!!!!
            #(data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) < -MaxDecline) | (              #Decline more than 4%
            (data['VolumeEMADiff'] > VolumeEMAThreashold) & (data['Volatility'] > VolatilityThreashold))      #Big volume and volatility  
    """
    # Add one day buy signal
    data['OneDayBuy'] = False
    data['OneDayBuy'] = (((data['Close'] <=  data['Close'].rolling(DownDays).min()) & (data['VolumeEMADiff'] < -VolumeEMAThreasholdBuy) & (LowVolumeBuy))) | ( #Low volume buy
            (data['Close'] < data['Close'].shift(1)) & (pd.to_datetime(data['Date']).dt.dayofweek == 0) & (MondayBuy))                                          #Monday buy
                                                                                                         


    # Calculate days when entering and exiting trades
    #data['OneDayArm'] = False

    for i, row in data.iterrows():
        if i == 0:
            data['HoldLong'].at[i] = False
            data['LongTradeOut'].at[i] = False
            data['TradeEntry'].at[i] = 0
            data['TradePnL'].at[i] = 0
            data['RollingPnL'].at[i] = start_capital
        else:
            data['HoldLong'].at[i] = ((data['HoldLong'].shift(1).at[i] and not data['LongTradeOut'].shift(1).at[i]) or ( #If previously long and not trade out on previous day
                                        data['LongTradeIn'].shift(1).at[i]) or (                                        #Or if Enetering trade on previous day
                                        data['OneDayBuy'].shift(1).at[i]))                                              #Or if one day buy on previous day??? Do we need this?
            data['LongTradeIn'].at[i] = (data['Buy'].at[i] or data['OneDayBuy'].at[i]) and not data['HoldLong'].at[i]
            data['DaysInTrade'].at[i] = data['DaysInTrade'].shift(1).at[i] + 1 if (data['HoldLong'].at[i] and i>0) else 0
            if (data['HoldLong'].at[i]):
                data['ProfitableCloses'].at[i] = data['ProfitableCloses'].shift(1).at[i] + 1 if (data['Close'].at[i] > data['Close'].shift(1).at[i]) else data['ProfitableCloses'].shift(1).at[i]
            data['TradeEntry'].at[i] = data['Close'].at[i] if (((data['LongTradeIn'].at[i] or data['OneDayBuy'].at[i])) and data['TradeEntry'].shift(1).at[i] == 0) else data['TradeEntry'].shift(1).at[i] if data['HoldLong'].at[i] else 0
            data['TradePnL'].at[i] = data['TradePnL'].shift(1).at[i] + data['%Change'].at[i] if data['HoldLong'].at[i] else 0
            #(data['Close'].at[i] - data['TradeEntry'].at[i]) / data['TradeEntry'].at[i] if data['HoldLong'].at[i] else 0
            if HoldOnBuySignal and data['Buy'].at[i] and data['HoldLong'].at[i]:
                data['LongTradeOut'].at[i] = False
            else:
                data['LongTradeOut'].at[i] = (data['Sell'].at[i] and data['HoldLong'].at[i]) or (                           #If sell signal and hold long
                                            data['OneDayBuy'].shift(1).at[i] and not data['HoldLong'].shift(1).at[i] and not data['Buy'].shift(1).at[i]) or ( #Or if one day buy and not hold long on previous day (and not buy signal today)
                                            data['TradePnL'].at[i] < -stop_loss)                                            #Or if hit stoploss 
                if (days > 0):
                    data['LongTradeOut'].at[i] = data['LongTradeOut'].at[i] or (data['DaysInTrade'].at[i] >= days) or (data['ProfitableCloses'].at[i] >= profit)
            #data['TradeEntry'].at[i] = data['Close'].at[i] if (data['LongTradeIn'].at[i] or data['OneDayBuy'].at[i]) else data['TradeEntry'].shift(1).at[i] if data['HoldLong'].at[i] else 0


        #Calculate rolling PnL for the strategy
        if i == 0:
            data['RollingPnL'].at[i] = start_capital
        elif data['HoldLong'].at[i]: 
            #signals['RollingPnL'].at[i] = signals['RollingPnL'].shift(1).at[i] + signals['Close'].at[i] - signals['Close'].shift(1).at[i]
            data['RollingPnL'].at[i] = (1+data['%Change'].at[i])*data['RollingPnL'].shift(1).at[i]
        else:
            data['RollingPnL'].at[i] = data['RollingPnL'].shift(1).at[i]   

    # Calculate the running maximum of the 'RollingPnL' column
    data['RunningMax'] = data['RollingPnL'].cummax()

    # Calculate the drawdown as the difference between the running maximum and the current 'RollingPnL' value
    data['Drawdown'] = (data['RunningMax'] - data['RollingPnL'])/data['RunningMax']
    #signals['TradePnL'] = signals['TradePnL'].apply(format_dollar_value)
    #signals['RollingPnL'] = signals['RollingPnL'].apply(format_dollar_value)

    return data

# def long_strat(data, days, prof_closes, is_long = True, start_capital = 15000, point_multiplier = point_multiplier):
#     signals = data
#     signals['LongTradeIn'] = False
#     signals['LongTradeOut'] = False
#     signals['HoldLong'] = False
#     signals['DaysInTrade'] = 0
#     signals['ProfitableCloses'] = 0
#     signals['RollingPnL'] = 0
#     signals['TradePnL'] = 0
#     signals['TradeEntry'] = 0
#     baddates = pd.DataFrame()
#     #signals['TradeInvestment'] = 0

#     for i, row in signals.iterrows():
#         signals['HoldLong'].at[i] = (signals['HoldLong'].shift(1).at[i] and not signals['LongTradeOut'].shift(1).at[i] and (i>0)) or ( #If previously long and not trade out on previous day
#                                     signals['LongTradeIn'].shift(1).at[i] and i>0)                                            #Or if Enetering trade on previous day
#         #signals['LongTradeIn'].at[i] = signals['Buy'].at[i] and not signals['HoldLong'].at[i]
#         signals['LongTradeIn'].at[i] = signals['Buy'].at[i] and not (signals['HoldLong'].at[i] and not signals['LongTradeOut'].at[i])
#         signals['DaysInTrade'].at[i] = signals['DaysInTrade'].shift(1).at[i] + 1 if (signals['HoldLong'].at[i] and i>0) else 0
#         #if i>0:
#         #    signals['TradeInvestment'].at[i] = signals['RollingPnL'].at[i] if (signals['LongTradeIn'].at[i]) else signals['TradeInvestment'].shift(1).at[i] if signals['HoldLong'].at[i] else 0
#         if (signals['HoldLong'].at[i] and i>0):
#             if (is_long):
#                #signals['ProfitableCloses'].at[i] = (signals['ProfitableCloses'].shift(1).at[i] + 1) if (signals['TradePnL'].at[i] > 0) else signals['ProfitableCloses'].shift(1).at[i]
#                signals['ProfitableCloses'].at[i] = signals['ProfitableCloses'].shift(1).at[i] + 1 if (signals['Close'].at[i] > signals['Close'].shift(1).at[i]) else signals['ProfitableCloses'].shift(1).at[i]
#             else:
#                signals['ProfitableCloses'].at[i] = signals['ProfitableCloses'].shift(1).at[i] + 1 if (signals['Close'].at[i] < signals['Close'].shift(1).at[i]) else signals['ProfitableCloses'].shift(1).at[i] 

#         signals['LongTradeOut'].at[i] = ((signals['Sell'].at[i] and signals['HoldLong'].at[i]) or (signals['DaysInTrade'].at[i] >= days) or (signals['ProfitableCloses'].at[i] >= prof_closes))
#         signals['TradeEntry'].at[i] = signals['Close'].at[i] if signals['LongTradeIn'].at[i] else signals['TradeEntry'].shift(1).at[i] if signals['HoldLong'].at[i] else 0
#         #signals['TradePnL'].at[i] = (signals['TradePnL'].shift(1).at[i] + (signals['Close'].at[i] - signals['Close'].shift(1).at[i])) if (signals['HoldLong'].at[i] and i>0) else 0
#         signals['TradePnL'].at[i] = (signals['Close'].at[i] - signals['TradeEntry'].at[i]) / signals['TradeEntry'].at[i] if signals['HoldLong'].at[i] else 0

#         #Calculate rolling PnL for the strategy
#         if i == 0:
#             signals['RollingPnL'].at[i] = start_capital
#         elif signals['HoldLong'].at[i]: 
#             #signals['RollingPnL'].at[i] = signals['RollingPnL'].shift(1).at[i] + signals['Close'].at[i] - signals['Close'].shift(1).at[i]
#             signals['RollingPnL'].at[i] = (1+signals['%Change'].at[i])*signals['RollingPnL'].shift(1).at[i] if is_long else (1-signals['%Change'].at[i])*signals['RollingPnL'].shift(1).at[i]
#         else:
#             signals['RollingPnL'].at[i] = signals['RollingPnL'].shift(1).at[i]    
#         #check if RollingPnL is N/A
#         #if pd.isnull(row['RollingPnL']):
#         #    baddates = baddates.append(row)


#     #signals['RollingPnL'] = signals['RollingPnL']#*point_multiplier
#     #signals['TradePnL'] = signals['TradePnL']*point_multiplier
#     signals['TradePnL'] = -1*Leverage*signals['TradePnL'] if not is_long else Leverage*signals['TradePnL']
#     # Calculate the running maximum of the 'RollingPnL' column
#     signals['RunningMax'] = signals['RollingPnL'].cummax()

#     # Calculate the drawdown as the difference between the running maximum and the current 'RollingPnL' value
#     signals['Drawdown'] = (signals['RunningMax'] - signals['RollingPnL'])/signals['RunningMax']
#     #signals['TradePnL'] = signals['TradePnL'].apply(format_dollar_value)
#     #signals['RollingPnL'] = signals['RollingPnL'].apply(format_dollar_value)
#     baddates.to_csv('baddates.csv')
#     return signals


def long_og_strat_proxy(data, days = 0, profit = 0, start_capital = 15000):

    data['OneDayBuy'] = False
    data['HoldLong'] = False
    data['LongTradeIn'] = False
    data['LongTradeOut'] = False
    data['DaysInTrade'] = 0
    data['ProfitableCloses'] = 0
    data['RollingPnL'] = 0.0
    data['TradePnL'] = 0.0
    data['TradeEntry'] = 0.0
    split_change = (data[ProxySymbol] - data[ProxySymbol].shift(1))*Leverage / data[ProxySymbol].shift(1) if data[ProxySymbol].shift(1).any() > 0 else 0
    if SplitLong:
        underlying_change = (data['%Change'])
        data['TrackChange'] = (split_change + underlying_change)/2
    else:
        data['TrackChange'] = split_change
    
    """
    if set_sell:
        data['Sell'] = ((data['RSI2'] > RSI2Sell) & (data['RSI5'] > RSI5Sell)) | (                            #RSI2 and RSI5 above threashold
            (data['Close'].shift(1) > data['EMA8'].shift(1)) & (data['Close'] < data['EMA8'])) | (            #Crossing EMA8 down
            (ExitOnVolatility) & ((data['VolumeEMADiff'] >= VolumeEMAThreashold))) | (                        #Volume more than EMAThreashold !!!!!!!!!!!DOESNT WORK - INVESTIGATE!!!!!!!!
            #(data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) < -MaxDecline) | (              #Decline more than 4%
            (data['VolumeEMADiff'] > VolumeEMAThreashold) & (data['Volatility'] > VolatilityThreashold))      #Big volume and volatility  
    """
    # Add one day buy signal
    data['OneDayBuy'] = False
    data['OneDayBuy'] = (((data['Close'] <=  data['Close'].rolling(DownDays).min()) & (data['VolumeEMADiff'] < -VolumeEMAThreasholdBuy) & (LowVolumeBuy))) | ( #Low volume buy
            (data['Close'] < data['Close'].shift(1)) & (pd.to_datetime(data['Date']).dt.dayofweek == 0) & (MondayBuy))                                          #Monday buy
                                                                                                         


    # Calculate days when entering and exiting trades
    #data['OneDayArm'] = False

    for i, row in data.iterrows():
        if i == 0:
            data['HoldLong'].at[i] = False
            data['LongTradeOut'].at[i] = False
            data['TradeEntry'].at[i] = 0
            data['TradePnL'].at[i] = 0
            data['RollingPnL'].at[i] = start_capital
        else:
            data['HoldLong'].at[i] = ((data['HoldLong'].shift(1).at[i] and not data['LongTradeOut'].shift(1).at[i]) or ( #If previously long and not trade out on previous day
                                        data['LongTradeIn'].shift(1).at[i]) or (                                        #Or if Enetering trade on previous day
                                        data['OneDayBuy'].shift(1).at[i]))                                              #Or if one day buy on previous day??? Do we need this?
            data['LongTradeIn'].at[i] = (data['Buy'].at[i] or data['OneDayBuy'].at[i]) and not data['HoldLong'].at[i]
            data['DaysInTrade'].at[i] = data['DaysInTrade'].shift(1).at[i] + 1 if (data['HoldLong'].at[i] and i>0) else 0
            if (data['HoldLong'].at[i]):
                data['ProfitableCloses'].at[i] = data['ProfitableCloses'].shift(1).at[i] + 1 if (data['Close'].at[i] > data['Close'].shift(1).at[i]) else data['ProfitableCloses'].shift(1).at[i]
            data['TradeEntry'].at[i] = data['Close'].at[i] if (((data['LongTradeIn'].at[i] or data['OneDayBuy'].at[i])) and data['TradeEntry'].shift(1).at[i] == 0) else data['TradeEntry'].shift(1).at[i] if data['HoldLong'].at[i] else 0
            data['TradePnL'].at[i] = data['TradePnL'].shift(1).at[i] + data['TrackChange'].at[i] if data['HoldLong'].at[i] else 0
            #(data['Close'].at[i] - data['TradeEntry'].at[i]) / data['TradeEntry'].at[i] if data['HoldLong'].at[i] else 0
            if HoldOnBuySignal and data['Buy'].at[i] and data['HoldLong'].at[i]:
                data['LongTradeOut'].at[i] = False
            else:
                data['LongTradeOut'].at[i] = (data['Sell'].at[i] and data['HoldLong'].at[i]) or (                           #If sell signal and hold long
                                            data['OneDayBuy'].shift(1).at[i] and not data['HoldLong'].shift(1).at[i] and not data['Buy'].shift(1).at[i]) or ( #Or if one day buy and not hold long on previous day (and not buy signal today)
                                            data['TradePnL'].at[i] < -stop_loss)                                            #Or if hit stoploss 
                if (days > 0):
                    data['LongTradeOut'].at[i] = data['LongTradeOut'].at[i] or (data['DaysInTrade'].at[i] >= days) or (data['ProfitableCloses'].at[i] >= profit)
            #data['TradeEntry'].at[i] = data['Close'].at[i] if (data['LongTradeIn'].at[i] or data['OneDayBuy'].at[i]) else data['TradeEntry'].shift(1).at[i] if data['HoldLong'].at[i] else 0


        #Calculate rolling PnL for the strategy
        if i == 0:
            data['RollingPnL'].at[i] = start_capital
        elif data['HoldLong'].at[i]: 
            #signals['RollingPnL'].at[i] = signals['RollingPnL'].shift(1).at[i] + signals['Close'].at[i] - signals['Close'].shift(1).at[i]
            data['RollingPnL'].at[i] = (1+data['TrackChange'].at[i])*data['RollingPnL'].shift(1).at[i]
        else:
            data['RollingPnL'].at[i] = data['RollingPnL'].shift(1).at[i]   
    # print (data['RollingPnL'])

    # Calculate the running maximum of the 'RollingPnL' column
    data['RunningMax'] = data['RollingPnL'].cummax()

    # Calculate the drawdown as the difference between the running maximum and the current 'RollingPnL' value
    data['Drawdown'] = (data['RunningMax'] - data['RollingPnL'])/data['RunningMax']
    #signals['TradePnL'] = signals['TradePnL'].apply(format_dollar_value)
    #signals['RollingPnL'] = signals['RollingPnL'].apply(format_dollar_value)
    # print (data.head(-10))

    return data

def long_strat(data, days, prof_closes, is_long = True, start_capital = 15000, point_multiplier = point_multiplier, *, pnl_column=None):
    signals = data
    n = len(signals)

    if pnl_column:
        split_change = (data[pnl_column] - data[pnl_column].shift(1))*Leverage / data[pnl_column].shift(1) if data[pnl_column].shift(1).any() > 0 else 0
        signals['TrackChange'] = split_change
    elif not UseProxyUnderlying:
        signals['TrackChange'] = signals['%Change']
    else:
        split_change = (data[ProxySymbol] - data[ProxySymbol].shift(1))*Leverage / data[ProxySymbol].shift(1) if data[ProxySymbol].shift(1).any() > 0 else 0
        if SplitLong:
            underlying_change = data['%Change']
            signals['TrackChange'] = (split_change + underlying_change)/2
        else:
            signals['TrackChange'] = split_change

    buy = np.asarray(signals['Buy'].values, dtype=bool)
    sell = np.asarray(signals['Sell'].values, dtype=bool)
    price_col = pnl_column or 'Close'
    close = np.asarray(signals[price_col].values, dtype=float)
    track_change = np.asarray(signals['TrackChange'].fillna(0).values, dtype=float)

    long_in = np.zeros(n, dtype=bool)
    long_out = np.zeros(n, dtype=bool)
    hold_long = np.zeros(n, dtype=bool)
    days_in_trade = np.zeros(n, dtype=int)
    profitable_closes = np.zeros(n, dtype=int)
    rolling_pnl = np.zeros(n, dtype=float)
    trade_pnl = np.zeros(n, dtype=float)
    trade_entry = np.zeros(n, dtype=float)

    def mark_price(i: int) -> float:
        price = close[i]
        if price and not np.isnan(price):
            return price
        if i <= 0:
            return price
        prior = close[:i]
        valid = prior[~np.isnan(prior) & (prior != 0)]
        return valid[-1] if valid.size else np.nan

    for i in range(n):
        if i > 0:
            hold_long[i] = (hold_long[i - 1] and not long_out[i - 1]) or long_in[i - 1]

        long_in[i] = buy[i] and not hold_long[i]
        days_in_trade[i] = days_in_trade[i - 1] + 1 if hold_long[i] and i > 0 else 0

        if hold_long[i] and i > 0:
            if is_long:
                profitable_closes[i] = profitable_closes[i - 1] + 1 if close[i] > close[i - 1] else profitable_closes[i - 1]
            else:
                profitable_closes[i] = profitable_closes[i - 1] + 1 if close[i] < close[i - 1] else profitable_closes[i - 1]

        if HoldOnBuySignal and buy[i] and hold_long[i]:
            long_out[i] = False
        else:
            long_out[i] = (sell[i] and hold_long[i]) or (days_in_trade[i] >= days) or (profitable_closes[i] >= prof_closes)
        if long_in[i]:
            trade_entry[i] = mark_price(i)
        elif hold_long[i]:
            trade_entry[i] = trade_entry[i - 1] if i > 0 else 0

        price = mark_price(i)
        entry = trade_entry[i]
        trade_pnl[i] = (price - entry) / entry if hold_long[i] and entry and not np.isnan(price) else 0

        if i == 0:
            rolling_pnl[i] = start_capital
        elif hold_long[i]:
            change = track_change[i]
            if np.isnan(change):
                change = 0.0
            if is_long:
                rolling_pnl[i] = (1 + change) * rolling_pnl[i - 1]
            else:
                rolling_pnl[i] = (1 - change) * rolling_pnl[i - 1]
        else:
            rolling_pnl[i] = rolling_pnl[i - 1]

    trade_pnl = Leverage * trade_pnl if is_long else -1 * Leverage * trade_pnl

    signals['LongTradeIn'] = long_in
    signals['LongTradeOut'] = long_out
    signals['HoldLong'] = hold_long
    signals['DaysInTrade'] = days_in_trade
    signals['ProfitableCloses'] = profitable_closes
    signals['RollingPnL'] = rolling_pnl
    signals['TradePnL'] = trade_pnl
    signals['TradeEntry'] = trade_entry
    signals['RunningMax'] = signals['RollingPnL'].cummax()
    signals['Drawdown'] = (signals['RunningMax'] - signals['RollingPnL'])/signals['RunningMax']
    return signals


def long_strat_reference(data, days, prof_closes, is_long = True, start_capital = 15000, point_multiplier = point_multiplier):
    signals = data
    signals['LongTradeIn'] = False
    signals['LongTradeOut'] = False
    signals['HoldLong'] = False
    signals['DaysInTrade'] = 0
    signals['ProfitableCloses'] = 0
    signals['RollingPnL'] = 0.0
    signals['TradePnL'] = 0.0
    signals['TradeEntry'] = 0.0
    baddates = pd.DataFrame()
    if not UseProxyUnderlying:
        data['TrackChange'] = data['%Change']
    else:
        split_change = (data[ProxySymbol] - data[ProxySymbol].shift(1))*Leverage / data[ProxySymbol].shift(1) if data[ProxySymbol].shift(1).any() > 0 else 0
        if SplitLong:
            underlying_change = (data['%Change'])
            data['TrackChange'] = (split_change + underlying_change)/2
        else:
            data['TrackChange'] = split_change
    #signals['TradeInvestment'] = 0

    for i, row in signals.iterrows():
        signals['HoldLong'].at[i] = (signals['HoldLong'].shift(1).at[i] and not signals['LongTradeOut'].shift(1).at[i] and (i>0)) or ( #If previously long and not trade out on previous day
                                    signals['LongTradeIn'].shift(1).at[i] and i>0)                                            #Or if Enetering trade on previous day
        #signals['LongTradeIn'].at[i] = signals['Buy'].at[i] and not signals['HoldLong'].at[i]
        signals['LongTradeIn'].at[i] = signals['Buy'].at[i] and not (signals['HoldLong'].at[i] and not signals['LongTradeOut'].at[i])
        signals['DaysInTrade'].at[i] = signals['DaysInTrade'].shift(1).at[i] + 1 if (signals['HoldLong'].at[i] and i>0) else 0
        #if i>0:
        #    signals['TradeInvestment'].at[i] = signals['RollingPnL'].at[i] if (signals['LongTradeIn'].at[i]) else signals['TradeInvestment'].shift(1).at[i] if signals['HoldLong'].at[i] else 0
        if (signals['HoldLong'].at[i] and i>0):
            if (is_long):
               #signals['ProfitableCloses'].at[i] = (signals['ProfitableCloses'].shift(1).at[i] + 1) if (signals['TradePnL'].at[i] > 0) else signals['ProfitableCloses'].shift(1).at[i]
               signals['ProfitableCloses'].at[i] = signals['ProfitableCloses'].shift(1).at[i] + 1 if (signals['Close'].at[i] > signals['Close'].shift(1).at[i]) else signals['ProfitableCloses'].shift(1).at[i]
            else:
               signals['ProfitableCloses'].at[i] = signals['ProfitableCloses'].shift(1).at[i] + 1 if (signals['Close'].at[i] < signals['Close'].shift(1).at[i]) else signals['ProfitableCloses'].shift(1).at[i] 

        signals['LongTradeOut'].at[i] = ((signals['Sell'].at[i] and signals['HoldLong'].at[i]) or (signals['DaysInTrade'].at[i] >= days) or (signals['ProfitableCloses'].at[i] >= prof_closes))
        signals['TradeEntry'].at[i] = signals['Close'].at[i] if signals['LongTradeIn'].at[i] else signals['TradeEntry'].shift(1).at[i] if signals['HoldLong'].at[i] else 0
        #signals['TradePnL'].at[i] = (signals['TradePnL'].shift(1).at[i] + (signals['Close'].at[i] - signals['Close'].shift(1).at[i])) if (signals['HoldLong'].at[i] and i>0) else 0
        signals['TradePnL'].at[i] = (signals['Close'].at[i] - signals['TradeEntry'].at[i]) / signals['TradeEntry'].at[i] if signals['HoldLong'].at[i] else 0

        #Calculate rolling PnL for the strategy
        if i == 0:
            signals['RollingPnL'].at[i] = start_capital
        elif signals['HoldLong'].at[i]: 
            #signals['RollingPnL'].at[i] = signals['RollingPnL'].shift(1).at[i] + signals['Close'].at[i] - signals['Close'].shift(1).at[i]
            signals['RollingPnL'].at[i] = (1+signals['TrackChange'].at[i])*signals['RollingPnL'].shift(1).at[i] if is_long else (1-signals['TrackChange'].at[i])*signals['RollingPnL'].shift(1).at[i]
        else:
            signals['RollingPnL'].at[i] = signals['RollingPnL'].shift(1).at[i]    
        #check if RollingPnL is N/A
        #if pd.isnull(row['RollingPnL']):
        #    baddates = baddates.append(row)


    #signals['RollingPnL'] = signals['RollingPnL']#*point_multiplier
    #signals['TradePnL'] = signals['TradePnL']*point_multiplier
    signals['TradePnL'] = -1*Leverage*signals['TradePnL'] if not is_long else Leverage*signals['TradePnL']
    # Calculate the running maximum of the 'RollingPnL' column
    signals['RunningMax'] = signals['RollingPnL'].cummax()

    # Calculate the drawdown as the difference between the running maximum and the current 'RollingPnL' value
    signals['Drawdown'] = (signals['RunningMax'] - signals['RollingPnL'])/signals['RunningMax']
    #signals['TradePnL'] = signals['TradePnL'].apply(format_dollar_value)
    #signals['RollingPnL'] = signals['RollingPnL'].apply(format_dollar_value)
    baddates.to_csv('baddates.csv')
    return signals