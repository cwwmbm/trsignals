import pandas as pd
#import nest_asyncio

import getdata as dt
import numpy as np
import indicators as ind
from config import *

import time

import backtest as bt


i=0
start_time = time.perf_counter()

signals = pd.DataFrame()
results = pd.DataFrame()
symbols = ['SPY', 'SMH', 'QQQ', 'SOXX','^VIX', 'XLI','XLU','XLE','XLF','RSP', 'IWM', 'FXI', 'AAPL', 'GDX', 'IBB', 'GLD', 'SLV', 'EEM', 'INDA', 'USO', 'CPER', 'UNG', 'XLK', 'IBM', 'CSCO']

buy_signal = ind.buy_signal4

yf_symbols = [symbol+'=F' if symbol in ['NQ', 'ES', 'RTY', 'CL', 'GC', 'SI', 'HG'] else symbol for symbol in symbols]
symbol_mapping = {symbol: yf_symbol for symbol, yf_symbol in zip(symbols, yf_symbols)}
full_data = dt.get_bulk_data(yf_symbols, years = 20)
vix_close = full_data['Close', '^VIX']
breadth = full_data['Close', 'RSP'] / full_data['Close', 'SPY']
qqq_to_spy = full_data['Close']['QQQ'] / full_data['Close']['SPY']
smh_to_spy = full_data['Close']['SMH'] / full_data['Close']['SPY']
xlf_to_spy = full_data['Close']['XLF'] / full_data['Close']['SPY']
xle_to_spy = full_data['Close']['XLE'] / full_data['Close']['SPY']
xlu_to_spy = full_data['Close']['XLU'] / full_data['Close']['SPY']
xli_to_spy = full_data['Close']['XLI'] / full_data['Close']['SPY']
gold_to_spy = full_data['Close']['GLD'] / full_data['Close']['SPY']
# full_data['Goldbreadth'] = gold_to_spy
spy50 = full_data['Close']['SPY'].rolling(50).mean()
spy200 = full_data['Close']['SPY'].rolling(200).mean()
        


for symbol, yf_symbol in symbol_mapping.items():
    if symbol in ['^VIX', 'RSP']:
        continue
    data= full_data.xs(yf_symbol, axis=1, level=1, drop_level=False)

    i+=0.04
    print('Getting data for ' + symbol + '...')
    data.columns = data.columns.droplevel(1)  # Reset column level
    data = data.copy()
    data['VIX'] = vix_close
    data['Breadth'] = breadth
    data['RiskBreadth'] = qqq_to_spy
    data['SemisBreadth'] = smh_to_spy
    data['FinancialsBreadth'] = xlf_to_spy
    data['EnergyBreadth'] = xle_to_spy
    data['UtilitiesBreadth'] = xlu_to_spy
    data['IndustrialsBreadth'] = xli_to_spy
    data['GoldBreadth'] = gold_to_spy
    data['SPYBull'] = np.where(spy50>spy200, 1, -1)
    data = dt.normalize_dataframe(data)
    data = data.drop(columns = ['Adj close'])
    data = dt.clean_holidays(data)

    data = ind.add_indicators(data)

    # data_temp = data.copy()
    data['Buy'], data['Sell'], days, profit, description, verdict, is_long, ignore = buy_signal(data, symbol)
    # data['Buy'] = data['Buy'] & (data['IBR'] < 0.1)

    data = bt.execute_strategy(data, days, profit)
    # number_of_trades = data['LongTradeOut'].value_counts().get(True, 0)
    # trade_out_rows = data[data['LongTradeOut']]
    # num_profitable_trades = (trade_out_rows['TradePnL'] > 0).sum()
    latest_rolling_pnl = ind.format_dollar_value(data['RollingPnL'].iloc[-1])
    max_drawdown = data['Drawdown'].max()
    formatted_drawdown = '{:.2%}'.format(max_drawdown)
    first_day = data.iloc[0]
    last_day = data.iloc[-1]
    cagr = (last_day['RollingPnL'] / first_day['RollingPnL']) ** (1 / (data.shape[0] / 252)) - 1
    # percentage_profitable_trades = (num_profitable_trades / number_of_trades) * 100
    # average_positive_trade_pnl = 100*trade_out_rows[trade_out_rows['TradePnL'] > 0]['TradePnL'].mean()
    # average_negative_trade_pnl = 100*trade_out_rows[trade_out_rows['TradePnL'] < 0]['TradePnL'].mean()
    sharpe = ind.sharpes_ratio(data)
    sortino = ind.sortino_ratio(data)
    # kelly = (percentage_profitable_trades/100 - ((1 - percentage_profitable_trades/100) / (average_positive_trade_pnl / (-average_negative_trade_pnl))))*100

    results = results._append({'Symbol': symbol,'PNL:': latest_rolling_pnl, 'Max Drawdown': formatted_drawdown, 'CAGR': cagr, 'Sharpe': sharpe, 'Sortino': sortino}, ignore_index=True)


# buy_signals = signals[signals['Buy signal?'] == True]
# sell_signals = signals[signals['Sell signal?'] == True]

# print (signals)
print (results)

end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")


