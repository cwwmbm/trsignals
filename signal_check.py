import pandas as pd
#import nest_asyncio
#nest_asyncio.apply()
import getdata as dt
import numpy as np
import indicators as ind
from config import *
#from ib_insync import IB, Future, util, Stock
#from pushbullet import Pushbullet
import time
import streamlit as st
import backtest as bt

#def send_push_notification(title, message, api_key):
#    pb = Pushbullet(api_key)
#    pb.push_note(title, message)

bar = st.progress(0)
status = st.empty()
i=0
start_time = time.perf_counter()
#ib = IB()
#Get historical data from yfinance and real time data for today from IB and combined them in one table
#data = dt.get_full_data(ib, False)
#data = dt.clean_holidays(data) #Remove holidays
#Add indicators to the table
#data = ind.add_indicators(data)
signals = pd.DataFrame()
symbols = ['SPY', 'SMH', 'QQQ', 'SOXX','^VIX', 'XLI','XLU','XLE','XLF','RSP', 'IWM', 'FXI', 'AAPL']

buy_signals = [ind.buy_signal1, ind.buy_signal2, ind.buy_signal3, ind.buy_signal4, ind.buy_signal5, ind.buy_signal6, ind.buy_signal7, ind.buy_signal8, ind.buy_signal9, ind.buy_signal10, 
               ind.buy_signal11, ind.buy_signal12, ind.buy_signal13, ind.buy_signal14, ind.buy_signal15, ind.buy_signal16, ind.buy_signal17, ind.buy_signal18, ind.buy_signal19, ind.buy_signal20, ind.buy_signal21, 
               ind.og_buy_signal, ind.og_new_buy_signal]

yf_symbols = [symbol+'=F' if symbol in ['NQ', 'ES', 'RTY', 'CL', 'GC', 'SI', 'HG'] else symbol for symbol in symbols]
symbol_mapping = {symbol: yf_symbol for symbol, yf_symbol in zip(symbols, yf_symbols)}
full_data = dt.get_bulk_data(yf_symbols, years = 1)
vix_close = full_data['Close', '^VIX']
breadth = full_data['Close', 'RSP'] / full_data['Close', 'SPY']
qqq_to_spy = full_data['Close']['QQQ'] / full_data['Close']['SPY']
smh_to_spy = full_data['Close']['SMH'] / full_data['Close']['SPY']
xlf_to_spy = full_data['Close']['XLF'] / full_data['Close']['SPY']
xle_to_spy = full_data['Close']['XLE'] / full_data['Close']['SPY']
xlu_to_spy = full_data['Close']['XLU'] / full_data['Close']['SPY']
xli_to_spy = full_data['Close']['XLI'] / full_data['Close']['SPY']
spy50 = full_data['Close']['SPY'].rolling(50).mean()
spy200 = full_data['Close']['SPY'].rolling(200).mean()
        


for symbol, yf_symbol in symbol_mapping.items():
    if symbol in ['^VIX', 'RSP', 'XLI','XLU','XLE','XLF']:
        continue
    data= full_data.xs(yf_symbol, axis=1, level=1, drop_level=False)
    status.text('Getting data for ' + symbol + '...')
    bar.progress(i)
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
    data['SPYBull'] = np.where(spy50>spy200, 1, -1)
    data = dt.normalize_dataframe(data)
    data = data.drop(columns = ['Adj close'])
    data = dt.clean_holidays(data)
    #print("Getting data for " + symbol + "...")
    #data = dt.get_full_data(ib, False, symbol = symbol)
    data = ind.add_indicators(data)
    #print(data.tail(5))
    #time.sleep(5)
    for buy_signal in buy_signals:
        data_temp = data.copy()
        data_temp['Buy'], data_temp['Sell'], days, profit, description, verdict, is_long, ignore = buy_signal(data_temp, symbol)
        if not ignore:
            #data_temp = ind.long_strat(data_temp, days, profit) if days>0 else ind.og_strat(data_temp, set_sell=False)
            data_temp = bt.execute_strategy(data_temp, days, profit)
            trade_pnl = str(round(data_temp['TradePnL'].iloc[-1]*100,2)) + '%'
            #Calculate Kelly Criterion
            number_of_trades = data_temp['LongTradeOut'].value_counts().get(True, 0)
            trade_out_rows = data_temp[data_temp['LongTradeOut']]
            num_profitable_trades = (trade_out_rows['TradePnL'] > 0).sum()
            percentage_profitable_trades = (num_profitable_trades / number_of_trades)
            average_positive_trade_pnl = trade_out_rows[trade_out_rows['TradePnL'] > 0]['TradePnL'].mean()
            average_negative_trade_pnl = trade_out_rows[trade_out_rows['TradePnL'] < 0]['TradePnL'].mean()
            kelly = (percentage_profitable_trades - ((1 - percentage_profitable_trades) / (average_positive_trade_pnl / (-average_negative_trade_pnl))))*100
            signals = signals._append([{'Symbol': symbol,'Signal': buy_signal.__name__, 'Buy signal?': data_temp['LongTradeIn'].iloc[-1], 'HoldLong?': data_temp['HoldLong'].iloc[-1], 'Sell signal?': data_temp['LongTradeOut'].iloc[-1],
                                        'Days': days, 'Profit': profit, 'TradePnL': trade_pnl,'Kelly': (str(round(kelly,2))+"%"),'Description': description,'Verdict': verdict}])

status.text('Done')
bar.progress(100)           
st.write(signals)
buy_signals = signals[signals['Buy signal?'] == True]
sell_signals = signals[signals['Sell signal?'] == True]
#''' PUSH NOTIFICATIONS 
if (len(buy_signals) == 0) & (len(sell_signals) == 0):
    title = "Signals Update"
    message = "No signals today"
    #extract signal name and description into a string message
    #message = "Signal is: " + str(signals['Signal'].tolist()) + "Description: " + str(signals['Description'].tolist())
    #message = "Buy signals: " + 
    #send_push_notification(title, message, api_key)
else:
    title = "Signals Update"
    message = "Buy signals: " + str(buy_signals['Signal'].tolist()) + "Sell signals: " + str(sell_signals['Signal'].tolist())
    #send_push_notification(title, message, api_key)
data1 = dt.get_data_yf('SPY', years=1, Local = False) #True for local data, False for Yahoo Finance
data1 = dt.normalize_dataframe(data1) #Capitalize the column names
data1 = dt.clean_holidays(data1) #Remove holidays
data1 = ind.add_indicators(data1)
st.write("SPY: " + str(round(100*data1['Close'].pct_change().iloc[-1], 2)) + "%")
st.write("QQQ: " + str(round(100*data1['Qqq'].pct_change().iloc[-1], 2)) + "%")
st.write("IWM: " + str(round(100*data1['Iwm'].pct_change().iloc[-1], 2)) + "%")
print("SOXX: " + str(round(100*data1['Soxx'].pct_change().iloc[-1], 2)) + "%")
print("Today's Close: " + str(round(data1['Close'].iloc[-1], 2)))
print("EMA8: " + str(round(data1['EMA8'].iloc[-1], 2)))
print("RSI2: " + str(round(data1['RSI2'].iloc[-1], 2)))
print("RSI5: " + str(round(data1['RSI5'].iloc[-1], 2)))
print("Stoch: " + str(round(data1['Stoch'].iloc[-1], 2)))
print("Volume EMA: " + str(round(100*data1['VolumeEMADiff'].iloc[-1], 2))+"%")
print (signals)
print(message)
#data['Buy'] = (data['Close'].shift(1) <= data['Close'].shift(3)) & (data['IBR'] <= 0.4) #& (data['Close'].pct_change(periods=10) < 0) #Hold 3 days profit 1
#data = ind.longStrat(data,3,1)
#data.to_csv('NQ_sgnl.csv')
#ib.disconnect()

end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")


