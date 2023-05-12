import pandas as pd
#import nest_asyncio
#nest_asyncio.apply()
import getdata as dt
import indicators as ind
from config import *
#from ib_insync import IB, Future, util, Stock
#from pushbullet import Pushbullet
import time
import streamlit as st

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
symbols = ['NQ', 'ES', 'CL', 'GC', 'SI', 'SPY', 'SMH', 'QQQ', '^VIX', 'RSP']

buy_signals = [ind.buy_signal1, ind.buy_signal2, ind.buy_signal3, ind.buy_signal4, ind.buy_signal5, ind.buy_signal6, ind.buy_signal7, ind.buy_signal8, ind.buy_signal9, ind.buy_signal10, 
               ind.buy_signal11, ind.buy_signal12, ind.buy_signal13, ind.buy_signal14, ind.buy_signal15, ind.buy_signal16, ind.buy_signal17, ind.buy_signal18, ind.buy_signal19, ind.og_buy_signal, ind.og_new_buy_signal]

yf_symbols = [symbol+'=F' if symbol in ['NQ', 'ES', 'RTY', 'CL', 'GC', 'SI', 'HG'] else symbol for symbol in symbols]
symbol_mapping = {symbol: yf_symbol for symbol, yf_symbol in zip(symbols, yf_symbols)}
full_data = dt.get_bulk_data(yf_symbols)
vix_close = full_data['Close', '^VIX']
breadth = full_data['Close', 'SPY'] / full_data['Close', 'RSP']

for symbol, yf_symbol in symbol_mapping.items():
    if symbol in ['^VIX', 'RSP']:
        continue
    data= full_data.xs(yf_symbol, axis=1, level=1, drop_level=False)
    status.text('Getting data for ' + symbol + '...')
    bar.progress(i)
    i+=0.125
    print('Getting data for ' + symbol + '...')
    data.columns = data.columns.droplevel(1)  # Reset column level
    data = data.copy()
    data['VIX'] = vix_close
    data['Breadth'] = breadth
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
        data_temp['Buy'], days, profit, description, verdict, is_long, ignore = buy_signal(data_temp, symbol)
        if buy_signal.__name__ == 'og_buy_signal':
            data_temp['Sell'] = ind.og_sell_signal(data_temp)
        elif buy_signal.__name__ == 'og_new_buy_signal':
            data_temp['Sell'] = ind.og_new_sell_signal(data_temp)
        if not ignore:
            data_temp = ind.long_strat(data_temp, days, profit) if days>0 else ind.og_strat(data_temp, set_sell=False)
            #find TradePnL in % and store it as string with % sign in the end
            trade_pnl = str(round(data_temp['TradePnL'].iloc[-1]*100,2)) + '%'
            signals = signals._append([{'Symbol': symbol,'Signal': buy_signal.__name__, 'Buy signal?': data_temp['LongTradeIn'].iloc[-1], 'HoldLong?': data_temp['HoldLong'].iloc[-1], 'Sell signal?': data_temp['LongTradeOut'].iloc[-1],
                                        'Days': days, 'Profit': profit, 'TradePnL': trade_pnl,'Description': description,'Verdict': verdict, 'Date': data_temp['Date'].iloc[-1]}])

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

print (signals)
print(message)
#data['Buy'] = (data['Close'].shift(1) <= data['Close'].shift(3)) & (data['IBR'] <= 0.4) #& (data['Close'].pct_change(periods=10) < 0) #Hold 3 days profit 1
#data = ind.longStrat(data,3,1)
#data.to_csv('NQ_sgnl.csv')
#ib.disconnect()

end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")


