import pandas as pd
import getdata as dt
import indicators as ind
from config import *
from ib_insync import IB, Future, util, Stock
from pushbullet import Pushbullet
import time
import backtest as bt

start_time = time.perf_counter()
if (ticker == 'NQ' or ticker == 'ES' or ticker == 'RTY' or ticker == 'CL' or ticker == 'GC' or ticker == 'SI' or ticker == 'HG'):
        yfticker = ticker + '=F'
elif (ticker == 'GBPUSD'):
        yfticker = ticker + '=X'
else:
        yfticker = ticker
data = dt.get_data_yf(yfticker, 20, False) #True for local data, False for Yahoo Finance
#vix_data = dt.get_data_yf('^VIX', 20, False)
#Add VIX data to dataframe
#data['VIX'] = vix_data['Close']
data = dt.normalize_dataframe(data) #Capitalize the column names
data = dt.clean_holidays(data) #Remove holidays
data = ind.add_indicators(data)
signals = pd.DataFrame()

buy_signals = [ind.buy_signal1, ind.buy_signal2, ind.buy_signal3, ind.buy_signal4, ind.buy_signal5, ind.buy_signal6, ind.buy_signal7, ind.buy_signal8, ind.buy_signal9, ind.buy_signal10, 
               ind.buy_signal11, ind.buy_signal12, ind.buy_signal13, ind.buy_signal14, ind.buy_signal15, ind.buy_signal16, ind.buy_signal17, ind.buy_signal18, ind.buy_signal19, ind.buy_signal20, ind.buy_signal21,
               ind.og_buy_signal, ind.og_new_buy_signal ]
for buy_signal in buy_signals:
    data_temp = data.copy()
    data_temp['Buy'], data_temp['Sell'], days, profit, description, verdict, is_long, ignore = buy_signal(data)

    #data_temp = ind.long_strat(data_temp, days, profit) if days>0 else ind.og_strat(data_temp, set_sell = False)
    data_temp = bt.execute_strategy(data_temp, days, profit, is_long)
    max_drawdown = data_temp['Drawdown'].max()
    formatted_drawdown = '{:.2%}'.format(max_drawdown)
    latest_rolling_pnl = ind.format_dollar_value(data_temp['RollingPnL'].iloc[-1] - 15000)
    number_of_trades = data_temp['LongTradeOut'].value_counts().get(True, 0)
    # Filter the rows where 'LongTradeOut' is True
    trade_out_rows = data_temp[data_temp['LongTradeOut']]
    # Calculate the number of profitable trades
    num_profitable_trades = (trade_out_rows['TradePnL'] > 0).sum()

    percentage_profitable_trades = (num_profitable_trades / number_of_trades)
    percentage_profitable_trades = '{:.2%}'.format(percentage_profitable_trades)


    signals = signals._append([{'Signal': buy_signal.__name__, 'RollingPnL': latest_rolling_pnl, 'MaxDD': formatted_drawdown, 'TradeNumber': number_of_trades, 'ProfitableTrades': percentage_profitable_trades,'Days': days, 'Profit': profit}])

print(signals)

end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")