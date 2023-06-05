from config import *
import indicators as ind
import numpy as np
import pandas as pd
import getdata as dt
import datetime as dtm

data = pd.DataFrame()
if (ticker == 'NQ' or ticker == 'ES' or ticker == 'RTY' or ticker == 'CL' or ticker == 'GC' or ticker == 'SI' or ticker == 'HG' or ticker == 'NG'):
        #contract = Future(ticker, '202306', 'CME')
        yfticker = ticker + '=F'
elif (ticker == 'GBPUSD'):
        #contract = Forex(ticker)
        yfticker = ticker + '=X'
else:
        #contract = Stock(ticker, 'ARCA')
        yfticker = ticker

data = dt.get_data_yf(yfticker, years=1, Local = False) #True for local data, False for Yahoo Finance
data = dt.normalize_dataframe(data) #Capitalize the column names
data = dt.clean_holidays(data) #Remove holidays
data = ind.add_indicators(data)

print("SPY: " + str(round(100*data['Close'].pct_change().iloc[-1], 2)) + "%")
print("QQQ: " + str(round(100*data['Qqq'].pct_change().iloc[-1], 2)) + "%")
print("SOXX: " + str(round(100*data['Soxx'].pct_change().iloc[-1], 2)) + "%")
print("Today's Close: " + str(round(data['Close'].iloc[-1], 2)))
print("EMA8: " + str(round(data['EMA8'].iloc[-1], 2)))
print("RSI2: " + str(round(data['RSI2'].iloc[-1], 2)))
print("RSI5: " + str(round(data['RSI5'].iloc[-1], 2)))
