import pandas as pd
import yfinance as yf
from config import *
import getdata as dt
import datetime
import indicators as ind

 #data = dt.get_data_ib(ib, nq, False, period = '1 Y') #True = use RTH data, False = use all data
if (ticker == 'NQ' or ticker == 'ES' or ticker == 'RTY' or ticker == 'CL' or ticker == 'GC' or ticker == 'SI' or ticker == 'HG'):
        #contract = Future(ticker, '202306', 'CME')
        yfticker = ticker + '=F'
elif (ticker == 'GBPUSD'):
        #contract = Forex(ticker)
        yfticker = ticker + '=X'
else:
        #contract = Stock(ticker, 'ARCA')
        yfticker = ticker

data = dt.get_data_yf(yfticker, 25, False) #True for local data, False for Yahoo Finance

data = dt.normalize_dataframe(data) #Capitalize the column names
data = dt.clean_holidays(data) #Remove holidays
data = ind.add_indicators(data)
#data = pd.read_csv('data.csv')

print(data['Date'].dt.year[0])
#print(data['Date'].year)