import yfinance as yf
import pandas as pd
import mplfinance as mpf
import ta
import math

#Setting up trade variables
#Base variables
Local = False #True if take local csv, False if take from yahoo finance
RSI2Buy = 15
RSI5Buy = 35
MacDecline = 0.04
RSI2Sell = 95
RSI5Sell = 70
VolatilityPeriod = 5
VolatilityThreashold = 0.1
VolumeEMAThreashold = 0.6
ExitOnVolatility = True

#One day buy variables
MondayBuy = True
LowVolumeBuy = True
VolumeEMAThreasholdBuy = 0.15
DownDays = 3
OneDayArm = False #No code written to support this yet

#Function to get the data
def getData(ticker, years=1, Local=False):
    start_date = pd.to_datetime("today") - pd.DateOffset(years=years)
    end_date = pd.to_datetime("today")
    if Local:
        print("Using local csv")
        data = pd.read_csv('SPY.csv', index_col='Date', parse_dates=True)
    else:
        print("Using yahoo finance")
        data = yf.download(ticker, start=start_date, end=end_date)
        data.to_csv('SPY.csv')
    return data


def main():
    # 1. Get 1y SPY data from csv or yahoo finance, based on Local variable
    data = getData('SPY', 1, Local)

    # Calculate indicators
    data['RSI2'] = ta.momentum.RSIIndicator(data['Close'], window=2).rsi()
    data['RSI5'] = ta.momentum.RSIIndicator(data['Close'], window=5).rsi()
    data['EMA8'] = ta.trend.ema_indicator(data['Close'], window=8)
    data['VolumeEMADiff'] = (data['Volume'] - ta.trend.ema_indicator(data['Volume'], window=8)) / ta.trend.ema_indicator(data['Volume'], window=8)

    #Calculating volatily with period of 5 days
    data['AdjustedChange'] = data['Close']/data['Close'].shift(1) - 1
    data['Volatility'] = data['AdjustedChange'].rolling(VolatilityPeriod).std() * math.sqrt(252)

    # Main Trading signals
    data['MainBuy'] = (data['RSI2'] < RSI2Buy) & (data['RSI5'] < RSI5Buy) & (                                 #RSI2 and RSI5 below threashold
            (data['VolumeEMADiff'] < VolumeEMAThreashold) | (data['Volatility'] < VolumeEMAThreashold)) & (  #Volume less than EMAThreashold
            ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) > -MacDecline))                #Decline less than 4%

    data['Sell'] = ((data['RSI2'] > RSI2Sell) & (data['RSI5'] > RSI5Sell)) | (                                #RSI2 and RSI5 above threashold
            (data['Close'].shift(1) > data['EMA8'].shift(1)) & (data['Close'] < data['EMA8'])) | (            #Crossing EMA8 down
            (ExitOnVolatility) & ((data['VolumeEMADiff'] >= VolumeEMAThreashold)))                             #Volume more than EMAThreashold !!!!!!!!!!!DOESNT WORK - INVESTIGATE!!!!!!!!

    # Add one day buy signal
    data['OneDayBuy'] = False
    data['OneDayBuy'] = (((data['Close'] <=  data['Close'].rolling(DownDays).min()) & (data['VolumeEMADiff'] < -VolumeEMAThreasholdBuy) & (LowVolumeBuy))) | ( #Low volume buy
            (data['Close'] < data['Close'].shift(1)) & (data.index.dayofweek == 0) & (MondayBuy))                                                                                                         #Monday buy


    # Calculate days when entering and exiting trades
    data['HoldLong'] = False
    data['TradeIn'] = False
    data['TradeOut'] = False
    #data['OneDayArm'] = False

    for i, row in data.iterrows():
        data['HoldLong'].at[i] = (data['HoldLong'].shift(1).at[i] and not data['TradeOut'].shift(1).at[i]) or ( #If previously long and not trade out on previous day
                                    data['TradeIn'].shift(1).at[i]) or (                                        #Or if Enetering trade on previous day
                                    data['OneDayBuy'].shift(1).at[i])                                           #Or if one day buy on previous day??? Do we need this?
        data['TradeIn'].at[i] = data['MainBuy'].at[i] and not data['HoldLong'].at[i]
        data['TradeOut'].at[i] = (data['Sell'].at[i] and data['HoldLong'].at[i]) or (                           #If sell signal and hold long
                                    data['OneDayBuy'].shift(1).at[i] and not data['HoldLong'].shift(1).at[i] and not data['MainBuy'].shift(1).at[i])   #Or if one day buy and not hold long on previous day (and not buy signal today)



    print(data.tail(30))

if __name__ == '__main__':
    main()



