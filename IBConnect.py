from ib_insync import IB, Future, util, Stock
import ta
import yfinance as yf
import pandas as pd
import numpy as np
import getdata as dt
from datetime import datetime as dtm
import indicators as ind
import math
import backtest as bt
import time
from config import *

# Print out stats for the strategy
def print_stats(data, days = 0, profit = 0, description = "Original Strategy"):
    number_of_trades = data['LongTradeOut'].value_counts().get(True, 0)
    print(f"Number of trades: {number_of_trades}")
    # Filter the rows where 'LongTradeOut' is True
    trade_out_rows = data[data['LongTradeOut']]

    # Calculate the number of profitable trades
    num_profitable_trades = (trade_out_rows['TradePnL'] > 0).sum()

    latest_rolling_pnl = ind.format_dollar_value(data['RollingPnL'].iloc[-1])
    print(f"Latest Rolling PnL: {latest_rolling_pnl}")

    # Find the maximum drawdown
    max_drawdown = data['Drawdown'].max()
    formatted_drawdown = '{:.2%}'.format(max_drawdown)
    print(f"Maximum drawdown: {formatted_drawdown}")


    # Calculate the percentage of profitable trades
    percentage_profitable_trades = (num_profitable_trades / number_of_trades) * 100

    print(f"Percentage of profitable trades: {percentage_profitable_trades:.2f}")

    #Calculate average positive trade PnL
    average_positive_trade_pnl = 100*trade_out_rows[trade_out_rows['TradePnL'] > 0]['TradePnL'].mean()
    print(f"Average positive trade PnL: {average_positive_trade_pnl:.2f}%")
    #print(f"Average positive trade PnL: {average_positive_trade_pnl:.2f}%")

    #Calculate average negative trade PnL
    average_negative_trade_pnl = 100*trade_out_rows[trade_out_rows['TradePnL'] < 0]['TradePnL'].mean()
    print(f"Average negative trade PnL: {average_negative_trade_pnl:.2f}%")
    #print(f"Average negative trade PnL: {average_negative_trade_pnl:.2f}%")

    #Print Sharpe Ratio
    sharpe = ind.sharpes_ratio(data)
    print(f"Sharpe ratio: {sharpe:.2f}")

    #Print Sortino Ratio
    sortino = ind.sortino_ratio(data)
    print(f"Sortino ratio: {sortino:.2f}")

    print(f"Hold for {days} days, profit {profit}")
    print(f"Description: {description}")
    def calculate_yearly_performance(data):
        first_day = data.iloc[0]
        last_day = data.iloc[-1]

        pnl_percent = ((last_day['RollingPnL'] - first_day['RollingPnL']) / first_day['RollingPnL']) * 100
        drawdown_percent = data['Drawdown'].max() * 100
        num_trades = data['LongTradeOut'].sum()
        pos_trades = data[(data['TradePnL'] > 0) & (data['LongTradeOut'])].shape[0]

        return pd.Series({'PnL%': f'{pnl_percent:.2f}%', 'Drawdown%': f'{drawdown_percent:.2f}%', 'Num_Trades': num_trades, 'Positive_Trades': pos_trades})

    def annual_performance(data):
        yearly_data = data.groupby(data['Date'].dt.year)
        yearly_performance = yearly_data.apply(calculate_yearly_performance)
        return yearly_performance


    yearly_stats = annual_performance(data)
    print(yearly_stats)

    #Print Treynor Ratio - need benchmark returns
    #treynor = ind.treynor_ratio(data)
    #print(f"Treynor ratio: {treynor:.6f}")

def indicator_tryout(data, days, profit, is_long, is_sell = False, og = False):
    results = bt.backtest_ind(data, days, profit, is_long, 'Hurst', 'both', 0.1, 0.9, 0.1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'Hurst', 'both', 0.1, 0.9, 0.1, og)
    print(results.head(20))
    results = bt.backtest_ind(data, days, profit, is_long, 'Stoch', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'Stoch', 'both', 10, 90, 10, og)
    print(results.head(20))
    results = bt.backtest_ind(data, days, profit, is_long, 'RSI5', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI5', 'both', 10, 90, 10, og)
    print(results.head(20))
    results = bt.backtest_ind(data, days, profit, is_long, 'ER', 'both', 0.1, 0.9, 0.1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'ER', 'both', 0.1, 0.9, 0.1, og)
    print(results.head(20))
    results = bt.backtest_ind(data, days, profit, is_long, 'CCI', 'both', -150, 150, 50, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'CCI', 'both', -150, 150, 50, og)
    print(results.head(20))
    results = bt.backtest_ind(data, days, profit, is_long, 'IBR', 'both', 0.1, 0.9, 0.1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'IBR', 'both', 0.1, 0.9, 0.1, og)
    print(results.head(20))
    results = bt.backtest_ind(data, days, profit, is_long, 'ValueCharts', 'both', -12, 12, 2, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'ValueCharts', 'both', -12, 12, 2, og)
    print(results.head(20))
    results = bt.backtest_ind(data, days, profit, is_long, 'MACDHist', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'MACDHist', 'both', 0, 0, 1, og)
    print(results.head(20))
    results = bt.backtest_ind(data, days, profit, is_long, 'StochOscilator', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'StochOscilator', 'both', 0, 0, 1, og)
    print(results.head(20))

def main():
    start_time = time.perf_counter()
    results = pd.DataFrame()
    #ib = dt.ib_connect()
    """"
    # Create an IB instance
    ib = dt.ib_connect()
    # Define the NQ futures contract
    nq = Future('NQ', '202306', 'CME')
    qqq = Stock('QQQ', 'ARCA')
    """

    #data = dt.get_data_ib(ib, nq, False, period = '1 Y') #True = use RTH data, False = use all data
    if (ticker == 'NQ' or ticker == 'ES' or ticker == 'RTY' or ticker == 'CL' or ticker == 'GC'):
        contract = Future(ticker, '202306', 'CME')
        yfticker = ticker + '=F'
    elif (ticker == 'GBPUSD'):
        #contract = Forex(ticker)
        yfticker = ticker + '=X'
    else:
        contract = Stock(ticker, 'ARCA')
        yfticker = ticker

    data = dt.get_data_yf(yfticker, 20, False) #True for local data, False for Yahoo Finance
    vix_data = dt.get_data_yf('^VIX', 20, False)
    #Add VIX data to dataframe
    data['VIX'] = vix_data['Close']
    data = dt.normalize_dataframe(data) #Capitalize the column names
    data = dt.clean_holidays(data) #Remove holidays
    data = ind.add_indicators(data)

   
    
    data['Buy'], days, profit, description, verdict, is_long, ignore = ind.buy_signal19(data)
    #data['Buy'], days, profit, description, verdict, is_long, ignore = ind.og_buy_signal(data)
    #data['Buy'] = data['Buy'] & (data['ValueCharts']>-12) #& (data['Stoch']<40)
    #data['Sell'] = ind.og_sell_signal(data) | (data['Hurst'] < 0.4)
    #data['Buy'] = data['Buy'] & (data['Stoch']<90)
    #data['Buy'] = data['Buy'] & (data['ER']<0.7)
    #data['Buy'] = data['Buy'] & (data['ValueCharts']>-10)
    #indicator_tryout(data, days, profit, is_long, is_sell = False, og = False)
    #results = bt.backtest_ind(data, days, profit, is_long, 'Hurst', 'both', 0, 1, 0.1, og = True)
    #print (results)
    #results = bt.backtest_sell_ind(data, days, profit, is_long, 'Hurst', 'both', 0, 1, 0.1, False)
    #data['Sell'] = (data['Hurst'] < 0.5)
    results = bt.backtest_days(data, 7, is_long)
    print(results.head(20))
    data = ind.long_strat(data,days,profit, is_long)
    #data = ind.og_strat(data, set_sell = False)
    print_stats(data, days, profit, description)
    data.to_csv('NQ.csv')

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
