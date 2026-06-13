import warnings
# Suppress FutureWarnings and other non-critical warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*auto_adjust.*')
warnings.filterwarnings('ignore', message='.*T.*is deprecated.*')
warnings.filterwarnings('ignore', message='.*Setting an item of incompatible dtype.*')
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

    #calculate CAGR
    first_day = data.iloc[0]
    last_day = data.iloc[-1]
    cagr = (last_day['RollingPnL'] / first_day['RollingPnL']) ** (1 / (data.shape[0] / 252)) - 1
    print(f"CAGR: {cagr:.2%}")

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
    #Calculate Kelly Criterion
    kelly = (percentage_profitable_trades/100 - ((1 - percentage_profitable_trades/100) / (average_positive_trade_pnl / (-average_negative_trade_pnl))))*100
    print(f"Kelly Criterion: {kelly:.2f}%")
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

def indicator_tryout(data, days, profit, is_long, is_sell = False, check_breadth = True, check_both = True):
    running_results = pd.DataFrame(columns=['Buysell', 'Indicator', 'Condition', 'Value', 'PnL', 'MaxDD', 'Trades', '%Pstv', 'CAGR','Sharpe', 'Sortino'])
    og = True if days == 0 else False
    if check_breadth:
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI2BondBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI2GoldBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI5BondBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI2GoldBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI14BondBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI2GoldBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI14FinancialsBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI14FinancialsBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI14EnergyBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI14EnergyBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI14UtilitiesBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI14UtilitiesBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI14IndustrialsBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI14IndustrialsBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI5FinancialsBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI5FinancialsBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI5EnergyBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI5EnergyBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI5UtilitiesBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI5UtilitiesBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI5IndustrialsBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI5IndustrialsBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI2GoldBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI2GoldBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI5GoldBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI2GoldBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI14GoldBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI2GoldBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        if (data['Date'].dt.year[0] >= 2003):
            results = bt.backtest_ind(data, days, profit, is_long, 'RSI14Breadth', 'both', 20, 80, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI14Breadth', 'both', 20, 80, 10, og)
            print (results.head(5))
            running_results = running_results._append(results.head(3))
            results = bt.backtest_ind(data, days, profit, is_long, 'RSI2Breadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI2Breadth', 'both', 10, 90, 10, og)
            print (results.head(5))
            running_results = running_results._append(results.head(3))
            results = bt.backtest_ind(data, days, profit, is_long, 'RSI5Breadth', 'both', 20, 80, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI5Breadth', 'both', 20, 80, 10, og)
            print (results.head(5))
            running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI14RiskBreadth', 'both', 20, 80, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI14RiskBreadth', 'both', 20, 80, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI5RiskBreadth', 'both', 20, 80, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI5RiskBreadth', 'both', 20, 80, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI2RiskBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI2RiskBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI5SemisBreadth', 'both', 20, 80, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI5SemisBreadth', 'both', 20, 80, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI14SemisBreadth', 'both', 20, 80, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI14SemisBreadth', 'both', 20, 80, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI2SemisBreadth', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI2SemisBreadth', 'both', 10, 90, 10, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
    if check_both or not check_breadth:
        results = bt.backtest_ind(data, days, profit, is_long, 'LowerCloses2', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'LowestClose2', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'LowerCloses3', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'LowestClose3', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'HigherCloses2', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'HighestClose2', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'HigherCloses3', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'HighestClose3', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, '%Change', 'both', -0.06, 0.06, 0.01, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, '%Change', 'both', -0.06, 0.06, 0.01, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))    
        results = bt.backtest_ind(data, days, profit, is_long, 'RSIBuy', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSISell', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'Close_EMA8', 'both', -10, 10, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'Close_EMA8', 'both', -10, 10, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))    
        results = bt.backtest_ind(data, days, profit, is_long, 'EMA8CrossUp', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'EMA8CrossUp', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'EMA8CrossDown', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'EMA8CrossDown', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'ATR20_ATR50', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'ATR20_ATR50', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'ChangeVelocity', 'both', -2, 2, 0.5, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'ChangeVelocity', 'both', -2, 2, 0.5, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'EMA20_EMA100', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'EMA20_EMA100', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'LowestClose2', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'LowestClose2', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'LowestClose3', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'LowestClose3', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'HighestClose2', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'HighestClose2', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'HighestClose3', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'HighestClose3', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))

        

        #"""
        results = bt.backtest_ind(data, days, profit, is_long, 'SMA50_SMA200', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'SMA50_SMA200', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        #"""
        results = bt.backtest_ind(data, days, profit, is_long, 'SMA20_SMA50', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'SMA20_SMA50', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'Close_SMA200', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'Close_SMA200', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'Close_SMA50', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'Close_SMA50', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'Close_SMA20', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'Close_SMA20', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        if (ticker not in ['NQ', 'ES', 'GC', 'SI', 'HG', 'RTY', 'YM', 'CL', 'SOXX', 'FXI']):
            results = bt.backtest_ind(data, days, profit, is_long, 'VFI80', 'both', -8, 8, 2, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'VFI80', 'both', -8, 8, 2, og)
            running_results = running_results._append(results.head(3))
            print (results.head(5))
            results = bt.backtest_ind(data, days, profit, is_long, 'VFI40', 'both', -8, 8, 2, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'VFI40', 'both', -8, 8, 2, og)
            running_results = running_results._append(results.head(3))
            print (results.head(5))
            results = bt.backtest_ind(data, days, profit, is_long, 'VFI10', 'both', -8, 8, 2, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'VFI10', 'both', -8, 8, 2, og)
            running_results = running_results._append(results.head(3))
            print (results.head(5))
            results = bt.backtest_ind(data, days, profit, is_long, 'VFI20', 'both', -8, 8, 2, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'VFI20', 'both', -8, 8, 2, og)
            print (results.head(5))
            running_results = running_results._append(results.head(3))

        results = bt.backtest_ind(data, days, profit, is_long, 'Vix', 'both', 10, 50, 5, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'Vix', 'both', 10, 50, 5, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        #Check if the first date is earlier than 2003

        #results = bt.backtest_ind(data, days, profit, is_long, 'Hurst', 'both', 0.3, 0.7, 0.1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'Hurst', 'both', 0.3, 0.7, 0.1, og)
        #running_results = running_results._append(results.head(3))
        #print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'Stoch', 'both', 10, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'Stoch', 'both', 10, 90, 10, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI14', 'both', 20, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI14', 'both', 20, 90, 10, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI5', 'both', 20, 90, 10, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI5', 'both', 20, 90, 10, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'RSI2', 'both', 10, 50, 5, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'RSI2', 'both', 50, 99, 5, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'ER', 'both', 0.1, 0.9, 0.1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'ER', 'both', 0.1, 0.9, 0.1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'CCI', 'both', -150, 150, 50, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'CCI', 'both', -150, 150, 50, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'IBR', 'both', 0.1, 0.9, 0.1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'IBR', 'both', 0.1, 0.9, 0.1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'IBR2', 'both', 0.1, 0.9, 0.1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'IBR2', 'both', 0.1, 0.9, 0.1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'IBR3', 'both', 0.1, 0.9, 0.1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'IBR3', 'both', 0.1, 0.9, 0.1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'ValueCharts', 'both', -12, 12, 2, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'ValueCharts', 'both', -12, 12, 2, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'MACDHist', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'MACDHist', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))
        results = bt.backtest_ind(data, days, profit, is_long, 'StochOscilator', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'StochOscilator', 'both', 0, 0, 1, og)
        running_results = running_results._append(results.head(3))
        print (results.head(5))

        results = bt.backtest_ind(data, days, profit, is_long, 'SPYBull', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'SPYBull', 'both', 0, 0, 1, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'HigherCloses2', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'HigherCloses2', 'both', 0, 0, 1, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'HigherCloses3', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'HigherCloses3', 'both', 0, 0, 1, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'LowerCloses2', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'LowerCloses2', 'both', 0, 0, 1, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))
        results = bt.backtest_ind(data, days, profit, is_long, 'LowerCloses3', 'both', 0, 0, 1, og) if not is_sell else bt.backtest_sell_ind(data, days, profit, is_long, 'LowerCloses3', 'both', 0, 0, 1, og)
        print (results.head(5))
        running_results = running_results._append(results.head(3))


    running_results = running_results.sort_values(by=['Sharpe'], ascending=False)
    print(running_results)
    return(running_results)
    #print(running_results)


def signal_combination_tryout(signal_a, signal_b, data, symbol=ticker):
    results = bt.backtest_signal_combinations(signal_a, signal_b, data, symbol)
    print(results)
    return results


def symbol_confirmation_tryout(buy_signal, primary_symbol, symbol_pool, years=25, confirm_sets=None):
    results = bt.backtest_symbol_confirmation_sweep(
        buy_signal, primary_symbol, symbol_pool, years=years, confirm_sets=confirm_sets
    )
    print(results)
    return results


def symbol_confirmation_detail(buy_signal, primary_symbol, confirm_symbols=None, years=25, save_csv=True):
    """
    Detailed backtest for one primary + confirm set, including per-year breakdown via print_stats.
    """
    confirm_symbols = confirm_symbols or []
    data, days, profit, description, is_long = bt.backtest_cross_symbol(
        buy_signal, primary_symbol, confirm_symbols, years=years
    )
    print_stats(data, days, profit, description)
    if save_csv:
        confirm_label = '+'.join(confirm_symbols) if confirm_symbols else 'none'
        filename = f'CSV/{primary_symbol}_{buy_signal.__name__}_confirm_{confirm_label}.csv'
        data.to_csv(filename)
        print(f"Saved to {filename}")
    return data


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
    if (ticker == 'NQ' or ticker == 'ES' or ticker == 'RTY' or ticker == 'CL' or ticker == 'GC' or ticker == 'SI' or ticker == 'HG' or ticker == 'NG'):
        contract = Future(ticker, '202306', 'CME')
        yfticker = ticker + '=F'
    elif (ticker == 'GBPUSD'):
        #contract = Forex(ticker)
        yfticker = ticker + '=X'
    else:
        contract = Stock(ticker, 'ARCA')
        yfticker = ticker

    data = dt.get_data_yf(yfticker, years=25, Local = False) #True for local data, False for Yahoo Finance
    data = dt.normalize_dataframe(data) #Capitalize the column names
    data = dt.clean_holidays(data) #Remove holidays
    data = ind.add_indicators(data)

    # buy_signal = ind.buy_signal4
    # buy_signal = ind.combined_signal(ind.buy_signal16, ind.buy_signal7, 'and')  # both must fire
    buy_signal = ind.combined_signal(ind.buy_signal16, ind.buy_signal7, 'or')   # either fires
    # buy_signal = ind.og_new_buy_signal

    # Sweep all 4 primary/secondary AND/OR combos (use combined_signal above for detailed single run)
    # results = signal_combination_tryout(ind.buy_signal7, ind.buy_signal16, data)

    # Cross-symbol confirmation sweep (primary traded at leverage; all confirm symbols must also buy)
    results = symbol_confirmation_tryout(
        ind.combined_signal(ind.buy_signal16, ind.buy_signal7, 'or'),
        primary_symbol='SOXX',
        symbol_pool=['SOXX', 'SMH', 'QQQ', 'SPY'],
        years=25,
    )
    # Custom confirm sets only: confirm_sets=[[], ['SMH'], ['SMH', 'QQQ']]

    # Detailed run for one chosen confirm set (includes per-year breakdown)
    symbol_confirmation_detail(
        ind.combined_signal(ind.buy_signal16, ind.buy_signal7, 'or'),
        primary_symbol='SOXX',
        confirm_symbols=[],
        years=25,
    )
    return  # uncomment to skip single-symbol run below

    data['Buy'], data['Sell'], days, profit, description, verdict, is_long, ignore = buy_signal(data)
    # print(data['BBUpper'])
    # data['Buy'] = data['Buy'] & (data['ValueCharts'] < 0) #& (data['ValueCharts'] < 0) #(data['RSI5SemisBreadth'] > 40) & 
    # data['Buy'] = data['Buy'] & (data['RSI14Breadth']<30) & (data['RSI2']>90) # & (data['Close'] < data['BBUpper'])#data['Close'].shift(1)# & (data['Close'] > data['High'].shift(1)) & (data['RSI2'].shift(1)<15)   #& (data['VFI40'] < 8) &   #& (data['RSI5IndustrialsBreadth']<80)
    # data['Sell'] = data['Sell'] | (data['SMA20'] > data['SMA200']) | (data['RSI14'] < 50)
    
    # results = indicator_tryout(data, days, profit, is_long, is_sell = False, check_breadth=False, check_both=False)    
    #results = results._append(indicator_tryout(data, days, profit, is_long, is_sell = True))
    # results = bt.backtest_ind(data, days, profit, is_long, 'RSI2GoldBreadth', 'both', 0, 100, 10, og=False)
    # print(results.head(20))
    # results = bt.backtest_ind(data, days, profit, is_long, 'RSI5GoldBreadth', 'both', 0, 100, 10, og=False)
    # print(results.head(20))
    # results = bt.backtest_ind(data, days, profit, is_long, 'RSI14GoldBreadth', 'both', 0, 100, 10, og=False)
    # print(results.head(20))


    # results = bt.backtest_days(data, 7, is_long)

    if len(results) >0:
        results = results.sort_values(by=['Sharpe'], ascending=False)
        # results.to_csv('CSV/backtest_results.csv')
        print(results.head(20))
    data = bt.execute_strategy(data, days, profit, is_long)
    # data.to_csv(f'CSV/{ticker}_{buy_signal.__name__}.csv')
    # print(data.head(20))
    print_stats(data, days, profit, description)
    data.to_csv(f'CSV/{ticker}_{buy_signal.__name__}.csv')

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
