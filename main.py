#from ib_insync import IB, Future, util, Stock
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
import streamlit as st

# Print out stats for the strategy
def print_stats(data, days = 0, profit = 0, description = "Original Strategy"):
    number_of_trades = data['LongTradeOut'].value_counts().get(True, 0)
    f"Number of trades: {number_of_trades}"
    # Filter the rows where 'LongTradeOut' is True
    trade_out_rows = data[data['LongTradeOut']]

    # Calculate the number of profitable trades
    num_profitable_trades = (trade_out_rows['TradePnL'] > 0).sum()

    latest_rolling_pnl = ind.format_dollar_value(data['RollingPnL'].iloc[-1])
    f"Latest Rolling PnL: {latest_rolling_pnl}"

    #calculate CAGR
    first_day = data.iloc[0]
    last_day = data.iloc[-1]
    cagr = (last_day['RollingPnL'] / first_day['RollingPnL']) ** (1 / (data.shape[0] / 252)) - 1
    f"CAGR: {cagr:.2%}"

    # Find the maximum drawdown
    max_drawdown = data['Drawdown'].max()
    formatted_drawdown = '{:.2%}'.format(max_drawdown)
    f"Maximum drawdown: {formatted_drawdown}"


    # Calculate the percentage of profitable trades
    percentage_profitable_trades = (num_profitable_trades / number_of_trades) * 100

    f"Percentage of profitable trades: {percentage_profitable_trades:.2f}"

    #Calculate average positive trade PnL
    average_positive_trade_pnl = 100*trade_out_rows[trade_out_rows['TradePnL'] > 0]['TradePnL'].mean()
    f"Average positive trade PnL: {average_positive_trade_pnl:.2f}%"
    #print(f"Average positive trade PnL: {average_positive_trade_pnl:.2f}%")

    #Calculate average negative trade PnL
    average_negative_trade_pnl = 100*trade_out_rows[trade_out_rows['TradePnL'] < 0]['TradePnL'].mean()
    f"Average negative trade PnL: {average_negative_trade_pnl:.2f}%"
    #print(f"Average negative trade PnL: {average_negative_trade_pnl:.2f}%")

    #Print Sharpe Ratio
    sharpe = ind.sharpes_ratio(data)
    f"Sharpe ratio: {sharpe:.2f}"

    #Print Sortino Ratio
    sortino = ind.sortino_ratio(data)
    f"Sortino ratio: {sortino:.2f}"

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

    yearly_stats

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

@st.cache_data
def load_data(years = 1):
    symbols = 'NQ', 'ES', 'CL', 'GC', 'SI', 'SPY', 'SMH', 'QQQ'
    yf_symbols = [symbol+'=F' if symbol in ['NQ', 'ES', 'RTY', 'CL', 'GC', 'SI', 'HG'] else symbol for symbol in symbols]
    #symbol_mapping = {symbol: yf_symbol for symbol, yf_symbol in zip(symbols, yf_symbols)}
    full_data = dt.get_bulk_data(yf_symbols, years)
    return full_data

def main():

    results = pd.DataFrame()
    with st.sidebar:
        smb = st.selectbox('Select symbol', ('NQ', 'ES', 'CL', 'GC', 'SI', 'SPY', 'SMH', 'QQQ'))
        years = st.selectbox('Select years', (1, 5, 10, 15, 20, 25))
        rsi2 = st.number_input('RSI2', 0,100,15)
        rsi5 = st.number_input('RSI5', 0,100,35)
        max_decline = st.number_input('Max Decline', 1, 10, 4)
        stop_loss = st.number_input('Stop Loss, %', 0, 100, 15)
        volatility_threshold = st.number_input('Volatility Threshold, %', 0, 100, 10)
        volume_ema_threshold = st.number_input('Volume EMA Threshold, %', 0, 100, 60)
        monday_buy = st.checkbox('Monday Buy')
        low_volume = st.checkbox('Low Volume Buy')

        #max_decline = st.slider('Max Decline', 1, )
        run_button = st.button('Calculate')
    if (run_button):
        full_data = load_data(years)
        yf_symbol = smb + '=F' if smb in ['NQ', 'ES', 'RTY', 'CL', 'GC', 'SI', 'HG'] else smb
        data= full_data.xs(yf_symbol, axis=1, level=1, drop_level=False)
        data.columns = data.columns.droplevel(1)  # Reset column level
        #data = dt.get_data_yf(yfticker, 20, False) #True for local data, False for Yahoo Finance
        data = dt.normalize_dataframe(data) #Capitalize the column names
        data = dt.clean_holidays(data) #Remove holidays
        data = ind.add_indicators(data) #Add indicators to the table
        #data['Buy'], days, profit, description, verdict, is_long, ignore = ind.buy_signal7(data)
        data['Buy'], days, profit, description, verdict, is_long, ignore = ind.og_buy_signal(data)

        print(results.head(20))
        #data = ind.long_strat(data,days,profit, is_long)
        data = ind.og_strat(data, set_sell = True)
        st.dataframe(data.tail(20), width=2000)
        print_stats(data, days, profit, description)
        #print(data.tail(10))
        #print (data.tail(10))
        #print_stats(data, days, profit, description)
        data.to_csv('NQ.csv')



if __name__ == '__main__':
    main()
