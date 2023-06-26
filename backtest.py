import pandas as pd
import indicators as ind
import numpy as np
from config import *


#Backtest function that iterates over number of days in trade / profitable days in trade
def backtest_days(data, max_days = 10, is_long = True, og = False):
    results = pd.DataFrame()
    for i in range(1, max_days+1):
        for k in range(1, i+1):
            signals = execute_strategy(data, i, k, is_long)
            results.at[i*10+k, 'Days'] = i
            results.at[i*10+k, 'Prf'] = k
            results.at[i*10+k, 'PnL'] = signals['RollingPnL'].iloc[-1]
            #results.at[i*10+k, 'PstvTrades'] = signals['TradePnL'].loc[signals['TradePnL'] > 0].count()
            results.at[i*10+k, 'MaxDD'] = signals['Drawdown'].max()
            #Total number of trades
            results.at[i*10+k, 'Trades'] = signals['LongTradeOut'].value_counts().get(True, 0)
            # % of profitable trades
            results.at[i*10+k, '%Pstv'] = (signals.loc[signals['LongTradeOut'] & (signals['TradePnL'] > 0), 'LongTradeOut'].count() / results.at[i*10+k, 'Trades']) * 100
            results.at[i*10+k, 'Sharpe'] = ind.sharpes_ratio(signals)
            results.at[i*10+k, 'Sortino'] = ind.sortino_ratio(signals)

    
    #sort by Sharpe ratio
    results = results.sort_values(by=['Sharpe'], ascending=False)
    #apply formating to PnL and DD
    results['PnL'] = results['PnL'].astype(int)
    #results['MaxDD'] = results['MaxDD'].astype(int)
    results['PnL'] = results['PnL'].apply(ind.format_dollar_value)
    results['MaxDD'] = results['MaxDD'].round(2)*100#.apply(ind.format_dollar_value)
    # Round the 'Sharpes' column to 2 decimal places
    results['Sharpe'] = results['Sharpe'].round(2)
    results['Sortino'] = results['Sortino'].round(2)
    results['%Pstv'] = results['%Pstv'].round(1)

    # Convert the 'Trades' column to integers
    results['Trades'] = results['Trades'].astype(int)
    results['Days'] = results['Days'].astype(int)
    results['Prf'] = results['Prf'].astype(int)

    return results

#Backtest function that iterates over input indicator and its value
def backtest_ind(data, days_in_trade, profitable_close, is_long, column_name, condition, min_value, max_value, step=0.1, og = False):
    results = pd.DataFrame(columns=['Indicator', 'Condition', 'Value', 'PnL', 'MaxDD', 'Trades', '%Pstv', 'Sharpe', 'Sortino'])
    
    for value in np.arange(min_value, max_value + step, step):
        data_copy = data.copy()
        
        if condition == 'less':
            data_copy['Buy'] = data_copy['Buy'] & (data_copy[column_name] <= value)
        elif condition == 'more':
            data_copy['Buy'] = data_copy['Buy'] & (data_copy[column_name] >= value)
        elif condition == 'both':
            data_copy['Buy'] = data_copy['Buy'] & (data_copy[column_name] >= value)
            data_copy = execute_strategy(data_copy, days_in_trade, profitable_close, is_long)
            #data_copy = ind.og_strat(data_copy, set_sell = False) if og else ind.long_strat(data_copy, days_in_trade, profitable_close, is_long)
            #data_copy = ind.long_strat(data_copy, days_in_trade, profitable_close, is_long)
            #data_copy = ind.og_strat(data_copy)
            rolling_pnl = data_copy['RollingPnL'].iloc[-1]
            max_drawdown = data_copy['Drawdown'].max()*100
            trades_number = data_copy['LongTradeOut'].value_counts().get(True, 0)
            trade_out_rows = data_copy[data_copy['LongTradeOut']]
            positive_trades = (trade_out_rows['TradePnL'] > 0).sum() / trades_number * 100 if trades_number > 0 else 0
            sharpe = ind.sharpes_ratio(data_copy)
            sortino = ind.sortino_ratio(data_copy)
            #calculate CAGR
            cagr = ind.cagr(data_copy)
            results = results._append({'Buysell': 'Buy','Indicator': column_name, 'Condition': 'more', 'Value': value, 'PnL': rolling_pnl, 'MaxDD': max_drawdown, 'Trades': trades_number, '%Pstv': positive_trades, 'CAGR': str(cagr)+'%','Sharpe': sharpe, 'Sortino': sortino}, ignore_index=True)
            data_copy = data.copy()
            data_copy['Buy'] = data_copy['Buy'] & (data_copy[column_name] <= value)    
        
        data_copy = execute_strategy(data_copy, days_in_trade, profitable_close, is_long)
        #data_copy = ind.long_strat(data_copy, days_in_trade, profitable_close, is_long)        
        #data_copy = ind.og_strat(data_copy)
        rolling_pnl = data_copy['RollingPnL'].iloc[-1]
        max_drawdown = data_copy['Drawdown'].max()*100
        trades_number = data_copy['LongTradeOut'].value_counts().get(True, 0)
        trade_out_rows = data_copy[data_copy['LongTradeOut']]
        positive_trades = (trade_out_rows['TradePnL'] > 0).sum() / trades_number * 100 if trades_number > 0 else 0
        sharpe = ind.sharpes_ratio(data_copy)
        sortino = ind.sortino_ratio(data_copy)
        cagr = ind.cagr(data_copy)
        cond = 'less' if condition == 'both' else condition
        
        results = results._append({'Buysell': 'Buy', 'Indicator': column_name, 'Condition': cond, 'Value': value, 'PnL': rolling_pnl, 'MaxDD': max_drawdown, 'Trades': trades_number, '%Pstv': positive_trades, 'CAGR': str(cagr)+'%','Sharpe': sharpe, 'Sortino': sortino}, ignore_index=True)
    
    results = results.sort_values(by=['Sharpe'], ascending=False)
    results['PnL'] = results['PnL'].astype(int)
    results['MaxDD'] = results['MaxDD'].round(2)
    results['PnL'] = results['PnL'].apply(ind.format_dollar_value)
    results['MaxDD'] = results['MaxDD'].astype(str)+'%'
    # Round the 'Sharpes' column to 2 decimal places
    results['Sharpe'] = pd.to_numeric(results['Sharpe'], errors='coerce')
    results['Sharpe'] = results['Sharpe'].round(2) if not (results['Sharpe'].isnull().values.any() or np.isinf(results['Sharpe']).any()) else results['Sharpe']
    #results['Sharpe'] = results['Sharpe'].round(2)
    results['Sortino'] = pd.to_numeric(results['Sortino'], errors='coerce')
    results['Sortino'] = results['Sortino'].round(2) if not (results['Sortino'].isnull().values.any() or np.isinf(results['Sortino']).any()) else results['Sortino']
    results['%Pstv'] = pd.to_numeric(results['%Pstv'], errors='coerce')
    results['%Pstv'] = results['%Pstv'].round(1) if not (results['%Pstv'].isnull().values.any() or np.isinf(results['%Pstv']).any()) else results['%Pstv']

    # Convert the 'Trades' column to integers
    results['Trades'] = results['Trades'].astype(int)
    
    return results

def backtest_sell_ind(data, days_in_trade, profitable_close, is_long, column_name, condition, min_value, max_value, step=0.1, og = False):
    results = pd.DataFrame(columns=['Indicator', 'Condition', 'Value', 'PnL', 'MaxDD', 'Trades', '%Pstv', 'Sharpe', 'Sortino'])
    for value in np.arange(min_value, max_value + step, step):
        data_copy = data.copy()
        
        if condition == 'less':
            data_copy['Sell'] = data_copy['Sell'] | (data_copy[column_name] <= value)
        elif condition == 'more':
            data_copy['Sell'] = data_copy['Sell'] | (data_copy[column_name] >= value)
        elif condition == 'both':
            data_copy['Sell'] = data_copy['Sell'] | (data_copy[column_name] >= value)
            data_copy = execute_strategy(data_copy, days_in_trade, profitable_close, is_long)
            #data_copy = ind.og_strat(data_copy, set_sell = False) if og else ind.long_strat(data_copy, days_in_trade, profitable_close, is_long)
            #data_copy = ind.long_strat(data_copy, days_in_trade, profitable_close, is_long)
            #data_copy = ind.og_strat(data_copy)
            rolling_pnl = data_copy['RollingPnL'].iloc[-1]
            max_drawdown = data_copy['Drawdown'].max()*100
            trades_number = data_copy['LongTradeOut'].value_counts().get(True, 0)
            trade_out_rows = data_copy[data_copy['LongTradeOut']]
            positive_trades = (trade_out_rows['TradePnL'] > 0).sum() / trades_number * 100 if trades_number > 0 else 0
            sharpe = ind.sharpes_ratio(data_copy)
            sortino = ind.sortino_ratio(data_copy)
            #calculate CAGR
            cagr = ind.cagr(data_copy)
            results = results._append({'Buysell': 'Sell', 'Indicator': column_name, 'Condition': 'more', 'Value': value, 'PnL': rolling_pnl, 'MaxDD': max_drawdown, 'Trades': trades_number, '%Pstv': positive_trades, 'CAGR': str(cagr)+'%','Sharpe': sharpe, 'Sortino': sortino}, ignore_index=True)
            data_copy = data.copy()
            data_copy['Sell'] = data_copy['Sell'] | (data_copy[column_name] <= value)    
        
        data_copy = execute_strategy(data_copy, days_in_trade, profitable_close, is_long)
        #data_copy = ind.long_strat(data_copy, days_in_trade, profitable_close, is_long)        
        #data_copy = ind.og_strat(data_copy)
        rolling_pnl = data_copy['RollingPnL'].iloc[-1]
        max_drawdown = data_copy['Drawdown'].max()*100
        trades_number = data_copy['LongTradeOut'].value_counts().get(True, 0)
        trade_out_rows = data_copy[data_copy['LongTradeOut']]
        positive_trades = (trade_out_rows['TradePnL'] > 0).sum() / trades_number * 100 if trades_number > 0 else 0
        sharpe = ind.sharpes_ratio(data_copy)
        sortino = ind.sortino_ratio(data_copy)
        cagr = ind.cagr(data_copy)
        cond = 'less' if condition == 'both' else condition
        
        results = results._append({'Buysell': 'Sell', 'Indicator': column_name, 'Condition': cond, 'Value': value, 'PnL': rolling_pnl, 'MaxDD': max_drawdown, 'Trades': trades_number, '%Pstv': positive_trades, 'CAGR': str(cagr)+'%','Sharpe': sharpe, 'Sortino': sortino}, ignore_index=True)
    
    results = results.sort_values(by=['Sharpe'], ascending=False)
    results['PnL'] = results['PnL'].astype(int)
    results['MaxDD'] = results['MaxDD'].round(2)
    results['PnL'] = results['PnL'].apply(ind.format_dollar_value)
    results['MaxDD'] = results['MaxDD'].astype(str)+'%'
    # Round the 'Sharpes' column to 2 decimal places if sharpe is not NaN
    results['Sharpe'] = pd.to_numeric(results['Sharpe'], errors='coerce')
    results['Sharpe'] = results['Sharpe'].round(2) if not (results['Sharpe'].isnull().values.any() or np.isinf(results['Sharpe']).any()) else results['Sharpe']
    #results['Sharpe'] = results['Sharpe'].round(2)
    results['Sortino'] = pd.to_numeric(results['Sortino'], errors='coerce')
    results['Sortino'] = results['Sortino'].round(2) if not (results['Sortino'].isnull().values.any() or np.isinf(results['Sortino']).any()) else results['Sortino']
    results['%Pstv'] = pd.to_numeric(results['%Pstv'], errors='coerce')
    results['%Pstv'] = results['%Pstv'].round(1) if not (results['%Pstv'].isnull().values.any() or np.isinf(results['%Pstv']).any()) else results['%Pstv']

    # Convert the 'Trades' column to integers
    results['Trades'] = results['Trades'].astype(int)
    
    return results

def execute_strategy (data, days, profit, is_long = True):
    if days == 0:
        results = og_strat(data)
        #print('og_strat'+str(is_long)+str(days)+str(profit))
    else:
        results = long_strat(data, days, profit, is_long)
        #print('long_strat'+str(is_long)+str(days)+str(profit))
    return results

#Original Main Strategy
def og_strat(data, days = 0, profit = 0, external_count = 0, start_capital = 15000):
    # Main Trading signals
    #data['MainBuy'] = False
    #if set_sell: data['Sell'] = False
    data['OneDayBuy'] = False
    data['HoldLong'] = False
    data['LongTradeIn'] = False
    data['LongTradeOut'] = False
    data['DaysInTrade'] = 0
    data['ProfitableCloses'] = 0
    data['RollingPnL'] = 0
    data['TradePnL'] = 0
    data['TradeEntry'] = 0
    ExternalBuy = False
    if external_count > 0:
        for i in range (0, external_count):
            buy_column = 'Buy' + str(i)
            hold_column = 'HoldLong' + str(i)
            ExternalBuy = ExternalBuy | data[buy_column] | data[hold_column]
    else:
        ExternalBuy = True
    """
    if set_sell:
        data['Sell'] = ((data['RSI2'] > RSI2Sell) & (data['RSI5'] > RSI5Sell)) | (                            #RSI2 and RSI5 above threashold
            (data['Close'].shift(1) > data['EMA8'].shift(1)) & (data['Close'] < data['EMA8'])) | (            #Crossing EMA8 down
            (ExitOnVolatility) & ((data['VolumeEMADiff'] >= VolumeEMAThreashold))) | (                        #Volume more than EMAThreashold !!!!!!!!!!!DOESNT WORK - INVESTIGATE!!!!!!!!
            #(data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) < -MaxDecline) | (              #Decline more than 4%
            (data['VolumeEMADiff'] > VolumeEMAThreashold) & (data['Volatility'] > VolatilityThreashold))      #Big volume and volatility  
    """
    # Add one day buy signal
    data['OneDayBuy'] = False
    data['OneDayBuy'] = (((data['Close'] <=  data['Close'].rolling(DownDays).min()) & (data['VolumeEMADiff'] < -VolumeEMAThreasholdBuy) & (LowVolumeBuy))) | ( #Low volume buy
            (data['Close'] < data['Close'].shift(1)) & (pd.to_datetime(data['Date']).dt.dayofweek == 0) & (MondayBuy))                                          #Monday buy
                                                                                                         


    # Calculate days when entering and exiting trades
    #data['OneDayArm'] = False

    for i, row in data.iterrows():
        if i == 0:
            data['HoldLong'].at[i] = False
            data['LongTradeOut'].at[i] = False
            data['TradeEntry'].at[i] = 0
            data['TradePnL'].at[i] = 0
            data['RollingPnL'].at[i] = start_capital
        else:
            data['HoldLong'].at[i] = ((data['HoldLong'].shift(1).at[i] and not data['LongTradeOut'].shift(1).at[i]) or ( #If previously long and not trade out on previous day
                                        data['LongTradeIn'].shift(1).at[i]) or (                                        #Or if Enetering trade on previous day
                                        data['OneDayBuy'].shift(1).at[i]))                                              #Or if one day buy on previous day??? Do we need this?
            data['LongTradeIn'].at[i] = (data['Buy'].at[i] or data['OneDayBuy'].at[i]) and not data['HoldLong'].at[i]
            data['DaysInTrade'].at[i] = data['DaysInTrade'].shift(1).at[i] + 1 if (data['HoldLong'].at[i] and i>0) else 0
            if (data['HoldLong'].at[i]):
                data['ProfitableCloses'].at[i] = data['ProfitableCloses'].shift(1).at[i] + 1 if (data['Close'].at[i] > data['Close'].shift(1).at[i]) else data['ProfitableCloses'].shift(1).at[i]
            data['TradeEntry'].at[i] = data['Close'].at[i] if (((data['LongTradeIn'].at[i] or data['OneDayBuy'].at[i])) and data['TradeEntry'].shift(1).at[i] == 0) else data['TradeEntry'].shift(1).at[i] if data['HoldLong'].at[i] else 0
            data['TradePnL'].at[i] = data['TradePnL'].shift(1).at[i] + data['%Change'].at[i] if data['HoldLong'].at[i] else 0
            #(data['Close'].at[i] - data['TradeEntry'].at[i]) / data['TradeEntry'].at[i] if data['HoldLong'].at[i] else 0
            data['LongTradeOut'].at[i] = (data['Sell'].at[i] and data['HoldLong'].at[i]) or (                           #If sell signal and hold long
                                        data['OneDayBuy'].shift(1).at[i] and not data['HoldLong'].shift(1).at[i] and not data['Buy'].shift(1).at[i]) or ( #Or if one day buy and not hold long on previous day (and not buy signal today)
                                        data['TradePnL'].at[i] < -stop_loss)                                            #Or if hit stoploss 
            if (days > 0):
                data['LongTradeOut'].at[i] = data['LongTradeOut'].at[i] or (data['DaysInTrade'].at[i] >= days) or (data['ProfitableCloses'].at[i] >= profit)
            #data['TradeEntry'].at[i] = data['Close'].at[i] if (data['LongTradeIn'].at[i] or data['OneDayBuy'].at[i]) else data['TradeEntry'].shift(1).at[i] if data['HoldLong'].at[i] else 0


        #Calculate rolling PnL for the strategy
        if i == 0:
            data['RollingPnL'].at[i] = start_capital
        elif data['HoldLong'].at[i]: 
            #signals['RollingPnL'].at[i] = signals['RollingPnL'].shift(1).at[i] + signals['Close'].at[i] - signals['Close'].shift(1).at[i]
            data['RollingPnL'].at[i] = (1+data['%Change'].at[i])*data['RollingPnL'].shift(1).at[i]
        else:
            data['RollingPnL'].at[i] = data['RollingPnL'].shift(1).at[i]   

    # Calculate the running maximum of the 'RollingPnL' column
    data['RunningMax'] = data['RollingPnL'].cummax()

    # Calculate the drawdown as the difference between the running maximum and the current 'RollingPnL' value
    data['Drawdown'] = (data['RunningMax'] - data['RollingPnL'])/data['RunningMax']
    #signals['TradePnL'] = signals['TradePnL'].apply(format_dollar_value)
    #signals['RollingPnL'] = signals['RollingPnL'].apply(format_dollar_value)

    return data

def long_strat(data, days, prof_closes, is_long = True, start_capital = 15000, point_multiplier = point_multiplier):
    signals = data
    signals['LongTradeIn'] = False
    signals['LongTradeOut'] = False
    signals['HoldLong'] = False
    signals['DaysInTrade'] = 0
    signals['ProfitableCloses'] = 0
    signals['RollingPnL'] = 0
    signals['TradePnL'] = 0
    signals['TradeEntry'] = 0
    baddates = pd.DataFrame()
    #signals['TradeInvestment'] = 0

    for i, row in signals.iterrows():
        signals['HoldLong'].at[i] = (signals['HoldLong'].shift(1).at[i] and not signals['LongTradeOut'].shift(1).at[i] and (i>0)) or ( #If previously long and not trade out on previous day
                                    signals['LongTradeIn'].shift(1).at[i] and i>0)                                            #Or if Enetering trade on previous day
        #signals['LongTradeIn'].at[i] = signals['Buy'].at[i] and not signals['HoldLong'].at[i]
        signals['LongTradeIn'].at[i] = signals['Buy'].at[i] and not (signals['HoldLong'].at[i] and not signals['LongTradeOut'].at[i])
        signals['DaysInTrade'].at[i] = signals['DaysInTrade'].shift(1).at[i] + 1 if (signals['HoldLong'].at[i] and i>0) else 0
        #if i>0:
        #    signals['TradeInvestment'].at[i] = signals['RollingPnL'].at[i] if (signals['LongTradeIn'].at[i]) else signals['TradeInvestment'].shift(1).at[i] if signals['HoldLong'].at[i] else 0
        if (signals['HoldLong'].at[i] and i>0):
            if (is_long):
               #signals['ProfitableCloses'].at[i] = (signals['ProfitableCloses'].shift(1).at[i] + 1) if (signals['TradePnL'].at[i] > 0) else signals['ProfitableCloses'].shift(1).at[i]
               signals['ProfitableCloses'].at[i] = signals['ProfitableCloses'].shift(1).at[i] + 1 if (signals['Close'].at[i] > signals['Close'].shift(1).at[i]) else signals['ProfitableCloses'].shift(1).at[i]
            else:
               signals['ProfitableCloses'].at[i] = signals['ProfitableCloses'].shift(1).at[i] + 1 if (signals['Close'].at[i] < signals['Close'].shift(1).at[i]) else signals['ProfitableCloses'].shift(1).at[i] 

        signals['LongTradeOut'].at[i] = ((signals['Sell'].at[i] and signals['HoldLong'].at[i]) or (signals['DaysInTrade'].at[i] >= days) or (signals['ProfitableCloses'].at[i] >= prof_closes))
        signals['TradeEntry'].at[i] = signals['Close'].at[i] if signals['LongTradeIn'].at[i] else signals['TradeEntry'].shift(1).at[i] if signals['HoldLong'].at[i] else 0
        #signals['TradePnL'].at[i] = (signals['TradePnL'].shift(1).at[i] + (signals['Close'].at[i] - signals['Close'].shift(1).at[i])) if (signals['HoldLong'].at[i] and i>0) else 0
        signals['TradePnL'].at[i] = (signals['Close'].at[i] - signals['TradeEntry'].at[i]) / signals['TradeEntry'].at[i] if signals['HoldLong'].at[i] else 0

        #Calculate rolling PnL for the strategy
        if i == 0:
            signals['RollingPnL'].at[i] = start_capital
        elif signals['HoldLong'].at[i]: 
            #signals['RollingPnL'].at[i] = signals['RollingPnL'].shift(1).at[i] + signals['Close'].at[i] - signals['Close'].shift(1).at[i]
            signals['RollingPnL'].at[i] = (1+signals['%Change'].at[i])*signals['RollingPnL'].shift(1).at[i] if is_long else (1-signals['%Change'].at[i])*signals['RollingPnL'].shift(1).at[i]
        else:
            signals['RollingPnL'].at[i] = signals['RollingPnL'].shift(1).at[i]    
        #check if RollingPnL is N/A
        #if pd.isnull(row['RollingPnL']):
        #    baddates = baddates.append(row)


    #signals['RollingPnL'] = signals['RollingPnL']#*point_multiplier
    #signals['TradePnL'] = signals['TradePnL']*point_multiplier
    signals['TradePnL'] = -1*Leverage*signals['TradePnL'] if not is_long else Leverage*signals['TradePnL']
    # Calculate the running maximum of the 'RollingPnL' column
    signals['RunningMax'] = signals['RollingPnL'].cummax()

    # Calculate the drawdown as the difference between the running maximum and the current 'RollingPnL' value
    signals['Drawdown'] = (signals['RunningMax'] - signals['RollingPnL'])/signals['RunningMax']
    #signals['TradePnL'] = signals['TradePnL'].apply(format_dollar_value)
    #signals['RollingPnL'] = signals['RollingPnL'].apply(format_dollar_value)
    baddates.to_csv('baddates.csv')
    return signals