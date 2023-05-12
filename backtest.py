import pandas as pd
import indicators as ind
import numpy as np

#Backtest function that iterates over number of days in trade / profitable days in trade
def backtest_days(data, max_days = 10, is_long = True, og = False):
    results = pd.DataFrame()
    for i in range(1, max_days+1):
        for k in range(1, i+1):
            signals = ind.og_strat(data, days = i, profit = k, set_sell=False) if og else ind.long_strat(data, i, k, is_long)
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
            data_copy = ind.og_strat(data_copy, set_sell = False) if og else ind.long_strat(data_copy, days_in_trade, profitable_close, is_long)
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
            results = results._append({'Indicator': column_name, 'Condition': 'more', 'Value': value, 'PnL': rolling_pnl, 'MaxDD': max_drawdown, 'Trades': trades_number, '%Pstv': positive_trades, 'CAGR': str(cagr)+'%','Sharpe': sharpe, 'Sortino': sortino}, ignore_index=True)
            data_copy = data.copy()
            data_copy['Buy'] = data_copy['Buy'] & (data_copy[column_name] <= value)    
        
        data_copy = ind.og_strat(data_copy, set_sell=False) if og else ind.long_strat(data_copy, days_in_trade, profitable_close, is_long)
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
        
        results = results._append({'Indicator': column_name, 'Condition': cond, 'Value': value, 'PnL': rolling_pnl, 'MaxDD': max_drawdown, 'Trades': trades_number, '%Pstv': positive_trades, 'CAGR': str(cagr)+'%','Sharpe': sharpe, 'Sortino': sortino}, ignore_index=True)
    
    results = results.sort_values(by=['Sharpe'], ascending=False)
    results['PnL'] = results['PnL'].astype(int)
    results['MaxDD'] = results['MaxDD'].round(2)
    results['PnL'] = results['PnL'].apply(ind.format_dollar_value)
    results['MaxDD'] = results['MaxDD'].astype(str)+'%'
    # Round the 'Sharpes' column to 2 decimal places
    results['Sharpe'] = results['Sharpe'].round(2)
    results['Sortino'] = results['Sortino'].round(2)
    results['%Pstv'] = results['%Pstv'].round(1)

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
            data_copy = ind.og_strat(data_copy, set_sell = False) if og else ind.long_strat(data_copy, days_in_trade, profitable_close, is_long)
            #data_copy = ind.long_strat(data_copy, days_in_trade, profitable_close, is_long)
            #data_copy = ind.og_strat(data_copy)
            rolling_pnl = data_copy['RollingPnL'].iloc[-1]
            max_drawdown = data_copy['Drawdown'].max()*100
            trades_number = data_copy['LongTradeOut'].value_counts().get(True, 0)
            trade_out_rows = data_copy[data_copy['LongTradeOut']]
            positive_trades = (trade_out_rows['TradePnL'] > 0).sum() / trades_number * 100 if trades_number > 0 else 0
            sharpe = ind.sharpes_ratio(data_copy)
            sortino = ind.sortino_ratio(data_copy)
            results = results._append({'Indicator': column_name, 'Condition': 'more', 'Value': value, 'PnL': rolling_pnl, 'MaxDD': max_drawdown, 'Trades': trades_number, '%Pstv': positive_trades, 'Sharpe': sharpe, 'Sortino': sortino}, ignore_index=True)
            data_copy = data.copy()
            data_copy['Sell'] = data_copy['Sell'] | (data_copy[column_name] <= value)    
        
        data_copy = ind.og_strat(data_copy, set_sell = False) if og else ind.long_strat(data_copy, days_in_trade, profitable_close, is_long)
        #data_copy = ind.long_strat(data_copy, days_in_trade, profitable_close, is_long)        
        #data_copy = ind.og_strat(data_copy)
        rolling_pnl = data_copy['RollingPnL'].iloc[-1]
        max_drawdown = data_copy['Drawdown'].max()*100
        trades_number = data_copy['LongTradeOut'].value_counts().get(True, 0)
        trade_out_rows = data_copy[data_copy['LongTradeOut']]
        positive_trades = (trade_out_rows['TradePnL'] > 0).sum() / trades_number * 100 if trades_number > 0 else 0
        sharpe = ind.sharpes_ratio(data_copy)
        sortino = ind.sortino_ratio(data_copy)
        cond = 'less' if condition == 'both' else condition
        
        results = results._append({'Indicator': column_name, 'Condition': cond, 'Value': value, 'PnL': rolling_pnl, 'MaxDD': max_drawdown, 'Trades': trades_number, '%Pstv': positive_trades, 'Sharpe': sharpe, 'Sortino': sortino}, ignore_index=True)
    
    results = results.sort_values(by=['Sharpe'], ascending=False)
    results['PnL'] = results['PnL'].astype(int)
    results['MaxDD'] = results['MaxDD'].round(2)
    results['PnL'] = results['PnL'].apply(ind.format_dollar_value)
    results['MaxDD'] = results['MaxDD'].astype(str)+'%'
    # Round the 'Sharpes' column to 2 decimal places
    results['Sharpe'] = results['Sharpe'].round(2)
    results['Sortino'] = results['Sortino'].round(2)
    results['%Pstv'] = results['%Pstv'].round(1)

    # Convert the 'Trades' column to integers
    results['Trades'] = results['Trades'].astype(int)
    
    return results
