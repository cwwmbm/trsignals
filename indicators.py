import math
import numpy as np
import pandas as pd
import ta
from config import *
import mplfinance as mpf
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from hurst import compute_Hc

#Calculate Sharpe Ratio
def sharpes_ratio(data, risk_free_rate=0, periods_per_year=252):
    """
    Calculate the Sharpe ratio of a set of returns.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    risk_free_rate : float
        Constant risk-free return throughout the period.
    periods_per_year : int, optional
        Number of periods per year.

    Returns
    -------
    float
        The Sharpe ratio.

    """
    # Calculate daily returns
    data['DailyReturn'] = data['RollingPnL'].pct_change()
    # Drop NaN values from the 'DailyReturn' column
    data = data.dropna(subset=['DailyReturn'])
    # Calculate the average daily return considering only non-zero RollingPnL values
    average_daily_return = data.loc[data['RollingPnL'] != 0, 'DailyReturn'].mean()

    # Calculate the standard deviation of daily returns considering only non-zero RollingPnL values
    daily_return_std = data.loc[data['RollingPnL'] != 0, 'DailyReturn'].std()

    # Calculate the Sharpe ratio
    sharpe_ratio = math.sqrt(periods_per_year)*(average_daily_return - risk_free_rate) / daily_return_std if daily_return_std != 0 else 0
    return sharpe_ratio

#Calculate Sortino Ratio
def sortino_ratio(data, risk_free_rate=0, periods_per_year=252):
    """
    Calculate the Sortino ratio of a set of returns.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the 'RollingPnL' column.
    risk_free_rate : float, optional
        Constant risk-free return throughout the period.
    periods_per_year : int, optional
        Number of periods per year.

    Returns
    -------
    float
        The Sortino ratio.

    """
    # Calculate daily returns
    data['DailyReturn'] = data['RollingPnL'].pct_change()
    # Drop NaN values from the 'DailyReturn' column
    data = data.dropna(subset=['DailyReturn'])
    # Calculate the average daily return considering only non-zero RollingPnL values
    average_daily_return = data.loc[data['RollingPnL'] != 0, 'DailyReturn'].mean()

    # Calculate the downside (negative) returns
    downside_returns = data.loc[data['RollingPnL'] != 0, 'DailyReturn'].apply(lambda x: min(x, 0))

    # Calculate the downside risk (standard deviation of downside returns)
    downside_risk = np.sqrt((downside_returns**2).mean())

    # Calculate the Sortino ratio
    sortino_ratio = math.sqrt(periods_per_year) * (average_daily_return - risk_free_rate) / downside_risk if downside_risk != 0 else 0
    return sortino_ratio

#Calculate Treynor Ratio - needs benchmark returns to work properly, currently not working
def treynor_ratio(data, benchmark_returns, risk_free_rate=0, periods_per_year=252):
    """
    Calculate the Treynor ratio of a set of returns.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the 'RollingPnL' column.
    benchmark_returns : pandas.Series
        Daily returns of the benchmark.
    risk_free_rate : float, optional
        Constant risk-free return throughout the period.
    periods_per_year : int, optional
        Number of periods per year.

    Returns
    -------
    float
        The Treynor ratio.

    """
    # Calculate daily returns
    data['DailyReturn'] = data['RollingPnL'].pct_change()
    # Drop NaN values from the 'DailyReturn' column
    data = data.dropna(subset=['DailyReturn'])
    
    # Calculate the average daily return considering only non-zero RollingPnL values
    average_daily_return = data.loc[data['RollingPnL'] != 0, 'DailyReturn'].mean()
    
    # Calculate the excess returns
    excess_returns = data['DailyReturn'] - risk_free_rate
    benchmark_excess_returns = benchmark_returns - risk_free_rate
    
    # Calculate the beta of the strategy
    beta = excess_returns.cov(benchmark_excess_returns) / benchmark_excess_returns.var() if benchmark_excess_returns.var() != 0 else 0
    
    # Calculate the Treynor ratio
    treynor_ratio = (average_daily_return - risk_free_rate) / beta if beta != 0 else 0
    return treynor_ratio

#Convert numbers to dollars
def format_dollar_value(value):
    return f"${value:,.2f}"

#Custom CCI calculation
def custom_cci(high, low, close, period=20):
    typical_price = (high + low + close) / 3
    sma_typical_price = typical_price.rolling(window=period).mean()

    # Calculate Mean Deviation
    mean_deviation = typical_price.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())

    cci = (typical_price - sma_typical_price) / (0.015 * mean_deviation)
    return cci

#Calculating CCI
def get_cci(data, period=20):
    #df = util.df(data)
    high = data['High']
    low = data['Low']
    close = data['Close']

    cci = custom_cci(high, low, close, period=period)
    return cci

#Calculating IBR
def internal_bar_ratio(row):
    high = row['High']
    low = row['Low']
    close = row['Close']

    if high == low:
        return 1
    else:
        return (close - low) / (high - low)

def value_charts(data, period=5):
    hl_avg = (data['High'] + data['Low']) / 2
    sma_hl_avg = hl_avg.rolling(window=period).mean()
    
    bar_range = data['High'] - data['Low']
    sma_bar_range = bar_range.rolling(window=period).mean()
    
    value_charts = (data['Close'] - sma_hl_avg) * period / sma_bar_range
    return value_charts

def stochastic_oscillator(data, k_period=14, d_period=3, column_high='High', column_low='Low', column_close='Close'):
    """
    Calculate the Stochastic Oscillator (%K and %D lines) for the given dataframe.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing high, low, and close prices.
    k_period : int, optional
        The lookback period for %K line (default is 14).
    d_period : int, optional
        The lookback period for %D line (default is 3).
    column_high : str, optional
        The column name for the high prices (default is 'High').
    column_low : str, optional
        The column name for the low prices (default is 'Low').
    column_close : str, optional
        The column name for the close prices (default is 'Close').

    Returns
    -------
    data : pandas.DataFrame
        Dataframe with the calculated Stochastic Oscillator (%K and %D) values.
    """

    stoch = ta.momentum.StochasticOscillator(
        data[column_high],
        data[column_low],
        data[column_close],
        n=k_period,
        d_n=d_period
    )

    #data['%K'] = stoch.stoch()
    data['%D'] = stoch.stoch_signal()

    return data

#Calculating signals for TradeIN and TradeOUT.
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
    #signals['TradeInvestment'] = 0

    for i, row in signals.iterrows():
        signals['HoldLong'].at[i] = (signals['HoldLong'].shift(1).at[i] and not signals['LongTradeOut'].shift(1).at[i] and (i>0)) or ( #If previously long and not trade out on previous day
                                    signals['LongTradeIn'].shift(1).at[i] and i>0)                                            #Or if Enetering trade on previous day
        signals['LongTradeIn'].at[i] = signals['Buy'].at[i] and not signals['HoldLong'].at[i]
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


    signals['RollingPnL'] = signals['RollingPnL']#*point_multiplier
    #signals['TradePnL'] = signals['TradePnL']*point_multiplier
    signals['TradePnL'] = -1*Leverage*signals['TradePnL'] if not is_long else Leverage*signals['TradePnL']
    # Calculate the running maximum of the 'RollingPnL' column
    signals['RunningMax'] = signals['RollingPnL'].cummax()

    # Calculate the drawdown as the difference between the running maximum and the current 'RollingPnL' value
    signals['Drawdown'] = (signals['RunningMax'] - signals['RollingPnL'])/signals['RunningMax']
    #signals['TradePnL'] = signals['TradePnL'].apply(format_dollar_value)
    #signals['RollingPnL'] = signals['RollingPnL'].apply(format_dollar_value)

    return signals

#Original Main Strategy
def og_strat(data, days = 0, profit = 0, set_sell = True, external_count = 0, start_capital = 15000):
    # Main Trading signals
    #data['MainBuy'] = False
    if set_sell: data['Sell'] = False
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
    if set_sell:
        data['Sell'] = ((data['RSI2'] > RSI2Sell) & (data['RSI5'] > RSI5Sell)) | (                            #RSI2 and RSI5 above threashold
            (data['Close'].shift(1) > data['EMA8'].shift(1)) & (data['Close'] < data['EMA8'])) | (            #Crossing EMA8 down
            (ExitOnVolatility) & ((data['VolumeEMADiff'] >= VolumeEMAThreashold))) | (                        #Volume more than EMAThreashold !!!!!!!!!!!DOESNT WORK - INVESTIGATE!!!!!!!!
            #(data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) < -MaxDecline) | (              #Decline more than 4%
            (data['VolumeEMADiff'] > VolumeEMAThreashold) & (data['Volatility'] > VolatilityThreashold))      #Big volume and volatility  
    
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
            data['LongTradeIn'].at[i] = data['Buy'].at[i] and not data['HoldLong'].at[i]
            data['DaysInTrade'].at[i] = data['DaysInTrade'].shift(1).at[i] + 1 if (data['HoldLong'].at[i] and i>0) else 0
            if (data['HoldLong'].at[i]):
                data['ProfitableCloses'].at[i] = data['ProfitableCloses'].shift(1).at[i] + 1 if (data['Close'].at[i] > data['Close'].shift(1).at[i]) else data['ProfitableCloses'].shift(1).at[i]
            data['TradeEntry'].at[i] = data['Close'].at[i] if (data['LongTradeIn'].at[i] or data['OneDayBuy'].at[i]) else data['TradeEntry'].shift(1).at[i] if data['HoldLong'].at[i] else 0
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


#Calculating Kaufman Efficiency Ratio
def kaufman_efficiency_ratio(data, period=10, column_close='Close'):
    """
    Calculate the Kaufman Efficiency Ratio for the given dataframe.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing close prices.
    period : int, optional
        The lookback period for the Efficiency Ratio (default is 14).
    column_close : str, optional
        The column name for the close prices (default is 'Close').

    Returns
    -------
    data : pandas.DataFrame
        Dataframe with the calculated Kaufman Efficiency Ratio.
    """

    # Calculate the absolute price change
    data['AbsPriceChange'] = data[column_close].diff(period).abs()

    # Calculate the absolute price change for each bar in the lookback period
    data['AbsBarChange'] = data[column_close].diff().abs()

    # Calculate the cumulative absolute price change for the lookback period
    data['CumAbsBarChange'] = data['AbsBarChange'].rolling(window=period).sum()

    # Calculate the Kaufman Efficiency Ratio
    data['ER'] = data['AbsPriceChange'] / data['CumAbsBarChange']

    # Clean up the temporary columns
    data.drop(columns=['AbsPriceChange', 'AbsBarChange', 'CumAbsBarChange'], inplace=True)

    return data

def hurst_exponent(data):
    H = lambda x: compute_Hc(x)[0]
    window = 200
    hurst = data['Close'].rolling(window).apply(H)
    return hurst

def macd_histogram(data, fast_period=12, slow_period=26, signal_period=9):
    macd_indicator = ta.trend.MACD(data['Close'], window_fast=fast_period, window_slow=slow_period, window_sign=signal_period)
    hist = macd_indicator.macd_diff()
    return hist

def add_indicators(data):
    data['%Change'] = Leverage*data['Close'].pct_change()
    data['Hurst'] = hurst_exponent(data)
    data['IBR'] = data.apply(internal_bar_ratio, axis=1)
    cci = get_cci(data, 20)
    data['CCI'] = cci  # Assign the cci Series to the CCI column in the data DataFrame
    # Calculate SMA(50) and SMA(200)
    data['SMA10'] = data['Close'].rolling(window=10).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA100'] = data['Close'].rolling(window=100).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    data['RSI2'] = ta.momentum.RSIIndicator(data['Close'], window=2).rsi()
    data['RSI5'] = ta.momentum.RSIIndicator(data['Close'], window=5).rsi()
    data['RSI14'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['EMA8'] = ta.trend.ema_indicator(data['Close'], window=8)
    data['EMA20'] = ta.trend.ema_indicator(data['Close'], window=20)
    data['EMA100'] = ta.trend.ema_indicator(data['Close'], window=100)
    data['VolumeEMADiff'] = (data['Volume'] - ta.trend.ema_indicator(data['Volume'], window=8)) / ta.trend.ema_indicator(data['Volume'], window=8)
    data['AdjustedChange'] = data['Close']/data['Close'].shift(1) - 1
    data['Volatility'] = data['AdjustedChange'].rolling(VolatilityPeriod).std() * math.sqrt(252)
    data = kaufman_efficiency_ratio(data)
    data['Stoch'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
    data['StochSlow'] = data['Stoch'].rolling(window=3).mean()
    data['StochOscilator'] = data['Stoch'] - data['StochSlow']
    data['ValueCharts'] = value_charts(data)
    data['MACDHist'] = macd_histogram(data)
    data = data.drop(columns=['StochSlow'])
    #data['StochFast'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3, fastd=True)
    data['Sell'] = False
    return data

def custom_return(buy_signal, days, profit, description, verdict):
    import inspect

    frame = inspect.currentframe().f_back
    assignments = frame.f_code.co_argcount
    return (buy_signal, days, profit, description, verdict) if assignments > 1 else buy_signal

def buy_signal1 (data, symbol = ticker):
    allowed_symbols = []
    ignore = False if symbol in allowed_symbols else True
    days = 5
    profit = 5
    description= 'Long NQ: Open <= High 1 day ago, High 3 days ago <= Open 8 days ago, EMA 100 > EMA 100 2 days ago, IBR < 0.4, ER <= 0.4'
    verdict = 'Bad results after 2020'
    buy = False #(data['Open'] <= data['High'].shift(1)) & (data['High'].shift(3) <= data['Open'].shift(8)) & (data['EMA100'] > data['EMA100'].shift(2))&(data['IBR']<0.4) & (data['ER'] <= 0.4)
    #return custom_return(buy, days, profit, description, verdict)
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore

def buy_signal2 (data, symbol = ticker):
    allowed_symbols = ['NQ', 'QQQ']
    ignore = False if symbol in allowed_symbols else True
    days = 2
    profit = 1
    description = 'Long NQ: Close < Close 2 days ago, Close < Close 10 days ago, IBR < 0.5'
    verdict = 'Good numbers overall'
    buy = (data['Close'].shift(1) < data['Close'].shift(2)) & (data['Close'].pct_change(periods=10).shift(1) < 0) & (data['IBR'] <= 0.5)
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore
    #return custom_return(buy, days, profit, description, verdict)
    #return (data['Close'].shift(1) < data['Close'].shift(2)) & (data['Close'].pct_change(periods=10).shift(1) < 0) & (data['IBR'] <= 0.5) #2/1 for ES, check for NQ, good numbers overall.

def buy_signal3 (data, symbol = ticker):
    allowed_symbols = []
    ignore = False if symbol in allowed_symbols else True

    days = 5
    profit = 5
    description = 'Long NQ: Close 2 days ago < Close 3 days ago, High 1 day ago == High 5 days ago'
    verdict = 'Big drawdown either way for NQ. For ES 6/6 or 6/5 gives some decent numbers.'
    buy = False #(data['Close'].shift(2) < data['Close'].shift(3)) & (data['High'].shift(1) == data['High'].rolling(window=5).max())
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore
    #return (data['Close'].shift(2) < data['Close'].shift(3)) & (data['High'].shift(1) == data['High'].rolling(window=5).max()) #5/5 or 7/5 but big drawdown either way for NQ. For ES 6/6 or 6/5 gives some decent numbers.

def buy_signal4 (data, symbol = ticker):
    allowed_symbols = ['NQ', 'ES']
    ignore = False if symbol in allowed_symbols else True
    days = 2
    profit = 2
    description = 'Long NQ: SMA50 > SMA200, Low = max low last 3 days, IBR < 0.8'
    verdict = 'Suspect for elimination'
    buy = (data['SMA50'] > data['SMA200']) & (data['Low'] == data['Low'].rolling(window=3).max()) & (data['IBR'] <= 0.8)
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore
    #return (data['SMA50'] > data['SMA200']) & (data['Low'] == data['Low'].rolling(window=3).max()) & (data['IBR'] <= 80) #Hold 2 days profit 2

def buy_signal5 (data, symbol = ticker):
    allowed_symbols = ['NQ', 'QQQ']
    ignore = False if symbol in allowed_symbols else True
    days = 2
    profit = 1
    description = 'Long NQ: ER >= 0.5, IBR <= 0.8'
    verdict = 'Low profit but low drawdown.'
    buy = (pd.notna(data['ER'])) & (data['ER'] >= 0.5) & (data['IBR'] <= 0.8)
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore
    #return (pd.notna(data['ER'])) & (data['ER'] >= 0.5) & (data['IBR'] <= 0.8)  #Hold 2 days profit 1

def buy_signal6 (data, symbol = ticker):
    allowed_symbols = ['NQ', 'ES']
    ignore = False if symbol in allowed_symbols else True  
    days = 1
    profit = 1
    description = 'Long NQ: CCI <= -150, IBR <= 0.4'
    verdict = 'Very low drawdowns. Decent results with OG strat'
    buy = (pd.notna(data['CCI'])) & (data['CCI'] <= -150) & (data['IBR'] <= 0.2)
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore
    #return (pd.notna(data['CCI'])) & (data['CCI'] <= -150) & (data['IBR'] <= 0.4) #Hold 4 days proft 1 (could do 3 and 1). Potentiall 9 and 5.

def buy_signal7 (data, symbol = ticker):
    allowed_symbols = ['NQ', 'SMH', 'ES', 'QQQ']
    ignore = False if symbol in allowed_symbols else True
    days = 3
    profit = 1
    description = 'Long NQ: Close 1 day ago <= Close 3 days ago, IBR <= 0.4'
    verdict = '3/3 for NQ, 3/1 for SMH, ES. Monster of a strategy.'
    buy = (data['Close'].shift(1) <= data['Close'].shift(3)) & (data['IBR'] <= 0.4) #& (data['Close'].pct_change(periods=10) < 0)
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore
    #return (data['Close'].shift(1) <= data['Close'].shift(3)) & (data['IBR'] <= 0.4) #& (data['Close'].pct_change(periods=10) < 0) #Hold 3 days profit 1

def buy_signal8(data, symbol = ticker):
    allowed_symbols = ['NQ']
    ignore = False if symbol in allowed_symbols else True
    days = 2
    profit = 1
    description = "Long NQ: ER(10) > 0.50, ValueCharts(5) > -12, RSI2 <= 90, IBR <= 0.8"
    verdict = "2/1 best combo, decent results, low drawdown after 2011."

    kama_er_condition = data['ER'] > 0.50
    value_charts_condition = data['ValueCharts'] > -12
    rsi_condition = data['RSI2'] <= 90

    buy = kama_er_condition & value_charts_condition & rsi_condition & (data['IBR'] <= 0.8)
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore

def buy_signal9(data, symbol = ticker):
    allowed_symbols = ['NQ', 'SMH']
    ignore = False if symbol in allowed_symbols else True
    days = 5
    profit = 2
    description = "Long NQ: Close<= EMA(50), Open[1] = Lowest Open in 3 days, IBR <= 20, RSI14 <= 50"
    verdict = "5/2. Great results. Low drawdown. Some discrepancy with build alpha"

    ema_50 = data['Close'].ewm(span=50).mean()
    lowest_open_3_days = data['Open'].rolling(window=3).min().shift(1)

    #close_condition = data['Close'] < ema_50
    #rsi_condition = 
    open_condition = data['Open'].shift(1) <= lowest_open_3_days
    ibr_condition = data['IBR'] <= 0.2

    buy =  open_condition & ibr_condition & (data['Close'] <= ema_50) & (data['RSI14'] <= 50)#close_condition
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore

def buy_signal10(data, symbol = ticker):
    allowed_symbols = ['NQ', 'SMH', 'ES', 'QQQ']
    ignore = False if symbol in allowed_symbols else True
    days = 3  # You can set the days_to_hold value here
    profit = 1  # You can set the profit target value here
    description = "Long NQ: Low 1 day ago <= Lowest Low in 2 days, IBR <= 50, ValueCharts <= 0"
    verdict = "3/1, great results, investigate overfitting."

    lowest_low_2_days = data['Low'].rolling(window=2).min().shift(1)

    low_condition = data['Low'].shift(1) <= lowest_low_2_days
    ibr_condition = data['IBR'] <= 0.50
    #macd_histogram_condition = data['MACDHist'] <= 0

    buy = low_condition & ibr_condition & (data['ValueCharts'] < 0)  #& macd_histogram_condition
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore
   
def buy_signal11(data, symbol = ticker):
    allowed_symbols = []
    ignore = False if symbol in allowed_symbols else True
    days = 2  # You can set the days_to_hold value here
    profit = 1  # You can set the profit target value here
    description = "Replacebale"
    verdict = "Doesn't work"
    is_long = False

    buy = False#(data['Close'] < data['SMA50']) & (data['Close'] < data['SMA200']) & (data['High'] < data['High'].shift(1)) & (data['High'].shift(1) < data['High'].shift(2)) & (data['RSI2'] > 10)

    return buy, days, profit, description, verdict, is_long, ignore

def buy_signal12(data, symbol = ticker):
    allowed_symbols = ['NQ']
    ignore = False if symbol in allowed_symbols else True
    days = 9  # You can set the days_to_hold value here
    profit = 9  # You can set the profit target value here
    description = "Long NQ: Volume < Volume 2 days ago, ValueCharts < 2, SMA50 > SMA200"
    verdict = "9/9 Very high returns for NQ but not anything else. Investigate overfitting."



    buy = (data['Volume'] < data['Volume'].shift(2)) & (data['ValueCharts'] < 2) & (data['SMA50'] > data['SMA200']) & (data['ER']<0.5)
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore

def buy_signal13(data, symbol = ticker):
    allowed_symbols = ['CL']
    ignore = False if symbol in allowed_symbols else True
    days = 2  # You can set the days_to_hold value here
    profit = 1  # You can set the profit target value here
    description = "Long CL: close[0] > SMA(close,100)[0], rsi(close,2)[0] <= 60, ValueClose(5)[0] > -4, IBR[0] <= 20"
    verdict = "2/1"

    buy = (data['Close'] > data['SMA100']) & (data['RSI2'] <= 60) & (data['ValueCharts'] > -4) & (data['IBR'] <= 0.2)
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore

def buy_signal14(data, symbol = ticker):
    allowed_symbols = ['SMH']
    ignore = False if symbol in allowed_symbols else True
    days = 1
    profit = 1
    description = "Long SMH: low[0] = lowest(low,4)[0], IBR[0] <= 30"
    verdict = "1/1. Very low drawdowns."

    buy = (data['Low'] == data['Low'].rolling(window=4).min()) & (data['IBR'] <= 0.3) & (data['ER'] > 0.3)
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore

def buy_signal15(data, symbol = ticker):
    allowed_symbols = ['CL']
    ignore = False if symbol in allowed_symbols else True
    days = 2
    profit = 1
    description = "Long CL: KaufmanEfficiencyRatio(10)[0] > 20, rsi(close,14)[0] >= 40, IBR[0] <= 20"
    verdict = "2/1"

    buy = (data['ER'] > 0.2) & (data['RSI14'] >= 40) & (data['IBR'] <= 0.2)
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore

def buy_signal16(data, symbol = ticker):
    allowed_symbols = ['SMH']
    ignore = False if symbol in allowed_symbols else True
    days = 3
    profit = 1
    description = "Long SMH: high[0] > close[1], IBR[0] <= 50, Close > SMA100"
    verdict = "2/1"

    buy = (data['High'] > data['Close'].shift(1)) & (data['IBR'] <= 0.5) & (data['Close']>data['SMA100']) #& (data['High'] < data['SMA10']) 
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore

def buy_signal17(data, symbol = ticker):
    allowed_symbols = ['SMH', 'NQ']
    ignore = False if symbol in allowed_symbols else True
    days = 2
    profit = 1
    description = "Long SMH: rsi(close,2)[0] <= 20, IBR[0] <= 30, ER>0.3"
    verdict = "2/1. Knowckout for SMH, ok for NQ"

    buy = (data['RSI2'] <= 20) & (data['IBR'] <= 0.3) & (data['ER'] >= 0.3) #& (data['Stoch'] <= 10)
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore

def buy_signal18(data, symbol = ticker):
    allowed_symbols = ['CL']
    ignore = False if symbol in allowed_symbols else True
    days = 2
    profit = 1
    description = "Long CL: open[1] <= lowest(open,2)[0], IBR[0] <= 20, Stochastics(14)[0] >= 10"
    verdict = "2/1"
    buy = (data['Open'].shift(1) <= data['Open'].rolling(window=2).min()) & (data['IBR'] <= 0.2) & (data['ER'] <= 0.6) & (data['Stoch'] >= 10)
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore

def buy_signal19(data, symbol = ticker):
    allowed_symbols = ['GC']
    ignore = False if symbol in allowed_symbols else True
    days = 1
    profit = 1
    description = "Long GC: Vix[0] <= Vix[1], rsi(close,14)[0] >= 30, IBR[0] <= 50"
    verdict = "1/1"
    buy = (data['Vix'] <= data['Vix'].shift(1)) & (data['RSI14'] >= 30) & (data['IBR'] <= 0.5) & (data['RSI5'] < 60)
    is_long = True
    return buy, days, profit, description, verdict, is_long, ignore


def og_buy_signal(data, symbol = ticker):
    allowed_symbols = ['SPY']
    ignore = False if symbol in allowed_symbols else True
    description = "OG Long Spy strategy"
    verdict = "OG Long Spy strategy"
    is_long = True
    
    buy = ((data['RSI2'] < RSI2Buy) & (data['RSI5'] < RSI5Buy) & (                                 #RSI2 and RSI5 below threashold
        (data['VolumeEMADiff'] < VolumeEMAThreashold) | (data['Volatility'] < VolumeEMAThreashold)) & (  #Volume less than EMAThreashold
        ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) > -MaxDecline)))               #Decline less than 4%    
    return buy, 0, 0, description, verdict, is_long, ignore

def og_sell_signal(data, symbol = ticker):
    allowed_symbols = ['SPY']
    
    sell = ((data['RSI2'] > RSI2Sell) & (data['RSI5'] > RSI5Sell)) | (                                #RSI2 and RSI5 above threashold
            (data['Close'].shift(1) > data['EMA8'].shift(1)) & (data['Close'] < data['EMA8'])) | (            #Crossing EMA8 down
            (ExitOnVolatility) & ((data['VolumeEMADiff'] >= VolumeEMAThreashold))) | (                        #Volume more than EMAThreashold !!!!!!!!!!!DOESNT WORK - INVESTIGATE!!!!!!!!
            #(data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) < -MaxDecline) | (              #Decline more than 4%
            (data['VolumeEMADiff'] > VolumeEMAThreashold) & (data['Volatility'] > VolatilityThreashold))      #Big volume and volatility  

    return sell#, 0, 0, description, verdict, is_long, ignore
