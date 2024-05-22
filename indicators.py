import math
import numpy as np
import pandas as pd
import ta
from config import *
# import mplfinance as mpf
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

#Calculate CAGR
def cagr(data, periods_per_year=252):
    """
    Calculate the CAGR of a set of returns.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the 'RollingPnL' column.
    periods_per_year : int, optional
        Number of periods per year.

    Returns
    -------
    float
        The CAGR.

    """
    first_day = data.iloc[0]
    last_day = data.iloc[-1]
    cagr = (last_day['RollingPnL'] / first_day['RollingPnL']) ** (1 / (data.shape[0] / 252)) - 1
    #round to 2 decimal places
    cagr = round(cagr*100, 2)
    return cagr

#Calculate Kelly Criterion - does't work properly
def kelly_criterion(data):
    """
    Calculate the Kelly Criterion of a set of returns.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the 'RollingPnL' column.

    Returns
    -------
    float
        The Kelly Criterion.

    """
    # Calculate daily returns
    data['DailyReturn'] = data['RollingPnL'].pct_change()
    # Drop NaN values from the 'DailyReturn' column
    data = data.dropna(subset=['DailyReturn'])
    # Calculate the average daily return considering only non-zero RollingPnL values
    average_daily_return = data.loc[data['RollingPnL'] != 0, 'DailyReturn'].mean()
    # Calculate the standard deviation of daily returns considering only non-zero RollingPnL values
    daily_return_std = data.loc[data['RollingPnL'] != 0, 'DailyReturn'].std()
    # Calculate the Kelly Criterion
    kelly_criterion = average_daily_return / daily_return_std**2
    return kelly_criterion

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

#calculate stoch_rsi
def stoch_rsi(data):
    stoch_rsi = ta.momentum.StochRSIIndicator(data['Close']).stochrsi()
    return stoch_rsi


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
    window = 100
    hurst = data['Close'].rolling(window).apply(H)
    return hurst

def vfi(df, period=40, coef=0.2, vcoef=2.5):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    inter = np.log(typical_price) - np.log(typical_price.shift(1))
    inter.fillna(0, inplace=True)

    vinter = inter.rolling(window=30).std()

    cutoff = coef * vinter * df['Close']
    
    vave = df['Volume'].rolling(window=period).mean()

    vmax = vave * vcoef

    vc = df['Volume'].where(df['Volume'] < vmax, vmax)
    
    mf = typical_price - typical_price.shift(1)

    vc = np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0))
    
    vfii_temp = pd.Series(vc).rolling(window=period).sum()
    vfii = vfii_temp / vave
    
    vfimov = vfii.ewm(span=3).mean()
    
    return vfimov

def macd_histogram(data, fast_period=12, slow_period=26, signal_period=9):
    macd_indicator = ta.trend.MACD(data['Close'], window_fast=fast_period, window_slow=slow_period, window_sign=signal_period)
    hist = macd_indicator.macd_diff()
    return hist

def add_indicators(data):
    data['%Change'] = Leverage*data['Close'].pct_change()
    data['SPYBull'] = data['Spybull']
    data['Hurst'] = hurst_exponent(data)
    data['IBR'] = data.apply(internal_bar_ratio, axis=1)
    data['IBR2'] = data['IBR'].rolling(window=2).mean()
    data['IBR3'] = data['IBR'].rolling(window=3).mean()
    cci = get_cci(data, 20)
    data['CCI'] = cci  # Assign the cci Series to the CCI column in the data DataFrame
    # Calculate SMA(50) and SMA(200)
    data['SMA10'] = data['Close'].rolling(window=10).mean()
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA100'] = data['Close'].rolling(window=100).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    data['RSI2'] = ta.momentum.RSIIndicator(data['Close'], window=2).rsi()
    data['RSI5'] = ta.momentum.RSIIndicator(data['Close'], window=5).rsi()
    data['RSI14'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['RSIBuy'] = np.where((data['RSI2'] <= 15) & (data['RSI5'] <= 35), 1, -1)
    data['RSISell'] = np.where((data['RSI2'] >= 95) & (data['RSI5'] >= 70), 1, -1)
    data['RSI2Breadth'] = ta.momentum.RSIIndicator(data['Breadth'], window=2).rsi()
    data['RSI5Breadth'] = ta.momentum.RSIIndicator(data['Breadth'], window=5).rsi()
    data['RSI14Breadth'] = ta.momentum.RSIIndicator(data['Breadth'], window=14).rsi()
    data['RSI2RiskBreadth'] = ta.momentum.RSIIndicator(data['Riskbreadth'], window=2).rsi()
    data['RSI5RiskBreadth'] = ta.momentum.RSIIndicator(data['Riskbreadth'], window=5).rsi()
    data['RSI14RiskBreadth'] = ta.momentum.RSIIndicator(data['Riskbreadth'], window=14).rsi()
    data['RSI2SemisBreadth'] = ta.momentum.RSIIndicator(data['Semisbreadth'], window=2).rsi()
    data['RSI5SemisBreadth'] = ta.momentum.RSIIndicator(data['Semisbreadth'], window=5).rsi()
    data['RSI14SemisBreadth'] = ta.momentum.RSIIndicator(data['Semisbreadth'], window=14).rsi()
    data['RSI14FinancialsBreadth'] = ta.momentum.RSIIndicator(data['Financialsbreadth'], window=14).rsi()
    data['RSI14EnergyBreadth'] = ta.momentum.RSIIndicator(data['Energybreadth'], window=14).rsi()
    data['RSI14UtilitiesBreadth'] = ta.momentum.RSIIndicator(data['Utilitiesbreadth'], window=14).rsi()
    data['RSI14IndustrialsBreadth'] = ta.momentum.RSIIndicator(data['Industrialsbreadth'], window=14).rsi()
    data['RSI5FinancialsBreadth'] = ta.momentum.RSIIndicator(data['Financialsbreadth'], window=5).rsi()
    data['RSI5EnergyBreadth'] = ta.momentum.RSIIndicator(data['Energybreadth'], window=5).rsi()
    data['RSI5UtilitiesBreadth'] = ta.momentum.RSIIndicator(data['Utilitiesbreadth'], window=5).rsi()
    data['RSI5IndustrialsBreadth'] = ta.momentum.RSIIndicator(data['Industrialsbreadth'], window=5).rsi()
    data['RSI2GoldBreadth'] = ta.momentum.RSIIndicator(data['Goldbreadth'], window=2).rsi()
    data['RSI5GoldBreadth'] = ta.momentum.RSIIndicator(data['Goldbreadth'], window=5).rsi()
    data['RSI14GoldBreadth'] = ta.momentum.RSIIndicator(data['Goldbreadth'], window=14).rsi()
    data['EMA8'] = ta.trend.ema_indicator(data['Close'], window=8)
    data['EMA8CrossUp'] = np.where((data['Close'] > data['EMA8']) & (data['Close'].shift(1) < data['EMA8'].shift(1)), 1, -1)
    data['EMA8CrossDown'] = np.where((data['Close'] < data['EMA8']) & (data['Close'].shift(1) > data['EMA8'].shift(1)), 1, -1)
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
    data['VFI40'] = vfi(data)
    data['VFI20'] = vfi(data, period=20)
    data['VFI80'] = vfi(data, period=80)
    data['VFI10'] = vfi(data, period=10)
    data['Close_EMA8'] = (data['Close'] - data['EMA8'])/data['Close']*100
    data['EMA20_EMA100'] = (data['EMA20'] - data['EMA100'])/data['EMA20']*100
    data['SMA50_SMA200'] = (data['SMA50'] - data['SMA200'])/data['SMA50']*100
    data['SMA20_SMA50'] = (data['SMA20'] - data['SMA50'])/data['SMA20']*100
    data['Close_SMA200'] = (data['Close'] - data['SMA200'])/data['Close']*100
    data['Close_SMA50'] = (data['Close'] - data['SMA50'])/data['Close']*100
    data['Close_SMA20'] = (data['Close'] - data['SMA20'])/data['Close']*100
    data['LowestClose2'] = np.where(data['Close'] <= data['Close'].shift(1), 1, -1)
    data['LowestClose3'] = np.where(data['Close'] <= data['Close'].rolling(window=3).min(), 1, -1)
    data['HighestClose2'] = np.where(data['Close'] >= data['Close'].shift(1), 1, -1)
    data['HighestClose3'] = np.where(data['Close'] >= data['Close'].rolling(window=3).max(), 1, -1)
    data['ATR20'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=20)
    data['ATR50'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=50)
    data['ATR20_ATR50'] = data['ATR20'] - data['ATR50']
    data['ChangeVelocity'] = (data['Close'] - data['Close'].shift(1)) / data['ATR20'].shift(1)
    data['HigherCloses2'] = np.where((data['Close'] > data['Close'].shift(1)) & (data['Close'].shift(1) > data['Close'].shift(2)), 1, -1)
    data['HigherCloses3'] = np.where((data['Close'] > data['Close'].shift(1)) & (data['Close'].shift(1) > data['Close'].shift(2)) & (data['Close'].shift(2) > data['Close'].shift(3)), 1, -1)
    data['LowerCloses2'] = np.where((data['Close'] < data['Close'].shift(1)) & (data['Close'].shift(1) < data['Close'].shift(2)), 1, -1)
    data['LowerCloses3'] = np.where((data['Close'] < data['Close'].shift(1)) & (data['Close'].shift(1) < data['Close'].shift(2)) & (data['Close'].shift(2) < data['Close'].shift(3)), 1, -1)
    data = data.drop(columns=['StochSlow'])
    data = data.drop(columns=['Spybull'])
    #data['StochFast'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3, fastd=True)
    data['Sell'] = False
    return data

def buy_signal1 (data, symbol = ticker):
    allowed_symbols = ['IBB']
    ignore = False if symbol in allowed_symbols else True
    days = 2
    profit = 1
    description= 'Long IBB: RSI5EnergyBreadth < 70, Close_EMA8 < 0, IBR < 0.4'
    verdict = ''
    buy = (data['RSI5EnergyBreadth']<70) & (data['Close_EMA8']<0) & (data['IBR']<0.4) #& (data['RSI2SemisBreadth']>10) #True #(data['Open'] <= data['High'].shift(1)) & (data['High'].shift(3) <= data['Open'].shift(8)) & (data['EMA100'] > data['EMA100'].shift(2))&(data['IBR']<0.4) & (data['ER'] <= 0.4)
    #return custom_return(buy, days, profit, description, verdict)
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal2 (data, symbol = ticker):
    allowed_symbols = ['SPY', 'QQQ']
    ignore = False if symbol in allowed_symbols else True
    days = 2
    profit = 1
    description = 'Long NQ: Close < Close 2 days ago, Close < Close 10 days ago, IBR < 0.5'
    verdict = 'Good numbers overall'
    buy = (data['Close'].shift(1) < data['Close'].shift(2)) & (data['Close'].pct_change(periods=10).shift(1) < 0) & (data['IBR'] <= 0.5)
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore
    #return custom_return(buy, days, profit, description, verdict)
    #return (data['Close'].shift(1) < data['Close'].shift(2)) & (data['Close'].pct_change(periods=10).shift(1) < 0) & (data['IBR'] <= 0.5) #2/1 for ES, check for NQ, good numbers overall.

def buy_signal3 (data, symbol = ticker):
    allowed_symbols = ['TLT']
    ignore = False if symbol in allowed_symbols else True

    days = 2
    profit = 1
    description = 'Long NQ: Close 2 days ago < Close 3 days ago, High 1 day ago == High 5 days ago'
    verdict = 'Big drawdown either way for NQ. For ES 6/6 or 6/5 gives some decent numbers.'
    buy = (data['RSI5EnergyBreadth']<60) & (data['IBR3'] < 0.7) & (data['Vix'] < 25) & (data['VFI10'] > 0)
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore
    #return (data['Close'].shift(2) < data['Close'].shift(3)) & (data['High'].shift(1) == data['High'].rolling(window=5).max()) #5/5 or 7/5 but big drawdown either way for NQ. For ES 6/6 or 6/5 gives some decent numbers.

def buy_signal4 (data, symbol = ticker):
    allowed_symbols = ['FXI']
    ignore = False if symbol in allowed_symbols else True
    days = 3
    profit = 1
    description = 'New strat for FXI: RSI2GoldBreadth > 50, Stoch < 90, RSI14SemisBreadth > 40'
    verdict = 'Seems good'
    buy = (data['RSI2GoldBreadth']>50) & (data['Stoch'] < 90) & (data['RSI14SemisBreadth'] > 40)
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore
   #return (data['SMA50'] > data['SMA200']) & (data['Low'] == data['Low'].rolling(window=3).max()) & (data['IBR'] <= 80) #Hold 2 days profit 2

def buy_signal5 (data, symbol = ticker):
    allowed_symbols = ['QQQ', 'SPY']
    ignore = False if symbol in allowed_symbols else True
    days = 2
    profit = 1
    description = 'Long NQ: ER >= 0.5, IBR <= 0.8'
    verdict = 'Low profit but low drawdown.'
    buy = (pd.notna(data['ER'])) & (data['ER'] >= 0.5) & (data['IBR'] <= 0.8) & (data['RSI14SemisBreadth'] >30)
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore
   #return (pd.notna(data['ER'])) & (data['ER'] >= 0.5) & (data['IBR'] <= 0.8)  #Hold 2 days profit 1

def buy_signal6 (data, symbol = ticker):
    allowed_symbols = ['NQ', 'ES']
    ignore = False if symbol in allowed_symbols else True  
    days = 4
    profit = 1
    description = 'Long NQ: CCI <= -150, IBR <= 0.4'
    verdict = 'Very low drawdowns. Decent results with OG strat'
    buy = (pd.notna(data['CCI'])) & (data['CCI'] <= -150) & (data['IBR'] <= 0.2)
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore
   #return (pd.notna(data['CCI'])) & (data['CCI'] <= -150) & (data['IBR'] <= 0.4) #Hold 4 days proft 1 (could do 3 and 1). Potentiall 9 and 5.

def buy_signal7 (data, symbol = ticker):
    allowed_symbols = ['NQ', 'SMH', 'ES', 'QQQ', 'FXI','AAPL', 'SOXX', 'MSFT']
    ignore = False if symbol in allowed_symbols else True
    days = 3
    profit = 1
    description = 'Long NQ: Close 1 day ago <= Close 3 days ago, IBR <= 0.4'
    verdict = '3/3 for NQ, 3/1 for SMH, ES. Monster of a strategy.'
    buy = (data['Close'].shift(1) <= data['Close'].shift(3)) & (data['IBR'] <= 0.4) #& (data['Close'].pct_change(periods=10) < 0)
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore
   #return (data['Close'].shift(1) <= data['Close'].shift(3)) & (data['IBR'] <= 0.4) #& (data['Close'].pct_change(periods=10) < 0) #Hold 3 days profit 1

def buy_signal8(data, symbol = ticker):
    allowed_symbols = ['QQQ', 'SPY']
    ignore = False if symbol in allowed_symbols else True
    days = 3
    profit = 1
    description = "Long NQ: ER(10) > 0.50, ValueCharts(5) > -12, RSI2 <= 90, IBR <= 0.8, RSI5Breadth < 80"
    verdict = "Good results for both NQ and ES. Low drawdown post 2008."

    kama_er_condition = data['ER'] > 0.50
    value_charts_condition = data['ValueCharts'] > -12
    rsi_condition = data['RSI2'] <= 90

    buy = kama_er_condition & value_charts_condition & rsi_condition & (data['IBR'] <= 0.8) #& (data['RSI5Breadth'] < 80)
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal9(data, symbol = ticker):
    allowed_symbols = ['SPY', 'QQQ']
    ignore = False if symbol in allowed_symbols else True
    days = 100
    profit = 100
    description = "Stoch < 30, MACD < 0, IBR <= 0.2"
    verdict = "Great results. Low drawdown."

    buy = (data['Stoch']<30) & (data['MACDHist']< 0) &(data['IBR'] <= 0.2) #& open_condition #& (data['Close'] <= ema_50) #& (data['RSI14'] <= 50)#close_condition
    is_long = True
    sell = (data['RSI14IndustrialsBreadth'] < 40) | (data['Stoch'] > 20)#False
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal10(data, symbol = ticker):
    allowed_symbols = ['SMH']
    ignore = False if symbol in allowed_symbols else True
    days = 3  # You can set the days_to_hold value here
    profit = 1  # You can set the profit target value here
    description = "Long NQ: Low 1 day ago <= Lowest Low in 2 days, IBR <= 50"
    verdict = "3/1, great results, investigate overfitting."

    lowest_low_2_days = data['Low'].rolling(window=2).min().shift(1)

    low_condition = data['Low'].shift(1) <= lowest_low_2_days
    ibr_condition = data['IBR'] <= 0.50
    #macd_histogram_condition = data['MACDHist'] <= 0

    buy = low_condition & ibr_condition #& (data['ValueCharts'] < 0)  #& macd_histogram_condition
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal11(data, symbol = ticker):
    allowed_symbols = []
    ignore = False if symbol in allowed_symbols else True
    days = 2  # You can set the days_to_hold value here
    profit = 1  # You can set the profit target value here
    description = "Replacebale"
    verdict = "Doesn't work"
    is_long = False

    buy = False#(data['Close'] < data['SMA50']) & (data['Close'] < data['SMA200']) & (data['High'] < data['High'].shift(1)) & (data['High'].shift(1) < data['High'].shift(2)) & (data['RSI2'] > 10)

    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal12(data, symbol = ticker):
    allowed_symbols = []
    ignore = False if symbol in allowed_symbols else True
    days = 9  # You can set the days_to_hold value here
    profit = 9  # You can set the profit target value here
    description = "Long NQ: Volume < Volume 2 days ago, ValueCharts < 2, SMA50 > SMA200"
    verdict = "9/9 Very high returns for NQ but not anything else. Investigate overfitting."



    buy = False#(data['Volume'] < data['Volume'].shift(2)) & (data['ValueCharts'] < 2) & (data['SMA50'] > data['SMA200']) & (data['ER']<0.5)
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal13(data, symbol = ticker):
    allowed_symbols = ['CL']
    ignore = False if symbol in allowed_symbols else True
    days = 2  # You can set the days_to_hold value here
    profit = 1  # You can set the profit target value here
    description = "Long CL: close[0] > SMA(close,100)[0], rsi(close,2)[0] <= 60, ValueClose(5)[0] > -4, IBR[0] <= 20"
    verdict = "2/1"

    buy = (data['Close'] > data['SMA100']) & (data['RSI2'] <= 60) & (data['ValueCharts'] > -4) & (data['IBR'] <= 0.2)
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal14(data, symbol = ticker):
    allowed_symbols = ['SMH']
    ignore = False if symbol in allowed_symbols else True
    days = 1
    profit = 1
    description = "Long SMH: low[0] = lowest(low,4)[0], IBR[0] <= 30"
    verdict = "1/1. Very low drawdowns."

    buy = (data['Low'] == data['Low'].rolling(window=4).min()) & (data['IBR'] <= 0.3) & (data['ER'] > 0.3)
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal15(data, symbol = ticker):
    allowed_symbols = ['CL']
    ignore = False if symbol in allowed_symbols else True
    days = 2
    profit = 1
    description = "Long CL: KaufmanEfficiencyRatio(10)[0] > 20, rsi(close,14)[0] >= 40, IBR[0] <= 20"
    verdict = "2/1"

    buy = (data['ER'] > 0.2) & (data['RSI14'] >= 40) & (data['IBR'] <= 0.2)
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal16(data, symbol = ticker):
    allowed_symbols = ['SMH', 'QQQ', 'ORCL', 'SOXX', 'MSFT', 'TECL']
    ignore = False if symbol in allowed_symbols else True
    days = 4
    profit = 1
    description = "high[0] > close[1], IBR[0] <= 50, SMA50>SMA200, Hurst > 0.4"
    verdict = "Great results for SMH but not for SOXX. Suspect."

    buy = (data['High'] > data['Close'].shift(1)) & (data['IBR'] <= 0.5) #& (data['SMA50_SMA200']>0) #& (data['Hurst'] > 0.4)#& (data['Close']>data['SMA100']) #& (data['High'] < data['SMA10']) 
    is_long = True
    sell = False#(data['VFI40'] < -2) | (data['RSI14RiskBreadth'] > 70)
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal17(data, symbol = ticker):
    allowed_symbols = ['SMH', 'NQ']
    ignore = False if symbol in allowed_symbols else True
    days = 2
    profit = 1
    description = "Long SMH: rsi(close,2)[0] <= 20, IBR[0] <= 30, ER>0.3"
    verdict = "2/1. Knowckout for SMH, ok for NQ"

    buy = (data['RSI2'] <= 20) & (data['IBR'] <= 0.3) & (data['ER'] >= 0.3) #& (data['Stoch'] <= 10)
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal18(data, symbol = ticker):
    allowed_symbols = ['GDX', 'INDA']
    ignore = False if symbol in allowed_symbols else True
    days = 2
    profit = 1
    description = "Long CL: open[1] <= lowest(open,2)[0], IBR[0] <= 20, Stochastics(14)[0] >= 10"
    verdict = "2/1"
    buy = (data['IBR'] <= 0.2) & (data['ER'] <= 0.6) & (data['Stoch'] >= 10) & (data['RSI14EnergyBreadth']<60) & (data['RSI14EnergyBreadth']>30)# & (data['Open'].shift(1) <= data['Open'].rolling(window=2).min()) & 
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal19(data, symbol = ticker):
    allowed_symbols = ['GC', 'SI']
    ignore = False if symbol in allowed_symbols else True
    days = 1
    profit = 1
    description = "Long GC: Vix[0] <= Vix[1], rsi(close,14)[0] >= 30, IBR[0] <= 50"
    verdict = "1/1"
    buy = (data['Vix'] <= data['Vix'].shift(1)) & (data['RSI14'] >= 30) & (data['IBR'] <= 0.5) & (data['RSI5'] < 90)
    is_long = True
    sell = False
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal20(data, symbol = ticker):
    allowed_symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'SOXX', 'CSCO', 'MSFT', 'FNGU']
    ignore = False if symbol in allowed_symbols else True
    days = 50

    profit = 50
    description = "Experimental Long signal"
    verdict = ""
    buy = (data['RSI2'] < 40) & (data['IBR'] < 0.2) & (data['RSI5Breadth'] < 60)# &(data['RSI2SemisBreadth']>20) #& (data['RSI5Breadth'] < 60) &(data['RSI2SemisBreadth']>20) & (data['Hurst'] > 0.4)#& (data['RSI5Breadth'] < 60)#(data['IBR'] <= 0.2) & (data['CCI'] < 100) #& (data['ValueCharts'] < 0)
    sell = ((data['RSI2SemisBreadth'].shift(1) > 50) &
            (data['RSI2SemisBreadth'] < 50)) | (data['Vix'] > 40) | (data['ChangeVelocity'] > 1) | (data['RSI2SemisBreadth'] < 10)
    is_long = True
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal21(data, symbol = ticker):
    allowed_symbols = ['SPY']
    ignore = False if symbol in allowed_symbols else True
    days = 1

    profit = 1
    description = "Close<200SMA, RSI2<40, RSI2SemisBreadth>30. Bear market long signal"
    verdict = "1/1"
    buy = (data['SMA50_SMA200'] < 0) & (data['RSI2'] < 40) & (data['RSI2SemisBreadth'] > 30)
    sell = False
    is_long = True
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal22(data, symbol = ticker): #Just some tests, not a real signal
    allowed_symbols = ['SPY', 'QQQ', 'ES', 'NQ']
    ignore = False if symbol in allowed_symbols else True
    days = 5

    profit = 5
    description = "Experimental Long signal"
    verdict = ""
    buy = (data['RSI2RiskBreadth'] > 95) & (data['RSI5RiskBreadth'] > 85)
    sell = False#((data['RSI2SemisBreadth'].shift(1) > 50) & (data['RSI2SemisBreadth'] < 50)) | (data['Vix'] > 40)
    is_long = True
    return buy, sell, days, profit, description, verdict, is_long, ignore    

def buy_signal23(data, symbol = ticker):
    allowed_symbols = ['FXI']
    ignore = False if symbol in allowed_symbols else True
    days = 2

    profit = 1
    description = "low[0] >= highest(low,2)[0], close[1] <= lowest(close,5)[1], rsi(close,2)[0] >= 15"
    verdict = ""
    buy = (data['LowerCloses2'] > 0) & (data['RSI5SemisBreadth']>40)#& (data['RSI14SemisBreadth']>40) #& (data['ValueCharts'] < 0) 
    #True; #(data['IBR']<0.2) & (data['RSI2']>15)#(data['Low'] >= data['Low'].rolling(window=2).max()) & (data['Close'].shift(1) <= data['Close'].rolling(window=5).min().shift(1)) & (data['RSI2'] >= 15) & (data['Vix'] < 20)
    #buy = (data['IBR'] <= 0.3) & (data['Hurst'] > 0.5) #& (data['ValueCharts'] <= 10) &
    sell = False #(data['Close_SMA200'] > 0)#((data['RSI2SemisBreadth'].shift(1) > 50) & (data['RSI2SemisBreadth'] < 50)) | (data['Vix'] > 40)
    is_long = True
    return buy, sell, days, profit, description, verdict, is_long, ignore

def buy_signal24(data, symbol = ticker):
    allowed_symbols = ['GDX']
    ignore = False if symbol in allowed_symbols else True
    days = 2

    profit = 1
    description = "Testing"
    verdict = ""
    buy = (data['RSI5GoldBreadth'] > 60) & (data['RSI14UtilitiesBreadth'] < 50)
    sell = False 
    is_long = True
    return buy, sell, days, profit, description, verdict, is_long, ignore

def og_buy_signal(data, symbol = ticker):
    allowed_symbols = ['SPY']
    ignore = False if symbol in allowed_symbols else True
    description = "OG Long Spy strategy"
    verdict = "OG Long Spy strategy"
    is_long = True
    
    buy = ((data['RSI2'] < RSI2Buy) & (data['RSI5'] < RSI5Buy) & (                                 #RSI2 and RSI5 below threashold
        (data['VolumeEMADiff'] < VolumeEMAThreashold) | (data['Volatility'] < VolumeEMAThreashold)) & (  #Volume less than EMAThreashold
        ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) > -MaxDecline)))               #Decline less than 4%    
    
    sell = og_sell_signal(data, symbol = symbol)
    return buy, sell, 0, 0, description, verdict, is_long, ignore

def og_new_buy_signal(data, symbol = ticker):
    allowed_symbols = ['SPY', 'IWM', 'QQQ']
    ignore = False if symbol in allowed_symbols else True
    description = "OG Long Spy strategy with few extra conditions"
    verdict = "OG Long Spy strategy with few extra conditions"
    is_long = True
    
    buy = ((data['RSI2'] < RSI2Buy) & #(data['RSI5'] < RSI5Buy) &                                  #RSI2 and RSI5 below threashold
        ((data['VolumeEMADiff'] < VolumeEMAThreashold) | (data['Volatility'] < VolumeEMAThreashold)) & (  #Volume less than EMAThreashold
        ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) > -MaxDecline))            #Decline less than 4%   
        & (data['Stoch']<30) & (data['SMA50_SMA200']>0) & (data['ER']<0.7))                          #New Conditions
        #(data['RSI5Breadth'] < 90) &(data['RSI5Breadth'] > 20) & (data['ValueCharts']>-12) & (data['Stoch']<40))    #New Conditions            
    sell = og_new_sell_signal(data, symbol = symbol)
    return buy, sell, 0, 0, description, verdict, is_long, ignore    

def og_sell_signal(data, symbol = ticker):
    allowed_symbols = ['SPY']
    
    sell = ((data['RSI2'] > RSI2Sell) & (data['RSI5'] > RSI5Sell)) | (                                #RSI2 and RSI5 above threashold
            (data['Close'].shift(1) > data['EMA8'].shift(1)) & (data['Close'] < data['EMA8'])) | (            #Crossing EMA8 down
            (ExitOnVolatility) & ((data['VolumeEMADiff'] >= VolumeEMAThreashold))) | (                        #Volume more than EMAThreashold !!!!!!!!!!!DOESNT WORK - INVESTIGATE!!!!!!!!
            #(data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) < -MaxDecline) | (              #Decline more than 4%
            (data['VolumeEMADiff'] > VolumeEMAThreashold) & (data['Volatility'] > VolatilityThreashold))      #Big volume and volatility  

    return sell#, 0, 0, description, verdict, is_long, ignore

def og_new_sell_signal(data, symbol = ticker):
    allowed_symbols = ['SPY', 'ES']
    
    sell = ((data['RSI2'] > RSI2Sell) & (data['RSI5'] > RSI5Sell)) | (                                #RSI2 and RSI5 above threashold
            #(data['Close'].shift(1) > data['EMA8'].shift(1)) & (data['Close'] < data['EMA8'])) | (            #Crossing EMA8 down
            (ExitOnVolatility) & ((data['VolumeEMADiff'] >= VolumeEMAThreashold))) | (                        #Volume more than EMAThreashold !!!!!!!!!!!DOESNT WORK - INVESTIGATE!!!!!!!!
            #(data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) < -MaxDecline) | (              #Decline more than 4%
            ((data['VolumeEMADiff'] > VolumeEMAThreashold) & (data['Volatility'] > VolatilityThreashold))       #Big volume and volatility  
            | (data['SMA50_SMA200'] <0) | (data['EMA8CrossDown'] > 0))                                           #New Condition

    return sell#, 0, 0, description, verdict, is_long, ignore