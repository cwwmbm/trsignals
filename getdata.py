#from ib_insync import IB, Future, util, Stock
import yfinance as yf
import quandl
import pandas as pd
import datetime
from config import *
import pandas_market_calendars as mcal
from datetime import datetime as dtm, timedelta
""""
def ib_connect():
    # Create an IB instance
    ib = IB()

    # Connect to the Interactive Brokers Gateway
    ib.connect('127.0.0.1', 4001, clientId=1)
    return ib

#Get list of accounts and positions in these accounts.
def get_accounts(ib):
    # Get the list of available accounts
    accounts = ib.managedAccounts()
    print("Available accounts:", accounts)

    for account in accounts:
        # Get the account values for the specified account
        account_values = ib.accountValues(account)

        # Get the current open positions for the specified account
        positions = ib.positions(account)

        # Filter the account values to get the net liquidation value
        net_liquidation = [value for value in account_values if value.tag == 'NetLiquidation']
        for value in net_liquidation:
            print(f"Account {value.account} Net Liquidation: {value.value} {value.currency}")

        # Print current open positions
        print("\nCurrent open positions:")
        for position in positions:
            print (position)

def get_quote_ib(ib, contract):
    # Request contract market data
    ib.reqMktData(contract, '', False, False)
    util.sleep(1)  # Wait for the market data to be received

    try:
        while True:

            # Get the current NQ quote
            ticker = ib.ticker(contract)
            last = ticker.last

            # Print the current time and NQ quote
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] NQ Last: {last}")
            # Wait for 1 minute
            util.sleep(10)

    except KeyboardInterrupt:
        # Disconnect from the Interactive Brokers Gateway when the user stops the script
        ib.disconnect()
        print("\nStopped and disconnected from Interactive Brokers Gateway.")

#Function to get the historical data from IB Gateway
def get_data_ib(ib, contract, useRTH = True, Local = False, period = '1 D', barSize = '1 day'):
    # Request historical data for the contract
    if Local:
        print("Using local csv")
        data = pd.read_csv('NQ_ib.csv', index_col='Date', parse_dates=True)
    else:
        print("Using IB Gateway")
        bars = ib.reqHistoricalData(
            contract, endDateTime='', durationStr=period, barSizeSetting=barSize,
            whatToShow='TRADES', useRTH=useRTH, formatDate=1, keepUpToDate=False
        )
        if not bars:
            print(f"No historical data for {contract.symbol}")
            return
        data = util.df(bars)
        data.to_csv('NQ_ib.csv')

    return data

    
    return data
"""
#Function to get the data from Yahoo Finance
def get_data_yf(ticker, years=1, Local=False):
    start_date = pd.to_datetime("today") - pd.DateOffset(years=years)
    end_date = (dtm.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    if Local:
        print("Using local csv")
        data = pd.read_csv('NQ_yf.csv', index_col='Date', parse_dates=True)
    else:
        print("Using yahoo finance")
        data = yf.download(ticker, start=start_date, end=end_date)
        data.to_csv('NQ_yf.csv')
    return data

def get_data_quandl():
    # Set your API key
    quandl.ApiConfig.api_key = '3hAmwu7_u6y5g4P37Sbo'
    # Get VIX futures data
    #vx1 = quandl.get('CHRIS/CBOE_VX1', start_date='2000-01-01', end_date='2023-04-11')
    #vx4 = quandl.get('CHRIS/CBOE_VX4', start_date='2000-01-01', end_date='2023-04-11')
    data = quandl.get('NASDAQOMX/NDX', start_date='2000-01-01', end_date='2023-04-20')
    #data = quandl.get('CHRIS/CME_NQ1', start_date='2000-01-01', end_date='2023-04-20')
    data = data.rename(columns={
        "Trade Date": "Date",
        "Index Value": "Close",
        "High": "High",
        "Low": "Low",
        "Total Market Value": "Volume",
        "Dividend Market Value": "Dividends"
    })
    data = normalize_dataframe(data)
    data = data[['Date', 'Open', 'High', 'Low', 'Last', 'Volume']]
    data.rename(columns={'Last': 'Close'}, inplace=True)
    #print(data)
    return data

def normalize_dataframe(df):
    #Capitalize the column names
    df = df.rename(columns=lambda x: x.capitalize())
    
    # Reset the index and move the 'Date' column from the index to a regular column
    if not isinstance(df.index, pd.RangeIndex):
        # Reset the index and move the 'Date' column from the index to a regular column
        df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def get_full_data(ib, Local = False, years = 1, symbol = ticker):
    # Request historical data for the contract
    if Local:
        print("Using local csv")
        data = pd.read_csv('NQ_ib.csv', index_col='Date', parse_dates=True)
    else:
        print("Using yahoo finance and IB Gateway")
        # Define contract class for IB and ticker for Yahoo Finance based on config variables
        if (symbol == 'NQ' or symbol == 'ES' or symbol == 'RTY' or symbol == 'CL' or symbol == 'GC' or symbol == 'SI' or symbol == 'HG'):
            #contract = Future(symbol, '202306', 'CME')
            yfticker = symbol + '=F'
        else:
            #contract = Stock(symbol, 'ARCA')
            yfticker = symbol


        #data_ib = get_data_ib(ib, contract, False, period = '1 D', barSize='1 day') #True = use RTH data, False = use all data
        #data_ib = normalize_dataframe(data_ib)

        data = get_data_yf(yfticker, years, False) #True for local data, False for Yahoo Finance
        data = normalize_dataframe(data)
        data = data.drop(columns = ['Adj close'])
        data = clean_holidays(data)
        #data_ib=data_ib.drop(columns = ['Average'])
        #data_ib=data_ib.drop(columns = ['Barcount'])
        #data = data._append(data_ib, ignore_index=True)
        data = data.drop_duplicates(subset=['Date'], keep='last')
    return data

def get_bulk_data(symbols, years = 1):
    today = datetime.datetime.now()
    end_date = (dtm.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = pd.to_datetime("today") - pd.DateOffset(years=years)
    data = yf.download(symbols, start=start_date, end=end_date)
    return data


def clean_holidays(data):
    # Remove holidays
    nyse = mcal.get_calendar("NYSE")

    # Calculate the date range for the historical data
    start_date = data['Date'].min()
    end_date = data['Date'].max()

    # Get the market schedule within the date range
    schedule = nyse.schedule(start_date, end_date)

    # Get the valid trading days within the date range
    valid_days = schedule.index

    # Convert the 'Date' column to pandas datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Filter out rows with holiday dates
    clean_data = data[data['Date'].isin(valid_days)]
    return clean_data

