#from ib_insync import IB, Future, util, Stock
import yfinance as yf
# import quandl
import pandas as pd
import datetime
from config import *
import pandas_market_calendars as mcal
from datetime import datetime as dtm, timedelta
import numpy as np
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
    start_date = (pd.to_datetime("today") - pd.DateOffset(years=years)).strftime("%Y")
    end_date = (dtm.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    start_date += "-01-01"
    # end_date = "2018-01-01"
    print(f"Start date: {start_date}, End date: {end_date}")
    if Local:
        print("Using local csv")
        data_symbol = pd.read_csv(f'CSV/{ticker}_yf.csv', index_col='Date', parse_dates=True)
    else:
        print("Using yahoo finance")
        tickers = [ticker, '^VIX', 'SPY', 'RSP', 'QQQ', 'SMH', 'IWM', 'XLF','XLE', 'XLU', 'XLI', 'SOXX', 'GLD', 'TLT', 'UVXY', 'SQQQ']
        data = yf.download(tickers, start=start_date, end=end_date)
        vix = data['Close']['^VIX']
        qqq = data['Close']['QQQ']
        soxx = data['Close']['SOXX']
        iwm = data['Close']['IWM']
        rsp_to_spy = data['Close']['RSP'] / data['Close']['SPY']
        qqq_to_spy = data['Close']['QQQ'] / data['Close']['SPY']
        smh_to_spy = data['Close']['SMH'] / data['Close']['SPY']
        xlf_to_spy = data['Close']['XLF'] / data['Close']['SPY']
        xle_to_spy = data['Close']['XLE'] / data['Close']['SPY']
        xlu_to_spy = data['Close']['XLU'] / data['Close']['SPY']
        xli_to_spy = data['Close']['XLI'] / data['Close']['SPY']
        iwm_to_spy = data['Close']['IWM'] / data['Close']['SPY']
        data_symbol = data.xs(ticker, axis=1, level=1, drop_level=False)
        data_symbol.columns = data_symbol.columns.droplevel(1)  # Reset column level
        data_symbol = data_symbol.copy()
        data_symbol['UVXY'] = data['Close']['UVXY']
        data_symbol['VIX'] = vix
        data_symbol['QQQ'] = qqq
        data_symbol['SOXX'] = soxx
        data_symbol['IWM'] = iwm
        data_symbol['SQQQ'] = data['Close']['SQQQ']
        data_symbol['SPY']  = data['Close']['SPY']
        data_symbol['Breadth'] = rsp_to_spy
        data_symbol['RiskBreadth'] = qqq_to_spy
        data_symbol['SemisBreadth'] = smh_to_spy
        data_symbol['FinancialsBreadth'] = xlf_to_spy
        data_symbol['EnergyBreadth'] = xle_to_spy
        data_symbol['UtilitiesBreadth'] = xlu_to_spy
        data_symbol['IndustrialsBreadth'] = xli_to_spy     
        data_symbol['IWMBreadth'] = iwm_to_spy
        data_symbol['BondBreadth'] = data['Close']['TLT'] / data['Close']['SPY']
        data_symbol['GoldBreadth'] = data['Close']['GLD'] / data['Close']['SPY']
        spy50 = data['Close']['SPY'].rolling(50).mean()
        spy200 = data['Close']['SPY'].rolling(200).mean()
        data_symbol['SPYBull'] = np.where(spy50>spy200, 1, -1)

        #remove rows with empty Close values
        data_symbol = data_symbol[data_symbol['Close'].notna()]
        # data_symbol.to_csv(f'CSV/{ticker}_yf.csv')
    #vix_data = dt.get_data_yf('^VIX', 20, False)
    
    return data_symbol

def get_data_quandl():
    # Set your API key
    # quandl.ApiConfig.api_key = '3hAmwu7_u6y5g4P37Sbo'
    # Get VIX futures data
    #vx1 = quandl.get('CHRIS/CBOE_VX1', start_date='2000-01-01', end_date='2023-04-11')
    #vx4 = quandl.get('CHRIS/CBOE_VX4', start_date='2000-01-01', end_date='2023-04-11')
    # data = quandl.get('NASDAQOMX/NDX', start_date='2000-01-01', end_date='2023-04-20')
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
        # Drop 'Adj close' column if it exists (not present in newer yfinance versions with auto_adjust=True)
        if 'Adj close' in data.columns:
            data = data.drop(columns = ['Adj close'])
        data = clean_holidays(data)
        #data_ib=data_ib.drop(columns = ['Average'])
        #data_ib=data_ib.drop(columns = ['Barcount'])
        #data = data._append(data_ib, ignore_index=True)
        data = data.drop_duplicates(subset=['Date'], keep='last')
    return data

def default_scan_bulk_csv_path():
    """Default Scan CSV path; SCAN_BULK_CSV env overrides when set."""
    import os
    from pathlib import Path

    env_path = os.environ.get("SCAN_BULK_CSV")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent / "Alpaca.Market" / "scan_bulk_yf.csv"


def load_bulk_csv(path):
    """Load a yfinance-style MultiIndex OHLCV CSV (Price x Ticker columns, Date index)."""
    data = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError(f"Expected MultiIndex columns in bulk CSV: {path}")
    # Drop empty name row artifacts; keep Price/Ticker level names when present.
    if data.columns.nlevels >= 2:
        data.columns = data.columns.set_names(["Price", "Ticker"])
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data.index.name = "Date"
    return data.sort_index()


def get_bulk_data(symbols, years=1, csv_path=None):
    """
    Download (or load) multi-ticker OHLCV in yfinance MultiIndex shape.

    If csv_path is set, load that CSV instead of calling yfinance.
    The CSV must already contain the needed symbols.
    """
    if csv_path:
        data = load_bulk_csv(csv_path)
        if symbols:
            available = set(data.columns.get_level_values(1))
            missing = [symbol for symbol in symbols if symbol not in available]
            if missing:
                raise ValueError(
                    f"Bulk CSV missing symbols: {', '.join(missing)} (file={csv_path})"
                )
            keep = [col for col in data.columns if col[1] in set(symbols)]
            data = data.loc[:, keep]
        return data

    end_date = (dtm.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (pd.to_datetime("today") - pd.DateOffset(years=years)).strftime("%Y")
    start_date += "-01-01"
    data = yf.download(symbols, start=start_date, end=end_date)
    return data


def ib_connection_settings():
    """Host/port/clientId for TWS/Gateway. Defaults match legacy getdata stubs."""
    import os

    host = os.environ.get("IB_HOST", "127.0.0.1")
    port = int(os.environ.get("IB_PORT", "4001"))
    client_id = int(os.environ.get("IB_CLIENT_ID", "31"))
    return host, port, client_id


def _ensure_ib_event_loop():
    """
    ib_insync/eventkit require a thread-local asyncio loop at import and runtime.

    FastAPI runs sync routes in AnyIO worker threads that have no loop by default,
    which raises: RuntimeError: There is no current event loop in thread ...
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ib_canonical_symbol(symbol: str) -> str:
    value = str(symbol).strip()
    if value.upper() in {"VIX", "^VIX"}:
        return "^VIX"
    return value.upper() if value != "^VIX" else value


def ib_contract_for_symbol(symbol: str):
    """Return (ib_insync contract, canonical ticker used in MultiIndex)."""
    _ensure_ib_event_loop()
    from ib_insync import Index, Stock

    canonical = ib_canonical_symbol(symbol)
    if canonical == "^VIX":
        return Index("VIX", "CBOE"), canonical
    return Stock(canonical, "SMART", "USD"), canonical


def ib_bars_to_frame(bars) -> pd.DataFrame:
    """Convert ib_insync BarDataList to OHLCV frame indexed by date."""
    if not bars:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    rows = []
    for bar in bars:
        date = pd.Timestamp(bar.date)
        if getattr(date, "tz", None) is not None:
            date = date.tz_convert(None)
        date = date.normalize()
        rows.append(
            {
                "Date": date,
                "Open": float(bar.open),
                "High": float(bar.high),
                "Low": float(bar.low),
                "Close": float(bar.close),
                "Volume": float(bar.volume) if bar.volume is not None else 0.0,
            }
        )
    frame = pd.DataFrame(rows).drop_duplicates(subset=["Date"], keep="last")
    return frame.set_index("Date").sort_index()


def build_bulk_multiindex(symbol_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble yfinance-shaped MultiIndex OHLCV from {ticker: OHLCV frame}."""
    pieces = []
    for symbol, frame in symbol_frames.items():
        if frame is None or frame.empty:
            continue
        part = frame[["Open", "High", "Low", "Close", "Volume"]].copy()
        part.columns = pd.MultiIndex.from_product(
            [part.columns, [symbol]], names=["Price", "Ticker"]
        )
        pieces.append(part)
    if not pieces:
        raise ValueError("No IB bars returned for any requested symbol")
    bulk = pd.concat(pieces, axis=1).sort_index()
    price_order = ["Close", "High", "Low", "Open", "Volume"]
    tickers = list(dict.fromkeys(symbol_frames.keys()))
    ordered = [
        (price, ticker)
        for price in price_order
        for ticker in tickers
        if (price, ticker) in bulk.columns
    ]
    bulk = bulk.reindex(columns=ordered)
    bulk.columns = pd.MultiIndex.from_tuples(ordered, names=["Price", "Ticker"])
    bulk.index.name = "Date"
    return bulk


def _ib_duration_str(years: int = 1) -> str:
    """
    IB rejects day-based durations longer than 365 days; use years instead.
    Scan's years=1 window starts Jan 1 of (today - years), so request 2Y to cover it.
    """
    if years <= 1:
        return "2 Y"
    return f"{int(years) + 1} Y"


def get_bulk_data_ib(symbols, years=1):
    """
    Fetch multi-ticker daily OHLCV from IB TWS/Gateway in yfinance MultiIndex shape.

    Requires a running Gateway/TWS. ^VIX is requested as CBOE Index('VIX').
    """
    _ensure_ib_event_loop()
    from ib_insync import IB

    if not symbols:
        raise ValueError("symbols must be non-empty")

    host, port, client_id = ib_connection_settings()
    duration = _ib_duration_str(years)
    ib = IB()
    symbol_frames: dict[str, pd.DataFrame] = {}
    try:
        try:
            ib.connect(host, port, clientId=client_id, timeout=5)
        except Exception as exc:  # noqa: BLE001 — surface as ValueError for API 400
            raise ValueError(
                f"Could not connect to Interactive Brokers at {host}:{port} "
                f"(clientId={client_id}). Is Gateway/TWS running? ({exc})"
            ) from exc

        for raw_symbol in symbols:
            contract, canonical = ib_contract_for_symbol(raw_symbol)
            try:
                qualified = ib.qualifyContracts(contract)
                if not qualified:
                    raise ValueError(f"IB could not qualify contract for {canonical}")
                bars = ib.reqHistoricalData(
                    qualified[0],
                    endDateTime="",
                    durationStr=duration,
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                    keepUpToDate=False,
                )
                symbol_frames[canonical] = ib_bars_to_frame(bars)
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"IB historical data failed for {canonical}: {exc}") from exc
    finally:
        if ib.isConnected():
            ib.disconnect()

    missing = [
        ib_canonical_symbol(symbol)
        for symbol in symbols
        if ib_canonical_symbol(symbol) not in symbol_frames
        or symbol_frames[ib_canonical_symbol(symbol)].empty
    ]
    # Deduplicate while preserving order
    missing = list(dict.fromkeys(missing))
    if missing:
        raise ValueError(f"IB returned no bars for: {', '.join(missing)}")

    ordered_frames = {
        ib_canonical_symbol(symbol): symbol_frames[ib_canonical_symbol(symbol)]
        for symbol in symbols
    }
    return build_bulk_multiindex(ordered_frames)


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


FUTURES_SYMBOLS = ['NQ', 'ES', 'RTY', 'CL', 'GC', 'SI', 'HG', 'NG']
MARKET_CONTEXT_SYMBOLS = ['^VIX', 'SPY', 'RSP', 'QQQ', 'SMH', 'XLF', 'XLE', 'XLU', 'XLI', 'GLD', 'TLT', 'SOXX']


def to_yf_symbol(symbol):
    if symbol in FUTURES_SYMBOLS:
        return symbol + '=F'
    if symbol == 'GBPUSD':
        return symbol + '=X'
    return symbol


def _bulk_close(full_data, yf_symbol):
    close = full_data['Close']
    if isinstance(close, pd.DataFrame):
        if yf_symbol in close.columns:
            return close[yf_symbol]
        reference = close['SPY'] if 'SPY' in close.columns else close.iloc[:, 0]
        return pd.Series(np.nan, index=reference.index)
    return close


def extract_market_context(full_data, symbol_to_yf):
    spy = _bulk_close(full_data, symbol_to_yf['SPY'])
    spy50 = spy.rolling(50).mean()
    spy200 = spy.rolling(200).mean()
    spy_bull = pd.Series(np.where(spy50 > spy200, 1, -1), index=spy.index)
    return {
        'vix_close': _bulk_close(full_data, symbol_to_yf['^VIX']),
        'breadth': _bulk_close(full_data, symbol_to_yf['RSP']) / spy,
        'qqq_to_spy': _bulk_close(full_data, symbol_to_yf['QQQ']) / spy,
        'smh_to_spy': _bulk_close(full_data, symbol_to_yf['SMH']) / spy,
        'xlf_to_spy': _bulk_close(full_data, symbol_to_yf['XLF']) / spy,
        'xle_to_spy': _bulk_close(full_data, symbol_to_yf['XLE']) / spy,
        'xlu_to_spy': _bulk_close(full_data, symbol_to_yf['XLU']) / spy,
        'xli_to_spy': _bulk_close(full_data, symbol_to_yf['XLI']) / spy,
        'gold_to_spy': _bulk_close(full_data, symbol_to_yf['GLD']) / spy,
        'bond_breadth': _bulk_close(full_data, symbol_to_yf['TLT']) / spy,
        'soxx': _bulk_close(full_data, symbol_to_yf['SOXX']),
        'qqq': _bulk_close(full_data, symbol_to_yf['QQQ']),
        'spy_bull': spy_bull,
    }


def symbol_frame_from_bulk(full_data, yf_symbol, market_context):
    data = full_data.xs(yf_symbol, axis=1, level=1, drop_level=False)
    data.columns = data.columns.droplevel(1)
    data = data.copy()
    primary_columns = [column for column in ['Open', 'High', 'Low', 'Close'] if column in data.columns]
    if primary_columns:
        data = data.dropna(subset=primary_columns)
    ctx = market_context
    data['VIX'] = ctx['vix_close']
    data['Breadth'] = ctx['breadth']
    data['RiskBreadth'] = ctx['qqq_to_spy']
    data['SemisBreadth'] = ctx['smh_to_spy']
    data['FinancialsBreadth'] = ctx['xlf_to_spy']
    data['EnergyBreadth'] = ctx['xle_to_spy']
    data['UtilitiesBreadth'] = ctx['xlu_to_spy']
    data['IndustrialsBreadth'] = ctx['xli_to_spy']
    data['GoldBreadth'] = ctx['gold_to_spy']
    data['BondBreadth'] = ctx['bond_breadth']
    data['Soxx'] = ctx['soxx']
    data['QQQ'] = ctx['qqq']
    data['SPYBull'] = ctx['spy_bull']
    data = normalize_dataframe(data)
    if 'Adj close' in data.columns:
        data = data.drop(columns=['Adj close'])
    return clean_holidays(data)

