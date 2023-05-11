import pandas as pd
import getdata as dt
import indicators as ind

def signal_check():
    signals = pd.DataFrame()
    symbols = ['NQ', 'ES', 'CL', 'GC', 'SI', 'SPY', 'SMH', 'QQQ']

    buy_signals = [ind.buy_signal1, ind.buy_signal2, ind.buy_signal3, ind.buy_signal4, ind.buy_signal5, ind.buy_signal6, ind.buy_signal7, ind.buy_signal8, ind.buy_signal9, ind.buy_signal10, 
                ind.buy_signal11, ind.buy_signal12, ind.buy_signal13, ind.buy_signal14, ind.buy_signal15, ind.buy_signal16, ind.buy_signal17, ind.buy_signal18, ind.og_buy_signal]

    yf_symbols = [symbol+'=F' if symbol in ['NQ', 'ES', 'RTY', 'CL', 'GC', 'SI', 'HG'] else symbol for symbol in symbols]
    symbol_mapping = {symbol: yf_symbol for symbol, yf_symbol in zip(symbols, yf_symbols)}
    full_data = dt.get_bulk_data(yf_symbols)
    for symbol, yf_symbol in symbol_mapping.items():
        data= full_data.xs(yf_symbol, axis=1, level=1, drop_level=False)
        print('Getting data for ' + symbol + '...')
        data.columns = data.columns.droplevel(1)  # Reset column level
        data = dt.normalize_dataframe(data)
        data = data.drop(columns = ['Adj close'])
        data = dt.clean_holidays(data)
        #print("Getting data for " + symbol + "...")
        #data = dt.get_full_data(ib, False, symbol = symbol)
        data = ind.add_indicators(data)
        #print(data.tail(5))
        #time.sleep(5)
        for buy_signal in buy_signals:
            data_temp = data.copy()
            data_temp['Buy'], days, profit, description, verdict, is_long, ignore = buy_signal(data_temp, symbol)
            if not ignore:
                data_temp = ind.long_strat(data_temp, days, profit) if days>0 else ind.og_strat(data_temp)
                signals = signals._append([{'Symbol': symbol,'Name': buy_signal.__name__, 'Days': days, 'Profit': profit,
                                            'Buy?': data_temp['LongTradeIn'].iloc[-1], 'Hold?': data_temp['HoldLong'].iloc[-1], 'Sell?': data_temp['LongTradeOut'].iloc[-1], 'Description': description, 'Verdict': verdict, 'Date': data_temp['Date'].iloc[-1]}])
    return signals