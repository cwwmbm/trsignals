import getdata as dt
import indicators as ind
import backtest as bt
from stats import print_stats


def load_ticker_data(symbol, years=25):
    yfticker = dt.to_yf_symbol(symbol)
    data = dt.get_data_yf(yfticker, years=years, Local=False)
    data = dt.normalize_dataframe(data)
    if 'Adj close' in data.columns:
        data = data.drop(columns=['Adj close'])
    data = dt.clean_holidays(data)
    return ind.add_indicators(data)


def run_single_symbol_backtest(buy_signal, symbol, years=25, save_csv=True):
    """Detailed backtest on one symbol using config ticker or an explicit symbol."""
    data = load_ticker_data(symbol, years=years)
    data['Buy'], data['Sell'], days, profit, description, _, is_long, _ = buy_signal(data, symbol)
    data = bt.execute_strategy(data, days, profit, is_long)
    print_stats(data, days, profit, description)
    if save_csv:
        path = f'CSV/{symbol}_{buy_signal.__name__}.csv'
        data.to_csv(path)
        print(f"Saved to {path}")
    return data


def signal_combination_tryout(signal_a, signal_b, data, symbol):
    results = bt.backtest_signal_combinations(signal_a, signal_b, data, symbol)
    print(results)
    return results


def symbol_confirmation_tryout(buy_signal, primary_symbol, symbol_pool, years=25, confirm_sets=None):
    results = bt.backtest_symbol_confirmation_sweep(
        buy_signal, primary_symbol, symbol_pool, years=years, confirm_sets=confirm_sets,
    )
    print(results)
    return results


def symbol_confirmation_detail(buy_signal, primary_symbol, confirm_symbols=None, years=25, save_csv=True):
    confirm_symbols = confirm_symbols or []
    data, days, profit, description, is_long = bt.backtest_cross_symbol(
        buy_signal, primary_symbol, confirm_symbols, years=years,
    )
    print_stats(data, days, profit, description)
    if save_csv:
        confirm_label = '+'.join(confirm_symbols) if confirm_symbols else 'none'
        path = f'CSV/{primary_symbol}_{buy_signal.__name__}_confirm_{confirm_label}.csv'
        data.to_csv(path)
        print(f"Saved to {path}")
    return data
