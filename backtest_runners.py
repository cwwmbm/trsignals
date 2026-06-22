import getdata as dt
import indicators as ind
import backtest as bt
from stats import print_stats
import pandas as pd

from api.market_data_cache import PROFILE_SINGLE, load as load_cached_frame, save as save_cached_frame


def load_ticker_data(symbol, years=25, *, use_cache=True):
    normalized = symbol.strip().upper()
    if use_cache:
        cached = load_cached_frame(normalized, years, PROFILE_SINGLE)
        if cached is not None:
            return cached

    yfticker = dt.to_yf_symbol(normalized)
    data = dt.get_data_yf(yfticker, years=years, Local=False)
    data = dt.normalize_dataframe(data)
    if 'Adj close' in data.columns:
        data = data.drop(columns=['Adj close'])
    data = dt.clean_holidays(data)
    prepared = ind.add_indicators(data)
    if use_cache:
        save_cached_frame(normalized, years, PROFILE_SINGLE, prepared)
    return prepared


def attach_proxy_column(
    data: pd.DataFrame,
    proxy_symbol: str,
    *,
    years: int = 25,
    bulk_data: pd.DataFrame | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Merge proxy close prices into the signal frame, keyed by Date."""
    proxy = proxy_symbol.strip().upper()
    if not proxy:
        return data
    if proxy in data.columns:
        return data

    def _merge_proxy_close(proxy_close: pd.Series) -> pd.DataFrame:
        proxy_frame = pd.DataFrame({"Date": proxy_close.index, proxy: proxy_close.values})
        proxy_frame["Date"] = pd.to_datetime(proxy_frame["Date"])
        merged = data.merge(proxy_frame, on="Date", how="left")
        data[proxy] = merged[proxy]
        return data

    if use_cache:
        from api.market_data_cache import load_close_column

        cached_close = load_close_column(proxy, years)
        if cached_close is not None:
            return _merge_proxy_close(cached_close)

    yf_symbol = dt.to_yf_symbol(proxy)
    if bulk_data is not None:
        try:
            proxy_close = dt._bulk_close(bulk_data, yf_symbol)
        except (KeyError, TypeError):
            proxy_close = None
        if proxy_close is not None:
            return _merge_proxy_close(proxy_close)

    full_data = dt.get_bulk_data([yf_symbol], years=years)
    proxy_close = dt._bulk_close(full_data, yf_symbol)
    return _merge_proxy_close(proxy_close)


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
