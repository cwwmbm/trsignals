import pandas as pd
import indicators as ind
from config import ExcludeBestReturnYear


def yearly_returns(data):
    """Calendar-year fractional returns from RollingPnL."""
    returns = {}
    for year, group in data.groupby(data['Date'].dt.year):
        if len(group) < 2:
            continue
        start_pnl = group.iloc[0]['RollingPnL']
        end_pnl = group.iloc[-1]['RollingPnL']
        if start_pnl <= 0:
            continue
        returns[year] = (end_pnl / start_pnl) - 1
    return returns


def best_return_year_to_exclude(data, returns=None):
    """Year with the highest positive annual return, or None."""
    returns = yearly_returns(data) if returns is None else returns
    if not returns:
        return None
    best_year = max(returns, key=returns.get)
    return best_year if returns[best_year] > 0 else None


def exclude_best_return_year(data, best_year=None):
    """Drop rows from the single best-return year (positive outlier only)."""
    if not ExcludeBestReturnYear:
        return data.copy()
    if best_year is None:
        best_year = best_return_year_to_exclude(data)
    if best_year is None:
        return data.copy()
    return data[data['Date'].dt.year != best_year].copy()


def cagr_decimal(data, returns=None, best_year=None):
    """CAGR as a decimal, optionally excluding the best-return year."""
    if not ExcludeBestReturnYear:
        first, last = data.iloc[0], data.iloc[-1]
        return (last['RollingPnL'] / first['RollingPnL']) ** (1 / (data.shape[0] / 252)) - 1

    returns = yearly_returns(data) if returns is None else returns
    best_year = best_return_year_to_exclude(data, returns) if best_year is None else best_year
    if best_year is None or len(returns) <= 1:
        first, last = data.iloc[0], data.iloc[-1]
        return (last['RollingPnL'] / first['RollingPnL']) ** (1 / (data.shape[0] / 252)) - 1

    remaining = [r for y, r in returns.items() if y != best_year]
    cumulative = 1.0
    for r in remaining:
        cumulative *= (1 + r)
    return cumulative ** (1 / len(remaining)) - 1


def cagr_percent(data, returns=None, best_year=None):
    """CAGR formatted like ind.cagr() — percentage number, e.g. 46.12."""
    return round(cagr_decimal(data, returns, best_year) * 100, 2)


def compute_aggregate_metrics(data):
    """
    Aggregate backtest metrics for ranking/comparison.
    Excludes the single best-return year when ExcludeBestReturnYear is True.
    Total PnL always reflects the full run.
    """
    returns = yearly_returns(data) if ExcludeBestReturnYear else None
    excluded_year = best_return_year_to_exclude(data, returns) if ExcludeBestReturnYear else None
    metrics_data = exclude_best_return_year(data, excluded_year)
    cagr = cagr_decimal(data, returns, excluded_year)

    trade_out = metrics_data[metrics_data['LongTradeOut']]
    trades = trade_out.shape[0]
    positive = (trade_out['TradePnL'] > 0).sum()
    pct_positive = (positive / trades * 100) if trades else 0

    avg_win = 100 * trade_out[trade_out['TradePnL'] > 0]['TradePnL'].mean() if positive else 0
    avg_loss = 100 * trade_out[trade_out['TradePnL'] < 0]['TradePnL'].mean()
    kelly = None
    if trades and avg_win and avg_loss:
        kelly = (pct_positive / 100 - ((1 - pct_positive / 100) / (avg_win / (-avg_loss)))) * 100

    return {
        'excluded_year': excluded_year,
        'rolling_pnl': data['RollingPnL'].iloc[-1],
        'max_drawdown': metrics_data['Drawdown'].max(),
        'trades': trades,
        'pct_positive': pct_positive,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'kelly': kelly,
        'cagr_decimal': cagr,
        'cagr_percent': round(cagr * 100, 2),
        'sharpe': ind.sharpes_ratio(metrics_data),
        'sortino': ind.sortino_ratio(metrics_data),
    }


def yearly_performance(data):
    def _year_stats(group):
        first_day = group.iloc[0]
        last_day = group.iloc[-1]
        pnl_percent = ((last_day['RollingPnL'] - first_day['RollingPnL']) / first_day['RollingPnL']) * 100
        return pd.Series({
            'PnL%': f'{pnl_percent:.2f}%',
            'Drawdown%': f'{group["Drawdown"].max() * 100:.2f}%',
            'Num_Trades': group['LongTradeOut'].sum(),
            'Positive_Trades': group[(group['TradePnL'] > 0) & group['LongTradeOut']].shape[0],
        })

    return data.groupby(data['Date'].dt.year).apply(_year_stats)


def print_stats(data, days=0, profit=0, description='Original Strategy'):
    m = compute_aggregate_metrics(data)

    print(f"Number of trades: {m['trades']}")
    print(f"Latest Rolling PnL: {ind.format_dollar_value(m['rolling_pnl'])}")
    print(f"Maximum drawdown: {m['max_drawdown']:.2%}")
    print(f"CAGR: {m['cagr_decimal']:.2%}")
    print(f"Percentage of profitable trades: {m['pct_positive']:.2f}")
    if m['trades']:
        print(f"Average positive trade PnL: {m['avg_win']:.2f}%")
        print(f"Average negative trade PnL: {m['avg_loss']:.2f}%")
    print(f"Sharpe ratio: {m['sharpe']:.2f}")
    print(f"Sortino ratio: {m['sortino']:.2f}")
    if m['kelly'] is not None:
        print(f"Kelly Criterion: {m['kelly']:.2f}%")
    if m['excluded_year'] is not None:
        best_return = yearly_returns(data)[m['excluded_year']] * 100
        print(f"(Aggregate metrics exclude {m['excluded_year']} — best year at {best_return:.2f}%)")

    print(f"Hold for {days} days, profit {profit}")
    print(f"Description: {description}")
    print(yearly_performance(data))
