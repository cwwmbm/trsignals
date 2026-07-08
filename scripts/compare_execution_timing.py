#!/usr/bin/env python3
"""
Compare close vs next-day-open execution for a saved strategy.

Signals are evaluated on the bar close (unchanged). Execution timing differs:
  - close: enter/exit on the same bar the signal fires (current engine default)
  - next_open: signal on day T -> enter/exit at day T+1 open

The next_open path only changes fill price (Open) and signal timing (+1 bar).
Profitable-close counting and daily returns stay on Close (%Change), matching close execution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import warn_config  # noqa: F401

import numpy as np
import backtest as bt
import indicators as ind
from config import Leverage
from api.builder_strategy import prepare_builder_refine_frame
from api.indicator_catalog import list_indicators
from api.strategy_store import get_strategy_by_id, list_strategies
from stats import compute_aggregate_metrics, yearly_performance


DEFAULT_STRATEGY_NAME = "Signal 16 Confirmed1"


def _find_strategy(name: str):
    matches = [s for s in list_strategies() if s.name.strip().lower() == name.strip().lower()]
    if not matches:
        available = sorted({s.name for s in list_strategies()})
        raise SystemExit(f"Strategy {name!r} not found. Available: {', '.join(available)}")
    if len(matches) > 1:
        raise SystemExit(f"Multiple strategies named {name!r}; pass --strategy-id instead.")
    return matches[0]


def _labels() -> dict[str, str]:
    labels = {item["id"]: item["label"] for item in list_indicators(builder_only=True)}
    labels.update({f"strategy:{saved.id}": saved.name for saved in list_strategies()})
    return labels


def _prepare_signal_frame(strategy, *, years: int):
    return prepare_builder_refine_frame(
        strategy,
        years=years,
        strategy_resolver=get_strategy_by_id,
        labels=_labels(),
    )


def _run_close_execution(frame, days: int, profit: int, is_long: bool, hold_on_buy: bool, pnl_column: str | None):
    original = bt.HoldOnBuySignal
    bt.HoldOnBuySignal = hold_on_buy
    try:
        return bt.execute_strategy(frame.copy(), days, profit, is_long, pnl_column=pnl_column)
    finally:
        bt.HoldOnBuySignal = original


def _mark_price(prices: np.ndarray, i: int) -> float:
    price = prices[i]
    if price and not np.isnan(price):
        return price
    if i <= 0:
        return price
    prior = prices[:i]
    valid = prior[~np.isnan(prior) & (prior != 0)]
    return valid[-1] if valid.size else np.nan


def _run_next_open_execution(
    frame,
    days: int,
    profit: int,
    is_long: bool,
    hold_on_buy: bool,
    *,
    start_capital: float = 15000,
):
    """
    long_strat-compatible simulator: signal on T acts on T+1 at Open.

    Only fill price and Buy/Sell timing differ from close execution. Exit rules
    (days held, profitable closes) and RollingPnL compounding use Close/%Change.
    """
    signals = frame.copy()
    n = len(signals)
    signals["TrackChange"] = signals["%Change"]

    buy = np.asarray(signals["Buy"].shift(1).fillna(False).astype(bool).values, dtype=bool)
    sell = np.asarray(signals["Sell"].shift(1).fillna(False).astype(bool).values, dtype=bool)
    open_px = np.asarray(signals["Open"].values, dtype=float)
    close_px = np.asarray(signals["Close"].values, dtype=float)
    track_change = np.asarray(signals["TrackChange"].fillna(0).values, dtype=float)

    long_in = np.zeros(n, dtype=bool)
    long_out = np.zeros(n, dtype=bool)
    hold_long = np.zeros(n, dtype=bool)
    days_in_trade = np.zeros(n, dtype=int)
    profitable_closes = np.zeros(n, dtype=int)
    rolling_pnl = np.zeros(n, dtype=float)
    trade_pnl = np.zeros(n, dtype=float)
    trade_entry = np.zeros(n, dtype=float)

    for i in range(n):
        if i > 0:
            hold_long[i] = (hold_long[i - 1] and not long_out[i - 1]) or long_in[i - 1]

        long_in[i] = buy[i] and not hold_long[i]
        days_in_trade[i] = days_in_trade[i - 1] + 1 if hold_long[i] and i > 0 else 0

        if hold_long[i] and i > 0:
            if is_long:
                profitable_closes[i] = (
                    profitable_closes[i - 1] + 1
                    if close_px[i] > close_px[i - 1]
                    else profitable_closes[i - 1]
                )
            else:
                profitable_closes[i] = (
                    profitable_closes[i - 1] + 1
                    if close_px[i] < close_px[i - 1]
                    else profitable_closes[i - 1]
                )

        if hold_on_buy and buy[i] and hold_long[i]:
            long_out[i] = False
        else:
            long_out[i] = (
                (sell[i] and hold_long[i])
                or (days_in_trade[i] >= days)
                or (profitable_closes[i] >= profit)
            )

        if long_in[i]:
            trade_entry[i] = _mark_price(open_px, i)
        elif hold_long[i]:
            trade_entry[i] = trade_entry[i - 1] if i > 0 else 0

        fill = _mark_price(open_px, i)
        entry = trade_entry[i]
        trade_pnl[i] = (fill - entry) / entry if hold_long[i] and entry and not np.isnan(fill) else 0

        if i == 0:
            rolling_pnl[i] = start_capital
        elif hold_long[i]:
            change = track_change[i]
            if np.isnan(change):
                change = 0.0
            if is_long:
                rolling_pnl[i] = (1 + change) * rolling_pnl[i - 1]
            else:
                rolling_pnl[i] = (1 - change) * rolling_pnl[i - 1]
        else:
            rolling_pnl[i] = rolling_pnl[i - 1]

    trade_pnl = Leverage * trade_pnl if is_long else -1 * Leverage * trade_pnl

    signals["Buy"] = buy
    signals["Sell"] = sell
    signals["LongTradeIn"] = long_in
    signals["LongTradeOut"] = long_out
    signals["HoldLong"] = hold_long
    signals["DaysInTrade"] = days_in_trade
    signals["ProfitableCloses"] = profitable_closes
    signals["RollingPnL"] = rolling_pnl
    signals["TradePnL"] = trade_pnl
    signals["TradeEntry"] = trade_entry
    signals["RunningMax"] = signals["RollingPnL"].cummax()
    signals["Drawdown"] = (signals["RunningMax"] - signals["RollingPnL"]) / signals["RunningMax"]
    return signals


def _metrics_row(label: str, data) -> dict:
    m = compute_aggregate_metrics(data)
    return {
        "execution": label,
        "rolling_pnl": m["rolling_pnl"],
        "max_drawdown_pct": round(m["max_drawdown"] * 100, 2),
        "trades": m["trades"],
        "pct_positive": round(m["pct_positive"], 1),
        "cagr_pct": m["cagr_percent"],
        "sharpe": round(m["sharpe"], 2) if m["sharpe"] is not None else None,
        "sortino": round(m["sortino"], 2) if m["sortino"] is not None else None,
        "kelly_pct": round(m["kelly"], 2) if m["kelly"] is not None else None,
        "excluded_year": m["excluded_year"],
    }


def _print_comparison(strategy, close_data, open_data, years: int) -> None:
    confirm = ", ".join(strategy.confirm_symbols or []) or "(none)"
    proxy = getattr(strategy, "proxy_symbol", None) or "(none)"
    print()
    print(f"Strategy: {strategy.name}")
    print(f"Symbol: {strategy.symbol}  |  Confirm: {confirm}  |  Proxy: {proxy}")
    print(f"Hold: {strategy.hold_days}d / {strategy.profit}p  |  Hold on buy: {strategy.hold_on_buy_signal}")
    print(f"History: {years} years")
    print(f"Close range: {close_data['Date'].iloc[0].date()} -> {close_data['Date'].iloc[-1].date()}")
    print("Next-open run: T+1 Open fills; profitable-close + daily returns still on Close.")
    print()

    close_row = _metrics_row("close (same bar)", close_data)
    open_row = _metrics_row("next_open (T+1 open)", open_data)

    headers = [
        ("Rolling PnL", "rolling_pnl", ind.format_dollar_value),
        ("Max drawdown", "max_drawdown_pct", lambda v: f"{v:.2f}%"),
        ("Trades", "trades", str),
        ("Win rate", "pct_positive", lambda v: f"{v:.1f}%"),
        ("CAGR", "cagr_pct", lambda v: f"{v:.2f}%"),
        ("Sharpe", "sharpe", str),
        ("Sortino", "sortino", str),
        ("Kelly", "kelly_pct", lambda v: f"{v:.2f}%" if v is not None else "n/a"),
    ]

    col_w = 22
    print(f"{'Metric':<18} {'Close':>{col_w}} {'Next-day open':>{col_w}} {'Delta':>{col_w}}")
    print("-" * (18 + col_w * 3 + 2))
    for title, key, fmt in headers:
        left = close_row[key]
        right = open_row[key]
        if key in {"rolling_pnl"}:
            delta = right - left
            delta_str = ind.format_dollar_value(delta)
        elif key in {"trades"}:
            delta_str = str(right - left)
        elif key in {"sharpe", "sortino", "kelly_pct"} and left is not None and right is not None:
            delta_str = f"{right - left:+.2f}"
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            delta_str = f"{right - left:+.2f}"
        else:
            delta_str = "—"
        print(f"{title:<18} {fmt(left):>{col_w}} {fmt(right):>{col_w}} {delta_str:>{col_w}}")

    excluded = close_row["excluded_year"]
    if excluded is not None:
        print(f"\nAggregate metrics exclude best-return year: {excluded}")

    print("\nYearly performance (close execution):")
    print(yearly_performance(close_data).to_string())
    print("\nYearly performance (next-day open execution):")
    print(yearly_performance(open_data).to_string())
    print()


def compare_strategy(strategy, *, years: int = 25, save_csv: bool = False):
    frame, days, profit, is_long, pnl_col = _prepare_signal_frame(strategy, years=years)
    hold_on_buy = bool(getattr(strategy, "hold_on_buy_signal", False))

    close_data = _run_close_execution(frame, days, profit, is_long, hold_on_buy, pnl_col)
    open_data = _run_next_open_execution(frame, days, profit, is_long, hold_on_buy)

    _print_comparison(strategy, close_data, open_data, years)

    if save_csv:
        out_dir = _ROOT / "CSV"
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = strategy.name.replace(" ", "_")
        close_path = out_dir / f"{strategy.symbol}_{slug}_close.csv"
        open_path = out_dir / f"{strategy.symbol}_{slug}_next_open.csv"
        close_data.to_csv(close_path, index=False)
        open_data.to_csv(open_path, index=False)
        print(f"Saved {close_path}")
        print(f"Saved {open_path}")

    return close_data, open_data


def main():
    parser = argparse.ArgumentParser(description="Compare close vs next-day-open execution timing.")
    parser.add_argument("--strategy-name", default=DEFAULT_STRATEGY_NAME)
    parser.add_argument("--strategy-id", default=None, help="Use a specific saved strategy id")
    parser.add_argument("--years", type=int, default=25)
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()

    if args.strategy_id:
        strategy = get_strategy_by_id(args.strategy_id)
        if strategy is None:
            raise SystemExit(f"Unknown strategy id: {args.strategy_id}")
    else:
        strategy = _find_strategy(args.strategy_name)

    compare_strategy(strategy, years=args.years, save_csv=args.save_csv)


if __name__ == "__main__":
    main()
