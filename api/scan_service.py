from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as bt
import getdata as dt
import indicators as ind

from api.indicator_catalog import list_indicators
from api.proxy_symbol import execute_with_proxy, with_proxy_description
from api.strategy_compiler import compile_buy_mask, compile_sell_mask, format_condition_preview
from api.strategy_store import get_strategy_by_id, list_strategies
from api.portfolio_store import list_portfolios


def _condition_dict(condition) -> dict:
    if hasattr(condition, "model_dump"):
        return condition.model_dump()
    if hasattr(condition, "dict"):
        return condition.dict()
    return dict(condition)


SCAN_SYMBOLS = [
    "SPY",
    "SMH",
    "QQQ",
    "SOXX",
    "^VIX",
    "XLI",
    "XLU",
    "XLE",
    "XLF",
    "RSP",
    "IWM",
    "FXI",
    "AAPL",
    "GDX",
    "MSFT",
    "GLD",
    "XBI",
    "TLT",
]

SKIP_SCAN_SYMBOLS = {"^VIX", "RSP", "XLI", "XLU", "XLE", "XLF"}

FUTURES_SYMBOLS = frozenset(["NQ", "ES", "RTY", "CL", "GC", "SI", "HG"])


def _yf_symbol(symbol: str) -> str:
    if symbol in FUTURES_SYMBOLS:
        return symbol + "=F"
    return symbol


def _builder_scan_symbols(strategies) -> set[str]:
    symbols: set[str] = set()
    for strategy in strategies:
        symbols.add(strategy.symbol.strip().upper())
        for item in getattr(strategy, "confirm_symbols", None) or []:
            value = item.strip().upper()
            if value:
                symbols.add(value)
        proxy = getattr(strategy, "proxy_symbol", None)
        if proxy:
            value = str(proxy).strip().upper()
            if value:
                symbols.add(value)
    return symbols


def _scan_download_symbols(strategies) -> list[str]:
    builder_symbols = _builder_scan_symbols(strategies)
    extra_symbols = sorted(
        symbol
        for symbol in builder_symbols
        if symbol not in SCAN_SYMBOLS and symbol not in SKIP_SCAN_SYMBOLS
    )
    return list(dict.fromkeys([*SCAN_SYMBOLS, *extra_symbols]))


SCAN_SYMBOL_ORDER = [symbol for symbol in SCAN_SYMBOLS if symbol not in SKIP_SCAN_SYMBOLS]
LEGACY_SCAN_SYMBOLS = frozenset(SCAN_SYMBOL_ORDER)

# Matches signal_check.py buy_signals order (symbol loop is outer, signal inner).
SIGNAL_ORDER = [
    "buy_signal1",
    "buy_signal2",
    "buy_signal3",
    "buy_signal4",
    "buy_signal5",
    "buy_signal6",
    "buy_signal7",
    "buy_signal8",
    "buy_signal9",
    "buy_signal10",
    "buy_signal11",
    "buy_signal12",
    "buy_signal13",
    "buy_signal14",
    "buy_signal15",
    "buy_signal16",
    "buy_signal17",
    "buy_signal18",
    "buy_signal19",
    "buy_signal20",
    "buy_signal21",
    "buy_signal24",
    "og_buy_signal",
    "og_new_buy_signal",
]

LEGACY_BUY_SIGNALS = [
    ind.og_buy_signal,
    ind.og_new_buy_signal,
]


def compute_kelly(executed: pd.DataFrame) -> float | None:
    number_of_trades = int(executed["LongTradeOut"].value_counts().get(True, 0))
    if number_of_trades == 0:
        return None

    trade_out_rows = executed[executed["LongTradeOut"]]
    num_profitable_trades = int((trade_out_rows["TradePnL"] > 0).sum())
    percentage_profitable_trades = num_profitable_trades / number_of_trades

    positive = trade_out_rows[trade_out_rows["TradePnL"] > 0]["TradePnL"]
    negative = trade_out_rows[trade_out_rows["TradePnL"] < 0]["TradePnL"]
    if positive.empty or negative.empty:
        return None

    average_positive_trade_pnl = positive.mean()
    average_negative_trade_pnl = negative.mean()
    if average_negative_trade_pnl == 0 or pd.isna(average_positive_trade_pnl) or pd.isna(average_negative_trade_pnl):
        return None

    kelly = (
        percentage_profitable_trades
        - ((1 - percentage_profitable_trades) / (average_positive_trade_pnl / (-average_negative_trade_pnl)))
    ) * 100
    if pd.isna(kelly):
        return None
    return round(float(kelly), 2)


def _scan_trade_pnl_pct(executed: pd.DataFrame, *, is_long: bool = True) -> float:
    if executed.empty:
        return 0.0

    value = executed["TradePnL"].iloc[-1]
    if value is not None and not pd.isna(value):
        return round(float(value) * 100, 2)

    last = executed.iloc[-1]
    if not bool(last.get("HoldLong", False)):
        return 0.0

    closes = executed.loc[executed["HoldLong"], "Close"].dropna()
    entry = last.get("TradeEntry")
    if closes.empty or entry is None or pd.isna(entry) or not entry:
        return 0.0

    from config import Leverage

    pnl = (float(closes.iloc[-1]) - float(entry)) / float(entry)
    if not is_long:
        pnl = -pnl
    pnl *= Leverage
    return round(pnl * 100, 2)


def _prepare_symbol_frame(full_data: pd.DataFrame, symbol: str, yf_symbol: str) -> pd.DataFrame:
    vix_close = full_data["Close", "^VIX"]
    breadth = full_data["Close", "RSP"] / full_data["Close", "SPY"]
    qqq_to_spy = full_data["Close"]["QQQ"] / full_data["Close"]["SPY"]
    smh_to_spy = full_data["Close"]["SMH"] / full_data["Close"]["SPY"]
    xlf_to_spy = full_data["Close"]["XLF"] / full_data["Close"]["SPY"]
    xle_to_spy = full_data["Close"]["XLE"] / full_data["Close"]["SPY"]
    xlu_to_spy = full_data["Close"]["XLU"] / full_data["Close"]["SPY"]
    xli_to_spy = full_data["Close"]["XLI"] / full_data["Close"]["SPY"]
    gold_to_spy = full_data["Close"]["GLD"] / full_data["Close"]["SPY"]
    spy50 = full_data["Close"]["SPY"].rolling(50).mean()
    spy200 = full_data["Close"]["SPY"].rolling(200).mean()

    data = full_data.xs(yf_symbol, axis=1, level=1, drop_level=False)
    data.columns = data.columns.droplevel(1)
    data = data.copy()
    data["VIX"] = vix_close
    data["Breadth"] = breadth
    data["RiskBreadth"] = qqq_to_spy
    data["SemisBreadth"] = smh_to_spy
    data["FinancialsBreadth"] = xlf_to_spy
    data["EnergyBreadth"] = xle_to_spy
    data["UtilitiesBreadth"] = xlu_to_spy
    data["IndustrialsBreadth"] = xli_to_spy
    data["GoldBreadth"] = gold_to_spy
    data["BondBreadth"] = full_data["Close"]["TLT"] / full_data["Close"]["SPY"]
    data["Soxx"] = full_data["Close"]["SOXX"]
    data["QQQ"] = full_data["Close"]["QQQ"]
    data["SPYBull"] = np.where(spy50 > spy200, 1, -1)
    data = dt.normalize_dataframe(data)
    if "Adj close" in data.columns:
        data = data.drop(columns=["Adj close"])
    data = dt.clean_holidays(data)
    return ind.add_indicators(data)


def _legacy_scan_row(
    symbol: str,
    buy_signal,
    data: pd.DataFrame,
) -> dict | None:
    data_temp = data.copy()
    buy, sell, days, profit, description, _, is_long, ignore = buy_signal(data_temp, symbol)
    if ignore:
        return None

    data_temp["Buy"] = buy
    data_temp["Sell"] = sell
    executed = bt.execute_strategy(data_temp, days, profit, is_long)
    signal_name = buy_signal.__name__
    return {
        "id": f"legacy:{symbol}:{signal_name}",
        "source": "legacy",
        "strategy_id": None,
        "symbol": symbol,
        "signal": signal_name,
        "buy_signal": bool(executed["LongTradeIn"].iloc[-1]),
        "hold_long": bool(executed["HoldLong"].iloc[-1]),
        "sell_signal": bool(executed["LongTradeOut"].iloc[-1]),
        "days": int(days),
        "profit": int(profit),
        "trade_pnl": _scan_trade_pnl_pct(executed, is_long=is_long),
        "kelly": compute_kelly(executed),
        "description": description,
    }


def execute_saved_strategy(
    data: pd.DataFrame,
    strategy,
    *,
    years: int = 25,
    bulk_data: pd.DataFrame | None = None,
    symbol_data: dict | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    original_hold_on_buy = bt.HoldOnBuySignal
    bt.HoldOnBuySignal = bool(getattr(strategy, "hold_on_buy_signal", False))
    try:
        confirm_symbols = getattr(strategy, "confirm_symbols", None) or []
        if confirm_symbols:
            from api.builder_strategy import builder_signal_callable

            labels = {item["id"]: item["label"] for item in list_indicators(builder_only=True)}
            labels.update(
                {
                    f"strategy:{saved.id}": saved.name
                    for saved in list_strategies()
                }
            )
            signal = builder_signal_callable(
                strategy,
                strategy_resolver=get_strategy_by_id,
                labels=labels,
            )
            primary_symbol = strategy.symbol.strip().upper()
            needed = list(dict.fromkeys([primary_symbol, *confirm_symbols]))
            if symbol_data is None:
                if bulk_data is not None:
                    symbol_data = bt.build_symbol_dataset(bulk_data, needed)
                else:
                    symbol_data = bt.load_symbol_dataset(needed, years=years, use_cache=use_cache)
            frame, days, profit, _, _, is_long, _ = bt.apply_cross_symbol_signal(
                signal,
                primary_symbol,
                confirm_symbols,
                symbol_data,
            )
            return execute_with_proxy(
                frame,
                strategy,
                days,
                profit,
                is_long,
                years=years,
                bulk_data=bulk_data,
            )

        conditions = [_condition_dict(condition) for condition in strategy.conditions]
        from api.builder_strategy import _compile_strategy_masks

        buy, sell = _compile_strategy_masks(
            data,
            strategy,
            strategy_resolver=get_strategy_by_id,
        )
        is_long = strategy.direction == "long"

        frame = data.copy()
        frame["Buy"] = buy
        frame["Sell"] = sell
        return execute_with_proxy(
            frame,
            strategy,
            strategy.hold_days,
            strategy.profit,
            is_long,
            years=years,
            bulk_data=bulk_data,
        )
    finally:
        bt.HoldOnBuySignal = original_hold_on_buy


def _builder_scan_row(
    strategy,
    data: pd.DataFrame,
    *,
    bulk_data: pd.DataFrame | None = None,
    symbol_data: dict | None = None,
) -> dict:
    conditions = [_condition_dict(condition) for condition in strategy.conditions]
    executed = execute_saved_strategy(
        data,
        strategy,
        years=1,
        bulk_data=bulk_data,
        symbol_data=symbol_data,
        use_cache=False,
    )
    is_long = strategy.direction == "long"

    labels = {item["id"]: item["label"] for item in list_indicators(builder_only=True)}
    labels.update(
        {
            f"strategy:{saved.id}": saved.name
            for saved in list_strategies()
        }
    )
    rule_preview = format_condition_preview(conditions, labels)
    description = with_proxy_description(
        strategy.description.strip() or rule_preview,
        strategy,
    )

    return {
        "id": f"builder:{strategy.id}",
        "source": "builder",
        "strategy_id": strategy.id,
        "symbol": strategy.symbol,
        "signal": strategy.name,
        "buy_signal": bool(executed["LongTradeIn"].iloc[-1]),
        "hold_long": bool(executed["HoldLong"].iloc[-1]),
        "sell_signal": bool(executed["LongTradeOut"].iloc[-1]),
        "days": int(strategy.hold_days),
        "profit": int(strategy.profit),
        "trade_pnl": _scan_trade_pnl_pct(executed, is_long=strategy.direction == "long"),
        "kelly": compute_kelly(executed),
        "description": description,
    }


def _portfolio_scan_row(portfolio) -> dict | None:
    from api.portfolio_service import build_portfolio_overlay_frame

    try:
        frame, strategies, description, max_hold, max_profit = build_portfolio_overlay_frame(
            portfolio,
            years=1,
            use_cache=False,
        )
    except ValueError:
        return None

    if frame.empty:
        return None

    symbols = sorted({strategy.symbol.strip().upper() for strategy in strategies})
    return {
        "id": f"portfolio:{portfolio.id}",
        "source": "portfolio",
        "strategy_id": None,
        "portfolio_id": portfolio.id,
        "symbol": "+".join(symbols),
        "signal": portfolio.name,
        "buy_signal": bool(frame["LongTradeIn"].iloc[-1]),
        "hold_long": bool(frame["HoldLong"].iloc[-1]),
        "sell_signal": bool(frame["LongTradeOut"].iloc[-1]),
        "days": int(max_hold),
        "profit": int(max_profit),
        "trade_pnl": _scan_trade_pnl_pct(frame, is_long=True),
        "kelly": compute_kelly(frame),
        "description": description,
    }


def _scan_sort_key(row: dict) -> tuple[int, int, str]:
    symbol = row["symbol"]
    signal = row["signal"]
    symbol_rank = (
        SCAN_SYMBOL_ORDER.index(symbol)
        if symbol in SCAN_SYMBOL_ORDER
        else len(SCAN_SYMBOL_ORDER)
    )
    signal_rank = SIGNAL_ORDER.index(signal) if signal in SIGNAL_ORDER else len(SIGNAL_ORDER)
    return symbol_rank, signal_rank, signal


def run_scan(
    *,
    full_data: pd.DataFrame | None = None,
    symbol_frames: dict[str, pd.DataFrame] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    strategies = list_strategies()

    if symbol_frames is None:
        download_symbols = _scan_download_symbols(strategies)
        yf_symbols = [_yf_symbol(symbol) for symbol in download_symbols]
        symbol_mapping = dict(zip(download_symbols, yf_symbols))
        if full_data is None:
            full_data = dt.get_bulk_data(yf_symbols, years=1)

        symbol_frames = {}
        for symbol, yf_symbol in symbol_mapping.items():
            if symbol in SKIP_SCAN_SYMBOLS:
                continue
            symbol_frames[symbol] = _prepare_symbol_frame(full_data, symbol, yf_symbol)

    for symbol, data in symbol_frames.items():
        if symbol not in LEGACY_SCAN_SYMBOLS:
            continue
        for buy_signal in LEGACY_BUY_SIGNALS:
            row = _legacy_scan_row(symbol, buy_signal, data)
            if row is not None:
                rows.append(row)

    symbol_dataset: dict | None = None
    if full_data is not None:
        cross_symbol_strategies = [
            strategy
            for strategy in strategies
            if getattr(strategy, "confirm_symbols", None)
        ]
        if cross_symbol_strategies:
            symbols_needed: set[str] = set()
            for strategy in cross_symbol_strategies:
                symbols_needed.add(strategy.symbol.strip().upper())
                symbols_needed.update(
                    symbol.strip().upper()
                    for symbol in strategy.confirm_symbols
                )
            symbol_dataset = bt.build_symbol_dataset(full_data, list(symbols_needed))

    for strategy in strategies:
        if strategy.symbol not in symbol_frames:
            continue
        rows.append(
            _builder_scan_row(
                strategy,
                symbol_frames[strategy.symbol],
                bulk_data=full_data,
                symbol_data=symbol_dataset,
            )
        )

    for portfolio in list_portfolios():
        row = _portfolio_scan_row(portfolio)
        if row is not None:
            rows.append(row)

    rows.sort(key=_scan_sort_key)
    return rows
