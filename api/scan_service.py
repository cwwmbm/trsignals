from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as bt
import getdata as dt
import indicators as ind

from api.indicator_catalog import list_indicators
from api.strategy_compiler import compile_buy_mask, compile_sell_mask, format_condition_preview
from api.strategy_store import list_strategies

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

SCAN_SYMBOL_ORDER = [symbol for symbol in SCAN_SYMBOLS if symbol not in SKIP_SCAN_SYMBOLS]

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
        "trade_pnl": round(float(executed["TradePnL"].iloc[-1]) * 100, 2),
        "kelly": compute_kelly(executed),
        "description": description,
    }


def execute_saved_strategy(data: pd.DataFrame, strategy) -> pd.DataFrame:
    conditions = [condition.model_dump() for condition in strategy.conditions]
    buy = compile_buy_mask(data, conditions)
    sell_conditions = [condition.model_dump() for condition in strategy.sell_conditions]
    sell = compile_sell_mask(data, sell_conditions) if sell_conditions else False
    is_long = strategy.direction == "long"

    frame = data.copy()
    frame["Buy"] = buy
    frame["Sell"] = sell
    return bt.execute_strategy(frame, strategy.hold_days, strategy.profit, is_long)


def _builder_scan_row(strategy, data: pd.DataFrame) -> dict:
    conditions = [condition.model_dump() for condition in strategy.conditions]
    executed = execute_saved_strategy(data, strategy)
    is_long = strategy.direction == "long"

    labels = {item["id"]: item["label"] for item in list_indicators(builder_only=True)}
    rule_preview = format_condition_preview(conditions, labels)
    description = strategy.description.strip() or rule_preview

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
        "trade_pnl": round(float(executed["TradePnL"].iloc[-1]) * 100, 2),
        "kelly": compute_kelly(executed),
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

    if symbol_frames is None:
        yf_symbols = [
            symbol + "=F" if symbol in ["NQ", "ES", "RTY", "CL", "GC", "SI", "HG"] else symbol
            for symbol in SCAN_SYMBOLS
        ]
        symbol_mapping = dict(zip(SCAN_SYMBOLS, yf_symbols))
        if full_data is None:
            full_data = dt.get_bulk_data(yf_symbols, years=1)

        symbol_frames = {}
        for symbol, yf_symbol in symbol_mapping.items():
            if symbol in SKIP_SCAN_SYMBOLS:
                continue
            symbol_frames[symbol] = _prepare_symbol_frame(full_data, symbol, yf_symbol)

    for symbol, data in symbol_frames.items():
        for buy_signal in LEGACY_BUY_SIGNALS:
            row = _legacy_scan_row(symbol, buy_signal, data)
            if row is not None:
                rows.append(row)

    for strategy in list_strategies():
        if strategy.symbol not in symbol_frames:
            continue
        rows.append(_builder_scan_row(strategy, symbol_frames[strategy.symbol]))

    rows.sort(key=_scan_sort_key)
    return rows
