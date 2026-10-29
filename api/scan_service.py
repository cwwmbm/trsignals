from __future__ import annotations

import copy
import hashlib
import threading
import time

import numpy as np
import pandas as pd

import backtest as bt
import getdata as dt
import indicators as ind

from api import vehicles
from api.scan_position import position_from_flags, signal_flags
from api.indicator_catalog import list_indicators
from api.proxy_symbol import execute_with_proxy, with_proxy_description
from api.strategy_compiler import (
    compile_buy_mask,
    compile_sell_mask,
    evaluate_condition_snapshot,
    format_condition_preview,
)
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
    "GDX",
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


def _collect_portfolio_symbols(portfolios) -> set[str]:
    from api.portfolio_service import _collect_symbols, _resolve_portfolio_strategies

    symbols: set[str] = set()
    by_id = {item.id: item for item in portfolios}
    visited = set()

    def collect(portfolio):
        if portfolio.id in visited:
            return
        visited.add(portfolio.id)
        if portfolio.strategy_ids:
            strategies = _resolve_portfolio_strategies(list(portfolio.strategy_ids))
            proxy = getattr(portfolio, "proxy_symbol", None)
            symbols.update(_collect_symbols(strategies, proxy.strip().upper() if proxy else None))
        for identity in getattr(portfolio, "portfolio_ids", []):
            if identity in by_id:
                collect(by_id[identity])

    for portfolio in portfolios:
        try:
            collect(portfolio)
        except ValueError:
            continue
    return symbols


def _scan_dataset_symbols(strategies, portfolios) -> set[str]:
    symbols: set[str] = set()
    for strategy in strategies:
        confirm = getattr(strategy, "confirm_symbols", None)
        if confirm:
            symbols.add(strategy.symbol.strip().upper())
            symbols.update(
                symbol.strip().upper()
                for symbol in confirm
                if symbol and str(symbol).strip()
            )
    symbols.update(_collect_portfolio_symbols(portfolios))
    return {symbol for symbol in symbols if symbol and symbol not in SKIP_SCAN_SYMBOLS}


def _scan_download_symbols(strategies, portfolios) -> list[str]:
    builder_symbols = _builder_scan_symbols(strategies)
    portfolio_symbols = _collect_portfolio_symbols(portfolios)
    all_extra = builder_symbols | portfolio_symbols
    extra_symbols = sorted(
        symbol
        for symbol in all_extra
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


def _last_day(frame: pd.DataFrame) -> str | None:
    if frame is None or frame.empty or "Date" not in frame.columns:
        return None
    return pd.Timestamp(frame["Date"].iloc[-1]).strftime("%Y-%m-%d")


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


def _prepare_symbol_frame(
    full_data: pd.DataFrame,
    symbol: str,
    yf_symbol: str,
    *,
    compute_hurst: bool = True,
) -> pd.DataFrame:
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
    return vehicles.stamp(ind.add_indicators(data, compute_hurst=compute_hurst, symbol=symbol), symbol)


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
    buy_now = bool(executed["LongTradeIn"].iloc[-1])
    hold_now = bool(executed["HoldLong"].iloc[-1])
    sell_now = bool(executed["LongTradeOut"].iloc[-1])
    return {
        "id": f"legacy:{symbol}:{signal_name}",
        "source": "legacy",
        "strategy_id": None,
        "symbol": symbol,
        "signal": signal_name,
        "buy_signal": buy_now,
        "hold_long": hold_now,
        "sell_signal": sell_now,
        "position": position_from_flags(buy_now, hold_now, sell_now, as_of=_last_day(executed)),
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
            from api.builder_strategy import (
                apply_primary_entry_filters,
                attach_strategy_entry_ready,
                builder_signal_callable,
                _confirm_strategy_for_cross_symbol,
            )

            labels = {item["id"]: item["label"] for item in list_indicators(builder_only=True)}
            labels.update(
                {
                    f"strategy:{saved.id}": saved.name
                    for saved in list_strategies()
                }
            )
            confirm_strategy, primary_filters = _confirm_strategy_for_cross_symbol(strategy)
            signal = builder_signal_callable(
                confirm_strategy,
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
            attach_strategy_entry_ready(
                frame,
                confirm_strategy,
                strategy_resolver=get_strategy_by_id,
                symbol_data=symbol_data,
            )
            frame = apply_primary_entry_filters(
                frame,
                primary_filters,
                strategy_resolver=get_strategy_by_id,
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
        from api.builder_strategy import _compile_strategy_masks, attach_strategy_entry_ready

        buy, sell = _compile_strategy_masks(
            data,
            strategy,
            strategy_resolver=get_strategy_by_id,
        )
        is_long = strategy.direction == "long"

        frame = data.copy()
        frame["Buy"] = buy
        frame["Sell"] = sell
        attach_strategy_entry_ready(
            frame,
            strategy,
            strategy_resolver=get_strategy_by_id,
        )
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
    condition_snapshot = evaluate_condition_snapshot(
        data,
        conditions,
        strategy_resolver=get_strategy_by_id,
        labels=labels,
    )
    passed_count = sum(1 for item in condition_snapshot if item["passed"])
    sell_conditions = [_condition_dict(condition) for condition in strategy.sell_conditions]
    sell_condition_snapshot = (
        evaluate_condition_snapshot(
            data,
            sell_conditions,
            strategy_resolver=get_strategy_by_id,
            labels=labels,
        )
        if sell_conditions
        else []
    )
    sell_passed_count = sum(1 for item in sell_condition_snapshot if item["passed"])
    as_of = data["Date"].iloc[-1] if not data.empty and "Date" in data.columns else None
    if as_of is not None:
        as_of = pd.Timestamp(as_of).strftime("%Y-%m-%d")

    buy_now = bool(executed["LongTradeIn"].iloc[-1])
    hold_now = bool(executed["HoldLong"].iloc[-1])
    sell_now = bool(executed["LongTradeOut"].iloc[-1])
    return {
        "id": f"builder:{strategy.id}",
        "source": "builder",
        "strategy_id": strategy.id,
        "symbol": strategy.symbol,
        "signal": strategy.name,
        "buy_signal": buy_now,
        "hold_long": hold_now,
        "sell_signal": sell_now,
        "position": position_from_flags(buy_now, hold_now, sell_now, as_of=as_of),
        "days": int(strategy.hold_days),
        "profit": int(strategy.profit),
        "trade_pnl": _scan_trade_pnl_pct(executed, is_long=strategy.direction == "long"),
        "kelly": compute_kelly(executed),
        "description": description,
        "condition_snapshot": condition_snapshot or None,
        "condition_passed_count": passed_count if condition_snapshot else None,
        "condition_total_count": len(condition_snapshot) if condition_snapshot else None,
        "sell_condition_snapshot": sell_condition_snapshot or None,
        "sell_condition_passed_count": sell_passed_count if sell_condition_snapshot else None,
        "sell_condition_total_count": len(sell_condition_snapshot) if sell_condition_snapshot else None,
        "condition_as_of": as_of,
    }


def _portfolio_scan_row(
    portfolio,
    *,
    bulk_data: pd.DataFrame | None = None,
    symbol_dataset: dict[str, pd.DataFrame] | None = None,
) -> dict | None:
    from api.portfolio_service import build_portfolio_scan_book, scan_position, signal_symbols

    try:
        book = build_portfolio_scan_book(portfolio, bulk_data=bulk_data, symbol_dataset=symbol_dataset)
    except ValueError:
        return None

    frame = book.frame
    if frame.empty:
        return None

    position = scan_position(book)
    symbols = signal_symbols(book.strategies)
    return {
        "id": f"portfolio:{portfolio.id}",
        "source": "portfolio",
        "strategy_id": None,
        "portfolio_id": portfolio.id,
        "symbol": "+".join(symbols),
        "signal": portfolio.name,
        **signal_flags(position["action"], position["held_before"]),
        "position": position,
        "days": int(book.max_hold),
        "profit": int(book.max_profit),
        "trade_pnl": _scan_trade_pnl_pct(frame, is_long=True),
        "kelly": compute_kelly(frame),
        "description": book.description,
    }


def exact_portfolio_position(portfolio_id: str) -> dict:
    """The portfolio's position today with its real sizing: the full 25 years, including the regime
    scores a regime-adjusted portfolio reads. Slower than a scan row, so only run on request."""
    from api.portfolio_service import build_portfolio_scan_book, scan_position
    from api.portfolio_store import get_portfolio_by_id

    portfolio = get_portfolio_by_id(portfolio_id)
    if portfolio is None:
        raise ValueError(f"Unknown portfolio: {portfolio_id}")
    book = build_portfolio_scan_book(portfolio, exact=True)
    if book.frame.empty:
        raise ValueError("No data for this portfolio")
    return scan_position(book)


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


def _strategies_need_hurst(strategies) -> bool:
    """True when any saved strategy reads the Hurst column. Nothing else in a scan does."""
    for strategy in strategies:
        for condition in [*strategy.conditions, *strategy.sell_conditions]:
            item = _condition_dict(condition)
            if "Hurst" in (item.get("left"), item.get("right")):
                return True
    return False


# One scan costs seconds and its inputs rarely change between refreshes, so the last result is reused
# while the saved strategies/portfolios and leverage settings are unchanged, the downloaded bars match
# the last scan's (within float noise) and a short clock window has not run out. The clock window
# bounds staleness from inputs the comparison cannot see: the regime calendar and the leverage-vehicle
# price files refresh on their own.
SCAN_CACHE_TTL_SECONDS = 300
SCAN_CACHE_RTOL = 1e-5
_scan_cache: dict = {"signature": None, "data": None, "rows": None, "at": 0.0}
_scan_cache_lock = threading.Lock()


def _file_stamp(path) -> tuple:
    try:
        stat = path.stat()
    except OSError:
        return (None, None)
    return (stat.st_mtime_ns, stat.st_size)


def _scan_signature(strategies, portfolios) -> tuple:
    """What, besides the price bars, the scan result depends on."""
    from api import leverage_settings

    content = hashlib.sha1(repr((strategies, portfolios)).encode()).hexdigest()
    return (content, _file_stamp(leverage_settings.DEFAULT_STORE_PATH))


def _frames_match(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    """Same bars, ignoring the ~1e-7 float noise Yahoo adds between downloads of identical history."""
    try:
        if old.shape != new.shape or not old.index.equals(new.index) or not old.columns.equals(new.columns):
            return False
        return bool(
            np.allclose(
                old.to_numpy(dtype=float),
                new.to_numpy(dtype=float),
                rtol=SCAN_CACHE_RTOL,
                atol=0.0,
                equal_nan=True,
            )
        )
    except (TypeError, ValueError):
        return False


def clear_scan_cache() -> None:
    with _scan_cache_lock:
        _scan_cache.update(signature=None, data=None, rows=None, at=0.0)


def _scan_cache_get(signature: tuple, full_data: pd.DataFrame):
    with _scan_cache_lock:
        if (
            _scan_cache["signature"] == signature
            and _scan_cache["rows"] is not None
            and time.monotonic() - _scan_cache["at"] < SCAN_CACHE_TTL_SECONDS
            and _frames_match(_scan_cache["data"], full_data)
        ):
            return copy.deepcopy(_scan_cache["rows"])
    return None


def _scan_cache_put(signature: tuple, full_data: pd.DataFrame, rows: list[dict]) -> None:
    with _scan_cache_lock:
        _scan_cache.update(
            signature=signature,
            data=full_data.copy(),
            rows=copy.deepcopy(rows),
            at=time.monotonic(),
        )


def run_scan(
    *,
    full_data: pd.DataFrame | None = None,
    symbol_frames: dict[str, pd.DataFrame] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    strategies = list_strategies()
    portfolios = list_portfolios()
    need_hurst = _strategies_need_hurst(strategies)
    cache_signature = None

    if symbol_frames is None:
        download_symbols = _scan_download_symbols(strategies, portfolios)
        yf_symbols = [_yf_symbol(symbol) for symbol in download_symbols]
        symbol_mapping = dict(zip(download_symbols, yf_symbols))
        if full_data is None:
            full_data = dt.get_bulk_data(yf_symbols, years=1)

        cache_signature = _scan_signature(strategies, portfolios)
        cached = _scan_cache_get(cache_signature, full_data)
        if cached is not None:
            return cached

        symbol_frames = {}
        for symbol, yf_symbol in symbol_mapping.items():
            if symbol in SKIP_SCAN_SYMBOLS:
                continue
            symbol_frames[symbol] = _prepare_symbol_frame(
                full_data, symbol, yf_symbol, compute_hurst=need_hurst
            )

    symbol_dataset: dict | None = None
    if full_data is not None:
        dataset_symbols = _scan_dataset_symbols(strategies, portfolios)
        if dataset_symbols:
            symbol_dataset = bt.build_symbol_dataset(
                full_data, sorted(dataset_symbols), compute_hurst=need_hurst
            )

    for symbol, data in symbol_frames.items():
        if symbol not in LEGACY_SCAN_SYMBOLS:
            continue
        for buy_signal in LEGACY_BUY_SIGNALS:
            row = _legacy_scan_row(symbol, buy_signal, data)
            if row is not None:
                rows.append(row)

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

    for portfolio in portfolios:
        row = _portfolio_scan_row(
            portfolio,
            bulk_data=full_data,
            symbol_dataset=symbol_dataset,
        )
        if row is not None:
            rows.append(row)

    rows.sort(key=_scan_sort_key)
    if cache_signature is not None:
        _scan_cache_put(cache_signature, full_data, rows)
    return rows
