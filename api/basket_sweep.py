"""Run one draft strategy on every symbol in a basket.

The question this answers: a strategy looks good on AAPL, does it hold up on other stocks? Each
symbol runs the same conditions, hold and exits over its full history, through the same loader and
engine as the Backtest button, so the draft's own symbol reproduces the Backtest numbers.

Each row also carries a shifted-signal null (``NullSharpe``, ``NullP``): the Sharpe the same entry
signals would earn at the wrong times on that symbol. A stock that rose a lot looks good under any
timing, so beating this is the evidence that the entries add something there. Without numba the null
is skipped and those columns are empty.

Per symbol the strategy is adapted, because it was written for another instrument:
* a confirm symbol equal to the basket symbol is dropped (a symbol never confirms itself);
* the proxy symbol is ignored, since it names the instrument traded for the draft's symbol.
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd

import backtest as bt
from api import fast_sweep
from api.builder_strategy import prepare_builder_refine_frame
from api.strategy_store import _normalize_confirm_symbols
from backtest_runners import load_ticker_data

MAX_BASKET_SYMBOLS = 40
MIN_BARS = 250
NULL_SHIFTS = 300
NULL_ALPHA = 0.05

# The loaders download through yfinance, which loses columns when two downloads overlap.
_LOAD_LOCK = threading.Lock()


def normalize_basket(symbols, draft_symbol: str) -> list[str]:
    """Upper-cased, de-duplicated symbols, without the draft's own symbol."""
    draft = draft_symbol.strip().upper()
    seen: list[str] = []
    for item in symbols or []:
        for part in str(item).replace(",", " ").split():
            value = part.strip().upper()
            if value and value != draft and value not in seen:
                seen.append(value)
    if len(seen) > MAX_BASKET_SYMBOLS:
        raise ValueError(f"A basket can hold at most {MAX_BASKET_SYMBOLS} symbols")
    return seen


def strategy_for_symbol(strategy, symbol: str):
    """The draft strategy retargeted to ``symbol``."""
    update = {
        "symbol": symbol,
        "confirm_symbols": _normalize_confirm_symbols(symbol, list(strategy.confirm_symbols or [])),
        "proxy_symbol": None,
    }
    clone = getattr(strategy, "model_copy", None) or strategy.copy
    return clone(update=update)


def _load_frame(variant, years: int, in_sample_end=None):
    """The executable frame for one symbol: (frame, days, profit, is_long, pnl_column)."""
    from api.sample_window import slice_frame_to_end, slice_symbol_data_to_end
    from api.strategy_store import get_strategy_by_id

    symbol = variant.symbol
    try:
        with _LOAD_LOCK:
            if variant.confirm_symbols:
                symbol_data = bt.load_symbol_dataset([symbol, *variant.confirm_symbols], years=years)
                data = None
            else:
                symbol_data = None
                data = load_ticker_data(symbol, years=years)
    except Exception as exc:  # noqa: BLE001 - yfinance/pandas raise unhelpful errors for unknown tickers
        raise ValueError(f"Could not load price data for {symbol}") from exc
    if in_sample_end is not None:
        if data is not None:
            data = slice_frame_to_end(data, in_sample_end)
        if symbol_data is not None:
            symbol_data = slice_symbol_data_to_end(symbol_data, in_sample_end)
    return prepare_builder_refine_frame(
        variant,
        years=years,
        data=data,
        symbol_data=symbol_data,
        strategy_resolver=get_strategy_by_id,
        labels=_labels(),
    )


_label_cache: dict | None = None


def _labels() -> dict[str, str]:
    global _label_cache
    if _label_cache is None:
        from api.services import _builder_condition_labels

        _label_cache = _builder_condition_labels()
    return _label_cache


def symbol_metrics(strategy, symbol: str, *, years: int, in_sample_end=None, include_null: bool = False) -> dict:
    """Ranking metrics (with ``Yearly``) for the strategy on one symbol, plus the null when asked."""
    variant = strategy_for_symbol(strategy, symbol)
    frame, days, profit, is_long, pnl_col = _load_frame(variant, years, in_sample_end)
    if len(frame) < MIN_BARS:
        raise ValueError(f"Only {len(frame)} bars of history")
    from api import fast_metrics

    metrics = fast_metrics.run_metrics(frame.copy(), days, profit, is_long, pnl_column=pnl_col, include_yearly=True)
    metrics["_bars"] = len(frame)
    metrics["_start"] = pd.Timestamp(frame["Date"].iloc[0]).strftime("%Y-%m-%d")
    metrics["_end"] = pd.Timestamp(frame["Date"].iloc[-1]).strftime("%Y-%m-%d")
    if include_null and fast_sweep.NUMBA_AVAILABLE and days > 0:
        try:
            metrics["_null"] = fast_sweep.shift_null(
                frame, days, profit, is_long, pnl_column=pnl_col, shifts=NULL_SHIFTS
            )
        except ValueError:
            metrics["_null"] = None
    return metrics


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _round_or_none(value, digits: int):
    number = _num(value)
    return None if not np.isfinite(number) else round(number, digits)


def _percent(value) -> float:
    try:
        return float(str(value).rstrip("%"))
    except (TypeError, ValueError):
        return float("nan")


def run_basket_sweep(strategy, symbols, *, years: int = 25) -> tuple[list[dict], dict]:
    """Evaluate ``strategy`` on its own symbol and on every basket symbol.

    Returns (rows, summary): the draft symbol first, then the basket ranked by Sharpe, then symbols
    that failed. A symbol that cannot be loaded or has too little history becomes a row with ``Error``
    set; it never fails the sweep.
    """
    draft_symbol = strategy.symbol.strip().upper()
    basket = normalize_basket(symbols, draft_symbol)
    if not basket:
        raise ValueError("Add at least one symbol to the basket")

    ordered = [(draft_symbol, "Draft symbol"), *[(s, "Basket") for s in basket]]
    good: list[dict] = []
    failed: list[dict] = []
    raw_rows: list[dict] = []
    for symbol, source in ordered:
        try:
            m = symbol_metrics(strategy, symbol, years=years, include_null=True)
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not sink the sweep
            failed.append({"symbol": symbol, "source": source, "error": str(exc)[:200] or type(exc).__name__})
            continue
        null = m.get("_null") or {}
        row = {
            "Symbol": symbol,
            "Source": source,
            "Trades": m["Trades"],
            "%Pstv": m["%Pstv"],
            "Sharpe": m["Sharpe"],
            "NullSharpe": null.get("null_median"),
            "NullP": null.get("p"),
            "CAGR": m["CAGR"],
            "MaxDD": m["MaxDD"],
            "PnL": m["PnL"],
            "Sortino": m["Sortino"],
            "From": m["_start"],
            "Yearly": m["Yearly"],
        }
        good.append(row)
        raw_rows.append({"source": source, "sharpe": _num(m["Sharpe"]), "cagr": _percent(m["CAGR"]), "p": null.get("p"), "trades": m["Trades"]})

    rows: list[dict] = []
    if good:
        frame = pd.DataFrame(good)
        formatted = bt._format_ranking_results(
            frame.drop(columns=["NullSharpe", "NullP", "From", "Yearly", "Source", "Symbol"])
        )
        # _format_ranking_results sorts by Sharpe and formats the metric columns; carry the rest across.
        frame = frame.loc[formatted.index].copy()
        for column in formatted.columns:
            frame[column] = formatted[column]
        columns = ["Symbol", "Source", "Trades", "%Pstv", "Sharpe", "NullSharpe", "NullP", "CAGR", "MaxDD", "PnL", "Sortino", "From", "Yearly"]
        for record in frame[columns].to_dict(orient="records"):
            record["Error"] = None
            record["Trades"] = int(record["Trades"])
            record["NullSharpe"] = _round_or_none(record["NullSharpe"], 2)
            record["NullP"] = _round_or_none(record["NullP"], 3)
            rows.append(record)
        # The draft symbol is the reference everything else is read against, so it leads.
        rows.sort(key=lambda r: 0 if r["Source"] == "Draft symbol" else 1)
    blank = {key: None for key in ("Trades", "%Pstv", "Sharpe", "NullSharpe", "NullP", "CAGR", "MaxDD", "PnL", "Sortino", "From", "Yearly")}
    rows += [{"Symbol": item["symbol"], "Source": item["source"], **blank, "Error": item["error"]} for item in failed]

    basket_rows = [r for r in raw_rows if r["source"] == "Basket"]
    draft_rows = [r for r in raw_rows if r["source"] == "Draft symbol"]
    with_null = [r for r in basket_rows if r["p"] is not None and np.isfinite(r["p"])]
    sharpes = [r["sharpe"] for r in basket_rows if np.isfinite(r["sharpe"])]
    summary = {
        "draft_symbol": draft_symbol,
        "requested": len(basket),
        "evaluated": len(basket_rows),
        "failed": failed,
        "draft_sharpe": draft_rows[0]["sharpe"] if draft_rows and np.isfinite(draft_rows[0]["sharpe"]) else None,
        "median_sharpe": float(np.median(sharpes)) if sharpes else None,
        "share_sharpe_positive": float(np.mean([s > 0 for s in sharpes])) if sharpes else None,
        "share_cagr_positive": float(np.mean([r["cagr"] > 0 for r in basket_rows if np.isfinite(r["cagr"])])) if basket_rows else None,
        "share_beating_null": float(np.mean([r["p"] <= NULL_ALPHA for r in with_null])) if with_null else None,
        "null_evaluated": len(with_null),
        "null_alpha": NULL_ALPHA,
        "null_shifts": NULL_SHIFTS,
        "ignored_proxy_symbol": strategy.proxy_symbol,
        "confirm_symbols": list(strategy.confirm_symbols or []),
    }
    return rows, summary
