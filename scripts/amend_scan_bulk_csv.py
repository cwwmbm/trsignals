#!/usr/bin/env python3
"""
Append one daily OHLCV row per symbol into Alpaca.Market/scan_bulk_yf.csv.

Input CSV columns (header required):
  symbol,open,high,low,close,volume
  # or Date,Symbol,Open,High,Low,Close,Volume

Example:
  python scripts/amend_scan_bulk_csv.py --date 2026-07-27 --bars today_bars.csv

Also fetches ^VIX for that date from yfinance unless --skip-vix.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from getdata import load_bulk_csv  # noqa: E402

DEFAULT_BULK = _ROOT / "Alpaca.Market" / "scan_bulk_yf.csv"
VIX_SYMBOL = "^VIX"


def _normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {c: c.strip().lower() for c in frame.columns}
    frame = frame.rename(columns=rename)
    if "ticker" in frame.columns and "symbol" not in frame.columns:
        frame = frame.rename(columns={"ticker": "symbol"})
    required = ["symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Bars CSV missing columns: {', '.join(missing)}")
    out = frame[required].copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["symbol"] = out["symbol"].replace({"VIX": VIX_SYMBOL})
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[required[1:]].isna().any().any():
        raise ValueError("Bars CSV contains non-numeric OHLCV values")
    return out


def _vix_row(date: str) -> dict | None:
    start = date
    end = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(VIX_SYMBOL, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    raw = raw.rename(columns=str.title)
    row = raw.iloc[-1]
    return {
        "symbol": VIX_SYMBOL,
        "open": float(row["Open"]),
        "high": float(row["High"]),
        "low": float(row["Low"]),
        "close": float(row["Close"]),
        "volume": float(row["Volume"]) if "Volume" in raw.columns else 0.0,
    }


def amend(bulk_path: Path, date: str, bars: pd.DataFrame) -> pd.DataFrame:
    bulk = load_bulk_csv(bulk_path)
    ts = pd.Timestamp(date)
    prices = ["Open", "High", "Low", "Close", "Volume"]
    field_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }

    # Ensure all target symbols exist as columns.
    for _, row in bars.iterrows():
        symbol = row["symbol"]
        for price in prices:
            col = (price, symbol)
            if col not in bulk.columns:
                bulk[col] = pd.NA

    if ts not in bulk.index:
        empty = pd.DataFrame([[pd.NA] * len(bulk.columns)], index=[ts], columns=bulk.columns)
        bulk = pd.concat([bulk, empty]).sort_index()

    for _, row in bars.iterrows():
        symbol = row["symbol"]
        for src, price in field_map.items():
            bulk.loc[ts, (price, symbol)] = float(row[src])

    # Stable-ish column order by price then ticker.
    tickers = sorted(set(bulk.columns.get_level_values(1)), key=lambda s: (s != VIX_SYMBOL, s))
    ordered = [(price, ticker) for price in prices for ticker in tickers if (price, ticker) in bulk.columns]
    bulk = bulk.reindex(columns=ordered)
    bulk.columns = pd.MultiIndex.from_tuples(ordered, names=["Price", "Ticker"])
    bulk.index.name = "Date"
    return bulk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk", type=Path, default=DEFAULT_BULK)
    parser.add_argument("--date", required=True, help="Session date YYYY-MM-DD to append/overwrite")
    parser.add_argument("--bars", type=Path, required=True, help="CSV with symbol,open,high,low,close,volume")
    parser.add_argument("--skip-vix", action="store_true", help="Do not fetch/append ^VIX from yfinance")
    args = parser.parse_args()

    bars = _normalize_bars(pd.read_csv(args.bars))
    if not args.skip_vix and VIX_SYMBOL not in set(bars["symbol"]):
        vix = _vix_row(args.date)
        if vix is None:
            print(f"Warning: no yfinance ^VIX bar for {args.date}; leaving VIX unchanged/empty")
        else:
            bars = pd.concat([bars, pd.DataFrame([vix])], ignore_index=True)
            print(f"Appended ^VIX from yfinance close={vix['close']}")

    updated = amend(args.bulk, args.date, bars)
    updated.to_csv(args.bulk)
    print(f"Updated {args.bulk} for {args.date}")
    print("Symbols written:", ", ".join(sorted(bars["symbol"].unique())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
