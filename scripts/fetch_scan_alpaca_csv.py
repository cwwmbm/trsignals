#!/usr/bin/env python3
"""
Fetch Scan-universe daily bars from Alpaca (ETFs) + ^VIX from yfinance.

Writes a yfinance-compatible MultiIndex CSV (Open/High/Low/Close/Volume x symbol)
excluding today's Eastern calendar date so a partial session is never baked in.

Usage:
  python scripts/fetch_scan_alpaca_csv.py
  python scripts/fetch_scan_alpaca_csv.py --out Alpaca.Market/scan_bulk_yf.csv --years 1
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ALPACA_SYMBOLS = [
    "SPY",
    "SMH",
    "QQQ",
    "SOXX",
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
VIX_SYMBOL = "^VIX"
DATA_HOST = "https://data.alpaca.markets"
DEFAULT_OUT = _ROOT / "Alpaca.Market" / "scan_bulk_yf.csv"
ENV_PATH = _ROOT / "Alpaca.Market" / ".env"
ET = ZoneInfo("America/New_York")


def _load_alpaca_creds(path: Path = ENV_PATH) -> tuple[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Alpaca credentials file: {path}")
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    api_key = values.get("api_key") or values.get("APCA_API_KEY_ID")
    api_secret = values.get("api_secret") or values.get("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise ValueError(f"api_key / api_secret not found in {path}")
    return api_key, api_secret


def _date_window(years: int) -> tuple[str, str, str]:
    """Return (start, end_exclusive_iso for Alpaca, today_et YYYY-MM-DD)."""
    now_et = datetime.now(ET)
    today = now_et.date()
    start = (pd.Timestamp(today) - pd.DateOffset(years=years)).strftime("%Y") + "-01-01"
    # Alpaca end is exclusive; use start of today ET so today's bar is excluded.
    end_exclusive = datetime(today.year, today.month, today.day, tzinfo=ET).isoformat()
    return start, end_exclusive, today.isoformat()


def _alpaca_get(path: str, params: dict, api_key: str, api_secret: str) -> dict:
    query = urllib.parse.urlencode(params, doseq=True)
    url = f"{DATA_HOST}{path}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Alpaca HTTP {exc.code} for {path}: {body}") from exc


def fetch_alpaca_daily(
    symbols: list[str],
    start: str,
    end_exclusive: str,
    api_key: str,
    api_secret: str,
    *,
    feed: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Return {symbol: DataFrame[Open,High,Low,Close,Volume]} indexed by date."""
    frames: dict[str, list[dict]] = {symbol: [] for symbol in symbols}
    page_token: str | None = None
    while True:
        params: dict = {
            "symbols": ",".join(symbols),
            "timeframe": "1Day",
            "start": start,
            "end": end_exclusive,
            "limit": 10000,
            "adjustment": "split",
            "sort": "asc",
        }
        if feed:
            params["feed"] = feed
        if page_token:
            params["page_token"] = page_token
        payload = _alpaca_get("/v2/stocks/bars", params, api_key, api_secret)
        bars_by_symbol = payload.get("bars") or {}
        for symbol, bars in bars_by_symbol.items():
            for bar in bars or []:
                frames.setdefault(symbol, []).append(bar)
        page_token = payload.get("next_page_token")
        if not page_token:
            break

    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        rows = frames.get(symbol) or []
        if not rows:
            out[symbol] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            continue
        frame = pd.DataFrame(rows)
        frame["Date"] = pd.to_datetime(frame["t"], utc=True).dt.tz_convert(ET).dt.normalize()
        frame["Date"] = frame["Date"].dt.tz_localize(None)
        frame = frame.rename(
            columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
        )
        frame = frame[["Date", "Open", "High", "Low", "Close", "Volume"]]
        frame = frame.drop_duplicates(subset=["Date"], keep="last").set_index("Date").sort_index()
        out[symbol] = frame
    return out


def fetch_vix(start: str, today: str) -> pd.DataFrame:
    # yfinance end is exclusive-ish; use tomorrow so we can still drop today explicitly.
    end = (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(VIX_SYMBOL, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError("yfinance returned no ^VIX data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    frame = raw.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]].copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame = frame[frame.index < pd.Timestamp(today)]
    frame.index.name = "Date"
    return frame


def _align_vix_to_equity_calendar(
    vix: pd.DataFrame, equity_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Reindex VIX onto equity dates; forward-fill trailing Yahoo gaps (e.g. missing Jul 24)."""
    if vix.empty or equity_index.empty:
        return vix
    aligned = vix.reindex(equity_index)
    missing = aligned["Close"].isna()
    if missing.any():
        dates = ", ".join(d.date().isoformat() for d in aligned.index[missing])
        print(f"Warning: ^VIX missing on {dates}; forward-filling from prior yfinance close")
        aligned = aligned.ffill()
        # Volume is not meaningful when filled; keep last known or 0.
        if "Volume" in aligned.columns:
            aligned["Volume"] = aligned["Volume"].fillna(0)
    return aligned.dropna(subset=["Close"])


def build_bulk_frame(symbol_frames: dict[str, pd.DataFrame], today: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(today)
    trimmed: dict[str, pd.DataFrame] = {}
    for symbol, frame in symbol_frames.items():
        if frame.empty:
            trimmed[symbol] = frame
            continue
        part = frame[frame.index < cutoff].copy()
        trimmed[symbol] = part

    equity_frames = {s: trimmed[s] for s in ALPACA_SYMBOLS if s in trimmed and not trimmed[s].empty}
    if equity_frames:
        equity_index = pd.DatetimeIndex(sorted(set().union(*(f.index for f in equity_frames.values()))))
        if VIX_SYMBOL in trimmed and not trimmed[VIX_SYMBOL].empty:
            trimmed[VIX_SYMBOL] = _align_vix_to_equity_calendar(trimmed[VIX_SYMBOL], equity_index)

    pieces = []
    for symbol, frame in trimmed.items():
        if frame.empty:
            continue
        renamed = frame.copy()
        renamed.columns = pd.MultiIndex.from_product(
            [renamed.columns, [symbol]], names=["Price", "Ticker"]
        )
        pieces.append(renamed)
    if not pieces:
        raise RuntimeError("No bars fetched for any symbol")
    bulk = pd.concat(pieces, axis=1).sort_index()
    # Stable column order: Price level then ticker (match yfinance-ish grouping).
    price_order = ["Close", "High", "Low", "Open", "Volume"]
    tickers = [s for s in [*ALPACA_SYMBOLS, VIX_SYMBOL] if s in bulk.columns.get_level_values(1)]
    ordered = []
    for price in price_order:
        for ticker in tickers:
            col = (price, ticker)
            if col in bulk.columns:
                ordered.append(col)
    bulk = bulk.reindex(columns=ordered)
    bulk.index.name = "Date"
    return bulk


def _report(bulk: pd.DataFrame, today: str) -> None:
    tickers = list(bulk.columns.get_level_values(1).unique())
    print(f"Today (ET, excluded): {today}")
    print(f"Rows: {len(bulk)}  Symbols: {len(tickers)}")
    close = bulk["Close"]
    print("\nLast date / July 24 check:")
    target = pd.Timestamp("2026-07-24")
    for symbol in tickers:
        series = close[symbol].dropna()
        last = series.index.max() if not series.empty else None
        has_jul24 = target in series.index if not series.empty else False
        print(f"  {symbol:6} last={last.date() if last is not None else 'n/a'}  jul24={has_jul24}")
    missing = [s for s in [*ALPACA_SYMBOLS, VIX_SYMBOL] if s not in tickers]
    if missing:
        print("Missing symbols:", ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output CSV path")
    parser.add_argument("--years", type=int, default=1, help="History window in years (Scan default=1)")
    parser.add_argument(
        "--feed",
        default=None,
        help="Optional Alpaca feed (iex/sip). Omit to use account default.",
    )
    args = parser.parse_args()

    start, end_exclusive, today = _date_window(args.years)
    print(f"Window start={start} end_exclusive={end_exclusive} (today={today} excluded)")

    api_key, api_secret = _load_alpaca_creds()
    feeds_to_try = [args.feed] if args.feed else [None, "iex", "sip"]
    alpaca_frames = None
    last_error: Exception | None = None
    for feed in feeds_to_try:
        try:
            label = feed or "default"
            print(f"Fetching Alpaca bars (feed={label})...")
            alpaca_frames = fetch_alpaca_daily(
                ALPACA_SYMBOLS,
                start,
                end_exclusive,
                api_key,
                api_secret,
                feed=feed,
            )
            nonempty = sum(1 for frame in alpaca_frames.values() if not frame.empty)
            if nonempty == 0:
                raise RuntimeError(f"No Alpaca bars returned for feed={label}")
            print(f"  Got data for {nonempty}/{len(ALPACA_SYMBOLS)} symbols")
            break
        except Exception as exc:  # noqa: BLE001 — try next feed
            last_error = exc
            print(f"  feed={feed or 'default'} failed: {exc}")
            alpaca_frames = None
    if alpaca_frames is None:
        raise RuntimeError(f"Alpaca fetch failed: {last_error}")

    print("Fetching ^VIX from yfinance...")
    vix = fetch_vix(start, today)
    print(f"  VIX rows={len(vix)} last={vix.index.max().date() if len(vix) else 'n/a'}")

    symbol_frames = {**alpaca_frames, VIX_SYMBOL: vix}
    bulk = build_bulk_frame(symbol_frames, today)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bulk.to_csv(args.out)
    print(f"\nWrote {args.out}")
    _report(bulk, today)
    print("\nSymbols needed for today's amend (Alpaca set):")
    print(" ", ", ".join(ALPACA_SYMBOLS))
    print("\nScan CSV mode:")
    print(f'  SCAN_BULK_CSV="{args.out}"  # then restart API / run_scan')
    print("Amend today's bars:")
    print(
        f"  python scripts/amend_scan_bulk_csv.py --date {today} --bars today_bars.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
