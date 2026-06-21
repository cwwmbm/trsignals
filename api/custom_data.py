from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

import indicators as ind
from api.indicator_catalog import INDICATOR_CATALOG, custom_data_only_indicator_ids
from api.session_masks import _to_eastern_timestamps, infer_market_timestamp_timezone
from getdata import normalize_dataframe

COMMON_INTERVAL_MINUTES = [1, 2, 3, 5, 10, 15, 30, 60, 120, 240]
MAX_STORED_DATASETS = 5
CUSTOM_INTRADAY_BACKTEST_LIMIT = 5000

MARKET_CONTEXT_COLUMN_IDS = [
    "Spybull",
    "Vix",
    "Spy",
    "Qqq",
    "Soxx",
    "Iwm",
    "Uvxy",
    "Sqqq",
    "Breadth",
    "Riskbreadth",
    "Semisbreadth",
    "Financialsbreadth",
    "Energybreadth",
    "Utilitiesbreadth",
    "Industrialsbreadth",
    "Goldbreadth",
    "Bondbreadth",
    "Iwmbreadth",
]


def unavailable_indicator_ids() -> list[str]:
    ids = list(MARKET_CONTEXT_COLUMN_IDS)
    for item in INDICATOR_CATALOG:
        if not item.get("builderEligible", True):
            continue
        kind = item.get("kind")
        if kind == "breadth":
            ids.append(item["id"])
    if "SPYBull" not in ids:
        ids.append("SPYBull")
    return sorted(set(ids))


def _column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {column.lower(): column for column in df.columns}


def _resolve_column(df: pd.DataFrame, names: list[str]) -> str | None:
    lookup = _column_lookup(df)
    for name in names:
        match = lookup.get(name.lower())
        if match is not None:
            return match
    return None


def _symbol_from_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    stem = filename.rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0]
    match = re.match(r"^([A-Za-z][A-Za-z0-9._=-]*)", stem)
    if not match:
        return None
    token = match.group(1).split("_")[0].split("-")[0].upper()
    return token if token and token != "CUSTOM" else None


def parse_intraday_csv(
    content: bytes,
    *,
    filename: str | None = None,
) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(content))
    if raw.empty:
        raise ValueError("CSV file is empty")

    timestamp_col = _resolve_column(raw, ["timestamp", "date", "datetime", "time"])
    if timestamp_col is None:
        raise ValueError("CSV must include a timestamp or date column")

    rename_map: dict[str, str] = {timestamp_col: "Date"}
    for source, target in [
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
        ("vwap", "VWAP"),
    ]:
        resolved = _resolve_column(raw, [source])
        if resolved is not None:
            rename_map[resolved] = target

    df = raw.rename(columns=rename_map)
    required = ["Date", "Open", "High", "Low", "Close"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    symbol_col = _resolve_column(raw, ["symbol"])
    if symbol_col is not None and raw[symbol_col].notna().any():
        df.attrs["symbol"] = str(raw[symbol_col].dropna().iloc[0]).strip().upper()
    else:
        inferred = _symbol_from_filename(filename)
        df.attrs["symbol"] = inferred or "CUSTOM"

    symbol_data_col = _resolve_column(df, ["symbol"])
    if symbol_data_col is not None:
        df = df.drop(columns=[symbol_data_col])

    df = normalize_dataframe(df)
    if "Vwap" in df.columns:
        df = df.rename(columns={"Vwap": "VWAP"})
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    if df.empty:
        raise ValueError("No valid OHLC rows found after parsing")

    df.attrs["timezone"] = infer_market_timestamp_timezone(df["Date"])
    return df


def detect_interval(df: pd.DataFrame) -> tuple[int, str, int]:
    diffs = df["Date"].diff().dropna()
    positive = diffs[diffs > pd.Timedelta(0)]
    if positive.empty:
        raise ValueError("Could not detect bar interval from timestamps")

    median_minutes = positive.dt.total_seconds().median() / 60.0
    interval_minutes = min(COMMON_INTERVAL_MINUTES, key=lambda value: abs(value - median_minutes))

    session_dates = df["Date"].dt.date
    bars_per_session = df.groupby(session_dates).size().median()
    periods_per_year = max(1, int(round(bars_per_session * 252)))

    if interval_minutes >= 60 and interval_minutes % 60 == 0:
        label = f"{interval_minutes // 60}h"
    else:
        label = f"{interval_minutes}min"

    return interval_minutes, label, periods_per_year


def has_vwap_column(df: pd.DataFrame) -> bool:
    column = _resolve_column(df, ["vwap", "VWAP", "Vwap"])
    return column is not None and df[column].notna().any()


def stub_market_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    for column_id in MARKET_CONTEXT_COLUMN_IDS:
        if column_id not in frame.columns:
            frame[column_id] = np.nan
    return frame


INDICATOR_READY_COLUMN = "RSI2"


def prepare_custom_intraday_frame(df: pd.DataFrame, periods_per_year: int) -> pd.DataFrame:
    frame = stub_market_context_columns(df)
    timezone = str(df.attrs.get("timezone", "UTC"))
    return ind.add_indicators(frame, periods_per_year=periods_per_year, source_timezone=timezone)


def is_indicators_prepared(df: pd.DataFrame) -> bool:
    return INDICATOR_READY_COLUMN in df.columns


def slice_intraday_backtest_window(
    data: pd.DataFrame,
    *,
    backtest_all_data: bool,
    limit: int = CUSTOM_INTRADAY_BACKTEST_LIMIT,
) -> pd.DataFrame:
    """Return the OHLC slice to backtest before indicator calculation."""
    if backtest_all_data or len(data) <= limit:
        return data.copy()
    sliced = data.iloc[-limit:].copy()
    if "timezone" in data.attrs:
        sliced.attrs["timezone"] = data.attrs["timezone"]
    return sliced


def pad_session_warmup(full_data: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
    """Prepend earlier same-session bars so session-cumulative VWAP std bands are correct."""
    if window.empty or full_data.empty:
        return window.copy()

    timezone = str(full_data.attrs.get("timezone", "UTC"))
    eastern_full = _to_eastern_timestamps(full_data["Date"], timezone)
    first_eastern = _to_eastern_timestamps(window["Date"].iloc[:1], timezone).iloc[0]
    session_date = first_eastern.date()

    session_mask = eastern_full.dt.date == session_date
    session_bars = full_data.loc[session_mask]
    window_start = window["Date"].iloc[0]
    earlier = session_bars[session_bars["Date"] < window_start]

    if earlier.empty:
        padded = window.copy()
    else:
        padded = pd.concat([earlier, window], ignore_index=False)
        padded = padded.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    if "timezone" in full_data.attrs:
        padded.attrs["timezone"] = full_data.attrs["timezone"]
    return padded


@dataclass
class CustomDataset:
    id: str
    symbol: str
    interval_minutes: int
    interval_label: str
    periods_per_year: int
    start: datetime
    end: datetime
    row_count: int
    unavailable_indicator_ids: list[str]
    has_vwap: bool = False
    timezone: str = "UTC"
    is_intraday: bool = True
    data: pd.DataFrame | None = None
    prepared_data: pd.DataFrame | None = None


class CustomDatasetStore:
    def __init__(self, max_datasets: int = MAX_STORED_DATASETS) -> None:
        self._max_datasets = max_datasets
        self._datasets: dict[str, CustomDataset] = {}
        self._order: list[str] = []

    def _evict_if_needed(self) -> None:
        while len(self._datasets) > self._max_datasets:
            oldest_id = self._order.pop(0)
            self._datasets.pop(oldest_id, None)

    def add_from_csv(
        self,
        content: bytes,
        *,
        filename: str | None = None,
    ) -> CustomDataset:
        parsed = parse_intraday_csv(content, filename=filename)
        interval_minutes, interval_label, periods_per_year = detect_interval(parsed)
        frame = stub_market_context_columns(parsed)
        symbol = str(parsed.attrs.get("symbol", "CUSTOM")).upper()
        timezone = str(parsed.attrs.get("timezone", "UTC"))
        dataset_id = str(uuid.uuid4())

        dataset = CustomDataset(
            id=dataset_id,
            symbol=symbol,
            interval_minutes=interval_minutes,
            interval_label=interval_label,
            periods_per_year=periods_per_year,
            start=frame["Date"].iloc[0],
            end=frame["Date"].iloc[-1],
            row_count=int(frame.shape[0]),
            unavailable_indicator_ids=unavailable_indicator_ids(),
            has_vwap=has_vwap_column(parsed),
            timezone=timezone,
            data=frame,
        )
        self._datasets[dataset_id] = dataset
        self._order.append(dataset_id)
        self._evict_if_needed()
        return dataset

    def ensure_prepared(self, dataset_id: str) -> CustomDataset:
        dataset = self.require(dataset_id)
        if dataset.data is None:
            raise ValueError(f"Custom dataset has no data: {dataset_id}")
        if dataset.prepared_data is not None and is_indicators_prepared(dataset.prepared_data):
            return dataset
        dataset.prepared_data = prepare_custom_intraday_frame(
            dataset.data.copy(),
            dataset.periods_per_year,
        )
        dataset.prepared_data.attrs["timezone"] = dataset.timezone
        return dataset

    def load_backtest_frame(self, dataset_id: str, *, backtest_all_data: bool) -> pd.DataFrame:
        dataset = self.require(dataset_id)
        if dataset.data is None:
            raise ValueError(f"Custom dataset has no data: {dataset_id}")

        if backtest_all_data:
            self.ensure_prepared(dataset_id)
            frame = dataset.prepared_data.copy()
        else:
            window = slice_intraday_backtest_window(
                dataset.data,
                backtest_all_data=False,
            )
            padded = pad_session_warmup(dataset.data, window)
            prepared = prepare_custom_intraday_frame(padded, dataset.periods_per_year)
            window_dates = set(window["Date"])
            frame = prepared[prepared["Date"].isin(window_dates)].copy()

        frame.attrs["timezone"] = dataset.timezone
        return frame

    def get(self, dataset_id: str) -> CustomDataset | None:
        return self._datasets.get(dataset_id)

    def require(self, dataset_id: str) -> CustomDataset:
        dataset = self.get(dataset_id)
        if dataset is None:
            raise ValueError(f"Custom dataset not found: {dataset_id}")
        return dataset

    def delete(self, dataset_id: str) -> bool:
        if dataset_id not in self._datasets:
            return False
        self._datasets.pop(dataset_id, None)
        self._order = [item for item in self._order if item != dataset_id]
        return True

    def metadata(self, dataset: CustomDataset) -> dict[str, Any]:
        return {
            "id": dataset.id,
            "symbol": dataset.symbol,
            "interval_minutes": dataset.interval_minutes,
            "interval_label": dataset.interval_label,
            "periods_per_year": dataset.periods_per_year,
            "start": pd.Timestamp(dataset.start).isoformat(),
            "end": pd.Timestamp(dataset.end).isoformat(),
            "row_count": dataset.row_count,
            "unavailable_indicator_ids": dataset.unavailable_indicator_ids,
            "has_vwap": dataset.has_vwap,
            "custom_data_only_indicator_ids": custom_data_only_indicator_ids(),
            "timezone": dataset.timezone,
            "is_intraday": dataset.is_intraday,
        }


custom_dataset_store = CustomDatasetStore()
