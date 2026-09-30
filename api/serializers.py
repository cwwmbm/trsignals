import math
import re
from typing import Any

import numpy as np
import pandas as pd

from api.market_regimes import (
    atr_regime_from_row,
    compute_market_regime_sharpe,
    current_market_regimes,
    load_regime_calendar,
    regime_coverage,
    market_regimes_for_timestamp,
)
from stats import compute_aggregate_metrics, monthly_performance, yearly_performance

MAX_EQUITY_CURVE_POINTS = 2000


def _clean_number(value: Any) -> Any:
    if isinstance(value, list):
        return [_clean_number(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_number(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return stripped
        if stripped.endswith("%"):
            try:
                return float(stripped[:-1].replace(",", ""))
            except ValueError:
                return value
        if "$" in stripped:
            cleaned = re.sub(r"[$,]", "", stripped)
            try:
                return float(cleaned)
            except ValueError:
                return value
    return value


def dataframe_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for row in df.to_dict(orient="records"):
        records.append({key: _clean_number(value) for key, value in row.items()})
    return records


def _format_timestamp(value, *, is_intraday: bool) -> str:
    ts = pd.to_datetime(value)
    if is_intraday and (ts.hour != 0 or ts.minute != 0 or ts.second != 0):
        return ts.strftime("%Y-%m-%d %H:%M")
    return ts.strftime("%Y-%m-%d")


def summary_payload(
    data: pd.DataFrame,
    days: int,
    profit: int,
    description: str,
    *,
    periods_per_year: int = 252,
) -> dict:
    metrics = compute_aggregate_metrics(data, periods_per_year=periods_per_year)
    return {
        "description": description,
        "days": days,
        "profit": profit,
        "excluded_year": metrics["excluded_year"],
        "rolling_pnl": metrics["rolling_pnl"],
        "max_drawdown": metrics["max_drawdown"],
        "trades": metrics["trades"],
        "pct_positive": metrics["pct_positive"],
        "avg_win": metrics["avg_win"],
        "avg_loss": metrics["avg_loss"],
        "kelly": metrics["kelly"],
        "cagr_decimal": metrics["cagr_decimal"],
        "cagr_percent": metrics["cagr_percent"],
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
    }


def yearly_payload(data: pd.DataFrame) -> list[dict]:
    yearly = yearly_performance(data).reset_index()
    yearly = yearly.rename(
        columns={
            "Date": "year",
            "PnL%": "pnl_percent",
            "Drawdown%": "drawdown_percent",
            "Num_Trades": "num_trades",
            "Positive_Trades": "positive_trades",
        }
    )
    return dataframe_records(yearly)


def monthly_payload(data: pd.DataFrame) -> list[dict]:
    monthly = monthly_performance(data).reset_index()
    monthly = monthly.rename(
        columns={
            "Date": "month",
            "PnL%": "pnl_percent",
            "Drawdown%": "drawdown_percent",
            "Num_Trades": "num_trades",
            "Positive_Trades": "positive_trades",
        }
    )
    if "month" in monthly.columns:
        monthly["month"] = monthly["month"].astype(str)
    return dataframe_records(monthly)


def _downsample_for_chart(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    indices = np.linspace(0, len(df) - 1, max_points, dtype=int)
    return df.iloc[indices]


def equity_curve_payload(
    data: pd.DataFrame,
    *,
    is_intraday: bool = False,
    max_points: int = MAX_EQUITY_CURVE_POINTS,
) -> list[dict]:
    rows = data[["Date", "RollingPnL", "Drawdown"]].copy()
    rows = _downsample_for_chart(rows, max_points)
    rows["Date"] = rows["Date"].apply(lambda value: _format_timestamp(value, is_intraday=is_intraday))
    rows = rows.rename(
        columns={
            "Date": "date",
            "RollingPnL": "rolling_pnl",
            "Drawdown": "drawdown",
        }
    )
    return dataframe_records(rows)


def _with_regimes(
    record: dict,
    entry: pd.Series,
    *,
    include_regimes: bool,
    regime_calendar: pd.DataFrame | None,
) -> dict:
    if not include_regimes:
        return record
    record.update(market_regimes_for_timestamp(entry["Date"], regime_calendar))
    record["atr_regime"] = atr_regime_from_row(entry)
    return record


def trade_payload(
    data: pd.DataFrame,
    *,
    is_intraday: bool = False,
    portfolio_equity: bool = False,
    include_regimes: bool = False,
    regime_calendar: pd.DataFrame | None = None,
) -> list[dict]:
    trades = []
    entry = None
    for _, row in data.iterrows():
        if bool(row.get("LongTradeIn", False)):
            entry = row
        if bool(row.get("LongTradeOut", False)) and entry is not None:
            if portfolio_equity:
                entry_price = float(entry["TradeEntry"])
                exit_price = float(row["RollingPnL"])
            else:
                entry_price = float(entry["Close"])
                exit_price = float(row["Close"])
            trades.append(
                _with_regimes(
                    {
                        "entry_date": _format_timestamp(entry["Date"], is_intraday=is_intraday),
                        "exit_date": _format_timestamp(row["Date"], is_intraday=is_intraday),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "trade_pnl": float(row["TradePnL"]),
                        "days_in_trade": int(row.get("DaysInTrade", 0)),
                        "status": "Closed",
                    },
                    entry,
                    include_regimes=include_regimes,
                    regime_calendar=regime_calendar,
                )
            )
            entry = None
    if entry is not None and bool(data.iloc[-1].get("HoldLong", False)):
        last = data.iloc[-1]
        if portfolio_equity:
            entry_price = float(entry["TradeEntry"])
            exit_price = float(last["RollingPnL"])
        else:
            entry_price = float(entry["Close"])
            exit_price = float(last["Close"])
        trades.append(
            _with_regimes(
                {
                    "entry_date": _format_timestamp(entry["Date"], is_intraday=is_intraday),
                    "exit_date": "Open",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "trade_pnl": float(last["TradePnL"]),
                    "days_in_trade": int(last.get("DaysInTrade", 0)),
                    "status": "Open",
                },
                entry,
                include_regimes=include_regimes,
                regime_calendar=regime_calendar,
            )
        )
    return trades


def detailed_backtest_payload(
    data: pd.DataFrame,
    days: int,
    profit: int,
    description: str,
    *,
    periods_per_year: int = 252,
    is_intraday: bool = False,
    portfolio_equity: bool = False,
    years: int | None = None,
) -> dict:
    total_bars = int(data.shape[0])
    equity_curve = equity_curve_payload(data, is_intraday=is_intraday)
    regime_calendar = load_regime_calendar(years) if years is not None else None
    payload = {
        "summary": summary_payload(
            data,
            days,
            profit,
            description,
            periods_per_year=periods_per_year,
        ),
        "yearly": yearly_payload(data),
        "monthly": monthly_payload(data),
        "equity_curve": equity_curve,
        "trades": trade_payload(
            data,
            is_intraday=is_intraday,
            portfolio_equity=portfolio_equity,
            include_regimes=years is not None,
            regime_calendar=regime_calendar,
        ),
        "equity_curve_total_points": total_bars,
        "equity_curve_shown_points": len(equity_curve),
    }
    if years is not None:
        payload["market_regime_sharpe"] = compute_market_regime_sharpe(
            data,
            regime_calendar,
            periods_per_year=periods_per_year,
        )
        payload["market_regime_current"] = current_market_regimes(data, regime_calendar)
        payload["market_regime_coverage"] = regime_coverage(data, regime_calendar)
    return payload
