import math
import re
from typing import Any

import pandas as pd

from stats import compute_aggregate_metrics, yearly_performance


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


def summary_payload(data: pd.DataFrame, days: int, profit: int, description: str) -> dict:
    metrics = compute_aggregate_metrics(data)
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


def equity_curve_payload(data: pd.DataFrame) -> list[dict]:
    rows = data[["Date", "RollingPnL", "Drawdown"]].copy()
    rows["Date"] = pd.to_datetime(rows["Date"]).dt.strftime("%Y-%m-%d")
    rows = rows.rename(
        columns={
            "Date": "date",
            "RollingPnL": "rolling_pnl",
            "Drawdown": "drawdown",
        }
    )
    return dataframe_records(rows)


def trade_payload(data: pd.DataFrame) -> list[dict]:
    trades = []
    entry = None
    for _, row in data.iterrows():
        if bool(row.get("LongTradeIn", False)):
            entry = row
        if bool(row.get("LongTradeOut", False)) and entry is not None:
            trades.append(
                {
                    "entry_date": pd.to_datetime(entry["Date"]).strftime("%Y-%m-%d"),
                    "exit_date": pd.to_datetime(row["Date"]).strftime("%Y-%m-%d"),
                    "entry_price": float(entry["Close"]),
                    "exit_price": float(row["Close"]),
                    "trade_pnl": float(row["TradePnL"]),
                    "days_in_trade": int(row.get("DaysInTrade", 0)),
                    "status": "Closed",
                }
            )
            entry = None
    if entry is not None and bool(data.iloc[-1].get("HoldLong", False)):
        last = data.iloc[-1]
        trades.append(
            {
                "entry_date": pd.to_datetime(entry["Date"]).strftime("%Y-%m-%d"),
                "exit_date": "Open",
                "entry_price": float(entry["Close"]),
                "exit_price": float(last["Close"]),
                "trade_pnl": float(last["TradePnL"]),
                "days_in_trade": int(last.get("DaysInTrade", 0)),
                "status": "Open",
            }
        )
    return trades


def detailed_backtest_payload(
    data: pd.DataFrame,
    days: int,
    profit: int,
    description: str,
) -> dict:
    return {
        "summary": summary_payload(data, days, profit, description),
        "yearly": yearly_payload(data),
        "equity_curve": equity_curve_payload(data),
        "trades": trade_payload(data),
    }
