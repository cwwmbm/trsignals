from __future__ import annotations

import backtest as bt
import pandas as pd

from backtest_runners import attach_proxy_column


def proxy_column(strategy) -> str | None:
    proxy = getattr(strategy, "proxy_symbol", None)
    if not proxy:
        return None
    value = str(proxy).strip().upper()
    if not value:
        return None
    primary = str(strategy.symbol).strip().upper()
    if value == primary:
        return None
    return value


def proxy_description_tag(strategy) -> str | None:
    pnl_col = proxy_column(strategy)
    if not pnl_col:
        return None
    primary = str(strategy.symbol).strip().upper()
    return f"[{primary} signals, {pnl_col} trade]"


def with_proxy_description(description: str, strategy) -> str:
    tag = proxy_description_tag(strategy)
    if not tag:
        return description
    if tag in description:
        return description
    return f"{tag} {description}".strip()


def execute_with_proxy(
    frame: pd.DataFrame,
    strategy,
    days: int,
    profit: int,
    is_long: bool,
    *,
    years: int = 25,
) -> pd.DataFrame:
    pnl_col = proxy_column(strategy)
    if pnl_col:
        frame = attach_proxy_column(frame, pnl_col, years=years)
    return bt.execute_strategy(frame, days, profit, is_long, pnl_column=pnl_col)
