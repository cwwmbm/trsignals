"""Incremental regime scores: one pass over a book's history, scored at any bars you name.

``market_regimes.score_regime_book`` scores a finished backtest. Regime-based allocation needs
the same scores *as they stood on each decision day*, using only the history up to that day.
Re-scoring from scratch at every decision is quadratic (about 0.4 s per decision per book, so
minutes per backtest). Every ingredient of the score can be kept as a running statistic, so a
single pass does it in about a second:

- Sortino (over invested bars): running sums of the returns and of the squared losses
- average trade: running sum and count of closed-trade returns
- max drawdown: a running peak of the bucket's compounded equity
- robustness: the bucket's equity at the first and last bar of each calendar year

The scores come from ``market_regimes.regime_score`` and ``robustness_regime_score`` themselves,
so there is one definition. ``tests/test_regime_scoring.py`` checks this module against the
direct scorer bar for bar.

A bucket's equity compounds only the bars of trades *entered* in that regime, from the first bar
after entry to the exit bar, and is flat otherwise. Bars of a trade that is still open count as
they happen (they are known by then); a trade's return and the regime that owns it are counted
when it closes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from api.market_regimes import regime_score, robustness_regime_score

PERIODS_PER_YEAR = 252


@dataclass(frozen=True)
class BucketScore:
    """Both scores for one regime state, plus how many closed trades stand behind them."""

    n_trades: int
    portfolio_score: float | None  # the four-part regime score (Sortino over invested bars)
    robustness_score: float | None  # average trade + robustness only


@dataclass
class EntityHistory:
    """What the scorer needs to know about one book, aligned bar for bar.

    ``labels[dimension][i]`` is the regime state on bar ``i`` (None where unlabeled).
    ``bar_returns`` are the book's own simple returns (flat bars are 0). ``trade_pnl`` is only
    read on exit bars.
    """

    dates: pd.DatetimeIndex
    bar_returns: np.ndarray
    long_in: np.ndarray
    hold: np.ndarray
    long_out: np.ndarray
    trade_pnl: np.ndarray
    labels: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        n = len(self.dates)
        for name in ("bar_returns", "long_in", "hold", "long_out", "trade_pnl"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have one value per bar")
        for dimension, labels in self.labels.items():
            if len(labels) != n:
                raise ValueError(f"labels for {dimension} must have one value per bar")


class _Bucket:
    __slots__ = (
        "equity", "peak", "drawdown", "sum_r", "sum_down2", "invested_bars",
        "trades", "sum_pnl", "year_first", "year_last",
    )

    def __init__(self) -> None:
        self.equity = 1.0
        self.peak = 1.0
        self.drawdown = 0.0
        self.sum_r = 0.0
        self.sum_down2 = 0.0
        self.invested_bars = 0
        self.trades = 0
        self.sum_pnl = 0.0
        self.year_first: dict[int, float] = {}
        self.year_last: dict[int, float] = {}


def score_buckets(
    history: EntityHistory,
    keys_by_dimension: Mapping[str, Sequence[str]],
    checkpoints: Iterable[int],
    *,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[int, dict[tuple[str, str], BucketScore | None]]:
    """Scores of every regime state, as they stood after each checkpoint bar.

    ``keys_by_dimension`` lists the states to track per dimension. A state that has never held
    a position maps to None. Each checkpoint is scored on the bars up to and including itself.
    """
    wanted = sorted({int(i) for i in checkpoints})
    n = len(history.dates)
    if wanted and (wanted[0] < 0 or wanted[-1] >= n):
        raise ValueError("Checkpoint outside the history")
    years = history.dates.year.to_numpy()
    first_bar: dict[int, int] = {}
    last_bar: dict[int, int] = {}
    for index, year in enumerate(years):
        first_bar.setdefault(int(year), index)
        last_bar[int(year)] = index

    state = {
        dimension: {key: _Bucket() for key in keys}
        for dimension, keys in keys_by_dimension.items()
    }
    owner: dict[str, str | None] = {dimension: None for dimension in state}
    results: dict[int, dict[tuple[str, str], BucketScore | None]] = {}
    pending = iter(wanted)
    next_checkpoint = next(pending, None)
    sqrt_periods = math.sqrt(periods_per_year)

    def score_all(i: int) -> dict[tuple[str, str], BucketScore | None]:
        out: dict[tuple[str, str], BucketScore | None] = {}
        current_year = int(years[i])
        for dimension, buckets in state.items():
            for key, b in buckets.items():
                if b.invested_bars == 0:
                    out[(dimension, key)] = None
                    continue
                downside = math.sqrt(b.sum_down2 / b.invested_bars)
                sortino = (
                    sqrt_periods * (b.sum_r / b.invested_bars) / downside if downside != 0 else 0.0
                )
                year_returns: dict[int, float] = {}
                for year, first in b.year_first.items():
                    in_progress = year == current_year
                    bars = (i if in_progress else last_bar[year]) - first_bar[year] + 1
                    if bars < 2 or first <= 0:  # yearly_returns() skips one-bar years
                        continue
                    last = b.equity if in_progress else b.year_last[year]
                    year_returns[year] = last / first - 1.0
                avg_trade = b.sum_pnl / b.trades if b.trades else None
                full = regime_score(sortino, avg_trade, b.drawdown, year_returns)
                robust = robustness_regime_score(avg_trade, year_returns)
                out[(dimension, key)] = BucketScore(
                    n_trades=b.trades,
                    portfolio_score=full["score"],
                    robustness_score=robust["score"],
                )
        return out

    for i in range(n):
        new_year = i == 0 or years[i] != years[i - 1]
        if i > 0 and years[i] != years[i - 1]:
            previous = int(years[i - 1])
            for buckets in state.values():
                for b in buckets.values():
                    b.year_last[previous] = b.equity
        r = history.bar_returns[i]
        r = float(r) if np.isfinite(r) else 0.0
        held = bool(history.hold[i])
        for dimension, buckets in state.items():
            if history.long_in[i]:
                owner[dimension] = history.labels[dimension][i]
            key = owner[dimension] if held else None
            if key is not None and i > 0 and key in buckets:
                b = buckets[key]
                b.equity *= 1.0 + r
                if b.equity > b.peak:
                    b.peak = b.equity
                drawdown = (b.peak - b.equity) / b.peak
                if drawdown > b.drawdown:
                    b.drawdown = drawdown
                b.sum_r += r
                if r < 0:
                    b.sum_down2 += r * r
                b.invested_bars += 1
            if history.long_out[i]:
                if key is not None and key in buckets and np.isfinite(history.trade_pnl[i]):
                    b = buckets[key]
                    b.trades += 1
                    b.sum_pnl += float(history.trade_pnl[i])
                owner[dimension] = None
        if new_year:
            year = int(years[i])
            for buckets in state.values():
                for b in buckets.values():
                    b.year_first[year] = b.equity
        if next_checkpoint == i:
            results[i] = score_all(i)
            next_checkpoint = next(pending, None)
    return results
