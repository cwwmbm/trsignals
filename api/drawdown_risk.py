"""How likely a drawdown is to exceed a given depth, from the book's own daily returns.

The backtest is one path, and its maximum drawdown is one draw from a wide distribution. This
reshuffles the daily returns of the finished equity curve in blocks (a circular block
bootstrap), compounds each reshuffle into a path as long as the backtest, and records the
worst peak-to-trough fall of every path. ``P(DD > x)`` is the share of paths whose worst fall
went beyond ``x``, so every threshold comes from the same set of paths at no extra cost.

Blocks keep the clustering of good and bad days that independent draws would destroy. The
returns come from the finished curve, so this works for any book, however it was sized: one
symbol or several, flat or nested. It resamples this history; it is not a forecast of regimes
the history never contained, and is most useful for comparing rules with one another.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from stats import performance_frame

THRESHOLDS = (0.40, 0.50, 0.75)
PATHS = 4000
# Fewer paths where several books are shown side by side.
COMPARISON_PATHS = 1500
# Long (intraday) histories get fewer paths, down to this many, so the work stays near
# PATHS x one daily history: about 26 million resampled bars.
MIN_PATHS = 500
CELL_BUDGET = 26_000_000
BLOCK_DAYS = 20
TRADING_DAYS = 252
SEED = 42
MIN_BARS = 60
# Keeps the working set near 30 MB however long the history is.
_CELLS_PER_CHUNK = 4_000_000
# A return at or below this would make the logarithm undefined; equity cannot lose more.
_FLOOR = -0.999999


def _block_indices(starts: np.ndarray, block: int, length: int) -> np.ndarray:
    """Circular block positions: each row is the concatenation of its blocks, cut to ``length``."""
    steps = starts[:, :, None] + np.arange(block)
    return steps.reshape(len(starts), -1)[:, :length] % length


def bootstrap_max_drawdowns(
    returns: np.ndarray,
    *,
    paths: int = PATHS,
    block: int = BLOCK_DAYS,
    seed: int = SEED,
) -> np.ndarray:
    """Worst peak-to-trough fall (a fraction) of each reshuffled path of ``returns``."""
    returns = np.clip(np.asarray(returns, dtype=float), _FLOOR, None)
    length = len(returns)
    block = max(1, min(int(block), length))
    blocks = -(-length // block)
    # Every path's starting points are drawn at once, so the answer does not depend on how the
    # work is split into chunks below.
    starts = np.random.default_rng(seed).integers(0, length, size=(int(paths), blocks))
    chunk = max(1, _CELLS_PER_CHUNK // length)
    worst = np.empty(int(paths))
    for lo in range(0, int(paths), chunk):
        picked = returns[_block_indices(starts[lo:lo + chunk], block, length)]
        equity = np.exp(np.cumsum(np.log1p(picked), axis=1))
        # The path starts at 1, so a fall from the very first bar counts.
        peak = np.maximum(np.maximum.accumulate(equity, axis=1), 1.0)
        worst[lo:lo + chunk] = (1.0 - equity / peak).max(axis=1)
    return worst


def drawdown_risk(
    equity: Sequence[float] | np.ndarray,
    *,
    thresholds: Sequence[float] = THRESHOLDS,
    paths: int = PATHS,
    block: int = BLOCK_DAYS,
    seed: int = SEED,
    periods_per_year: float = TRADING_DAYS,
) -> dict[str, Any] | None:
    """Probability that the worst drawdown of a backtest-length path exceeds each threshold.

    ``equity`` is the book's value per bar. ``block`` is in trading days: with intraday bars
    (``periods_per_year`` above 252) it spans that many days' worth of bars, and the path count
    is lowered so a long history stays quick. Returns ``None`` when there is too little history
    (or no usable equity) for the question to mean anything.
    """
    values = np.asarray(equity, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < MIN_BARS + 1 or (values <= 0).any():
        return None
    returns = values[1:] / values[:-1] - 1.0
    bars_per_day = max(1.0, float(periods_per_year) / TRADING_DAYS)
    block_bars = max(1, min(int(round(block * bars_per_day)), len(returns)))
    paths = min(int(paths), max(MIN_PATHS, CELL_BUDGET // len(returns)))
    worst = bootstrap_max_drawdowns(returns, paths=paths, block=block_bars, seed=seed)
    n = len(worst)
    rows = []
    for threshold in thresholds:
        probability = float((worst > threshold).mean())
        rows.append(
            {
                "drawdown": float(threshold),
                "probability": probability,
                "std_error": float(np.sqrt(probability * (1.0 - probability) / n)),
            }
        )
    realized = float((1.0 - values / np.maximum.accumulate(values)).max())
    return {
        "method": "block_bootstrap",
        "paths": int(n),
        "block_days": int(round(block_bars / bars_per_day)),
        "block_bars": block_bars,
        "horizon_bars": int(len(returns)),
        "periods_per_year": float(periods_per_year),
        "seed": int(seed),
        "historical_max_drawdown": realized,
        "median_max_drawdown": float(np.median(worst)),
        "p95_max_drawdown": float(np.percentile(worst, 95)),
        "thresholds": rows,
    }


def drawdown_risk_for_frame(frame: pd.DataFrame | None, **kwargs: Any) -> dict[str, Any] | None:
    """The same, for a backtest frame, from the first bar whose entry conditions could be read."""
    if frame is None or frame.empty or "RollingPnL" not in frame.columns:
        return None
    scored = performance_frame(frame)
    return drawdown_risk(scored["RollingPnL"].to_numpy(dtype=float), **kwargs)
