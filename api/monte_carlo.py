from __future__ import annotations

from typing import Literal

import numpy as np

MonteCarloMethod = Literal["shuffle", "bootstrap"]

DEFAULT_N_SIMS = 1000
DEFAULT_START_CAPITAL = 15000.0
MIN_TRADES = 2
MIN_N_SIMS = 100
MAX_N_SIMS = 10_000
PERCENTILES = (5, 25, 50, 75, 95)
RNG_SEED = 42
CONFIDENCE_PERCENTILE = 95


def clamp_n_sims(n_sims: int) -> int:
    return max(MIN_N_SIMS, min(MAX_N_SIMS, int(n_sims)))


def _percentile_block(values: np.ndarray) -> dict[str, float]:
    ps = np.percentile(values, list(PERCENTILES))
    return {
        "p5": float(ps[0]),
        "p25": float(ps[1]),
        "p50": float(ps[2]),
        "p75": float(ps[3]),
        "p95": float(ps[4]),
        "mean": float(np.mean(values)),
    }


def _equity_paths(returns_matrix: np.ndarray, start_capital: float) -> np.ndarray:
    """Compound trade returns into equity paths. Shape: (n_sims, n_trades + 1)."""
    growth = 1.0 + returns_matrix
    paths = np.empty((returns_matrix.shape[0], returns_matrix.shape[1] + 1), dtype=float)
    paths[:, 0] = start_capital
    paths[:, 1:] = start_capital * np.cumprod(growth, axis=1)
    return paths


def _max_drawdowns(paths: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (running_max - paths) / np.where(running_max > 0, running_max, 1.0)
    return drawdowns.max(axis=1)


def _sample_returns(
    returns: np.ndarray,
    *,
    method: MonteCarloMethod,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_trades = returns.shape[0]
    if method == "shuffle":
        matrix = np.empty((n_sims, n_trades), dtype=float)
        for i in range(n_sims):
            matrix[i] = rng.permutation(returns)
        return matrix
    if method == "bootstrap":
        indices = rng.integers(0, n_trades, size=(n_sims, n_trades))
        return returns[indices]
    raise ValueError(f"Unsupported Monte Carlo method: {method}")


def _drawdown_distribution(max_dds: np.ndarray) -> tuple[list[dict], dict]:
    """Histogram (counts) + CDF (%) of max drawdowns, plus confidence marker."""
    dd_pct = np.asarray(max_dds, dtype=float) * 100.0
    n_sims = int(dd_pct.size)
    hi = float(max(np.ceil(dd_pct.max() * 2) / 2, 1.0))  # round up to 0.5%
    # Aim for ~1% bins, capped for readability.
    bin_width = 1.0 if hi <= 50 else (2.0 if hi <= 100 else max(hi / 40.0, 1.0))
    n_bins = max(int(np.ceil(hi / bin_width)), 1)
    edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)
    counts, edges = np.histogram(dd_pct, bins=edges)
    cumulative = np.cumsum(counts)
    cumulative_pct = cumulative / n_sims * 100.0

    distribution = [
        {
            "drawdown_pct": float(edges[i + 1]),
            "count": int(counts[i]),
            "cumulative_pct": float(cumulative_pct[i]),
        }
        for i in range(len(counts))
    ]

    p95 = float(np.percentile(dd_pct, CONFIDENCE_PERCENTILE))
    marker = {
        "percentile": CONFIDENCE_PERCENTILE,
        "drawdown_pct": p95,
        "cumulative_pct": float(CONFIDENCE_PERCENTILE),
    }
    return distribution, marker


def run_monte_carlo(
    trade_returns: list[float] | np.ndarray,
    *,
    method: MonteCarloMethod = "shuffle",
    n_sims: int = DEFAULT_N_SIMS,
    start_capital: float = DEFAULT_START_CAPITAL,
    seed: int = RNG_SEED,
) -> dict:
    returns = np.asarray(trade_returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError("trade_returns must be a 1-D list of closed-trade returns")
    if returns.size < MIN_TRADES:
        raise ValueError(f"Monte Carlo requires at least {MIN_TRADES} closed trades")
    if not np.isfinite(returns).all():
        raise ValueError("trade_returns must be finite numbers")
    if start_capital <= 0:
        raise ValueError("start_capital must be positive")
    if method not in ("shuffle", "bootstrap"):
        raise ValueError("method must be 'shuffle' or 'bootstrap'")

    n_sims = clamp_n_sims(n_sims)
    rng = np.random.default_rng(seed)

    actual_paths = _equity_paths(returns.reshape(1, -1), start_capital)
    actual_final = float(actual_paths[0, -1])
    actual_max_dd = float(_max_drawdowns(actual_paths)[0])

    sampled = _sample_returns(returns, method=method, n_sims=n_sims, rng=rng)
    paths = _equity_paths(sampled, start_capital)
    finals = paths[:, -1]
    max_dds = _max_drawdowns(paths)
    distribution, confidence_marker = _drawdown_distribution(max_dds)

    return {
        "method": method,
        "n_sims": n_sims,
        "n_trades": int(returns.size),
        "start_capital": float(start_capital),
        "actual": {
            "final_equity": actual_final,
            "max_drawdown": actual_max_dd,
        },
        "summary": {
            "final_equity": _percentile_block(finals),
            "max_drawdown": _percentile_block(max_dds),
            "pct_sims_final_equity_ge_actual": float(np.mean(finals >= actual_final) * 100.0),
            "pct_sims_max_drawdown_le_actual": float(np.mean(max_dds <= actual_max_dd) * 100.0),
        },
        "drawdown_distribution": distribution,
        "confidence_marker": confidence_marker,
    }
