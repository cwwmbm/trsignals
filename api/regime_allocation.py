"""Regime-based allocation (V2): confidence from how each regime has treated the strategies.

How it decides (all of it known on the decision day, none of it from later bars):

1. Every active strategy (in trade, or with a BUY today) is scored against the regime states in
   force today. For each state, the strategy's own history of trades entered in that state gives
   a score. Low scores cost confidence, and confidence starts at 100%:

       score <= low cutoff              -> low penalty
       low cutoff < score <= high cutoff -> high penalty
       otherwise (or fewer than ``min_trades`` closed trades behind the score) -> no penalty

   States with too little history count as score 100, so a young system starts fully invested
   and tightens as evidence builds.
2. Strategy confidence is the plain average of the active strategies' confidences.
3. Portfolio consensus is the same calculation on the portfolio's own history: the all-in
   portfolio of the same strategies, which does not depend on the allocation (otherwise the
   decision would feed on itself).
4. ``allocation.RegimeAdjustedModel`` turns the two into an exposure:
   ``strategy confidence * (0.70 + 0.30 * portfolio consensus)``.

Decisions are made whenever the set of active strategies changes while the portfolio is in
trade: a strategy joins, or a strategy leaves while another stays. The position moves to the new
exposure at that bar's close. When nothing is active the book is flat.

Which score: ``robustness`` (average trade return + robustness, the default) or ``portfolio``
(the four-part regime score).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from api.allocation import PORTFOLIO_CONSENSUS, STRATEGY_CONFIDENCE
from api.market_regimes import (
    _SHARPE_BUCKETS,
    _bool_column,
    _regime_label_sets,
    _strategy_bar_returns,
)
from api.regime_scoring import BucketScore, EntityHistory, score_buckets

SCORE_ROBUSTNESS = "robustness"
SCORE_PORTFOLIO = "portfolio"
SCORE_KINDS = (SCORE_ROBUSTNESS, SCORE_PORTFOLIO)

# The 15 regimes the allocation reads: SPY trend, VIX, equity-risk breadth, sector deviation from
# its 50-day average, copper/gold, materials strength, credit risk-on, bond duration, housing
# strength, housing vs defensive, rate shock, 10Y-3M curve, rate shock x curve, 10Y-3M curve
# change, and dollar + rates.
ALLOCATION_REGIME_DIMENSIONS = (
    "spy",
    "vix",
    "equity_risk_breadth",
    "sector_trend_50",
    "copper_gold",
    "materials_breadth",
    "credit_risk_on",
    "bond_duration",
    "housing_strength",
    "housing_defensive",
    "rate_shock",
    "curve_10y3m",
    "rate_curve",
    "curve_change_20",
    "dollar_rates",
)

DEFAULT_MIN_TRADES = 20

# How hard a regime state that has treated a strategy badly is punished. Each preset is
# (label, low cutoff, low penalty, high cutoff, high penalty): a state scoring at or below the
# low cutoff costs the low penalty, one above it but at or below the high cutoff costs the high
# penalty. Penalties are fractions of full confidence.
PENALTY_PRESETS: dict[str, tuple[str, float, float, float, float]] = {
    "harsh": ("Harsh", 10.0, 1.00, 25.0, 0.00),
    "strict": ("Strict", 10.0, 0.50, 25.0, 0.25),
    "medium": ("Medium", 10.0, 1 / 3, 25.0, 0.10),
    "low": ("Low", 10.0, 0.25, 25.0, 0.05),
}
DEFAULT_PENALTY_PRESET = "medium"

DEFAULT_LOW_CUTOFF, DEFAULT_LOW_PENALTY, DEFAULT_HIGH_CUTOFF, DEFAULT_HIGH_PENALTY = PENALTY_PRESETS[
    DEFAULT_PENALTY_PRESET
][1:]


@dataclass(frozen=True)
class RegimePenaltySettings:
    """Which score to read, where the cutoffs sit, and what each tier costs.

    Users pick one of ``PENALTY_PRESETS``; the numbers are kept explicit so the rules can be
    tested and shown. Penalties are fractions of full confidence (0.5 = take off 50 points).
    """

    score: str = SCORE_ROBUSTNESS
    low_cutoff: float = DEFAULT_LOW_CUTOFF
    low_penalty: float = DEFAULT_LOW_PENALTY
    high_cutoff: float = DEFAULT_HIGH_CUTOFF
    high_penalty: float = DEFAULT_HIGH_PENALTY
    min_trades: int = DEFAULT_MIN_TRADES
    preset: str | None = None

    def __post_init__(self) -> None:
        if self.score not in SCORE_KINDS:
            raise ValueError(f"Regime score must be one of {', '.join(SCORE_KINDS)}")
        if not 0 < self.low_cutoff < self.high_cutoff <= 100:
            raise ValueError("Regime cutoffs must satisfy 0 < low cutoff < high cutoff <= 100")
        for name, value in (("low", self.low_penalty), ("high", self.high_penalty)):
            if not 0 <= value <= 1:
                raise ValueError(f"The {name} regime penalty must be between 0% and 100%")
        if int(self.min_trades) != self.min_trades or self.min_trades < 1:
            raise ValueError("Minimum trades per regime state must be a whole number of at least 1")

    @classmethod
    def from_preset(cls, preset: str | None = None, score: str | None = None) -> "RegimePenaltySettings":
        name = preset or DEFAULT_PENALTY_PRESET
        if name not in PENALTY_PRESETS:
            raise ValueError(f"Regime penalty must be one of {', '.join(PENALTY_PRESETS)}")
        _, low_cutoff, low_penalty, high_cutoff, high_penalty = PENALTY_PRESETS[name]
        return cls(
            score=score or SCORE_ROBUSTNESS,
            low_cutoff=low_cutoff,
            low_penalty=low_penalty,
            high_cutoff=high_cutoff,
            high_penalty=high_penalty,
            min_trades=DEFAULT_MIN_TRADES,
            preset=name,
        )

    @classmethod
    def from_source(cls, source: Any) -> "RegimePenaltySettings":
        """Read ``regime_penalty`` and ``regime_score`` off a request or a saved portfolio."""
        return cls.from_preset(
            getattr(source, "regime_penalty", None), getattr(source, "regime_score", None)
        )

    def penalty(self, score: float) -> float:
        if score <= self.low_cutoff:
            return self.low_penalty
        if score <= self.high_cutoff:
            return self.high_penalty
        return 0.0

    def as_dict(self) -> dict[str, Any]:
        """For the response; penalties in percent."""
        return {
            "preset": self.preset,
            "score": self.score,
            "low_cutoff": self.low_cutoff,
            "low_penalty": self.low_penalty * 100.0,
            "high_cutoff": self.high_cutoff,
            "high_penalty": self.high_penalty * 100.0,
            "min_trades": self.min_trades,
        }


def _score_of(state: BucketScore, kind: str) -> float | None:
    return state.robustness_score if kind == SCORE_ROBUSTNESS else state.portfolio_score


def regime_confidence(
    states: Iterable[BucketScore | None],
    settings: RegimePenaltySettings,
) -> float:
    """Confidence in [0, 1] given the score of each regime state in force.

    A state with no history, no score, or fewer than ``min_trades`` closed trades is ignored.
    """
    confidence = 1.0
    for state in states:
        if state is None or state.n_trades < settings.min_trades:
            continue
        score = _score_of(state, settings.score)
        if score is None:
            continue
        confidence -= settings.penalty(score)
    return max(0.0, confidence)


def active_set_change_bars(active: np.ndarray) -> np.ndarray:
    """Bars where some strategy is active and the set of active strategies differs from the
    previous bar's. These are the bars V2 re-decides on, including the first entry."""
    active = np.asarray(active, dtype=bool)
    n = active.shape[1]
    bars = []
    previous = np.zeros(active.shape[0], dtype=bool)
    for i in range(n):
        column = active[:, i]
        if column.any() and not np.array_equal(column, previous):
            bars.append(i)
        previous = column
    return np.asarray(bars, dtype=int)


def regime_labels(
    dates: pd.Series | pd.DatetimeIndex,
    calendar: pd.DataFrame | None,
    dimensions: Sequence[str] = ALLOCATION_REGIME_DIMENSIONS,
) -> dict[str, np.ndarray]:
    """Regime state on each date, per dimension (None where it cannot be labeled)."""
    frame = pd.DataFrame({"Date": pd.Series(pd.DatetimeIndex(dates))})
    sets = _regime_label_sets(frame, calendar)
    return {dimension: sets[dimension].to_numpy(dtype=object) for dimension in dimensions}


def history_from_frame(
    frame: pd.DataFrame,
    labels: Mapping[str, np.ndarray],
) -> EntityHistory:
    """An ``EntityHistory`` from a backtest frame (RollingPnL, LongTradeIn/Out, HoldLong, TradePnL)."""
    n = len(frame)
    trade_pnl = (
        pd.to_numeric(frame["TradePnL"], errors="coerce").to_numpy(dtype=float)
        if "TradePnL" in frame.columns
        else np.full(n, np.nan)
    )
    return EntityHistory(
        dates=pd.DatetimeIndex(pd.to_datetime(frame["Date"])),
        bar_returns=_strategy_bar_returns(pd.to_numeric(frame["RollingPnL"], errors="coerce")),
        long_in=_bool_column(frame, "LongTradeIn"),
        hold=_bool_column(frame, "HoldLong"),
        long_out=_bool_column(frame, "LongTradeOut"),
        trade_pnl=trade_pnl,
        labels=labels,
    )


class RegimeAllocationContext:
    """Turns histories and today's regimes into the model's per-decision inputs.

    ``strategy_history(id)`` builds one strategy's standalone history. ``baseline_history(ids)``
    builds the all-in portfolio of those strategies (cached by the caller or here). Both are
    aligned to their own dates; scores are looked up on the master calendar by date.

    Strategy scores are computed once, at every bar where the *full* set of strategies
    changes. Any smaller portfolio (leave-one-out, Shapley) changes on a subset of those bars,
    so it reuses them. Only the portfolio baseline has to be re-scored per subset.
    """

    def __init__(
        self,
        settings: RegimePenaltySettings,
        *,
        strategy_ids: Sequence[str],
        master_dates: pd.DatetimeIndex,
        full_active: np.ndarray,
        strategy_history: Callable[[str], EntityHistory],
        baseline_history: Callable[[tuple[str, ...]], EntityHistory],
        current_labels: Mapping[str, np.ndarray],
        dimensions: Sequence[str] = ALLOCATION_REGIME_DIMENSIONS,
    ) -> None:
        self.settings = settings
        self.master_dates = pd.DatetimeIndex(master_dates)
        self.dimensions = tuple(dimensions)
        self._ids = list(strategy_ids)
        self._full_active = np.asarray(full_active, dtype=bool)
        self._strategy_history = strategy_history
        self._baseline_history = baseline_history
        self._labels = current_labels
        self._keys = {dimension: tuple(_SHARPE_BUCKETS[dimension]) for dimension in self.dimensions}
        self._full_events = active_set_change_bars(self._full_active)
        self._strategy_scores: dict[str, dict[int, Mapping[tuple[str, str], BucketScore | None]]] = {}
        self._strategy_requested: dict[str, set[int]] = {}
        self._baseline_scores: dict[tuple[str, ...], dict[int, Mapping[tuple[str, str], BucketScore | None]]] = {}
        self._baseline_requested: dict[tuple[str, ...], set[int]] = {}

    def _scores_at(
        self,
        history: EntityHistory,
        bars: Sequence[int],
    ) -> dict[int, Mapping[tuple[str, str], BucketScore | None]]:
        """Scores keyed by master bar; bars the history has no row for are left out."""
        bars = list(bars)
        rows = history.dates.get_indexer(self.master_dates[bars])
        wanted = {int(row): int(bar) for bar, row in zip(bars, rows) if row >= 0}
        if not wanted:
            return {}
        scored = score_buckets(history, self._keys, wanted.keys())
        return {bar: scored[row] for row, bar in wanted.items()}

    def _strategy_at(self, strategy_id: str, bars: Sequence[int]) -> dict[int, Mapping]:
        requested = self._strategy_requested.setdefault(strategy_id, set())
        if not {int(b) for b in bars} <= requested:
            # One pass covers every bar this strategy is active on in the full set. A smaller
            # portfolio only changes on a subset of those bars, so it never needs another.
            index = self._ids.index(strategy_id)
            full = {int(b) for b in self._full_events if self._full_active[index, b]}
            requested |= full | {int(b) for b in bars}
            self._strategy_scores[strategy_id] = self._scores_at(
                self._strategy_history(strategy_id), sorted(requested)
            )
        return self._strategy_scores[strategy_id]

    def _baseline_at(self, ids: tuple[str, ...], bars: Sequence[int]) -> dict[int, Mapping]:
        requested = self._baseline_requested.setdefault(ids, set())
        if not {int(b) for b in bars} <= requested:
            requested |= {int(b) for b in bars}
            self._baseline_scores[ids] = self._scores_at(self._baseline_history(ids), sorted(requested))
        return self._baseline_scores[ids]

    def _states(self, scores: Mapping | None, bar: int) -> list[BucketScore | None]:
        if scores is None:
            return []
        states = []
        for dimension in self.dimensions:
            key = self._labels[dimension][bar]
            if key is None or (isinstance(key, float) and np.isnan(key)):
                continue
            states.append(scores.get((dimension, key)))
        return states

    def decisions(
        self,
        ids: Sequence[str],
        active: np.ndarray,
        dates: pd.DatetimeIndex | None = None,
        settings: RegimePenaltySettings | None = None,
    ) -> dict[int, dict[str, float]]:
        """``{bar: {strategy_confidence, portfolio_consensus}}`` for every decision bar.

        ``active`` is the (strategies, bars) matrix for ``ids``, in the same row order, on the
        context's calendar (``dates``, when given, must be that calendar). The scores do not
        depend on the penalties, so ``settings`` can differ from the context's own and every
        preset reuses one scoring pass.
        """
        settings = settings or self.settings
        ids = tuple(ids)
        active = np.asarray(active, dtype=bool)
        if dates is not None and not self.master_dates.equals(pd.DatetimeIndex(dates)):
            raise ValueError("Regime allocation needs the same calendar as the portfolio it was built for")
        if active.shape != (len(ids), len(self.master_dates)):
            raise ValueError("active must be (strategies, bars) for the given ids")
        bars = [int(b) for b in active_set_change_bars(active)]
        if not bars:
            return {}
        per_strategy = {sid: self._strategy_at(sid, bars) for sid in ids}
        baseline = self._baseline_at(ids, bars)
        out: dict[int, dict[str, float]] = {}
        for bar in bars:
            confidences = []
            for k, sid in enumerate(ids):
                if not active[k, bar]:
                    continue
                scores = per_strategy[sid].get(bar)
                confidences.append(
                    regime_confidence(self._states(scores, bar), settings) if scores is not None else 1.0
                )
            consensus_scores = baseline.get(bar)
            out[bar] = {
                STRATEGY_CONFIDENCE: float(np.mean(confidences)),
                PORTFOLIO_CONSENSUS: (
                    regime_confidence(self._states(consensus_scores, bar), settings)
                    if consensus_scores is not None
                    else 1.0
                ),
            }
        return out
