import unittest

import numpy as np
import pandas as pd

from api import market_regimes as m
from api.regime_scoring import BucketScore, EntityHistory, score_buckets

KEYS = {"vix": ("low", "mid", "high"), "spy": ("bull", "bear")}


def _history(n=1400, seed=11, with_gaps=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-03-01", periods=n)
    hold = np.zeros(n, bool)
    long_in = np.zeros(n, bool)
    long_out = np.zeros(n, bool)
    i = 4
    while i < n - 8:
        length = int(rng.integers(1, 7))
        long_in[i] = True
        hold[i + 1 : i + 1 + length] = True
        long_out[i + length] = True
        i += length + int(rng.integers(2, 25 if with_gaps else 4))
    returns = np.where(hold, rng.normal(0.003, 0.025, n), 0.0)
    equity = 10_000 * np.cumprod(1 + returns)
    pnl = np.zeros(n)
    entry = None
    for k in range(n):
        if long_in[k]:
            entry = equity[k]
        if long_out[k] and entry:
            pnl[k] = equity[k] / entry - 1
    labels = {
        dim: np.array([keys[int(x)] for x in rng.integers(0, len(keys), n)], dtype=object)
        for dim, keys in KEYS.items()
    }
    return EntityHistory(dates, returns, long_in, hold, long_out, pnl, labels)


def _direct(history: EntityHistory, end: int, dimension: str, key: str):
    """The existing scorer, run on the history up to and including bar ``end``."""
    sl = slice(0, end + 1)
    labels = history.labels[dimension][sl]
    held = m._hold_entry_labels(labels, history.long_in[sl], history.hold[sl], history.long_out[sl])
    mask = held == key
    if not mask.any():
        return None
    summary = m._summary_metrics_of_trades(
        pd.Series(history.dates[sl]), history.bar_returns[sl], mask, 252
    )
    closed = mask & history.long_out[sl] & np.isfinite(history.trade_pnl[sl])
    avg = float(np.mean(history.trade_pnl[sl][closed])) if closed.any() else None
    full = m.regime_score(summary["score_sortino"], avg, summary["score_max_drawdown"], summary["year_returns"])
    robust = m.robustness_regime_score(avg, summary["year_returns"])
    return int(closed.sum()), full["score"], robust["score"]


class IncrementalMatchesDirectTests(unittest.TestCase):
    def _compare(self, history, checkpoints):
        got = score_buckets(history, KEYS, checkpoints)
        compared = 0
        for end in checkpoints:
            for dimension, keys in KEYS.items():
                for key in keys:
                    expected = _direct(history, end, dimension, key)
                    actual = got[end][(dimension, key)]
                    if expected is None:
                        self.assertIsNone(actual, (end, dimension, key))
                        continue
                    n, full, robust = expected
                    self.assertEqual(actual.n_trades, n, (end, dimension, key))
                    for want, have, label in ((full, actual.portfolio_score, "portfolio"), (robust, actual.robustness_score, "robustness")):
                        if want is None:
                            self.assertIsNone(have, (end, dimension, key, label))
                        else:
                            self.assertAlmostEqual(have, want, places=9, msg=(end, dimension, key, label))
                            compared += 1
        return compared

    def test_matches_the_direct_scorer_at_many_checkpoints(self):
        history = _history()
        compared = self._compare(history, [30, 200, 411, 640, 900, 1150, 1399])
        self.assertGreater(compared, 60)  # the comparison actually covered scored buckets

    def test_matches_with_dense_trading(self):
        self.assertGreater(self._compare(_history(n=900, seed=3, with_gaps=False), [100, 450, 899]), 20)

    def test_matches_when_a_checkpoint_is_the_first_bar_of_a_year(self):
        history = _history()
        years = history.dates.year.to_numpy()
        firsts = [int(np.flatnonzero(years == y)[0]) for y in sorted(set(years.tolist()))[1:4]]
        self._compare(history, firsts + [f - 1 for f in firsts])

    def test_each_checkpoint_only_sees_the_past(self):
        history = _history()
        early = score_buckets(history, KEYS, [500])[500]
        # Cutting the history off at the checkpoint changes nothing.
        cut = EntityHistory(
            history.dates[:501], history.bar_returns[:501], history.long_in[:501], history.hold[:501],
            history.long_out[:501], history.trade_pnl[:501], {k: v[:501] for k, v in history.labels.items()},
        )
        self.assertEqual(early, score_buckets(cut, KEYS, [500])[500])

    def test_never_invested_state_has_no_score(self):
        history = _history()
        got = score_buckets(history, {"vix": ("low", "nothing")}, [1399])[1399]
        self.assertIsNone(got[("vix", "nothing")])
        self.assertIsInstance(got[("vix", "low")], BucketScore)

    def test_rejects_bad_input(self):
        history = _history(n=100)
        with self.assertRaises(ValueError):
            score_buckets(history, KEYS, [100])
        with self.assertRaises(ValueError):
            EntityHistory(history.dates, history.bar_returns[:-1], history.long_in, history.hold, history.long_out, history.trade_pnl, history.labels)


if __name__ == "__main__":
    unittest.main()
