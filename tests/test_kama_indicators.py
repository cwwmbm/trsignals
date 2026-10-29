import unittest

import numpy as np
import pandas as pd

from api.indicator_catalog import is_compare_indicator_allowed, list_indicators
from api.strategy_compiler import compile_buy_mask
from indicator_sweep import PRICE_SWEEPS
from indicators import INDICATOR_CACHE_VERSION, signed_cross_age


def _kama(close):
    import ta

    return ta.momentum.KAMAIndicator(pd.Series(close, dtype=float), window=10, pow1=2, pow2=30).kama()


class KamaBehaviourTests(unittest.TestCase):
    def test_tracks_a_clean_trend_closely(self):
        close = np.linspace(100, 160, 80)
        kama = _kama(close)
        # ER is 1 in a straight line, so KAMA moves like a 2-bar EMA and lags very little.
        gap = (close[-1] - kama.iloc[-1]) / close[-1] * 100
        self.assertLess(gap, 1.0)

    def test_goes_flat_in_chop(self):
        close = np.r_[np.linspace(100, 120, 40), 120 + np.tile([1.0, -1.0], 30)]
        kama = _kama(close)
        slope = kama.pct_change(5) * 100
        self.assertLess(abs(slope.iloc[-1]), 0.2)
        # A plain 10-bar EMA keeps reacting to every swing; KAMA barely moves.
        ema = pd.Series(close).ewm(span=10, adjust=False).mean()
        self.assertLess(kama.iloc[-20:].std(), ema.iloc[-20:].std())

    def test_cache_version_was_bumped(self):
        self.assertGreaterEqual(INDICATOR_CACHE_VERSION, 8)


class KamaCrossTests(unittest.TestCase):
    def test_cross_age_counts_days_on_each_side(self):
        close = pd.Series([9.0, 9.0, 11.0, 12.0, 8.0, 8.0, 12.0])
        kama = pd.Series([10.0] * 7)
        age = signed_cross_age(close - kama)
        # The opening run started before the data, so its age is unknown.
        self.assertTrue(age.iloc[:2].isna().all())
        self.assertEqual(age.iloc[2:].tolist(), [1.0, 2.0, -1.0, -2.0, 1.0])

    def test_touching_kama_keeps_the_prior_side(self):
        age = signed_cross_age(pd.Series([-1.0, 1.0, 0.0, 1.0, -1.0]))
        self.assertEqual(age.iloc[1:].tolist(), [1.0, 2.0, 3.0, -1.0])


if __name__ == "__main__":
    unittest.main()
