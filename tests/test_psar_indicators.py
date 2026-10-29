import unittest

import numpy as np
import pandas as pd

from api.indicator_catalog import list_indicators
from indicator_sweep import PRICE_SWEEPS
from indicators import INDICATOR_CACHE_VERSION, psar_columns


class PsarColumnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A rally, a slide, then another rally: PSAR has to flip down and back up.
        close = pd.Series(np.r_[np.linspace(100, 140, 60), np.linspace(140, 110, 30), np.linspace(110, 150, 40)])
        cls.data = pd.DataFrame(psar_columns(close + 0.5, close - 0.5, close)).assign(Close=close)

    def test_flip_age_follows_the_sar_side(self):
        age = self.data["PSARFlipAge"].dropna()
        below = self.data.loc[age.index, "PSAR"] < self.data.loc[age.index, "Close"]
        self.assertTrue(((age > 0) == below).all())
        # Flips down once in the slide and back up in the second rally.
        self.assertEqual((age == -1).sum(), 1)
        self.assertEqual((age == 1).sum(), 1)
        self.assertGreater(age.iloc[-1], 1)

    def test_distance_sign_matches_side(self):
        d = self.data.dropna(subset=["PSARFlipAge", "Close_PSAR"])
        self.assertTrue(((d["Close_PSAR"] > 0) == (d["PSARFlipAge"] > 0)).all())


class PsarWiringTests(unittest.TestCase):
    def test_swept_and_listed(self):
        swept = {spec[0] for spec in PRICE_SWEEPS}
        self.assertTrue({"PSARFlipAge", "Close_PSAR"} <= swept)
        items = {item["id"]: item for item in list_indicators(builder_only=True, include_regimes=False)}
        self.assertIn("typicalRange", items["PSARFlipAge"])
        self.assertIn("typicalRange", items["Close_PSAR"])

    def test_cache_version_was_bumped(self):
        self.assertGreaterEqual(INDICATOR_CACHE_VERSION, 9)


if __name__ == "__main__":
    unittest.main()
