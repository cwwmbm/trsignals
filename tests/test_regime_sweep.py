"""Market regime sweep: each regime state ANDed into the entry rule, one at a time."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import indicator_sweep
from api.builder_strategy import (
    _regime_trial_frame,
    builder_regime_tryout,
    draft_to_saved_strategy,
    regime_needs_full_path,
)
from api.market_regimes import (
    REGIME_INDICATOR_IDS,
    REGIME_INDICATOR_SPECS,
    regime_flag_frame,
    regime_spec,
)
from api.schemas import BuilderBacktestRequest, BuilderCondition


def _frame(rows: int = 8) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    return pd.DataFrame(
        {
            "Date": dates,
            "Close": np.linspace(100, 110, rows),
            "Buy": [True] * rows,
            "Sell": [False] * rows,
            "EntryReady": [True] * rows,
        }
    )


def _strategy(**overrides):
    payload = {
        "symbol": "SPY",
        "years": 25,
        "conditions": [BuilderCondition(left="RSI2", operator="<=", right="20", logic="AND")],
    }
    payload.update(overrides)
    return draft_to_saved_strategy(BuilderBacktestRequest(**payload))


class RegimeFlagFrameTests(unittest.TestCase):
    def test_flags_match_the_label_for_every_state(self):
        frame = _frame(4)
        calendar = pd.DataFrame(
            {"vix": [10.0, 25.0, np.nan, 40.0]}, index=frame["Date"]
        )
        with patch("api.market_regimes.load_regime_calendar", return_value=calendar):
            flags = regime_flag_frame(frame)
        self.assertEqual(flags["Regime_vix_le_15"].tolist(), [1, -1, 0, -1])
        self.assertEqual(flags["Regime_vix_20_30"].tolist(), [-1, 1, 0, -1])
        self.assertEqual(flags["Regime_vix_gt_30"].tolist(), [-1, -1, 0, 1])
        self.assertEqual(set(flags.columns), set(REGIME_INDICATOR_IDS))

    def test_missing_calendar_raises_unless_only_atr_is_asked_for(self):
        frame = _frame(3)
        with patch("api.market_regimes.load_regime_calendar", return_value=None):
            with self.assertRaises(ValueError):
                regime_flag_frame(frame)
            atr = tuple(spec for spec in REGIME_INDICATOR_SPECS if spec.dimension == "atr")
            self.assertEqual(len(regime_flag_frame(frame, atr).columns), 2)


class RegimeTrialFrameTests(unittest.TestCase):
    def test_state_is_anded_into_buy_and_entry_ready(self):
        frame = _frame(4)
        frame["Buy"] = [True, True, False, True]
        flag = pd.Series([1, -1, 1, 0], index=frame.index)
        trial = _regime_trial_frame(frame, flag)
        self.assertEqual(trial["Buy"].tolist(), [True, False, False, False])
        # Unlabeled days (0) are not ready, other states are.
        self.assertEqual(trial["EntryReady"].tolist(), [True, True, True, False])
        # "is false" keeps the days labeled as some other state, never unlabeled days.
        other = _regime_trial_frame(frame, flag, "is false")
        self.assertEqual(other["Buy"].tolist(), [False, True, False, False])
        self.assertEqual(other["EntryReady"].tolist(), [True, True, True, False])
        with self.assertRaises(ValueError):
            _regime_trial_frame(frame, flag, "<")
        # The refine frame itself is untouched.
        self.assertEqual(frame["Buy"].tolist(), [True, True, False, True])


class RegimeTryoutTests(unittest.TestCase):
    def _run(self, strategy, frame=None):
        frame = _frame() if frame is None else frame
        seen: list[pd.DataFrame] = []

        def fake_execute(data, days, profit, is_long, pnl_column=None):
            seen.append(data)
            return data

        def fake_metrics(data, include_yearly=True):
            return {
                "PnL": 1000.0 + int(data["Buy"].sum()),
                "MaxDD": 10.0,
                "Trades": int(data["Buy"].sum()),
                "%Pstv": 60.0,
                "CAGR": "5%",
                "Sharpe": float(data["Buy"].sum()),
                "Sortino": 1.0,
                "Yearly": [],
            }

        flags = pd.DataFrame(
            {spec.id: -1 for spec in REGIME_INDICATOR_SPECS}, index=frame.index
        )
        flags["Regime_vix_le_15"] = [1, 1, 1, 0, 0, -1, -1, -1]
        with (
            patch("api.market_regimes.regime_flag_frame", side_effect=lambda f, specs=None: flags[[s.id for s in (specs or REGIME_INDICATOR_SPECS)]]),
            patch("api.builder_strategy.bt.execute_strategy", side_effect=fake_execute),
            patch("api.builder_strategy.bt._ranking_metrics", side_effect=fake_metrics),
        ):
            result = builder_regime_tryout(strategy, frame, 2, 1, True, years=5)
        return result, seen

    def test_every_state_is_tried_as_is_and_as_is_not(self):
        result, seen = self._run(_strategy())
        self.assertEqual(len(result), 2 * len(REGIME_INDICATOR_SPECS))
        self.assertEqual(len(seen), 2 * len(REGIME_INDICATOR_SPECS))
        rows = result[result["RegimeId"] == "Regime_vix_le_15"].set_index("RegimeOperator")
        self.assertEqual(set(rows.index), {"is true", "is false"})
        self.assertEqual(rows.loc["is true", "Regime"], "VIX")
        self.assertEqual(rows.loc["is true", "State"], "≤ 15")
        self.assertEqual(rows.loc["is true", "Operator"], "Is")
        self.assertEqual(rows.loc["is false", "Operator"], "Is not")
        self.assertEqual(rows.loc["is true", "Buysell"], "Buy")
        # Flags are [1, 1, 1, 0, 0, -1, -1, -1]: three days in the state, three in another,
        # two unlabeled (neither side).
        self.assertEqual(int(rows.loc["is true", "Trades"]), 3)
        self.assertEqual(int(rows.loc["is false", "Trades"]), 3)
        # Every other state is -1 on every day, so "is not" keeps all eight days.
        other = result[
            (result["RegimeId"] == "Regime_vix_15_20") & (result["RegimeOperator"] == "is false")
        ].iloc[0]
        self.assertEqual(int(other["Trades"]), 8)
        # Ranked by Sharpe, best first.
        self.assertEqual(int(result.iloc[0]["Trades"]), 8)

    def test_states_already_in_the_draft_are_skipped(self):
        strategy = _strategy(
            conditions=[
                BuilderCondition(left="RSI2", operator="<=", right="20", logic="AND"),
                BuilderCondition(left="Regime_vix_le_15", operator="is true", right="", logic="AND"),
            ]
        )
        result, _ = self._run(strategy)
        self.assertNotIn("Regime_vix_le_15", set(result["RegimeId"]))
        self.assertEqual(len(result), 2 * (len(REGIME_INDICATOR_SPECS) - 1))

    def test_per_symbol_states_take_the_full_path_only_with_confirm_symbols(self):
        atr = regime_spec("Regime_atr_expanding")
        vix = regime_spec("Regime_vix_le_15")
        self.assertFalse(regime_needs_full_path(_strategy(), atr))
        confirmed = _strategy(confirm_symbols=["QQQ"])
        self.assertTrue(regime_needs_full_path(confirmed, atr))
        self.assertFalse(regime_needs_full_path(confirmed, vix))


class RegimesAreNotSweptAsIndicators(unittest.TestCase):
    def test_indicator_sweep_specs_never_name_a_regime(self):
        names = set()
        for group in (
            indicator_sweep.BREADTH_SWEEPS,
            indicator_sweep.POST_2003_BREADTH_SWEEPS,
            indicator_sweep.PRICE_SWEEPS,
            indicator_sweep.VFI_SWEEPS,
            indicator_sweep.VWAP_SWEEPS,
        ):
            for spec in group:
                names.update(str(item) for item in spec[:2])
        self.assertFalse(names & REGIME_INDICATOR_IDS)
        self.assertFalse({name for name in names if name.startswith("Regime_")})


if __name__ == "__main__":
    unittest.main()
