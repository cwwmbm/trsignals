import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from api.allocation import (
    PORTFOLIO_CONSENSUS,
    REGIME_ADJUSTED,
    STRATEGY_CONFIDENCE,
    AllInModel,
    RegimeAdjustedModel,
    SignalState,
    resolve_allocation_model,
)
from api.allocation_engine import simulate_allocation
from api.portfolio_service import (
    _prepare_portfolio_simulation,
    allocation_settings,
    build_portfolio_overlay_frame,
    simulate_portfolio,
    simulate_portfolio_allocation,
)
from api.portfolio_store import create_portfolio, list_portfolios, update_portfolio
from api.regime_allocation import (
    PENALTY_PRESETS,
    RegimeAllocationContext,
    RegimePenaltySettings,
    active_set_change_bars,
    history_from_frame,
    regime_confidence,
)
from api.regime_scoring import BucketScore, EntityHistory
from api.schemas import SavePortfolioRequest, UpdatePortfolioRequest
from tests.test_allocation import (
    DATES,
    _market,
    _patched_portfolio,
    _signals,
    _strategy,
    _two_strategy_case,
)


def _score(n_trades, robustness=None, portfolio=None):
    return BucketScore(n_trades=n_trades, portfolio_score=portfolio, robustness_score=robustness)


class PenaltySettingsTests(unittest.TestCase):
    def test_defaults_are_the_medium_preset(self):
        settings = RegimePenaltySettings()
        self.assertEqual(settings.score, "robustness")
        self.assertEqual(settings.min_trades, 20)
        self.assertEqual(RegimePenaltySettings.from_preset().preset, "medium")
        self.assertEqual(RegimePenaltySettings.from_source(SimpleNamespace()).preset, "medium")

    def test_presets_have_the_agreed_tiers(self):
        expected = {
            # preset: (penalty at or below 10, penalty above 10 up to 25)
            "harsh": (1.0, 0.0),
            "strict": (0.5, 0.25),
            "medium": (1 / 3, 0.10),
            "low": (0.25, 0.05),
        }
        self.assertEqual(list(PENALTY_PRESETS), list(expected))
        for name, (low, high) in expected.items():
            settings = RegimePenaltySettings.from_preset(name)
            self.assertEqual((settings.low_cutoff, settings.high_cutoff), (10.0, 25.0), name)
            self.assertAlmostEqual(settings.penalty(5.0), low, msg=name)
            self.assertAlmostEqual(settings.penalty(10.0), low, msg=name)
            self.assertAlmostEqual(settings.penalty(15.0), high, msg=name)
            self.assertAlmostEqual(settings.penalty(25.0), high, msg=name)
            self.assertEqual(settings.penalty(40.0), 0.0, name)
            self.assertEqual(settings.min_trades, 20)

    def test_harsh_zeroes_the_confidence_of_one_poor_state(self):
        harsh = RegimePenaltySettings.from_preset("harsh")
        self.assertEqual(regime_confidence([_score(50, robustness=5.0)], harsh), 0.0)
        self.assertEqual(regime_confidence([_score(50, robustness=15.0)], harsh), 1.0)

    def test_from_source_reads_the_preset_and_score(self):
        settings = RegimePenaltySettings.from_source(
            SimpleNamespace(regime_penalty="low", regime_score="portfolio")
        )
        self.assertEqual((settings.preset, settings.score), ("low", "portfolio"))
        self.assertAlmostEqual(settings.low_penalty, 0.25)
        self.assertEqual(RegimePenaltySettings.from_source(SimpleNamespace(regime_penalty=None)).preset, "medium")

    def test_as_dict_reports_percent(self):
        data = RegimePenaltySettings.from_preset("strict").as_dict()
        self.assertEqual(data["preset"], "strict")
        self.assertAlmostEqual(data["low_penalty"], 50.0)
        self.assertAlmostEqual(data["high_penalty"], 25.0)
        self.assertEqual(data["min_trades"], 20)

    def test_rejects_settings_that_cannot_run(self):
        with self.assertRaises(ValueError):
            RegimePenaltySettings.from_preset("brutal")
        for kwargs in (
            {"score": "sharpe"},
            {"low_cutoff": 0},
            {"low_cutoff": 30, "high_cutoff": 25},
            {"low_cutoff": 25, "high_cutoff": 25},
            {"high_cutoff": 101},
            {"low_penalty": -0.1},
            {"high_penalty": 1.1},
            {"min_trades": 0},
            {"min_trades": 2.5},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RegimePenaltySettings(**kwargs)


class ConfidenceTests(unittest.TestCase):
    settings = RegimePenaltySettings.from_preset("strict")

    def test_no_evidence_means_full_confidence(self):
        self.assertEqual(regime_confidence([], self.settings), 1.0)
        self.assertEqual(regime_confidence([None, None], self.settings), 1.0)

    def test_tiers_and_boundaries_are_inclusive(self):
        def confidence(score):
            return regime_confidence([_score(50, robustness=score)], self.settings)

        self.assertAlmostEqual(confidence(5.0), 0.5)
        self.assertAlmostEqual(confidence(10.0), 0.5)  # at the low cutoff
        self.assertAlmostEqual(confidence(10.01), 0.75)
        self.assertAlmostEqual(confidence(25.0), 0.75)  # at the high cutoff
        self.assertAlmostEqual(confidence(25.01), 1.0)
        self.assertAlmostEqual(confidence(100.0), 1.0)

    def test_states_with_too_few_trades_cost_nothing(self):
        self.assertEqual(regime_confidence([_score(19, robustness=5.0)], self.settings), 1.0)
        self.assertAlmostEqual(regime_confidence([_score(20, robustness=5.0)], self.settings), 0.5)

    def test_missing_score_is_skipped(self):
        self.assertEqual(regime_confidence([_score(99, robustness=None)], self.settings), 1.0)

    def test_penalties_add_up_and_floor_at_zero(self):
        states = [_score(50, robustness=5.0), _score(50, robustness=20.0)]
        self.assertAlmostEqual(regime_confidence(states, self.settings), 0.25)
        states = [_score(50, robustness=5.0)] * 3
        self.assertEqual(regime_confidence(states, self.settings), 0.0)

    def test_score_kind_picks_the_score(self):
        state = _score(50, robustness=90.0, portfolio=5.0)
        self.assertEqual(regime_confidence([state], RegimePenaltySettings.from_preset("strict", "robustness")), 1.0)
        self.assertAlmostEqual(regime_confidence([state], RegimePenaltySettings.from_preset("strict", "portfolio")), 0.5)


class ActiveSetTests(unittest.TestCase):
    def test_marks_joins_leaves_and_swaps_but_not_the_final_exit(self):
        active = np.array(
            [
                [1, 1, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 0, 1, 1],
            ],
            dtype=bool,
        )
        # bar 0: first entry. bar 2: b joins. bar 3: a leaves while b stays.
        # bar 4: everyone is out (flat, nothing to decide). bar 5: a re-enters.
        # bar 6: swap a -> b. bar 7: unchanged.
        self.assertEqual(list(active_set_change_bars(active)), [0, 2, 3, 5, 6])

    def test_nothing_ever_active(self):
        self.assertEqual(len(active_set_change_bars(np.zeros((2, 5), dtype=bool))), 0)


class ModelTests(unittest.TestCase):
    model = RegimeAdjustedModel()

    def _exposure(self, active=1, eligible=2, **context):
        return self.model.target_exposure(SignalState(active, eligible, context))

    def test_formula(self):
        self.assertAlmostEqual(self._exposure(**{STRATEGY_CONFIDENCE: 1.0, PORTFOLIO_CONSENSUS: 1.0}), 1.0)
        self.assertAlmostEqual(self._exposure(**{STRATEGY_CONFIDENCE: 1.0, PORTFOLIO_CONSENSUS: 0.0}), 0.7)
        self.assertAlmostEqual(self._exposure(**{STRATEGY_CONFIDENCE: 0.5, PORTFOLIO_CONSENSUS: 0.5}), 0.425)
        self.assertAlmostEqual(self._exposure(**{STRATEGY_CONFIDENCE: 0.0, PORTFOLIO_CONSENSUS: 1.0}), 0.0)

    def test_flat_when_nothing_is_active_and_neutral_without_context(self):
        self.assertEqual(self._exposure(active=0), 0.0)
        self.assertEqual(self._exposure(active=1), 1.0)  # no evidence: behaves like all-in

    def test_clamped_and_pure(self):
        self.assertEqual(self._exposure(**{STRATEGY_CONFIDENCE: 9.0, PORTFOLIO_CONSENSUS: 9.0}), 1.0)
        self.assertEqual(self._exposure(**{STRATEGY_CONFIDENCE: float("nan")}), 0.0)
        context = {STRATEGY_CONFIDENCE: 0.6, PORTFOLIO_CONSENSUS: 0.4}
        state = SignalState(1, 2, context)
        self.assertEqual(self.model.target_exposure(state), self.model.target_exposure(state))
        self.assertEqual(context, {STRATEGY_CONFIDENCE: 0.6, PORTFOLIO_CONSENSUS: 0.4})

    def test_resolves_by_name_and_only_asks_again_when_the_active_set_changes(self):
        model = resolve_allocation_model(REGIME_ADJUSTED)
        self.assertIsInstance(model, RegimeAdjustedModel)
        self.assertFalse(model.reacts_to_eligibility)
        self.assertTrue(getattr(AllInModel(), "reacts_to_eligibility", True))


def _engine_inputs(active, eligible):
    n = active.shape[1]
    track, close = _market(changes=[0.0] * n, dates=DATES[:n])
    return dict(
        dates=DATES[:n],
        active=active,
        eligible=eligible,
        pnl_symbols=["AAA"] * active.shape[0],
        track_changes=track,
        close_series=close,
        start_capital=1000.0,
    )


class EngineTests(unittest.TestCase):
    def test_decisions_set_the_exposure_and_a_leaving_strategy_can_raise_it(self):
        strategies, signals = _two_strategy_case()
        from api.portfolio_service import _effective_signal_matrices

        active, eligible, _ = _effective_signal_matrices(strategies, signals, DATES)
        self.assertEqual(list(active_set_change_bars(active)), [0, 1, 3])
        decisions = {
            0: {STRATEGY_CONFIDENCE: 0.5, PORTFOLIO_CONSENSUS: 1.0},
            1: {STRATEGY_CONFIDENCE: 0.8, PORTFOLIO_CONSENSUS: 0.0},
            3: {STRATEGY_CONFIDENCE: 1.0, PORTFOLIO_CONSENSUS: 1.0},
        }
        simulation = simulate_allocation(
            **_engine_inputs(active, eligible),
            model=RegimeAdjustedModel(),
            decisions=decisions,
        )
        targets = simulation.target_exposure
        self.assertAlmostEqual(targets[0], 0.5)
        self.assertAlmostEqual(targets[1], 0.56)
        self.assertTrue(np.isnan(targets[2]))  # nothing changed, nothing decided
        self.assertAlmostEqual(targets[3], 1.0)  # a left, b stays: exposure went up
        self.assertAlmostEqual(targets[4], 0.0)
        np.testing.assert_allclose(simulation.post_weight[:5], [0.5, 0.56, 0.56, 1.0, 0.0], atol=1e-9)

    def test_eligibility_changes_alone_do_not_trigger_a_new_decision(self):
        active = np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=bool)
        eligible = np.array([[1, 1, 1, 1], [0, 0, 1, 1]], dtype=bool)
        decisions = {0: {STRATEGY_CONFIDENCE: 0.5, PORTFOLIO_CONSENSUS: 1.0}}
        regime = simulate_allocation(
            **_engine_inputs(active, eligible), model=RegimeAdjustedModel(), decisions=decisions
        )
        self.assertAlmostEqual(regime.target_exposure[0], 0.5)
        self.assertTrue(np.isnan(regime.target_exposure[2]))  # strategy 2 became eligible
        # Models that size from the eligible count do re-decide there.
        all_in = simulate_allocation(**_engine_inputs(active, eligible), model=AllInModel())
        self.assertFalse(np.isnan(all_in.target_exposure[2]))

    def test_without_decisions_it_is_all_in(self):
        strategies, signals = _two_strategy_case()
        from api.portfolio_service import _effective_signal_matrices

        active, eligible, _ = _effective_signal_matrices(strategies, signals, DATES)
        regime = simulate_allocation(**_engine_inputs(active, eligible), model=RegimeAdjustedModel())
        reference = simulate_allocation(**_engine_inputs(active, eligible), model=AllInModel())
        np.testing.assert_allclose(regime.post_weight, reference.post_weight)


N_BARS = 900
REGIME_DATES = pd.bdate_range("2020-01-01", periods=N_BARS)
# Alternating 20-bar blocks. A trade belongs to the block it was entered in.
SPY_LABELS = np.array(["bull" if (i // 20) % 2 == 0 else "bear" for i in range(N_BARS)], dtype=object)


def _history(*, bear_return=-0.01, bull_return=0.01, flip_after=None, dates=REGIME_DATES):
    """One trade every 10 bars: enter on bar 10k, hold the next 4 bars, exit on bar 10k+4."""
    n = len(dates)
    long_in = np.zeros(n, dtype=bool)
    hold = np.zeros(n, dtype=bool)
    long_out = np.zeros(n, dtype=bool)
    returns = np.zeros(n)
    trade_pnl = np.zeros(n)
    for start in range(0, n - 5, 10):
        step = bull_return if SPY_LABELS[start] == "bull" else bear_return
        if flip_after is not None and start > flip_after:
            step = abs(step)
        long_in[start] = True
        for bar in range(start + 1, start + 5):
            hold[bar] = True
            returns[bar] = step
        long_out[start + 4] = True
        trade_pnl[start + 4] = (1 + step) ** 4 - 1
    return EntityHistory(
        dates=pd.DatetimeIndex(dates),
        bar_returns=returns,
        long_in=long_in,
        hold=hold,
        long_out=long_out,
        trade_pnl=trade_pnl,
        labels={"spy": SPY_LABELS},
    )


def _active_of(history):
    return ((history.hold & ~history.long_out) | history.long_in)[None, :]


def _context(histories, *, settings=None, calls=None):
    ids = list(histories)
    first = histories[ids[0]]
    full_active = np.vstack([_active_of(histories[i]) for i in ids])

    def strategy_history(strategy_id):
        if calls is not None:
            calls.setdefault("strategy", []).append(strategy_id)
        return histories[strategy_id]

    def baseline_history(member_ids):
        if calls is not None:
            calls.setdefault("baseline", []).append(member_ids)
        return histories[member_ids[0]]

    context = RegimeAllocationContext(
        settings or replace(RegimePenaltySettings.from_preset("strict"), min_trades=5),
        strategy_ids=ids,
        master_dates=first.dates,
        full_active=full_active,
        strategy_history=strategy_history,
        baseline_history=baseline_history,
        current_labels={"spy": SPY_LABELS},
        dimensions=("spy",),
    )
    return context, full_active


class ContextTests(unittest.TestCase):
    def test_penalizes_the_regime_that_has_hurt_and_leaves_the_other_alone(self):
        context, active = _context({"s": _history()})
        decisions = context.decisions(["s"], active)
        # Bar 10: too little history -> no evidence -> full confidence.
        self.assertEqual(decisions[10], {STRATEGY_CONFIDENCE: 1.0, PORTFOLIO_CONSENSUS: 1.0})
        # Bar 720 is a bull block, bar 710 a bear block, long after both have history.
        self.assertEqual(SPY_LABELS[720], "bull")
        self.assertEqual(SPY_LABELS[710], "bear")
        self.assertAlmostEqual(decisions[720][STRATEGY_CONFIDENCE], 1.0)
        self.assertAlmostEqual(decisions[710][STRATEGY_CONFIDENCE], 0.5)  # low-score penalty
        self.assertAlmostEqual(decisions[710][PORTFOLIO_CONSENSUS], 0.5)
        exposure = RegimeAdjustedModel().target_exposure(SignalState(1, 1, decisions[710]))
        self.assertAlmostEqual(exposure, 0.5 * (0.7 + 0.3 * 0.5))

    def test_strategy_confidence_averages_the_active_strategies(self):
        always_wins = _history(bear_return=0.01)
        context, active = _context({"hurt": _history(), "fine": always_wins})
        both = context.decisions(["hurt", "fine"], active)
        self.assertEqual(SPY_LABELS[710], "bear")
        self.assertAlmostEqual(both[710][STRATEGY_CONFIDENCE], (0.5 + 1.0) / 2)

    def test_decisions_only_use_the_past(self):
        base = _history()
        changed = _history(flip_after=705)
        context_a, active = _context({"s": base})
        context_b, _ = _context({"s": changed})
        a = context_a.decisions(["s"], active)
        b = context_b.decisions(["s"], active)
        before = [bar for bar in a if bar <= 705]
        self.assertGreater(len(before), 50)
        for bar in before:
            self.assertEqual(a[bar], b[bar], f"bar {bar} looked at the future")
        # The trades after bar 705 do matter later on: flip nearly all of them and the bear
        # regime stops being penalized.
        early_flip, _ = _context({"s": _history(flip_after=100)})
        self.assertEqual(SPY_LABELS[870], "bear")
        self.assertAlmostEqual(a[870][STRATEGY_CONFIDENCE], 0.5)
        self.assertAlmostEqual(early_flip.decisions(["s"], active)[870][STRATEGY_CONFIDENCE], 1.0)

    def test_smaller_portfolios_reuse_strategy_scores(self):
        calls = {}
        context, active = _context({"a": _history(), "b": _history(bear_return=0.01)}, calls=calls)
        context.decisions(["a", "b"], active)
        context.decisions(["a"], active[:1])
        context.decisions(["b"], active[1:])
        self.assertEqual(sorted(calls["strategy"]), ["a", "b"])  # each scored once
        self.assertEqual(len(calls["baseline"]), 3)  # one baseline per distinct portfolio
        again = context.decisions(["a", "b"], active)
        self.assertEqual(len(calls["baseline"]), 3)
        self.assertEqual(again[710][STRATEGY_CONFIDENCE], context.decisions(["a", "b"], active)[710][STRATEGY_CONFIDENCE])

    def test_refuses_a_different_calendar(self):
        context, active = _context({"s": _history()})
        with self.assertRaises(ValueError):
            context.decisions(["s"], active, REGIME_DATES + pd.Timedelta(days=1))
        with self.assertRaises(ValueError):
            context.decisions(["s"], active[:, :-1])

    def test_unlabeled_states_are_skipped(self):
        context, active = _context({"s": _history()})
        context._labels = {"spy": np.array([None] * N_BARS, dtype=object)}
        decisions = context.decisions(["s"], active)
        self.assertTrue(all(row[STRATEGY_CONFIDENCE] == 1.0 for row in decisions.values()))

    def test_history_from_frame_matches_the_columns_scoring_reads(self):
        history = _history()
        frame = pd.DataFrame(
            {
                "Date": history.dates,
                "RollingPnL": 1000.0 * np.cumprod(1.0 + history.bar_returns),
                "LongTradeIn": history.long_in,
                "HoldLong": history.hold,
                "LongTradeOut": history.long_out,
                "TradePnL": history.trade_pnl,
            }
        )
        rebuilt = history_from_frame(frame, history.labels)
        np.testing.assert_allclose(rebuilt.bar_returns, history.bar_returns, atol=1e-12)
        np.testing.assert_array_equal(rebuilt.long_in, history.long_in)
        np.testing.assert_array_equal(rebuilt.hold, history.hold)
        np.testing.assert_array_equal(rebuilt.long_out, history.long_out)


class _StubContext:
    """Stands in for the scorer so service-level tests don't need market data."""

    def __init__(self, decisions):
        self._decisions = decisions

    def decisions(self, ids, active, dates=None, settings=None):
        return self._decisions


TWO_STRATEGY_DECISIONS = {
    0: {STRATEGY_CONFIDENCE: 0.5, PORTFOLIO_CONSENSUS: 1.0},
    1: {STRATEGY_CONFIDENCE: 0.8, PORTFOLIO_CONSENSUS: 0.0},
    3: {STRATEGY_CONFIDENCE: 1.0, PORTFOLIO_CONSENSUS: 1.0},
}


def _bind_stub(allocation, *args, **kwargs):
    if not allocation.is_regime_adjusted:
        return allocation
    return replace(allocation, regime_context=_StubContext(TWO_STRATEGY_DECISIONS))


class ServiceTests(unittest.TestCase):
    def test_settings_carry_the_penalties_only_for_the_regime_model(self):
        regime = allocation_settings(
            SimpleNamespace(allocation_model="regime_adjusted", regime_penalty="low")
        )
        self.assertTrue(regime.is_regime_adjusted)
        self.assertAlmostEqual(regime.regime.low_penalty, 0.25)
        self.assertEqual(regime.regime.min_trades, 20)
        self.assertIsNone(allocation_settings(SimpleNamespace(allocation_model="proportional")).regime)
        self.assertIsNone(allocation_settings(SimpleNamespace()).regime)
        with self.assertRaises(ValueError):
            allocation_settings(SimpleNamespace(allocation_model="regime_adjusted", regime_penalty="brutal"))

    def test_the_model_cannot_run_without_its_scorer(self):
        strategies, signals = _two_strategy_case()
        track, close = _market()
        with self.assertRaises(ValueError):
            simulate_portfolio_allocation(
                strategies,
                signals,
                model=RegimeAdjustedModel(),
                global_proxy=None,
                track_changes=track,
                close_series=close,
            )

    def test_payload_reports_the_model_the_penalties_and_the_exposure_steps(self):
        request = _patched_portfolio(
            self,
            dict(
                allocation_model="regime_adjusted",
                regime_score="portfolio",
                regime_penalty="strict",
                transaction_cost_bps=0.0,
            ),
        )
        with patch("api.portfolio_service.bind_regime_context", side_effect=_bind_stub):
            payload = simulate_portfolio(request)
        allocation = payload["allocation"]
        self.assertEqual(allocation["model"], "regime_adjusted")
        self.assertEqual(allocation["regime"]["score"], "portfolio")
        self.assertEqual(allocation["regime"]["preset"], "strict")
        self.assertAlmostEqual(allocation["regime"]["low_penalty"], 50.0)
        self.assertAlmostEqual(allocation["regime"]["high_penalty"], 25.0)
        steps = [step["exposure"] for step in allocation["exposure_steps"]]
        self.assertEqual([round(x, 6) for x in steps], [0.5, 0.56, 1.0, 0.0])
        self.assertIn("regime adjusted", payload["summary"]["description"])
        self.assertEqual(len(payload["contribution"]), 2)
        self.assertGreater(len(payload["positions"]["episodes"]), 0)

    def test_all_in_payload_has_no_regime_section(self):
        payload = simulate_portfolio(_patched_portfolio(self))
        self.assertNotIn("regime", payload["allocation"])

    def test_strategy_backtests_are_kept_only_when_the_scorer_needs_them(self):
        seen = []

        def load(strategy, **kwargs):
            seen.append(kwargs["keep_executed"])
            return _two_strategy_case()[1][strategy.id]

        for model, expected in (("all_in", False), ("proportional", False), ("regime_adjusted", True)):
            request = _patched_portfolio(self, dict(allocation_model=model))
            seen.clear()
            with patch("api.portfolio_service._load_strategy_signals", side_effect=load):
                _prepare_portfolio_simulation(request, years=25)
            self.assertEqual(seen, [expected, expected], model)

    def test_the_scan_sizes_a_regime_portfolio_like_a_linear_one(self):
        request = _patched_portfolio(self, dict(allocation_model="regime_adjusted"))
        frame, *_ = build_portfolio_overlay_frame(request, years=1, signals_only=True)
        self.assertEqual(
            [round(x, 6) for x in frame["TargetExposure"].iloc[:5]], [0.5, 1.0, 1.0, 0.5, 0.0]
        )


class _PenaltyStubContext:
    """Confidence falls with the preset's low penalty on every decision bar."""

    def decisions(self, ids, active, dates=None, settings=None):
        confidence = 1.0 - (settings.low_penalty if settings else 0.0)
        return {
            int(bar): {STRATEGY_CONFIDENCE: confidence, PORTFOLIO_CONSENSUS: 1.0}
            for bar in active_set_change_bars(active)
        }


class ComparisonTests(unittest.TestCase):
    def _compare(self, **kwargs):
        from api.portfolio_service import simulate_portfolio_allocation_comparison

        overlap = kwargs.pop("overlap_mode", None)
        request = _patched_portfolio(self, kwargs)
        if overlap:
            request.overlap_mode = overlap

        def bind(allocation, *args, **kw):
            return replace(allocation, regime_context=_PenaltyStubContext())

        with patch("api.portfolio_service.bind_regime_context", side_effect=bind):
            return simulate_portfolio_allocation_comparison(request)

    def test_lists_all_in_then_every_regime_preset_then_proportional(self):
        payload = self._compare()
        self.assertEqual(
            [row["model"] for row in payload["models"]],
            [
                "all_in",
                "regime_adjusted_harsh",
                "regime_adjusted_strict",
                "regime_adjusted_medium",
                "regime_adjusted_low",
                "proportional_linear",
                "proportional_early_conviction",
                "proportional_confirmation_required",
            ],
        )
        labels = {row["model"]: row["label"] for row in payload["models"]}
        self.assertEqual(labels["regime_adjusted_harsh"], "Regime · Harsh")
        first = payload["equity_curves"][0]
        for row in payload["models"]:
            self.assertAlmostEqual(first[row["model"]], 15000, msg=row["model"])

    def test_presets_run_progressively_less_exposure_as_they_get_harsher(self):
        by_model = {row["model"]: row for row in self._compare()["models"]}
        exposure = [
            by_model[f"regime_adjusted_{name}"]["analytics"]["average_exposure_pct"]
            for name in ("harsh", "strict", "medium", "low")
        ]
        self.assertEqual(exposure[0], 0.0)  # harsh: a poor state takes all confidence
        self.assertTrue(all(a < b for a, b in zip(exposure, exposure[1:])), exposure)
        self.assertGreater(by_model["all_in"]["analytics"]["average_exposure_pct"], exposure[-1])

    def test_all_in_row_is_real_all_in_whatever_model_or_overlap_the_request_carries(self):
        rows = []
        for model, overlap in (("all_in", "hold_until_all_exit"), ("regime_adjusted", "first_signal_only")):
            payload = self._compare(allocation_model=model, overlap_mode=overlap)
            rows.append(payload["models"][0])
        self.assertEqual(rows[0]["model"], "all_in")
        self.assertAlmostEqual(rows[0]["metrics"]["rolling_pnl"], rows[1]["metrics"]["rolling_pnl"])
        self.assertAlmostEqual(rows[1]["analytics"]["average_exposure_when_invested_pct"], 100.0)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "portfolios.json"
        patcher = patch(
            "api.portfolio_store.get_strategy_by_id",
            return_value=SimpleNamespace(id="s1", name="Long SPY", direction="long"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _request(self, **kwargs):
        return SavePortfolioRequest(name="Regime book", strategy_ids=["s1"], **kwargs)

    def test_saving_a_portfolio_keeps_its_allocation_model_and_penalty(self):
        saved = create_portfolio(
            self._request(
                allocation_model="regime_adjusted",
                transaction_cost_bps=2.5,
                regime_score="portfolio",
                regime_penalty="harsh",
            ),
            store_path=self.path,
        )
        [loaded] = list_portfolios(store_path=self.path)
        self.assertEqual(loaded.id, saved.id)
        self.assertEqual(loaded.allocation_model, "regime_adjusted")
        self.assertEqual(loaded.transaction_cost_bps, 2.5)
        self.assertEqual((loaded.regime_score, loaded.regime_penalty), ("portfolio", "harsh"))
        # And the loaded portfolio drives the simulation the same way the form does.
        settings = allocation_settings(loaded)
        self.assertEqual((settings.regime.preset, settings.regime.score), ("harsh", "portfolio"))
        self.assertAlmostEqual(settings.regime.low_penalty, 1.0)

    def test_regime_penalty_defaults_to_medium(self):
        create_portfolio(self._request(allocation_model="regime_adjusted"), store_path=self.path)
        [loaded] = list_portfolios(store_path=self.path)
        self.assertEqual((loaded.regime_penalty, loaded.regime_score), ("medium", "robustness"))

    def test_proportional_and_all_in_portfolios_round_trip_too(self):
        create_portfolio(
            self._request(allocation_model="proportional", proportional_mode="confirmation_required"),
            store_path=self.path,
        )
        [loaded] = list_portfolios(store_path=self.path)
        self.assertEqual((loaded.allocation_model, loaded.proportional_mode), ("proportional", "confirmation_required"))

    def test_updating_changes_only_what_was_sent(self):
        saved = create_portfolio(self._request(allocation_model="regime_adjusted"), store_path=self.path)
        updated = update_portfolio(
            saved.id, UpdatePortfolioRequest(regime_penalty="low", regime_score="portfolio"), store_path=self.path
        )
        self.assertEqual((updated.regime_penalty, updated.regime_score), ("low", "portfolio"))
        self.assertEqual(updated.allocation_model, "regime_adjusted")
        switched = update_portfolio(saved.id, UpdatePortfolioRequest(allocation_model="all_in"), store_path=self.path)
        self.assertEqual(switched.allocation_model, "all_in")
        self.assertEqual(switched.regime_penalty, "low")  # kept, in case it is switched back

    def test_unknown_penalty_names_are_rejected_by_the_schema(self):
        with self.assertRaises(ValueError):
            self._request(allocation_model="regime_adjusted", regime_penalty="brutal")
        with self.assertRaises(ValueError):
            UpdatePortfolioRequest(regime_penalty="brutal")

    def test_portfolios_saved_before_regime_allocation_load_with_defaults(self):
        self.path.write_text(
            '[{"id": "old", "name": "Old", "strategy_ids": ["s1"], "overlap_mode": "first_signal_only",'
            ' "created_at": "2024-01-01T00:00:00+00:00", "updated_at": "2024-01-01T00:00:00+00:00"}]'
        )
        [loaded] = list_portfolios(store_path=self.path)
        self.assertEqual(loaded.allocation_model, "all_in")
        self.assertEqual(loaded.regime_penalty, "medium")

    def test_portfolios_saved_with_numeric_penalties_still_load(self):
        self.path.write_text(
            '[{"id": "old", "name": "Old", "strategy_ids": ["s1"], "allocation_model": "regime_adjusted",'
            ' "regime_low_cutoff": 10, "regime_low_penalty": 50, "regime_min_trades": 20,'
            ' "created_at": "2024-01-01T00:00:00+00:00", "updated_at": "2024-01-01T00:00:00+00:00"}]'
        )
        [loaded] = list_portfolios(store_path=self.path)
        self.assertEqual(loaded.regime_penalty, "medium")


class OverlapTests(unittest.TestCase):
    def test_every_portfolio_holds_until_all_exit(self):
        for stored in ("first_signal_only", "hold_until_all_exit"):
            with self.subTest(stored=stored):
                request = _patched_portfolio(self)
                request.overlap_mode = stored
                prepared = _prepare_portfolio_simulation(request, years=25)
                self.assertEqual(prepared[3], "hold_until_all_exit")

    def test_the_all_in_result_ignores_a_stored_first_signal_setting(self):
        results = []
        for stored in ("first_signal_only", "hold_until_all_exit"):
            request = _patched_portfolio(self)
            request.overlap_mode = stored
            results.append(simulate_portfolio(request)["summary"]["rolling_pnl"])
        self.assertAlmostEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
