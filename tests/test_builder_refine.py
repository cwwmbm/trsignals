import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
from api.builder_strategy import (
    backtest_builder_signal_combinations,
    backtest_builder_signal_sweep,
    combine_builder_buy_masks,
    draft_to_saved_strategy,
)
from api.schemas import (
    BuilderBacktestRequest,
    BuilderCondition,
    BuilderRefineOutcomeRequest,
    BuilderRefineRequest,
    SaveStrategyRequest,
)
from api.services import run_builder_backtest, run_builder_refine, run_builder_refine_outcome
from api.strategy_store import create_strategy, get_strategy_by_id


# Refine drops rows with fewer in-sample trades than config.MIN_IN_SAMPLE_TRADES, and a row with no
# trade count counts as zero. Mocked sweep rows therefore carry a count that clears it.
ENOUGH_TRADES = config.MIN_IN_SAMPLE_TRADES


def _draft_request(**overrides) -> BuilderBacktestRequest:
    payload = {
        "symbol": "SPY",
        "years": 25,
        "direction": "long",
        "hold_days": 2,
        "profit": 1,
        "name": "Draft RSI",
        "description": "",
        "conditions": [
            BuilderCondition(left="RSI2", operator="<=", right="20", logic="AND"),
        ],
        "sell_conditions": [],
    }
    payload.update(overrides)
    return BuilderBacktestRequest(**payload)


def _sample_data(rows: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=rows, freq="B"),
            "Close": np.linspace(100, 110, rows),
            "RSI2": np.linspace(10, 30, rows),
            "SMA200": np.linspace(99, 108, rows),
        }
    )


class BuilderStrategyHelperTests(unittest.TestCase):
    def test_draft_to_saved_strategy_uses_name_and_conditions(self):
        saved = draft_to_saved_strategy(_draft_request())
        self.assertEqual(saved.id, "draft")
        self.assertEqual(saved.name, "Draft RSI")
        self.assertEqual(saved.symbol, "SPY")
        self.assertIn("RSI(2)", saved.description)

    def test_draft_to_saved_strategy_normalizes_confirm_symbols(self):
        saved = draft_to_saved_strategy(
            _draft_request(confirm_symbols=["spy", "smh", "QQQ"])
        )
        self.assertEqual(saved.confirm_symbols, ["SMH", "QQQ"])

    def test_combine_builder_buy_masks_and_mode(self):
        data = _sample_data()
        primary = draft_to_saved_strategy(_draft_request())
        secondary = draft_to_saved_strategy(
            _draft_request(name="Other", conditions=[BuilderCondition(left="RSI2", operator="<=", right="25", logic="AND")])
        )
        buy_and, _, days, profit, description, _, is_long, _ = combine_builder_buy_masks(
            primary, secondary, data, "and"
        )
        buy_or, *_ = combine_builder_buy_masks(primary, secondary, data, "or")
        self.assertEqual(days, 2)
        self.assertEqual(profit, 1)
        self.assertTrue(is_long)
        self.assertIn("AND", description)
        self.assertTrue(buy_or.sum() >= buy_and.sum())


class BuilderRefineServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.temp_dir.name) / "strategies.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_combo_sweep_requires_same_symbol_saved_strategies(self):
        request = BuilderRefineRequest(
            mode="signal-combo-sweep",
            strategy=_draft_request(),
        )
        with patch("api.services.list_strategies", return_value=[]):
            with self.assertRaisesRegex(ValueError, "No saved strategies found for SPY"):
                run_builder_refine(request)

    def test_get_strategy_by_id(self):
        saved = create_strategy(
            SaveStrategyRequest(
                name="Saved RSI",
                symbol="SPY",
                direction="long",
                hold_days=2,
                profit=1,
                conditions=[BuilderCondition(left="RSI2", operator="<=", right="20", logic="AND")],
            ),
            store_path=self.store_path,
        )
        loaded = get_strategy_by_id(saved.id, store_path=self.store_path)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.name, "Saved RSI")

    def test_combo_sweep_uses_all_same_symbol_strategies(self):
        secondary_a = create_strategy(
            SaveStrategyRequest(
                name="Secondary A",
                symbol="SPY",
                direction="long",
                hold_days=2,
                profit=1,
                conditions=[BuilderCondition(left="RSI2", operator="<=", right="25", logic="AND")],
            ),
            store_path=self.store_path,
        )
        secondary_b = create_strategy(
            SaveStrategyRequest(
                name="Secondary B",
                symbol="SPY",
                direction="long",
                hold_days=2,
                profit=1,
                conditions=[BuilderCondition(left="Close", operator="<", right="SMA200", logic="AND")],
            ),
            store_path=self.store_path,
        )
        other_symbol = create_strategy(
            SaveStrategyRequest(
                name="Other Symbol",
                symbol="QQQ",
                direction="long",
                hold_days=2,
                profit=1,
                conditions=[BuilderCondition(left="RSI2", operator="<=", right="25", logic="AND")],
            ),
            store_path=self.store_path,
        )
        request = BuilderRefineRequest(
            mode="signal-combo-sweep",
            strategy=_draft_request(),
        )
        data = _sample_data()

        captured_secondaries = []

        def fake_sweep(_primary, secondaries, *_args, **_kwargs):
            captured_secondaries.extend(secondaries)
            return pd.DataFrame(
                [
                    {"Primary": "Draft RSI", "Secondary": "Secondary A", "Mode": "AND", "Sharpe": 1.2, "Trades": ENOUGH_TRADES},
                    {"Primary": "Draft RSI", "Secondary": "Secondary A", "Mode": "OR", "Sharpe": 1.0, "Trades": ENOUGH_TRADES},
                    {"Primary": "Draft RSI", "Secondary": "Secondary B", "Mode": "AND", "Sharpe": 0.9, "Trades": ENOUGH_TRADES},
                    {"Primary": "Draft RSI", "Secondary": "Secondary B", "Mode": "OR", "Sharpe": 0.8, "Trades": ENOUGH_TRADES},
                ]
            )

        with patch("api.services.load_ticker_data", return_value=data.copy()):
            with patch("api.services.list_strategies", return_value=[secondary_a, secondary_b, other_symbol]):
                with patch("api.services.backtest_builder_signal_sweep", side_effect=fake_sweep) as combo:
                    payload = run_builder_refine(request)

        self.assertEqual(len(payload["rows"]), 4)
        self.assertEqual(payload["meta"]["sample"], "in_sample")
        self.assertIn("in_sample_end", payload["meta"])
        combo.assert_called_once()
        self.assertIsNotNone(combo.call_args.kwargs.get("in_sample_end"))
        self.assertEqual([item.id for item in captured_secondaries], [secondary_a.id, secondary_b.id])

    def test_hold_days_refine_uses_prepared_frame_with_confirm_and_proxy(self):
        request = BuilderRefineRequest(
            mode="hold-days-sweep",
            strategy=_draft_request(
                confirm_symbols=["SMH", "QQQ"],
                proxy_symbol="SOXX",
                hold_on_buy_signal=True,
            ),
            max_days=3,
        )
        prepared = _sample_data()
        prepared["Buy"] = True
        prepared["Sell"] = False

        with patch(
            "api.services.prepare_builder_refine_frame",
            return_value=(prepared, 2, 1, True, "SOXX"),
        ) as prepare:
            with patch("api.services.bt.backtest_days") as backtest_days:
                backtest_days.return_value = pd.DataFrame(
                    [{"Days": 2, "Prf": 1, "Sharpe": 1.0, "Trades": ENOUGH_TRADES}]
                )
                payload = run_builder_refine(request)

        self.assertGreaterEqual(prepare.call_count, 2)
        self.assertIsNotNone(prepare.call_args.kwargs.get("in_sample_end"))
        backtest_days.assert_called_once_with(
            prepared,
            3,
            True,
            pnl_column="SOXX",
        )
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["meta"]["sample"], "in_sample")

    def test_indicator_sweep_returns_rows(self):
        request = BuilderRefineRequest(
            mode="indicator-sweep",
            strategy=_draft_request(),
            check_both=False,
            check_breadth=False,
        )
        data = _sample_data()

        with patch("api.services.load_ticker_data", return_value=data.copy()):
            with patch("api.services.indicator_tryout") as tryout:
                tryout.return_value = pd.DataFrame(
                    [
                        {"Indicator": "RSI2", "Value": 20, "Sharpe": 1.1, "Trades": ENOUGH_TRADES},
                        {"Indicator": "RSI5", "Value": 25, "Sharpe": 0.9, "Trades": ENOUGH_TRADES},
                    ]
                )
                payload = run_builder_refine(request)

        self.assertEqual(len(payload["rows"]), 2)
        self.assertEqual(payload["meta"]["sample"], "in_sample")
        tryout.assert_called_once()

    def test_indicator_sweep_uses_backtest_all_data_from_strategy(self):
        from types import SimpleNamespace

        request = BuilderRefineRequest(
            mode="indicator-sweep",
            strategy=_draft_request(
                custom_dataset_id="dataset-1",
                backtest_all_data=False,
            ),
            check_both=False,
            check_breadth=False,
        )
        data = _sample_data(rows=100)
        custom_dataset = SimpleNamespace(unavailable_indicator_ids=["Vix"], has_vwap=False)

        with patch("api.custom_data.custom_dataset_store") as store:
            store.require.return_value = custom_dataset
            store.load_backtest_frame.return_value = data.copy()
            with patch("api.services.indicator_tryout") as tryout:
                tryout.return_value = pd.DataFrame(
                    [{"Indicator": "RSI2", "Value": 20, "Sharpe": 1.1, "Trades": ENOUGH_TRADES}]
                )
                payload = run_builder_refine(request)

        store.load_backtest_frame.assert_called_once_with(
            "dataset-1",
            backtest_all_data=False,
        )
        self.assertEqual(len(payload["rows"]), 1)
        tryout.assert_called_once()
        self.assertFalse(tryout.call_args.kwargs.get("include_vwap_sweeps"))

    def test_indicator_sweep_includes_vwap_sweeps_for_custom_dataset_with_vwap(self):
        from types import SimpleNamespace

        request = BuilderRefineRequest(
            mode="indicator-sweep",
            strategy=_draft_request(
                custom_dataset_id="dataset-1",
                backtest_all_data=False,
            ),
            check_both=False,
            check_breadth=False,
        )
        data = _sample_data(rows=100)
        custom_dataset = SimpleNamespace(unavailable_indicator_ids=["Vix"], has_vwap=True)

        with patch("api.custom_data.custom_dataset_store") as store:
            store.require.return_value = custom_dataset
            store.load_backtest_frame.return_value = data.copy()
            with patch("api.services.indicator_tryout") as tryout:
                tryout.return_value = pd.DataFrame(
                    [{"Indicator": "Close_VWAP", "Value": 0.5, "Sharpe": 1.2, "Trades": ENOUGH_TRADES}]
                )
                payload = run_builder_refine(request)

        self.assertEqual(len(payload["rows"]), 1)
        tryout.assert_called_once()
        self.assertTrue(tryout.call_args.kwargs.get("include_vwap_sweeps"))

    def test_run_builder_backtest_uses_cross_symbol_when_confirm_symbols_set(self):
        request = _draft_request(confirm_symbols=["SMH", "QQQ"])
        data = _sample_data()
        executed = data.copy()

        with patch("api.services.load_ticker_data", return_value=data.copy()):
            with patch("api.services.compile_buy_mask"):
                with patch("api.services.compile_sell_mask"):
                    with patch("api.services.execute_saved_strategy") as execute_saved:
                        execute_saved.return_value = executed
                        with patch("api.services.detailed_backtest_payload") as payload:
                            payload.return_value = {"summary": {"PnL": 1}}
                            result = run_builder_backtest(request)

        execute_saved.assert_called_once()
        saved_strategy = execute_saved.call_args[0][1]
        self.assertEqual(saved_strategy.symbol, "SPY")
        self.assertEqual(saved_strategy.confirm_symbols, ["SMH", "QQQ"])
        payload.assert_called_once()
        self.assertEqual(result, {"summary": {"PnL": 1}})

    def test_refine_outcome_full_vs_in_sample_meta(self):
        request = BuilderRefineOutcomeRequest(
            mode="indicator-sweep",
            strategy=_draft_request(),
            sample="full",
            row={"Indicator": "RSI2", "Condition": "less", "Value": 20, "Buysell": "Buy"},
        )
        data = _sample_data(rows=20)
        prepared = data.copy()
        prepared["Buy"] = True
        prepared["Sell"] = False

        with patch("api.services.load_ticker_data", return_value=data.copy()):
            with patch(
                "api.services.prepare_builder_refine_frame",
                return_value=(prepared, 2, 1, True, None),
            ) as prepare:
                with patch("api.services.bt._run_indicator_threshold") as threshold:
                    threshold.return_value = {
                        "PnL": 1234.5,
                        "MaxDD": 12.34,
                        "Trades": 3,
                        "%Pstv": 66.6,
                        "CAGR": "10%",
                        "Sharpe": 1.23,
                        "Sortino": 1.45,
                        "Yearly": [{"year": 2020, "pnl_percent": 1.0}],
                    }
                    full_payload = run_builder_refine_outcome(request)

        self.assertEqual(full_payload["meta"]["sample"], "full")
        self.assertEqual(full_payload["meta"]["period_end"], "2020-01-28")
        self.assertIsNone(prepare.call_args.kwargs.get("in_sample_end"))
        self.assertEqual(full_payload["metrics"]["Trades"], 3)
        self.assertTrue(str(full_payload["metrics"]["PnL"]).startswith("$"))

        request.sample = "in_sample"
        with patch("api.services.load_ticker_data", return_value=data.copy()):
            with patch(
                "api.services.prepare_builder_refine_frame",
                return_value=(prepared, 2, 1, True, None),
            ) as prepare:
                with patch("api.services.bt._run_indicator_threshold") as threshold:
                    threshold.return_value = {
                        "PnL": 100.0,
                        "MaxDD": 5.0,
                        "Trades": 1,
                        "%Pstv": 100.0,
                        "CAGR": "5%",
                        "Sharpe": 0.5,
                        "Sortino": 0.6,
                        "Yearly": [],
                    }
                    is_payload = run_builder_refine_outcome(request)

        self.assertEqual(is_payload["meta"]["sample"], "in_sample")
        self.assertIsNotNone(prepare.call_args.kwargs.get("in_sample_end"))
        self.assertNotEqual(
            full_payload["meta"]["in_sample_end"],
            full_payload["meta"]["period_end"],
        )

    def test_refine_filters_rows_below_min_in_sample_trades(self):
        from api.services import _filter_rows_by_min_in_sample_trades

        rows = [
            {"Indicator": "RSI2", "Trades": 150, "Sharpe": 1.2},
            {"Indicator": "RSI5", "Trades": 99, "Sharpe": 1.5},
            {"Indicator": "ADX14", "Trades": 100, "Sharpe": 1.1},
        ]
        filtered = _filter_rows_by_min_in_sample_trades(rows, minimum=100)
        self.assertEqual([row["Indicator"] for row in filtered], ["RSI2", "ADX14"])

    def test_refine_counts_a_row_without_a_trade_count_as_zero_trades(self):
        from api.services import _filter_rows_by_min_in_sample_trades

        rows = [
            {"Indicator": "RSI2", "Sharpe": 1.2},
            {"Indicator": "RSI5", "Trades": None, "Sharpe": 1.1},
            {"Indicator": "ADX14", "Trades": ENOUGH_TRADES, "Sharpe": 1.0},
        ]
        self.assertEqual([r["Indicator"] for r in _filter_rows_by_min_in_sample_trades(rows)], ["ADX14"])
        self.assertEqual(len(_filter_rows_by_min_in_sample_trades(rows, minimum=0)), 3)

    def test_refine_applies_the_configured_minimum_and_reports_it(self):
        request = BuilderRefineRequest(
            mode="indicator-sweep", strategy=_draft_request(), check_both=False, check_breadth=False
        )
        rows = pd.DataFrame(
            [
                {"Indicator": "RSI2", "Value": 20, "Sharpe": 1.1, "Trades": ENOUGH_TRADES},
                {"Indicator": "RSI5", "Value": 25, "Sharpe": 1.5, "Trades": ENOUGH_TRADES - 1},
            ]
        )
        with patch("api.services.load_ticker_data", return_value=_sample_data()):
            with patch("api.services.indicator_tryout", return_value=rows):
                payload = run_builder_refine(request)
        self.assertEqual([row["Indicator"] for row in payload["rows"]], ["RSI2"])
        self.assertEqual(payload["meta"]["min_in_sample_trades"], ENOUGH_TRADES)

    def test_refine_attaches_full_period_for_rows_with_yearly(self):
        request = BuilderRefineRequest(
            mode="indicator-sweep",
            strategy=_draft_request(),
            check_both=False,
            check_breadth=False,
        )
        data = _sample_data(rows=20)
        prepared = data.copy()
        prepared["Buy"] = True
        prepared["Sell"] = False

        with patch("api.services.load_ticker_data", return_value=data.copy()):
            with patch(
                "api.services.prepare_builder_refine_frame",
                return_value=(prepared, 2, 1, True, None),
            ):
                with patch("api.services.indicator_tryout") as tryout:
                    tryout.return_value = pd.DataFrame(
                        [
                            {
                                "Indicator": "RSI2",
                                "Condition": "less",
                                "Value": 20,
                                "Buysell": "Buy",
                                "Sharpe": 1.1,
                                "Trades": ENOUGH_TRADES,
                                "Yearly": [{"year": 2020, "pnl_percent": 1.0}],
                            },
                            {
                                "Indicator": "RSI5",
                                "Condition": "less",
                                "Value": 25,
                                "Buysell": "Buy",
                                "Sharpe": 0.9,
                                "Trades": ENOUGH_TRADES,
                            },
                        ]
                    )
                    with patch("api.services._evaluate_refine_row_metrics") as evaluate:
                        evaluate.return_value = {
                            "PnL": 9999,
                            "MaxDD": 11.1,
                            "Trades": 7,
                            "%Pstv": 70.0,
                            "CAGR": "9%",
                            "Sharpe": 1.5,
                            "Sortino": 1.6,
                            "Yearly": [{"year": 2021, "pnl_percent": 2.0}],
                        }
                        payload = run_builder_refine(request)

        self.assertIn("FullPeriod", payload["rows"][0])
        self.assertEqual(payload["rows"][0]["FullPeriod"]["Trades"], 7)
        self.assertEqual(
            payload["rows"][0]["FullPeriod"]["Yearly"],
            [{"year": 2021, "pnl_percent": 2.0}],
        )
        self.assertNotIn("FullPeriod", payload["rows"][1])
        evaluate.assert_called_once()


class DraftBaselineTests(unittest.TestCase):
    """The draft as it stands is scored beside the sweep rows, on the same two windows."""

    @staticmethod
    def _frames():
        """Several years of bars, so the best-year exclusion leaves something to score."""
        rng = np.random.default_rng(5)
        dates = pd.bdate_range("2015-01-01", periods=900)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0007, 0.012, len(dates))), index=dates)
        buy = np.zeros(len(dates), dtype=bool)
        buy[10::15] = True
        full = pd.DataFrame(
            {
                "Date": dates,
                "Close": close.to_numpy(),
                "%Change": config.Leverage * close.pct_change().fillna(0.0).to_numpy(),
                "Buy": buy,
                "Sell": False,
            }
        )
        full.attrs["symbol"] = "SOXX"
        cut = dates[600]

        def prepare(strategy, **kwargs):
            end = kwargs.get("in_sample_end")
            data = full if end is None else full[full["Date"] <= end].copy()
            data.attrs["symbol"] = "SOXX"
            return data, 3, 2, True, None

        return full, cut, prepare

    def test_baseline_is_the_drafts_own_ranking_metrics_on_both_windows(self):
        from api import fast_metrics
        from api.services import _draft_baseline
        from tests.test_vehicles import settings

        full, cut, prepare = self._frames()
        primary = draft_to_saved_strategy(_draft_request())
        with settings(), patch("api.services.prepare_builder_refine_frame", side_effect=prepare):
            baseline = _draft_baseline(primary, None, years=25, labels={}, in_sample_end=cut)
            expected = {
                "in_sample": fast_metrics.run_metrics(full[full["Date"] <= cut].copy(), 3, 2, True),
                "full_period": fast_metrics.run_metrics(full.copy(), 3, 2, True),
            }
        for key, metrics in expected.items():
            self.assertEqual(baseline[key]["Trades"], metrics["Trades"])
            self.assertEqual(baseline[key]["%Pstv"], round(float(metrics["%Pstv"]), 1))
            self.assertEqual(baseline[key]["Sharpe"], round(float(metrics["Sharpe"]), 2))
            self.assertIn("PnL", baseline[key])
        self.assertLess(baseline["in_sample"]["Trades"], baseline["full_period"]["Trades"])

    def test_a_draft_that_cannot_be_scored_gives_none_and_does_not_raise(self):
        from api.services import _draft_baseline

        primary = draft_to_saved_strategy(_draft_request())
        with patch("api.services.prepare_builder_refine_frame", side_effect=ValueError("no data")):
            with self.assertLogs("api.services", level="WARNING"):
                self.assertIsNone(_draft_baseline(primary, None, years=25, labels={}, in_sample_end=None))

    def test_refine_meta_carries_the_baseline_except_for_a_basket_test(self):
        from tests.test_vehicles import settings

        _, _, prepare = self._frames()
        request = BuilderRefineRequest(
            mode="hold-days-sweep",
            strategy=_draft_request(confirm_symbols=["SMH"]),
            max_days=3,
        )
        rows = pd.DataFrame([{"Days": 2, "Prf": 1, "Sharpe": 1.0, "Trades": ENOUGH_TRADES}])
        with settings(), patch("api.services.prepare_builder_refine_frame", side_effect=prepare):
            with patch("api.services.bt.backtest_days", return_value=rows):
                payload = run_builder_refine(request)
        draft = payload["meta"]["draft"]
        self.assertEqual(set(draft), {"in_sample", "full_period"})
        self.assertGreater(draft["full_period"]["Trades"], 0)

        basket = BuilderRefineRequest(
            mode="basket-sweep", strategy=_draft_request(), basket_symbols=["QQQ"]
        )
        with patch("api.services.load_ticker_data", return_value=_sample_data()):
            with patch("api.services.run_basket_sweep", return_value=([{"Symbol": "QQQ"}], {})):
                self.assertNotIn("draft", run_builder_refine(basket)["meta"])


class BuilderComboSweepTests(unittest.TestCase):
    def test_backtest_builder_signal_combinations_produces_four_rows(self):
        primary = draft_to_saved_strategy(_draft_request())
        secondary = draft_to_saved_strategy(
            _draft_request(name="Secondary", conditions=[BuilderCondition(left="RSI2", operator="<=", right="25", logic="AND")])
        )
        data = _sample_data()

        with patch("api.builder_strategy.bt.execute_strategy") as execute_strategy:
            with patch("api.builder_strategy.bt._ranking_metrics") as ranking:
                execute_strategy.side_effect = lambda frame, *_args, **_kwargs: frame
                ranking.return_value = {
                    "PnL": 1000,
                    "MaxDD": 10.0,
                    "Trades": 5,
                    "%Pstv": 60.0,
                    "CAGR": 12.0,
                    "Sharpe": 1.0,
                    "Sortino": 1.1,
                    "Yearly": [],
                }
                results = backtest_builder_signal_combinations(primary, secondary, data, "SPY")

        self.assertEqual(len(results), 4)
        self.assertEqual(set(results["Mode"]), {"AND", "OR"})

    def test_backtest_builder_signal_sweep_produces_two_rows_per_secondary(self):
        primary = draft_to_saved_strategy(_draft_request())
        secondaries = [
            draft_to_saved_strategy(
                _draft_request(name="Secondary A", conditions=[BuilderCondition(left="RSI2", operator="<=", right="25", logic="AND")])
            ),
            draft_to_saved_strategy(
                _draft_request(name="Secondary B", conditions=[BuilderCondition(left="Close", operator="<", right="SMA200", logic="AND")])
            ),
        ]
        data = _sample_data()

        with patch("api.builder_strategy.bt.execute_strategy") as execute_strategy:
            with patch("api.builder_strategy.bt._ranking_metrics") as ranking:
                execute_strategy.side_effect = lambda frame, *_args, **_kwargs: frame
                ranking.return_value = {
                    "PnL": 1000,
                    "MaxDD": 10.0,
                    "Trades": 5,
                    "%Pstv": 60.0,
                    "CAGR": 12.0,
                    "Sharpe": 1.0,
                    "Sortino": 1.1,
                    "Yearly": [],
                }
                results = backtest_builder_signal_sweep(primary, secondaries, data, "SPY")

        self.assertEqual(len(results), 4)
        self.assertEqual(set(results["Primary"]), {"Draft RSI"})
        self.assertEqual(set(results["Secondary"]), {"Secondary A", "Secondary B"})
        self.assertEqual(set(results["Mode"]), {"AND", "OR"})


if __name__ == "__main__":
    unittest.main()
