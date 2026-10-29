import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from api.allocation import AllInModel, RegimeAdjustedModel, proportional_model
from api.portfolio_service import (
    AllocationSettings, StrategySignals, _effective_signal_matrices, simulate_portfolio_allocation,
    simulate_portfolio_frame, _positions_payload, START_CAPITAL,
)
from api.schemas import SavePortfolioRequest, UpdatePortfolioRequest, PortfolioSimulateRequest
from api.regime_allocation import ALLOCATION_REGIME_DIMENSIONS
from api.portfolio_store import create_portfolio, update_portfolio, delete_portfolio, list_portfolios
from tests.test_allocation import _signals, _strategy

DATES = pd.bdate_range('2020-01-01', periods=10)


def signal(identity, symbol, start=0, end=8):
    events = [''] * len(DATES)
    if start is not None:
        events[start] = 'buy'
    if end is not None and start is not None:
        events[end] = 'sell'
    return _signals(identity, DATES, events, symbol=symbol)


def child(members, signals, model=None, name='Child', cost=0, context=None):
    active, eligible, _ = _effective_signal_matrices(members, signals, DATES)
    active, eligible = active.any(axis=0), eligible.any(axis=0)
    previous = np.r_[False, active[:-1]]
    identity = f'portfolio:{name}'
    member = SimpleNamespace(id=identity, name=name, symbol='Portfolio', direction='long',
        proxy_symbol=None, hold_days=5, profit=1,
        nested_spec=SimpleNamespace(members=members, signals=signals, proxy=None, years=1,
            allocation=AllocationSettings(model=model or AllInModel(), cost_bps=cost,
                                          regime_context=context)))
    state = StrategySignals(identity, name, 'Portfolio', DATES, active & ~previous,
                            previous, previous & ~active, entry_ready=eligible)
    return member, state


def run(members, states, model=None, returns=None, cost=0):
    symbols = ('AAA', 'BBB', 'CCC')
    track = {s: pd.Series((returns or {}).get(s, np.zeros(len(DATES))), index=DATES) for s in symbols}
    closes = {s: pd.Series(100 * np.cumprod(1 + track[s]), index=DATES) for s in symbols}
    return simulate_portfolio_allocation(members, states, model=model or AllInModel(), cost_bps=cost,
        global_proxy=None, track_changes=track, close_series=closes, master_dates=DATES)


class NestedEngineTests(unittest.TestCase):
    def fixture(self, child_model=None, cost=0):
        a, idle, b = _strategy('a', 'AAA'), _strategy('idle', 'CCC'), _strategy('b', 'BBB')
        c, cs = child([a, idle], {'a': signal('a', 'AAA'), 'idle': signal('idle', 'CCC', None)},
                      model=child_model or proportional_model('linear'), cost=cost)
        return [c, b], {c.id: cs, b.id: signal('b', 'BBB')}

    def test_child_exposure_multiplies_parent_budget(self):
        members, states = self.fixture()
        result = run(members, states)
        self.assertAlmostEqual(result.asset_weight_post['AAA'][0], .25)
        self.assertAlmostEqual(result.asset_weight_post['BBB'][0], .50)
        self.assertAlmostEqual(result.cash[0] / result.equity[0], .25)

    def test_rejection_keeps_budget_cash(self):
        class Reject:
            id = 'reject'
            label = 'Reject'
            def target_exposure(self, state):
                return 0.0
        members, states = self.fixture(Reject())
        result = run(members, states)
        self.assertAlmostEqual(result.post_weight[0], .5)
        self.assertAlmostEqual(result.cash[0], START_CAPITAL / 2)
        self.assertEqual(result.active_count[0], 2)
        self.assertFalse(any(t.symbol == 'AAA' for t in result.transactions))

    def test_nonlinear_parent_counts_child_once(self):
        members, states = self.fixture()
        idle = _strategy('root-idle', 'CCC')
        members.append(idle)
        states[idle.id] = signal(idle.id, 'CCC', None)
        result = run(members, states, proportional_model('early_conviction'))
        self.assertAlmostEqual(result.post_weight[0], np.sqrt(2/3) * .75)

    def test_child_changes_without_parent_active_change(self):
        a, a2, b = _strategy('a', 'AAA'), _strategy('a2', 'AAA'), _strategy('b', 'BBB')
        c, cs = child([a, a2], {'a': signal('a', 'AAA'), 'a2': signal('a2', 'AAA', 3)},
                      proportional_model('linear'))
        result = run([c, b], {c.id: cs, b.id: signal('b', 'BBB')})
        self.assertAlmostEqual(result.post_weight[2], .75)
        self.assertAlmostEqual(result.post_weight[3], 1)
        self.assertEqual(result.active_count[2], result.active_count[3])
        self.assertTrue(result.rebalanced[3])

    def test_parent_join_reduces_existing_child(self):
        members, states = self.fixture(AllInModel())
        states['b'] = signal('b', 'BBB', 3)
        result = run(members, states)
        self.assertAlmostEqual(result.asset_weight_post['AAA'][2], 1)
        self.assertAlmostEqual(result.asset_weight_post['AAA'][3], .5)
        self.assertAlmostEqual(result.asset_weight_post['BBB'][3], .5)

    def test_drift_is_not_daily_rebalanced(self):
        members, states = self.fixture(AllInModel())
        result = run(members, states, returns={'AAA': [0] + [.1] * 9})
        self.assertGreater(result.asset_weight_post['AAA'][2], .5)
        self.assertFalse(result.rebalanced[1:8].any())
        self.assertAlmostEqual(result.equity[1], START_CAPITAL * 1.05)

    def test_three_levels_and_parent_only_children(self):
        members, states = self.fixture()
        wrapper, ws = child(members, states, proportional_model('linear'), name='Wrapper')
        result = run([wrapper], {wrapper.id: ws})
        self.assertAlmostEqual(result.post_weight[0], .75)
        self.assertAlmostEqual(result.post_weight[8], 0)

    def test_fees_and_liquidation_reconcile(self):
        members, states = self.fixture(AllInModel(), cost=17)
        states['b'] = signal('b', 'BBB', 3, 7)
        result = run(members, states, cost=31, returns={'AAA': [0] + [.02] * 9})
        self.assertGreater(result.total_costs[-1], 0)
        self.assertGreaterEqual(result.cash.min(), -1e-7)
        self.assertAlmostEqual(result.invested[-1], 0)
        np.testing.assert_allclose(result.equity, result.cash + result.invested)
        np.testing.assert_allclose(result.equity - START_CAPITAL,
            result.realized_pnl + result.unrealized_pnl - result.total_costs, atol=1e-7)
        episodes = _positions_payload(result)
        total = sum(e['pnl'] for e in episodes['episodes'])
        self.assertAlmostEqual(total, result.equity[-1] - START_CAPITAL)

    def test_same_symbol_sleeves_reconcile_open_positions(self):
        a, b = _strategy('a', 'AAA'), _strategy('b', 'AAA')
        c, cs = child([a], {'a': signal('a', 'AAA', 0, 5)}, cost=10)
        result = run([c, b], {c.id: cs, b.id: signal('b', 'AAA', 2, None)},
                     returns={'AAA': [0] + [.03] * 9}, cost=20)
        np.testing.assert_allclose(result.equity - START_CAPITAL,
            result.realized_pnl + result.unrealized_pnl - result.total_costs, atol=1e-7)
        episodes = _positions_payload(result)['episodes']
        self.assertAlmostEqual(sum(e['pnl'] for e in episodes), result.equity[-1] - START_CAPITAL)

    def test_child_future_changes_do_not_change_past(self):
        members, states = self.fixture()
        before = run(members, states)
        members[0].nested_spec.signals['idle'] = signal('idle', 'CCC', 6)
        after = run(members, states)
        np.testing.assert_allclose(before.equity[:6], after.equity[:6])
        np.testing.assert_allclose(before.post_weight[:6], after.post_weight[:6])

    def test_nested_with_one_symbol_does_not_use_legacy_overlay(self):
        a = _strategy('a', 'AAA')
        c, cs = child([a], {'a': signal('a', 'AAA')})
        from api.portfolio_service import is_single_symbol, trading_symbols
        self.assertFalse(is_single_symbol([c], None))
        self.assertEqual(trading_symbols([c], None), ['AAA'])


class NestedStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'portfolios.json'
        self.mock = patch('api.portfolio_store.get_strategy_by_id', return_value=_strategy('a', 'AAA'))
        self.mock.start()
        self.addCleanup(self.mock.stop)
        self.addCleanup(self.temp.cleanup)

    def create(self, name='Leaf', **kwargs):
        return create_portfolio(SavePortfolioRequest(name=name, **kwargs), store_path=self.path)

    def test_save_load_update_mixed_and_child_only(self):
        leaf = self.create(strategy_ids=['a'], allocation_model='proportional', proportional_mode='early_conviction')
        parent = self.create('Parent', portfolio_ids=[leaf.id], allocation_model='regime_adjusted')
        loaded = list_portfolios(self.path)[1]
        self.assertEqual(loaded.portfolio_ids, [leaf.id])
        self.assertEqual(loaded.allocation_model, 'regime_adjusted')
        updated = update_portfolio(parent.id, UpdatePortfolioRequest(strategy_ids=['a']), self.path)
        self.assertEqual(updated.portfolio_ids, [leaf.id])
        self.assertEqual(list_portfolios(self.path)[0].proportional_mode, 'early_conviction')

    def test_rejects_direct_and_indirect_cycles(self):
        leaf = self.create(strategy_ids=['a'])
        parent = self.create('Parent', portfolio_ids=[leaf.id])
        for children in ([leaf.id], [parent.id]):
            with self.assertRaisesRegex(ValueError, 'cycle'):
                update_portfolio(leaf.id, UpdatePortfolioRequest(portfolio_ids=children), self.path)
        self.assertEqual(list_portfolios(self.path)[0].portfolio_ids, [])

    def test_rejects_missing_duplicate_and_empty(self):
        leaf = self.create(strategy_ids=['a'])
        for children in (['missing'], [leaf.id, leaf.id], []):
            with self.assertRaises(ValueError):
                self.create('Bad', portfolio_ids=children)

    def test_referenced_child_cannot_be_deleted(self):
        leaf = self.create(strategy_ids=['a'])
        parent = self.create('Parent', portfolio_ids=[leaf.id])
        with self.assertRaisesRegex(ValueError, 'used by'):
            delete_portfolio(leaf.id, self.path)
        delete_portfolio(parent.id, self.path)
        self.assertTrue(delete_portfolio(leaf.id, self.path))

class ChildEquivalenceTests(unittest.TestCase):
    def test_wrapped_child_matches_standalone_for_all_proportional_modes_and_all_in(self):
        for mode in (AllInModel(), proportional_model('linear'), proportional_model('early_conviction'),
                     proportional_model('confirmation_required')):
            for same_symbol in (True, False):
                with self.subTest(model=mode.id, same_symbol=same_symbol):
                    a, b = _strategy('a', 'AAA'), _strategy('b', 'AAA' if same_symbol else 'BBB')
                    states = {'a': signal('a', a.symbol, 0, 8), 'b': signal('b', b.symbol, 3, 6)}
                    wrapper, ws = child([a, b], states, mode, cost=23)
                    returns = {'AAA': [0] + [.03] * 9, 'BBB': [0] + [-.02] * 9}
                    standalone = run([a, b], states, mode, cost=23, returns=returns)
                    nested = run([wrapper], {wrapper.id: ws}, returns=returns)
                    np.testing.assert_allclose(nested.equity, standalone.equity, rtol=1e-10)
                    np.testing.assert_allclose(nested.total_costs, standalone.total_costs, rtol=1e-9, atol=1e-8)
                    self.assertEqual(len(nested.transactions), len(standalone.transactions))


class NestedIntegrationTests(unittest.TestCase):
    def setUp(self):
        from api.schemas import SavedPortfolio
        self.members = {key: _strategy(key, symbol) for key, symbol in [('a', 'AAA'), ('idle', 'CCC'), ('b', 'BBB')]}
        self.signals = {key: signal(key, member.symbol, None if key == 'idle' else 0) for key, member in self.members.items()}
        self.saved = SavedPortfolio(id='child', name='Child', strategy_ids=['a', 'idle'],
            allocation_model='proportional', proportional_mode='linear', created_at='now', updated_at='now')
        self.track = {symbol: pd.Series(np.zeros(len(DATES)), index=DATES) for symbol in ('AAA', 'BBB', 'CCC')}
        self.close = {symbol: pd.Series(np.full(len(DATES), 100.), index=DATES) for symbol in self.track}
        # Real scorer, with no labeled regimes: history is valid but cannot penalize anything.
        from api.portfolio_service import simulate_portfolio_allocation
        for key, member in self.members.items():
            self.signals[key].executed = simulate_portfolio_allocation([member], {key: self.signals[key]},
                model=AllInModel(), global_proxy=None, track_changes=self.track, close_series=self.close).frame
        patches = [
            patch('api.nested_portfolios.list_portfolios', return_value=[self.saved]),
            patch('api.portfolio_service.get_strategy_by_id', side_effect=lambda key: self.members[key]),
            patch('api.portfolio_service._load_strategy_signals', side_effect=lambda member, **kwargs: self.signals[member.id]),
            patch('api.portfolio_service._load_portfolio_market_data', return_value=(None, {}, self.track, self.close)),
            patch('api.market_regimes.load_regime_calendar', return_value=pd.DataFrame({'Date': DATES})),
            patch('api.portfolio_service.regime_labels', side_effect=lambda dates, calendar: {
                key: np.full(len(dates), None, dtype=object)
                for key in ALLOCATION_REGIME_DIMENSIONS
            }),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_simulation_contribution_comparison_and_shapley_accept_child_only(self):
        from api.portfolio_service import simulate_portfolio, simulate_portfolio_allocation_comparison, simulate_portfolio_shapley
        request = PortfolioSimulateRequest(portfolio_ids=['child'], years=1)
        payload = simulate_portfolio(request)
        self.assertEqual(payload['allocation']['nested_cash_policy'], 'respect_child_rejection')
        self.assertEqual(len(payload['contribution']), 1)
        self.assertEqual(payload['contribution'][0]['strategy_id'], 'portfolio:child')
        comparison = simulate_portfolio_allocation_comparison(request)
        self.assertEqual(len(comparison['models']), 9)
        self.assertAlmostEqual(comparison['models'][0]['analytics']['average_exposure_when_invested_pct'], 50)
        self.assertTrue(simulate_portfolio_shapley(request))

    def test_regime_parent_and_regime_child_both_run(self):
        from api.portfolio_service import simulate_portfolio
        self.saved.allocation_model = 'regime_adjusted'
        request = PortfolioSimulateRequest(strategy_ids=['b'], portfolio_ids=['child'], years=1,
                                          allocation_model='regime_adjusted')
        result = simulate_portfolio(request)
        self.assertEqual(result['allocation']['model'], 'regime_adjusted')
        self.assertEqual(len(result['allocation']['members']), 2)

    def test_parent_proxy_does_not_override_child(self):
        from api.portfolio_service import simulate_portfolio
        result = simulate_portfolio(PortfolioSimulateRequest(strategy_ids=['b'], portfolio_ids=['child'],
                                                              proxy_symbol='CCC', years=1))
        symbols = {episode['symbol'] for episode in result['positions']['episodes']}
        self.assertEqual(symbols, {'AAA', 'CCC'})

class NestedRegimeAndCalendarTests(unittest.TestCase):
    def test_distinct_parent_and_child_regime_confidences_multiply(self):
        class Context:
            master_dates = DATES
            def __init__(self, confidence, consensus):
                self.confidence, self.consensus = confidence, consensus
            def decisions(self, ids, active, dates=None, settings=None):
                return {bar: {'strategy_confidence': self.confidence, 'portfolio_consensus': self.consensus}
                        for bar in range(len(DATES))}
        a = _strategy('a', 'AAA')
        wrapper, ws = child([a], {'a': signal('a', 'AAA')}, RegimeAdjustedModel(), context=Context(.5, 1))
        tracks = {'AAA': pd.Series(np.zeros(len(DATES)), index=DATES)}
        closes = {'AAA': pd.Series(np.full(len(DATES), 100.), index=DATES)}
        result = simulate_portfolio_allocation([wrapper], {wrapper.id: ws}, model=RegimeAdjustedModel(),
            global_proxy=None, track_changes=tracks, close_series=closes, regime_context=Context(.8, .5))
        self.assertAlmostEqual(result.post_weight[0], .5 * .8 * (.7 + .3 * .5))

    def test_not_yet_eligible_child_is_excluded_until_ready_even_without_entry(self):
        a, b = _strategy('a', 'AAA'), _strategy('b', 'BBB')
        states = signal('a', 'AAA', None)
        states.entry_ready = np.array([False] * 3 + [True] * 7)
        c, cs = child([a], {'a': states})
        result = run([c, b], {c.id: cs, 'b': signal('b', 'BBB')}, proportional_model('linear'))
        self.assertAlmostEqual(result.post_weight[2], 1.)
        self.assertAlmostEqual(result.post_weight[3], .5)
        self.assertEqual(result.eligible_count[2], 1)
        self.assertEqual(result.eligible_count[3], 2)
        self.assertEqual(result.eligibility[0]['first_eligible_date'], DATES[3].strftime('%Y-%m-%d'))

    def test_multisymbol_multilevel_redemptions_preserve_equity_with_costs(self):
        a, b, c = _strategy('a', 'AAA'), _strategy('b', 'BBB'), _strategy('c', 'CCC')
        inner, ins = child([a, b], {'a': signal('a', 'AAA', 0, 8), 'b': signal('b', 'BBB', 1, 7)},
                           proportional_model('early_conviction'), cost=37)
        outer, outs = child([inner], {inner.id: ins}, name='Outer')
        result = run([outer, c], {outer.id: outs, 'c': signal('c', 'CCC', 3, 6)}, cost=25,
                     returns={'AAA': [0] + [.02] * 9, 'BBB': [0] + [-.015] * 9})
        np.testing.assert_allclose(result.equity - START_CAPITAL,
            result.realized_pnl + result.unrealized_pnl - result.total_costs, atol=1e-7)
        self.assertGreaterEqual(result.cash.min(), -1e-7)
        self.assertEqual(result.invested[-1], 0)
        for transaction in result.transactions:
            self.assertGreater(transaction.equity_before, 0)
            self.assertLessEqual(transaction.exposure_after, 1 + 1e-9)


class NettedExecutionTests(unittest.TestCase):
    def compare_flat(self, mode, costs, second_symbol='AAA', starts=(0, 0), ends=(1, 8)):
        a, b = _strategy('a', 'AAA'), _strategy('b', second_symbol)
        states = {'a': signal('a', 'AAA', starts[0], ends[0]),
                  'b': signal('b', second_symbol, starts[1], ends[1])}
        ca, sa = child([a], {'a': states['a']}, name='A', cost=costs)
        cb, sb = child([b], {'b': states['b']}, name='B', cost=costs)
        returns = {'AAA': [0, .03, -.02, .01, .04, -.015, .02, -.01, .03, 0],
                   'BBB': [0, -.02, .01, .03, -.01, .02, -.03, .04, -.02, 0]}
        nested = run([ca, cb], {ca.id: sa, cb.id: sb}, mode, returns, costs)
        flat = run([a, b], states, mode, returns, costs)
        for field in ('equity', 'cash', 'invested', 'realized_pnl', 'unrealized_pnl',
                      'total_costs', 'traded_notional', 'post_weight', 'rebalanced'):
            np.testing.assert_allclose(getattr(nested, field), getattr(flat, field),
                                       rtol=1e-10, atol=1e-8, err_msg=field)
        self.assertEqual(len(nested.transactions), len(flat.transactions))
        for actual, expected in zip(nested.transactions, flat.transactions):
            self.assertEqual((actual.bar, actual.symbol), (expected.bar, expected.symbol))
            for field in ('units_delta', 'units_before', 'units_after', 'realized_pnl',
                          'fee', 'avg_cost_after', 'exposure_before', 'exposure_after'):
                self.assertAlmostEqual(getattr(actual, field), getattr(expected, field), places=7)
        return nested

    def test_matches_flat_execution_across_models_symbols_fees_and_signal_changes(self):
        for mode in [AllInModel()] + [proportional_model(m) for m in
                                     ('linear', 'early_conviction', 'confirmation_required')]:
            for cost in (0, 23):
                for symbol in ('AAA', 'BBB'):
                    for starts, ends in [((0, 0), (1, 8)), ((0, 3), (5, 8)), ((0, 4), (4, 8))]:
                        with self.subTest(mode=mode.id, cost=cost, symbol=symbol, starts=starts, ends=ends):
                            self.compare_flat(mode, cost, symbol, starts, ends)

    def test_early_conviction_has_one_enter_reduce_and_exit(self):
        result = self.compare_flat(proportional_model('early_conviction'), 0)
        self.assertEqual([t.bar for t in result.transactions], [0, 1, 8])
        np.testing.assert_allclose([t.exposure_after for t in result.transactions], [1, np.sqrt(.5), 0])
        self.assertLess(result.transactions[1].units_delta, 0)
        self.assertAlmostEqual(result.transactions[1].units_delta / result.transactions[1].units_before,
                               np.sqrt(.5) - 1)

    def test_internal_handoff_has_no_trade_fee_or_realized_profit(self):
        result = self.compare_flat(AllInModel(), 23, starts=(0, 4), ends=(4, 8))
        self.assertEqual([t.bar for t in result.transactions], [0, 8])
        self.assertEqual(result.cost[4], 0)
        self.assertEqual(result.traded_notional[4], 0)
        self.assertEqual(result.realized_pnl[4], 0)

    def test_different_child_fees_apply_only_to_net_orders(self):
        a, b = _strategy('a', 'AAA'), _strategy('b', 'AAA')
        ca, sa = child([a], {'a': signal('a', 'AAA', 0, 1)}, name='A', cost=10)
        cb, sb = child([b], {'b': signal('b', 'AAA', 0, 8)}, name='B', cost=30)
        result = run([ca, cb], {ca.id: sa, cb.id: sb}, proportional_model('early_conviction'))
        self.assertEqual([t.bar for t in result.transactions], [0, 1, 8])
        for transaction, bps in zip(result.transactions, (20, 10, 30)):
            self.assertAlmostEqual(transaction.fee,
                                   abs(transaction.units_delta) * transaction.price * bps / 10000)
        np.testing.assert_allclose(result.equity - START_CAPITAL,
            result.realized_pnl + result.unrealized_pnl - result.total_costs, atol=1e-8)
