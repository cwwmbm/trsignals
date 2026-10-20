"""Resolve references into immediate members; never flatten allocation boundaries."""
from types import SimpleNamespace
import numpy as np

from api.portfolio_graph import validate_graph
from api.portfolio_store import list_portfolios, _model_dump


def prepare_nested(portfolio, *, years, use_cache=True, symbol_dataset=None, bulk_data=None, _saved=None, _cache=None, **kwargs):
    from api import portfolio_service as ps
    saved = list_portfolios() if _saved is None else _saved
    cache = {} if _cache is None else _cache
    items = [_model_dump(item) for item in saved]
    root = dict(id='__request__', strategy_ids=list(portfolio.strategy_ids),
                portfolio_ids=list(portfolio.portfolio_ids))
    if _saved is None:
        validate_graph([*items, root])
    by_id = {item.id: item for item in saved}
    strategies, ids, signals, tracks, closes = [], [], {}, {}, {}
    proxy = ps._normalize_proxy(getattr(portfolio, 'proxy_symbol', None))
    if portfolio.strategy_ids:
        direct = SimpleNamespace(**{**vars(portfolio), 'portfolio_ids': []})
        strategies, ids, signals, _, _, tracks, closes = ps._prepare_portfolio_simulation(
            direct, years=years, use_cache=use_cache, symbol_dataset=symbol_dataset,
            bulk_data=bulk_data, keep_executed=True,
        )
    for identity in portfolio.portfolio_ids:
        child = by_id[identity]
        if identity in cache:
            member, state, child_tracks, child_closes = cache[identity]
            strategies.append(member)
            ids.append(member.id)
            signals[member.id] = state
            tracks.update(child_tracks)
            closes.update(child_closes)
            continue
        prepare = prepare_nested if child.portfolio_ids else ps._prepare_portfolio_simulation
        extra = dict(_saved=saved, _cache=cache) if child.portfolio_ids else dict(keep_executed=True)
        members, _, child_signals, _, child_proxy, child_tracks, child_closes = prepare(
            child, years=years, use_cache=use_cache, symbol_dataset=symbol_dataset,
            bulk_data=bulk_data, **extra,
        )
        dates = ps._master_dates_for(child_signals, None)
        allocation = ps.bind_regime_context(ps.allocation_settings(child), members, child_signals,
            global_proxy=child_proxy, track_changes=child_tracks, close_series=child_closes, years=years)
        executed = ps.simulate_portfolio_frame(members, child_signals, allocation=allocation,
            overlap_mode=ps.PORTFOLIO_OVERLAP_MODE, global_proxy=child_proxy,
            track_changes=child_tracks, close_series=child_closes, strategy_order=[m.id for m in members])
        active, eligible, _ = ps._effective_signal_matrices(members, child_signals, dates)
        active, eligible = active.any(axis=0), eligible.any(axis=0)
        previous = np.concatenate(([False], active[:-1]))
        member_id = f'portfolio:{identity}'
        member = SimpleNamespace(id=member_id, name=child.name, symbol='Portfolio', direction='long',
            proxy_symbol=None, hold_days=max(m.hold_days for m in members), profit=max(m.profit for m in members),
            nested_spec=SimpleNamespace(members=members, signals=child_signals, allocation=allocation,
                                        proxy=child_proxy, years=years))
        strategies.append(member)
        ids.append(member_id)
        signals[member_id] = ps.StrategySignals(member_id, child.name, 'Portfolio', dates,
            active & ~previous, previous, previous & ~active, executed=executed, entry_ready=eligible)
        cache[identity] = (member, signals[member_id], child_tracks, child_closes)
        tracks.update(child_tracks)
        closes.update(child_closes)
    return strategies, ids, signals, ps.PORTFOLIO_OVERLAP_MODE, proxy, tracks, closes
