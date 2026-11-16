"""Is a strategy's backtest a real edge or a fit to noise? One report, several independent checks.

* Random timing: the entry mask (and, separately, entries and exits together) circularly shifted
  against prices. Frequency and clustering survive, the link to what the market does next does not.
* Parameter neighbourhood: every numeric threshold moved +-10% and +-25%, hold/profit limits nudged.
  A real effect sits on a plateau; a fitted one on a spike.
* Ablation: each condition removed once. A condition whose removal leaves Sharpe where it was is
  selection fat.
* Time stability: Sharpe by thirds of the history and recent windows, years positive, the
  annualised return with any single year left out.
* Trade concentration: how much of the growth the best ten trades carry.
* Execution: entry delayed a bar or two, entry at the next open, costs charged per leg.

Every run goes through the same frame builder as the Backtest button and the compiled loop from
``api.fast_sweep``; ``_simulate_fills`` with close fills and no cost is that loop exactly.
Bars before the base strategy's first ready entry are left out of every score, so a variant with a
shorter warm-up is not judged on extra early years.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import backtest as bt
import stats
from api import fast_sweep as fs
from api import leverage_settings
from api.builder_strategy import prepare_builder_refine_frame
from api.fast_sweep import START_CAPITAL, njit
from api.strategy_compiler import format_condition_preview
from backtest_runners import load_ticker_data

NULL_SHIFTS = 2000
MIN_SHIFT = 60
HISTOGRAM_BINS = 30
THRESHOLD_STEPS = (-0.25, -0.10, 0.10, 0.25)
DELAYS = (1, 2)
# Cost per leg as a share of the position; 0.05% is a tight ETF fill, 0.125% a conservative one.
COSTS_PER_LEG = (0.0, 0.0005, 0.00125)
# Removing a condition must cost more than this share of the base Sharpe (and at least 0.05) for the
# condition to count as doing work.
ABLATION_TOLERANCE = 0.10
PERIODS_PER_YEAR = 252


@njit(cache=True, error_model="numpy")
def _simulate_fills(
    buy, sell, close, track, use_stream, v_ok, v_use, v_ret, s_ret, days, prof, is_long, hold_on_buy,
    first_change, use_first, cost_leg, pnl, entered, exited,
):
    """``fast_sweep._simulate`` with two execution options and trade bookkeeping.

    ``use_first``: the first held bar earns ``first_change`` (an entry filled at that bar's open)
    instead of the close-to-close return. ``cost_leg`` is charged on the entry and on the exit bar.
    ``entered`` / ``exited`` are filled with the bars a trade opened and closed on.
    """
    n = buy.shape[0]
    trades = 0
    prev_hold = False
    prev_out = False
    prev_in = False
    prev_dfe = 0
    prev_pc = 0
    in_vehicle = False
    for i in range(n):
        hold = False
        if i > 0:
            hold = (prev_hold and not prev_out) or prev_in
        b = buy[i]
        lin = b and not hold
        if lin:
            in_vehicle = use_stream and v_ok[i]
        dfe = 0
        pc = 0
        if hold and i > 0:
            dfe = prev_dfe + 1
            if is_long:
                up = close[i] > close[i - 1]
            else:
                up = close[i] < close[i - 1]
            pc = prev_pc + 1 if up else prev_pc
        would_exit = (sell[i] and hold) or (dfe >= days) or (pc >= prof)
        out = would_exit
        if hold_on_buy and b and hold and would_exit:
            out = False
            dfe = 0
            pc = 0
        if i == 0:
            pnl[i] = START_CAPITAL
        elif hold:
            if prev_in and use_first:
                change = first_change[i]
            elif use_stream:
                if in_vehicle and v_use[i]:
                    change = v_ret[i]
                else:
                    change = s_ret[i]
            else:
                change = track[i]
            if np.isnan(change):
                change = 0.0
            if is_long:
                pnl[i] = (1.0 + change) * pnl[i - 1]
            else:
                pnl[i] = (1.0 - change) * pnl[i - 1]
        else:
            pnl[i] = pnl[i - 1]
        if lin:
            pnl[i] = pnl[i] * (1.0 - cost_leg)
        if out:
            pnl[i] = pnl[i] * (1.0 - cost_leg)
            trades += 1
        entered[i] = lin
        exited[i] = out
        prev_hold = hold
        prev_out = out
        prev_in = lin
        prev_dfe = dfe
        prev_pc = pc
    return trades


class _Run:
    """One strategy's frame, ready to be re-scored with other masks and fills."""

    def __init__(self, frame: pd.DataFrame, days: int, profit: int, is_long: bool, pnl_col: str | None, first: int):
        self.frame = frame
        self.pnl_col = pnl_col
        self.inputs = fs.build_sim_inputs(frame, days, profit, is_long, pnl_column=pnl_col)
        self.n = len(frame)
        self.first = first
        years = pd.to_datetime(frame["Date"]).dt.year.to_numpy()[first:]
        self.years = years
        self.ystart = np.flatnonzero(np.r_[True, np.diff(years) != 0]).astype(np.int64)
        self.yend = np.r_[self.ystart[1:] - 1, len(years) - 1].astype(np.int64)
        self.exclude_best = bool(stats.ExcludeBestReturnYear)
        self.no_first = np.zeros(self.n)

    @property
    def buy(self) -> np.ndarray:
        return self.inputs.base

    @property
    def sell(self) -> np.ndarray:
        return self.inputs.sell

    def simulate(self, buy=None, sell=None, *, first_change=None, cost_leg: float = 0.0):
        i = self.inputs
        pnl = np.empty(self.n)
        entered = np.zeros(self.n, dtype=np.bool_)
        exited = np.zeros(self.n, dtype=np.bool_)
        trades = _simulate_fills(
            np.ascontiguousarray(self.buy if buy is None else buy, dtype=np.bool_),
            np.ascontiguousarray(self.sell if sell is None else sell, dtype=np.bool_),
            i.close, i.track, i.use_stream, i.v_ok, i.v_use, i.v_ret, i.s_ret,
            i.days, i.prof, i.is_long, i.hold_on_buy,
            self.no_first if first_change is None else first_change, first_change is not None, float(cost_leg),
            pnl, entered, exited,
        )
        f = self.first
        return pnl[f:], int(trades), entered[f:], exited[f:]

    def sharpe(self, pnl: np.ndarray) -> float:
        return float(fs._sharpe(pnl, self.ystart, self.yend, self.exclude_best))

    def year_returns(self, pnl: np.ndarray) -> np.ndarray:
        out = np.full(len(self.ystart), np.nan)
        for k, (s, e) in enumerate(zip(self.ystart, self.yend)):
            if e - s + 1 >= 2 and pnl[s] > 0:
                out[k] = pnl[e] / pnl[s] - 1
        return out

    def cagr(self, pnl: np.ndarray) -> float:
        """Annualised return, leaving out the best positive year like the app does."""
        rets = self.year_returns(pnl)
        rets = rets[np.isfinite(rets)]
        if rets.size == 0:
            return float("nan")
        if self.exclude_best and rets.size > 1 and rets.max() > 0:
            rets = np.delete(rets, int(np.argmax(rets)))
        return float(np.prod(1 + rets) ** (1 / rets.size) - 1)

    def score(self, buy=None, sell=None, **kwargs) -> dict:
        pnl, trades, _, _ = self.simulate(buy, sell, **kwargs)
        return {"sharpe": _finite(self.sharpe(pnl)), "cagr": _finite(self.cagr(pnl)), "trades": trades}


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _first_ready(frame: pd.DataFrame) -> int:
    if "EntryReady" not in frame.columns:
        return 0
    ready = frame["EntryReady"].fillna(False).to_numpy(bool)
    return int(np.argmax(ready)) if ready.any() else 0


def _plain_sharpe(returns: np.ndarray) -> float | None:
    returns = returns[np.isfinite(returns)]
    if returns.size < 30 or returns.std(ddof=1) == 0:
        return None
    return _finite(math.sqrt(PERIODS_PER_YEAR) * returns.mean() / returns.std(ddof=1))


def _status(value: float | None, good: float, ok: float, *, higher_is_better: bool = True) -> str:
    if value is None:
        return "skip"
    if not higher_is_better:
        value, good, ok = -value, -good, -ok
    if value >= good:
        return "pass"
    if value >= ok:
        return "warn"
    return "fail"


def _p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _multiple(x: float) -> str:
    return f"{x:,.0f}x" if x < 1e6 else f"{x:.2g}x"


def _fmt(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


class _Builder:
    """Builds and scores variants of one strategy on one data set."""

    def __init__(self, strategy, data, *, years, labels, strategy_resolver, symbol_data=None):
        self.strategy = strategy
        self.data = data
        self.years = years
        self.labels = labels
        self.resolver = strategy_resolver
        if strategy.confirm_symbols and symbol_data is None:
            needed = list(dict.fromkeys([strategy.symbol.strip().upper(), *strategy.confirm_symbols]))
            symbol_data = bt.load_symbol_dataset(needed, years=years)
        self.symbol_data = symbol_data
        self.base = self.run(strategy)
        self.base_start = pd.to_datetime(self.base.frame["Date"]).to_numpy()[self.base.first]

    def run(self, strategy, *, align: bool = False) -> _Run:
        frame, days, profit, is_long, pnl_col = prepare_builder_refine_frame(
            strategy, years=self.years, data=None if strategy.confirm_symbols else self.data,
            symbol_data=self.symbol_data, strategy_resolver=self.resolver, labels=self.labels,
        )
        if align:
            first = int(np.searchsorted(pd.to_datetime(frame["Date"]).to_numpy(), self.base_start))
        else:
            first = _first_ready(frame)
        return _Run(frame, days, profit, is_long, pnl_col, first)

    def score_variant(self, strategy) -> dict:
        try:
            return self.run(strategy, align=True).score()
        except Exception as exc:  # a nudged value can make a nested condition invalid
            return {"sharpe": None, "cagr": None, "trades": 0, "error": str(exc)[:120]}


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------


def _histogram(values: np.ndarray, actual: float) -> list[dict]:
    lo = min(float(values.min()), actual)
    hi = max(float(values.max()), actual)
    if hi <= lo:
        hi = lo + 1e-9
    counts, edges = np.histogram(values, bins=HISTOGRAM_BINS, range=(lo, hi))
    return [{"lo": float(edges[k]), "hi": float(edges[k + 1]), "count": int(counts[k])} for k in range(len(counts))]


def random_timing(run: _Run, *, shift_exits: bool, shifts: int = NULL_SHIFTS, seed: int = 5) -> dict | None:
    base = run.score()
    length = run.n - run.first
    if base["sharpe"] is None or length <= 2 * MIN_SHIFT + 2 or not run.buy[run.first:].any():
        return None
    rng = np.random.default_rng(seed)
    f = run.first
    sharpes = np.empty(int(shifts))
    for k in range(int(shifts)):
        shift = int(rng.integers(MIN_SHIFT, length - MIN_SHIFT))
        buy = run.buy.copy()
        buy[f:] = np.roll(run.buy[f:], shift)
        sell = None
        if shift_exits:
            sell = run.sell.copy()
            sell[f:] = np.roll(run.sell[f:], shift)
        pnl, _, _, _ = run.simulate(buy, sell)
        sharpes[k] = run.sharpe(pnl)
    sharpes = sharpes[np.isfinite(sharpes)]
    if sharpes.size == 0:
        return None
    actual = base["sharpe"]
    return {
        "shift_exits": shift_exits,
        "sharpe": actual,
        "null_median": float(np.median(sharpes)),
        "null_p95": float(np.quantile(sharpes, 0.95)),
        "null_p99": float(np.quantile(sharpes, 0.99)),
        "p": float((1 + (sharpes >= actual).sum()) / (1 + sharpes.size)),
        "shifts": int(sharpes.size),
        "histogram": _histogram(sharpes, actual),
    }


def _threshold_variants(strategy):
    for side in ("conditions", "sell_conditions"):
        conds = list(getattr(strategy, side))
        for i, c in enumerate(conds):
            try:
                base = float(c.right)
            except (TypeError, ValueError):
                continue
            if c.operator not in {"<", "<=", ">", ">="}:
                continue
            for rel in THRESHOLD_STEPS:
                value = base * (1 + rel) if base != 0 else rel
                new = [x.model_copy(update={"right": _fmt(value)}) if j == i else x for j, x in enumerate(conds)]
                yield {
                    "group": "exit threshold" if side == "sell_conditions" else "entry threshold",
                    "condition": f"{c.left} {c.operator} {c.right}",
                    "label": f"{c.left} {c.operator} {_fmt(value)}",
                    "step": rel,
                }, strategy.model_copy(update={side: new})


def _limit_variants(strategy):
    # A limit at the cap (50 days / 50 profitable closes) is effectively off; nudging it says nothing.
    if strategy.hold_days < 50:
        for d in (-2, -1, 1, 2):
            if strategy.hold_days + d >= 1:
                yield {"group": "hold days", "condition": f"hold {strategy.hold_days}", "label": f"hold {strategy.hold_days + d}", "step": d}, \
                    strategy.model_copy(update={"hold_days": strategy.hold_days + d})
    if 0 < strategy.profit < 50:
        for d in (-1, 1, 2):
            if strategy.profit + d >= 1:
                yield {"group": "profitable closes", "condition": f"profit {strategy.profit}", "label": f"profit {strategy.profit + d}", "step": d}, \
                    strategy.model_copy(update={"profit": strategy.profit + d})


def neighbourhood(builder: _Builder, base: dict) -> dict:
    rows = []
    for meta, variant in [*_threshold_variants(builder.strategy), *_limit_variants(builder.strategy)]:
        rows.append({**meta, **builder.score_variant(variant)})
    sharpes = np.array([r["sharpe"] for r in rows if r["sharpe"] is not None], dtype=float)
    summary = None
    if sharpes.size and base["sharpe"]:
        b = base["sharpe"]
        summary = {
            "variants": int(sharpes.size),
            "median_sharpe": float(np.median(sharpes)),
            "min_sharpe": float(sharpes.min()),
            "max_sharpe": float(sharpes.max()),
            "share_within_75pct": float((sharpes >= 0.75 * b).mean()),
        }
    return {"rows": rows, "summary": summary}


def ablation(builder: _Builder, base: dict) -> list[dict]:
    s = builder.strategy
    tolerance = max(0.05, ABLATION_TOLERANCE * abs(base["sharpe"] or 0.0))
    rows = []
    for side in ("conditions", "sell_conditions"):
        conds = list(getattr(s, side))
        for i, c in enumerate(conds):
            if side == "conditions" and len(conds) == 1:
                continue
            new = [x for j, x in enumerate(conds) if j != i]
            if side == "conditions" and i == 0:
                new[0] = new[0].model_copy(update={"logic": "AND"})
            r = builder.score_variant(s.model_copy(update={side: new}))
            delta = None if r["sharpe"] is None or base["sharpe"] is None else base["sharpe"] - r["sharpe"]
            rows.append({
                "side": "exit" if side == "sell_conditions" else "entry",
                "condition": format_condition_preview([c.model_dump()], builder.labels),
                **r,
                "sharpe_lost": delta,
                "earns_place": None if delta is None else bool(delta > tolerance),
            })
    return rows


def stability(run: _Run) -> dict:
    pnl, trades, entered, exited = run.simulate()
    rets = pnl[1:] / pnl[:-1] - 1
    dates = pd.to_datetime(run.frame["Date"]).to_numpy()[run.first:]
    n = len(rets)
    periods = []
    cuts = [0, n // 3, 2 * n // 3, n]
    for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        periods.append({"label": f"Third {k + 1}", "start": str(pd.Timestamp(dates[a + 1]).date()),
                        "end": str(pd.Timestamp(dates[b]).date()), "sharpe": _plain_sharpe(rets[a:b])})
    last = pd.Timestamp(dates[-1])
    for yrs in (10, 5, 3):
        since = (dates[1:] > np.datetime64(last - pd.DateOffset(years=yrs)))
        periods.append({"label": f"Last {yrs}y", "start": str((last - pd.DateOffset(years=yrs)).date()),
                        "end": str(last.date()), "sharpe": _plain_sharpe(rets[since])})

    yr = run.year_returns(pnl)
    year_labels = run.years[run.ystart]
    held = np.zeros(len(pnl), dtype=bool)
    ins = np.flatnonzero(entered)
    outs = np.flatnonzero(exited)
    trade_rets = []
    for a in ins:
        later = outs[outs > a]
        b = int(later[0]) if later.size else len(pnl) - 1
        held[a + 1:b + 1] = True
        if later.size:
            trade_rets.append(pnl[b] / pnl[a] - 1)
    trade_rets = np.array(trade_rets, dtype=float)
    year_trades = [int(((outs >= s) & (outs <= e)).sum()) for s, e in zip(run.ystart, run.yend)]
    years = [{"year": int(y), "return": _finite(r), "trades": t} for y, r, t in zip(year_labels, yr, year_trades)]

    valid = yr[np.isfinite(yr)]
    loyo = None
    if valid.size > 2:
        g = [np.prod(1 + np.delete(valid, k)) ** (1 / (valid.size - 1)) - 1 for k in range(valid.size)]
        loyo = {"min": float(min(g)), "max": float(max(g))}

    peak = np.maximum.accumulate(pnl)
    concentration = None
    if trade_rets.size:
        logs = np.log1p(np.clip(trade_rets, -0.999, None))
        total = float(logs.sum())
        top = float(np.sort(logs)[::-1][:10].sum())
        wins, losses = trade_rets[trade_rets > 0], trade_rets[trade_rets < 0]
        streak = cur = 0
        for v in trade_rets < 0:
            cur = cur + 1 if v else 0
            streak = max(streak, cur)
        concentration = {
            "trades": int(trade_rets.size),
            "win_rate": float((trade_rets > 0).mean()),
            "avg_win": _finite(wins.mean()) if wins.size else None,
            "avg_loss": _finite(losses.mean()) if losses.size else None,
            "profit_factor": _finite(wins.sum() / -losses.sum()) if losses.size else None,
            "top10_share_of_growth": _finite(top / total) if total > 0 else None,
            "growth_multiple": _finite(math.exp(total)),
            "growth_multiple_ex_top10": _finite(math.exp(total - top)),
            "longest_losing_streak": int(streak),
        }
    return {
        "periods": periods,
        "years": years,
        "years_positive": int((valid > 0).sum()),
        "years_counted": int(valid.size),
        "cagr_leave_one_year_out": loyo,
        "time_in_market": float(held.mean()),
        "max_drawdown": float((1 - pnl / peak).max()),
        "concentration": concentration,
    }


def _next_open_change(builder: _Builder, run: _Run) -> np.ndarray | None:
    """What the first held bar earns when the entry fills at that bar's open: leverage x open-to-close."""
    symbol = run.pnl_col or builder.strategy.symbol.strip().upper()
    if run.pnl_col is None and "Open" in run.frame.columns:
        opens, closes = run.frame["Open"].to_numpy(float), run.frame["Close"].to_numpy(float)
    else:
        raw = load_ticker_data(symbol, years=builder.years)
        raw = raw.set_index(pd.to_datetime(raw["Date"]))
        dates = pd.to_datetime(run.frame["Date"])
        opens = raw["Open"].reindex(dates).to_numpy(float)
        closes = raw["Close"].reindex(dates).to_numpy(float)
    if not np.isfinite(opens).any():
        return None
    intraday = np.nan_to_num(closes / opens - 1, nan=0.0)
    return leverage_settings.general_leverage() * intraday


def execution(builder: _Builder, run: _Run) -> dict:
    delays = []
    for k in DELAYS:
        buy = np.zeros_like(run.buy)
        buy[k:] = run.buy[:-k]
        delays.append({"delay_bars": k, **run.score(buy)})
    fills = []
    next_open = _next_open_change(builder, run)
    for fill, change in (("close", None), ("next open", next_open)):
        if fill == "next open" and change is None:
            continue
        for cost in COSTS_PER_LEG:
            fills.append({"fill": fill, "cost_per_leg": cost, **run.score(first_change=change, cost_leg=cost)})
    return {"delays": delays, "fills": fills}


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


def _verdict(base, nulls, neigh, abl, stab, exe) -> tuple[str, list[dict]]:
    checks = []
    b = base["sharpe"] or 0.0

    primary = nulls[-1] if nulls else None  # entries+exits shifted is the stricter null when there are exits
    if primary:
        st = "pass" if primary["p"] <= 0.01 else "warn" if primary["p"] <= 0.05 else "fail"
        checks.append({"id": "random-timing", "title": "Beats random timing", "status": st,
                       "headline": f"{_p_text(primary['p'])}: Sharpe {b:.2f} vs {primary['null_p95']:.2f} at the 95th percentile of {primary['shifts']} shifted runs"})
    else:
        checks.append({"id": "random-timing", "title": "Beats random timing", "status": "skip", "headline": "Not enough history or no trades"})

    summ = neigh["summary"]
    if summ:
        share = summ["share_within_75pct"]
        checks.append({"id": "neighbourhood", "title": "Plateau, not a spike", "status": _status(share, 0.8, 0.5),
                       "headline": f"{share:.0%} of {summ['variants']} nudged variants keep >= 75% of the Sharpe (range {summ['min_sharpe']:.2f} to {summ['max_sharpe']:.2f})"})
    else:
        checks.append({"id": "neighbourhood", "title": "Plateau, not a spike", "status": "skip", "headline": "No numeric thresholds to nudge"})

    idle = [r["condition"] for r in abl if r["earns_place"] is False]
    if abl:
        checks.append({"id": "ablation", "title": "Every condition does work", "status": "pass" if not idle else "warn",
                       "headline": "Removing any condition costs Sharpe" if not idle else "Costs under 10% of the Sharpe to remove: " + ", ".join(idle)})

    thirds = [p["sharpe"] for p in stab["periods"][:3]]
    if all(t is not None for t in thirds) and b > 0:
        worst = min(thirds)
        checks.append({"id": "stability", "title": "Holds across time", "status": _status(worst / b, 0.5, 0.0),
                       "headline": f"Sharpe by thirds {' / '.join(f'{t:.2f}' for t in thirds)}; {stab['years_positive']}/{stab['years_counted']} years positive"})

    conc = stab["concentration"]
    if conc and conc["growth_multiple_ex_top10"] is not None:
        ex = conc["growth_multiple_ex_top10"]
        share = conc["top10_share_of_growth"]
        st = "fail" if ex <= 1 else "warn" if (share or 0) > 0.5 else "pass"
        checks.append({"id": "concentration", "title": "Not carried by a few trades", "status": st,
                       "headline": f"Best 10 of {conc['trades']} trades carry {share:.0%} of log growth; without them equity still grows {_multiple(ex)}"})

    worst_fill = [f for f in exe["fills"] if f["fill"] == "next open" and f["cost_per_leg"] == max(COSTS_PER_LEG)]
    worst_fill = worst_fill[0] if worst_fill else (exe["fills"][-1] if exe["fills"] else None)
    if worst_fill and worst_fill["sharpe"] is not None and b > 0:
        delay1 = exe["delays"][0]["sharpe"] if exe["delays"] else None
        checks.append({"id": "execution", "title": "Survives realistic fills", "status": _status(worst_fill["sharpe"] / b, 0.75, 0.5),
                       "headline": f"{worst_fill['fill']} entry, {worst_fill['cost_per_leg'] * 100:.3g}% per leg: Sharpe {worst_fill['sharpe']:.2f}"
                                   + (f"; entry a full bar late: {delay1:.2f}" if delay1 is not None else "")})

    statuses = [c["status"] for c in checks]
    if checks[0]["status"] == "fail" or statuses.count("fail") >= 2:
        overall = "likely overfit"
    elif "fail" in statuses or statuses.count("warn") >= 2:
        overall = "mixed"
    else:
        overall = "robust"
    return overall, checks


def robustness_report(strategy, data, *, years: int, labels: dict, strategy_resolver, shifts: int = NULL_SHIFTS) -> dict:
    """Every check for one strategy. ``bt.HoldOnBuySignal`` must already match the strategy."""
    builder = _Builder(strategy, data, years=years, labels=labels, strategy_resolver=strategy_resolver)
    run = builder.base
    base = run.score()
    if not base["trades"]:
        raise ValueError("The strategy never trades, so there is nothing to test")
    nulls = [n for n in (random_timing(run, shift_exits=False, shifts=shifts),) if n]
    if strategy.sell_conditions:
        n2 = random_timing(run, shift_exits=True, shifts=shifts, seed=6)
        if n2:
            nulls.append(n2)
    neigh = neighbourhood(builder, base)
    abl = ablation(builder, base)
    stab = stability(run)
    exe = execution(builder, run)
    overall, checks = _verdict(base, nulls, neigh, abl, stab, exe)
    return {
        "baseline": {**base, "start": str(pd.Timestamp(builder.base_start).date()),
                     "traded_symbol": run.pnl_col or strategy.symbol.strip().upper()},
        "verdict": overall,
        "checks": checks,
        "random_timing": nulls,
        "neighbourhood": neigh,
        "ablation": abl,
        "stability": stab,
        "execution": exe,
    }
