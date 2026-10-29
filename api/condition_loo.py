"""What each entry condition adds: the draft rerun once without each of them (leave one out).

Every row is the draft with one entry condition removed, scored in-sample and out-of-sample
with the same frame preparation and metric code as the sweeps. A condition earns its place when
removing it makes the strategy worse in both halves. Sharpe is the deciding figure: removing a
filter adds trades, so PnL and drawdown grow with exposure alone, while Sharpe compares the
variants at equal risk.

Every variant starts trading on the draft's first tradable bar, so a removed condition with a
long warm-up (an SMA200, say) does not hand its variant extra early years to be judged on.
"""
from __future__ import annotations

import math

import pandas as pd

from api.builder_strategy import prepare_builder_refine_frame
from api.strategy_compiler import format_condition_preview

# Removing a condition must move Sharpe by more than this, in the same direction in both halves,
# for the row to count as helping or hurting.
SHARPE_TOLERANCE = 0.05
DRAFT_LABEL = "None (draft as is)"


def _condition_dicts(conditions) -> list[dict]:
    return [c.dict() if hasattr(c, "dict") else dict(c) for c in conditions]


def _first_ready_date(frame: pd.DataFrame) -> pd.Timestamp | None:
    if "EntryReady" not in frame.columns:
        return None
    ready = frame["EntryReady"].fillna(False).astype(bool).to_numpy()
    if not ready.any():
        return None
    return pd.Timestamp(pd.to_datetime(frame["Date"]).iloc[int(ready.argmax())])


def _start_at(frame: pd.DataFrame, start: pd.Timestamp | None) -> pd.DataFrame:
    """No entries, and no scoring, before `start`."""
    if start is None:
        return frame
    frame = frame.copy()
    before = (pd.to_datetime(frame["Date"]) < start).to_numpy()
    frame.loc[before, "Buy"] = False
    if "EntryReady" in frame.columns:
        frame.loc[before, "EntryReady"] = False
    else:
        frame["EntryReady"] = ~before
    return frame


def _sharpe(metrics: dict | None) -> float | None:
    if not metrics:
        return None
    value = metrics.get("Sharpe")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _verdict(delta_in: float | None, delta_out: float | None, tolerance: float) -> str:
    """How the draft fares without the condition, read from the Sharpe change in both halves."""
    if delta_in is None or delta_out is None:
        return "-"
    if delta_in < -tolerance and delta_out < -tolerance:
        return "Helps"
    if delta_in > tolerance and delta_out > tolerance:
        return "Hurts"
    if abs(delta_in) <= tolerance and abs(delta_out) <= tolerance:
        return "No effect"
    return "Mixed"


def _round(value: float | None, digits: int = 2):
    return None if value is None else round(value, digits)


def leave_one_out_rows(
    primary,
    data,
    *,
    years: int,
    labels: dict[str, str],
    in_sample_end: pd.Timestamp,
    strategy_resolver=None,
    tolerance: float = SHARPE_TOLERANCE,
) -> list[dict]:
    """The draft first, then one row per entry condition with that condition removed."""
    from api import fast_metrics
    from api.services import _format_outcome_metrics, _full_period_summary

    conditions = list(primary.conditions)
    condition_dicts = _condition_dicts(conditions)
    variants = [(DRAFT_LABEL, primary, None)]
    for index, condition in enumerate(condition_dicts):
        label = format_condition_preview([{**condition, "logic": "AND"}], labels)
        if len(conditions) == 1:
            variants.append((label, None, None))
            continue
        kept = conditions[:index] + conditions[index + 1:]
        # Lets the builder find and delete the condition: its place in the draft, and what it is
        # in case the draft has changed since.
        removed = {"index": index, **{key: condition[key] for key in ("left", "operator", "right")}}
        variants.append((label, primary.copy(update={"conditions": kept}), removed))

    start: pd.Timestamp | None = None
    draft_metrics: dict[str, dict | None] = {}
    draft_entries = 0
    rows: list[dict] = []
    for order, (label, strategy, removed) in enumerate(variants):
        row: dict = {"#": order, "Removed": label, "RemovedCondition": removed}
        if strategy is None:
            row["Error"] = "The only entry condition: nothing is left to compare against"
            rows.append(row)
            continue
        try:
            frame, days, profit, is_long, pnl_col = prepare_builder_refine_frame(
                strategy,
                years=years,
                data=data,
                strategy_resolver=strategy_resolver,
                labels=labels,
            )
        except ValueError as exc:
            if order == 0:
                raise
            row["Error"] = str(exc)
            rows.append(row)
            continue
        if order == 0:
            start = _first_ready_date(frame)
        frame = _start_at(frame, start)
        dates = pd.to_datetime(frame["Date"])
        windows = {
            "in": frame.loc[(dates <= in_sample_end).to_numpy()],
            "out": frame.loc[(dates > in_sample_end).to_numpy()],
            "full": frame,
        }
        metrics: dict[str, dict | None] = {}
        for key, window in windows.items():
            metrics[key] = (
                fast_metrics.run_metrics(
                    window.copy(), days, profit, is_long, pnl_column=pnl_col, include_yearly=True
                )
                if len(window) >= 3
                else None
            )
        entries = int(frame["Buy"].fillna(False).astype(bool).sum())
        if order == 0:
            draft_metrics = metrics
            draft_entries = entries

        for prefix, key in (("", "in"), ("OOS ", "out")):
            scored = metrics[key]
            formatted = _format_outcome_metrics(scored) if scored else None
            sharpe = _sharpe(scored)
            row[f"{prefix}Trades"] = formatted["Trades"] if formatted else None
            row[f"{prefix}Sharpe"] = _round(sharpe)
            if order:
                base = _sharpe(draft_metrics.get(key))
                row[f"{prefix}ΔSharpe"] = (
                    None if sharpe is None or base is None else _round(sharpe - base)
                )
            row[f"{prefix}CAGR"] = formatted["CAGR"] if formatted else None
            row[f"{prefix}MaxDD"] = formatted["MaxDD"] if formatted else None
            row[f"{prefix}%Pstv"] = formatted["%Pstv"] if formatted else None
        if order:
            # The share of the entries the other conditions allow that this one blocks.
            row["Blocks"] = f"{round((1 - draft_entries / entries) * 100)}%" if entries else None
            row["Verdict"] = _verdict(row.get("ΔSharpe"), row.get("OOS ΔSharpe"), tolerance)
        row["Yearly"] = (metrics["in"] or {}).get("Yearly") or []
        if metrics["full"]:
            row["FullPeriod"] = _full_period_summary(metrics["full"])
        rows.append(row)

    # The table takes its columns from the first row, and the draft row has no change columns.
    columns = ["#", "Removed"]
    for prefix in ("", "OOS "):
        columns += [f"{prefix}{name}" for name in ("Trades", "Sharpe", "ΔSharpe", "CAGR", "MaxDD", "%Pstv")]
    columns += ["Blocks", "Verdict", "Error", "Yearly", "FullPeriod", "RemovedCondition"]
    return [{column: row.get(column) for column in columns} for row in rows]
