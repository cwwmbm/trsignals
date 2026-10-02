from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from api.indicator_catalog import (
    builder_eligible_ids,
    is_primary_only_indicator,
    validate_compare_target,
)
from api.market_regimes import REGIME_INDICATOR_IDS, ensure_regime_indicator_columns


VALID_OPERATORS = {"<", "<=", ">", ">=", "=", "crosses above", "crosses below", "is true", "is false"}
FLAG_OPERATORS = {"is true", "is false"}
STRATEGY_INDICATOR_PREFIX = "strategy:"
_INDICATOR_IDS = set(builder_eligible_ids())
StrategyResolver = Callable[[str], Any | None]


class ConditionSnapshotItem(TypedDict):
    label: str
    logic: str | None
    passed: bool
    left_value: str | None
    right_value: str | None
    operator: str
    left: str
    right: str


def is_strategy_indicator(value: str) -> bool:
    return value.startswith(STRATEGY_INDICATOR_PREFIX)


def strategy_indicator_id(value: str) -> str:
    return value.removeprefix(STRATEGY_INDICATOR_PREFIX)


def _condition_dicts(conditions) -> list[dict]:
    return [
        condition.model_dump() if hasattr(condition, "model_dump") else dict(condition)
        for condition in conditions
    ]


def _resolve_operand(data: pd.DataFrame, value: str):
    if value in _INDICATOR_IDS:
        if value not in data.columns:
            raise ValueError(f"Indicator column not found in data: {value}")
        return data[value]
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Right operand must be an indicator id or numeric literal: {value}") from exc


def _compare(left: pd.Series, operator: str, right) -> pd.Series:
    if operator == "<":
        result = left < right
    elif operator == "<=":
        result = left <= right
    elif operator == ">":
        result = left > right
    elif operator == ">=":
        result = left >= right
    elif operator == "=":
        result = left == right
    elif operator == "crosses above":
        if isinstance(right, (int, float)):
            result = (left > right) & (left.shift(1) <= right)
        else:
            result = (left > right) & (left.shift(1) <= right.shift(1))
    elif operator == "crosses below":
        if isinstance(right, (int, float)):
            result = (left < right) & (left.shift(1) >= right)
        else:
            result = (left < right) & (left.shift(1) >= right.shift(1))
    else:
        raise ValueError(f"Unsupported operator: {operator}")
    return result.fillna(False)


def _indicator_ready(data: pd.DataFrame, indicator_id: str) -> pd.Series:
    """True once this input exists. Regime 0 is unlabeled, so it is not ready."""
    numeric = pd.to_numeric(data[indicator_id], errors="coerce")
    ready = numeric.notna()
    if indicator_id in REGIME_INDICATOR_IDS:
        ready = ready & (numeric != 0)
    return ready


def _operand_ready(data: pd.DataFrame, value: str) -> pd.Series:
    if value in _INDICATOR_IDS:
        if value not in data.columns:
            raise ValueError(f"Indicator column not found in data: {value}")
        return _indicator_ready(data, value)
    try:
        float(value)
    except ValueError as exc:
        raise ValueError(f"Right operand must be an indicator id or numeric literal: {value}") from exc
    return pd.Series(True, index=data.index)


def _ready_with_prior(ready: pd.Series) -> pd.Series:
    prior = ready.shift(1)
    prior = prior.where(prior.notna(), False).astype(bool)
    return ready & prior


def _compile_condition_state(
    data: pd.DataFrame,
    conditions: list[dict],
    *,
    strategy_resolver: StrategyResolver | None = None,
    seen_strategy_ids: set[str] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return (condition is true, inputs exist so a trade could be taken)."""
    if not conditions:
        raise ValueError("At least one condition is required")

    ensure_regime_indicator_columns(data, conditions)
    seen_strategy_ids = seen_strategy_ids or set()
    mask: pd.Series | None = None
    ready: pd.Series | None = None
    for condition in conditions:
        left_id = condition["left"]
        operator = condition["operator"]
        right_value = condition["right"]
        logic = condition.get("logic", "AND")

        if operator not in VALID_OPERATORS:
            raise ValueError(f"Unsupported operator: {operator}")

        if is_strategy_indicator(left_id):
            part, part_ready = _compile_strategy_indicator_state(
                data,
                left_id,
                operator,
                strategy_resolver,
                seen_strategy_ids,
            )
        elif left_id not in _INDICATOR_IDS:
            raise ValueError(f"Unknown left indicator: {left_id}")
        elif left_id not in data.columns:
            raise ValueError(f"Indicator column not found in data: {left_id}")
        else:
            left_series = data[left_id]
            part_ready = _indicator_ready(data, left_id)
            if operator in FLAG_OPERATORS:
                part = left_series == (1 if operator == "is true" else -1)
            else:
                validate_compare_target(left_id, right_value)
                right_operand = _resolve_operand(data, right_value)
                part = _compare(left_series, operator, right_operand)
                right_ready = _operand_ready(data, right_value)
                if operator in {"crosses above", "crosses below"}:
                    part_ready = _ready_with_prior(part_ready)
                    if right_value in _INDICATOR_IDS:
                        right_ready = _ready_with_prior(right_ready)
                part_ready = part_ready & right_ready
        part = part.fillna(False)
        part_ready = part_ready.fillna(False)

        if mask is None:
            mask = part
            ready = part_ready
            continue

        if logic == "AND":
            mask = mask & part
            ready = ready & part_ready
        elif logic == "OR":
            mask = mask | part
            ready = ready | part_ready
        else:
            raise ValueError(f"Unsupported logic: {logic}")

    assert mask is not None and ready is not None
    return mask, ready


def _compile_strategy_indicator_state(
    data: pd.DataFrame,
    left_id: str,
    operator: str,
    strategy_resolver: StrategyResolver | None,
    seen_strategy_ids: set[str],
) -> tuple[pd.Series, pd.Series]:
    if operator not in FLAG_OPERATORS:
        raise ValueError("Saved strategy indicators only support flag operators")
    if strategy_resolver is None:
        raise ValueError("Saved strategy resolver is required for strategy indicators")

    strategy_id = strategy_indicator_id(left_id)
    if not strategy_id:
        raise ValueError("Saved strategy indicator is missing a strategy id")
    if strategy_id in seen_strategy_ids:
        raise ValueError(f"Circular saved strategy reference: {strategy_id}")

    strategy = strategy_resolver(strategy_id)
    if strategy is None:
        raise ValueError(f"Unknown saved strategy: {strategy_id}")

    next_seen = {*seen_strategy_ids, strategy_id}
    nested, nested_ready = _compile_condition_state(
        data,
        _condition_dicts(strategy.conditions),
        strategy_resolver=strategy_resolver,
        seen_strategy_ids=next_seen,
    )
    part = nested if operator == "is true" else ~nested
    return part, nested_ready


def compile_condition_mask(
    data: pd.DataFrame,
    conditions: list[dict],
    *,
    strategy_resolver: StrategyResolver | None = None,
    seen_strategy_ids: set[str] | None = None,
) -> pd.Series:
    mask, _ready = _compile_condition_state(
        data,
        conditions,
        strategy_resolver=strategy_resolver,
        seen_strategy_ids=seen_strategy_ids,
    )
    return mask


def compile_entry_ready_mask(
    data: pd.DataFrame,
    conditions: list[dict],
    *,
    strategy_resolver: StrategyResolver | None = None,
    seen_strategy_ids: set[str] | None = None,
) -> pd.Series:
    """True on bars where every required entry input exists, whether or not it passes.

    Leading False bars are the stretch before a trade could be taken: an instrument
    is not listed yet, a regime day is unlabeled, or an indicator is still missing.
    A later False signal on a complete input stays True here.
    """
    _mask, ready = _compile_condition_state(
        data,
        conditions,
        strategy_resolver=strategy_resolver,
        seen_strategy_ids=seen_strategy_ids,
    )
    return ready


def compile_buy_mask(
    data: pd.DataFrame,
    conditions: list[dict],
    *,
    strategy_resolver: StrategyResolver | None = None,
) -> pd.Series:
    return compile_condition_mask(data, conditions, strategy_resolver=strategy_resolver)


def is_primary_only_entry_filter(condition: dict) -> bool:
    """Market-wide filters applied on the primary frame after cross-symbol confirm.

    Symbol-specific conditions (High, IBR, ADX, RSI2, strategy refs, …) stay inside
    the confirm buy_signal and run on every symbol. Breadth / Vix / SPYBull gate the
    merged primary Buy so they match indicator-sweep without changing confirm holds.
    """
    left_id = condition["left"]
    if is_strategy_indicator(left_id):
        return False
    return is_primary_only_indicator(left_id)


def split_entry_conditions_for_confirm(conditions) -> tuple[list[dict], list[dict]]:
    condition_dicts = _condition_dicts(conditions)
    confirm_conditions: list[dict] = []
    primary_filters: list[dict] = []
    for condition in condition_dicts:
        if is_primary_only_entry_filter(condition):
            primary_filters.append(condition)
        else:
            confirm_conditions.append(condition)
    return confirm_conditions, primary_filters


def uses_split_confirm_entry_filters(conditions, confirm_symbols: list[str] | None) -> bool:
    if not confirm_symbols:
        return False
    confirm_conditions, primary_filters = split_entry_conditions_for_confirm(conditions)
    return bool(confirm_conditions) and bool(primary_filters)


def compile_sell_mask(
    data: pd.DataFrame,
    conditions: list[dict],
    *,
    strategy_resolver: StrategyResolver | None = None,
) -> pd.Series:
    return compile_condition_mask(data, conditions, strategy_resolver=strategy_resolver)


def format_condition_preview(conditions: list[dict], labels: dict[str, str] | None = None) -> str:
    labels = labels or {}
    parts: list[str] = []
    for index, condition in enumerate(conditions):
        left = labels.get(condition["left"], condition["left"])
        operator = condition["operator"]
        if operator in FLAG_OPERATORS:
            snippet = f"{left} {operator}"
        else:
            right = labels.get(condition["right"], condition["right"])
            snippet = f"{left} {operator} {right}"
        if index == 0:
            parts.append(snippet)
        else:
            parts.append(f" {condition.get('logic', 'AND')} {snippet}")
    return "".join(parts)


def _format_snapshot_value(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


def _snapshot_operand_value(data: pd.DataFrame, value: str):
    if value in _INDICATOR_IDS:
        if value not in data.columns:
            return None
        return data[value].iloc[-1]
    try:
        return float(value)
    except ValueError:
        return None


def _snapshot_values_for_condition(
    data: pd.DataFrame,
    condition: dict,
) -> tuple[str | None, str | None]:
    left_id = condition["left"]
    operator = condition["operator"]
    right_value = condition["right"]

    if is_strategy_indicator(left_id) or operator in FLAG_OPERATORS:
        if is_strategy_indicator(left_id):
            return None, None
        if left_id not in data.columns:
            return None, None
        return _format_snapshot_value(data[left_id].iloc[-1]), None

    if left_id not in data.columns:
        return None, None

    left_value = _format_snapshot_value(data[left_id].iloc[-1])
    validate_compare_target(left_id, right_value)
    right_raw = _snapshot_operand_value(data, right_value)
    return left_value, _format_snapshot_value(right_raw)


def evaluate_condition_snapshot(
    data: pd.DataFrame,
    conditions: list[dict],
    *,
    strategy_resolver: StrategyResolver | None = None,
    labels: dict[str, str] | None = None,
) -> list[ConditionSnapshotItem]:
    if data.empty or not conditions:
        return []

    labels = labels or {}
    condition_dicts = _condition_dicts(conditions)
    results: list[ConditionSnapshotItem] = []

    for index, condition in enumerate(condition_dicts):
        part = compile_condition_mask(
            data,
            [condition],
            strategy_resolver=strategy_resolver,
        )
        passed = bool(part.iloc[-1])
        left_value, right_value = _snapshot_values_for_condition(data, condition)
        logic = None if index == 0 else condition.get("logic", "AND")

        results.append(
            {
                "label": format_condition_preview([condition], labels),
                "logic": logic,
                "passed": passed,
                "left_value": left_value,
                "right_value": right_value,
                "operator": condition["operator"],
                "left": condition["left"],
                "right": condition["right"],
            }
        )

    return results
