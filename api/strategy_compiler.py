from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from api.indicator_catalog import builder_eligible_ids, validate_compare_target


VALID_OPERATORS = {"<", "<=", ">", ">=", "=", "crosses above", "crosses below", "is true", "is false"}
FLAG_OPERATORS = {"is true", "is false"}
STRATEGY_INDICATOR_PREFIX = "strategy:"
_INDICATOR_IDS = set(builder_eligible_ids())
StrategyResolver = Callable[[str], Any | None]


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


def _compile_strategy_indicator(
    data: pd.DataFrame,
    left_id: str,
    operator: str,
    strategy_resolver: StrategyResolver | None,
    seen_strategy_ids: set[str],
) -> pd.Series:
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
    nested = compile_condition_mask(
        data,
        _condition_dicts(strategy.conditions),
        strategy_resolver=strategy_resolver,
        seen_strategy_ids=next_seen,
    )
    return nested if operator == "is true" else ~nested


def compile_condition_mask(
    data: pd.DataFrame,
    conditions: list[dict],
    *,
    strategy_resolver: StrategyResolver | None = None,
    seen_strategy_ids: set[str] | None = None,
) -> pd.Series:
    if not conditions:
        raise ValueError("At least one condition is required")

    seen_strategy_ids = seen_strategy_ids or set()
    mask: pd.Series | None = None
    for index, condition in enumerate(conditions):
        left_id = condition["left"]
        operator = condition["operator"]
        right_value = condition["right"]
        logic = condition.get("logic", "AND")

        if operator not in VALID_OPERATORS:
            raise ValueError(f"Unsupported operator: {operator}")

        if is_strategy_indicator(left_id):
            part = _compile_strategy_indicator(
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
            if operator in FLAG_OPERATORS:
                part = left_series == (1 if operator == "is true" else -1)
            else:
                validate_compare_target(left_id, right_value)
                right_operand = _resolve_operand(data, right_value)
                part = _compare(left_series, operator, right_operand)
        part = part.fillna(False)

        if mask is None:
            mask = part
            continue

        if logic == "AND":
            mask = mask & part
        elif logic == "OR":
            mask = mask | part
        else:
            raise ValueError(f"Unsupported logic: {logic}")

    assert mask is not None
    return mask


def compile_buy_mask(
    data: pd.DataFrame,
    conditions: list[dict],
    *,
    strategy_resolver: StrategyResolver | None = None,
) -> pd.Series:
    return compile_condition_mask(data, conditions, strategy_resolver=strategy_resolver)


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
