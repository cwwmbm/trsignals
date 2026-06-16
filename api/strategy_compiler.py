from __future__ import annotations

import pandas as pd

from api.indicator_catalog import builder_eligible_ids


VALID_OPERATORS = {"<", "<=", ">", ">=", "=", "crosses above", "crosses below", "is true", "is false"}
FLAG_OPERATORS = {"is true", "is false"}
_INDICATOR_IDS = set(builder_eligible_ids())


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


def compile_condition_mask(data: pd.DataFrame, conditions: list[dict]) -> pd.Series:
    if not conditions:
        raise ValueError("At least one condition is required")

    mask: pd.Series | None = None
    for index, condition in enumerate(conditions):
        left_id = condition["left"]
        operator = condition["operator"]
        right_value = condition["right"]
        logic = condition.get("logic", "AND")

        if left_id not in _INDICATOR_IDS:
            raise ValueError(f"Unknown left indicator: {left_id}")
        if left_id not in data.columns:
            raise ValueError(f"Indicator column not found in data: {left_id}")
        if operator not in VALID_OPERATORS:
            raise ValueError(f"Unsupported operator: {operator}")

        left_series = data[left_id]
        if operator in FLAG_OPERATORS:
            part = left_series == (1 if operator == "is true" else -1)
        else:
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


def compile_buy_mask(data: pd.DataFrame, conditions: list[dict]) -> pd.Series:
    return compile_condition_mask(data, conditions)


def compile_sell_mask(data: pd.DataFrame, conditions: list[dict]) -> pd.Series:
    return compile_condition_mask(data, conditions)


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
