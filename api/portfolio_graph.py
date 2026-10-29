"""Validate saved portfolio references without flattening their allocation settings."""
from __future__ import annotations


def validate_graph(items: list[dict]) -> None:
    by_id = {item["id"]: item for item in items}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(identity: str) -> None:
        if identity in visiting:
            raise ValueError("Portfolio nesting cannot contain a cycle")
        if identity in visited:
            return
        if identity not in by_id:
            raise ValueError(f"Unknown child portfolio: {identity}")
        item = by_id[identity]
        children = item.get("portfolio_ids", [])
        if not item.get("strategy_ids") and not children:
            raise ValueError("At least one strategy or portfolio is required")
        if len(set(children)) != len(children):
            raise ValueError("A child portfolio can only be selected once per parent")
        visiting.add(identity)
        for child in children:
            visit(child)
        visiting.remove(identity)
        visited.add(identity)

    for identity in by_id:
        visit(identity)
