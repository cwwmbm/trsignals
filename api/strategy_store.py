from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from api.schemas import SaveStrategyRequest, SavedStrategy

DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "strategies.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Strategy store must contain a JSON array")
    return payload


def _write_raw(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(items, handle, indent=2)
        handle.write("\n")


def list_strategies(store_path: Path | None = None) -> list[SavedStrategy]:
    path = store_path or DEFAULT_STORE_PATH
    return [SavedStrategy.model_validate(item) for item in _load_raw(path)]


def get_strategy_by_id(strategy_id: str, store_path: Path | None = None) -> SavedStrategy | None:
    path = store_path or DEFAULT_STORE_PATH
    for item in _load_raw(path):
        if item.get("id") == strategy_id:
            return SavedStrategy.model_validate(item)
    return None


def create_strategy(
    request: SaveStrategyRequest,
    store_path: Path | None = None,
) -> SavedStrategy:
    path = store_path or DEFAULT_STORE_PATH
    now = _now_iso()
    saved = SavedStrategy(
        id=str(uuid4()),
        name=request.name.strip(),
        symbol=request.symbol.strip().upper(),
        direction=request.direction,
        hold_days=request.hold_days,
        profit=request.profit,
        description=request.description.strip(),
        conditions=request.conditions,
        sell_conditions=request.sell_conditions,
        created_at=now,
        updated_at=now,
    )
    items = _load_raw(path)
    items.append(saved.model_dump())
    _write_raw(path, items)
    return saved
