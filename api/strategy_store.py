from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from api.schemas import SaveStrategyRequest, SavedStrategy, UpdateStrategyRequest

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


def _model_dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _normalize_confirm_symbols(symbol: str, confirm_symbols: list[str]) -> list[str]:
    primary = symbol.strip().upper()
    normalized: list[str] = []
    for item in confirm_symbols:
        value = item.strip().upper()
        if not value or value == primary or value in normalized:
            continue
        normalized.append(value)
    return normalized


def _normalize_proxy_symbol(symbol: str, proxy_symbol: str | None) -> str | None:
    primary = symbol.strip().upper()
    if not proxy_symbol:
        return None
    value = proxy_symbol.strip().upper()
    if not value or value == primary:
        return None
    return value


def _validate_saved_strategy(item: dict) -> SavedStrategy:
    if hasattr(SavedStrategy, "model_validate"):
        return SavedStrategy.model_validate(item)
    return SavedStrategy.parse_obj(item)


def list_strategies(store_path: Path | None = None) -> list[SavedStrategy]:
    path = store_path or DEFAULT_STORE_PATH
    return [_validate_saved_strategy(item) for item in _load_raw(path)]


def get_strategy_by_id(strategy_id: str, store_path: Path | None = None) -> SavedStrategy | None:
    path = store_path or DEFAULT_STORE_PATH
    for item in _load_raw(path):
        if item.get("id") == strategy_id:
            return _validate_saved_strategy(item)
    return None


def create_strategy(
    request: SaveStrategyRequest,
    store_path: Path | None = None,
) -> SavedStrategy:
    path = store_path or DEFAULT_STORE_PATH
    now = _now_iso()
    symbol = request.symbol.strip().upper()
    saved = SavedStrategy(
        id=str(uuid4()),
        name=request.name.strip(),
        symbol=symbol,
        direction=request.direction,
        hold_days=request.hold_days,
        profit=request.profit,
        description=request.description.strip(),
        conditions=request.conditions,
        sell_conditions=request.sell_conditions,
        confirm_symbols=_normalize_confirm_symbols(symbol, request.confirm_symbols),
        proxy_symbol=_normalize_proxy_symbol(symbol, request.proxy_symbol),
        created_at=now,
        updated_at=now,
    )
    items = _load_raw(path)
    items.append(_model_dump(saved))
    _write_raw(path, items)
    return saved


def update_strategy(
    strategy_id: str,
    request: UpdateStrategyRequest,
    store_path: Path | None = None,
) -> SavedStrategy | None:
    path = store_path or DEFAULT_STORE_PATH
    items = _load_raw(path)
    for index, item in enumerate(items):
        if item.get("id") != strategy_id:
            continue
        updated = {
            **item,
            "description": request.description.strip(),
            "updated_at": _now_iso(),
        }
        items[index] = updated
        _write_raw(path, items)
        return _validate_saved_strategy(updated)
    return None


def delete_strategy(strategy_id: str, store_path: Path | None = None) -> bool:
    path = store_path or DEFAULT_STORE_PATH
    items = _load_raw(path)
    next_items = [item for item in items if item.get("id") != strategy_id]
    if len(next_items) == len(items):
        return False
    _write_raw(path, next_items)
    return True
