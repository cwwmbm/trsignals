from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from api.schemas import SaveStrategyRequest, SavedStrategy, UpdateStrategyRequest

DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "strategies.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def _store_lock(path: Path):
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Strategy store JSON is invalid: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("Strategy store must contain a JSON array")
    return payload


def _write_raw(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(items, handle, indent=2)
        handle.write("\n")
    temp_path.replace(path)


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


def find_strategy_by_name_symbol(
    name: str,
    symbol: str,
    store_path: Path | None = None,
) -> SavedStrategy | None:
    """Return the most recently updated strategy matching name+symbol, if any."""
    needle_name = name.strip().casefold()
    needle_symbol = symbol.strip().upper()
    if not needle_name or not needle_symbol:
        return None
    matches = [
        strategy
        for strategy in list_strategies(store_path=store_path)
        if strategy.name.strip().casefold() == needle_name
        and strategy.symbol.strip().upper() == needle_symbol
    ]
    if not matches:
        return None
    return max(matches, key=lambda strategy: strategy.updated_at)


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
        hold_on_buy_signal=request.hold_on_buy_signal,
        rth_entries_only=request.rth_entries_only,
        eod_exit=request.eod_exit,
        scan_lane="testing",
        scan_sort_order=0,
        created_at=now,
        updated_at=now,
    )
    with _store_lock(path):
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
    with _store_lock(path):
        items = _load_raw(path)
        for index, item in enumerate(items):
            if item.get("id") != strategy_id:
                continue
            if hasattr(request, "model_dump"):
                payload = request.model_dump(exclude_unset=True)
            else:
                payload = request.dict(exclude_unset=True)
            if not payload:
                return _validate_saved_strategy(item)
            updated = {**item, "updated_at": _now_iso()}
            if "name" in payload and payload["name"] is not None:
                updated["name"] = payload["name"].strip()
            if "symbol" in payload and payload["symbol"] is not None:
                updated["symbol"] = payload["symbol"].strip().upper()
            if "direction" in payload and payload["direction"] is not None:
                updated["direction"] = payload["direction"]
            if "hold_days" in payload and payload["hold_days"] is not None:
                updated["hold_days"] = payload["hold_days"]
            if "profit" in payload and payload["profit"] is not None:
                updated["profit"] = payload["profit"]
            if "description" in payload and payload["description"] is not None:
                updated["description"] = payload["description"].strip()
            if "conditions" in payload and payload["conditions"] is not None:
                updated["conditions"] = payload["conditions"]
            if "sell_conditions" in payload and payload["sell_conditions"] is not None:
                updated["sell_conditions"] = payload["sell_conditions"]
            symbol_for_norm = updated.get("symbol", item.get("symbol", ""))
            if "confirm_symbols" in payload and payload["confirm_symbols"] is not None:
                updated["confirm_symbols"] = _normalize_confirm_symbols(
                    symbol_for_norm, list(payload["confirm_symbols"])
                )
            if "proxy_symbol" in payload:
                updated["proxy_symbol"] = _normalize_proxy_symbol(
                    symbol_for_norm, payload["proxy_symbol"]
                )
            if "hold_on_buy_signal" in payload and payload["hold_on_buy_signal"] is not None:
                updated["hold_on_buy_signal"] = payload["hold_on_buy_signal"]
            if "rth_entries_only" in payload and payload["rth_entries_only"] is not None:
                updated["rth_entries_only"] = payload["rth_entries_only"]
            if "eod_exit" in payload and payload["eod_exit"] is not None:
                updated["eod_exit"] = payload["eod_exit"]
            if "scan_lane" in payload and payload["scan_lane"] is not None:
                updated["scan_lane"] = payload["scan_lane"]
            if "scan_sort_order" in payload and payload["scan_sort_order"] is not None:
                updated["scan_sort_order"] = payload["scan_sort_order"]
            # Re-normalize confirm/proxy if symbol changed but those fields were not sent.
            if "symbol" in payload and "confirm_symbols" not in payload:
                updated["confirm_symbols"] = _normalize_confirm_symbols(
                    updated["symbol"], list(updated.get("confirm_symbols") or [])
                )
            if "symbol" in payload and "proxy_symbol" not in payload:
                updated["proxy_symbol"] = _normalize_proxy_symbol(
                    updated["symbol"], updated.get("proxy_symbol")
                )
            items[index] = updated
            _write_raw(path, items)
            return _validate_saved_strategy(updated)
    return None


def delete_strategy(strategy_id: str, store_path: Path | None = None) -> bool:
    path = store_path or DEFAULT_STORE_PATH
    with _store_lock(path):
        items = _load_raw(path)
        next_items = [item for item in items if item.get("id") != strategy_id]
        if len(next_items) == len(items):
            return False
        _write_raw(path, next_items)
    return True
