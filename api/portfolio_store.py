from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from api.portfolio_graph import validate_graph
from api.regime_allocation import RegimePenaltySettings
from api.schemas import (
    RegimePenaltyFields,
    SavePortfolioRequest,
    SavedPortfolio,
    UpdatePortfolioRequest,
)
from api.strategy_store import get_strategy_by_id

# A portfolio owns its allocation: model, proportional mode, costs, and the regime penalties.
# Strategies carry none of it, so saving a portfolio is what keeps these settings.
REGIME_FIELDS = tuple(RegimePenaltyFields.__fields__)

DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "portfolios.json"


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
            raise ValueError(f"Portfolio store JSON is invalid: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("Portfolio store must contain a JSON array")
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


def _check_regime_settings(values) -> None:
    """Reject regime penalties that cannot run, so a bad portfolio is never saved."""
    get = values.get if isinstance(values, dict) else lambda key, default=None: getattr(values, key, default)
    if get("allocation_model") != "regime_adjusted":
        return

    class _Source:
        pass

    source = _Source()
    for key in REGIME_FIELDS:
        setattr(source, key, get(key))
    RegimePenaltySettings.from_source(source)


def _normalize_proxy_symbol(proxy_symbol: str | None) -> str | None:
    if not proxy_symbol:
        return None
    value = proxy_symbol.strip().upper()
    return value or None


def _validate_strategy_ids(strategy_ids: list[str]) -> list[str]:
    normalized: list[str] = []
    for strategy_id in strategy_ids:
        strategy = get_strategy_by_id(strategy_id)
        if strategy is None:
            raise ValueError(f"Unknown strategy: {strategy_id}")
        if strategy.direction != "long":
            raise ValueError(
                f"Short strategies are not supported in portfolios: {strategy.name}"
            )
        normalized.append(strategy_id)
    return normalized


def _validate_saved_portfolio(item: dict) -> SavedPortfolio:
    if hasattr(SavedPortfolio, "model_validate"):
        return SavedPortfolio.model_validate(item)
    return SavedPortfolio.parse_obj(item)


def list_portfolios(store_path: Path | None = None) -> list[SavedPortfolio]:
    path = store_path or DEFAULT_STORE_PATH
    return [_validate_saved_portfolio(item) for item in _load_raw(path)]


def get_portfolio_by_id(portfolio_id: str, store_path: Path | None = None) -> SavedPortfolio | None:
    path = store_path or DEFAULT_STORE_PATH
    for item in _load_raw(path):
        if item.get("id") == portfolio_id:
            return _validate_saved_portfolio(item)
    return None


def find_portfolio_by_name(name: str, store_path: Path | None = None) -> SavedPortfolio | None:
    """Return the most recently updated portfolio matching name, if any."""
    needle = name.strip().casefold()
    if not needle:
        return None
    matches = [
        portfolio
        for portfolio in list_portfolios(store_path=store_path)
        if portfolio.name.strip().casefold() == needle
    ]
    if not matches:
        return None
    return max(matches, key=lambda portfolio: portfolio.updated_at)


def create_portfolio(
    request: SavePortfolioRequest,
    store_path: Path | None = None,
) -> SavedPortfolio:
    path = store_path or DEFAULT_STORE_PATH
    strategy_ids = _validate_strategy_ids(list(request.strategy_ids))
    _check_regime_settings(request)
    now = _now_iso()
    saved = SavedPortfolio(
        id=str(uuid4()),
        name=request.name.strip(),
        description=request.description.strip(),
        strategy_ids=strategy_ids,
        portfolio_ids=list(request.portfolio_ids),
        overlap_mode=request.overlap_mode,
        proxy_symbol=_normalize_proxy_symbol(request.proxy_symbol),
        allocation_model=request.allocation_model,
        proportional_mode=request.proportional_mode,
        tier_single_pct=request.tier_single_pct,
        transaction_cost_bps=request.transaction_cost_bps,
        **{key: getattr(request, key) for key in REGIME_FIELDS},
        scan_lane="testing",
        scan_sort_order=0,
        created_at=now,
        updated_at=now,
    )
    with _store_lock(path):
        items = _load_raw(path)
        items.append(_model_dump(saved))
        validate_graph(items)
        _write_raw(path, items)
    return saved


def update_portfolio(
    portfolio_id: str,
    request: UpdatePortfolioRequest,
    store_path: Path | None = None,
) -> SavedPortfolio | None:
    path = store_path or DEFAULT_STORE_PATH
    with _store_lock(path):
        items = _load_raw(path)
        for index, item in enumerate(items):
            if item.get("id") != portfolio_id:
                continue
            if hasattr(request, "model_dump"):
                payload = request.model_dump(exclude_unset=True)
            else:
                payload = request.dict(exclude_unset=True)
            if not payload:
                return _validate_saved_portfolio(item)
            updated = {**item, "updated_at": _now_iso()}
            if "name" in payload and payload["name"] is not None:
                updated["name"] = payload["name"].strip()
            if "description" in payload and payload["description"] is not None:
                updated["description"] = payload["description"].strip()
            if "scan_lane" in payload and payload["scan_lane"] is not None:
                updated["scan_lane"] = payload["scan_lane"]
            if "scan_sort_order" in payload and payload["scan_sort_order"] is not None:
                updated["scan_sort_order"] = payload["scan_sort_order"]
            if "strategy_ids" in payload and payload["strategy_ids"] is not None:
                updated["strategy_ids"] = _validate_strategy_ids(list(payload["strategy_ids"]))
            if "portfolio_ids" in payload and payload["portfolio_ids"] is not None:
                updated["portfolio_ids"] = list(payload["portfolio_ids"])
            if "overlap_mode" in payload and payload["overlap_mode"] is not None:
                updated["overlap_mode"] = payload["overlap_mode"]
            if "proxy_symbol" in payload:
                updated["proxy_symbol"] = _normalize_proxy_symbol(payload["proxy_symbol"])
            for key in ("allocation_model", "proportional_mode", "tier_single_pct", "transaction_cost_bps", *REGIME_FIELDS):
                if key in payload and payload[key] is not None:
                    updated[key] = payload[key]
            _check_regime_settings(updated)
            items[index] = updated
            validate_graph(items)
            _write_raw(path, items)
            return _validate_saved_portfolio(updated)
    return None


def delete_portfolio(portfolio_id: str, store_path: Path | None = None) -> bool:
    path = store_path or DEFAULT_STORE_PATH
    with _store_lock(path):
        items = _load_raw(path)
        parents = [item["name"] for item in items if portfolio_id in item.get("portfolio_ids", [])]
        if parents:
            raise ValueError(f"Portfolio is used by: {', '.join(parents)}")
        next_items = [item for item in items if item.get("id") != portfolio_id]
        if len(next_items) == len(items):
            return False
        _write_raw(path, next_items)
    return True
