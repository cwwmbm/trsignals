"""Map legacy buy_signal functions to builder JSON strategies.

og_buy_signal and og_new_buy_signal are intentionally excluded — they stay in
signal_check.py and scan_service LEGACY_BUY_SIGNALS only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid5, NAMESPACE_URL

from pathlib import Path

from api.schemas import BuilderCondition, SavedStrategy

ConditionDict = dict[str, str]


@dataclass(frozen=True)
class LegacyStrategyTemplate:
    legacy_signal: str
    allowed_symbols: tuple[str, ...]
    hold_days: int
    profit: int
    description: str
    conditions: tuple[ConditionDict, ...]
    sell_conditions: tuple[ConditionDict, ...] = field(default_factory=tuple)


def _c(left: str, operator: str, right: str = "", logic: str = "AND") -> ConditionDict:
    return {"left": left, "operator": operator, "right": right, "logic": logic}


LEGACY_STRATEGY_TEMPLATES: tuple[LegacyStrategyTemplate, ...] = (
    LegacyStrategyTemplate(
        legacy_signal="buy_signal1",
        allowed_symbols=("XBI",),
        hold_days=2,
        profit=1,
        description="Long IBB: RSI5EnergyBreadth < 70, Close_EMA8 < 0, IBR < 0.4",
        conditions=(
            _c("Close_EMA8", "<", "0"),
            _c("IBR", "<", "0.4", logic="AND"),
            _c("ChangeVelocity", ">", "-1", logic="AND"),
            _c("RSI5IndustrialsBreadth", ">", "20", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal3",
        allowed_symbols=(),
        hold_days=2,
        profit=1,
        description="Long NQ: RSI5EnergyBreadth < 60, IBR3 < 0.7, Vix < 25, VFI10 > 0",
        conditions=(
            _c("RSI5EnergyBreadth", "<", "60"),
            _c("IBR3", "<", "0.7", logic="AND"),
            _c("Vix", "<", "25", logic="AND"),
            _c("VFI10", ">", "0", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal4",
        allowed_symbols=("FXI",),
        hold_days=3,
        profit=1,
        description="For SPY, add ValueCharts<0 condition. New strat for FXI: RSI2GoldBreadth > 50, Stoch < 90, RSI14SemisBreadth > 40",
        conditions=(
            _c("RSI2GoldBreadth", ">", "50"),
            _c("RSI14SemisBreadth", ">", "40", logic="AND"),
            _c("Stoch", "<", "90", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal6",
        allowed_symbols=(),
        hold_days=2,
        profit=1,
        description="Long NQ: RSI5RiskBreadth > 60, ValueCharts > 0",
        conditions=(
            _c("RSI5RiskBreadth", ">", "60"),
            _c("ValueCharts", ">", "0", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal7",
        allowed_symbols=("SMH", "QQQ", "FXI", "SOXX", "SPY"),
        hold_days=2,
        profit=1,
        description="Long NQ: Close 1 day ago <= Close 3 days ago, IBR <= 0.4",
        conditions=(
            _c("CloseLag1", "<=", "CloseLag3"),
            _c("IBR", "<=", "0.4", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal8",
        allowed_symbols=("QQQ", "SPY"),
        hold_days=3,
        profit=1,
        description="Long NQ: ER(10) > 0.50, ValueCharts(5) > -12, RSI2 <= 90, IBR <= 0.8, RSI5Breadth < 80",
        conditions=(
            _c("ER", ">", "0.50"),
            _c("ValueCharts", ">", "-12", logic="AND"),
            _c("RSI2", "<=", "90", logic="AND"),
            _c("IBR", "<=", "0.8", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal9",
        allowed_symbols=("SPY", "QQQ"),
        hold_days=100,
        profit=100,
        description="Stoch < 30, MACD < 0, IBR <= 0.2",
        conditions=(
            _c("Stoch", "<", "30"),
            _c("MACDHist", "<", "0", logic="AND"),
            _c("IBR", "<=", "0.2", logic="AND"),
        ),
        sell_conditions=(
            _c("RSI14IndustrialsBreadth", "<", "40"),
            _c("Stoch", ">", "20", logic="OR"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal10",
        allowed_symbols=("SMH", "SPY", "SOXX", "QQQ"),
        hold_days=3,
        profit=1,
        description="Long NQ: Low 1 day ago <= Lowest Low in 2 days, IBR <= 50",
        conditions=(
            _c("LowLag1", "<=", "LowMin2Lag1"),
            _c("IBR", "<=", "0.50", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal11",
        allowed_symbols=(),
        hold_days=2,
        profit=1,
        description="Replacebale",
        conditions=(
            _c("VFI80", ">", "0"),
            _c("EMA8CrossDown", "is false", logic="AND"),
            _c("IBR3", "<", "0.8", logic="AND"),
            _c("RSI5IndustrialsBreadth", "<", "80", logic="AND"),
            _c("RSI5SemisBreadth", ">", "20", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal13",
        allowed_symbols=("CL",),
        hold_days=2,
        profit=1,
        description="Long CL: close[0] > SMA(close,100)[0], rsi(close,2)[0] <= 60, ValueClose(5)[0] > -4, IBR[0] <= 20",
        conditions=(
            _c("Close", ">", "SMA100"),
            _c("RSI2", "<=", "60", logic="AND"),
            _c("ValueCharts", ">", "-4", logic="AND"),
            _c("IBR", "<=", "0.2", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal14",
        allowed_symbols=(),
        hold_days=1,
        profit=1,
        description="Long SMH: low[0] = lowest(low,4)[0], IBR[0] <= 30",
        conditions=(
            _c("Low", "=", "Low"),
            _c("IBR", "<=", "0.3", logic="AND"),
            _c("ER", ">", "0.3", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal15",
        allowed_symbols=("CL",),
        hold_days=2,
        profit=1,
        description="Long CL: KaufmanEfficiencyRatio(10)[0] > 20, rsi(close,14)[0] >= 40, IBR[0] <= 20",
        conditions=(
            _c("ER", ">", "0.2"),
            _c("RSI14", ">=", "40", logic="AND"),
            _c("IBR", "<=", "0.2", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal16",
        allowed_symbols=("SMH", "QQQ", "SOXX"),
        hold_days=4,
        profit=1,
        description="high[0] > close[1], IBR[0] <= 50, SMA50>SMA200, Hurst > 0.4",
        conditions=(
            _c("High", ">", "CloseLag1"),
            _c("IBR", "<=", "0.5", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal17",
        allowed_symbols=("SMH", "SOXX", "QQQ"),
        hold_days=2,
        profit=1,
        description="Long SMH: rsi(close,2)[0] <= 20, IBR[0] <= 30, ER>0.3",
        conditions=(
            _c("RSI2", "<=", "20"),
            _c("IBR", "<=", "0.3", logic="AND"),
            _c("ER", ">=", "0.3", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal18",
        allowed_symbols=("GDX",),
        hold_days=2,
        profit=1,
        description="Long CL: open[1] <= lowest(open,2)[0], IBR[0] <= 20, Stochastics(14)[0] >= 10",
        conditions=(
            _c("IBR", "<=", "0.2"),
            _c("ER", "<=", "0.6", logic="AND"),
            _c("Stoch", ">=", "10", logic="AND"),
            _c("RSI14EnergyBreadth", "<", "60", logic="AND"),
            _c("RSI14EnergyBreadth", ">", "30", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal19",
        allowed_symbols=(),
        hold_days=100,
        profit=100,
        description="Long GC: Vix[0] <= Vix[1], rsi(close,14)[0] >= 30, IBR[0] <= 50",
        conditions=(
            _c("Vix", "<=", "10"),
        ),
        sell_conditions=(
            _c("Vix", ">", "11"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal20",
        allowed_symbols=("SPY", "QQQ", "IWM"),
        hold_days=50,
        profit=50,
        description="Experimental Long signal",
        conditions=(
            _c("RSI2", "<", "40"),
            _c("IBR", "<", "0.2", logic="AND"),
            _c("RSI5Breadth", "<", "60", logic="AND"),
        ),
        sell_conditions=(
            _c("RSI2SemisBreadth", "crosses below", "50"),
            _c("Vix", ">", "40", logic="OR"),
            _c("ChangeVelocity", ">", "1", logic="OR"),
            _c("RSI2SemisBreadth", "<", "10", logic="OR"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal21",
        allowed_symbols=("SPY",),
        hold_days=1,
        profit=1,
        description="Close<200SMA, RSI2<40, RSI2SemisBreadth>30. Bear market long signal",
        conditions=(
            _c("SMA50_SMA200", "<", "0"),
            _c("RSI2", "<", "40", logic="AND"),
            _c("RSI2SemisBreadth", ">", "30", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal22",
        allowed_symbols=("SPY", "QQQ", "ES", "NQ"),
        hold_days=5,
        profit=5,
        description="Experimental Long signal",
        conditions=(
            _c("RSI2RiskBreadth", ">", "95"),
            _c("RSI5RiskBreadth", ">", "85", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal23",
        allowed_symbols=("FXI",),
        hold_days=2,
        profit=1,
        description="low[0] >= highest(low,2)[0], close[1] <= lowest(close,5)[1], rsi(close,2)[0] >= 15",
        conditions=(
            _c("LowerCloses2", "is true"),
            _c("RSI5SemisBreadth", ">", "40", logic="AND"),
        ),
    ),
    LegacyStrategyTemplate(
        legacy_signal="buy_signal24",
        allowed_symbols=("GDX",),
        hold_days=2,
        profit=1,
        description="Testing",
        conditions=(
            _c("RSI5GoldBreadth", ">", "60"),
            _c("RSI14UtilitiesBreadth", "<", "50", logic="AND"),
        ),
    ),
)

# buy_signal2, buy_signal5, buy_signal12 use buy=True or empty allowed_symbols with
# degenerate logic — not migrated. buy_signal14 uses Low == rolling min which needs
# a dedicated column; kept as template with empty symbols for reference only.

MIGRATED_LEGACY_SIGNALS = frozenset(template.legacy_signal for template in LEGACY_STRATEGY_TEMPLATES)

SCAN_SYMBOLS = frozenset(
    {
        "SPY",
        "SMH",
        "QQQ",
        "SOXX",
        "^VIX",
        "XLI",
        "XLU",
        "XLE",
        "XLF",
        "RSP",
        "IWM",
        "FXI",
        "GDX",
        "GLD",
        "XBI",
        "TLT",
    }
)


def _stable_id(legacy_signal: str, symbol: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"legacy-strategy:{legacy_signal}:{symbol}"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_migrated_strategies(*, include_non_scan: bool = True) -> list[SavedStrategy]:
    """Build SavedStrategy entries for every legacy template × allowed symbol."""
    now = _now_iso()
    strategies: list[SavedStrategy] = []

    for template in LEGACY_STRATEGY_TEMPLATES:
        for symbol in template.allowed_symbols:
            if not include_non_scan and symbol not in SCAN_SYMBOLS:
                continue
            conditions = [BuilderCondition.model_validate(item) for item in template.conditions]
            sell_conditions = [
                BuilderCondition.model_validate(item) for item in template.sell_conditions
            ]
            strategies.append(
                SavedStrategy(
                    id=_stable_id(template.legacy_signal, symbol),
                    name=template.legacy_signal,
                    symbol=symbol,
                    direction="long",
                    hold_days=template.hold_days,
                    profit=template.profit,
                    description=template.description,
                    conditions=conditions,
                    sell_conditions=sell_conditions,
                    legacy_signal=template.legacy_signal,
                    created_at=now,
                    updated_at=now,
                )
            )

    return strategies


def build_migrated_strategies_payload(*, include_non_scan: bool = True) -> list[dict]:
    return [strategy.model_dump() for strategy in build_migrated_strategies(include_non_scan=include_non_scan)]


def is_custom_strategy(item: dict) -> bool:
    """Builder-created strategies have no legacy_signal; migrated ones always do."""
    return not item.get("legacy_signal")


def merge_migrated_with_store(
    existing: list[dict],
    *,
    include_non_scan: bool = True,
) -> list[dict]:
    """Keep custom strategies and replace all legacy-migrated entries."""
    custom = [item for item in existing if is_custom_strategy(item)]
    migrated = build_migrated_strategies_payload(include_non_scan=include_non_scan)
    return custom + migrated


def sync_migrated_strategies(
    store_path: Path | None = None,
    *,
    include_non_scan: bool = True,
) -> tuple[int, int]:
    """Write merged strategies.json. Returns (custom_count, migrated_count)."""
    from api.strategy_store import DEFAULT_STORE_PATH, _load_raw, _write_raw

    path = store_path or DEFAULT_STORE_PATH
    existing = _load_raw(path)
    merged = merge_migrated_with_store(existing, include_non_scan=include_non_scan)
    _write_raw(path, merged)
    custom_count = sum(1 for item in merged if is_custom_strategy(item))
    migrated_count = len(merged) - custom_count
    return custom_count, migrated_count
