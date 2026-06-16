from __future__ import annotations

from typing import Literal, TypedDict


IndicatorKind = Literal[
    "price",
    "volume",
    "reference",
    "breadth",
    "momentum",
    "trend",
    "volatility",
    "pattern",
    "composite",
    "signal_flag",
]

ValueType = Literal["continuous", "percent", "ratio", "flag"]


class TypicalRange(TypedDict):
    min: float
    max: float


class IndicatorDef(TypedDict, total=False):
    id: str
    label: str
    kind: IndicatorKind
    valueType: ValueType
    category: str
    description: str
    typicalRange: TypicalRange
    builderEligible: bool
    aliases: list[str]


def _entry(
    id: str,
    label: str,
    category: str,
    kind: IndicatorKind,
    value_type: ValueType,
    *,
    description: str = "",
    typical_range: TypicalRange | None = None,
    builder_eligible: bool = True,
    aliases: list[str] | None = None,
) -> IndicatorDef:
    item: IndicatorDef = {
        "id": id,
        "label": label,
        "category": category,
        "kind": kind,
        "valueType": value_type,
        "builderEligible": builder_eligible,
    }
    if description:
        item["description"] = description
    if typical_range is not None:
        item["typicalRange"] = typical_range
    if aliases:
        item["aliases"] = aliases
    return item


# Threshold hints copied from indicator_sweep.py where available.
_SWEEP = {
    "RSI2": {"min": 10, "max": 50},
    "RSI5": {"min": 20, "max": 90},
    "RSI14": {"min": 20, "max": 90},
    "IBR": {"min": 0.1, "max": 0.9},
    "IBR2": {"min": 0.1, "max": 0.9},
    "IBR3": {"min": 0.1, "max": 0.9},
    "CCI": {"min": -150, "max": 150},
    "Stoch": {"min": 10, "max": 90},
    "ValueCharts": {"min": -12, "max": 12},
    "ER": {"min": 0.1, "max": 0.9},
    "Vix": {"min": 10, "max": 50},
    "ChangeVelocity": {"min": -2, "max": 2},
    "Close_EMA8": {"min": -10, "max": 10},
    "VFI10": {"min": -8, "max": 8},
    "VFI20": {"min": -8, "max": 8},
    "VFI40": {"min": -8, "max": 8},
    "VFI80": {"min": -8, "max": 8},
    "%Change": {"min": -0.06, "max": 0.06},
}


def _range(indicator_id: str) -> TypicalRange | None:
    return _SWEEP.get(indicator_id)


def _breadth_rsi(window: int, source: str, label_source: str) -> IndicatorDef:
    column = f"RSI{window}{source}Breadth"
    return _entry(
        column,
        f"RSI({window}) {label_source} breadth",
        "Breadth RSI",
        "breadth",
        "continuous",
        typical_range={"min": 10, "max": 90},
    )


INDICATOR_CATALOG: list[IndicatorDef] = [
    # Price
    _entry("Open", "Open", "Price", "price", "continuous"),
    _entry("High", "High", "Price", "price", "continuous"),
    _entry("Low", "Low", "Price", "price", "continuous"),
    _entry("Close", "Close", "Price", "price", "continuous"),
    # Volume
    _entry("Volume", "Volume", "Volume", "volume", "continuous"),
    _entry(
        "VolumeEMADiff",
        "Volume EMA diff %",
        "Volume",
        "volume",
        "percent",
        description="Volume deviation from 8-day EMA",
    ),
    # Reference markets
    _entry("Vix", "VIX", "Reference markets", "reference", "continuous", typical_range=_range("Vix"), aliases=["VIX"]),
    _entry("Spy", "SPY", "Reference markets", "reference", "continuous", aliases=["SPY"]),
    _entry("Qqq", "QQQ", "Reference markets", "reference", "continuous", aliases=["QQQ"]),
    _entry("Soxx", "SOXX", "Reference markets", "reference", "continuous", aliases=["SOXX"]),
    _entry("Iwm", "IWM", "Reference markets", "reference", "continuous", aliases=["IWM"]),
    _entry("Uvxy", "UVXY", "Reference markets", "reference", "continuous", aliases=["UVXY"]),
    _entry("Sqqq", "SQQQ", "Reference markets", "reference", "continuous", aliases=["SQQQ"]),
    # Breadth (raw ratios)
    _entry("Breadth", "RSP/SPY breadth", "Breadth (raw ratios)", "breadth", "ratio"),
    _entry("Riskbreadth", "QQQ/SPY risk breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["RiskBreadth"]),
    _entry("Semisbreadth", "SMH/SPY semis breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["SemisBreadth"]),
    _entry("Financialsbreadth", "XLF/SPY financials breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["FinancialsBreadth"]),
    _entry("Energybreadth", "XLE/SPY energy breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["EnergyBreadth"]),
    _entry("Utilitiesbreadth", "XLU/SPY utilities breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["UtilitiesBreadth"]),
    _entry("Industrialsbreadth", "XLI/SPY industrials breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["IndustrialsBreadth"]),
    _entry("Goldbreadth", "GLD/SPY gold breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["GoldBreadth"]),
    _entry("Bondbreadth", "TLT/SPY bond breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["BondBreadth"]),
    _entry("Iwmbreadth", "IWM/SPY breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["IWMBreadth"]),
    # Moving averages
    _entry("SMA10", "SMA(10)", "Moving averages", "trend", "continuous"),
    _entry("SMA20", "SMA(20)", "Moving averages", "trend", "continuous"),
    _entry("SMA50", "SMA(50)", "Moving averages", "trend", "continuous"),
    _entry("SMA100", "SMA(100)", "Moving averages", "trend", "continuous"),
    _entry("SMA200", "SMA(200)", "Moving averages", "trend", "continuous"),
    _entry("EMA8", "EMA(8)", "Moving averages", "trend", "continuous"),
    _entry("EMA20", "EMA(20)", "Moving averages", "trend", "continuous"),
    _entry("EMA100", "EMA(100)", "Moving averages", "trend", "continuous"),
    _entry("BBUpper", "Bollinger upper", "Moving averages", "trend", "continuous"),
    _entry("BBMiddle", "Bollinger middle", "Moving averages", "trend", "continuous"),
    _entry("BBLower", "Bollinger lower", "Moving averages", "trend", "continuous"),
    # Momentum / oscillators
    _entry("RSI2", "RSI(2)", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("RSI2")),
    _entry("RSI5", "RSI(5)", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("RSI5")),
    _entry("RSI14", "RSI(14)", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("RSI14")),
    _entry("Stoch", "Stochastic", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("Stoch")),
    _entry("StochOscilator", "Stochastic oscillator", "Momentum / oscillators", "momentum", "continuous"),
    _entry("CCI", "CCI(20)", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("CCI")),
    _entry("MACDHist", "MACD histogram", "Momentum / oscillators", "momentum", "continuous"),
    _entry("ValueCharts", "ValueCharts(5)", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("ValueCharts")),
    _entry("Hurst", "Hurst exponent", "Momentum / oscillators", "momentum", "continuous"),
    # Volatility / risk
    _entry("%Change", "% change", "Volatility / risk", "volatility", "percent", typical_range=_range("%Change")),
    _entry("ATR20", "ATR(20)", "Volatility / risk", "volatility", "continuous"),
    _entry("ATR50", "ATR(50)", "Volatility / risk", "volatility", "continuous"),
    _entry("Volatility", "Realized volatility", "Volatility / risk", "volatility", "continuous"),
    _entry("ChangeVelocity", "Change velocity", "Volatility / risk", "volatility", "continuous", typical_range=_range("ChangeVelocity")),
    # Efficiency / flow
    _entry("ER", "Efficiency ratio", "Efficiency / flow", "momentum", "continuous", typical_range=_range("ER")),
    _entry("IBR", "IBR", "Efficiency / flow", "momentum", "continuous", typical_range=_range("IBR")),
    _entry("IBR2", "IBR(2)", "Efficiency / flow", "momentum", "continuous", typical_range=_range("IBR2")),
    _entry("IBR3", "IBR(3)", "Efficiency / flow", "momentum", "continuous", typical_range=_range("IBR3")),
    _entry("VFI10", "VFI(10)", "Efficiency / flow", "momentum", "continuous", typical_range=_range("VFI10")),
    _entry("VFI20", "VFI(20)", "Efficiency / flow", "momentum", "continuous", typical_range=_range("VFI20")),
    _entry("VFI40", "VFI(40)", "Efficiency / flow", "momentum", "continuous", typical_range=_range("VFI40")),
    _entry("VFI80", "VFI(80)", "Efficiency / flow", "momentum", "continuous", typical_range=_range("VFI80")),
    # Spreads & composites
    _entry("Close_EMA8", "Close vs EMA(8) %", "Spreads & composites", "composite", "percent", typical_range=_range("Close_EMA8")),
    _entry("Close_SMA20", "Close vs SMA(20) %", "Spreads & composites", "composite", "percent"),
    _entry("Close_SMA50", "Close vs SMA(50) %", "Spreads & composites", "composite", "percent"),
    _entry("Close_SMA200", "Close vs SMA(200) %", "Spreads & composites", "composite", "percent"),
    _entry("EMA20_EMA100", "EMA(20) vs EMA(100) %", "Spreads & composites", "composite", "percent"),
    _entry("SMA20_SMA50", "SMA(20) vs SMA(50) %", "Spreads & composites", "composite", "percent"),
    _entry("SMA50_SMA200", "SMA(50) vs SMA(200) %", "Spreads & composites", "composite", "percent"),
    _entry("ATR20_ATR50", "ATR(20) - ATR(50)", "Spreads & composites", "composite", "continuous"),
    # Pattern / signal flags
    _entry("HigherCloses2", "Higher closes (2)", "Pattern / signal flags", "pattern", "flag"),
    _entry("HigherCloses3", "Higher closes (3)", "Pattern / signal flags", "pattern", "flag"),
    _entry("LowerCloses2", "Lower closes (2)", "Pattern / signal flags", "pattern", "flag"),
    _entry("LowerCloses3", "Lower closes (3)", "Pattern / signal flags", "pattern", "flag"),
    _entry("HighestClose2", "Highest close (2)", "Pattern / signal flags", "pattern", "flag"),
    _entry("HighestClose3", "Highest close (3)", "Pattern / signal flags", "pattern", "flag"),
    _entry("LowestClose2", "Lowest close (2)", "Pattern / signal flags", "pattern", "flag"),
    _entry("LowestClose3", "Lowest close (3)", "Pattern / signal flags", "pattern", "flag"),
    _entry("CloseLag1", "Close (1 bar ago)", "Pattern / signal flags", "pattern", "continuous"),
    _entry("CloseLag2", "Close (2 bars ago)", "Pattern / signal flags", "pattern", "continuous"),
    _entry("CloseLag3", "Close (3 bars ago)", "Pattern / signal flags", "pattern", "continuous"),
    _entry("LowLag1", "Low (1 bar ago)", "Pattern / signal flags", "pattern", "continuous"),
    _entry("LowMin2Lag1", "2-bar low min (1 bar ago)", "Pattern / signal flags", "pattern", "continuous"),
    _entry("EMA8CrossUp", "EMA(8) cross up", "Pattern / signal flags", "pattern", "flag"),
    _entry("EMA8CrossDown", "EMA(8) cross down", "Pattern / signal flags", "pattern", "flag"),
    _entry("RSIBuy", "RSI buy flag", "Pattern / signal flags", "signal_flag", "flag"),
    _entry("RSISell", "RSI sell flag", "Pattern / signal flags", "signal_flag", "flag"),
    _entry("SPYBull", "SPY bull regime", "Pattern / signal flags", "signal_flag", "flag"),
    # Breadth RSI
    _breadth_rsi(2, "", "market"),
    _breadth_rsi(5, "", "market"),
    _breadth_rsi(14, "", "market"),
    _breadth_rsi(2, "Risk", "risk"),
    _breadth_rsi(5, "Risk", "risk"),
    _breadth_rsi(14, "Risk", "risk"),
    _breadth_rsi(2, "Semis", "semis"),
    _breadth_rsi(5, "Semis", "semis"),
    _breadth_rsi(14, "Semis", "semis"),
    _breadth_rsi(14, "Financials", "financials"),
    _breadth_rsi(14, "Energy", "energy"),
    _breadth_rsi(14, "Utilities", "utilities"),
    _breadth_rsi(14, "Industrials", "industrials"),
    _breadth_rsi(5, "Financials", "financials"),
    _breadth_rsi(5, "Energy", "energy"),
    _breadth_rsi(5, "Utilities", "utilities"),
    _breadth_rsi(5, "Industrials", "industrials"),
    _breadth_rsi(2, "Gold", "gold"),
    _breadth_rsi(5, "Gold", "gold"),
    _breadth_rsi(14, "Gold", "gold"),
    _breadth_rsi(2, "Bond", "bond"),
    _breadth_rsi(5, "Bond", "bond"),
    _breadth_rsi(14, "Bond", "bond"),
    # Internal / non-builder columns kept for completeness
    _entry("Date", "Date", "Internal", "reference", "continuous", builder_eligible=False),
    _entry("Sell", "Sell placeholder", "Internal", "signal_flag", "flag", builder_eligible=False),
    _entry("AdjustedChange", "Adjusted change", "Internal", "volatility", "percent", builder_eligible=False),
]

_CATEGORY_ORDER = [
    "Price",
    "Volume",
    "Reference markets",
    "Breadth (raw ratios)",
    "Moving averages",
    "Momentum / oscillators",
    "Volatility / risk",
    "Efficiency / flow",
    "Spreads & composites",
    "Pattern / signal flags",
    "Breadth RSI",
    "Internal",
]


def list_indicators(*, builder_only: bool = False) -> list[IndicatorDef]:
    items = INDICATOR_CATALOG
    if builder_only:
        items = [item for item in items if item.get("builderEligible", True)]
    order = {category: index for index, category in enumerate(_CATEGORY_ORDER)}
    return sorted(
        items,
        key=lambda item: (order.get(item["category"], 999), item["label"].lower()),
    )


def builder_eligible_ids() -> list[str]:
    return [item["id"] for item in INDICATOR_CATALOG if item.get("builderEligible", True)]
