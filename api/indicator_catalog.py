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
    "ADX14": {"min": 10, "max": 50},
    "WilliamsR14": {"min": -90, "max": -10},
    "ROC20": {"min": -10, "max": 10},
    "CMF20": {"min": -0.4, "max": 0.4},
    "BBPercentB": {"min": 0, "max": 1},
    "BBWidth": {"min": 0, "max": 0.2},
    "LinRegSlope20": {"min": -2, "max": 2},
    "OBVSlope20": {"min": -5, "max": 5},
    "VolatilityPercentile": {"min": 0, "max": 100},
}


def _range(indicator_id: str) -> TypicalRange | None:
    return _SWEEP.get(indicator_id)


_BREADTH_RSI_SOURCES: dict[str, str] = {
    "market": "RSP/SPY market breadth",
    "risk": "QQQ/SPY risk breadth",
    "semis": "SMH/SPY semis breadth",
    "financials": "XLF/SPY financials breadth",
    "energy": "XLE/SPY energy breadth",
    "utilities": "XLU/SPY utilities breadth",
    "industrials": "XLI/SPY industrials breadth",
    "gold": "GLD/SPY gold breadth",
    "bond": "TLT/SPY bond breadth",
}


def _breadth_rsi(window: int, source: str, label_source: str) -> IndicatorDef:
    column = f"RSI{window}{source}Breadth"
    breadth_label = _BREADTH_RSI_SOURCES[label_source]
    return _entry(
        column,
        f"RSI({window}) {label_source} breadth",
        "Breadth RSI",
        "breadth",
        "continuous",
        description=(
            f"RSI({window}) applied to the {breadth_label} ratio. "
            "Ranges from 0 (oversold) to 100 (overbought)."
        ),
        typical_range={"min": 10, "max": 90},
    )


INDICATOR_CATALOG: list[IndicatorDef] = [
    # Price
    _entry("Open", "Open", "Price", "price", "continuous", description="Opening price of the traded symbol for the bar."),
    _entry("High", "High", "Price", "price", "continuous", description="Highest price of the traded symbol for the bar."),
    _entry("Low", "Low", "Price", "price", "continuous", description="Lowest price of the traded symbol for the bar."),
    _entry("Close", "Close", "Price", "price", "continuous", description="Closing price of the traded symbol for the bar."),
    # Volume
    _entry("Volume", "Volume", "Volume", "volume", "continuous", description="Share volume traded for the symbol on the bar."),
    _entry("VolumeEMADiff",
        "Volume EMA diff %",
        "Volume",
        "volume",
        "percent",
        description="Percent deviation of volume from its 8-day EMA. Positive values mean above-average volume.",
    ),
    _entry("OBV", "OBV", "Volume", "volume", "continuous", description="On-Balance Volume — cumulative signed volume based on close direction."),
    _entry(
        "OBVSlope20",
        "OBV slope (20)",
        "Volume",
        "volume",
        "percent",
        typical_range=_range("OBVSlope20"),
        description="20-bar percent change in OBV — trend of accumulation vs distribution.",
    ),
    _entry(
        "CMF20",
        "Chaikin MF(20)",
        "Volume",
        "volume",
        "continuous",
        typical_range=_range("CMF20"),
        description="20-period Chaikin Money Flow — volume-weighted close position within the bar's range. Ranges roughly −1 to +1.",
    ),
    # Reference markets
    _entry("Vix", "VIX", "Reference markets", "reference", "continuous", typical_range=_range("Vix"), aliases=["VIX"], description="CBOE Volatility Index closing level — a measure of expected market volatility."),
    _entry("Spy", "SPY", "Reference markets", "reference", "continuous", aliases=["SPY"], description="SPY ETF closing price as a broad U.S. equity market reference."),
    _entry("Qqq", "QQQ", "Reference markets", "reference", "continuous", aliases=["QQQ"], description="QQQ ETF closing price as a large-cap growth / Nasdaq proxy."),
    _entry("Soxx", "SOXX", "Reference markets", "reference", "continuous", aliases=["SOXX"], description="SOXX ETF closing price as a semiconductor sector reference."),
    _entry("Iwm", "IWM", "Reference markets", "reference", "continuous", aliases=["IWM"], description="IWM ETF closing price as a small-cap U.S. equity reference."),
    _entry("Uvxy", "UVXY", "Reference markets", "reference", "continuous", aliases=["UVXY"], description="UVXY ETF closing price — a leveraged short-term VIX futures product."),
    _entry("Sqqq", "SQQQ", "Reference markets", "reference", "continuous", aliases=["SQQQ"], description="SQQQ ETF closing price — an inverse Nasdaq-100 product."),
    # Breadth (raw ratios)
    _entry("Breadth", "RSP/SPY breadth", "Breadth (raw ratios)", "breadth", "ratio", description="RSP close divided by SPY close. Measures equal-weight vs cap-weight market participation."),
    _entry("Riskbreadth", "QQQ/SPY risk breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["RiskBreadth"], description="QQQ close divided by SPY close. Tracks growth/risk-on leadership relative to the broad market."),
    _entry("Semisbreadth", "SMH/SPY semis breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["SemisBreadth"], description="SMH close divided by SPY close. Measures semiconductor sector strength vs the broad market."),
    _entry("Financialsbreadth", "XLF/SPY financials breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["FinancialsBreadth"], description="XLF close divided by SPY close. Tracks financial sector participation."),
    _entry("Energybreadth", "XLE/SPY energy breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["EnergyBreadth"], description="XLE close divided by SPY close. Tracks energy sector participation."),
    _entry("Utilitiesbreadth", "XLU/SPY utilities breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["UtilitiesBreadth"], description="XLU close divided by SPY close. Tracks utilities sector participation."),
    _entry("Industrialsbreadth", "XLI/SPY industrials breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["IndustrialsBreadth"], description="XLI close divided by SPY close. Tracks industrials sector participation."),
    _entry("Goldbreadth", "GLD/SPY gold breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["GoldBreadth"], description="GLD close divided by SPY close. Measures gold vs equity relative strength."),
    _entry("Bondbreadth", "TLT/SPY bond breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["BondBreadth"], description="TLT close divided by SPY close. Measures long-duration bonds vs equities."),
    _entry("Iwmbreadth", "IWM/SPY breadth", "Breadth (raw ratios)", "breadth", "ratio", aliases=["IWMBreadth"], description="IWM close divided by SPY close. Tracks small-cap vs large-cap participation."),
    # Moving averages
    _entry("SMA10", "SMA(10)", "Moving averages", "trend", "continuous", description="10-day simple moving average of close."),
    _entry("SMA20", "SMA(20)", "Moving averages", "trend", "continuous", description="20-day simple moving average of close."),
    _entry("SMA50", "SMA(50)", "Moving averages", "trend", "continuous", description="50-day simple moving average of close."),
    _entry("SMA100", "SMA(100)", "Moving averages", "trend", "continuous", description="100-day simple moving average of close."),
    _entry("SMA200", "SMA(200)", "Moving averages", "trend", "continuous", description="200-day simple moving average of close."),
    _entry("EMA8", "EMA(8)", "Moving averages", "trend", "continuous", description="8-day exponential moving average of close."),
    _entry("EMA20", "EMA(20)", "Moving averages", "trend", "continuous", description="20-day exponential moving average of close."),
    _entry("EMA100", "EMA(100)", "Moving averages", "trend", "continuous", description="100-day exponential moving average of close."),
    _entry("BBUpper", "Bollinger upper", "Moving averages", "trend", "continuous", description="Upper Bollinger band (default 20-period, 2 std dev) on close."),
    _entry("BBMiddle", "Bollinger middle", "Moving averages", "trend", "continuous", description="Middle Bollinger band — the 20-day SMA of close."),
    _entry("BBLower", "Bollinger lower", "Moving averages", "trend", "continuous", description="Lower Bollinger band (default 20-period, 2 std dev) on close."),
    _entry("DonchianUpper20", "Donchian upper (20)", "Moving averages", "trend", "continuous", description="Prior 20-day highest high (shifted one bar) — standard Donchian breakout upper level."),
    _entry("DonchianLower20", "Donchian lower (20)", "Moving averages", "trend", "continuous", description="Prior 20-day lowest low (shifted one bar) — standard Donchian breakout lower level."),
    _entry("DonchianUpper55", "Donchian upper (55)", "Moving averages", "trend", "continuous", description="Prior 55-day highest high (shifted one bar) — standard Donchian breakout upper level."),
    _entry("DonchianLower55", "Donchian lower (55)", "Moving averages", "trend", "continuous", description="Prior 55-day lowest low (shifted one bar) — standard Donchian breakout lower level."),
    _entry("KCUpper20", "Keltner upper (20)", "Moving averages", "trend", "continuous", description="20-day Keltner upper band — EMA(20) plus 1.5× ATR(20) (TTM Squeeze standard)."),
    _entry("KCMiddle20", "Keltner middle (20)", "Moving averages", "trend", "continuous", description="20-day Keltner middle line — EMA(20) of close."),
    _entry("KCLower20", "Keltner lower (20)", "Moving averages", "trend", "continuous", description="20-day Keltner lower band — EMA(20) minus 1.5× ATR(20) (TTM Squeeze standard)."),
    _entry("PSAR", "Parabolic SAR", "Moving averages", "trend", "continuous", description="Parabolic SAR trailing stop/reversal level in price units."),
    # Momentum / oscillators
    _entry("RSI2", "RSI(2)", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("RSI2"), description="2-period Relative Strength Index on close. Ranges 0–100."),
    _entry("RSI5", "RSI(5)", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("RSI5"), description="5-period Relative Strength Index on close. Ranges 0–100."),
    _entry("RSI14", "RSI(14)", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("RSI14"), description="14-period Relative Strength Index on close. Ranges 0–100."),
    _entry("Stoch", "Stochastic", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("Stoch"), description="14-period stochastic %K with 3-bar smoothing. Ranges 0–100."),
    _entry("StochOscilator", "Stochastic oscillator", "Momentum / oscillators", "momentum", "continuous", description="Difference between stochastic %K and its 3-bar SMA — a fast/slow stochastic spread."),
    _entry("CCI", "CCI(20)", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("CCI"), description="20-period Commodity Channel Index. Positive values indicate price above its recent average."),
    _entry("MACDHist", "MACD histogram", "Momentum / oscillators", "momentum", "continuous", description="MACD histogram (12/26/9) — the difference between the MACD line and its signal line."),
    _entry("ValueCharts", "ValueCharts(5)", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("ValueCharts"), description="5-period ValueCharts oscillator measuring close deviation from a smoothed mid-range."),
    _entry("Hurst", "Hurst exponent", "Momentum / oscillators", "momentum", "continuous", description="Rolling 100-bar Hurst exponent on close. Above 0.5 suggests trending; below 0.5 suggests mean-reverting."),
    _entry("ADX14", "ADX(14)", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("ADX14"), description="14-period Average Directional Index — trend strength from 0 (no trend) to 100 (strong trend)."),
    _entry("WilliamsR14", "Williams %R(14)", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("WilliamsR14"), description="14-period Williams %R — oversold/overbought oscillator ranging from −100 (oversold) to 0 (overbought)."),
    _entry("ROC20", "ROC(20)", "Momentum / oscillators", "momentum", "continuous", typical_range=_range("ROC20"), description="20-bar rate of change of close — plain momentum distinct from leveraged daily % change."),
    _entry("TRIX", "TRIX(15)", "Momentum / oscillators", "momentum", "continuous", description="15-period TRIX — triple-smoothed EMA rate of change; less noisy than MACD histogram alone."),
    _entry(
        "LinRegSlope20",
        "Lin reg slope (20)",
        "Momentum / oscillators",
        "momentum",
        "continuous",
        typical_range=_range("LinRegSlope20"),
        description="20-bar linear regression slope of close, normalized as percent of price per bar.",
    ),
    # Volatility / risk
    _entry("%Change", "% change", "Volatility / risk", "volatility", "percent", typical_range=_range("%Change"), description="Leveraged daily percent change in close (leverage factor from config)."),
    _entry("ATR20", "ATR(20)", "Volatility / risk", "volatility", "continuous", description="20-day Average True Range — average daily price range in price units."),
    _entry("ATR50", "ATR(50)", "Volatility / risk", "volatility", "continuous", description="50-day Average True Range — average daily price range in price units."),
    _entry("Volatility", "Realized volatility", "Volatility / risk", "volatility", "continuous", description="Annualized realized volatility from a 5-day rolling std of daily returns."),
    _entry(
        "VolatilityPercentile",
        "Vol realized vol percentile",
        "Volatility / risk",
        "volatility",
        "continuous",
        typical_range=_range("VolatilityPercentile"),
        description="Percentile rank (0–100) of current realized volatility vs the trailing 252 trading days.",
    ),
    _entry("BBWidth", "Bollinger width", "Volatility / risk", "volatility", "continuous", typical_range=_range("BBWidth"), description="Bollinger band width as a fraction of the middle band — low values indicate compression."),
    _entry("BBPercentB", "Bollinger %B", "Volatility / risk", "volatility", "continuous", typical_range=_range("BBPercentB"), description="Position of close within the Bollinger bands: 0 = lower band, 1 = upper band, 0.5 = middle."),
    _entry("ChangeVelocity", "Change velocity", "Volatility / risk", "volatility", "continuous", typical_range=_range("ChangeVelocity"), description="One-day price change divided by the prior bar's ATR(20). Normalizes momentum by recent volatility."),
    # Efficiency / flow
    _entry("ER", "Efficiency ratio", "Efficiency / flow", "momentum", "continuous", typical_range=_range("ER"), description="Kaufman Efficiency Ratio over 10 bars. Near 1 means directional; near 0 means choppy."),
    _entry("IBR", "IBR", "Efficiency / flow", "momentum", "continuous", typical_range=_range("IBR"), description="Internal Bar Range — where close sits within the day's high-low range (0 = at low, 1 = at high)."),
    _entry("IBR2", "IBR(2)", "Efficiency / flow", "momentum", "continuous", typical_range=_range("IBR2"), description="2-day rolling average of Internal Bar Range."),
    _entry("IBR3", "IBR(3)", "Efficiency / flow", "momentum", "continuous", typical_range=_range("IBR3"), description="3-day rolling average of Internal Bar Range."),
    _entry("VFI10", "VFI(10)", "Efficiency / flow", "momentum", "continuous", typical_range=_range("VFI10"), description="10-period Volume Flow Indicator — smoothed signed volume relative to its moving average."),
    _entry("VFI20", "VFI(20)", "Efficiency / flow", "momentum", "continuous", typical_range=_range("VFI20"), description="20-period Volume Flow Indicator — smoothed signed volume relative to its moving average."),
    _entry("VFI40", "VFI(40)", "Efficiency / flow", "momentum", "continuous", typical_range=_range("VFI40"), description="40-period Volume Flow Indicator — smoothed signed volume relative to its moving average."),
    _entry("VFI80", "VFI(80)", "Efficiency / flow", "momentum", "continuous", typical_range=_range("VFI80"), description="80-period Volume Flow Indicator — smoothed signed volume relative to its moving average."),
    # Spreads & composites
    _entry("Close_EMA8", "Close vs EMA(8) %", "Spreads & composites", "composite", "percent", typical_range=_range("Close_EMA8"), description="Percent distance of close from EMA(8): (close − EMA8) / close × 100."),
    _entry("Close_SMA20", "Close vs SMA(20) %", "Spreads & composites", "composite", "percent", description="Percent distance of close from SMA(20): (close − SMA20) / close × 100."),
    _entry("Close_SMA50", "Close vs SMA(50) %", "Spreads & composites", "composite", "percent", description="Percent distance of close from SMA(50): (close − SMA50) / close × 100."),
    _entry("Close_SMA200", "Close vs SMA(200) %", "Spreads & composites", "composite", "percent", description="Percent distance of close from SMA(200): (close − SMA200) / close × 100."),
    _entry("EMA20_EMA100", "EMA(20) vs EMA(100) %", "Spreads & composites", "composite", "percent", description="Percent spread between EMA(20) and EMA(100): (EMA20 − EMA100) / EMA20 × 100."),
    _entry("SMA20_SMA50", "SMA(20) vs SMA(50) %", "Spreads & composites", "composite", "percent", description="Percent spread between SMA(20) and SMA(50): (SMA20 − SMA50) / SMA20 × 100."),
    _entry("SMA50_SMA200", "SMA(50) vs SMA(200) %", "Spreads & composites", "composite", "percent", description="Percent spread between SMA(50) and SMA(200): (SMA50 − SMA200) / SMA50 × 100."),
    _entry("ATR20_ATR50", "ATR(20) - ATR(50)", "Spreads & composites", "composite", "continuous", description="Difference between ATR(20) and ATR(50). Positive means short-term volatility exceeds longer-term."),
    # Pattern / signal flags
    _entry("HigherCloses2", "Higher closes (2)", "Pattern / signal flags", "pattern", "flag", description="Flag (1) when close is higher than each of the prior two closes; otherwise −1."),
    _entry("HigherCloses3", "Higher closes (3)", "Pattern / signal flags", "pattern", "flag", description="Flag (1) when close is higher than each of the prior three closes; otherwise −1."),
    _entry("LowerCloses2", "Lower closes (2)", "Pattern / signal flags", "pattern", "flag", description="Flag (1) when close is lower than each of the prior two closes; otherwise −1."),
    _entry("LowerCloses3", "Lower closes (3)", "Pattern / signal flags", "pattern", "flag", description="Flag (1) when close is lower than each of the prior three closes; otherwise −1."),
    _entry("HighestClose2", "Highest close (2)", "Pattern / signal flags", "pattern", "flag", description="Flag (1) when close is at or above the prior bar's close; otherwise −1."),
    _entry("HighestClose3", "Highest close (3)", "Pattern / signal flags", "pattern", "flag", description="Flag (1) when close is at or above the 3-bar rolling maximum; otherwise −1."),
    _entry("LowestClose2", "Lowest close (2)", "Pattern / signal flags", "pattern", "flag", description="Flag (1) when close is at or below the prior bar's close; otherwise −1."),
    _entry("LowestClose3", "Lowest close (3)", "Pattern / signal flags", "pattern", "flag", description="Flag (1) when close is at or below the 3-bar rolling minimum; otherwise −1."),
    _entry("CloseLag1", "Close (1 bar ago)", "Pattern / signal flags", "pattern", "continuous", description="Close price from one bar ago."),
    _entry("CloseLag2", "Close (2 bars ago)", "Pattern / signal flags", "pattern", "continuous", description="Close price from two bars ago."),
    _entry("CloseLag3", "Close (3 bars ago)", "Pattern / signal flags", "pattern", "continuous", description="Close price from three bars ago."),
    _entry("LowLag1", "Low (1 bar ago)", "Pattern / signal flags", "pattern", "continuous", description="Low price from one bar ago."),
    _entry("LowMin2Lag1", "2-bar low min (1 bar ago)", "Pattern / signal flags", "pattern", "continuous", description="Minimum low over the prior two bars, shifted back one bar."),
    _entry("EMA8CrossUp", "EMA(8) cross up", "Pattern / signal flags", "pattern", "flag", description="Flag (1) when close crosses above EMA(8) from below; otherwise −1."),
    _entry("EMA8CrossDown", "EMA(8) cross down", "Pattern / signal flags", "pattern", "flag", description="Flag (1) when close crosses below EMA(8) from above; otherwise −1."),
    _entry("RSIBuy", "RSI buy flag", "Pattern / signal flags", "signal_flag", "flag", description="Flag (1) when RSI(2) ≤ 15 and RSI(5) ≤ 35 (oversold buy zone); otherwise −1."),
    _entry("RSISell", "RSI sell flag", "Pattern / signal flags", "signal_flag", "flag", description="Flag (1) when RSI(2) ≥ 95 and RSI(5) ≥ 70 (overbought sell zone); otherwise −1."),
    _entry("SPYBull", "SPY bull regime", "Pattern / signal flags", "signal_flag", "flag", description="Flag (1) when SPY SMA(50) > SMA(200) (golden cross regime); otherwise −1."),
    _entry("BBSqueeze", "BB/Keltner squeeze", "Pattern / signal flags", "pattern", "flag", description="TTM Squeeze flag (1) when Bollinger bands (20, 2σ) are inside Keltner channels (20 EMA, 1.5× ATR(20)); otherwise −1."),
    _entry(
        "CloseAboveDonchianUpper20",
        "Close above Donchian upper (20)",
        "Pattern / signal flags",
        "pattern",
        "flag",
        description="Flag (1) when close breaks above the prior 20-day Donchian upper; equivalent to Close > DonchianUpper20.",
    ),
    _entry(
        "CloseBelowDonchianLower20",
        "Close below Donchian lower (20)",
        "Pattern / signal flags",
        "pattern",
        "flag",
        description="Flag (1) when close breaks below the prior 20-day Donchian lower; equivalent to Close < DonchianLower20.",
    ),
    _entry(
        "CloseAboveDonchianUpper55",
        "Close above Donchian upper (55)",
        "Pattern / signal flags",
        "pattern",
        "flag",
        description="Flag (1) when close breaks above the prior 55-day Donchian upper; equivalent to Close > DonchianUpper55.",
    ),
    _entry(
        "CloseBelowDonchianLower55",
        "Close below Donchian lower (55)",
        "Pattern / signal flags",
        "pattern",
        "flag",
        description="Flag (1) when close breaks below the prior 55-day Donchian lower; equivalent to Close < DonchianLower55.",
    ),
    _entry(
        "CloseAbovePSAR",
        "Close above PSAR",
        "Pattern / signal flags",
        "pattern",
        "flag",
        description="Flag (1) when close > Parabolic SAR (bullish trend); equivalent to Close > PSAR.",
    ),
    _entry(
        "CloseAboveKCUpper20",
        "Close above Keltner upper (20)",
        "Pattern / signal flags",
        "pattern",
        "flag",
        description="Flag (1) when close > 20-day Keltner upper band; equivalent to Close > KCUpper20.",
    ),
    _entry(
        "CloseBelowKCLower20",
        "Close below Keltner lower (20)",
        "Pattern / signal flags",
        "pattern",
        "flag",
        description="Flag (1) when close < 20-day Keltner lower band; equivalent to Close < KCLower20.",
    ),
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
