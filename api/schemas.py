from typing import Literal

from pydantic import BaseModel, Field


class RuntimeOptions(BaseModel):
    monday_buy: bool | None = None
    low_volume_buy: bool | None = None


class SingleSignalExpression(BaseModel):
    kind: Literal["single"] = "single"
    name: str


class CombinedSignalExpression(BaseModel):
    kind: Literal["combined"] = "combined"
    primary: str
    secondary: str
    mode: Literal["and", "or"] = "and"


SignalExpression = SingleSignalExpression | CombinedSignalExpression


class SingleBacktestRequest(BaseModel):
    symbol: str = "SOXX"
    years: int = Field(default=25, ge=1, le=100)
    signal: SignalExpression
    runtime_options: RuntimeOptions = Field(default_factory=RuntimeOptions)


class SignalComboSweepRequest(BaseModel):
    symbol: str = "SOXX"
    years: int = Field(default=25, ge=1, le=100)
    signal_a: str
    signal_b: str
    runtime_options: RuntimeOptions = Field(default_factory=RuntimeOptions)


class SymbolConfirmSweepRequest(BaseModel):
    primary_symbol: str = "SOXX"
    symbol_pool: list[str] = Field(default_factory=lambda: ["SOXX", "SMH", "QQQ"])
    years: int = Field(default=25, ge=1, le=100)
    signal: SignalExpression
    runtime_options: RuntimeOptions = Field(default_factory=RuntimeOptions)


class SymbolConfirmDetailRequest(BaseModel):
    primary_symbol: str = "SOXX"
    confirm_symbols: list[str] = Field(default_factory=list)
    years: int = Field(default=25, ge=1, le=100)
    signal: SignalExpression
    runtime_options: RuntimeOptions = Field(default_factory=RuntimeOptions)


class HoldDaysSweepRequest(BaseModel):
    symbol: str = "SOXX"
    years: int = Field(default=25, ge=1, le=100)
    max_days: int = Field(default=7, ge=1, le=100)
    signal: SignalExpression
    runtime_options: RuntimeOptions = Field(default_factory=RuntimeOptions)


class IndicatorSweepRequest(BaseModel):
    symbol: str = "SOXX"
    years: int = Field(default=25, ge=1, le=100)
    signal: SignalExpression
    is_sell: bool = False
    check_breadth: bool = False
    check_both: bool = False
    runtime_options: RuntimeOptions = Field(default_factory=RuntimeOptions)


class BuilderCondition(BaseModel):
    left: str
    operator: Literal["<", "<=", ">", ">=", "=", "crosses above", "crosses below", "is true", "is false"]
    right: str = ""
    logic: Literal["AND", "OR"] = "AND"


class BuilderBacktestRequest(BaseModel):
    symbol: str = "SPY"
    years: int = Field(default=25, ge=1, le=100)
    direction: Literal["long", "short"] = "long"
    hold_days: int = Field(default=2, ge=1, le=100)
    profit: int = Field(default=1, ge=0, le=100)
    name: str = ""
    description: str = ""
    conditions: list[BuilderCondition] = Field(min_items=1)
    sell_conditions: list[BuilderCondition] = Field(default_factory=list)
    confirm_symbols: list[str] = Field(default_factory=list)
    proxy_symbol: str | None = None


BuilderRefineMode = Literal[
    "signal-combo-sweep",
    "symbol-confirm-sweep",
    "hold-days-sweep",
    "indicator-sweep",
]


class BuilderRefineRequest(BaseModel):
    mode: BuilderRefineMode
    strategy: BuilderBacktestRequest
    secondary_strategy_id: str | None = None
    primary_symbol: str | None = None
    symbol_pool: list[str] | None = None
    max_days: int = Field(default=7, ge=1, le=100)
    is_sell: bool = False
    check_breadth: bool = False
    check_both: bool = False


class SavedStrategy(BaseModel):
    id: str
    name: str
    symbol: str
    direction: Literal["long", "short"]
    hold_days: int
    profit: int
    description: str
    conditions: list[BuilderCondition]
    sell_conditions: list[BuilderCondition] = Field(default_factory=list)
    confirm_symbols: list[str] = Field(default_factory=list)
    proxy_symbol: str | None = None
    legacy_signal: str | None = None
    created_at: str
    updated_at: str


class SaveStrategyRequest(BaseModel):
    name: str = Field(min_length=1)
    symbol: str
    direction: Literal["long", "short"] = "long"
    hold_days: int = Field(default=2, ge=1, le=100)
    profit: int = Field(default=1, ge=0, le=100)
    description: str = ""
    conditions: list[BuilderCondition] = Field(min_items=1)
    sell_conditions: list[BuilderCondition] = Field(default_factory=list)
    confirm_symbols: list[str] = Field(default_factory=list)
    proxy_symbol: str | None = None


class UpdateStrategyRequest(BaseModel):
    description: str = ""


class ScanRowResponse(BaseModel):
    id: str
    source: Literal["legacy", "builder"]
    strategy_id: str | None = None
    symbol: str
    signal: str
    buy_signal: bool
    hold_long: bool
    sell_signal: bool
    days: int
    profit: int
    trade_pnl: float
    kelly: float | None
    description: str
