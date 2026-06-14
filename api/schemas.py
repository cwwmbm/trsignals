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
