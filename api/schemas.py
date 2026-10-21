from typing import Literal

from pydantic import BaseModel, Field

from api.allocation import DEFAULT_TIER_SINGLE_PCT

ScanLane = Literal["active", "testing", "archived"]


class RuntimeOptions(BaseModel):
    monday_buy: bool | None = None
    low_volume_buy: bool | None = None
    hold_on_buy_signal: bool | None = None


class RegimeRequestOptions(BaseModel):
    """Ask a backtest for market regime figures instead of its normal result."""

    regime_mode: Literal["overview", "analyze"] | None = None
    regime_id: str | None = None


class RegimeAnalyzeRequest(BaseModel):
    regime_id: str
    years: int = Field(default=25, ge=1, le=100)


class SingleSignalExpression(BaseModel):
    kind: Literal["single"] = "single"
    name: str


class CombinedSignalExpression(BaseModel):
    kind: Literal["combined"] = "combined"
    primary: str
    secondary: str
    mode: Literal["and", "or"] = "and"


SignalExpression = SingleSignalExpression | CombinedSignalExpression


class SingleBacktestRequest(RegimeRequestOptions):
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


class SymbolConfirmDetailRequest(RegimeRequestOptions):
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


class MonteCarloRequest(BaseModel):
    trade_returns: list[float]
    method: Literal["shuffle", "bootstrap"] = "shuffle"
    n_sims: int = Field(default=1000, ge=1, le=10_000)
    start_capital: float = Field(default=15000.0, gt=0)


class BuilderCondition(BaseModel):
    left: str
    operator: Literal["<", "<=", ">", ">=", "=", "crosses above", "crosses below", "is true", "is false"]
    right: str = ""
    logic: Literal["AND", "OR"] = "AND"


class BuilderBacktestRequest(RegimeRequestOptions):
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
    custom_dataset_id: str | None = None
    rth_entries_only: bool = True
    eod_exit: bool = True
    backtest_all_data: bool = False
    hold_on_buy_signal: bool = False


class CustomDatasetResponse(BaseModel):
    id: str
    symbol: str
    interval_minutes: int
    interval_label: str
    periods_per_year: int
    start: str
    end: str
    row_count: int
    unavailable_indicator_ids: list[str]
    has_vwap: bool = False
    custom_data_only_indicator_ids: list[str] = []
    timezone: str = "UTC"
    is_intraday: bool = True


BuilderRefineMode = Literal[
    "signal-combo-sweep",
    "symbol-confirm-sweep",
    "hold-days-sweep",
    "indicator-sweep",
    "indicator-combo-sweep",
    "basket-sweep",
    "market-regime-sweep",
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
    # indicator-combo-sweep: filters per combination, rows returned, and trio beam width.
    combo_size: int = Field(default=2, ge=2, le=3)
    combo_top_n: int = Field(default=50, ge=1, le=200)
    combo_beam_width: int = Field(default=150, ge=10, le=1000)
    # basket-sweep: symbols to run the draft strategy on (comma or space separated entries are fine).
    basket_symbols: list[str] = Field(default_factory=list)


RefineSampleMode = Literal["in_sample", "full"]


class BuilderRefineOutcomeRequest(BaseModel):
    mode: BuilderRefineMode
    strategy: BuilderBacktestRequest
    row: dict
    sample: RefineSampleMode = "full"
    primary_symbol: str | None = None
    symbol_pool: list[str] | None = None
    max_days: int = Field(default=7, ge=1, le=100)
    is_sell: bool = False


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
    hold_on_buy_signal: bool = False
    rth_entries_only: bool = False
    eod_exit: bool = False
    scan_lane: ScanLane = "testing"
    scan_sort_order: int = 0
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
    hold_on_buy_signal: bool = False
    rth_entries_only: bool = False
    eod_exit: bool = False


class UpdateStrategyRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    symbol: str | None = None
    direction: Literal["long", "short"] | None = None
    hold_days: int | None = Field(default=None, ge=1, le=100)
    profit: int | None = Field(default=None, ge=0, le=100)
    description: str | None = None
    conditions: list[BuilderCondition] | None = None
    sell_conditions: list[BuilderCondition] | None = None
    confirm_symbols: list[str] | None = None
    proxy_symbol: str | None = None
    hold_on_buy_signal: bool | None = None
    rth_entries_only: bool | None = None
    eod_exit: bool | None = None
    scan_lane: ScanLane | None = None
    scan_sort_order: int | None = None


# Portfolios always hold until every strategy exits. The field is still accepted (and stored
# portfolios keep theirs) so older clients and files load, but it is ignored.
PortfolioOverlapMode = Literal["first_signal_only", "hold_until_all_exit"]
PortfolioAllocationModel = Literal["all_in", "proportional", "regime_adjusted", "tiered"]
PortfolioProportionalMode = Literal["linear", "early_conviction", "confirmation_required"]
RegimeScoreKind = Literal["robustness", "portfolio"]
RegimePenaltyPreset = Literal["harsh", "strict", "medium", "low"]


class RegimePenaltyFields(BaseModel):
    """Regime-adjusted allocation settings, used only when allocation_model is regime_adjusted.

    ``regime_penalty`` picks how hard a poorly-scoring regime state is punished (see
    ``api.regime_allocation.PENALTY_PRESETS``); ``regime_score`` picks the score that is read.
    """

    regime_score: RegimeScoreKind = "robustness"
    regime_penalty: RegimePenaltyPreset = "medium"


class PortfolioSimulateRequest(RegimePenaltyFields, RegimeRequestOptions):
    strategy_ids: list[str] = Field(default_factory=list)
    portfolio_ids: list[str] = Field(default_factory=list)
    overlap_mode: PortfolioOverlapMode = "hold_until_all_exit"
    proxy_symbol: str | None = None
    years: int = Field(default=25, ge=1, le=100)
    # All-in is the original behavior. Proportional sizes the book by active/total strategies
    # and ignores overlap_mode.
    allocation_model: PortfolioAllocationModel = "all_in"
    proportional_mode: PortfolioProportionalMode = "linear"
    # Tiered model only: weight in percent when exactly one strategy is active (two or more is 100%).
    tier_single_pct: float = Field(default=DEFAULT_TIER_SINGLE_PCT, ge=0, le=100)
    transaction_cost_bps: float = Field(default=0.0, ge=0, le=1000)


class PortfolioShapleyRequest(PortfolioSimulateRequest):
    samples: int = Field(default=64, ge=1, le=512)
    seed: int | None = None


class PortfolioRegimeRequest(PortfolioSimulateRequest):
    """On-demand regime contribution analysis (same inputs as portfolio simulate)."""
    pass


class PortfolioAllocationComparisonRequest(PortfolioSimulateRequest):
    """On-demand All-In vs proportional comparison (same inputs as portfolio simulate)."""
    pass


class SavedPortfolio(RegimePenaltyFields):
    id: str
    name: str
    description: str = ""
    strategy_ids: list[str]
    portfolio_ids: list[str] = Field(default_factory=list)
    overlap_mode: PortfolioOverlapMode = "hold_until_all_exit"
    proxy_symbol: str | None = None
    allocation_model: PortfolioAllocationModel = "all_in"
    proportional_mode: PortfolioProportionalMode = "linear"
    # Tiered model only: weight in percent when exactly one strategy is active (two or more is 100%).
    tier_single_pct: float = Field(default=DEFAULT_TIER_SINGLE_PCT, ge=0, le=100)
    transaction_cost_bps: float = Field(default=0.0, ge=0, le=1000)
    scan_lane: ScanLane = "testing"
    scan_sort_order: int = 0
    created_at: str
    updated_at: str


class SavePortfolioRequest(RegimePenaltyFields):
    name: str = Field(min_length=1)
    description: str = ""
    strategy_ids: list[str] = Field(default_factory=list)
    portfolio_ids: list[str] = Field(default_factory=list)
    overlap_mode: PortfolioOverlapMode = "hold_until_all_exit"
    proxy_symbol: str | None = None
    allocation_model: PortfolioAllocationModel = "all_in"
    proportional_mode: PortfolioProportionalMode = "linear"
    # Tiered model only: weight in percent when exactly one strategy is active (two or more is 100%).
    tier_single_pct: float = Field(default=DEFAULT_TIER_SINGLE_PCT, ge=0, le=100)
    transaction_cost_bps: float = Field(default=0.0, ge=0, le=1000)


class UpdatePortfolioRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    description: str | None = None
    scan_lane: ScanLane | None = None
    scan_sort_order: int | None = None
    strategy_ids: list[str] | None = None
    portfolio_ids: list[str] | None = None
    overlap_mode: PortfolioOverlapMode | None = None
    proxy_symbol: str | None = None
    allocation_model: PortfolioAllocationModel | None = None
    proportional_mode: PortfolioProportionalMode | None = None
    tier_single_pct: float | None = Field(default=None, ge=0, le=100)
    transaction_cost_bps: float | None = Field(default=None, ge=0, le=1000)
    regime_score: RegimeScoreKind | None = None
    regime_penalty: RegimePenaltyPreset | None = None


class SymbolLeverageModel(BaseModel):
    """One symbol and the leveraged product traded in its place (see api.leverage_settings)."""

    symbol: str
    leverage_symbol: str | None = None  # blank or None: no product, the symbol is traded itself
    leverage: float
    beta: float | None = None
    annual_fee: float = 0.0
    financing_multiple: float = 0.0


class LeverageSettingsModel(BaseModel):
    general_leverage: float
    symbols: list[SymbolLeverageModel] = Field(default_factory=list)


class LeverageEstimateRequest(BaseModel):
    symbol: str
    leverage_symbol: str


class ConditionSnapshotItem(BaseModel):
    label: str
    logic: Literal["AND", "OR"] | None = None
    passed: bool
    left_value: str | None = None
    right_value: str | None = None
    operator: str
    left: str
    right: str


class ScanOrder(BaseModel):
    instrument: str
    underlying: str
    kind: Literal["vehicle", "synthetic"]
    leverage: float | None = None
    side: Literal["buy", "sell"]
    event: str
    weight_before: float
    weight_after: float
    price: float | None = None


class ScanHolding(BaseModel):
    instrument: str
    underlying: str
    kind: Literal["vehicle", "synthetic"]
    leverage: float | None = None
    weight: float
    price: float | None = None


class ScanPosition(BaseModel):
    """What to do with a row today, as shares of that row's own book (1.0 is all of it)."""

    action: Literal["Enter", "Add", "Reduce", "Exit", "Rebalance", "Hold", "Flat"]
    held_before: float
    held_now: float
    target: float
    delta: float
    active: int
    total: int
    active_strategies: list[str] = Field(default_factory=list)
    entered: list[str] = Field(default_factory=list)
    exited: list[str] = Field(default_factory=list)
    reason: str
    orders: list[ScanOrder] = Field(default_factory=list)
    holdings: list[ScanHolding] = Field(default_factory=list)
    model: str
    as_of: str | None = None
    approximate: str | None = None


class ScanRowResponse(BaseModel):
    id: str
    source: Literal["legacy", "builder", "portfolio"]
    strategy_id: str | None = None
    portfolio_id: str | None = None
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
    condition_snapshot: list[ConditionSnapshotItem] | None = None
    condition_passed_count: int | None = None
    condition_total_count: int | None = None
    sell_condition_snapshot: list[ConditionSnapshotItem] | None = None
    sell_condition_passed_count: int | None = None
    sell_condition_total_count: int | None = None
    condition_as_of: str | None = None
    position: ScanPosition | None = None


class QuoteSymbolResponse(BaseModel):
    symbol: str
    as_of: str | None = None
    close: float | None = None
    pct_change: float | None = None
    ibr: float | None = None
    rsi2: float | None = None
    rsi5: float | None = None
    stoch: float | None = None
    missing_days: list[str] = Field(default_factory=list)


class QuoteResponse(BaseModel):
    as_of: str | None = None
    quotes: list[QuoteSymbolResponse] = Field(default_factory=list)


class QuoteSymbolRequest(BaseModel):
    symbol: str
