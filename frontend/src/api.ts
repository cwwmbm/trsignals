const API_URL = import.meta.env.VITE_API_URL ?? "";

export type SignalExpression =
  | { kind: "single"; name: string }
  | { kind: "combined"; primary: string; secondary: string; mode: "and" | "or" };

export type RunMode =
  | "single"
  | "signal-combo-sweep"
  | "symbol-confirm-sweep"
  | "symbol-confirm-detail"
  | "hold-days-sweep"
  | "indicator-sweep";

export type { BuilderRefineMode } from "@/lib/builder-refine-config";

export interface SignalInfo {
  name: string;
  label: string;
}

export type IndicatorKind =
  | "price"
  | "volume"
  | "reference"
  | "breadth"
  | "momentum"
  | "trend"
  | "volatility"
  | "pattern"
  | "composite"
  | "signal_flag";

export type IndicatorValueType = "continuous" | "percent" | "ratio" | "flag";

export type CompareMode = "none" | "number" | "indicator" | "both";

export interface IndicatorInfo {
  id: string;
  label: string;
  kind: IndicatorKind;
  valueType: IndicatorValueType;
  category: string;
  description?: string;
  typicalRange?: { min: number; max: number };
  builderEligible: boolean;
  customDataOnly?: boolean;
  aliases?: string[];
  available?: boolean;
  compareMode?: CompareMode;
  compareIndicatorKinds?: IndicatorKind[];
  defaultCompareIndicator?: string;
  defaultCompareNumber?: string;
}

export type ContributionMetrics = {
  cagr_percent: number | null;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown: number | null;
  rolling_pnl: number | null;
  time_in_market_percent: number | null;
  trades: number;
  worst_calendar_year_pnl_percent: number | null;
  avg_days_in_trade: number | null;
  median_days_in_trade: number | null;
  calmar: number | null;
  ulcer_index: number | null;
  time_under_water_percent: number | null;
  trades_per_year: number | null;
  utility: number | null;
  exposure_adjusted_return: number | null;
};

export type StrategyContribution = {
  strategy_id: string;
  strategy_name: string;
  full: ContributionMetrics;
  without_strategy: ContributionMetrics;
  cagr_contribution_pp: number | null;
  sharpe_delta: number | null;
  sortino_delta: number | null;
  calmar_delta: number | null;
  ulcer_index_delta: number | null;
  time_under_water_delta_pp: number | null;
  max_drawdown_effect_pp: number | null;
  final_equity_delta: number | null;
  added_exposure_pp: number | null;
  added_portfolio_trades: number | null;
  worst_calendar_year_pnl_delta_pp: number | null;
  avg_days_in_trade_delta: number | null;
  median_days_in_trade_delta: number | null;
  marginal_utility: number | null;
  exposure_adjusted_return_delta: number | null;
  marginal_cagr_per_10pp_exposure: number | null;
  candidate_holding_days: number;
  overlapping_holding_days: number;
  unique_holding_days: number;
  unique_holding_percent: number | null;
  redundant_holding_percent: number | null;
};

export type PositionEventKind = "Enter" | "Add" | "Reduce" | "Exit";

export type PositionSignalChange = {
  strategy_id: string;
  strategy_name: string;
  change: "entered" | "exited";
};

/** One executed transaction, as the portfolio simulator's ledger recorded it. */
export type PositionEvent = {
  date: string;
  event: PositionEventKind;
  symbol: string;
  signal_changes: PositionSignalChange[];
  active_strategies: string[];
  active_signals: number;
  eligible_signals: number;
  exposure_before: number;
  exposure_after: number;
  /** The underlying's close on the trade date: the price to show. */
  underlying_close: number | null;
  position_value_after: number;
  unrealized_pnl_after: number;
  realized_pnl: number;
  /** Episode P&L so far, as of this event, net of fees. */
  episode_pnl_to_date: number;
  episode_return_to_date: number | null;
  fees: number;
  cash_before: number;
  cash_after: number;
  equity_before: number;
  equity_after: number;
};

/** A continuous period of nonzero exposure to one instrument. */
/** What a position held: the leveraged stand-in for a symbol, or its real leverage symbol. */
export type EpisodeInstrument = {
  symbol: string;
  underlying: string;
  kind: "synthetic" | "vehicle";
  leverage: number | null;
  /** The underlying has a leverage-settings entry. */
  configured: boolean;
};

export type PositionEpisode = {
  id: string;
  symbol: string;
  instrument?: EpisodeInstrument | null;
  status: "Open" | "Closed";
  start: string;
  end: string | null;
  days: number;
  max_exposure: number;
  avg_exposure: number;
  pnl: number;
  return: number | null;
  equity_before_entry: number;
  rebalance_count: number;
  realized_pnl: number;
  unrealized_pnl: number;
  fees: number;
  /** The underlying's close on the days of first entry and final exit. */
  initial_underlying_close: number | null;
  final_underlying_close: number | null;
  events: PositionEvent[];
};

export type PositionsPayload = {
  episodes: PositionEpisode[];
  reconciliation: {
    episode_pnl_total: number;
    equity_change: number;
    episode_vs_equity_difference: number;
    cash_difference: number;
    episode_count: number;
    transaction_count: number;
    ledger_vs_simulation_max_relative_difference?: number;
  };
  signal_changes_without_transaction: number;
  price_basis: { leverage: number; description: string };
};

/** Present when the backtest traded a symbol with a leverage-settings entry. */
export type TradedVehicle = {
  symbol: string;
  /** Null when the symbol is traded itself at its leverage: there is no separate product. */
  leverage_symbol: string | null;
  leverage: number;
  first_date: string | null;
  available: boolean;
  trades_on_leverage_symbol: number;
  trades_on_stand_in: number;
};

/** How likely the worst drawdown of a backtest-length path is to pass each depth (all fractions). */
export type DrawdownRisk = {
  method: "block_bootstrap";
  paths: number;
  block_days: number;
  /** The block length in bars; more than block_days when the bars are intraday. */
  block_bars?: number;
  horizon_bars: number;
  /** Bars per year of the history, to turn horizon_bars into years. */
  periods_per_year?: number;
  seed: number;
  historical_max_drawdown: number;
  median_max_drawdown: number;
  p95_max_drawdown: number;
  thresholds: Array<{ drawdown: number; probability: number; std_error: number }>;
};

export type BenchmarkMetrics = {
  total_return: number | null;
  /** CAGR, Sharpe and Sortino leave out the series' own best year, like the result cards. */
  cagr_percent: number | null;
  sharpe: number | null;
  sortino: number | null;
  /** Fraction, over every year. */
  max_drawdown: number | null;
  full_cagr_percent: number | null;
  full_sharpe: number | null;
  full_sortino: number | null;
  excluded_year: number | null;
};

export type BenchmarkSeries = {
  id: "underlying" | "vehicle";
  label: string;
  symbol: string;
  leverage: number;
  metrics: BenchmarkMetrics;
};

/** Buy-and-hold of the traded symbol, over the same bars as the result's metrics. */
export type BenchmarkPayload = {
  symbol: string;
  series: BenchmarkSeries[];
  time_in_market_percent: number | null;
  /** Portfolios: every symbol the book trades. The benchmark is the one most strategies trade. */
  trading_symbols?: string[];
};

export type EquityCurvePoint = {
  date: string;
  rolling_pnl: number;
  drawdown: number;
  benchmark_underlying?: number | null;
  benchmark_vehicle?: number | null;
};

export interface DetailedResult {
  vehicle?: TradedVehicle;
  benchmark?: BenchmarkPayload | null;
  summary: Record<string, number | string | null>;
  yearly: Array<Record<string, number | string | null>>;
  monthly: Array<Record<string, number | string | null>>;
  equity_curve: EquityCurvePoint[];
  trades: Array<Record<string, number | string | null>>;
  equity_curve_total_points?: number;
  equity_curve_shown_points?: number;
  contribution?: StrategyContribution[] | null;
  /** Present on portfolio simulations. */
  allocation?: AllocationSummary | null;
  /** Present on portfolio simulations: episodes and their rebalance events. */
  positions?: PositionsPayload | null;
  /** Present on portfolio simulations with enough history to resample. */
  drawdown_risk?: DrawdownRisk | null;
}

export type MarketRegimeCurrent = {
  as_of: string | null;
  regimes: Partial<Record<string, string | string[] | null>>;
  readings?: Partial<Record<string, number | null>>;
};

export type MarketRegimeCoverage = Partial<Record<string, { start: string; end: string } | null>>;

export type RegimeSharpeBucket = {
  key: string;
  sharpe: number | null;
  sortino: number | null;
  days: number;
  max_drawdown: number | null;
  cagr: number | null;
  calmar: number | null;
  regime_score: number | null;
  regime_score_sortino: number | null;
  regime_score_return: number | null;
  regime_score_drawdown: number | null;
  regime_score_robustness: number | null;
  /** Average-trade and robustness parts only; the default regime score. */
  robustness_score: number | null;
};

export type MarketRegimeSharpe = Partial<Record<string, RegimeSharpeBucket[]>>;

/** A request that produced a backtest result, so regimes can be computed for it on demand. */
export type RegimeSource = {
  path: string;
  payload: object;
};

/** Everything the Market regimes charts need. Trades carry their entry-day regime labels. */
export type RegimeOverview = {
  market_regime_sharpe: MarketRegimeSharpe;
  market_regime_current: MarketRegimeCurrent;
  market_regime_coverage: MarketRegimeCoverage;
  trades: DetailedResult["trades"];
};

export type RegimeStateDef = { key: string; label: string };

export type RegimeYearRow = {
  year: number;
  days: Record<string, number>;
  total_days: number;
  changes: number;
};

export type RegimeMarketAnalysis = {
  regime: { id: string; title: string; hint: string };
  states: RegimeStateDef[];
  rows: RegimeYearRow[];
  coverage: { start: string; end: string } | null;
};

export type RegimeTradeCount = { trades: number; profitable: number };

export type RegimeStrategyYearRow = RegimeYearRow & {
  trades: Record<string, RegimeTradeCount>;
  total_trades: number;
  profitable_trades: number;
  return: number | null;
};

export type RegimeGranularState = RegimeSharpeBucket & {
  label: string;
  trades: number;
  profitable: number;
  /** Average closed-trade return, as a fraction. */
  avg_trade_return: number | null;
  /** Average closed-trade return per day held, as a fraction. */
  exposure: number | null;
};

export type RegimeStrategyAnalysis = {
  regime: { id: string; title: string; hint: string };
  states: RegimeStateDef[];
  rows: RegimeStrategyYearRow[];
  period: { start: string; end: string };
  granular: {
    refined: boolean;
    states: RegimeGranularState[];
    rows: RegimeStrategyYearRow[];
    base_score: number | null;
    base_robustness_score: number | null;
  };
};

export type MonteCarloMethod = "shuffle" | "bootstrap";

export type MonteCarloPercentiles = {
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
  mean: number;
};

export type MonteCarloDrawdownBin = {
  drawdown_pct: number;
  count: number;
  cumulative_pct: number;
};

export type MonteCarloConfidenceMarker = {
  percentile: number;
  drawdown_pct: number;
  cumulative_pct: number;
};

export type MonteCarloResult = {
  method: MonteCarloMethod;
  n_sims: number;
  n_trades: number;
  start_capital: number;
  actual: {
    final_equity: number;
    max_drawdown: number;
  };
  summary: {
    final_equity: MonteCarloPercentiles;
    max_drawdown: MonteCarloPercentiles;
    pct_sims_final_equity_ge_actual: number;
    pct_sims_max_drawdown_le_actual: number;
  };
  drawdown_distribution: MonteCarloDrawdownBin[];
  confidence_marker: MonteCarloConfidenceMarker;
};

export type MonteCarloPayload = {
  trade_returns: number[];
  method: MonteCarloMethod;
  n_sims?: number;
  start_capital?: number;
};

export interface CustomDatasetInfo {
  id: string;
  symbol: string;
  interval_minutes: number;
  interval_label: string;
  periods_per_year: number;
  start: string;
  end: string;
  row_count: number;
  unavailable_indicator_ids: string[];
  has_vwap: boolean;
  custom_data_only_indicator_ids: string[];
  timezone: string;
  is_intraday: boolean;
}

export type SweepResult = Array<Record<string, number | string | null>>;

export type BuilderConditionPayload = {
  left: string;
  operator:
    | "<"
    | "<="
    | ">"
    | ">="
    | "="
    | "crosses above"
    | "crosses below"
    | "is true"
    | "is false";
  right: string;
  logic: "AND" | "OR";
};

export const FLAG_OPERATORS = ["is true", "is false"] as const;
export type FlagOperator = (typeof FLAG_OPERATORS)[number];

export function isFlagOperator(operator: string): operator is FlagOperator {
  return FLAG_OPERATORS.includes(operator as FlagOperator);
}

export type BuilderBacktestPayload = {
  symbol: string;
  years?: number;
  direction: "long" | "short";
  hold_days: number;
  profit?: number;
  name?: string;
  description?: string;
  conditions: BuilderConditionPayload[];
  sell_conditions?: BuilderConditionPayload[];
  confirm_symbols?: string[];
  proxy_symbol?: string;
  custom_dataset_id?: string;
  rth_entries_only?: boolean;
  eod_exit?: boolean;
  backtest_all_data?: boolean;
  hold_on_buy_signal?: boolean;
};

export type BuilderRefinePayload = {
  mode: import("@/lib/builder-refine-config").BuilderRefineMode;
  strategy: BuilderBacktestPayload;
  secondary_strategy_id?: string;
  primary_symbol?: string;
  symbol_pool?: string[];
  max_days?: number;
  is_sell?: boolean;
  check_breadth?: boolean;
  check_both?: boolean;
  combo_size?: 2 | 3;
  combo_top_n?: number;
  combo_beam_width?: number;
  basket_symbols?: string[];
};

export type RefineSampleMode = "in_sample" | "full";

/** The draft scored like a sweep row: same columns, same metric code. */
export type DraftBaselineMetrics = {
  PnL: string;
  MaxDD: string;
  Trades: number;
  "%Pstv": number;
  CAGR: string;
  Sharpe: number | string | null;
  Sortino: number | string | null;
};

/** The unmodified draft on the window the rows are ranked on and on the full period. */
export type DraftBaseline = {
  in_sample: DraftBaselineMetrics;
  full_period: DraftBaselineMetrics;
};

export type SweepMeta = {
  sample: RefineSampleMode;
  /** Not sent for a basket test, whose own symbol is its reference row. */
  draft?: DraftBaseline | null;
  in_sample_fraction: number;
  in_sample_end: string;
  period_start: string;
  period_end: string;
  min_in_sample_trades?: number;
  combo_search?: ComboSearchMeta;
  basket?: BasketSummary;
};

/** What a basket test found across the basket (the draft's own symbol is reported separately). */
export type BasketSummary = {
  draft_symbol: string;
  requested: number;
  evaluated: number;
  failed: Array<{ symbol: string; source: string; error: string }>;
  draft_sharpe: number | null;
  median_sharpe: number | null;
  share_sharpe_positive: number | null;
  share_cagr_positive: number | null;
  share_beating_null: number | null;
  null_evaluated: number;
  null_alpha: number;
  null_shifts: number;
  ignored_proxy_symbol: string | null;
  confirm_symbols: string[];
};

/** How much an indicator combo sweep searched. Rank by it: more tested means more luck. */
export type ComboSearchMeta = {
  size: number;
  grid_points: number;
  candidates: number;
  combinations_tested: number;
  search: string;
  engine: string;
  market_wide_only: boolean;
};

export type SweepResultResponse = {
  rows: SweepResult;
  meta: SweepMeta;
};

export type BuilderRefineOutcomePayload = {
  mode: import("@/lib/builder-refine-config").BuilderRefineMode;
  strategy: BuilderBacktestPayload;
  row: Record<string, unknown>;
  sample: RefineSampleMode;
  primary_symbol?: string;
  symbol_pool?: string[];
  max_days?: number;
  is_sell?: boolean;
};

export type BuilderRefineOutcomeResponse = {
  metrics: Record<string, unknown>;
  meta: SweepMeta;
};

export type ScanLane = "active" | "testing" | "archived";

export type SaveStrategyPayload = {
  name: string;
  symbol: string;
  direction: "long" | "short";
  hold_days: number;
  profit: number;
  description?: string;
  conditions: BuilderConditionPayload[];
  sell_conditions?: BuilderConditionPayload[];
  confirm_symbols?: string[];
  proxy_symbol?: string;
  hold_on_buy_signal?: boolean;
  rth_entries_only?: boolean;
  eod_exit?: boolean;
  scan_lane?: ScanLane;
  scan_sort_order?: number;
};

export type SavedStrategy = SaveStrategyPayload & {
  id: string;
  legacy_signal?: string | null;
  created_at: string;
  updated_at: string;
};

export type UpdateStrategyPayload = {
  name?: string;
  symbol?: string;
  direction?: "long" | "short";
  hold_days?: number;
  profit?: number;
  description?: string;
  conditions?: BuilderConditionPayload[];
  sell_conditions?: BuilderConditionPayload[];
  confirm_symbols?: string[];
  proxy_symbol?: string | null;
  hold_on_buy_signal?: boolean;
  rth_entries_only?: boolean;
  eod_exit?: boolean;
  scan_lane?: ScanLane;
  scan_sort_order?: number;
};

export type ConditionSnapshotItem = {
  label: string;
  logic?: "AND" | "OR" | null;
  passed: boolean;
  left_value?: string | null;
  right_value?: string | null;
  operator: string;
  left: string;
  right: string;
};

/** What to do with a row today. Enter / Add / Reduce / Exit change the position; Hold and Flat do not. */
export type ScanAction = "Enter" | "Add" | "Reduce" | "Exit" | "Rebalance" | "Hold" | "Flat";

export type ScanOrder = {
  instrument: string;
  underlying: string;
  kind: "vehicle" | "synthetic";
  leverage: number | null;
  side: "buy" | "sell";
  event: string;
  weight_before: number;
  weight_after: number;
  price: number | null;
};

export type ScanHolding = {
  instrument: string;
  underlying: string;
  kind: "vehicle" | "synthetic";
  leverage: number | null;
  weight: number;
  price: number | null;
};

/** Shares of the row's own book (1 is all of it), never of an account. */
export type ScanPosition = {
  action: ScanAction;
  /** Held going into today's decision, drifted to today's close. */
  held_before: number;
  /** Held after today's decision. With no trade it is the drifted share. */
  held_now: number;
  /** What the model asked for at its last change. */
  target: number;
  delta: number;
  active: number;
  total: number;
  active_strategies: string[];
  entered: string[];
  exited: string[];
  reason: string;
  orders: ScanOrder[];
  holdings: ScanHolding[];
  model: string;
  as_of: string | null;
  /** Why the sizing is approximate (regime-adjusted read as linear), or null when exact. */
  approximate: string | null;
};

export type ScanRow = {
  id: string;
  source: "legacy" | "builder" | "portfolio";
  strategy_id: string | null;
  portfolio_id?: string | null;
  symbol: string;
  signal: string;
  buy_signal: boolean;
  hold_long: boolean;
  sell_signal: boolean;
  days: number;
  profit: number;
  trade_pnl: number;
  kelly: number | null;
  description: string;
  condition_snapshot?: ConditionSnapshotItem[] | null;
  condition_passed_count?: number | null;
  condition_total_count?: number | null;
  sell_condition_snapshot?: ConditionSnapshotItem[] | null;
  sell_condition_passed_count?: number | null;
  sell_condition_total_count?: number | null;
  condition_as_of?: string | null;
  position?: ScanPosition | null;
};

export type PortfolioOverlapMode = "first_signal_only" | "hold_until_all_exit";

/**
 * All-in is the original behavior. Proportional sizes the book by active / total strategies.
 * Regime-adjusted scales the position by how well the current market regimes have treated the
 * active strategies and the portfolio. Tiered puts `tier_single_pct` of the book in when exactly
 * one member is active and all of it when two or more are.
 */
export type PortfolioAllocationModel = "all_in" | "proportional" | "regime_adjusted" | "tiered";
export type PortfolioProportionalMode = "linear" | "early_conviction" | "confirmation_required";
export type RegimeScoreKind = "robustness" | "portfolio";
/** How hard a regime state that has scored poorly is punished. */
export type RegimePenaltyPreset = "harsh" | "strict" | "medium" | "low";

/** Regime-adjusted settings. Ignored by other models. */
export type RegimePenaltyFields = {
  regime_score?: RegimeScoreKind;
  regime_penalty?: RegimePenaltyPreset;
};

export type SavePortfolioPayload = RegimePenaltyFields & {
  name: string;
  description?: string;
  strategy_ids: string[];
  portfolio_ids?: string[];
  overlap_mode: PortfolioOverlapMode;
  proxy_symbol?: string;
  allocation_model?: PortfolioAllocationModel;
  proportional_mode?: PortfolioProportionalMode;
  /** Tiered only: percent of the book when exactly one member is active (default 55). */
  tier_single_pct?: number;
  transaction_cost_bps?: number;
};

export type SavedPortfolio = SavePortfolioPayload & {
  id: string;
  scan_lane?: ScanLane;
  scan_sort_order?: number;
  created_at: string;
  updated_at: string;
};

export type UpdatePortfolioPayload = RegimePenaltyFields & {
  name?: string;
  description?: string;
  scan_lane?: ScanLane;
  scan_sort_order?: number;
  strategy_ids?: string[];
  portfolio_ids?: string[];
  overlap_mode?: PortfolioOverlapMode;
  proxy_symbol?: string | null;
  allocation_model?: PortfolioAllocationModel;
  proportional_mode?: PortfolioProportionalMode;
  tier_single_pct?: number;
  transaction_cost_bps?: number;
};

export type PortfolioSimulatePayload = RegimePenaltyFields & {
  strategy_ids: string[];
  portfolio_ids?: string[];
  overlap_mode: PortfolioOverlapMode;
  proxy_symbol?: string;
  years?: number;
  allocation_model?: PortfolioAllocationModel;
  proportional_mode?: PortfolioProportionalMode;
  tier_single_pct?: number;
  transaction_cost_bps?: number;
};

export type AllocationAnalytics = {
  bars: number;
  average_exposure_pct: number;
  average_exposure_when_invested_pct: number | null;
  max_exposure_pct: number;
  time_invested_pct: number;
  rebalance_count: number;
  entry_count: number;
  exit_count: number;
  adjustment_count: number;
  total_traded_notional: number;
  turnover_total_pct: number | null;
  turnover_annualized_pct: number | null;
  total_transaction_costs: number;
  transaction_costs_pct_of_avg_equity: number | null;
};

export type AllocationAttribution = {
  reference: string;
  allocation_effect_pp: number;
  cost_effect_pp: number;
  other_pp: number;
  total_return_gap_pp: number;
  final_equity_delta: number;
  final_equity_delta_pct: number | null;
  days_below_reference: number;
  days_partially_invested: number;
  average_eligible_strategies: number;
  by_active_count: Array<{
    active_count: number;
    days: number;
    average_exposure_pct: number;
    allocation_effect_pp: number;
  }>;
};

export type AllocationEligibility = {
  strategy_id: string;
  strategy_name: string;
  first_eligible_date: string | null;
  last_available_date: string | null;
  warmup_bars: number;
  eligible_bars: number;
  unavailable_bars: number;
};

export type AllocationLedger = {
  cash: number;
  invested_capital: number;
  equity: number;
  realized_pnl: number;
  unrealized_pnl: number;
  transaction_costs: number;
  positions: Array<{
    symbol: string;
    quantity: number;
    price: number;
    market_value: number;
    average_cost: number;
    unrealized_pnl: number;
  }>;
};

export type AllocationSummary = {
  nested_cash_policy?: "respect_child_rejection";
  members?: Array<{ id: string; name: string; budget_weight: number; invested_weight: number; cash_weight: number }>;
  model: string;
  label: string;
  transaction_cost_bps: number;
  /** The instruments the portfolio trades (after proxies). More than one means a mixed book. */
  trading_symbols?: string[];
  general_leverage?: number;
  /** One row per symbol with a leverage-settings entry: which leverage symbol, and since when. */
  vehicles?: Array<{
    symbol: string;
    leverage_symbol: string | null;
    leverage: number;
    first_date: string | null;
    /** False: the leverage symbol has no price history, so the stand-in carried every trade. */
    available: boolean;
  }>;
  /** The penalties the run used; present for regime-adjusted only. Penalties are in percent. */
  regime?: {
    preset: RegimePenaltyPreset | null;
    score: RegimeScoreKind;
    low_cutoff: number;
    low_penalty: number;
    high_cutoff: number;
    high_penalty: number;
    min_trades: number;
  };
  analytics: AllocationAnalytics;
  /** Null for all-in, which is its own reference. */
  attribution: AllocationAttribution | null;
  eligibility: AllocationEligibility[] | null;
  ledger: AllocationLedger | null;
  /** Target exposure as a step function: one row per change. */
  exposure_steps: Array<{
    date: string;
    exposure: number;
    active_count?: number;
    eligible_count?: number;
  }>;
};

export type AllocationComparisonRow = {
  model: string;
  label: string;
  metrics: Record<string, number | null>;
  analytics: AllocationAnalytics;
  attribution: AllocationAttribution | null;
  drawdown_risk?: DrawdownRisk | null;
};

export type AllocationComparison = {
  start_date: string;
  start_capital: number;
  transaction_cost_bps: number;
  /** The score the regime-adjusted rows read. */
  regime_score?: RegimeScoreKind;
  trading_symbols?: string[];
  single_symbol?: boolean;
  models: AllocationComparisonRow[];
  /** One row per date: `date` plus each model id's rebased equity. */
  equity_curves: Array<Record<string, number | string>>;
};

export type PortfolioShapleyPayload = PortfolioSimulatePayload & {
  samples?: number;
  seed?: number;
};

export type PortfolioShapleyResult = {
  samples_used: number;
  exact: boolean;
  shapley: StrategyContribution[];
};

export type RegimeBookMetrics = {
  regime_score: number | null;
  robustness_score: number | null;
  sortino: number | null;
  avg_trade_return: number | null;
  max_drawdown: number | null;
  robustness: number | null;
  exposure: number | null;
  cagr: number | null;
  trades: number;
};

export type RegimeMetricDelta = {
  regime_score: number | null;
  robustness_score: number | null;
  sortino: number | null;
  avg_trade_return: number | null;
  max_drawdown: number | null;
  robustness: number | null;
  exposure: number | null;
  cagr: number | null;
};

export type RegimeBook = {
  base: RegimeBookMetrics;
  regimes: Record<string, Record<string, RegimeBookMetrics>>;
};

export type RegimeContributionCell = {
  delta: RegimeMetricDelta;
  with: RegimeBookMetrics;
  without: RegimeBookMetrics;
};

export type RegimeContributionBook = {
  base: RegimeContributionCell;
  regimes: Record<string, Record<string, RegimeContributionCell>>;
};

export type StrategyRegimeAnalysis = {
  strategy_id: string;
  strategy_name: string;
  quality: RegimeBook;
  contribution: RegimeContributionBook;
};

export type PortfolioRegimeResult = {
  portfolio: RegimeBook;
  strategies: StrategyRegimeAnalysis[];
};

export type PortfolioRegimePayload = PortfolioSimulatePayload;

export async function getIndicators(builderOnly = true): Promise<IndicatorInfo[]> {
  const response = await fetch(`${API_URL}/indicators?builder_only=${builderOnly}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

/** The server's `detail` message when it sent one, otherwise the raw body. */
async function requestError(response: Response): Promise<Error> {
  const text = await response.text();
  try {
    const body = JSON.parse(text) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail.trim()) {
      return new Error(body.detail);
    }
  } catch {
    // Response body is already plain text.
  }
  return new Error(text || response.statusText);
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw await requestError(response);
  return response.json();
}

async function patchJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw await requestError(response);
  return response.json();
}

async function putJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw await requestError(response);
  return response.json();
}

async function deleteJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "DELETE",
  });
  if (!response.ok) throw await requestError(response);
  return response.json();
}

/** Re-run the request behind a backtest result and return only its regime figures. */
export function fetchRegimeOverview(source: RegimeSource): Promise<RegimeOverview> {
  return postJson<RegimeOverview>(source.path, { ...source.payload, regime_mode: "overview" });
}

export async function fetchRegimeStrategyAnalysis(
  source: RegimeSource,
  regimeId: string,
): Promise<RegimeStrategyAnalysis> {
  const response = await postJson<{ regime_analysis: RegimeStrategyAnalysis }>(source.path, {
    ...source.payload,
    regime_mode: "analyze",
    regime_id: regimeId,
  });
  return response.regime_analysis;
}

export function analyzeMarketRegime(payload: {
  regime_id: string;
  years: number;
}): Promise<RegimeMarketAnalysis> {
  return postJson<RegimeMarketAnalysis>("/regimes/analyze", payload);
}

export function runMonteCarlo(payload: MonteCarloPayload) {
  return postJson<MonteCarloResult>("/backtests/monte-carlo", payload);
}

export function runBuilderBacktest(payload: BuilderBacktestPayload): Promise<DetailedResult> {
  return postJson<DetailedResult>("/backtests/builder", payload);
}

export function runBuilderRefine(
  payload: BuilderRefinePayload,
): Promise<SweepResultResponse> {
  return postJson<SweepResultResponse>("/backtests/builder/refine", payload);
}

export function fetchBuilderRefineOutcome(
  payload: BuilderRefineOutcomePayload,
): Promise<BuilderRefineOutcomeResponse> {
  return postJson<BuilderRefineOutcomeResponse>(
    "/backtests/builder/refine/outcome",
    payload,
  );
}

export function runPortfolioSimulation(
  payload: PortfolioSimulatePayload,
): Promise<DetailedResult> {
  return postJson<DetailedResult>("/portfolios/simulate", payload);
}

export function runPortfolioShapley(
  payload: PortfolioShapleyPayload,
): Promise<PortfolioShapleyResult> {
  return postJson<PortfolioShapleyResult>("/portfolios/shapley", payload);
}

export function runPortfolioAllocationComparison(
  payload: PortfolioSimulatePayload,
): Promise<AllocationComparison> {
  return postJson<AllocationComparison>("/portfolios/allocation-comparison", payload);
}

export function runPortfolioRegimes(
  payload: PortfolioRegimePayload,
): Promise<PortfolioRegimeResult> {
  return postJson<PortfolioRegimeResult>("/portfolios/regimes", payload);
}

export function savePortfolio(payload: SavePortfolioPayload): Promise<SavedPortfolio> {
  return postJson<SavedPortfolio>("/portfolios", payload);
}

export function updatePortfolio(
  id: string,
  payload: UpdatePortfolioPayload,
): Promise<SavedPortfolio> {
  return patchJson<SavedPortfolio>(`/portfolios/${id}`, payload);
}

export function deletePortfolio(id: string): Promise<{ deleted: boolean }> {
  return deleteJson<{ deleted: boolean }>(`/portfolios/${id}`);
}

export async function getSavedPortfolios(): Promise<SavedPortfolio[]> {
  const response = await fetch(`${API_URL}/portfolios`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export function saveStrategy(payload: SaveStrategyPayload): Promise<SavedStrategy> {
  return postJson<SavedStrategy>("/strategies", payload);
}

export function updateStrategy(
  id: string,
  payload: UpdateStrategyPayload,
): Promise<SavedStrategy> {
  return patchJson<SavedStrategy>(`/strategies/${id}`, payload);
}

export function deleteStrategy(id: string): Promise<{ deleted: boolean }> {
  return deleteJson<{ deleted: boolean }>(`/strategies/${id}`);
}

export async function getSavedStrategies(): Promise<SavedStrategy[]> {
  const response = await fetch(`${API_URL}/strategies`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function getScan(source: "yahoo" | "ib" = "yahoo"): Promise<ScanRow[]> {
  const params = new URLSearchParams({ source });
  const response = await fetch(`${API_URL}/api/scan?${params}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

/** A portfolio's position today at its real sizing: the full history, slower than a scan row. */
export async function getExactScanPosition(portfolioId: string): Promise<ScanPosition> {
  const response = await fetch(`${API_URL}/api/scan/portfolios/${encodeURIComponent(portfolioId)}/position`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export type QuoteSymbol = {
  symbol: string;
  as_of: string | null;
  close: number | null;
  pct_change: number | null;
  ibr: number | null;
  rsi2: number | null;
  rsi5: number | null;
  stoch: number | null;
  missing_days: string[];
};

export type QuoteSnapshot = {
  as_of: string | null;
  quotes: QuoteSymbol[];
};

export async function getQuote(): Promise<QuoteSnapshot> {
  const response = await fetch(`${API_URL}/api/quote`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

const quoteRequestError = requestError;

export async function addQuoteSymbol(symbol: string): Promise<QuoteSnapshot> {
  const response = await fetch(`${API_URL}/api/quote/symbols`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol }),
  });
  if (!response.ok) throw await quoteRequestError(response);
  return response.json();
}

export async function deleteQuoteSymbol(symbol: string): Promise<QuoteSnapshot> {
  const response = await fetch(
    `${API_URL}/api/quote/symbols/${encodeURIComponent(symbol)}`,
    { method: "DELETE" },
  );
  if (!response.ok) throw await quoteRequestError(response);
  return response.json();
}

export async function uploadCustomDataset(file: File): Promise<CustomDatasetInfo> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_URL}/datasets/custom`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function deleteCustomDataset(datasetId: string): Promise<{ deleted: boolean }> {
  return deleteJson<{ deleted: boolean }>(`/datasets/custom/${datasetId}`);
}

/** One symbol and the leveraged product traded in its place. */
export type SymbolLeverage = {
  symbol: string;
  /** Null: no leveraged product, the symbol itself is traded at `leverage` (1 for an unleveraged stock). */
  leverage_symbol: string | null;
  leverage: number;
  /** Effective beta of the stand-in used before the leverage symbol exists; null means `leverage`. */
  beta: number | null;
  /** Fraction per year (0.019 = 1.9%). */
  annual_fee: number;
  /** Multiples of the 3-month T-bill rate, per year. */
  financing_multiple: number;
};

export type LeverageSettings = {
  general_leverage: number;
  symbols: SymbolLeverage[];
};

export type LeverageEstimate = {
  beta: number;
  fee: number;
  financing_multiple: number;
  days: number;
  first_date: string;
  last_date: string;
  correlation: number;
  tracking_error: number;
  symbol_first_date: string;
  leverage_symbol_first_date: string;
};

export function getLeverageSettings(): Promise<LeverageSettings> {
  return fetch(`${API_URL}/settings/leverage`).then(async (response) => {
    if (!response.ok) throw await requestError(response);
    return response.json();
  });
}

export function saveLeverageSettings(settings: LeverageSettings): Promise<LeverageSettings> {
  return putJson<LeverageSettings>("/settings/leverage", settings);
}

export function estimateLeverage(symbol: string, leverageSymbol: string): Promise<LeverageEstimate> {
  return postJson<LeverageEstimate>("/settings/leverage/estimate", {
    symbol,
    leverage_symbol: leverageSymbol,
  });
}
