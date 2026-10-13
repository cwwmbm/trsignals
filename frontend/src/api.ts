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
export type PositionEpisode = {
  id: string;
  symbol: string;
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

export interface DetailedResult {
  summary: Record<string, number | string | null>;
  yearly: Array<Record<string, number | string | null>>;
  monthly: Array<Record<string, number | string | null>>;
  equity_curve: Array<{ date: string; rolling_pnl: number; drawdown: number }>;
  trades: Array<Record<string, number | string | null>>;
  equity_curve_total_points?: number;
  equity_curve_shown_points?: number;
  contribution?: StrategyContribution[] | null;
  /** Present on portfolio simulations. */
  allocation?: AllocationSummary | null;
  /** Present on portfolio simulations: episodes and their rebalance events. */
  positions?: PositionsPayload | null;
}

export type MarketRegimeCurrent = {
  as_of: string | null;
  regimes: Partial<Record<string, string | null>>;
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
};

export type RefineSampleMode = "in_sample" | "full";

export type SweepMeta = {
  sample: RefineSampleMode;
  in_sample_fraction: number;
  in_sample_end: string;
  period_start: string;
  period_end: string;
  min_in_sample_trades?: number;
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
};

export type PortfolioOverlapMode = "first_signal_only" | "hold_until_all_exit";

/** All-in is the original behavior. Proportional sizes the book by active / total strategies. */
export type PortfolioAllocationModel = "all_in" | "proportional";
export type PortfolioProportionalMode = "linear" | "early_conviction" | "confirmation_required";

export type SavePortfolioPayload = {
  name: string;
  description?: string;
  strategy_ids: string[];
  overlap_mode: PortfolioOverlapMode;
  proxy_symbol?: string;
  allocation_model?: PortfolioAllocationModel;
  proportional_mode?: PortfolioProportionalMode;
  transaction_cost_bps?: number;
};

export type SavedPortfolio = SavePortfolioPayload & {
  id: string;
  scan_lane?: ScanLane;
  scan_sort_order?: number;
  created_at: string;
  updated_at: string;
};

export type UpdatePortfolioPayload = {
  name?: string;
  description?: string;
  scan_lane?: ScanLane;
  scan_sort_order?: number;
  strategy_ids?: string[];
  overlap_mode?: PortfolioOverlapMode;
  proxy_symbol?: string | null;
  allocation_model?: PortfolioAllocationModel;
  proportional_mode?: PortfolioProportionalMode;
  transaction_cost_bps?: number;
};

export type PortfolioSimulatePayload = {
  strategy_ids: string[];
  overlap_mode: PortfolioOverlapMode;
  proxy_symbol?: string;
  years?: number;
  allocation_model?: PortfolioAllocationModel;
  proportional_mode?: PortfolioProportionalMode;
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
  model: string;
  label: string;
  transaction_cost_bps: number;
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
};

export type AllocationComparison = {
  start_date: string;
  start_capital: number;
  transaction_cost_bps: number;
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

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function patchJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function deleteJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error(await response.text());
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
