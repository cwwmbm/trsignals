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

export interface DetailedResult {
  summary: Record<string, number | string | null>;
  yearly: Array<Record<string, number | string | null>>;
  monthly: Array<Record<string, number | string | null>>;
  equity_curve: Array<{ date: string; rolling_pnl: number; drawdown: number }>;
  trades: Array<Record<string, number | string | null>>;
  equity_curve_total_points?: number;
  equity_curve_shown_points?: number;
  contribution?: StrategyContribution[] | null;
}

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

export type SavePortfolioPayload = {
  name: string;
  description?: string;
  strategy_ids: string[];
  overlap_mode: PortfolioOverlapMode;
  proxy_symbol?: string;
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
};

export type PortfolioSimulatePayload = {
  strategy_ids: string[];
  overlap_mode: PortfolioOverlapMode;
  proxy_symbol?: string;
  years?: number;
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

export async function getSignals(): Promise<SignalInfo[]> {
  const response = await fetch(`${API_URL}/signals`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

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

export function runBacktest(mode: RunMode, payload: Record<string, unknown>) {
  const pathByMode: Record<RunMode, string> = {
    single: "/backtests/single",
    "signal-combo-sweep": "/backtests/signal-combo-sweep",
    "symbol-confirm-sweep": "/backtests/symbol-confirm-sweep",
    "symbol-confirm-detail": "/backtests/symbol-confirm-detail",
    "hold-days-sweep": "/backtests/hold-days-sweep",
    "indicator-sweep": "/backtests/indicator-sweep",
  };

  return postJson<DetailedResult | SweepResult>(pathByMode[mode], payload);
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

export async function getScan(): Promise<ScanRow[]> {
  const response = await fetch(`${API_URL}/api/scan`);
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
