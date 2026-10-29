export type RegimeField =
  | "vix_regime"
  | "vxn_regime"
  | "vix_change_regime"
  | "vix_rv_regime"
  | "atr_regime"
  | "spy_regime"
  | "market_breadth_regime"
  | "semis_breadth_regime"
  | "equity_risk_breadth_regime"
  | "credit_risk_breadth_regime"
  | "credit_risk_on_regime"
  | "bond_duration_regime"
  | "copper_gold_regime"
  | "materials_breadth_regime"
  | "housing_strength_regime"
  | "housing_defensive_regime"
  | "sector_breadth_50_regime"
  | "sector_breadth_200_regime"
  | "sector_trend_50_regime"
  | "rate_shock_regime"
  | "curve_10y3m_regime"
  | "curve_change_20_regime"
  | "rate_curve_regime"
  | "dollar_shock_regime"
  | "dollar_rates_regime"
  | "inflation_regime"
  | "inflation_yield_regime";

export type RegimeBucketDef = {
  key: string;
  label: string;
};

export type RegimeDimension = {
  id: string;
  title: string;
  hint: string;
  field: RegimeField;
  buckets: RegimeBucketDef[];
  unavailableWhenEmpty: boolean;
  /** Buckets overlap: a day or trade can sit in several (trade field is a list of keys). */
  overlapping?: boolean;
};

export type RegimeSharpeBucket = {
  key: string;
  sharpe: number | null;
  sortino?: number | null;
  days: number;
  max_drawdown?: number | null;
  cagr?: number | null;
  calmar?: number | null;
  regime_score?: number | null;
  regime_score_sortino?: number | null;
  regime_score_return?: number | null;
  regime_score_drawdown?: number | null;
  regime_score_robustness?: number | null;
  robustness_score?: number | null;
};

export type RegimeTradeRow = Record<string, number | string | string[] | null>;

export type MarketRegimeSharpe = Partial<Record<string, RegimeSharpeBucket[]>>;

export type RegimeChartPoint = {
  key: string;
  label: string;
  sharpe: number | null;
  sortino: number | null;
  days: number;
  maxDrawdown: number | null;
  cagr: number | null;
  calmar: number | null;
  trades: number;
  pctPositive: number | null;
  avgReturnPct: number | null;
  avgTradeReturnPct: number | null;
  regimeScore: number | null;
  regimeScoreSortino: number | null;
  regimeScoreReturn: number | null;
  regimeScoreDrawdown: number | null;
  regimeScoreRobustness: number | null;
  robustnessScore: number | null;
};

export type RegimeYMetric =
  | "robustness_score"
  | "regime_score"
  | "sharpe"
  | "sortino"
  | "calmar"
  | "avg_return"
  | "avg_trade_return"
  | "max_drawdown";

const VOL_BUCKETS: RegimeBucketDef[] = [
  { key: "le_15", label: "≤ 15" },
  { key: "15_20", label: "15–20" },
  { key: "20_30", label: "20–30" },
  { key: "gt_30", label: "> 30" },
];

const VIX_RV_BUCKETS: RegimeBucketDef[] = [
  { key: "le_10", label: "≤ 10th" },
  { key: "10_30", label: "10–30th" },
  { key: "30_70", label: "30–70th" },
  { key: "70_90", label: "70–90th" },
  { key: "gt_90", label: "> 90th" },
];

const VIX_CHANGE_BUCKETS: RegimeBucketDef[] = [
  { key: "le_neg_15", label: "≤ -15%" },
  { key: "le_neg_10", label: "≤ -10%" },
  { key: "le_neg_5", label: "≤ -5%" },
  { key: "le_0", label: "≤ 0%" },
  { key: "ge_0", label: "≥ 0%" },
  { key: "ge_5", label: "≥ 5%" },
  { key: "ge_10", label: "≥ 10%" },
  { key: "ge_15", label: "≥ 15%" },
  { key: "ge_20", label: "≥ 20%" },
];

const SECTOR_BREADTH_BUCKETS: RegimeBucketDef[] = [
  { key: "le_25", label: "≤ 25%" },
  { key: "25_50", label: "25–50%" },
  { key: "50_75", label: "50–75%" },
  { key: "gt_75", label: "> 75%" },
];

const SECTOR_TREND_BUCKETS: RegimeBucketDef[] = [
  { key: "lt_neg_5", label: "< -5%" },
  { key: "neg_5_to_0", label: "-5% to 0%" },
  { key: "zero_to_pos_5", label: "0% to +5%" },
  { key: "gt_pos_5", label: "> +5%" },
];

const DOLLAR_RATES_BUCKETS: RegimeBucketDef[] = [
  { key: "tnx_nonpos_dollar_nonpos", label: "≤ 0, ≤ 0" },
  { key: "tnx_nonpos_dollar_pos", label: "≤ 0, > 0" },
  { key: "tnx_pos_dollar_nonpos", label: "> 0, ≤ 0" },
  { key: "tnx_pos_dollar_pos", label: "> 0, > 0" },
];

const INFLATION_YIELD_BUCKETS: RegimeBucketDef[] = [
  { key: "tnx_rising_inflation_rising", label: "> 0, > 0" },
  { key: "tnx_rising_inflation_falling", label: "> 0, ≤ 0" },
  { key: "tnx_falling_inflation_rising", label: "≤ 0, > 0" },
  { key: "tnx_falling_inflation_falling", label: "≤ 0, ≤ 0" },
];

const SHOCK_BUCKETS: RegimeBucketDef[] = [
  { key: "lt_neg_1", label: "< -1" },
  { key: "neg_1_to_1", label: "-1 to 1" },
  { key: "gt_1", label: "> 1" },
];

function logRatioHint(pair: string, high?: string, low?: string) {
  let hint = `20-day change in log(${pair}), divided by the 63-day standard deviation of daily changes, times √20`;
  if (high) hint += `. > 1 — ${high}`;
  if (low) hint += `. < -1 — ${low}`;
  return hint;
}

const CURVE_CHANGE_BUCKETS: RegimeBucketDef[] = [
  { key: "le_neg_50", label: "≤ -50 bp" },
  { key: "neg_50_neg_10", label: "-50 to -10" },
  { key: "neg_10_pos_10", label: "-10 to +10" },
  { key: "pos_10_pos_50", label: "+10 to +50" },
  { key: "ge_pos_50", label: "≥ +50 bp" },
];

export const REGIME_TRADE_FIELDS: RegimeField[] = [
  "vix_regime",
  "vxn_regime",
  "vix_change_regime",
  "vix_rv_regime",
  "atr_regime",
  "spy_regime",
  "market_breadth_regime",
  "semis_breadth_regime",
  "equity_risk_breadth_regime",
  "credit_risk_breadth_regime",
  "credit_risk_on_regime",
  "bond_duration_regime",
  "copper_gold_regime",
  "materials_breadth_regime",
  "housing_strength_regime",
  "housing_defensive_regime",
  "sector_breadth_50_regime",
  "sector_breadth_200_regime",
  "sector_trend_50_regime",
  "rate_shock_regime",
  "curve_10y3m_regime",
  "curve_change_20_regime",
  "rate_curve_regime",
  "dollar_shock_regime",
  "dollar_rates_regime",
  "inflation_regime",
  "inflation_yield_regime",
];

export const REGIME_DIMENSIONS: RegimeDimension[] = [
  {
    id: "vix",
    title: "VIX",
    hint: "VIX close that day",
    field: "vix_regime",
    buckets: VOL_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "vxn",
    title: "VXN",
    hint: "VXN close that day",
    field: "vxn_regime",
    buckets: VOL_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "vix_rv",
    title: "VIX vs realized vol",
    hint: "Percentile of VIX / SPY 21-day realized volatility among all earlier days. ≤ 10th — extreme realized dominance. 10–30th — realized dominance. 30–70th — typical. 70–90th — elevated implied premium. > 90th — extreme implied premium",
    field: "vix_rv_regime",
    buckets: VIX_RV_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "vix_change",
    title: "VIX daily change",
    hint: "VIX close-to-close % change that day. Buckets overlap: a -12% day is in ≤ -10%, ≤ -5% and ≤ 0%",
    field: "vix_change_regime",
    buckets: VIX_CHANGE_BUCKETS,
    unavailableWhenEmpty: false,
    overlapping: true,
  },
  {
    id: "atr",
    title: "ATR",
    hint: "Traded symbol ATR(20) vs ATR(50) that day",
    field: "atr_regime",
    buckets: [
      { key: "expanding", label: "ATR(20) > ATR(50)" },
      { key: "contracting", label: "ATR(20) ≤ ATR(50)" },
    ],
    unavailableWhenEmpty: true,
  },
  {
    id: "spy",
    title: "SPY regime",
    hint: "Bull: SMA50 ≥ SMA200 · Bear: SMA50 < SMA200",
    field: "spy_regime",
    buckets: [
      { key: "bull", label: "Bull" },
      { key: "bear", label: "Bear" },
    ],
    unavailableWhenEmpty: false,
  },
  {
    id: "market_breadth",
    title: "Market breadth",
    hint: "20-day change in log(RSP / SPY), divided by the 63-day standard deviation of daily changes, times √20",
    field: "market_breadth_regime",
    buckets: SHOCK_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "semis_breadth",
    title: "Semis breadth",
    hint: logRatioHint(
      "SMH / SPY",
      "strong Semis overperformance compared to SPY",
      "strong Semis underperformance compared to SPY",
    ),
    field: "semis_breadth_regime",
    buckets: SHOCK_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "equity_risk_breadth",
    title: "Equity Risk Breadth",
    hint: logRatioHint(
      "XLY / XLP",
      "strong shift toward risk-on (XLY) / cyclical equities",
      "strong shift toward defensive equities (XLP)",
    ),
    field: "equity_risk_breadth_regime",
    buckets: SHOCK_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "credit_risk_breadth",
    title: "Credit Quality Breadth",
    hint: logRatioHint(
      "HYG / LQD",
      "unusually strong shift toward lower-quality/high-yield credit → increasing credit risk appetite",
      "unusually strong shift toward higher-quality credit → defensive / risk-off credit behavior",
    ),
    field: "credit_risk_breadth_regime",
    buckets: SHOCK_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "credit_risk_on",
    title: "Credit Risk On",
    hint: logRatioHint(
      "HYG / TLT",
      "unusually strong move toward credit/risk assets",
      "unusually strong move toward long-term Treasuries / defensive positioning",
    ),
    field: "credit_risk_on_regime",
    buckets: SHOCK_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "bond_duration",
    title: "Bond Duration Regime",
    hint: logRatioHint(
      "TLT / SHY",
      "long-duration Treasuries are unusually strong relative to short-duration bonds, generally associated with falling long-term yields",
      "long-duration Treasuries are unusually weak, generally associated with rising long-term yields",
    ),
    field: "bond_duration_regime",
    buckets: SHOCK_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "copper_gold",
    title: "Copper/Gold",
    hint: "20-day change in log(HG=F / GC=F), divided by the 63-day standard deviation of daily changes, times √20",
    field: "copper_gold_regime",
    buckets: SHOCK_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "materials_breadth",
    title: "Materials Relative Strength",
    hint: logRatioHint(
      "XLB / SPY",
      "unusually strong materials/cyclical leadership",
      "unusually strong materials underperformance, consistent with weaker cyclical positioning",
    ),
    field: "materials_breadth_regime",
    buckets: SHOCK_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "housing_strength",
    title: "Housing Strength",
    hint: logRatioHint("HGX / SPY"),
    field: "housing_strength_regime",
    buckets: SHOCK_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "housing_defensive",
    title: "Housing vs defensive strength",
    hint: logRatioHint("HGX / XLU"),
    field: "housing_defensive_regime",
    buckets: SHOCK_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "sector_breadth_50",
    title: "Sector breadth 50",
    hint: "Share of XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLK, XLU above their 50-day SMA",
    field: "sector_breadth_50_regime",
    buckets: SECTOR_BREADTH_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "sector_breadth_200",
    title: "Sector breadth 200",
    hint: "Share of XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLK, XLU above their 200-day SMA",
    field: "sector_breadth_200_regime",
    buckets: SECTOR_BREADTH_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "sector_trend_50",
    title: "Sector deviation from SMA50",
    hint: "Mean of log(close / SMA50) across XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLK, XLU. −0.05 is −5%",
    field: "sector_trend_50_regime",
    buckets: SECTOR_TREND_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "rate_shock",
    title: "Rate shock",
    hint: "20-day ^TNX change divided by the 63-day standard deviation of daily changes, times √20",
    field: "rate_shock_regime",
    buckets: SHOCK_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "curve_10y3m",
    title: "10Y–3M curve",
    hint: "^TNX minus ^IRX. Below 0 is inverted",
    field: "curve_10y3m_regime",
    buckets: [
      { key: "inverted", label: "Inverted" },
      { key: "normal", label: "Normal" },
    ],
    unavailableWhenEmpty: false,
  },
  {
    id: "curve_change_20",
    title: "10Y–3M curve change",
    hint: "20-day change in ^TNX minus ^IRX. 0.10 is 10 bp",
    field: "curve_change_20_regime",
    buckets: CURVE_CHANGE_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "rate_curve",
    title: "Rate shock × curve",
    hint: "Rate shock, then the 20-day curve change. A curve change of 0 counts as ≤ 0. A rate shock of 0 is left out",
    field: "rate_curve_regime",
    buckets: [
      { key: "shock_pos_curve_pos", label: "> 0, > 0" },
      { key: "shock_pos_curve_nonpos", label: "> 0, ≤ 0" },
      { key: "shock_neg_curve_pos", label: "< 0, > 0" },
      { key: "shock_neg_curve_nonpos", label: "< 0, ≤ 0" },
    ],
    unavailableWhenEmpty: false,
  },
  {
    id: "dollar_shock",
    title: "Dollar shock",
    hint: "20-day UUP change divided by the 63-day standard deviation of daily changes, times √20",
    field: "dollar_shock_regime",
    buckets: SHOCK_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "dollar_rates",
    title: "Dollar + rates",
    hint: "Rate shock, then the dollar shock. A value of 0 counts as ≤ 0",
    field: "dollar_rates_regime",
    buckets: DOLLAR_RATES_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "inflation",
    title: "Inflation trend",
    hint: "SMA(log(TIP / IEF), 20) minus SMA(log(TIP / IEF), 100). At or below 0 is the lower bucket",
    field: "inflation_regime",
    buckets: [
      { key: "nonpos", label: "≤ 0" },
      { key: "pos", label: "> 0" },
    ],
    unavailableWhenEmpty: false,
  },
  {
    id: "inflation_yield",
    title: "Inflation and yield",
    hint: "Rate shock, then the inflation trend. Above 0 is rising. Zero counts as falling",
    field: "inflation_yield_regime",
    buckets: INFLATION_YIELD_BUCKETS,
    unavailableWhenEmpty: false,
  },
];

export const MIN_REGIME_TRADES = 20;

function closedTradeStats(trades: RegimeTradeRow[], field: RegimeField, key: string) {
  let count = 0;
  let wins = 0;
  let tradePnl = 0;
  let exposurePnl = 0;
  let exposureDays = 0;
  for (const trade of trades) {
    if (String(trade.status ?? "Closed") !== "Closed") continue;
    const held = trade[field];
    if (Array.isArray(held) ? !held.includes(key) : held !== key) continue;
    const pnl = Number(trade.trade_pnl);
    if (!Number.isFinite(pnl)) continue;
    count += 1;
    tradePnl += pnl;
    if (pnl > 0) wins += 1;
    const days = Number(trade.days_in_trade);
    if (Number.isFinite(days) && days > 0) {
      exposurePnl += pnl;
      exposureDays += days;
    }
  }
  return {
    trades: count,
    pctPositive: count > 0 ? (wins / count) * 100 : null,
    avgReturnPct: exposureDays > 0 ? (exposurePnl / exposureDays) * 100 : null,
    avgTradeReturnPct: count > 0 ? (tradePnl / count) * 100 : null,
  };
}

export function regimeChartPoints(
  dimension: RegimeDimension,
  sharpe: MarketRegimeSharpe | null | undefined,
  trades: RegimeTradeRow[] = [],
  includeSmallSamples = false,
): RegimeChartPoint[] {
  const rows = new Map((sharpe?.[dimension.id] ?? []).map((row) => [row.key, row]));
  return dimension.buckets.map((bucket) => {
    const row = rows.get(bucket.key);
    const tradeStats = closedTradeStats(trades, dimension.field, bucket.key);
    const enough = includeSmallSamples || tradeStats.trades >= MIN_REGIME_TRADES;
    return {
      key: bucket.key,
      label: bucket.label,
      sharpe: enough ? (row?.sharpe ?? null) : null,
      sortino: enough ? (row?.sortino ?? null) : null,
      days: row?.days ?? 0,
      maxDrawdown: enough ? (row?.max_drawdown ?? null) : null,
      cagr: enough ? (row?.cagr ?? null) : null,
      calmar: enough ? (row?.calmar ?? null) : null,
      trades: tradeStats.trades,
      pctPositive: enough ? tradeStats.pctPositive : null,
      avgReturnPct: enough ? tradeStats.avgReturnPct : null,
      avgTradeReturnPct: enough ? tradeStats.avgTradeReturnPct : null,
      regimeScore: enough ? (row?.regime_score ?? null) : null,
      regimeScoreSortino: enough ? (row?.regime_score_sortino ?? null) : null,
      regimeScoreReturn: enough ? (row?.regime_score_return ?? null) : null,
      regimeScoreDrawdown: enough ? (row?.regime_score_drawdown ?? null) : null,
      regimeScoreRobustness: enough ? (row?.regime_score_robustness ?? null) : null,
      robustnessScore: enough ? (row?.robustness_score ?? null) : null,
    };
  });
}

export function regimeIndicatorId(dimensionId: string, bucketKey: string) {
  return `Regime_${dimensionId}_${bucketKey}`;
}

export function regimeIndicatorLabel(title: string, bucketLabel: string) {
  return `${title} · ${bucketLabel}`;
}

export function regimeYValue(point: RegimeChartPoint, metric: RegimeYMetric): number | null {
  if (metric === "robustness_score") return point.robustnessScore;
  if (metric === "regime_score") return point.regimeScore;
  if (metric === "sharpe") return point.sharpe;
  if (metric === "sortino") return point.sortino;
  if (metric === "calmar") return point.calmar;
  if (metric === "avg_return") return point.avgReturnPct;
  if (metric === "avg_trade_return") return point.avgTradeReturnPct;
  if (point.maxDrawdown == null) return null;
  return -point.maxDrawdown * 100;
}

/** Regimes that exist independent of any traded symbol, so they have a market-wide history. */
export const MARKET_WIDE_REGIMES = REGIME_DIMENSIONS.filter((dimension) => dimension.id !== "atr");

export const SCORE_WEAK = "#dc2626";
export const SCORE_DECENT = "#f59e0b";
export const SCORE_GOOD = "#4ade80";
export const SCORE_VERY_GOOD = "#15803d";

export function regimeScoreColor(value: number) {
  if (value < 25) return SCORE_WEAK;
  if (value < 50) return SCORE_DECENT;
  if (value < 75) return SCORE_GOOD;
  return SCORE_VERY_GOOD;
}

const REGIME_INDICATOR_PREFIX = "Regime_";

export function isRegimeIndicatorId(id: string) {
  return id.startsWith(REGIME_INDICATOR_PREFIX);
}

const REGIME_STATE_BY_ID = new Map(
  REGIME_DIMENSIONS.flatMap((dimension) =>
    dimension.buckets.map(
      (bucket) => [regimeIndicatorId(dimension.id, bucket.key), { dimension, bucket }] as const,
    ),
  ),
);

/** The regime and state a `Regime_<regime>_<state>` condition id stands for. */
export function parseRegimeIndicatorId(id: string) {
  return REGIME_STATE_BY_ID.get(id) ?? null;
}

/** "VIX · ≤ 15" for a regime condition id, or null for any other id. */
export function regimeConditionLabel(id: string): string | null {
  const parsed = parseRegimeIndicatorId(id);
  return parsed ? regimeIndicatorLabel(parsed.dimension.title, parsed.bucket.label) : null;
}
