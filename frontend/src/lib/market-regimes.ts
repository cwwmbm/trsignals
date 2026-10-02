export type RegimeField =
  | "vix_regime"
  | "vxn_regime"
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
};

export type RegimeTradeRow = Record<string, number | string | null>;

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
};

export type RegimeYMetric =
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

const CURVE_CHANGE_BUCKETS: RegimeBucketDef[] = [
  { key: "le_neg_50", label: "≤ -50 bp" },
  { key: "neg_50_neg_10", label: "-50 to -10" },
  { key: "neg_10_pos_10", label: "-10 to +10" },
  { key: "pos_10_pos_50", label: "+10 to +50" },
  { key: "ge_pos_50", label: "≥ +50 bp" },
];

const BREADTH_BUCKETS: RegimeBucketDef[] = [
  { key: "lt_40", label: "< 40" },
  { key: "40_50", label: "40–50" },
  { key: "50_60", label: "50–60" },
  { key: "gt_60", label: "> 60" },
];

export const REGIME_TRADE_FIELDS: RegimeField[] = [
  "vix_regime",
  "vxn_regime",
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
    hint: "RSI(14) of RSP / SPY",
    field: "market_breadth_regime",
    buckets: BREADTH_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "semis_breadth",
    title: "Semis breadth",
    hint: "RSI(14) of log(SMH / SPY)",
    field: "semis_breadth_regime",
    buckets: BREADTH_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "equity_risk_breadth",
    title: "Equity Risk Breadth",
    hint: "RSI(14) of log(XLY / XLP)",
    field: "equity_risk_breadth_regime",
    buckets: BREADTH_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "credit_risk_breadth",
    title: "Credit Quality Breadth",
    hint: "RSI(14) of log(HYG / LQD)",
    field: "credit_risk_breadth_regime",
    buckets: BREADTH_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "credit_risk_on",
    title: "Credit Risk On",
    hint: "RSI(14) of log(HYG / TLT)",
    field: "credit_risk_on_regime",
    buckets: BREADTH_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "bond_duration",
    title: "Bond Duration Regime",
    hint: "RSI(14) of log(TLT / SHY)",
    field: "bond_duration_regime",
    buckets: BREADTH_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "copper_gold",
    title: "Copper/Gold",
    hint: "RSI(14) of log(HG=F / GC=F)",
    field: "copper_gold_regime",
    buckets: BREADTH_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "materials_breadth",
    title: "Sensitive Materials Breadth",
    hint: "RSI(14) of log(XLB / SPY)",
    field: "materials_breadth_regime",
    buckets: BREADTH_BUCKETS,
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

const MIN_REGIME_TRADES = 20;

function closedTradeStats(trades: RegimeTradeRow[], field: RegimeField, key: string) {
  let count = 0;
  let wins = 0;
  let tradePnl = 0;
  let exposurePnl = 0;
  let exposureDays = 0;
  for (const trade of trades) {
    if (String(trade.status ?? "Closed") !== "Closed") continue;
    if (trade[field] !== key) continue;
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
  if (metric === "regime_score") return point.regimeScore;
  if (metric === "sharpe") return point.sharpe;
  if (metric === "sortino") return point.sortino;
  if (metric === "calmar") return point.calmar;
  if (metric === "avg_return") return point.avgReturnPct;
  if (metric === "avg_trade_return") return point.avgTradeReturnPct;
  if (point.maxDrawdown == null) return null;
  return -point.maxDrawdown * 100;
}
