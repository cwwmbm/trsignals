export type RegimeField =
  | "vix_regime"
  | "vxn_regime"
  | "atr_regime"
  | "spy_regime"
  | "breadth_new_regime"
  | "breadth_old_regime"
  | "semis_breadth_regime"
  | "equity_risk_breadth_regime"
  | "credit_risk_breadth_regime"
  | "credit_risk_on_regime"
  | "bond_duration_regime"
  | "copper_gold_regime"
  | "materials_breadth_regime"
  | "sector_breadth_50_regime"
  | "sector_breadth_200_regime";

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
};

export type RegimeYMetric = "sharpe" | "sortino" | "calmar" | "avg_return" | "avg_trade_return" | "max_drawdown";

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
  "breadth_new_regime",
  "breadth_old_regime",
  "semis_breadth_regime",
  "equity_risk_breadth_regime",
  "credit_risk_breadth_regime",
  "credit_risk_on_regime",
  "bond_duration_regime",
  "copper_gold_regime",
  "materials_breadth_regime",
  "sector_breadth_50_regime",
  "sector_breadth_200_regime",
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
    id: "breadth_new",
    title: "New market breadth",
    hint: "RSI(14) of log(RSP / SPY)",
    field: "breadth_new_regime",
    buckets: BREADTH_BUCKETS,
    unavailableWhenEmpty: false,
  },
  {
    id: "breadth_old",
    title: "Old market breadth",
    hint: "RSI(14) of RSP / SPY",
    field: "breadth_old_regime",
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
): RegimeChartPoint[] {
  const rows = new Map((sharpe?.[dimension.id] ?? []).map((row) => [row.key, row]));
  return dimension.buckets.map((bucket) => {
    const row = rows.get(bucket.key);
    const tradeStats = closedTradeStats(trades, dimension.field, bucket.key);
    const enough = tradeStats.trades >= MIN_REGIME_TRADES;
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
    };
  });
}

export function regimeYValue(point: RegimeChartPoint, metric: RegimeYMetric): number | null {
  if (metric === "sharpe") return point.sharpe;
  if (metric === "sortino") return point.sortino;
  if (metric === "calmar") return point.calmar;
  if (metric === "avg_return") return point.avgReturnPct;
  if (metric === "avg_trade_return") return point.avgTradeReturnPct;
  if (point.maxDrawdown == null) return null;
  return -point.maxDrawdown * 100;
}
