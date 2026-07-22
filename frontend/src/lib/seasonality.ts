export type SeasonalityTrade = {
  entry_date?: string | number | null;
  trade_pnl?: string | number | null;
  status?: string | number | null;
};

export type SeasonalityBucket = {
  key: string;
  label: string;
  avgPct: number;
  sumPct: number;
  count: number;
  wins: number;
  winRate: number;
};

const WEEKDAYS = [
  { key: "mon", label: "Monday", day: 1 },
  { key: "tue", label: "Tuesday", day: 2 },
  { key: "wed", label: "Wednesday", day: 3 },
  { key: "thu", label: "Thursday", day: 4 },
  { key: "fri", label: "Friday", day: 5 },
] as const;

const MONTHS = [
  { key: "jan", label: "January", month: 0 },
  { key: "feb", label: "February", month: 1 },
  { key: "mar", label: "March", month: 2 },
  { key: "apr", label: "April", month: 3 },
  { key: "may", label: "May", month: 4 },
  { key: "jun", label: "June", month: 5 },
  { key: "jul", label: "July", month: 6 },
  { key: "aug", label: "August", month: 7 },
  { key: "sep", label: "September", month: 8 },
  { key: "oct", label: "October", month: 9 },
  { key: "nov", label: "November", month: 10 },
  { key: "dec", label: "December", month: 11 },
] as const;

const QUARTERS = [
  { key: "q1", label: "Quarter 1", quarter: 1 },
  { key: "q2", label: "Quarter 2", quarter: 2 },
  { key: "q3", label: "Quarter 3", quarter: 3 },
  { key: "q4", label: "Quarter 4", quarter: 4 },
] as const;

function emptyBucket(key: string, label: string): SeasonalityBucket {
  return { key, label, avgPct: 0, sumPct: 0, count: 0, wins: 0, winRate: 0 };
}

function parseEntryDate(value: string | number | null | undefined): Date | null {
  if (value == null) return null;
  const raw = String(value).trim();
  if (!raw || raw === "Open") return null;
  // Prefer date-only parse to avoid timezone shifting calendar day.
  const dateOnly = raw.slice(0, 10);
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateOnly);
  if (match) {
    const year = Number(match[1]);
    const month = Number(match[2]) - 1;
    const day = Number(match[3]);
    return new Date(year, month, day);
  }
  const parsed = new Date(raw);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function closedTradeReturns(trades: SeasonalityTrade[]): Array<{ date: Date; pnlFrac: number }> {
  const rows: Array<{ date: Date; pnlFrac: number }> = [];
  for (const trade of trades) {
    if (String(trade.status ?? "Closed") !== "Closed") continue;
    const date = parseEntryDate(trade.entry_date);
    const pnlFrac = Number(trade.trade_pnl);
    if (!date || !Number.isFinite(pnlFrac)) continue;
    rows.push({ date, pnlFrac });
  }
  return rows;
}

function addToBucket(bucket: SeasonalityBucket, pnlFrac: number) {
  const pct = pnlFrac * 100;
  bucket.sumPct += pct;
  bucket.count += 1;
  if (pnlFrac > 0) bucket.wins += 1;
}

function finalize(buckets: SeasonalityBucket[]): SeasonalityBucket[] {
  return buckets.map((bucket) => ({
    ...bucket,
    avgPct: bucket.count > 0 ? bucket.sumPct / bucket.count : 0,
    winRate: bucket.count > 0 ? (bucket.wins / bucket.count) * 100 : 0,
  }));
}

export function aggregateSeasonality(trades: SeasonalityTrade[]): {
  days: SeasonalityBucket[];
  months: SeasonalityBucket[];
  quarters: SeasonalityBucket[];
} {
  const days = WEEKDAYS.map((item) => emptyBucket(item.key, item.label));
  const months = MONTHS.map((item) => emptyBucket(item.key, item.label));
  const quarters = QUARTERS.map((item) => emptyBucket(item.key, item.label));

  const dayIndex = new Map<number, number>(WEEKDAYS.map((item, index) => [item.day, index]));
  const monthIndex = new Map<number, number>(MONTHS.map((item, index) => [item.month, index]));
  const quarterIndex = new Map<number, number>(QUARTERS.map((item, index) => [item.quarter, index]));

  for (const { date, pnlFrac } of closedTradeReturns(trades)) {
    const weekday = date.getDay(); // 0=Sun … 6=Sat
    const dayIdx = dayIndex.get(weekday);
    if (dayIdx != null) addToBucket(days[dayIdx], pnlFrac);

    const monthIdx = monthIndex.get(date.getMonth());
    if (monthIdx != null) addToBucket(months[monthIdx], pnlFrac);

    const quarter = Math.floor(date.getMonth() / 3) + 1;
    const quarterIdx = quarterIndex.get(quarter);
    if (quarterIdx != null) addToBucket(quarters[quarterIdx], pnlFrac);
  }

  return {
    days: finalize(days),
    months: finalize(months),
    quarters: finalize(quarters),
  };
}
