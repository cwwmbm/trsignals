export function formatMetric(value: unknown, column?: string) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const integerColumns = new Set([
    "year",
    "num_trades",
    "positive_trades",
    "days_in_trade",
    "Days",
    "Profit",
    "Trades",
    "Prf",
  ]);
  const percentColumns = new Set(["pnl_percent", "drawdown_percent", "cagr_percent", "pct_positive"]);
  const fractionalPercentColumns = new Set(["trade_pnl", "max_drawdown", "drawdown"]);
  if (typeof value === "number") {
    if (column && integerColumns.has(column)) return String(Math.trunc(value));
    if (column && fractionalPercentColumns.has(column)) return `${(value * 100).toFixed(2)}%`;
    if (column && percentColumns.has(column)) return `${value.toFixed(2)}%`;
    if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
    return value.toFixed(2);
  }
  return String(value);
}

/** Format a percentage-point delta, e.g. +2.4 pp / −4.1 pp. */
export function formatPpDelta(value: unknown, digits = 1): string {
  if (value === null || value === undefined || typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  const sign = value > 0 ? "+" : value < 0 ? "\u2212" : "";
  const abs = Math.abs(value).toFixed(digits);
  return `${sign}${abs} pp`;
}

/** Format a signed numeric delta (ratios, utility, etc.). */
export function formatSignedDelta(value: unknown, digits = 2): string {
  if (value === null || value === undefined || typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  const sign = value > 0 ? "+" : value < 0 ? "\u2212" : "";
  return `${sign}${Math.abs(value).toFixed(digits)}`;
}

/** Format a holding-share fraction (0–1) as a percent. */
export function formatHoldingPercent(value: unknown, digits = 0): string {
  if (value === null || value === undefined || typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  return `${(value * 100).toFixed(digits)}%`;
}
