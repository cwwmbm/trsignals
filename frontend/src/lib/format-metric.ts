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
