import { Card } from "@/components/ui/card";
import { formatMetric } from "@/lib/format-metric";
import { cn } from "@/lib/utils";

export function StatCard({
  label,
  value,
  tone = "neutral",
  compact = false,
}: {
  label: string;
  value: string;
  tone?: "neutral" | "gain" | "loss";
  compact?: boolean;
}) {
  return (
    <Card
      className={cn(
        "gap-0.5 rounded-md border-border/60 bg-card/60",
        compact ? "p-2" : "gap-1 rounded-lg p-4",
      )}
    >
      <span
        className={cn(
          "font-medium uppercase tracking-wider text-muted-foreground",
          compact ? "text-[10px]" : "text-[11px]",
        )}
      >
        {label}
      </span>
      <span
        className={cn(
          "font-mono font-semibold tabular-nums",
          compact ? "text-sm" : "text-xl",
          tone === "gain" && "text-[var(--gain)]",
          tone === "loss" && "text-[var(--loss)]",
        )}
      >
        {value}
      </span>
    </Card>
  );
}

type Summary = Record<string, number | string | null>;

export type SummaryStat = {
  label: string;
  value: string;
  tone: "neutral" | "gain" | "loss";
};

function summaryTone(key: string, value: unknown): "neutral" | "gain" | "loss" {
  if (typeof value !== "number") return "neutral";
  if (key === "max_drawdown") return "loss";
  if (key === "rolling_pnl" || key === "cagr_percent") return value >= 0 ? "gain" : "loss";
  return "neutral";
}

function bestYearStat(summary: Summary): SummaryStat | null {
  const year = summary.full_best_year;
  if (year === undefined || year === null) return null;
  const share = summary.full_best_year_share;
  const value =
    typeof share === "number" ? `${Math.round(share * 100)}% (${year})` : String(year);
  return { label: "best year share", value, tone: "neutral" };
}

/**
 * The cards under a result. CAGR, Sharpe and Sortino leave out the best year (what sweeps rank
 * on). Max drawdown covers every year, so the worst fall shows whichever year it happened in.
 */
export function summaryStats(summary: Summary): SummaryStat[] {
  const stat = (label: string, key: string, value: unknown = summary[key]): SummaryStat => ({
    label,
    value: formatMetric(value, key),
    tone: summaryTone(key, value),
  });
  const worstDrawdown = summary.full_max_drawdown ?? summary.max_drawdown;
  const stats = [
    stat("rolling pnl", "rolling_pnl"),
    stat("cagr percent", "cagr_percent"),
    stat("sharpe", "sharpe"),
    stat("sortino", "sortino"),
    stat("max drawdown", "max_drawdown", worstDrawdown),
    stat("trades", "trades"),
    stat("pct positive", "pct_positive"),
  ];
  const best = bestYearStat(summary);
  if (best) stats.push(best);
  return stats;
}

/**
 * What the ratios above would be with the best year left in. Null when no year was left out or
 * the result predates the full-period figures.
 */
export function excludedYearNote(summary: Summary): string | null {
  const year = summary.excluded_year;
  if (year === undefined || year === null || summary.full_cagr_percent === undefined) return null;
  const best = summary.full_best_year_return;
  const bestText = typeof best === "number" ? `, ${best >= 0 ? "+" : ""}${(best * 100).toFixed(0)}%` : "";
  const drawdownText =
    typeof summary.max_drawdown === "number" &&
    typeof summary.full_max_drawdown === "number" &&
    summary.max_drawdown < summary.full_max_drawdown
      ? ` Max drawdown covers every year; without ${year} it is ${formatMetric(summary.max_drawdown, "max_drawdown")}.`
      : " Max drawdown covers every year.";
  return (
    `CAGR, Sharpe and Sortino leave out ${year} (best year${bestText}). ` +
    `With it: CAGR ${formatMetric(summary.full_cagr_percent, "cagr_percent")}, ` +
    `Sharpe ${formatMetric(summary.full_sharpe, "sharpe")}, ` +
    `Sortino ${formatMetric(summary.full_sortino, "sortino")}.` +
    drawdownText
  );
}
