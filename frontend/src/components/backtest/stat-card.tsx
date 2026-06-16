import { Card } from "@/components/ui/card";
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

export const SUMMARY_KEYS = [
  "rolling_pnl",
  "cagr_percent",
  "sharpe",
  "sortino",
  "max_drawdown",
  "trades",
  "pct_positive",
  "excluded_year",
] as const;

export function summaryLabel(key: string) {
  return key.replace(/_/g, " ");
}

export function summaryTone(key: string, value: unknown): "neutral" | "gain" | "loss" {
  if (typeof value !== "number") return "neutral";
  if (key === "max_drawdown") return "loss";
  if (key === "rolling_pnl" || key === "cagr_percent") return value >= 0 ? "gain" : "loss";
  return "neutral";
}
