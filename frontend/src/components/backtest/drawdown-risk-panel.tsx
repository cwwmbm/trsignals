import type { DrawdownRisk } from "@/api";
import { StatCard } from "@/components/backtest/stat-card";
import { cn } from "@/lib/utils";

export function riskLabel(threshold: number) {
  return `P(DD > ${Math.round(threshold * 100)}%)`;
}

function percent(value: number, digits = 0) {
  return `${(value * 100).toFixed(digits)}%`;
}

/** Share of resampled paths whose worst drawdown passed a depth. Small shares keep a decimal. */
export function formatRisk(probability: number) {
  const pct = probability * 100;
  if (pct === 0) return "0%";
  if (pct < 1) return "<1%";
  return `${pct.toFixed(pct < 10 ? 1 : 0)}%`;
}

export function riskTone(probability: number): "neutral" | "gain" | "loss" {
  if (probability >= 0.5) return "loss";
  if (probability <= 0.1) return "gain";
  return "neutral";
}

export function DrawdownRiskPanel({ risk, className }: { risk: DrawdownRisk; className?: string }) {
  const years = risk.horizon_bars / (risk.periods_per_year || 252);
  const worstError = Math.max(...risk.thresholds.map((row) => row.std_error));
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h3 className="text-xs font-medium text-muted-foreground">Drawdown risk</h3>
        <span className="text-[11px] text-muted-foreground">
          Median worst drawdown {percent(risk.median_max_drawdown, 1)} · 95th percentile{" "}
          {percent(risk.p95_max_drawdown, 1)} · this backtest {percent(risk.historical_max_drawdown, 1)}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-1.5 sm:max-w-md">
        {risk.thresholds.map((row) => (
          <StatCard
            key={row.drawdown}
            compact
            label={riskLabel(row.drawdown)}
            value={formatRisk(row.probability)}
            tone={riskTone(row.probability)}
          />
        ))}
      </div>
      <p className="text-[11px] text-muted-foreground">
        Chance that the worst fall from a peak, over {years.toFixed(1)} years, goes beyond that depth. From{" "}
        {risk.paths.toLocaleString()} paths made by reshuffling this backtest&apos;s daily returns in{" "}
        {risk.block_days}-day blocks (±{(worstError * 100).toFixed(1)} points at most). It resamples this history, so
        it is a way to compare rules, not a forecast.
      </p>
    </div>
  );
}
