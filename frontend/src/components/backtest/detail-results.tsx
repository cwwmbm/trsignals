import { memo } from "react";
import type { DetailedResult } from "@/api";
import { DrawdownCurve } from "@/components/drawdown-curve";
import { EquityCurve } from "@/components/equity-curve";
import { MonteCarloPanel } from "@/components/backtest/monte-carlo-panel";
import { negativeYearRowClass, ResultsTable, tradeRowClass } from "@/components/backtest/results-table";
import { StatCard, SUMMARY_KEYS, summaryLabel, summaryTone } from "@/components/backtest/stat-card";
import { formatMetric } from "@/lib/format-metric";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export const DetailResults = memo(function DetailResults({
  result,
  embedded = false,
}: {
  result: DetailedResult;
  embedded?: boolean;
}) {
  const latestTrades = [...result.trades].reverse().slice(0, 100);
  const latestYears = [...result.yearly].reverse();
  const latestMonths = [...(result.monthly ?? [])].reverse();
  const description = String(result.summary.description ?? "");

  const content = (
    <>
      {!embedded && (
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-semibold">Results</h2>
          <span className="font-mono text-[10px] text-muted-foreground">Detailed</span>
        </div>
      )}

      <div className={cn("grid grid-cols-2 gap-1.5 sm:grid-cols-4 lg:grid-cols-8", !embedded && "mt-2")}>
        {SUMMARY_KEYS.map((key) => (
          <StatCard
            key={key}
            compact
            label={summaryLabel(key)}
            value={formatMetric(result.summary[key], key)}
            tone={summaryTone(key, result.summary[key])}
          />
        ))}
      </div>

      {description ? (
        <p className="mt-2 truncate font-mono text-[11px] text-muted-foreground" title={description}>
          {description}
        </p>
      ) : null}

      <div className="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-3">
        <div className="flex flex-col gap-1">
          <h3 className="text-xs font-medium text-muted-foreground">Yearly breakdown</h3>
          <ResultsTable compact visibleRows={30} rows={latestYears} rowClassName={negativeYearRowClass} sortable={false} />
        </div>
        <div className="flex flex-col gap-1">
          <h3 className="text-xs font-medium text-muted-foreground">Monthly breakdown</h3>
          <ResultsTable compact visibleRows={30} rows={latestMonths} rowClassName={negativeYearRowClass} sortable={false} />
        </div>
        <div className="flex flex-col gap-1">
          <h3 className="text-xs font-medium text-muted-foreground">Trades</h3>
          <ResultsTable compact visibleRows={30} rows={latestTrades} rowClassName={tradeRowClass} sortable={false} />
        </div>
      </div>

      <div className="mt-3 rounded-md border border-border/60 bg-muted/20 p-3">
        <EquityCurve
          data={result.equity_curve}
          totalPoints={result.equity_curve_total_points}
        />
      </div>

      <div className="mt-3 rounded-md border border-border/60 bg-muted/20 p-3">
        <DrawdownCurve
          data={result.equity_curve}
          totalPoints={result.equity_curve_total_points}
        />
      </div>

      <div className="mt-3 rounded-md border border-border/60 bg-muted/20 p-3">
        <MonteCarloPanel trades={result.trades} />
      </div>
    </>
  );

  if (embedded) {
    return content;
  }

  return <Card className="border-border/60 p-3">{content}</Card>;
});
