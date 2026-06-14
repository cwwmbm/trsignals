import type { DetailedResult } from "@/api";
import { EquityCurve } from "@/components/equity-curve";
import { negativeYearRowClass, ResultsTable, tradeRowClass } from "@/components/backtest/results-table";
import { StatCard, SUMMARY_KEYS, summaryLabel, summaryTone } from "@/components/backtest/stat-card";
import { formatMetric } from "@/lib/format-metric";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

export function DetailResults({ result }: { result: DetailedResult }) {
  const latestTrades = [...result.trades].reverse().slice(0, 100);
  const latestYears = [...result.yearly].reverse();

  return (
    <Card className="border-border/60 p-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">Results</h2>
        <Badge variant="outline" className="font-mono text-xs">
          Detailed
        </Badge>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-8">
        {SUMMARY_KEYS.map((key) => (
          <StatCard
            key={key}
            label={summaryLabel(key)}
            value={formatMetric(result.summary[key], key)}
            tone={summaryTone(key, result.summary[key])}
          />
        ))}
      </div>

      <p className="mt-4 font-mono text-sm text-muted-foreground">
        {String(result.summary.description ?? "")}
      </p>

      <div className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-2">
        <div className="flex flex-col gap-3">
          <h3 className="text-sm font-semibold">Yearly Breakdown</h3>
          <ResultsTable rows={latestYears} rowClassName={negativeYearRowClass} />
        </div>
        <div className="flex flex-col gap-3">
          <h3 className="text-sm font-semibold">Trades</h3>
          <ResultsTable rows={latestTrades} rowClassName={tradeRowClass} />
        </div>
      </div>

      <div className="mt-6 rounded-lg border border-border/60 bg-muted/20 p-5">
        <EquityCurve data={result.equity_curve} />
      </div>
    </Card>
  );
}
