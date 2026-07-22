import { memo, useState } from "react";
import type { DetailedResult, PortfolioSimulatePayload } from "@/api";
import { ContributionPanel } from "@/components/backtest/contribution-panel";
import { EquityDrawdownPanel } from "@/components/backtest/equity-drawdown-panel";
import { MonteCarloPanel } from "@/components/backtest/monte-carlo-panel";
import { SeasonalityPanel } from "@/components/backtest/seasonality-panel";
import { negativeYearRowClass, ResultsTable, tradeRowClass } from "@/components/backtest/results-table";
import { StatCard, SUMMARY_KEYS, summaryLabel, summaryTone } from "@/components/backtest/stat-card";
import { formatMetric } from "@/lib/format-metric";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

type AnalysisTab = "equity-drawdown" | "seasonality" | "monte-carlo";
type ResultsTab = "overview" | "contribution";

export const DetailResults = memo(function DetailResults({
  result,
  embedded = false,
  shapleyRequest = null,
}: {
  result: DetailedResult;
  embedded?: boolean;
  shapleyRequest?: PortfolioSimulatePayload | null;
}) {
  const [analysisTab, setAnalysisTab] = useState<AnalysisTab>("equity-drawdown");
  const [resultsTab, setResultsTab] = useState<ResultsTab>("overview");
  const latestTrades = [...result.trades].reverse().slice(0, 100);
  const latestYears = [...result.yearly].reverse();
  const latestMonths = [...(result.monthly ?? [])].reverse();
  const description = String(result.summary.description ?? "");
  const contribution = result.contribution ?? null;
  const hasContribution = Array.isArray(contribution) && contribution.length > 0;

  const overview = (
    <>
      <div className={cn("grid grid-cols-2 gap-1.5 sm:grid-cols-4 lg:grid-cols-8", !embedded && !hasContribution && "mt-2")}>
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
        <Tabs
          value={analysisTab}
          onValueChange={(value) => value && setAnalysisTab(value as AnalysisTab)}
        >
          <TabsList
            variant="line"
            className="h-8 w-full justify-start rounded-none border-b border-border/60 bg-transparent p-0"
          >
            <TabsTrigger value="equity-drawdown" className="h-8 rounded-none px-3 text-xs">
              Equity curve / drawdown
            </TabsTrigger>
            <TabsTrigger value="seasonality" className="h-8 rounded-none px-3 text-xs">
              Seasonality
            </TabsTrigger>
            <TabsTrigger value="monte-carlo" className="h-8 rounded-none px-3 text-xs">
              Monte Carlo
            </TabsTrigger>
          </TabsList>

          <TabsContent value="equity-drawdown" className="mt-3">
            {analysisTab === "equity-drawdown" ? (
              <EquityDrawdownPanel
                data={result.equity_curve}
                totalPoints={result.equity_curve_total_points}
              />
            ) : null}
          </TabsContent>
          <TabsContent value="seasonality" className="mt-3">
            {analysisTab === "seasonality" ? <SeasonalityPanel trades={result.trades} /> : null}
          </TabsContent>
          <TabsContent value="monte-carlo" className="mt-3">
            {analysisTab === "monte-carlo" ? <MonteCarloPanel trades={result.trades} /> : null}
          </TabsContent>
        </Tabs>
      </div>
    </>
  );

  const content = (
    <>
      {!embedded && (
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-semibold">Results</h2>
          <span className="font-mono text-[10px] text-muted-foreground">Detailed</span>
        </div>
      )}

      {hasContribution ? (
        <Tabs
          value={resultsTab}
          onValueChange={(value) => value && setResultsTab(value as ResultsTab)}
          className={cn(!embedded && "mt-2")}
        >
          <TabsList
            variant="line"
            className="h-8 w-full justify-start rounded-none border-b border-border/60 bg-transparent p-0"
          >
            <TabsTrigger value="overview" className="h-8 rounded-none px-3 text-xs">
              Overview
            </TabsTrigger>
            <TabsTrigger value="contribution" className="h-8 rounded-none px-3 text-xs">
              Contribution
            </TabsTrigger>
          </TabsList>
          {/*
            Keep both panels in one grid cell so container height stays at the
            taller Overview size when switching tabs (avoids layout jump).
          */}
          <div className="mt-3 grid">
            <div
              className={cn(
                "col-start-1 row-start-1",
                resultsTab !== "overview" && "invisible pointer-events-none",
              )}
              aria-hidden={resultsTab !== "overview"}
            >
              {overview}
            </div>
            <div
              className={cn(
                "col-start-1 row-start-1 flex min-h-0 flex-col",
                resultsTab !== "contribution" && "invisible pointer-events-none",
              )}
              aria-hidden={resultsTab !== "contribution"}
            >
              <ContributionPanel
                rows={contribution}
                shapleyRequest={shapleyRequest}
                className="min-h-0 flex-1"
              />
            </div>
          </div>
        </Tabs>
      ) : (
        overview
      )}
    </>
  );

  if (embedded) {
    return content;
  }

  return <Card className="border-border/60 p-3">{content}</Card>;
});
