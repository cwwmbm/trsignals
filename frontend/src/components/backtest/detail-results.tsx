import { useMutation } from "@tanstack/react-query";
import { memo, useState } from "react";
import {
  fetchRegimeOverview,
  type DetailedResult,
  type PortfolioSimulatePayload,
  type RegimeOverview,
  type RegimeSource,
} from "@/api";
import { AllocationPanel } from "@/components/backtest/allocation-panel";
import { BenchmarkComparison } from "@/components/backtest/benchmark-comparison";
import { ContributionPanel } from "@/components/backtest/contribution-panel";
import { DrawdownRiskPanel } from "@/components/backtest/drawdown-risk-panel";
import { EquityDrawdownPanel } from "@/components/backtest/equity-drawdown-panel";
import { MarketRegimesPanel } from "@/components/backtest/market-regimes-panel";
import { MonteCarloPanel } from "@/components/backtest/monte-carlo-panel";
import { PositionEpisodesTable } from "@/components/backtest/position-episodes-table";
import { SeasonalityPanel } from "@/components/backtest/seasonality-panel";
import { negativeYearRowClass, ResultsTable, tradeRowClass } from "@/components/backtest/results-table";
import { excludedYearNote, StatCard, summaryStats } from "@/components/backtest/stat-card";
import { REGIME_TRADE_FIELDS } from "@/lib/market-regimes";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

type AnalysisTab = "equity-drawdown" | "seasonality" | "market-regimes" | "monte-carlo";
type ResultsTab = "overview" | "contribution" | "allocation";

export const DetailResults = memo(function DetailResults({
  result,
  embedded = false,
  shapleyRequest = null,
  regimeSource = null,
}: {
  result: DetailedResult;
  embedded?: boolean;
  shapleyRequest?: PortfolioSimulatePayload | null;
  /** The request that produced `result`. Regimes are computed from it, and only on demand. */
  regimeSource?: RegimeSource | null;
}) {
  const [analysisTab, setAnalysisTab] = useState<AnalysisTab>("equity-drawdown");
  const [resultsTab, setResultsTab] = useState<ResultsTab>("overview");
  // Held here, not in the tab, so switching tabs keeps the calculation. Tied to the
  // result it was computed for, so a new backtest starts uncalculated.
  const [regimeState, setRegimeState] = useState<{ result: DetailedResult; overview: RegimeOverview } | null>(
    null,
  );
  const regimeOverview = regimeState?.result === result ? regimeState.overview : null;
  const regimeMutation = useMutation({
    mutationFn: ({ source }: { source: RegimeSource; forResult: DetailedResult }) =>
      fetchRegimeOverview(source),
    onSuccess: (overview, { forResult }) => setRegimeState({ result: forResult, overview }),
  });
  const calculatingRegimes = regimeMutation.isPending && regimeMutation.variables?.forResult === result;
  const latestTrades = [...result.trades].reverse().slice(0, 100);
  const latestYears = [...result.yearly].reverse();
  const latestMonths = [...(result.monthly ?? [])].reverse();
  const description = String(result.summary.description ?? "");
  const contribution = result.contribution ?? null;
  const hasContribution = Array.isArray(contribution) && contribution.length > 0;
  const allocation = result.allocation ?? null;
  const positions = result.positions ?? null;
  const hasTabs = hasContribution || allocation !== null;

  const overview = (
    <>
      <div className={cn("grid grid-cols-2 gap-1.5 sm:grid-cols-4 lg:grid-cols-8", !embedded && !hasTabs && "mt-2")}>
        {summaryStats(result.summary).map((stat) => (
          <StatCard key={stat.label} compact label={stat.label} value={stat.value} tone={stat.tone} />
        ))}
      </div>
      {excludedYearNote(result.summary) ? (
        <p className="mt-1 text-[11px] text-muted-foreground">{excludedYearNote(result.summary)}</p>
      ) : null}

      {result.benchmark && result.benchmark.series.length > 0 ? (
        <BenchmarkComparison summary={result.summary} curve={result.equity_curve} benchmark={result.benchmark} />
      ) : null}

      {description ? (
        <p className="mt-2 truncate font-mono text-[11px] text-muted-foreground" title={description}>
          {description}
        </p>
      ) : null}
      {result.vehicle ? (
        <p className="mt-1 text-[11px] text-muted-foreground">
          {result.vehicle.leverage_symbol == null
            ? `${result.vehicle.symbol} is traded itself at ${result.vehicle.leverage}x, with no separate leveraged product.`
            : result.vehicle.available
            ? `${result.vehicle.symbol} is traded as ${result.vehicle.leverage_symbol} from ${result.vehicle.first_date}: ` +
              `${result.vehicle.trades_on_leverage_symbol} trades in ${result.vehicle.leverage_symbol}, ` +
              `${result.vehicle.trades_on_stand_in} in ${result.vehicle.symbol} × ${result.vehicle.leverage} (opened before then).`
            : `${result.vehicle.leverage_symbol} has no price history, so ${result.vehicle.symbol} × ${result.vehicle.leverage} stood in for every trade.`}
        </p>
      ) : null}
      {typeof result.summary.metrics_start === "string" && result.summary.metrics_start ? (
        <p className="mt-1 text-[11px] text-muted-foreground">
          Performance starts {result.summary.metrics_start}, the first day the entry conditions could be evaluated.
        </p>
      ) : null}

      {result.drawdown_risk ? <DrawdownRiskPanel risk={result.drawdown_risk} className="mt-3" /> : null}

      <div
        className={cn(
          "mt-3 grid grid-cols-1 gap-3",
          positions ? "xl:grid-cols-2" : "xl:grid-cols-3",
        )}
      >
        <div className="flex flex-col gap-1">
          <h3 className="text-xs font-medium text-muted-foreground">Yearly breakdown</h3>
          <ResultsTable compact visibleRows={30} rows={latestYears} rowClassName={negativeYearRowClass} sortable={false} />
        </div>
        <div className="flex flex-col gap-1">
          <h3 className="text-xs font-medium text-muted-foreground">Monthly breakdown</h3>
          <ResultsTable compact visibleRows={30} rows={latestMonths} rowClassName={negativeYearRowClass} sortable={false} />
        </div>
        {positions ? null : (
          <div className="flex flex-col gap-1">
            <h3 className="text-xs font-medium text-muted-foreground">Trades</h3>
            <ResultsTable
              compact
              visibleRows={30}
              rows={latestTrades}
              hiddenColumns={REGIME_TRADE_FIELDS}
              rowClassName={tradeRowClass}
              sortable={false}
            />
          </div>
        )}
      </div>

      {positions ? <PositionEpisodesTable positions={positions} className="mt-3" /> : null}

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
            <TabsTrigger value="market-regimes" className="h-8 rounded-none px-3 text-xs">
              Market regimes
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
                benchmark={result.benchmark}
              />
            ) : null}
          </TabsContent>
          <TabsContent value="seasonality" className="mt-3">
            {analysisTab === "seasonality" ? <SeasonalityPanel trades={result.trades} /> : null}
          </TabsContent>
          <TabsContent value="market-regimes" className="mt-3">
            {analysisTab === "market-regimes" ? (
              <MarketRegimesPanel
                overview={regimeOverview}
                source={regimeSource}
                isCalculating={calculatingRegimes}
                error={regimeMutation.variables?.forResult === result ? regimeMutation.error : null}
                onCalculate={() => {
                  if (regimeSource) regimeMutation.mutate({ source: regimeSource, forResult: result });
                }}
              />
            ) : null}
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

      {hasTabs ? (
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
            {hasContribution ? (
              <TabsTrigger value="contribution" className="h-8 rounded-none px-3 text-xs">
                Contribution
              </TabsTrigger>
            ) : null}
            {allocation ? (
              <TabsTrigger value="allocation" className="h-8 rounded-none px-3 text-xs">
                Allocation
              </TabsTrigger>
            ) : null}
          </TabsList>
          {/*
            Keep both panels in one grid cell so container height stays at the
            taller Overview size when switching tabs (avoids layout jump).
          */}
          <div className="mt-3 grid grid-cols-[minmax(0,1fr)]">
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
              {hasContribution ? (
                <ContributionPanel
                  rows={contribution}
                  shapleyRequest={shapleyRequest}
                  className="min-h-0 flex-1"
                />
              ) : null}
            </div>
            {allocation ? (
              <div
                className={cn(
                  "col-start-1 row-start-1 flex min-h-0 flex-col",
                  resultsTab !== "allocation" && "invisible pointer-events-none",
                )}
                aria-hidden={resultsTab !== "allocation"}
              >
                <AllocationPanel allocation={allocation} request={shapleyRequest} />
              </div>
            ) : null}
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
