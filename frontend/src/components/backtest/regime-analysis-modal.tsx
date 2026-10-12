import { useQuery } from "@tanstack/react-query";
import { Loader2, X } from "lucide-react";
import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  fetchRegimeStrategyAnalysis,
  type RegimeGranularState,
  type RegimeSource,
  type RegimeStrategyAnalysis,
} from "@/api";
import { formatProfitablePct, RegimeYearTable } from "@/components/backtest/regime-year-table";
import { Button } from "@/components/ui/button";
import { ChartContainer, type ChartConfig } from "@/components/ui/chart";
import { MIN_REGIME_TRADES, regimeScoreColor, type RegimeDimension } from "@/lib/market-regimes";

const chartConfig = { score: { label: "Regime score", color: "var(--chart-1)" } } satisfies ChartConfig;

function formatScore(value: number | null | undefined) {
  return value == null ? "—" : value.toFixed(1);
}

function formatPct(value: number | null | undefined, digits = 1) {
  return value == null ? "—" : `${(value * 100).toFixed(digits)}%`;
}

function formatRatio(value: number | null | undefined) {
  return value == null ? "—" : value.toFixed(2);
}

/** Same formats as the Market regimes tab: drawdown as a plain percent, averages signed. */
function formatDrawdown(value: number | null | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(2)}%`;
}

function formatSignedPct(value: number | null | undefined) {
  if (value == null) return "—";
  const pct = value * 100;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(2)}%`;
}

type ChartRow = RegimeGranularState & { score: number | null; sparse: boolean };

function GranularTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload?: ChartRow }> }) {
  const row = payload?.[0]?.payload;
  if (!active || !row) return null;
  return (
    <div className="grid min-w-40 gap-0.5 rounded-lg border border-border/50 bg-background px-2.5 py-1.5 text-xs shadow-xl">
      <div className="mb-0.5 font-medium">{row.label}</div>
      <div>Regime score {formatScore(row.regime_score)}</div>
      <div className="pl-2 text-muted-foreground">Sortino score {formatScore(row.regime_score_sortino)}</div>
      <div className="pl-2 text-muted-foreground">Return score {formatScore(row.regime_score_return)}</div>
      <div className="pl-2 text-muted-foreground">Drawdown score {formatScore(row.regime_score_drawdown)}</div>
      <div className="pl-2 text-muted-foreground">Robustness score {formatScore(row.regime_score_robustness)}</div>
      <div>{row.days} days in this state</div>
      <div>
        {row.trades} trades, {formatProfitablePct(row.profitable, row.trades)} profitable
      </div>
      <div>Sharpe {formatRatio(row.sharpe)}</div>
      <div>Sortino {formatRatio(row.sortino)}</div>
      <div>Avg / exposure day {formatSignedPct(row.exposure)}</div>
      <div>Avg / trade {formatSignedPct(row.avg_trade_return)}</div>
      <div>CAGR {formatPct(row.cagr)}</div>
      <div>Max drawdown {formatPct(row.max_drawdown)}</div>
      {row.sparse ? (
        <div className="text-muted-foreground">Under {MIN_REGIME_TRADES} trades, read with care</div>
      ) : null}
    </div>
  );
}

function GranularSection({ analysis }: { analysis: RegimeStrategyAnalysis }) {
  const { granular } = analysis;
  const rows: ChartRow[] = granular.states.map((state) => ({
    ...state,
    score: state.regime_score ?? null,
    sparse: state.trades < MIN_REGIME_TRADES,
  }));
  const angled = rows.length >= 5;
  const base = granular.base_score;
  return (
    <section className="flex flex-col gap-2">
      <div>
        <h4 className="text-xs font-semibold">
          {granular.refined ? `${rows.length} finer states` : "States"} · regime score
        </h4>
        <p className="text-[11px] text-muted-foreground">
          {granular.refined
            ? "The regime split into finer slices, scored on trades entered in each slice. "
            : "No finer split is defined for this regime, so these are its standard states. "}
          Faded bars have under {MIN_REGIME_TRADES} trades.
          {base != null ? ` The dashed line is the score across all trades (${base.toFixed(1)}).` : ""}
        </p>
      </div>
      <ChartContainer config={chartConfig} className="h-[260px] w-full">
        <BarChart data={rows} margin={{ left: 4, right: 12, top: 12, bottom: angled ? 40 : 8 }}>
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          <XAxis
            dataKey="label"
            tickLine={false}
            axisLine={false}
            interval={0}
            tickMargin={angled ? 2 : 10}
            tick={{ fontSize: 10, angle: angled ? -35 : 0, textAnchor: angled ? "end" : "middle" } as object}
          />
          <YAxis
            domain={[0, 100]}
            tickLine={false}
            axisLine={false}
            width={36}
            tickMargin={8}
            tick={{ fontSize: 10 }}
          />
          {base != null ? (
            <ReferenceLine y={base} stroke="var(--muted-foreground)" strokeDasharray="4 3" strokeOpacity={0.8} />
          ) : null}
          <Tooltip cursor={false} isAnimationActive={false} content={<GranularTooltip />} />
          <Bar dataKey="score" radius={[3, 3, 0, 0]} maxBarSize={72} isAnimationActive={false}>
            {rows.map((row) => (
              <Cell
                key={row.key}
                fill={row.score == null ? "transparent" : regimeScoreColor(row.score)}
                fillOpacity={row.sparse ? 0.35 : 0.9}
              />
            ))}
          </Bar>
        </BarChart>
      </ChartContainer>
      <div className="overflow-x-auto">
        <table className="w-full text-xs tabular-nums">
          <thead>
            <tr className="border-b text-[11px] text-muted-foreground">
              <th className="px-2 py-1 text-left font-medium">State</th>
              <th className="px-2 py-1 text-right font-medium">Days</th>
              <th className="px-2 py-1 text-right font-medium">Trades</th>
              <th className="px-2 py-1 text-right font-medium">% Profitable</th>
              <th className="px-2 py-1 text-right font-medium">Regime score</th>
              <th className="px-2 py-1 text-right font-medium">Sharpe</th>
              <th className="px-2 py-1 text-right font-medium">Sortino</th>
              <th className="px-2 py-1 text-right font-medium">MaxDD</th>
              <th className="px-2 py-1 text-right font-medium">Avg % / exposure day</th>
              <th className="px-2 py-1 text-right font-medium">Avg % / trade</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key} className="border-b border-border/40 last:border-0">
                <td className="px-2 py-1 whitespace-nowrap">{row.label}</td>
                <td className="px-2 py-1 text-right">{row.days}</td>
                <td className="px-2 py-1 text-right">{row.trades}</td>
                <td className="px-2 py-1 text-right">{formatProfitablePct(row.profitable, row.trades)}</td>
                <td className="px-2 py-1 text-right">{formatScore(row.regime_score)}</td>
                <td className="px-2 py-1 text-right">{formatRatio(row.sharpe)}</td>
                <td className="px-2 py-1 text-right">{formatRatio(row.sortino)}</td>
                <td className="px-2 py-1 text-right">{formatDrawdown(row.max_drawdown)}</td>
                <td className="px-2 py-1 text-right">{formatSignedPct(row.exposure)}</td>
                <td className="px-2 py-1 text-right">{formatSignedPct(row.avg_trade_return)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

type StatesView = "broad" | "granular";

function StatesToggle({ value, onChange }: { value: StatesView; onChange: (value: StatesView) => void }) {
  const options: { id: StatesView; label: string }[] = [
    { id: "broad", label: "Broad states" },
    { id: "granular", label: "Granular states" },
  ];
  return (
    <div role="group" aria-label="State detail" className="inline-flex w-fit rounded-md border border-border/60 p-0.5">
      {options.map((option) => (
        <Button
          key={option.id}
          type="button"
          size="xs"
          variant={value === option.id ? "secondary" : "ghost"}
          aria-pressed={value === option.id}
          onClick={() => onChange(option.id)}
        >
          {option.label}
        </Button>
      ))}
    </div>
  );
}

export function RegimeAnalysisModal({
  dimension,
  source,
  onClose,
}: {
  dimension: RegimeDimension;
  source: RegimeSource;
  onClose: () => void;
}) {
  const query = useQuery({
    queryKey: ["regime-strategy-analysis", source.path, JSON.stringify(source.payload), dimension.id],
    queryFn: () => fetchRegimeStrategyAnalysis(source, dimension.id),
    staleTime: Infinity,
    retry: false,
  });

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const [view, setView] = useState<StatesView>("broad");
  const analysis = query.data;
  const showGranular = view === "granular" && analysis?.granular.refined === true;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/70 p-4"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-labelledby="regime-analysis-title"
        className="flex max-h-[90vh] w-full max-w-7xl flex-col rounded-lg border border-border bg-background shadow-xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3 border-b border-border/60 px-4 py-3">
          <div className="min-w-0">
            <h3 id="regime-analysis-title" className="text-sm font-semibold">
              {dimension.title} · regime analysis
            </h3>
            <p className="text-[11px] text-muted-foreground">{dimension.hint}</p>
          </div>
          <Button type="button" variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close">
            <X className="size-4" />
          </Button>
        </div>

        <div className="flex flex-col gap-5 overflow-y-auto px-4 py-3">
          {query.isPending ? (
            <p className="flex items-center gap-2 py-8 text-xs text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              Analyzing {dimension.title}...
            </p>
          ) : null}
          {query.error ? (
            <pre className="whitespace-pre-wrap text-xs text-destructive">{String(query.error)}</pre>
          ) : null}
          {analysis ? (
            <>
              <section className="flex flex-col gap-2">
                <p className="text-[11px] text-muted-foreground">
                  {analysis.period.start} to {analysis.period.end}. Days are sessions each state was active.
                  Trades are counted in the state on their entry day and the year of that day. Profitable
                  means a closed trade with a positive return. CAGR is the strategy's return for the year.
                </p>
                {analysis.granular.refined ? <StatesToggle value={view} onChange={setView} /> : null}
                <RegimeYearTable
                  states={showGranular ? analysis.granular.states : analysis.states}
                  rows={showGranular ? analysis.granular.rows : analysis.rows}
                />
              </section>
              <GranularSection analysis={analysis} />
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
