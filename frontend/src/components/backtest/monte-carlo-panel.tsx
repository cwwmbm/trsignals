import { useEffect, useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  Scatter,
  XAxis,
  YAxis,
} from "recharts";
import {
  runMonteCarlo,
  type DetailedResult,
  type MonteCarloMethod,
  type MonteCarloResult,
} from "@/api";
import { StatCard } from "@/components/backtest/stat-card";
import { Button } from "@/components/ui/button";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { formatMetric } from "@/lib/format-metric";

const chartConfig = {
  count: { label: "Drawdown % PMF", color: "var(--chart-1)" },
  cumulative_pct: { label: "Cumulative drawdowns", color: "var(--chart-3)" },
  confidence: { label: "95% confidence", color: "var(--chart-2)" },
} satisfies ChartConfig;

const DEFAULT_N_SIMS = 1000;

function closedTradeReturns(trades: DetailedResult["trades"]): number[] {
  return trades
    .filter((trade) => String(trade.status ?? "Closed") === "Closed")
    .map((trade) => Number(trade.trade_pnl))
    .filter((value) => Number.isFinite(value));
}

function methodBlurb(method: MonteCarloMethod) {
  if (method === "shuffle") {
    return "Permute closed-trade returns (no replacement) to test sequence / path luck.";
  }
  return "Resample closed-trade returns with replacement to test sampling uncertainty.";
}

function ConfidenceMark(props: {
  cx?: number;
  cy?: number;
  fill?: string;
}) {
  const { cx = 0, cy = 0, fill = "var(--chart-2)" } = props;
  const size = 7;
  return (
    <g>
      <line x1={cx - size} y1={cy - size} x2={cx + size} y2={cy + size} stroke={fill} strokeWidth={2.5} />
      <line x1={cx - size} y1={cy + size} x2={cx + size} y2={cy - size} stroke={fill} strokeWidth={2.5} />
    </g>
  );
}

export function MonteCarloPanel({ trades }: { trades: DetailedResult["trades"] }) {
  const [method, setMethod] = useState<MonteCarloMethod>("shuffle");
  const [result, setResult] = useState<MonteCarloResult | null>(null);

  const tradeReturns = useMemo(() => closedTradeReturns(trades), [trades]);
  const canRun = tradeReturns.length >= 2;

  const mutation = useMutation({
    mutationFn: () =>
      runMonteCarlo({
        trade_returns: tradeReturns,
        method,
        n_sims: DEFAULT_N_SIMS,
      }),
    onSuccess: (payload) => setResult(payload),
  });

  useEffect(() => {
    setResult(null);
  }, [trades]);

  const chartData = useMemo(() => {
    if (!result) return [];
    return result.drawdown_distribution.map((bin) => ({
      drawdown_pct: Number(bin.drawdown_pct.toFixed(2)),
      count: bin.count,
      cumulative_pct: Number(bin.cumulative_pct.toFixed(2)),
    }));
  }, [result]);

  const confidencePoints = useMemo(() => {
    if (!result) return [];
    return [
      {
        drawdown_pct: Number(result.confidence_marker.drawdown_pct.toFixed(2)),
        cumulative_pct: result.confidence_marker.cumulative_pct,
      },
    ];
  }, [result]);

  const maxCount = useMemo(
    () => Math.max(1, ...chartData.map((row) => row.count)),
    [chartData],
  );

  function handleMethodChange(next: MonteCarloMethod) {
    setMethod(next);
    setResult(null);
    mutation.reset();
  }

  const actualDdPct = result ? result.actual.max_drawdown * 100 : null;
  const confidenceDd = result?.confidence_marker.drawdown_pct;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold">Monte Carlo</h3>
          <p className="text-xs text-muted-foreground">{methodBlurb(method)}</p>
          <p className="mt-1 font-mono text-[10px] text-muted-foreground">
            {tradeReturns.length} closed trades
            {!canRun ? " · need at least 2 to run" : null}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <RadioGroup
            value={method}
            onValueChange={(value) => handleMethodChange(value as MonteCarloMethod)}
            className="flex flex-row gap-4"
          >
            <div className="flex items-center gap-2">
              <RadioGroupItem value="shuffle" id="mc-shuffle" />
              <Label htmlFor="mc-shuffle" className="text-xs font-normal">
                Shuffle
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <RadioGroupItem value="bootstrap" id="mc-bootstrap" />
              <Label htmlFor="mc-bootstrap" className="text-xs font-normal">
                Bootstrap
              </Label>
            </div>
          </RadioGroup>
          <Button
            size="sm"
            disabled={!canRun || mutation.isPending}
            onClick={() => mutation.mutate()}
          >
            {mutation.isPending ? "Running…" : "Run Monte Carlo"}
          </Button>
        </div>
      </div>

      {mutation.isError ? (
        <p className="text-xs text-[var(--loss)]">
          {mutation.error instanceof Error ? mutation.error.message : "Monte Carlo failed"}
        </p>
      ) : null}

      {result ? (
        <>
          <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
            <StatCard
              compact
              label="median max DD"
              value={formatMetric(result.summary.max_drawdown.p50, "max_drawdown")}
            />
            <StatCard
              compact
              label="max DD p5–p95"
              value={`${formatMetric(result.summary.max_drawdown.p5, "max_drawdown")} – ${formatMetric(result.summary.max_drawdown.p95, "max_drawdown")}`}
            />
            <StatCard
              compact
              label="% sims ≥ actual final"
              value={`${result.summary.pct_sims_final_equity_ge_actual.toFixed(1)}%`}
            />
            <StatCard
              compact
              label="% sims ≤ actual max DD"
              value={`${result.summary.pct_sims_max_drawdown_le_actual.toFixed(1)}%`}
            />
          </div>

          <ChartContainer config={chartConfig} className="h-[320px] w-full">
            <ComposedChart data={chartData} margin={{ left: 8, right: 12, top: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="drawdown_pct"
                type="number"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                domain={[0, "auto"]}
                tickFormatter={(value: number) => `${Math.round(value)}%`}
                label={{ value: "Max drawdown %", position: "insideBottom", offset: -2, fontSize: 10 }}
              />
              <YAxis
                yAxisId="count"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                width={40}
                domain={[0, Math.ceil(maxCount * 1.15)]}
              />
              <YAxis
                yAxisId="cdf"
                orientation="right"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                width={36}
                domain={[0, 100]}
                tickFormatter={(value: number) => `${value}%`}
              />
              <ChartTooltip
                cursor={false}
                content={
                  <ChartTooltipContent
                    labelFormatter={(label) => `Max DD ≤ ${label}%`}
                    formatter={(value, name) => {
                      if (name === "cumulative_pct") return `${Number(value).toFixed(1)}%`;
                      return String(value);
                    }}
                  />
                }
              />
              <Legend
                verticalAlign="top"
                height={28}
                formatter={(value) =>
                  value === "count"
                    ? "Drawdown % PMF"
                    : value === "cumulative_pct"
                      ? "Cumulative drawdowns"
                      : value === "confidence"
                        ? "95% confidence"
                        : value
                }
              />
              <Bar
                yAxisId="count"
                dataKey="count"
                fill="var(--color-count)"
                fillOpacity={0.75}
                isAnimationActive={false}
                name="count"
              />
              <Line
                yAxisId="cdf"
                type="monotone"
                dataKey="cumulative_pct"
                stroke="var(--color-cumulative_pct)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                name="cumulative_pct"
              />
              <Scatter
                yAxisId="cdf"
                data={confidencePoints}
                dataKey="cumulative_pct"
                fill="var(--color-confidence)"
                shape={ConfidenceMark}
                isAnimationActive={false}
                name="confidence"
              />
              {confidenceDd != null ? (
                <ReferenceLine
                  yAxisId="cdf"
                  x={confidenceDd}
                  stroke="var(--color-confidence)"
                  strokeDasharray="3 3"
                  strokeOpacity={0.5}
                />
              ) : null}
              {actualDdPct != null ? (
                <ReferenceLine
                  yAxisId="cdf"
                  x={actualDdPct}
                  stroke="var(--muted-foreground)"
                  strokeDasharray="4 3"
                  strokeOpacity={0.7}
                  label={{
                    value: "Actual",
                    position: "insideTopLeft",
                    fill: "var(--muted-foreground)",
                    fontSize: 10,
                  }}
                />
              ) : null}
            </ComposedChart>
          </ChartContainer>
          <p className="text-xs text-muted-foreground">
            Green bars = frequency of max drawdowns across {result.n_sims.toLocaleString()} sims.
            Blue line = cumulative %. Red × at{" "}
            {result.confidence_marker.drawdown_pct.toFixed(1)}% DD means ~95% of sims had a milder
            max drawdown.
          </p>
        </>
      ) : null}
    </div>
  );
}
