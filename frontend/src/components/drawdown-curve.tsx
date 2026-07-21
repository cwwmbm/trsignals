import { useMemo } from "react";
import { Area, AreaChart, CartesianGrid, ReferenceLine, XAxis, YAxis } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";

const chartConfig = {
  drawdown: {
    label: "Drawdown",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig;

export type DrawdownPoint = { date: string; drawdown: number };

const MAX_CHART_POINTS = 2000;

function downsampleChartPoints<T>(points: T[], maxPoints: number): T[] {
  if (points.length <= maxPoints) return points;
  const lastIndex = points.length - 1;
  return Array.from({ length: maxPoints }, (_, index) => {
    const sourceIndex = Math.round((index / (maxPoints - 1)) * lastIndex);
    return points[sourceIndex];
  });
}

function formatPercent(value: number) {
  return `${value.toFixed(1)}%`;
}

export function DrawdownCurve({
  data,
  totalPoints,
}: {
  data: DrawdownPoint[];
  totalPoints?: number;
}) {
  const chartData = useMemo(
    () =>
      downsampleChartPoints(
        data.map((point) => ({
          date: point.date,
          // Backend stores fractional drawdown from peak (0 = at high-water).
          // Plot as negative % so the series dips below the zero baseline.
          drawdown: -Math.abs(Number(point.drawdown) || 0) * 100,
        })),
        MAX_CHART_POINTS,
      ),
    [data],
  );
  const shownTotal = totalPoints ?? data.length;
  const isDownsampled = shownTotal > chartData.length;
  const minDrawdown = useMemo(
    () => Math.min(0, ...chartData.map((point) => point.drawdown)),
    [chartData],
  );

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h3 className="text-sm font-semibold">Drawdown</h3>
        <p className="text-xs text-muted-foreground">
          Percent below peak equity over the test window
          {isDownsampled
            ? ` (chart shows ${chartData.length.toLocaleString()} of ${shownTotal.toLocaleString()} bars)`
            : null}
        </p>
      </div>

      <ChartContainer config={chartConfig} className="h-[240px] w-full">
        <AreaChart data={chartData} margin={{ left: 8, right: 8, top: 8 }}>
          <defs>
            <linearGradient id="fillDrawdown" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--color-drawdown)" stopOpacity={0.08} />
              <stop offset="95%" stopColor="var(--color-drawdown)" stopOpacity={0.4} />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            minTickGap={48}
            tickFormatter={(v: string) => v.slice(0, 4)}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            width={52}
            domain={[Math.floor(minDrawdown * 1.05), 0]}
            tickFormatter={formatPercent}
          />
          <ReferenceLine y={0} stroke="var(--muted-foreground)" strokeOpacity={0.5} />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                labelFormatter={(label) => `Date: ${label}`}
                formatter={(value) => formatPercent(Number(value))}
              />
            }
          />
          <Area
            dataKey="drawdown"
            type="monotone"
            fill="url(#fillDrawdown)"
            stroke="var(--color-drawdown)"
            strokeWidth={2}
            dot={false}
            baseValue={0}
          />
        </AreaChart>
      </ChartContainer>
    </div>
  );
}
