import { useMemo, useState } from "react";
import { Area, AreaChart, CartesianGrid, ReferenceLine, XAxis, YAxis } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

const equityConfig = {
  equity: {
    label: "Equity",
    color: "var(--chart-1)",
  },
} satisfies ChartConfig;

const drawdownConfig = {
  drawdown: {
    label: "Drawdown",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig;

export type EquityDrawdownPoint = {
  date: string;
  rolling_pnl: number;
  drawdown: number;
};

const MAX_CHART_POINTS = 2000;

function downsampleChartPoints<T>(points: T[], maxPoints: number): T[] {
  if (points.length <= maxPoints) return points;
  const lastIndex = points.length - 1;
  return Array.from({ length: maxPoints }, (_, index) => {
    const sourceIndex = Math.round((index / (maxPoints - 1)) * lastIndex);
    return points[sourceIndex];
  });
}

function formatCurrency(value: number) {
  if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `$${(value / 1_000).toFixed(0)}k`;
  return `$${value.toFixed(0)}`;
}

function formatPercent(value: number) {
  return `${value.toFixed(1)}%`;
}

export function EquityDrawdownPanel({
  data,
  totalPoints,
}: {
  data: EquityDrawdownPoint[];
  totalPoints?: number;
}) {
  const [logScale, setLogScale] = useState(true);

  const chartData = useMemo(
    () =>
      downsampleChartPoints(
        data.map((point) => ({
          date: point.date,
          equity: point.rolling_pnl,
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
      <div className="flex items-center justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold">Equity / Drawdown</h3>
          <p className="text-xs text-muted-foreground">
            Compounded equity with underwater drawdown on the same timeline
            {isDownsampled
              ? ` (chart shows ${chartData.length.toLocaleString()} of ${shownTotal.toLocaleString()} bars)`
              : null}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Label htmlFor="equity-dd-log-scale" className="text-xs text-muted-foreground">
            Equity log scale
          </Label>
          <Switch
            id="equity-dd-log-scale"
            checked={logScale}
            onCheckedChange={setLogScale}
          />
        </div>
      </div>

      <div className="flex flex-col gap-1">
        <p className="text-[11px] font-medium text-muted-foreground">Equity</p>
        <ChartContainer config={equityConfig} className="h-[280px] w-full">
          <AreaChart
            data={chartData}
            margin={{ left: 8, right: 8, top: 8, bottom: 0 }}
          >
            <defs>
              <linearGradient id="fillEquityCombined" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--color-equity)" stopOpacity={0.35} />
                <stop offset="95%" stopColor="var(--color-equity)" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="date" hide />
            <YAxis
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              width={52}
              scale={logScale ? "log" : "auto"}
              domain={logScale ? ["auto", "auto"] : [0, "auto"]}
              allowDataOverflow={logScale}
              tickFormatter={formatCurrency}
            />
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  labelFormatter={(label) => `Date: ${label}`}
                  formatter={(value) => formatCurrency(Number(value))}
                />
              }
            />
            <Area
              dataKey="equity"
              type="monotone"
              fill="url(#fillEquityCombined)"
              stroke="var(--color-equity)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ChartContainer>
      </div>

      <div className="flex flex-col gap-1">
        <p className="text-[11px] font-medium text-muted-foreground">Drawdown</p>
        <ChartContainer config={drawdownConfig} className="h-[180px] w-full">
          <AreaChart
            data={chartData}
            margin={{ left: 8, right: 8, top: 4, bottom: 0 }}
          >
            <defs>
              <linearGradient id="fillDrawdownCombined" x1="0" y1="0" x2="0" y2="1">
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
              scale="linear"
              type="number"
              domain={[Math.min(-1, Math.floor(minDrawdown * 1.05)), 0]}
              allowDataOverflow={false}
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
              fill="url(#fillDrawdownCombined)"
              stroke="var(--color-drawdown)"
              strokeWidth={2}
              dot={false}
              baseValue={0}
              isAnimationActive={false}
            />
          </AreaChart>
        </ChartContainer>
      </div>
    </div>
  );
}
