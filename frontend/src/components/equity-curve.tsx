import { useState } from "react";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

const chartConfig = {
  equity: {
    label: "Equity",
    color: "var(--chart-1)",
  },
} satisfies ChartConfig;

function formatCurrency(value: number) {
  if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `$${(value / 1_000).toFixed(0)}k`;
  return `$${value.toFixed(0)}`;
}

export type EquityPoint = { date: string; rolling_pnl: number };

export function EquityCurve({ data }: { data: EquityPoint[] }) {
  const [logScale, setLogScale] = useState(false);
  const chartData = data.map((point) => ({ date: point.date, equity: point.rolling_pnl }));

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold">Equity Curve</h3>
          <p className="text-xs text-muted-foreground">
            Compounded account value over the test window
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Label htmlFor="log-scale" className="text-xs text-muted-foreground">
            Log scale
          </Label>
          <Switch id="log-scale" checked={logScale} onCheckedChange={setLogScale} />
        </div>
      </div>

      <ChartContainer config={chartConfig} className="h-[320px] w-full">
        <AreaChart data={chartData} margin={{ left: 8, right: 8, top: 8 }}>
          <defs>
            <linearGradient id="fillEquity" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--color-equity)" stopOpacity={0.35} />
              <stop offset="95%" stopColor="var(--color-equity)" stopOpacity={0.02} />
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
            fill="url(#fillEquity)"
            stroke="var(--color-equity)"
            strokeWidth={2}
            dot={false}
          />
        </AreaChart>
      </ChartContainer>
    </div>
  );
}
