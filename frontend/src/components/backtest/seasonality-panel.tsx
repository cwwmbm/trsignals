import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, XAxis, YAxis } from "recharts";
import type { DetailedResult } from "@/api";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { aggregateSeasonality, type SeasonalityBucket } from "@/lib/seasonality";

const chartConfig = {
  avgPct: { label: "Avg return", color: "var(--chart-1)" },
} satisfies ChartConfig;

function formatPct(value: number) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function SeasonalityBarChart({
  title,
  data,
  tickFormatter,
}: {
  title: string;
  data: SeasonalityBucket[];
  tickFormatter?: (label: string) => string;
}) {
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <h4 className="text-xs font-medium text-muted-foreground">{title}</h4>
      <ChartContainer config={chartConfig} className="h-[220px] w-full">
        <BarChart data={data} margin={{ left: 4, right: 8, top: 8, bottom: 0 }}>
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          <XAxis
            dataKey="label"
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            interval={0}
            tickFormatter={tickFormatter}
            tick={{ fontSize: 10 }}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            width={48}
            tickFormatter={(value: number) => `${value.toFixed(1)}%`}
            tick={{ fontSize: 10 }}
          />
          <ReferenceLine y={0} stroke="var(--muted-foreground)" strokeOpacity={0.45} />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                labelFormatter={(label) => String(label)}
                formatter={(_value, _name, item) => {
                  const bucket = item?.payload as SeasonalityBucket | undefined;
                  if (!bucket) return formatPct(Number(_value));
                  return [
                    `avg ${formatPct(bucket.avgPct)}`,
                    `${bucket.count} trades`,
                    `win ${bucket.winRate.toFixed(0)}%`,
                    `sum ${formatPct(bucket.sumPct)}`,
                  ].join(" · ");
                }}
              />
            }
          />
          <Bar dataKey="avgPct" radius={[3, 3, 0, 0]} isAnimationActive={false}>
            {data.map((bucket) => (
              <Cell
                key={bucket.key}
                fill={bucket.avgPct >= 0 ? "var(--chart-1)" : "var(--chart-2)"}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ChartContainer>
    </div>
  );
}

export function SeasonalityPanel({ trades }: { trades: DetailedResult["trades"] }) {
  const { days, months, quarters } = useMemo(() => aggregateSeasonality(trades), [trades]);
  const closedCount = useMemo(
    () => trades.filter((trade) => String(trade.status ?? "Closed") === "Closed").length,
    [trades],
  );

  return (
    <div className="flex flex-col gap-3">
      <div>
        <h3 className="text-sm font-semibold">Seasonality</h3>
        <p className="text-xs text-muted-foreground">
          Average closed-trade return by entry day · {closedCount} closed trades
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <SeasonalityBarChart
          title="Days"
          data={days}
          tickFormatter={(label) => label.slice(0, 3)}
        />
        <SeasonalityBarChart
          title="Months"
          data={months}
          tickFormatter={(label) => label.slice(0, 3)}
        />
        <SeasonalityBarChart title="Quarters" data={quarters} />
      </div>
    </div>
  );
}
