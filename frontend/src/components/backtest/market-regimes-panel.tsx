import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  usePlotArea,
  useXAxisDomain,
  useXAxisScale,
  useYAxisScale,
  XAxis,
  YAxis,
} from "recharts";
import type { DetailedResult, MarketRegimeCoverage, MarketRegimeCurrent, MarketRegimeSharpe } from "@/api";
import {
  ChartContainer,
  ChartTooltip,
  type ChartConfig,
} from "@/components/ui/chart";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  REGIME_DIMENSIONS,
  regimeChartPoints,
  regimeYValue,
  type RegimeChartPoint,
  type RegimeDimension,
  type RegimeYMetric,
} from "@/lib/market-regimes";

const chartConfig = {
  value: { label: "Value", color: "var(--chart-1)" },
} satisfies ChartConfig;

const ATR_UNAVAILABLE = "Traded-symbol ATR is not available for this result.";

const METRIC_OPTIONS: { id: RegimeYMetric; label: string; subtitle: string }[] = [
  {
    id: "sharpe",
    label: "Sharpe",
    subtitle: "Sharpe of daily returns on trades entered in each regime",
  },
  {
    id: "sortino",
    label: "Sortino",
    subtitle: "Sortino of daily returns on trades entered in each regime",
  },
  {
    id: "calmar",
    label: "Calmar",
    subtitle: "Calmar of trades entered in each regime (CAGR / max drawdown)",
  },
  {
    id: "avg_return",
    label: "Avg % / exposure day",
    subtitle: "Average closed-trade return per day held, for trades entered in each regime",
  },
  {
    id: "avg_trade_return",
    label: "Avg % / trade",
    subtitle: "Average closed-trade return for trades entered in each regime",
  },
  {
    id: "max_drawdown",
    label: "Max drawdown",
    subtitle: "Max drawdown of the trades entered in each regime",
  },
];

const SHARPE_WEAK = "#dc2626";
const SHARPE_DECENT = "#f59e0b";
const SHARPE_GOOD = "#4ade80";
const SHARPE_VERY_GOOD = "#15803d";

function formatSharpe(value: number | null) {
  if (value == null) return "—";
  return value.toFixed(2);
}

function formatPct(value: number | null, digits = 2) {
  if (value == null) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}%`;
}

function formatDrawdown(value: number | null) {
  if (value == null) return "—";
  return `${(value * 100).toFixed(2)}%`;
}

function sharpeColor(value: number) {
  if (value < 0.5) return SHARPE_WEAK;
  if (value < 1) return SHARPE_DECENT;
  if (value <= 1.5) return SHARPE_GOOD;
  return SHARPE_VERY_GOOD;
}

function sortinoColor(value: number) {
  if (value < 1) return SHARPE_WEAK;
  if (value < 2) return SHARPE_DECENT;
  if (value <= 3) return SHARPE_GOOD;
  return SHARPE_VERY_GOOD;
}

function calmarColor(value: number) {
  if (value < 0.5) return SHARPE_WEAK;
  if (value < 1) return SHARPE_DECENT;
  if (value <= 2) return SHARPE_GOOD;
  return SHARPE_VERY_GOOD;
}

function drawdownColor(magnitudePct: number) {
  if (magnitudePct > 60) return SHARPE_WEAK;
  if (magnitudePct > 50) return SHARPE_DECENT;
  if (magnitudePct >= 40) return SHARPE_GOOD;
  return SHARPE_VERY_GOOD;
}

function exposureReturnColor(pct: number) {
  if (pct <= 0) return SHARPE_WEAK;
  if (pct <= 0.75) return SHARPE_DECENT;
  if (pct <= 1.5) return SHARPE_GOOD;
  return SHARPE_VERY_GOOD;
}

function barColor(metric: RegimeYMetric, value: number) {
  if (metric === "sharpe") return sharpeColor(value);
  if (metric === "sortino") return sortinoColor(value);
  if (metric === "calmar") return calmarColor(value);
  if (metric === "max_drawdown") return drawdownColor(Math.abs(value));
  return exposureReturnColor(value);
}

function RegimeTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload?: RegimeChartPoint }>;
}) {
  const point = payload?.[0]?.payload;
  if (!active || !point) return null;
  return (
    <div className="grid min-w-36 gap-0.5 rounded-lg border border-border/50 bg-background px-2.5 py-1.5 text-xs shadow-xl">
      <div className="mb-0.5 font-medium">{point.label}</div>
      <div>{point.trades} trades</div>
      <div>{point.pctPositive == null ? "— positive" : `${point.pctPositive.toFixed(0)}% positive`}</div>
      <div>Avg / exposure day {formatPct(point.avgReturnPct)}</div>
      <div>Avg / trade {formatPct(point.avgTradeReturnPct)}</div>
      <div>Sharpe {formatSharpe(point.sharpe)}</div>
      <div>Sortino {formatSharpe(point.sortino)}</div>
      <div>CAGR {formatPct(point.cagr == null ? null : point.cagr * 100)}</div>
      <div>Calmar {formatSharpe(point.calmar)}</div>
      <div>MaxDD {formatDrawdown(point.maxDrawdown)}</div>
    </div>
  );
}

function nowCaption(dimensionId: string, regimeKey: string | null, reading: number | null) {
  if (regimeKey == null) return null;
  if (dimensionId === "spy") {
    if (regimeKey === "bull") return "NOW · Bull";
    if (regimeKey === "bear") return "NOW · Bear";
  }
  if (reading == null || !Number.isFinite(reading)) return "NOW";
  if (dimensionId === "sector_breadth_50" || dimensionId === "sector_breadth_200") {
    return `NOW · ${Math.round(reading * 100)}%`;
  }
  if (dimensionId === "atr") return `NOW · ${reading.toFixed(2)}`;
  return `NOW · ${reading.toFixed(1)}`;
}

function RegimeTick({
  x = 0,
  y = 0,
  payload,
  currentLabel,
}: {
  x?: number | string;
  y?: number | string;
  payload?: { value?: string };
  currentLabel: string | null;
}) {
  const label = payload?.value ?? "";
  const now = currentLabel != null && label === currentLabel;
  return (
    <text
      x={x}
      y={y}
      textAnchor="middle"
      fontSize={10}
      fontWeight={now ? 700 : 400}
      fill={now ? "var(--regime-now-ink)" : "currentColor"}
      style={now ? { fill: "var(--regime-now-ink)" } : undefined}
    >
      {label}
    </text>
  );
}

type NowFrame = {
  x: number;
  y: number;
  width: number;
  height: number;
  center: number;
  axisY: number;
};

function useNowFrame(currentLabel: string | null, barValue: number | null): NowFrame | null {
  const plot = usePlotArea();
  const domain = useXAxisDomain();
  const xScale = useXAxisScale();
  const yScale = useYAxisScale();
  if (
    currentLabel == null ||
    !plot ||
    !Array.isArray(domain) ||
    domain.length === 0 ||
    !xScale ||
    !yScale
  ) {
    return null;
  }
  const index = domain.indexOf(currentLabel);
  if (index < 0) return null;
  const start = xScale(currentLabel, { position: "start" });
  const end = xScale(currentLabel, { position: "end" });
  if (start == null || end == null) return null;
  const center = (start + end) / 2;
  const slot = plot.width / domain.length;
  const band = Math.abs(end - start);
  const width = Math.min(slot - 6, Math.max(band + 18, slot * 0.78));
  const axisY = plot.y + plot.height;
  let barTop = axisY;
  let barBottom = axisY;
  if (barValue != null && Number.isFinite(barValue) && barValue !== 0) {
    const yValue = yScale(barValue);
    const yZero = yScale(0);
    if (yValue != null && yZero != null) {
      barTop = Math.min(yValue, yZero);
      barBottom = Math.max(yValue, yZero);
    }
  }
  const topPad = 22;
  const y = Math.max(2, Math.min(barTop - topPad, axisY - 88));
  const bottom = Math.max(barBottom, axisY) + 28;
  return { x: center - width / 2, y, width, height: Math.max(bottom - y, 1), center, axisY };
}

function NowOutline({ currentLabel, barValue }: { currentLabel: string | null; barValue: number | null }) {
  const frame = useNowFrame(currentLabel, barValue);
  if (!frame) return null;
  return (
    <rect
      x={frame.x}
      y={frame.y}
      width={frame.width}
      height={frame.height}
      rx={10}
      fill="var(--regime-now-fill)"
      stroke="var(--regime-now-stroke)"
      strokeWidth={1}
    />
  );
}

function NowMarker({
  currentLabel,
  barValue,
  caption,
}: {
  currentLabel: string | null;
  barValue: number | null;
  caption: string | null;
}) {
  const frame = useNowFrame(currentLabel, barValue);
  if (!frame || !caption) return null;
  const pillHeight = 16;
  const pillWidth = Math.max(46, caption.length * 5.7 + 14);
  const pillX = frame.center - pillWidth / 2;
  const pillY = frame.y + 4;
  return (
    <g>
      <circle cx={frame.center} cy={frame.axisY + 5} r={2.2} fill="var(--regime-now-ink)" />
      <rect
        x={pillX}
        y={pillY}
        width={pillWidth}
        height={pillHeight}
        rx={7}
        fill="var(--regime-now-pill)"
        stroke="var(--regime-now-stroke)"
        strokeWidth={1}
      />
      <text
        x={frame.center}
        y={pillY + 11}
        textAnchor="middle"
        fontSize={9}
        fontWeight={600}
        fill="var(--regime-now-ink)"
      >
        {caption}
      </text>
    </g>
  );
}

function RegimeBarChart({
  dimension,
  points,
  metric,
  currentKey,
  reading,
  coverage,
}: {
  dimension: RegimeDimension;
  points: RegimeChartPoint[];
  metric: RegimeYMetric;
  currentKey: string | null;
  reading: number | null;
  coverage?: { start: string; end: string } | null;
}) {
  const data = points.map((point) => ({ ...point, value: regimeYValue(point, metric) }));
  const currentLabel = points.find((point) => point.key === currentKey)?.label ?? null;
  const currentValue = data.find((point) => point.label === currentLabel)?.value ?? null;
  const caption = nowCaption(dimension.id, currentKey, reading);
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div>
        <h4 className="text-xs font-medium text-muted-foreground">{dimension.title}</h4>
        <p className="text-[10px] text-muted-foreground">{dimension.hint}</p>
      </div>
      <ChartContainer config={chartConfig} className="h-[220px] w-full">
        <BarChart data={data} margin={{ left: 4, right: 8, top: 26, bottom: 8 }}>
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          <NowOutline currentLabel={currentLabel} barValue={currentValue} />
          <XAxis
            dataKey="label"
            tickLine={false}
            axisLine={false}
            tickMargin={10}
            interval={0}
            tick={(props) => <RegimeTick {...props} currentLabel={currentLabel} />}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            width={48}
            domain={
              metric === "max_drawdown"
                ? [(min: number) => (Number.isFinite(min) && min < 0 ? min : 0), 0]
                : undefined
            }
            tickFormatter={(value: number) =>
              metric === "sharpe" || metric === "sortino" || metric === "calmar"
                ? value.toFixed(1)
                : metric === "max_drawdown"
                  ? `${value.toFixed(0)}%`
                  : `${value.toFixed(1)}%`
            }
            tick={{ fontSize: 10 }}
          />
          <ReferenceLine y={0} stroke="var(--muted-foreground)" strokeOpacity={0.45} />
          <ChartTooltip cursor={false} content={<RegimeTooltip />} />
          <Bar dataKey="value" radius={[3, 3, 0, 0]} isAnimationActive={false}>
            {data.map((point) => (
              <Cell
                key={point.key}
                fill={point.value == null ? "transparent" : barColor(metric, point.value)}
                fillOpacity={0.9}
              />
            ))}
          </Bar>
          <NowMarker currentLabel={currentLabel} barValue={currentValue} caption={caption} />
        </BarChart>
      </ChartContainer>
      {coverage?.start && coverage?.end ? (
        <p className="text-[10px] text-muted-foreground">
          Calculated from {coverage.start} to {coverage.end}. Trades entered before that are left out.
        </p>
      ) : null}
    </div>
  );
}

export function MarketRegimesPanel({
  sharpe,
  trades,
  current,
  coverage,
}: {
  sharpe?: MarketRegimeSharpe | null;
  trades: DetailedResult["trades"];
  current?: MarketRegimeCurrent | null;
  coverage?: MarketRegimeCoverage | null;
}) {
  const [metric, setMetric] = useState<RegimeYMetric>("sharpe");
  const selected = METRIC_OPTIONS.find((option) => option.id === metric) ?? METRIC_OPTIONS[0];
  const charts = useMemo(
    () =>
      REGIME_DIMENSIONS.map((dimension) => ({
        dimension,
        points: regimeChartPoints(dimension, sharpe, trades),
      })),
    [sharpe, trades],
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">Market regimes</h3>
          <p className="text-xs text-muted-foreground">{selected.subtitle}</p>
          <p className="text-[10px] text-muted-foreground">
            Buckets with fewer than 20 closed trades are left blank.
          </p>
        </div>
        <Select value={metric} onValueChange={(value) => value && setMetric(value as RegimeYMetric)}>
          <SelectTrigger size="sm" className="h-7 text-xs" aria-label="Y axis metric">
            <SelectValue>
              {(value: string) => METRIC_OPTIONS.find((option) => option.id === value)?.label ?? value}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {METRIC_OPTIONS.map((option) => (
              <SelectItem key={option.id} value={option.id}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {sharpe ? (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {charts.map(({ dimension, points }) =>
            dimension.unavailableWhenEmpty && points.every((point) => point.days === 0) ? (
              <div key={dimension.id} className="flex min-w-0 flex-col gap-2">
                <h4 className="text-xs font-medium text-muted-foreground">{dimension.title}</h4>
                <p className="text-xs text-muted-foreground">{ATR_UNAVAILABLE}</p>
              </div>
            ) : (
              <RegimeBarChart
                key={dimension.id}
                dimension={dimension}
                points={points}
                metric={metric}
                currentKey={current?.regimes?.[dimension.id] ?? null}
                reading={current?.readings?.[dimension.id] ?? null}
                coverage={coverage?.[dimension.id]}
              />
            ),
          )}
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">Regime stats are not available for this result.</p>
      )}
    </div>
  );
}
