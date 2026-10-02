import { useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import { createPortal } from "react-dom";
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
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { requestRegimeCondition } from "@/lib/regime-condition";
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
  regimeIndicatorId,
  regimeIndicatorLabel,
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
    id: "regime_score",
    label: "Regime score",
    subtitle:
      "Sortino, average trade return, max drawdown, and how much of the average yearly log return remains after dropping the best year",
  },
  {
    id: "sharpe",
    label: "Sharpe",
    subtitle: "Summary Sharpe for a run that only enters in that regime",
  },
  {
    id: "sortino",
    label: "Sortino",
    subtitle: "Summary Sortino for a run that only enters in that regime",
  },
  {
    id: "calmar",
    label: "Calmar",
    subtitle: "Summary Calmar for a run that only enters in that regime",
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
    subtitle: "Summary max drawdown for a run that only enters in that regime",
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

function formatPart(value: number | null) {
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

function regimeScoreColor(value: number) {
  if (value < 25) return SHARPE_WEAK;
  if (value < 50) return SHARPE_DECENT;
  if (value < 75) return SHARPE_GOOD;
  return SHARPE_VERY_GOOD;
}

function barColor(metric: RegimeYMetric, value: number) {
  if (metric === "regime_score") return regimeScoreColor(value);
  if (metric === "sharpe") return sharpeColor(value);
  if (metric === "sortino") return sortinoColor(value);
  if (metric === "calmar") return calmarColor(value);
  if (metric === "max_drawdown") return drawdownColor(Math.abs(value));
  return exposureReturnColor(value);
}

function RegimeTooltip({
  active,
  payload,
  coordinate,
  anchorRef,
}: {
  active?: boolean;
  payload?: Array<{ payload?: RegimeChartPoint }>;
  coordinate?: { x?: number; y?: number };
  anchorRef: RefObject<HTMLDivElement | null>;
}) {
  const point = payload?.[0]?.payload;
  const tipRef = useRef<HTMLDivElement>(null);
  const x = coordinate?.x;
  const y = coordinate?.y;
  const pointKey = point?.key;

  useLayoutEffect(() => {
    const tip = tipRef.current;
    const anchor = anchorRef.current;
    if (!tip || !anchor || x == null || y == null) return;
    const origin = anchor.getBoundingClientRect();
    const box = tip.getBoundingClientRect();
    const margin = 8;
    const gap = 12;
    let left = origin.left + x + gap;
    let top = origin.top + y + gap;
    if (left + box.width > window.innerWidth - margin) {
      left = origin.left + x - gap - box.width;
    }
    left = Math.max(margin, Math.min(left, window.innerWidth - margin - box.width));
    if (top + box.height > window.innerHeight - margin) {
      top = origin.top + y - gap - box.height;
    }
    top = Math.max(margin, Math.min(top, window.innerHeight - margin - box.height));
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
    tip.style.visibility = "visible";
  }, [anchorRef, pointKey, x, y]);

  if (!active || !point || typeof document === "undefined") return null;
  return createPortal(
    <div
      ref={tipRef}
      className="pointer-events-none grid min-w-36 gap-0.5 rounded-lg border border-border/50 bg-background px-2.5 py-1.5 text-xs shadow-xl"
      style={{
        position: "fixed",
        left: 0,
        top: 0,
        zIndex: 70,
        visibility: "hidden",
      }}
    >
      <div className="mb-0.5 font-medium">{point.label}</div>
      <div>Regime score {formatSharpe(point.regimeScore)}</div>
      <div className="pl-2 text-muted-foreground">Sortino score {formatPart(point.regimeScoreSortino)}</div>
      <div className="pl-2 text-muted-foreground">Return score {formatPart(point.regimeScoreReturn)}</div>
      <div className="pl-2 text-muted-foreground">Drawdown score {formatPart(point.regimeScoreDrawdown)}</div>
      <div className="pl-2 text-muted-foreground">Robustness score {formatPart(point.regimeScoreRobustness)}</div>
      <div>{point.trades} trades</div>
      <div>{point.pctPositive == null ? "— positive" : `${point.pctPositive.toFixed(0)}% positive`}</div>
      <div>Avg / exposure day {formatPct(point.avgReturnPct)}</div>
      <div>Avg / trade {formatPct(point.avgTradeReturnPct)}</div>
      <div>Sharpe {formatSharpe(point.sharpe)}</div>
      <div>Sortino {formatSharpe(point.sortino)}</div>
      <div>CAGR {formatPct(point.cagr == null ? null : point.cagr * 100)}</div>
      <div>Calmar {formatSharpe(point.calmar)}</div>
      <div>MaxDD {formatDrawdown(point.maxDrawdown)}</div>
    </div>,
    document.body,
  );
}

function nowCaption(dimensionId: string, regimeKey: string | null, reading: number | null) {
  if (regimeKey == null) return null;
  const bucketNow = dimensionId === "spy" || dimensionId === "rate_curve" || dimensionId === "dollar_rates" || dimensionId === "inflation_yield";
  if (bucketNow) {
    const label = REGIME_DIMENSIONS.find((item) => item.id === dimensionId)?.buckets.find((bucket) => bucket.key === regimeKey)?.label;
    return label ? `NOW · ${label}` : "NOW";
  }
  if (reading == null || !Number.isFinite(reading)) return "NOW";
  if (dimensionId === "sector_breadth_50" || dimensionId === "sector_breadth_200") {
    return `NOW · ${Math.round(reading * 100)}%`;
  }
  if (dimensionId === "sector_trend_50") {
    const pct = reading * 100;
    const sign = pct > 0 ? "+" : "";
    return `NOW · ${sign}${pct.toFixed(1)}%`;
  }
  if (dimensionId === "inflation") {
    const pct = reading * 100;
    const sign = pct > 0 ? "+" : "";
    return `NOW · ${sign}${pct.toFixed(2)}%`;
  }
  if (dimensionId === "atr") return `NOW · ${reading.toFixed(2)}`;
  if (
    dimensionId === "rate_shock" ||
    dimensionId === "dollar_shock" ||
    dimensionId === "curve_10y3m" ||
    dimensionId === "curve_change_20"
  ) {
    return `NOW · ${reading.toFixed(2)}`;
  }
  return `NOW · ${reading.toFixed(1)}`;
}

function RegimeTick({
  x = 0,
  y = 0,
  payload,
  currentLabel,
  onSelectLabel,
  angled = false,
}: {
  x?: number | string;
  y?: number | string;
  payload?: { value?: string };
  currentLabel: string | null;
  onSelectLabel?: (label: string) => void;
  angled?: boolean;
}) {
  const label = payload?.value ?? "";
  const now = currentLabel != null && label === currentLabel;
  return (
    <text
      x={x}
      y={y}
      textAnchor={angled ? "end" : "middle"}
      transform={angled ? `rotate(-40, ${x}, ${y})` : undefined}
      fontSize={angled ? 9 : 10}
      fontWeight={now ? 700 : 400}
      fill={now ? "var(--regime-now-ink)" : "currentColor"}
      style={{ cursor: "pointer", fill: now ? "var(--regime-now-ink)" : undefined }}
      onClick={() => onSelectLabel?.(label)}
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
  onSelectBucket,
}: {
  dimension: RegimeDimension;
  points: RegimeChartPoint[];
  metric: RegimeYMetric;
  currentKey: string | null;
  reading: number | null;
  coverage?: { start: string; end: string } | null;
  onSelectBucket: (point: RegimeChartPoint) => void;
}) {
  const data = points.map((point) => ({ ...point, value: regimeYValue(point, metric) }));
  const currentLabel = points.find((point) => point.key === currentKey)?.label ?? null;
  const currentValue = data.find((point) => point.label === currentLabel)?.value ?? null;
  const caption = nowCaption(dimension.id, currentKey, reading);
  const angledTicks = dimension.buckets.length >= 5;
  const chartRef = useRef<HTMLDivElement>(null);
  const selectLabel = (label: string | undefined) => {
    const point = points.find((item) => item.label === label);
    if (point) onSelectBucket(point);
  };
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div>
        <h4 className="text-xs font-medium text-muted-foreground">{dimension.title}</h4>
        <p className="text-[10px] text-muted-foreground">{dimension.hint}</p>
      </div>
      <div ref={chartRef}>
      <ChartContainer config={chartConfig} className="h-[220px] w-full">
        <BarChart
          data={data}
          margin={{ left: 4, right: 8, top: 26, bottom: angledTicks ? 36 : 8 }}
          onClick={(state) => {
            const label =
              state && typeof state === "object" && "activeLabel" in state
                ? String((state as { activeLabel?: string }).activeLabel ?? "")
                : "";
            if (label) selectLabel(label);
          }}
        >
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          <NowOutline currentLabel={currentLabel} barValue={currentValue} />
          <XAxis
            dataKey="label"
            tickLine={false}
            axisLine={false}
            tickMargin={angledTicks ? 2 : 10}
            interval={0}
            tick={(props) => (
              <RegimeTick
                {...props}
                angled={angledTicks}
                currentLabel={currentLabel}
                onSelectLabel={selectLabel}
              />
            )}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            width={48}
            domain={
              metric === "max_drawdown"
                ? [(min: number) => (Number.isFinite(min) && min < 0 ? min : 0), 0]
                : metric === "regime_score"
                  ? [0, 100]
                  : undefined
            }
            tickFormatter={(value: number) =>
              metric === "regime_score"
                ? value.toFixed(0)
                : metric === "sharpe" || metric === "sortino" || metric === "calmar"
                  ? value.toFixed(1)
                  : metric === "max_drawdown"
                    ? `${value.toFixed(0)}%`
                    : `${value.toFixed(1)}%`
            }
            tick={{ fontSize: 10 }}
          />
          <ReferenceLine y={0} stroke="var(--muted-foreground)" strokeOpacity={0.45} />
          <ChartTooltip
            cursor={false}
            isAnimationActive={false}
            allowEscapeViewBox={{ x: true, y: true }}
            content={<RegimeTooltip anchorRef={chartRef} />}
          />
          <Bar
            dataKey="value"
            radius={[3, 3, 0, 0]}
            isAnimationActive={false}
            cursor="pointer"
            onClick={(bar) => {
              const payload =
                bar && typeof bar === "object" && "payload" in bar
                  ? (bar as { payload?: { label?: string } }).payload
                  : undefined;
              selectLabel(payload?.label);
            }}
          >
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
      </div>
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
  const [metric, setMetric] = useState<RegimeYMetric>("regime_score");
  const [showSmallSamples, setShowSmallSamples] = useState(false);
  const [prompt, setPrompt] = useState<{ dimension: RegimeDimension; point: RegimeChartPoint } | null>(
    null,
  );
  const selected = METRIC_OPTIONS.find((option) => option.id === metric) ?? METRIC_OPTIONS[0];
  const charts = useMemo(
    () =>
      REGIME_DIMENSIONS.map((dimension) => ({
        dimension,
        points: regimeChartPoints(dimension, sharpe, trades, showSmallSamples),
      })),
    [sharpe, trades, showSmallSamples],
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">Market regimes</h3>
          <p className="text-xs text-muted-foreground">{selected.subtitle}</p>
          <p className="text-[10px] text-muted-foreground">
            {showSmallSamples
              ? "Including buckets with fewer than 20 closed trades."
              : "Buckets with fewer than 20 closed trades are left blank."}
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <Checkbox
              checked={showSmallSamples}
              onCheckedChange={(checked) => setShowSmallSamples(checked === true)}
            />
            Show under 20 trades
          </label>
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
      </div>

      {sharpe ? (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-4">
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
                onSelectBucket={(point) => setPrompt({ dimension, point })}
              />
            ),
          )}
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">Regime stats are not available for this result.</p>
      )}

      {prompt ? (
        <RegimeConditionPrompt
          dimension={prompt.dimension}
          point={prompt.point}
          onClose={() => setPrompt(null)}
        />
      ) : null}
    </div>
  );
}

function RegimeConditionPrompt({
  dimension,
  point,
  onClose,
}: {
  dimension: RegimeDimension;
  point: RegimeChartPoint;
  onClose: () => void;
}) {
  const label = regimeIndicatorLabel(dimension.title, point.label);
  const choose = (operator: "is true" | "is false") => {
    requestRegimeCondition(regimeIndicatorId(dimension.id, point.key), operator, label);
    onClose();
  };
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/70 p-4"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-labelledby="regime-condition-title"
        className="w-full max-w-sm rounded-lg border border-border bg-background p-4 shadow-xl"
        onClick={(event) => event.stopPropagation()}
      >
        <h3 id="regime-condition-title" className="text-sm font-semibold">
          {label}
        </h3>
        <p className="mt-1 text-xs text-muted-foreground">
          Add an entry condition to the strategy builder. Days before this series exists stay out either way.
        </p>
        <div className="mt-4 flex flex-col gap-2">
          <Button type="button" onClick={() => choose("is true")}>
            Only this regime
          </Button>
          <Button type="button" variant="outline" onClick={() => choose("is false")}>
            Everything except this regime
          </Button>
          <Button type="button" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
        </div>
      </div>
    </div>
  );
}
