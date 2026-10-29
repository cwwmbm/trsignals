import { useMutation } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, ChevronsUpDown, Loader2 } from "lucide-react";
import { Fragment, useEffect, useMemo, useState } from "react";
import { Area, AreaChart, CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import {
  runPortfolioAllocationComparison,
  type AllocationAnalytics,
  type AllocationAttribution,
  type AllocationComparison,
  type AllocationComparisonRow,
  type ComparisonNesting,
  type AllocationSummary,
  type PortfolioSimulatePayload,
} from "@/api";
import { formatRisk, riskLabel } from "@/components/backtest/drawdown-risk-panel";
import { StatCard } from "@/components/backtest/stat-card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatPpDelta as formatSignedPp } from "@/lib/format-metric";
import { cn } from "@/lib/utils";

const compactHead = "h-7 px-1.5 py-0 text-[11px] font-medium";
const compactCell = "px-1.5 py-0.5 font-mono text-[11px] tabular-nums";

// Regime rows are four shades of one hue, from the darkest (Harsh) to the lightest (Low).
const MODEL_COLORS: Record<string, string> = {
  all_in: "var(--chart-5)",
  regime_adjusted_harsh: "oklch(0.42 0.14 158)",
  regime_adjusted_strict: "oklch(0.56 0.16 158)",
  regime_adjusted_medium: "oklch(0.7 0.17 158)",
  regime_adjusted_low: "oklch(0.84 0.15 158)",
  proportional_linear: "var(--chart-3)",
  proportional_early_conviction: "var(--chart-4)",
  proportional_confirmation_required: "var(--chart-2)",
  tiered: "var(--chart-1)",
};

/** Percentage-point delta, with float noise around zero shown as a plain 0 rather than "-0.0". */
function formatPpDelta(value: number | null | undefined) {
  if (value != null && Math.abs(value) < 0.05) return formatSignedPp(0);
  return formatSignedPp(value);
}

function pct(value: number | null | undefined, digits = 1) {
  return value == null || !Number.isFinite(value) ? "-" : `${value.toFixed(digits)}%`;
}

function money(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  const sign = value < 0 ? "−" : "";
  if (abs >= 1_000_000_000) return `${sign}$${(abs / 1_000_000_000).toFixed(2)}B`;
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(1)}k`;
  return `${sign}$${abs.toFixed(0)}`;
}

function compactMoney(value: number) {
  if (value >= 1_000_000_000) return `$${(value / 1_000_000_000).toFixed(1)}B`;
  if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `$${(value / 1_000).toFixed(0)}k`;
  return `$${value.toFixed(0)}`;
}

function AnalyticsCards({ analytics }: { analytics: AllocationAnalytics }) {
  return (
    <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4 lg:grid-cols-6">
      <StatCard compact label="Avg exposure" value={pct(analytics.average_exposure_pct)} />
      <StatCard compact label="Time invested" value={pct(analytics.time_invested_pct)} />
      <StatCard
        compact
        label="Exposure when in"
        value={pct(analytics.average_exposure_when_invested_pct)}
      />
      <StatCard
        compact
        label="Rebalances"
        value={`${analytics.rebalance_count} (${analytics.adjustment_count} partial)`}
      />
      <StatCard compact label="Turnover / yr" value={pct(analytics.turnover_annualized_pct, 0)} />
      <StatCard compact label="Trading costs" value={money(analytics.total_transaction_costs)} />
    </div>
  );
}

function AttributionBlock({ attribution, nested = false }: { attribution: AllocationAttribution; nested?: boolean }) {
  const rows: Array<[string, string, string]> = [
    [
      "Allocation effect",
      formatPpDelta(attribution.allocation_effect_pp),
      nested
        ? "Return difference from parent sizing, with child allocation models unchanged."
        : "Return given up (or gained) by holding less than everything on days all-in was fully invested.",
    ],
    [
      "Cost effect",
      formatPpDelta(attribution.cost_effect_pp),
      "Trading-cost difference versus all-in. Positive means this model paid less.",
    ],
    [
      "Other",
      formatPpDelta(attribution.other_pp),
      "Basket drift when holdings in several instruments move apart between rebalances.",
    ],
    ["Total gap", formatPpDelta(attribution.total_return_gap_pp), "Sum of daily returns versus all-in."],
  ];
  return (
    <div className="flex flex-col gap-1">
      <h3 className="text-xs font-medium text-muted-foreground">
        Performance from allocation changes (vs all-in on the same signals)
      </h3>
      <div className="overflow-x-auto rounded-md border border-border/60">
        <Table>
          <TableBody>
            {rows.map(([label, value, hint]) => (
              <TableRow key={label}>
                <TableCell className="px-1.5 py-0.5 text-[11px]" title={hint}>
                  {label}
                </TableCell>
                <TableCell className={cn(compactCell, "text-right")}>{value}</TableCell>
              </TableRow>
            ))}
            <TableRow>
              <TableCell className="px-1.5 py-0.5 text-[11px]">Final equity vs all-in</TableCell>
              <TableCell className={cn(compactCell, "text-right")}>
                {money(attribution.final_equity_delta)}
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </div>
      <div className="mt-1 overflow-x-auto rounded-md border border-border/60">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className={compactHead}>{nested ? "Active members" : "Active strategies"}</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>Days</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>Avg exposure</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>Allocation effect</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {attribution.by_active_count.map((row) => (
              <TableRow key={row.active_count}>
                <TableCell className={compactCell}>{row.active_count}</TableCell>
                <TableCell className={cn(compactCell, "text-right")}>
                  {row.days.toLocaleString()}
                </TableCell>
                <TableCell className={cn(compactCell, "text-right")}>
                  {pct(row.average_exposure_pct, 0)}
                </TableCell>
                <TableCell className={cn(compactCell, "text-right")}>
                  {formatPpDelta(row.allocation_effect_pp)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

function EligibilityBlock({ rows, nested = false }: { rows: NonNullable<AllocationSummary["eligibility"]>; nested?: boolean }) {
  return (
    <div className="flex flex-col gap-1">
      <h3 className="text-xs font-medium text-muted-foreground">
        {nested ? "Member" : "Strategy"} warmup (counted in the denominator once eligible)
      </h3>
      <div className="overflow-x-auto rounded-md border border-border/60">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className={compactHead}>{nested ? "Member" : "Strategy"}</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>Eligible from</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>Warmup bars</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>Data ends</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>Unavailable bars</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.strategy_id}>
                <TableCell className="px-1.5 py-0.5 text-[11px]">{row.strategy_name}</TableCell>
                <TableCell className={cn(compactCell, "text-right")}>
                  {row.first_eligible_date ?? "never"}
                </TableCell>
                <TableCell className={cn(compactCell, "text-right")}>
                  {row.warmup_bars.toLocaleString()}
                </TableCell>
                <TableCell className={cn(compactCell, "text-right")}>
                  {row.last_available_date ?? "-"}
                </TableCell>
                <TableCell className={cn(compactCell, "text-right")}>
                  {row.unavailable_bars.toLocaleString()}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

// Module level so ChartContainer sees a stable config between renders.
const exposureConfig = {
  exposure: { label: "Avg target exposure", color: "var(--chart-1)" },
} satisfies ChartConfig;

const DAY_MS = 86_400_000;

/**
 * Time-weighted average target exposure per calendar quarter, from the exposure step
 * function. Thousands of daily rebalances draw as a solid block, so the chart shows the
 * allocation regimes instead.
 */
export function quarterlyAverageExposure(
  steps: AllocationSummary["exposure_steps"],
): Array<{ date: string; pct: number }> {
  if (steps.length === 0) return [];
  const start = Date.parse(steps[0].date);
  const last = Date.parse(steps[steps.length - 1].date);
  const totals = new Map<string, { weighted: number; days: number }>();
  let index = 0;
  for (let t = start; t <= last; t += DAY_MS) {
    while (index + 1 < steps.length && Date.parse(steps[index + 1].date) <= t) index += 1;
    const day = new Date(t);
    const quarter = Math.floor(day.getUTCMonth() / 3);
    const key = `${day.getUTCFullYear()}-${String(quarter * 3 + 1).padStart(2, "0")}-01`;
    const entry = totals.get(key) ?? { weighted: 0, days: 0 };
    entry.weighted += steps[index].exposure;
    entry.days += 1;
    totals.set(key, entry);
  }
  return [...totals.entries()].map(([date, { weighted, days }]) => ({
    date,
    pct: (weighted / days) * 100,
  }));
}

function ExposureChart({ steps, nested = false }: { steps: AllocationSummary["exposure_steps"]; nested?: boolean }) {
  const data = useMemo(() => quarterlyAverageExposure(steps), [steps]);
  return (
    <div className="flex flex-col gap-1">
      <h3 className="text-xs font-medium text-muted-foreground">
        {nested ? "Average actual exposure by quarter" : "Average target exposure by quarter"}
      </h3>
      <ChartContainer config={exposureConfig} className="h-[160px] w-full">
        <AreaChart data={data} margin={{ left: 8, right: 8, top: 8 }}>
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tickLine={false}
            axisLine={false}
            minTickGap={48}
            tickFormatter={(v: string) => v.slice(0, 4)}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            width={40}
            domain={[0, 100]}
            tickFormatter={(v: number) => `${v}%`}
          />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                labelFormatter={(label) => `Quarter of ${label}`}
                formatter={(value) => `${Number(value).toFixed(0)}%`}
              />
            }
          />
          <Area
            dataKey="pct"
            name="exposure"
            type="monotone"
            fill="var(--color-exposure)"
            fillOpacity={0.25}
            stroke="var(--color-exposure)"
            strokeWidth={1.5}
            dot={false}
          />
        </AreaChart>
      </ChartContainer>
    </div>
  );
}

const TOP_COUNT = 5;

/** The colour of a row's line and dot. Flat models keep their fixed colours; variants spread around the hue wheel. */
function rowColor(row: AllocationComparisonRow, index: number): string {
  return MODEL_COLORS[row.model] ?? `oklch(0.7 0.15 ${(index * 47) % 360})`;
}

function sharpeOf(row: AllocationComparisonRow): number {
  return row.metrics.sharpe ?? Number.NEGATIVE_INFINITY;
}

/** The ids of the best rows by Sharpe: what the chart shows until you pick others. */
function topBySharpe(rows: AllocationComparisonRow[], count = TOP_COUNT): string[] {
  return [...rows].sort((a, b) => sharpeOf(b) - sharpeOf(a)).slice(0, count).map((row) => row.model);
}

type SortColumn = {
  id: string;
  label: string;
  title?: string;
  align: "left" | "right";
  /** Which end is best, so the first click puts the best rows first. */
  better: "high" | "low" | "alpha";
  value: (row: AllocationComparisonRow) => number | string | null;
};
type SortState = { id: string; dir: "asc" | "desc" };

function sortColumns(riskThresholds: number[]): SortColumn[] {
  return [
    { id: "model", label: "Model", align: "left", better: "alpha", value: (row) => row.label },
    { id: "final", label: "Final equity", align: "right", better: "high", value: (row) => row.metrics.rolling_pnl },
    { id: "cagr", label: "CAGR", align: "right", better: "high", value: (row) => row.metrics.cagr_percent },
    { id: "sharpe", label: "Sharpe", align: "right", better: "high", value: (row) => row.metrics.sharpe },
    { id: "maxdd", label: "Max DD", align: "right", better: "low", value: (row) => row.metrics.max_drawdown },
    ...riskThresholds.map((threshold): SortColumn => ({
      id: `risk-${threshold}`,
      label: riskLabel(threshold),
      title: "Chance that the worst drawdown of a backtest-length path goes beyond this depth, from reshuffled daily returns",
      align: "right",
      better: "low",
      value: (row) => row.drawdown_risk?.thresholds.find((entry) => entry.drawdown === threshold)?.probability ?? null,
    })),
    { id: "exposure", label: "Avg exposure", align: "right", better: "high", value: (row) => row.analytics.average_exposure_pct },
    { id: "rebalances", label: "Rebalances", align: "right", better: "low", value: (row) => row.analytics.rebalance_count },
    { id: "turnover", label: "Turnover / yr", align: "right", better: "low", value: (row) => row.analytics.turnover_annualized_pct },
    { id: "effect", label: "Allocation effect", align: "right", better: "high", value: (row) => row.attribution?.allocation_effect_pp ?? null },
  ];
}

/** The direction a first click on this column sorts in: best first. */
function firstDirection(column: SortColumn): "asc" | "desc" {
  return column.better === "high" ? "desc" : "asc";
}

/** Click once for best first, again to reverse, a third time to go back to the default order. */
function nextSort(current: SortState | null, column: SortColumn): SortState | null {
  const first = firstDirection(column);
  if (current?.id !== column.id) return { id: column.id, dir: first };
  if (current.dir === first) return { id: column.id, dir: first === "asc" ? "desc" : "asc" };
  return null;
}

/** Rows ordered by a column. Missing values go last whichever way it runs; ties keep their order. */
function sortRows(rows: AllocationComparisonRow[], column: SortColumn, dir: "asc" | "desc"): AllocationComparisonRow[] {
  const sign = dir === "asc" ? 1 : -1;
  const missing = (value: number | string | null) =>
    value === null || value === undefined || (typeof value === "number" && !Number.isFinite(value));
  return rows
    .map((row, index) => ({ row, index, value: column.value(row) }))
    .sort((a, b) => {
      const aMissing = missing(a.value);
      const bMissing = missing(b.value);
      if (aMissing || bMissing) return aMissing === bMissing ? a.index - b.index : aMissing ? 1 : -1;
      const order =
        typeof a.value === "string" || typeof b.value === "string"
          ? String(a.value).localeCompare(String(b.value), undefined, { numeric: true })
          : (a.value as number) - (b.value as number);
      return order === 0 ? a.index - b.index : order * sign;
    })
    .map((entry) => entry.row);
}

function SortableHead({
  column,
  sort,
  onSort,
}: {
  column: SortColumn;
  sort: SortState | null;
  onSort: (column: SortColumn) => void;
}) {
  const active = sort?.id === column.id;
  const Icon = !active ? ChevronsUpDown : sort.dir === "asc" ? ArrowUp : ArrowDown;
  return (
    <TableHead
      className={cn(compactHead, column.align === "right" && "text-right")}
      title={column.title}
      aria-sort={active ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        onClick={() => onSort(column)}
        className={cn(
          "inline-flex items-center gap-0.5 hover:text-foreground",
          column.align === "right" && "flex-row-reverse",
          active && "text-foreground",
        )}
      >
        {column.label}
        <Icon className={cn("size-3", !active && "opacity-30")} />
      </button>
    </TableHead>
  );
}

function NestingNote({ nesting }: { nesting: ComparisonNesting }) {
  if (nesting.skipped) {
    return <p className="text-[11px] text-muted-foreground">Nesting not run: {nesting.skipped}</p>;
  }
  return (
    <div className="flex flex-col gap-0.5 text-[11px] text-muted-foreground">
      <p>
        Each nested row is a grouping, then the parent model and the child model:{" "}
        <span className="font-mono">A+B | C+D · Linear / All-in</span> is two child portfolios (A+B and C+D) held
        in proportion to how many are active, each child all-in on its own members.{" "}
        {nesting.variants} variants: {nesting.structures} grouping{nesting.structures === 1 ? "" : "s"} ×{" "}
        {(nesting.models?.length ?? 0) ** 2} parent/child model pairs, run in {nesting.seconds}s. With this many
        variants the best ones look better than they are: the top rows are the luckiest of many.
        {nesting.truncated
          ? ` Only ${nesting.structures} of ${nesting.total_structures} groupings and ${nesting.models?.length} models were run, to keep it quick.`
          : ""}
      </p>
      <p className="font-mono">
        {nesting.strategies.map((item) => `${item.letter} ${item.name}`).join(" · ")}
      </p>
    </div>
  );
}

function ComparisonBlock({ comparison, nested = false }: { comparison: AllocationComparison; nested?: boolean }) {
  // Flat models first, as listed; nested variants after them, best Sharpe first.
  const baseRows = useMemo(() => {
    const flat = comparison.models.filter((row) => row.group !== "nested");
    const variants = comparison.models
      .filter((row) => row.group === "nested")
      .sort((a, b) => sharpeOf(b) - sharpeOf(a));
    return [...flat, ...variants];
  }, [comparison.models]);
  // Colours follow the default order, so sorting the table never changes a row's colour.
  const colors = useMemo(
    () => new Map(baseRows.map((row, index) => [row.model, rowColor(row, index)])),
    [baseRows],
  );
  const riskThresholds = useMemo(
    () => comparison.models.find((row) => row.drawdown_risk)?.drawdown_risk?.thresholds.map((row) => row.drawdown) ?? [],
    [comparison.models],
  );
  const columns = useMemo(() => sortColumns(riskThresholds), [riskThresholds]);
  const [sort, setSort] = useState<SortState | null>(null);
  useEffect(() => {
    setSort(null);
  }, [comparison.models]);
  const sortColumn = columns.find((column) => column.id === sort?.id);
  const rows = useMemo(
    () => (sort && sortColumn ? sortRows(baseRows, sortColumn, sort.dir) : baseRows),
    [baseRows, sort, sortColumn],
  );
  // Which rows are on the chart. A new comparison starts from the top five by Sharpe.
  const [selected, setSelected] = useState<Set<string>>(() => new Set(topBySharpe(comparison.models)));
  useEffect(() => {
    setSelected(new Set(topBySharpe(comparison.models)));
  }, [comparison.models]);
  const toggle = (model: string) =>
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(model)) next.delete(model);
      else next.add(model);
      return next;
    });
  const plotted = baseRows.filter((row) => selected.has(row.model));

  const config = useMemo(() => {
    const entries: ChartConfig = {};
    for (const row of baseRows) entries[row.model] = { label: row.label, color: colors.get(row.model) ?? "var(--chart-5)" };
    return entries;
  }, [baseRows, colors]);
  const hasVariants = baseRows.some((row) => row.group === "nested");

  return (
    <div className="flex flex-col gap-3">
      <p className="text-[11px] text-muted-foreground">
        All models use the same signals from {comparison.start_date}, rebased to{" "}
        {money(comparison.start_capital)}.{" "}
        {nested
          ? "Parent All-in assigns its full budget to active members; child allocations and reserved cash remain unchanged."
          : `All-in is 100% while any strategy is active${comparison.single_symbol === false
              ? `, split across ${(comparison.trading_symbols ?? []).join(" and ")} by how many strategies are active in each`
              : ""}.`}

        Regime rows read the {comparison.regime_score === "portfolio" ? "regime" : "robustness"}{" "}
        score.
        {nested
          ? ` Direct strategies use ${comparison.transaction_cost_bps} bps; children retain their own trading costs.`
          : comparison.transaction_cost_bps > 0
          ? ` Each pays ${comparison.transaction_cost_bps} bps on traded notional.`
          : " No trading costs."}
        {riskThresholds.length
          ? " P(DD > x) is the chance that the worst drawdown of a path as long as the backtest goes beyond x, from reshuffled daily returns."
          : ""}
      </p>
      {comparison.nesting ? <NestingNote nesting={comparison.nesting} /> : null}

      <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
        <span>
          Chart: {plotted.length} of {baseRows.length} selected
        </span>
        <Button
          size="sm"
          variant="outline"
          className="h-6 px-2 text-[11px]"
          onClick={() => setSelected(new Set(topBySharpe(comparison.models)))}
        >
          Top {TOP_COUNT} by Sharpe
        </Button>
        <Button size="sm" variant="ghost" className="h-6 px-2 text-[11px]" onClick={() => setSelected(new Set())}>
          Clear
        </Button>
        {hasVariants ? (
          <Button
            size="sm"
            variant="ghost"
            className="h-6 px-2 text-[11px]"
            onClick={() => setSelected(new Set(baseRows.filter((row) => row.group !== "nested").map((row) => row.model)))}
          >
            Flat models only
          </Button>
        ) : null}
        <span className="ml-auto">
          {sort && sortColumn ? (
            <>
              Sorted by {sortColumn.label}{" "}
              <button type="button" className="underline decoration-dotted underline-offset-2 hover:text-foreground" onClick={() => setSort(null)}>
                default order
              </button>
            </>
          ) : (
            "Click a column heading to sort"
          )}
        </span>
      </div>

      <div className="max-h-[420px] overflow-auto rounded-md border border-border/60">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              {columns.map((column) => (
                <SortableHead
                  key={column.id}
                  column={column}
                  sort={sort}
                  onSort={(picked) => setSort((current) => nextSort(current, picked))}
                />
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row, index) => (
              <Fragment key={row.model}>
                {!sort && row.group === "nested" && rows[index - 1]?.group !== "nested" ? (
                  <TableRow className="hover:bg-transparent">
                    <TableCell
                      colSpan={10 + riskThresholds.length}
                      className="bg-muted/30 px-1.5 py-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground"
                    >
                      Child portfolio variants ({baseRows.filter((item) => item.group === "nested").length}), best Sharpe first
                    </TableCell>
                  </TableRow>
                ) : null}
                <TableRow>
                  <TableCell className="px-1.5 py-0.5 text-[11px] font-medium">
                    <label className="flex cursor-pointer items-center gap-1.5">
                      <Checkbox
                        checked={selected.has(row.model)}
                        onCheckedChange={() => toggle(row.model)}
                        aria-label={`Show ${row.label} on the chart`}
                      />
                      <span
                        className="inline-block size-2 shrink-0 rounded-full"
                        style={{ background: colors.get(row.model) }}
                      />
                      <span className={cn(row.group === "nested" && "font-mono")}>{row.label}</span>
                    </label>
                  </TableCell>
                  <TableCell className={cn(compactCell, "text-right")}>{money(row.metrics.rolling_pnl)}</TableCell>
                  <TableCell className={cn(compactCell, "text-right")}>{pct(row.metrics.cagr_percent, 1)}</TableCell>
                  <TableCell className={cn(compactCell, "text-right")}>
                    {row.metrics.sharpe == null ? "-" : row.metrics.sharpe.toFixed(2)}
                  </TableCell>
                  <TableCell className={cn(compactCell, "text-right")}>
                    {row.metrics.max_drawdown == null ? "-" : pct(row.metrics.max_drawdown * 100, 1)}
                  </TableCell>
                  {riskThresholds.map((threshold) => {
                    const hit = row.drawdown_risk?.thresholds.find((entry) => entry.drawdown === threshold);
                    return (
                      <TableCell key={threshold} className={cn(compactCell, "text-right")}>
                        {hit ? formatRisk(hit.probability) : "-"}
                      </TableCell>
                    );
                  })}
                  <TableCell className={cn(compactCell, "text-right")}>
                    {pct(row.analytics.average_exposure_pct)}
                  </TableCell>
                  <TableCell className={cn(compactCell, "text-right")}>{row.analytics.rebalance_count}</TableCell>
                  <TableCell className={cn(compactCell, "text-right")}>
                    {pct(row.analytics.turnover_annualized_pct, 0)}
                  </TableCell>
                  <TableCell className={cn(compactCell, "text-right")}>
                    {row.attribution ? formatPpDelta(row.attribution.allocation_effect_pp) : "-"}
                  </TableCell>
                </TableRow>
              </Fragment>
            ))}
          </TableBody>
        </Table>
      </div>
      {plotted.length === 0 ? (
        <p className="py-6 text-center text-[11px] text-muted-foreground">
          Tick rows in the table to plot them, or use Top {TOP_COUNT} by Sharpe.
        </p>
      ) : (
        <ChartContainer config={config} className="h-[300px] w-full">
          <LineChart data={comparison.equity_curves} margin={{ left: 8, right: 8, top: 8 }}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={false}
              minTickGap={48}
              tickFormatter={(v: string) => v.slice(0, 4)}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              width={56}
              scale="log"
              domain={["auto", "auto"]}
              allowDataOverflow
              tickFormatter={compactMoney}
            />
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  labelFormatter={(label) => `Date: ${label}`}
                  formatter={(value, name) => `${config[String(name)]?.label ?? name}: ${compactMoney(Number(value))}`}
                />
              }
            />
            <ChartLegend content={<ChartLegendContent />} />
            {plotted.map((row) => (
              <Line
                key={row.model}
                dataKey={row.model}
                name={row.model}
                type="monotone"
                stroke={colors.get(row.model) ?? "var(--chart-5)"}
                strokeWidth={row.model === "all_in" ? 1.5 : 2}
                strokeDasharray={row.model === "all_in" ? "4 3" : undefined}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ChartContainer>
      )}
    </div>
  );
}

export function AllocationPanel({
  allocation,
  request,
}: {
  allocation: AllocationSummary;
  /** The simulation request, so models can be compared on the same inputs. */
  request: PortfolioSimulatePayload | null;
}) {
  const comparisonMutation = useMutation({ mutationFn: runPortfolioAllocationComparison });
  const proportional = allocation.attribution != null;
  const [includeNesting, setIncludeNesting] = useState(false);
  // Regrouping needs a flat portfolio of at least three strategies.
  const strategyCount = request?.strategy_ids.length ?? 0;
  const nestingUnavailable =
    !!allocation.nested_cash_policy || (request?.portfolio_ids?.length ?? 0) > 0
      ? "This portfolio already has child portfolios."
      : strategyCount < 3
        ? "Needs at least 3 strategies."
        : null;
  const nestingOn = includeNesting && nestingUnavailable === null;
  // A comparison belongs to the request that produced it; a new simulation hides it.
  const { include_nesting: _ignored, ...askedFor } = (comparisonMutation.variables ?? {}) as Record<string, unknown>;
  const comparison =
    comparisonMutation.data && JSON.stringify(askedFor) === JSON.stringify(request)
      ? comparisonMutation.data
      : null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">{allocation.label}</h3>
          <p className="text-[11px] text-muted-foreground">
            {allocation.nested_cash_policy
              ? "Each member receives a capital budget. Child models control investment within that budget; rejected allocations stay in cash."
              : proportional
              ? "Exposure is a target share of portfolio equity. Each change in who is active rebalances the position."
              : (allocation.trading_symbols?.length ?? 1) > 1
                ? `100% invested while any strategy is active, split across ${allocation.trading_symbols?.join(" and ")} by how many strategies are active in each.`
                : "100% invested while any strategy is active, otherwise cash."}
          </p>
        </div>
      </div>

      {allocation.vehicles?.length ? (
        <ul className="space-y-0.5 text-[11px] text-muted-foreground">
          {allocation.vehicles.map((vehicle) => (
            <li key={vehicle.symbol}>
              {vehicle.leverage_symbol == null
                ? `${vehicle.symbol} is traded itself at ${vehicle.leverage}x, with no separate leveraged product.`
                : vehicle.available
                ? `${vehicle.symbol} is traded as ${vehicle.leverage_symbol} from ${vehicle.first_date}; ${vehicle.symbol} × ${vehicle.leverage} before that, and for any trade already open on that day.`
                : `${vehicle.leverage_symbol} has no price history, so ${vehicle.symbol} × ${vehicle.leverage} stood in for every trade.`}
            </li>
          ))}
        </ul>
      ) : null}
      <AnalyticsCards analytics={allocation.analytics} />
      <ExposureChart steps={allocation.exposure_steps} nested={!!allocation.nested_cash_policy} />
      {allocation.members ? (
        <div className="overflow-x-auto">
          <h3 className="mb-1 text-xs font-medium">Member allocation at end · % of total equity</h3>
          <Table>
            <TableHeader><TableRow>
              <TableHead className={compactHead}>Member</TableHead>
              <TableHead className={compactHead}>Budget</TableHead>
              <TableHead className={compactHead}>Invested</TableHead>
              <TableHead className={compactHead}>Cash</TableHead>
            </TableRow></TableHeader>
            <TableBody>{allocation.members.map((member) => (
              <TableRow key={member.id}>
                <TableCell className={compactCell}>{member.name}</TableCell>
                <TableCell className={compactCell}>{pct(member.budget_weight * 100)}</TableCell>
                <TableCell className={compactCell}>{pct(member.invested_weight * 100)}</TableCell>
                <TableCell className={compactCell}>{pct(member.cash_weight * 100)}</TableCell>
              </TableRow>
            ))}
              <TableRow>
                <TableCell className={compactCell}>Parent unallocated cash</TableCell>
                <TableCell className={compactCell}>{pct(Math.max(0, 1 - allocation.members.reduce((sum, member) => sum + member.budget_weight, 0)) * 100)}</TableCell>
                <TableCell className={compactCell}>0.0%</TableCell>
                <TableCell className={compactCell}>{pct(Math.max(0, 1 - allocation.members.reduce((sum, member) => sum + member.budget_weight, 0)) * 100)}</TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </div>
      ) : null}

      {allocation.attribution ? <AttributionBlock attribution={allocation.attribution} nested={!!allocation.nested_cash_policy} /> : null}
      {allocation.eligibility ? <EligibilityBlock rows={allocation.eligibility} nested={!!allocation.nested_cash_policy} /> : null}

      {allocation.ledger ? (
        <p className="font-mono text-[11px] text-muted-foreground">
          Ledger at end: cash {money(allocation.ledger.cash)} · invested{" "}
          {money(allocation.ledger.invested_capital)} · realized P&amp;L{" "}
          {money(allocation.ledger.realized_pnl)} · unrealized {money(allocation.ledger.unrealized_pnl)} ·
          costs {money(allocation.ledger.transaction_costs)}
        </p>
      ) : null}

      <div className="flex flex-col gap-2 border-t border-border/60 pt-3">
        <div className="flex flex-wrap items-center gap-3">
          <h3 className="text-xs font-semibold">Compare allocation models</h3>
          <label
            className={cn(
              "flex items-center gap-1.5 text-[11px]",
              nestingUnavailable ? "cursor-not-allowed text-muted-foreground/60" : "cursor-pointer",
            )}
            title={nestingUnavailable ?? "Also regroup the strategies into child portfolios, each with its own allocation model."}
          >
            <Checkbox
              checked={nestingOn}
              disabled={nestingUnavailable !== null || comparisonMutation.isPending}
              onCheckedChange={(checked) => setIncludeNesting(checked === true)}
            />
            Include child portfolio nesting
          </label>
          <Button
            size="sm"
            variant="outline"
            className="h-7 gap-1.5 text-[11px]"
            disabled={!request || comparisonMutation.isPending}
            onClick={() => request && comparisonMutation.mutate(nestingOn ? { ...request, include_nesting: true } : request)}
          >
            {comparisonMutation.isPending ? <Loader2 className="size-3.5 animate-spin" /> : null}
            {comparison ? "Recalculate" : "Calculate"}
          </Button>
          {comparisonMutation.isPending && comparisonMutation.variables?.include_nesting ? (
            <span className="text-[11px] text-muted-foreground">Running the nested variants, about a minute…</span>
          ) : null}
        </div>
        {nestingOn ? (
          <p className="text-[11px] text-muted-foreground">
            Regroups the {strategyCount} strategies into two child portfolios in every way that splits them evenly, each
            group and the parent trying every model (regime-adjusted is left out). That is up to 75 more rows and takes
            about a minute. The chart then shows the top {TOP_COUNT} by Sharpe, and you tick others to add them.
          </p>
        ) : null}
        {allocation.nested_cash_policy ? (
          <p className="text-[11px] text-muted-foreground">
            Comparisons change only the parent model. Child models stay unchanged, so parent All-in can still hold child cash.
          </p>
        ) : null}
        {comparisonMutation.error ? (
          <p className="text-xs text-destructive">{String(comparisonMutation.error)}</p>
        ) : null}
        {comparison ? <ComparisonBlock comparison={comparison} nested={!!allocation.nested_cash_policy} /> : null}
      </div>
    </div>
  );
}
