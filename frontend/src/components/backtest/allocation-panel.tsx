import { useMutation } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { useMemo } from "react";
import { Area, AreaChart, CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import {
  runPortfolioAllocationComparison,
  type AllocationAnalytics,
  type AllocationAttribution,
  type AllocationComparison,
  type AllocationSummary,
  type PortfolioSimulatePayload,
} from "@/api";
import { StatCard } from "@/components/backtest/stat-card";
import { Button } from "@/components/ui/button";
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

function ComparisonBlock({ comparison, nested = false }: { comparison: AllocationComparison; nested?: boolean }) {
  const config = useMemo(() => {
    const entries: ChartConfig = {};
    for (const row of comparison.models) {
      entries[row.model] = { label: row.label, color: MODEL_COLORS[row.model] ?? "var(--chart-5)" };
    }
    return entries;
  }, [comparison.models]);

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
      </p>
      <div className="overflow-x-auto rounded-md border border-border/60">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className={compactHead}>Model</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>Final equity</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>CAGR</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>Sharpe</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>Max DD</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>Avg exposure</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>Rebalances</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>Turnover / yr</TableHead>
              <TableHead className={cn(compactHead, "text-right")}>Allocation effect</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {comparison.models.map((row) => (
              <TableRow key={row.model}>
                <TableCell className="px-1.5 py-0.5 text-[11px] font-medium">
                  <span
                    className="mr-1.5 inline-block size-2 rounded-full align-middle"
                    style={{ background: MODEL_COLORS[row.model] }}
                  />
                  {row.label}
                </TableCell>
                <TableCell className={cn(compactCell, "text-right")}>{money(row.metrics.rolling_pnl)}</TableCell>
                <TableCell className={cn(compactCell, "text-right")}>{pct(row.metrics.cagr_percent, 1)}</TableCell>
                <TableCell className={cn(compactCell, "text-right")}>
                  {row.metrics.sharpe == null ? "-" : row.metrics.sharpe.toFixed(2)}
                </TableCell>
                <TableCell className={cn(compactCell, "text-right")}>
                  {row.metrics.max_drawdown == null ? "-" : pct(row.metrics.max_drawdown * 100, 1)}
                </TableCell>
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
            ))}
          </TableBody>
        </Table>
      </div>
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
          {comparison.models.map((row) => (
            <Line
              key={row.model}
              dataKey={row.model}
              name={row.model}
              type="monotone"
              stroke={MODEL_COLORS[row.model] ?? "var(--chart-5)"}
              strokeWidth={row.model === "all_in" ? 1.5 : 2}
              strokeDasharray={row.model === "all_in" ? "4 3" : undefined}
              dot={false}
            />
          ))}
        </LineChart>
      </ChartContainer>
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
  // A comparison belongs to the request that produced it; a new simulation hides it.
  const comparison =
    comparisonMutation.data &&
    JSON.stringify(comparisonMutation.variables) === JSON.stringify(request)
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
              {vehicle.available
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
          <Button
            size="sm"
            variant="outline"
            className="h-7 gap-1.5 text-[11px]"
            disabled={!request || comparisonMutation.isPending}
            onClick={() => request && comparisonMutation.mutate(request)}
          >
            {comparisonMutation.isPending ? <Loader2 className="size-3.5 animate-spin" /> : null}
            {comparison ? "Recalculate" : "Calculate"}
          </Button>
        </div>
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
