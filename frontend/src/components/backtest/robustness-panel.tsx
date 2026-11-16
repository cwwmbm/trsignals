import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, XAxis, YAxis } from "recharts";
import { Loader2 } from "lucide-react";
import type { RobustnessNull, RobustnessReport, RobustnessStatus } from "@/api";
import { StatCard } from "@/components/backtest/stat-card";
import { Button } from "@/components/ui/button";
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";

const nullChartConfig = {
  count: { label: "Shifted runs", color: "var(--chart-1)" },
} satisfies ChartConfig;

const yearChartConfig = {
  pct: { label: "Return", color: "var(--chart-1)" },
} satisfies ChartConfig;

const STATUS_STYLE: Record<RobustnessStatus, string> = {
  pass: "bg-[var(--gain)]/15 text-[var(--gain)]",
  warn: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  fail: "bg-[var(--loss)]/15 text-[var(--loss)]",
  skip: "bg-muted text-muted-foreground",
};

const VERDICT_STATUS: Record<RobustnessReport["verdict"], RobustnessStatus> = {
  robust: "pass",
  mixed: "warn",
  "likely overfit": "fail",
};

function StatusPill({ status, children }: { status: RobustnessStatus; children?: React.ReactNode }) {
  return (
    <span
      className={cn(
        "inline-flex min-w-11 justify-center rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase",
        STATUS_STYLE[status],
      )}
    >
      {children ?? status}
    </span>
  );
}

function num(value: number | null | undefined, digits = 2) {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function multiple(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "—";
  return value < 1e6 ? `${Math.round(value).toLocaleString()}x` : `${value.toPrecision(2)}x`;
}

function pct(value: number | null | undefined, digits = 1) {
  return value == null || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(digits)}%`;
}

/** Sharpe cell tinted by how much of the base Sharpe it keeps. */
function sharpeTone(value: number | null, base: number | null) {
  if (value == null || base == null || base <= 0) return "";
  const kept = value / base;
  if (kept >= 0.75) return "text-[var(--gain)]";
  if (kept >= 0.5) return "text-amber-600 dark:text-amber-400";
  return "text-[var(--loss)]";
}

function Section({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-1.5">
      <h3 className="text-xs font-medium text-muted-foreground">{title}</h3>
      {note ? <p className="text-[11px] text-muted-foreground">{note}</p> : null}
      {children}
    </section>
  );
}

function NullChart({ item }: { item: RobustnessNull }) {
  const data = item.histogram.map((bin) => ({ mid: (bin.lo + bin.hi) / 2, count: bin.count }));
  const bins = item.histogram;
  const width = bins.length ? bins[0].hi - bins[0].lo : 0;
  // The bins span the null and the actual Sharpe; headroom on the right keeps the "Actual" label in view.
  const lo = bins.length ? bins[0].lo : 0;
  const hi = bins.length ? bins[bins.length - 1].hi : 1;
  const domain: [number, number] = [lo, hi + (hi - lo) * 0.08];
  return (
    <div className="flex flex-col gap-1 rounded-md border border-border/60 p-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2 text-[11px]">
        <span className="font-medium">{item.shift_exits ? "Entries and exits shifted" : "Entries shifted"}</span>
        <span className="font-mono text-muted-foreground">
          {item.p < 0.001 ? "p < 0.001" : `p = ${item.p.toFixed(3)}`} · 95th pct {num(item.null_p95)} · actual {num(item.sharpe)}
        </span>
      </div>
      <ChartContainer config={nullChartConfig} className="h-[150px] w-full">
        <BarChart data={data} margin={{ left: 0, right: 8, top: 14, bottom: 0 }} barCategoryGap={1}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="mid"
            type="number"
            domain={domain}
            allowDataOverflow
            tickLine={false}
            axisLine={false}
            tickFormatter={(v: number) => v.toFixed(1)}
            fontSize={10}
          />
          <YAxis tickLine={false} axisLine={false} width={32} fontSize={10} />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                labelFormatter={(_, payload) => {
                  const mid = Number(payload?.[0]?.payload?.mid);
                  return `Sharpe ${(mid - width / 2).toFixed(2)} to ${(mid + width / 2).toFixed(2)}`;
                }}
              />
            }
          />
          <Bar dataKey="count" fill="var(--color-count)" fillOpacity={0.7} isAnimationActive={false} />
          <ReferenceLine
            x={item.sharpe}
            stroke="var(--gain)"
            strokeWidth={2}
            label={{ value: "Actual", position: "top", fontSize: 10, fill: "var(--gain)" }}
          />
          <ReferenceLine x={item.null_p95} stroke="var(--muted-foreground)" strokeDasharray="4 3" />
        </BarChart>
      </ChartContainer>
    </div>
  );
}

function YearChart({ years }: { years: RobustnessReport["stability"]["years"] }) {
  const data = years.map((y) => ({ year: String(y.year), pct: y.return == null ? 0 : y.return * 100, trades: y.trades }));
  return (
    <ChartContainer config={yearChartConfig} className="h-[160px] w-full">
      <BarChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="year" tickLine={false} axisLine={false} fontSize={10} interval="preserveStartEnd" />
        <YAxis tickLine={false} axisLine={false} width={40} fontSize={10} tickFormatter={(v: number) => `${v}%`} />
        <ChartTooltip
          cursor={false}
          content={
            <ChartTooltipContent
              formatter={(value, _name, item) => `${Number(value).toFixed(1)}% · ${item.payload.trades} trades`}
            />
          }
        />
        <ReferenceLine y={0} stroke="var(--border)" />
        <Bar dataKey="pct" isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.year} fill={d.pct >= 0 ? "var(--gain)" : "var(--loss)"} fillOpacity={0.75} />
          ))}
        </Bar>
      </BarChart>
    </ChartContainer>
  );
}

function ScoreTable({
  head,
  rows,
  base,
}: {
  head: string[];
  rows: { key: string; cells: React.ReactNode[]; sharpe: number | null; extra?: React.ReactNode }[];
  base: number | null;
}) {
  return (
    <div className="max-h-[320px] overflow-auto rounded-md border border-border/60">
      <Table className="text-xs">
        <TableHeader>
          <TableRow>
            {head.map((h) => (
              <TableHead key={h} className="h-7 px-2 text-[11px]">
                {h}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.key}>
              {row.cells.map((cell, i) => (
                <TableCell key={i} className="px-2 py-1 font-mono tabular-nums">
                  {cell}
                </TableCell>
              ))}
              <TableCell className={cn("px-2 py-1 font-mono font-semibold tabular-nums", sharpeTone(row.sharpe, base))}>
                {num(row.sharpe)}
              </TableCell>
              {row.extra}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export function RobustnessPanel({
  report,
  canRun,
  isRunning,
  error,
  onRun,
}: {
  report: RobustnessReport | null;
  canRun: boolean;
  isRunning: boolean;
  error: unknown;
  onRun: () => void;
}) {
  const base = report?.baseline.sharpe ?? null;
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <p className="max-w-2xl text-xs text-muted-foreground">
          Tests whether the backtest reflects a real edge or a fit to noise: random-timing nulls, nudged
          thresholds, each condition removed, stability over time, trade concentration, and execution
          with later fills and costs.
        </p>
        <Button size="sm" disabled={!canRun || isRunning} onClick={onRun}>
          {isRunning ? <Loader2 className="size-3.5 animate-spin" /> : null}
          {isRunning ? "Running…" : report ? "Run again" : "Run robustness checks"}
        </Button>
      </div>
      {!canRun ? (
        <p className="text-xs text-muted-foreground">Available for Strategy Builder backtests on daily data.</p>
      ) : null}
      {error ? (
        <p className="text-xs text-[var(--loss)]">{error instanceof Error ? error.message : "Robustness checks failed"}</p>
      ) : null}

      {report ? (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill status={VERDICT_STATUS[report.verdict]}>{report.verdict}</StatusPill>
            <span className="font-mono text-[11px] text-muted-foreground">
              Sharpe {num(base)} · CAGR {pct(report.baseline.cagr)} · {report.baseline.trades} trades in{" "}
              {report.baseline.traded_symbol} since {report.baseline.start}
            </span>
          </div>

          <ul className="flex flex-col gap-1.5">
            {report.checks.map((check) => (
              <li key={check.id} className="flex items-start gap-2 text-xs">
                <StatusPill status={check.status} />
                <span>
                  <span className="font-medium">{check.title}.</span>{" "}
                  <span className="text-muted-foreground">{check.headline}</span>
                </span>
              </li>
            ))}
          </ul>

          {report.random_timing.length ? (
            <Section
              title="Random timing"
              note="The same entry signals circularly shifted against prices. Frequency and clustering survive; the link to what happens next does not. The actual Sharpe should sit far right of the pile."
            >
              <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
                {report.random_timing.map((item) => (
                  <NullChart key={String(item.shift_exits)} item={item} />
                ))}
              </div>
            </Section>
          ) : null}

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <Section
              title="Parameter neighbourhood"
              note={
                report.neighbourhood.summary
                  ? `Each threshold moved ±10% and ±25%, limits nudged. Median Sharpe ${num(report.neighbourhood.summary.median_sharpe)}, ${pct(report.neighbourhood.summary.share_within_75pct, 0)} keep ≥ 75% of the base.`
                  : "No numeric thresholds to nudge."
              }
            >
              <ScoreTable
                head={["Original", "Variant", "CAGR", "Trades", "Sharpe"]}
                base={base}
                rows={report.neighbourhood.rows.map((r) => ({
                  key: `${r.condition}-${r.label}`,
                  cells: [r.condition, r.label, pct(r.cagr), r.trades],
                  sharpe: r.sharpe,
                }))}
              />
            </Section>
            <Section
              title="Ablation"
              note="Each condition removed once. A condition earns its place when removing it costs more than 10% of the Sharpe."
            >
              <ScoreTable
                head={["Removed", "CAGR", "Trades", "Sharpe", "Sharpe lost", ""]}
                base={base}
                rows={report.ablation.map((r) => ({
                  key: `${r.side}-${r.condition}`,
                  cells: [`${r.side === "exit" ? "Exit: " : ""}${r.condition}`, pct(r.cagr), r.trades],
                  sharpe: r.sharpe,
                  extra: (
                    <>
                      <TableCell className="px-2 py-1 font-mono tabular-nums">{num(r.sharpe_lost)}</TableCell>
                      <TableCell className="px-2 py-1">
                        {r.earns_place == null ? null : (
                          <StatusPill status={r.earns_place ? "pass" : "warn"}>{r.earns_place ? "works" : "idle"}</StatusPill>
                        )}
                      </TableCell>
                    </>
                  ),
                }))}
              />
            </Section>
          </div>

          <Section
            title="Stability over time"
            note="Sharpe over plain daily returns per window (no year left out)."
          >
            <div className="grid grid-cols-3 gap-1.5 sm:grid-cols-6">
              {report.stability.periods.map((p) => (
                <StatCard key={p.label} compact label={p.label} value={num(p.sharpe)} />
              ))}
            </div>
            <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
              <StatCard
                compact
                label="Years positive"
                value={`${report.stability.years_positive}/${report.stability.years_counted}`}
              />
              <StatCard
                compact
                label="CAGR, any year left out"
                value={
                  report.stability.cagr_leave_one_year_out
                    ? `${pct(report.stability.cagr_leave_one_year_out.min)} – ${pct(report.stability.cagr_leave_one_year_out.max)}`
                    : "—"
                }
              />
              <StatCard compact label="Time in market" value={pct(report.stability.time_in_market)} />
              <StatCard compact label="Max drawdown" value={pct(report.stability.max_drawdown)} tone="loss" />
            </div>
            <YearChart years={report.stability.years} />
          </Section>

          {report.stability.concentration ? (
            <Section title="Trade concentration">
              <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4 lg:grid-cols-7">
                <StatCard compact label="Win rate" value={pct(report.stability.concentration.win_rate)} />
                <StatCard
                  compact
                  label="Avg win / loss"
                  value={`${pct(report.stability.concentration.avg_win)} / ${pct(report.stability.concentration.avg_loss)}`}
                />
                <StatCard compact label="Profit factor" value={num(report.stability.concentration.profit_factor)} />
                <StatCard
                  compact
                  label="Top 10 trades' share"
                  value={pct(report.stability.concentration.top10_share_of_growth, 0)}
                />
                <StatCard
                  compact
                  label="Growth"
                  value={multiple(report.stability.concentration.growth_multiple)}
                />
                <StatCard
                  compact
                  label="Growth w/o top 10"
                  value={multiple(report.stability.concentration.growth_multiple_ex_top10)}
                />
                <StatCard
                  compact
                  label="Longest losing streak"
                  value={String(report.stability.concentration.longest_losing_streak)}
                />
              </div>
            </Section>
          ) : null}

          <Section
            title="Execution"
            note="Next-open entry: the first held bar earns only open-to-close at the general leverage. Costs are charged on entry and on exit, as a share of the position."
          >
            <ScoreTable
              head={["Entry fill", "Cost per leg", "CAGR", "Trades", "Sharpe"]}
              base={base}
              rows={[
                ...report.execution.fills.map((r) => ({
                  key: `${r.fill}-${r.cost_per_leg}`,
                  cells: [
                    r.fill === "close" ? "Signal close" : "Next open",
                    `${(r.cost_per_leg * 100).toFixed(3)}%`,
                    pct(r.cagr),
                    r.trades,
                  ],
                  sharpe: r.sharpe,
                })),
                ...report.execution.delays.map((r) => ({
                  key: `delay-${r.delay_bars}`,
                  cells: [`Close, ${r.delay_bars} bar${r.delay_bars > 1 ? "s" : ""} late`, "0%", pct(r.cagr), r.trades],
                  sharpe: r.sharpe,
                })),
              ]}
            />
          </Section>
        </>
      ) : null}
    </div>
  );
}
