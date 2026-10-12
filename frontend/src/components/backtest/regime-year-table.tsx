import type { RegimeStateDef, RegimeStrategyYearRow, RegimeYearRow } from "@/api";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

type Row = RegimeYearRow | RegimeStrategyYearRow;

function isStrategyRow(row: Row): row is RegimeStrategyYearRow {
  return "trades" in row;
}

const head = "h-7 px-1 py-0 text-[11px] font-medium";
// A strategy table has up to 30 state columns, so cells hug their numbers.
const cell = "px-1 py-0.5 text-[11px] tabular-nums";

/** Column label turned to read bottom-to-top, so the column is only as wide as its numbers. */
function VerticalLabel({ children }: { children: string }) {
  return (
    <span className="inline-block rotate-180 whitespace-nowrap [writing-mode:vertical-rl]">{children}</span>
  );
}

function formatReturn(value: number | null) {
  if (value == null) return "—";
  const pct = value * 100;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(1)}%`;
}

/** Profitable trades as a share of trades. */
export function formatProfitablePct(profitable: number, trades: number) {
  return trades > 0 ? `${Math.round((profitable / trades) * 100)}%` : "—";
}

function Count({ value }: { value: number }) {
  return <span className={value === 0 ? "text-muted-foreground/60" : undefined}>{value}</span>;
}

/** Shade a days cell by how much of that year the state was active. */
function shadeStyle(days: number, total: number) {
  if (total <= 0 || days <= 0) return undefined;
  const share = Math.min(days / total, 1);
  return { backgroundColor: `color-mix(in oklab, var(--primary) ${(share * 40).toFixed(0)}%, transparent)` };
}

/**
 * One row per calendar year, one column per regime state holding the number of
 * sessions the state was active, plus how many times the regime changed state.
 * Rows that carry trade figures (a strategy analysis) add trades and profitable
 * trades per state, the year's total trades, and its return.
 */
export function RegimeYearTable({
  states,
  rows,
  className,
}: {
  states: RegimeStateDef[];
  rows: Row[];
  className?: string;
}) {
  const withTrades = rows.some(isStrategyRow);
  const newestFirst = [...rows].sort((a, b) => b.year - a.year);

  const totalDays = (key: string) => rows.reduce((sum, row) => sum + (row.days[key] ?? 0), 0);
  const totalChanges = rows.reduce((sum, row) => sum + row.changes, 0);
  const sumTrades = (key: string, field: "trades" | "profitable") =>
    rows.reduce((sum, row) => sum + (isStrategyRow(row) ? (row.trades[key]?.[field] ?? 0) : 0), 0);
  const allTrades = rows.reduce((sum, row) => sum + (isStrategyRow(row) ? row.total_trades : 0), 0);
  const stateSpan = withTrades ? 3 : 1;

  return (
    <Table className={cn("border-separate border-spacing-0", withTrades && "w-auto", className)}>
      <TableHeader>
        <TableRow className="hover:bg-transparent">
          <TableHead rowSpan={withTrades ? 2 : 1} className={head}>
            Year
          </TableHead>
          {states.map((state) => (
            <TableHead
              key={state.key}
              colSpan={stateSpan}
              className={cn(head, "whitespace-nowrap text-center", withTrades && "border-l border-border/60")}
            >
              {state.label}
            </TableHead>
          ))}
          {withTrades ? (
            <>
              <TableHead rowSpan={2} className={cn(head, "h-auto border-l border-border/60 px-1 py-1 text-center align-bottom")}>
                <VerticalLabel>Regime changes</VerticalLabel>
              </TableHead>
              <TableHead rowSpan={2} className={cn(head, "h-auto px-1 py-1 text-center align-bottom")}>
                <VerticalLabel>Total trades</VerticalLabel>
              </TableHead>
              <TableHead
                rowSpan={2}
                className={cn(head, "text-right align-bottom")}
                title="Return for the calendar year (not annualized for partial years)"
              >
                CAGR
              </TableHead>
            </>
          ) : (
            <TableHead className={cn(head, "border-l border-border/60 text-right")}>Regime changes</TableHead>
          )}
        </TableRow>
        {withTrades ? (
          <TableRow className="hover:bg-transparent">
            {states.map((state) => (
              <SubHeads key={state.key} />
            ))}
          </TableRow>
        ) : null}
      </TableHeader>
      <TableBody>
        {newestFirst.map((row) => (
          <TableRow key={row.year}>
            <TableCell className={cn(cell, "font-medium")}>{row.year}</TableCell>
            {states.map((state) => {
              const days = row.days[state.key] ?? 0;
              const counts = isStrategyRow(row) ? row.trades[state.key] : undefined;
              return (
                <StateCells
                  key={state.key}
                  days={days}
                  shade={shadeStyle(days, row.total_days)}
                  counts={withTrades ? (counts ?? { trades: 0, profitable: 0 }) : undefined}
                />
              );
            })}
            <TableCell className={cn(cell, "border-l border-border/60 text-right")}>
              <Count value={row.changes} />
            </TableCell>
            {withTrades && isStrategyRow(row) ? (
              <>
                <TableCell className={cn(cell, "text-right")}>
                  <Count value={row.total_trades} />
                </TableCell>
                <TableCell
                  className={cn(
                    cell,
                    "text-right",
                    row.return != null && (row.return >= 0 ? "text-[var(--gain)]" : "text-[var(--loss)]"),
                  )}
                >
                  {formatReturn(row.return)}
                </TableCell>
              </>
            ) : null}
          </TableRow>
        ))}
        <TableRow className="border-t-2 border-border font-medium hover:bg-transparent">
          <TableCell className={cell}>All years</TableCell>
          {states.map((state) => (
            <StateCells
              key={state.key}
              days={totalDays(state.key)}
              counts={
                withTrades
                  ? { trades: sumTrades(state.key, "trades"), profitable: sumTrades(state.key, "profitable") }
                  : undefined
              }
            />
          ))}
          <TableCell className={cn(cell, "border-l border-border/60 text-right")}>{totalChanges}</TableCell>
          {withTrades ? (
            <>
              <TableCell className={cn(cell, "text-right")}>{allTrades}</TableCell>
              <TableCell className={cn(cell, "text-right text-muted-foreground")}>—</TableCell>
            </>
          ) : null}
        </TableRow>
      </TableBody>
    </Table>
  );
}

function SubHeads() {
  const sub = "h-auto px-1 py-1 text-center align-bottom font-normal text-muted-foreground";
  return (
    <>
      <TableHead className={cn(head, sub, "border-l border-border/60")}>
        <VerticalLabel>Days</VerticalLabel>
      </TableHead>
      <TableHead className={cn(head, sub)}>
        <VerticalLabel>Trades</VerticalLabel>
      </TableHead>
      <TableHead className={cn(head, sub)}>
        <VerticalLabel>% Profitable</VerticalLabel>
      </TableHead>
    </>
  );
}

function StateCells({
  days,
  shade,
  counts,
}: {
  days: number;
  shade?: { backgroundColor: string };
  counts?: { trades: number; profitable: number };
}) {
  return (
    <>
      <TableCell
        className={cn(cell, "text-right", counts && "border-l border-border/60")}
        style={shade}
      >
        <Count value={days} />
      </TableCell>
      {counts ? (
        <>
          <TableCell className={cn(cell, "text-right")}>
            <Count value={counts.trades} />
          </TableCell>
          <TableCell className={cn(cell, "text-right", counts.trades === 0 && "text-muted-foreground/60")}>
            {formatProfitablePct(counts.profitable, counts.trades)}
          </TableCell>
        </>
      ) : null}
    </>
  );
}
