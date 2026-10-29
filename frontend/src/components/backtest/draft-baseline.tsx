import type { DraftBaseline, DraftBaselineMetrics, SweepMeta } from "@/api";

const COLUMNS: Array<{ key: keyof DraftBaselineMetrics; label: string }> = [
  { key: "PnL", label: "PnL" },
  { key: "MaxDD", label: "MaxDD" },
  { key: "Trades", label: "Trades" },
  { key: "%Pstv", label: "%Pstv" },
  { key: "CAGR", label: "CAGR" },
  { key: "Sharpe", label: "Sharpe" },
  { key: "Sortino", label: "Sortino" },
];

function show(value: unknown) {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  return String(value);
}

function numeric(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  return Number.isFinite(parsed) ? parsed : null;
}

/** How many sweep rows score above the draft's in-sample Sharpe, or null when that cannot be said. */
export function rowsBeatingDraft(rows: Array<Record<string, unknown>>, draft: DraftBaseline) {
  const draftSharpe = numeric(draft.in_sample.Sharpe);
  if (draftSharpe === null) return null;
  const scored = rows.map((row) => numeric(row.Sharpe)).filter((value): value is number => value !== null);
  if (scored.length === 0) return null;
  return { beating: scored.filter((value) => value > draftSharpe).length, total: scored.length, draftSharpe };
}

/**
 * The draft as it stands, in the sweep's own columns, so a row can be read against it. Ranking
 * figures here and in the rows leave out each series' best year, like every sweep.
 */
export function DraftBaselineView({
  draft,
  meta,
  rows,
}: {
  draft: DraftBaseline;
  meta?: SweepMeta;
  rows: Array<Record<string, unknown>>;
}) {
  const comparison = rowsBeatingDraft(rows, draft);
  const head = "px-2 py-1 text-right font-medium";
  const lines = [
    {
      key: "in_sample",
      label: meta ? `Current draft, in-sample through ${meta.in_sample_end}` : "Current draft, in-sample",
      metrics: draft.in_sample,
      emphasis: true,
    },
    {
      key: "full_period",
      label: meta ? `Current draft, full period through ${meta.period_end}` : "Current draft, full period",
      metrics: draft.full_period,
      emphasis: false,
    },
  ];
  return (
    <div className="mb-2 rounded-md border border-border/50 bg-muted/15 px-2 py-1.5">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] border-collapse text-[11px]">
          <thead className="text-muted-foreground">
            <tr className="border-b border-border/50">
              <th className="px-2 py-1 text-left font-medium">Draft as it stands</th>
              {COLUMNS.map((column) => (
                <th key={column.key} className={head}>
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {lines.map((line) => (
              <tr key={line.key} className={line.emphasis ? "font-semibold" : "text-muted-foreground"}>
                <td className="whitespace-nowrap px-2 py-1 text-left font-sans">{line.label}</td>
                {COLUMNS.map((column) => (
                  <td key={column.key} className="px-2 py-1 text-right">
                    {show(line.metrics[column.key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {comparison ? (
        <p className="px-2 pt-1 text-[11px] text-muted-foreground">
          {comparison.beating} of {comparison.total} rows score above the draft's in-sample Sharpe (
          {comparison.draftSharpe.toFixed(2)}). The more rows tested, the more are expected to by chance.
        </p>
      ) : null}
    </div>
  );
}
