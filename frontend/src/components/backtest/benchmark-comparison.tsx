import type { BenchmarkPayload, EquityCurvePoint } from "@/api";
import { formatMetric } from "@/lib/format-metric";

type Summary = Record<string, number | string | null>;

const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });

/** Growth of the starting capital, as a multiple: x2.5 is +150%. */
function formatMultiple(totalReturn: number | null | undefined) {
  if (typeof totalReturn !== "number" || !Number.isFinite(totalReturn)) return "-";
  const multiple = 1 + totalReturn;
  return `×${multiple >= 1000 ? compact.format(multiple) : multiple.toFixed(multiple >= 10 ? 1 : 2)}`;
}

type Row = {
  key: string;
  label: string;
  totalReturn: number | null;
  cagr: number | null;
  sharpe: number | null;
  sortino: number | null;
  maxDrawdown: number | null;
  timeInMarket: number | null;
  emphasis?: boolean;
};

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * The result next to holding the instrument all the time. Every row uses the result cards' basis:
 * CAGR, Sharpe and Sortino leave out that series' own best year, drawdown covers every year.
 */
export function BenchmarkComparison({
  summary,
  curve,
  benchmark,
}: {
  summary: Summary;
  curve: EquityCurvePoint[];
  benchmark: BenchmarkPayload;
}) {
  const first = curve[0]?.rolling_pnl;
  const last = curve[curve.length - 1]?.rolling_pnl;
  const rows: Row[] = [
    {
      key: "strategy",
      label: "This result",
      totalReturn: first && last ? last / first - 1 : null,
      cagr: num(summary.cagr_percent),
      sharpe: num(summary.sharpe),
      sortino: num(summary.sortino),
      maxDrawdown: num(summary.full_max_drawdown) ?? num(summary.max_drawdown),
      timeInMarket: benchmark.time_in_market_percent,
      emphasis: true,
    },
    ...benchmark.series.map((item) => ({
      key: item.id,
      label: item.label,
      totalReturn: item.metrics.total_return,
      cagr: item.metrics.cagr_percent,
      sharpe: item.metrics.sharpe,
      sortino: item.metrics.sortino,
      maxDrawdown: item.metrics.max_drawdown,
      timeInMarket: 100,
    })),
  ];
  const head = "px-2 py-1 text-right font-medium";
  return (
    <div className="mt-2 overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-[11px]">
        <thead className="text-muted-foreground">
          <tr className="border-b border-border/60">
            <th className="px-2 py-1 text-left font-medium">Versus buy &amp; hold</th>
            <th className={head}>Growth</th>
            <th className={head}>CAGR</th>
            <th className={head}>Sharpe</th>
            <th className={head}>Sortino</th>
            <th className={head}>Max drawdown</th>
            <th className={head}>In market</th>
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {rows.map((row) => (
            <tr key={row.key} className={row.emphasis ? "font-semibold" : "text-muted-foreground"}>
              <td className="px-2 py-1 text-left font-sans">{row.label}</td>
              <td className="px-2 py-1 text-right">{formatMultiple(row.totalReturn)}</td>
              <td className="px-2 py-1 text-right">{formatMetric(row.cagr, "cagr_percent")}</td>
              <td className="px-2 py-1 text-right">{formatMetric(row.sharpe, "sharpe")}</td>
              <td className="px-2 py-1 text-right">{formatMetric(row.sortino, "sortino")}</td>
              <td className="px-2 py-1 text-right">{formatMetric(row.maxDrawdown, "max_drawdown")}</td>
              <td className="px-2 py-1 text-right">
                {row.timeInMarket === null ? "-" : `${row.timeInMarket.toFixed(0)}%`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
