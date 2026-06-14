import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatMetric } from "@/lib/format-metric";
import { cn } from "@/lib/utils";

export function ResultsTable({
  rows,
  hiddenColumns = [],
  rowClassName,
  onRowClick,
  selectedRow,
}: {
  rows: Array<Record<string, unknown>>;
  hiddenColumns?: string[];
  rowClassName?: (row: Record<string, unknown>) => string | undefined;
  onRowClick?: (row: Record<string, unknown>) => void;
  selectedRow?: Record<string, unknown>;
}) {
  if (!rows.length) {
    return <p className="text-sm text-muted-foreground">No rows returned.</p>;
  }

  const columns = Object.keys(rows[0]).filter((column) => !hiddenColumns.includes(column));

  return (
    <div className="max-h-[360px] overflow-auto rounded-lg border border-border/60">
      <Table>
        <TableHeader className="sticky top-0 z-10 bg-card">
          <TableRow>
            {columns.map((column) => (
              <TableHead key={column} className="whitespace-nowrap">
                {column}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row, idx) => {
            const extra = rowClassName?.(row);
            const selected = selectedRow === row;
            return (
              <TableRow
                key={idx}
                className={cn(
                  extra,
                  onRowClick && "cursor-pointer hover:bg-muted/50",
                  selected && "bg-primary/10",
                )}
                onClick={() => onRowClick?.(row)}
              >
                {columns.map((column) => (
                  <TableCell key={column} className="font-mono text-xs tabular-nums">
                    {formatMetric(row[column], column)}
                  </TableCell>
                ))}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

export function negativeYearRowClass(row: Record<string, unknown>) {
  return typeof row.pnl_percent === "number" && row.pnl_percent < 0 ? "bg-[var(--loss)]/8" : undefined;
}

export function tradeRowClass(row: Record<string, unknown>) {
  if (row.status === "Open") return "bg-muted/40";
  return typeof row.trade_pnl === "number" && row.trade_pnl < 0 ? "bg-[var(--loss)]/8" : undefined;
}
