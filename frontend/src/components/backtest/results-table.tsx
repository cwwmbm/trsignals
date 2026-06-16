import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";
import { formatMetric } from "@/lib/format-metric";
import { cn } from "@/lib/utils";

const compactHead = "h-7 px-1.5 py-0 text-[11px] font-medium";
const compactCell = "px-1.5 py-0.5";
const COMPACT_HEADER_PX = 28;
const COMPACT_ROW_PX = 22;

export function compactTableHeightPx(visibleRows: number) {
  return COMPACT_HEADER_PX + visibleRows * COMPACT_ROW_PX;
}

export function ResultsTable({
  rows,
  hiddenColumns = [],
  rowClassName,
  onRowClick,
  selectedRow,
  compact = false,
  visibleRows,
  onAddRow,
  canAddRow,
  addRowLabel,
}: {
  rows: Array<Record<string, unknown>>;
  hiddenColumns?: string[];
  rowClassName?: (row: Record<string, unknown>) => string | undefined;
  onRowClick?: (row: Record<string, unknown>) => void;
  selectedRow?: Record<string, unknown>;
  compact?: boolean;
  /** Fixed viewport height in compact mode (header + N data rows). */
  visibleRows?: number;
  onAddRow?: (row: Record<string, unknown>) => void;
  canAddRow?: (row: Record<string, unknown>) => boolean;
  addRowLabel?: (row: Record<string, unknown>) => string;
}) {
  if (!rows.length) {
    return <p className="text-xs text-muted-foreground">No rows returned.</p>;
  }

  const columns = Object.keys(rows[0]).filter((column) => !hiddenColumns.includes(column));
  const fixedHeightPx =
    compact && visibleRows !== undefined ? compactTableHeightPx(visibleRows) : undefined;
  const showAddColumn = Boolean(onAddRow);

  return (
    <div
      className={cn(
        "overflow-auto rounded-md border border-border/60",
        compact && fixedHeightPx === undefined && "max-h-[280px]",
        !compact && "max-h-[360px] rounded-lg",
      )}
      style={
        fixedHeightPx !== undefined
          ? { height: fixedHeightPx, maxHeight: fixedHeightPx }
          : undefined
      }
    >
      <Table className={compact ? "text-xs" : undefined}>
        <TableHeader className={cn("sticky top-0 z-10 bg-card", compact && "bg-muted/30")}>
          <TableRow className={compact ? "hover:bg-transparent" : undefined}>
            {showAddColumn && (
              <TableHead className={cn("w-8 px-1", compact ? compactHead : undefined)} />
            )}
            {columns.map((column) => (
              <TableHead
                key={column}
                className={cn("whitespace-nowrap", compact ? compactHead : undefined)}
              >
                {column}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row, idx) => {
            const extra = rowClassName?.(row);
            const selected = selectedRow === row;
            const addable = showAddColumn && (canAddRow?.(row) ?? true);
            return (
              <TableRow
                key={idx}
                className={cn(
                  extra,
                  compact && "hover:bg-muted/30",
                  onRowClick && "cursor-pointer hover:bg-muted/50",
                  selected && "bg-primary/10",
                )}
                onClick={() => onRowClick?.(row)}
              >
                {showAddColumn && (
                  <TableCell className={cn("w-8 px-1", compact ? compactCell : undefined)}>
                    {addable ? (
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-sm"
                        className="size-6 text-muted-foreground hover:text-foreground"
                        title={addRowLabel?.(row) ?? "Add condition"}
                        aria-label={addRowLabel?.(row) ?? "Add condition"}
                        onClick={(event) => {
                          event.stopPropagation();
                          onAddRow?.(row);
                        }}
                      >
                        <Plus className="size-3.5" />
                      </Button>
                    ) : null}
                  </TableCell>
                )}
                {columns.map((column) => (
                  <TableCell
                    key={column}
                    className={cn(
                      "font-mono tabular-nums",
                      compact ? cn(compactCell, "text-[11px]") : "text-xs",
                    )}
                  >
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
