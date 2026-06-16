import { useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
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
import {
  defaultSortColumn,
  defaultSortDirectionForColumn,
  sortTableRows,
  tableRowsSignature,
  type SortDirection,
} from "@/lib/sort-table-rows";
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
  sortable = true,
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
  /** Enable click-to-sort column headers. Defaults to Sharpe descending when present. */
  sortable?: boolean;
}) {
  const columns = useMemo(
    () => (rows.length ? Object.keys(rows[0]).filter((column) => !hiddenColumns.includes(column)) : []),
    [rows, hiddenColumns],
  );

  const [sort, setSort] = useState<{ column: string; direction: SortDirection } | null>(null);

  const rowsSignature = useMemo(() => tableRowsSignature(rows), [rows]);
  const columnKey = columns.join("\0");

  useEffect(() => {
    if (!sortable || !columns.length) {
      setSort(null);
      return;
    }
    const column = defaultSortColumn(columns);
    if (!column) {
      setSort(null);
      return;
    }
    setSort({
      column,
      direction: defaultSortDirectionForColumn(rows, column),
    });
  }, [rowsSignature, columnKey, sortable]);

  const displayRows = useMemo(() => {
    if (!sortable || !sort) return rows;
    return sortTableRows(rows, sort.column, sort.direction);
  }, [rows, sort, sortable]);

  if (!rows.length) {
    return <p className="text-xs text-muted-foreground">No rows returned.</p>;
  }

  const fixedHeightPx =
    compact && visibleRows !== undefined ? compactTableHeightPx(visibleRows) : undefined;
  const showAddColumn = Boolean(onAddRow);

  function handleSort(column: string) {
    setSort((current) => {
      if (current?.column === column) {
        return { column, direction: current.direction === "asc" ? "desc" : "asc" };
      }
      return {
        column,
        direction: defaultSortDirectionForColumn(rows, column),
      };
    });
  }

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
            {columns.map((column) => {
              const active = sort?.column === column;
              const SortIcon = active
                ? sort.direction === "asc"
                  ? ArrowUp
                  : ArrowDown
                : ArrowUpDown;
              return (
                <TableHead
                  key={column}
                  className={cn("whitespace-nowrap", compact ? compactHead : undefined)}
                  aria-sort={
                    sortable && active
                      ? sort.direction === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  {sortable ? (
                    <button
                      type="button"
                      className={cn(
                        "inline-flex items-center gap-1 text-left font-medium hover:text-foreground",
                        active ? "text-foreground" : "text-muted-foreground",
                      )}
                      onClick={() => handleSort(column)}
                    >
                      <span>{column}</span>
                      <SortIcon className={cn("size-3 shrink-0", active ? "opacity-100" : "opacity-40")} />
                    </button>
                  ) : (
                    column
                  )}
                </TableHead>
              );
            })}
          </TableRow>
        </TableHeader>
        <TableBody>
          {displayRows.map((row, idx) => {
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
