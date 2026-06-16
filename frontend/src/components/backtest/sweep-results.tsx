import { memo } from "react";
import type { SweepResult } from "@/api";
import { negativeYearRowClass, ResultsTable } from "@/components/backtest/results-table";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export const SweepResults = memo(function SweepResults({
  rows,
  selectedRow,
  onSelectRow,
  embedded = false,
  onAddRow,
  canAddRow,
  addRowLabel,
}: {
  rows: SweepResult;
  selectedRow?: Record<string, unknown>;
  onSelectRow: (row: Record<string, unknown>) => void;
  embedded?: boolean;
  onAddRow?: (row: Record<string, unknown>) => void;
  canAddRow?: (row: Record<string, unknown>) => boolean;
  addRowLabel?: (row: Record<string, unknown>) => string;
}) {
  const yearlyRows = Array.isArray(selectedRow?.Yearly)
    ? [...(selectedRow.Yearly as Array<Record<string, unknown>>)].reverse()
    : [];

  const content = (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2">
        {!embedded && <h2 className="text-sm font-semibold">Sweep results</h2>}
        <Badge variant="outline" className="font-mono text-[10px]">
          {rows.length} rows
        </Badge>
      </div>

      <div className={cn(!embedded && "mt-2")}>
        <ResultsTable
          compact
          rows={rows}
          hiddenColumns={["Yearly", "SecondaryId"]}
          selectedRow={selectedRow}
          onRowClick={onSelectRow}
          onAddRow={onAddRow}
          canAddRow={canAddRow}
          addRowLabel={addRowLabel}
        />
      </div>

      {yearlyRows.length > 0 && (
        <div className="mt-3 flex flex-col gap-1">
          <h3 className="text-xs font-medium text-muted-foreground">Selected outcome — yearly</h3>
          <ResultsTable compact visibleRows={30} rows={yearlyRows} rowClassName={negativeYearRowClass} />
        </div>
      )}
    </>
  );

  if (embedded) {
    return content;
  }

  return <Card className="border-border/60 p-3">{content}</Card>;
});
