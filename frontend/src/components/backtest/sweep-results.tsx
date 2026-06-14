import type { SweepResult } from "@/api";
import { negativeYearRowClass, ResultsTable } from "@/components/backtest/results-table";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

export function SweepResults({
  rows,
  selectedRow,
  onSelectRow,
}: {
  rows: SweepResult;
  selectedRow?: Record<string, unknown>;
  onSelectRow: (row: Record<string, unknown>) => void;
}) {
  const yearlyRows = Array.isArray(selectedRow?.Yearly)
    ? [...(selectedRow.Yearly as Array<Record<string, unknown>>)].reverse()
    : [];

  return (
    <Card className="border-border/60 p-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">Sweep Results</h2>
        <Badge variant="outline" className="font-mono text-xs">
          {rows.length} rows
        </Badge>
      </div>

      <div className="mt-4">
        <ResultsTable
          rows={rows}
          hiddenColumns={["Yearly"]}
          selectedRow={selectedRow}
          onRowClick={onSelectRow}
        />
      </div>

      {yearlyRows.length > 0 && (
        <div className="mt-6 flex flex-col gap-3">
          <h3 className="text-sm font-semibold">Selected Outcome Annual Breakdown</h3>
          <ResultsTable rows={yearlyRows} rowClassName={negativeYearRowClass} />
        </div>
      )}
    </Card>
  );
}
