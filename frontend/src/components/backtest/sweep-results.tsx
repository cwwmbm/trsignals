import { memo, useEffect, useMemo, useState } from "react";
import type { SweepMeta, SweepResult } from "@/api";
import { negativeYearRowClass, ResultsTable } from "@/components/backtest/results-table";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

type FullPeriodPayload = {
  PnL?: unknown;
  MaxDD?: unknown;
  Trades?: unknown;
  Sharpe?: unknown;
  Yearly?: Array<Record<string, unknown>>;
};

function fullPeriodFromRow(row?: Record<string, unknown>): FullPeriodPayload | undefined {
  const value = row?.FullPeriod;
  if (!value || typeof value !== "object") return undefined;
  return value as FullPeriodPayload;
}

export const SweepResults = memo(function SweepResults({
  rows,
  selectedRow,
  onSelectRow,
  embedded = false,
  onAddRow,
  canAddRow,
  addRowLabel,
  meta,
}: {
  rows: SweepResult;
  selectedRow?: Record<string, unknown>;
  onSelectRow: (row: Record<string, unknown>) => void;
  embedded?: boolean;
  onAddRow?: (row: Record<string, unknown>) => void;
  canAddRow?: (row: Record<string, unknown>) => boolean;
  addRowLabel?: (row: Record<string, unknown>) => string;
  meta?: SweepMeta;
}) {
  const [showFullPeriod, setShowFullPeriod] = useState(false);

  useEffect(() => {
    setShowFullPeriod(false);
  }, [selectedRow]);

  useEffect(() => {
    setShowFullPeriod(false);
  }, [rows, meta?.in_sample_end, meta?.period_end]);

  const fullPeriod = fullPeriodFromRow(selectedRow);
  const canToggleFullPeriod = Boolean(meta && fullPeriod);

  const yearlyRows = useMemo(() => {
    const source = showFullPeriod
      ? fullPeriod?.Yearly
      : (selectedRow?.Yearly as Array<Record<string, unknown>> | undefined);
    if (!Array.isArray(source)) return [];
    return [...source].reverse();
  }, [showFullPeriod, fullPeriod, selectedRow]);

  const fractionPct = meta
    ? Math.round(Number(meta.in_sample_fraction) * 100)
    : null;

  const content = (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2">
        {!embedded && <h2 className="text-sm font-semibold">Sweep results</h2>}
        <div className="flex flex-wrap items-center gap-2">
          {meta && fractionPct !== null && (
            <Badge variant="outline" className="font-mono text-[10px]">
              In-sample {fractionPct}% through {meta.in_sample_end}
            </Badge>
          )}
          <Badge variant="outline" className="font-mono text-[10px]">
            {rows.length} rows
          </Badge>
        </div>
      </div>

      <div className={cn(!embedded && "mt-2")}>
        <ResultsTable
          compact
          rows={rows}
          hiddenColumns={["Yearly", "SecondaryId", "FullPeriod"]}
          selectedRow={selectedRow}
          onRowClick={onSelectRow}
          onAddRow={onAddRow}
          canAddRow={canAddRow}
          addRowLabel={addRowLabel}
        />
      </div>

      {selectedRow && (
        <div className="mt-3 flex flex-col gap-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-xs font-medium text-muted-foreground">
              Selected outcome — yearly
              {showFullPeriod
                ? meta
                  ? ` (full period through ${meta.period_end})`
                  : " (full period)"
                : meta
                  ? ` (in-sample through ${meta.in_sample_end})`
                  : " (in-sample)"}
            </h3>
            {canToggleFullPeriod && (
              <div className="flex items-center gap-2">
                <Switch
                  id="refine-full-period"
                  size="sm"
                  checked={showFullPeriod}
                  onCheckedChange={(checked) => setShowFullPeriod(Boolean(checked))}
                />
                <Label
                  htmlFor="refine-full-period"
                  className="text-xs font-normal text-muted-foreground"
                >
                  Show full period
                </Label>
              </div>
            )}
          </div>

          {showFullPeriod && fullPeriod && (
            <div className="flex flex-wrap gap-3 font-mono text-[11px] text-muted-foreground">
              <span>PnL {String(fullPeriod.PnL)}</span>
              <span>MaxDD {String(fullPeriod.MaxDD)}</span>
              <span>Trades {String(fullPeriod.Trades)}</span>
              <span>Sharpe {String(fullPeriod.Sharpe)}</span>
            </div>
          )}

          {yearlyRows.length > 0 ? (
            <ResultsTable
              compact
              visibleRows={30}
              rows={yearlyRows}
              rowClassName={negativeYearRowClass}
              sortable={false}
            />
          ) : (
            <p className="text-xs text-muted-foreground">No yearly breakdown for this row.</p>
          )}
        </div>
      )}
    </>
  );

  if (embedded) {
    return content;
  }

  return <Card className="border-border/60 p-3">{content}</Card>;
});
