import type { IndicatorInfo } from "@/api";
import type { ConditionRow } from "@/components/strategy-builder/condition-list";
import { newConditionRow } from "@/components/strategy-builder/condition-list";
import { isFlagIndicator } from "@/lib/strategy-builder";

export function isIndicatorSweepRow(row: Record<string, unknown>): boolean {
  return (
    row.Indicator !== undefined &&
    row.Indicator !== null &&
    String(row.Indicator).trim() !== "" &&
    row.Condition !== undefined &&
    row.Value !== undefined
  );
}

/** An indicator combo sweep row: several threshold filters that are added to the entry together. */
export function isComboSweepRow(row: Record<string, unknown>): boolean {
  return Array.isArray(row.Filters) && row.Filters.length > 1;
}

export function comboSweepRowToConditionRows(
  row: Record<string, unknown>,
  indicators: IndicatorInfo[],
): ConditionRow[] | null {
  if (!isComboSweepRow(row)) return null;
  const mapped: ConditionRow[] = [];
  for (const filter of row.Filters as Array<Record<string, unknown>>) {
    const next = sweepRowToConditionRow({ ...filter, Buysell: "Buy" }, indicators);
    if (!next) return null;
    mapped.push(next);
  }
  return mapped;
}

/** A Market regime sweep row. Its `RegimeId` is the `Regime_<regime>_<state>` condition id. */
export function isRegimeSweepRow(row: Record<string, unknown>): boolean {
  return typeof row.RegimeId === "string" && row.RegimeId.trim() !== "";
}

/** "is true" keeps only that state. "is false" keeps every other labeled state. */
export function regimeSweepRowOperator(row: Record<string, unknown>): "is true" | "is false" {
  return row.RegimeOperator === "is false" ? "is false" : "is true";
}

export function regimeSweepRowId(row: Record<string, unknown>): string | null {
  return isRegimeSweepRow(row) ? String(row.RegimeId).trim() : null;
}

export function isHoldDaysSweepRow(row: Record<string, unknown>): boolean {
  return row.Days !== undefined && row.Prf !== undefined;
}

export function isStrategyComboSweepRow(row: Record<string, unknown>): boolean {
  const mode = String(row.Mode ?? "").toUpperCase();
  return (
    row.SecondaryId !== undefined &&
    row.SecondaryId !== null &&
    String(row.SecondaryId).trim() !== "" &&
    (mode === "AND" || mode === "OR")
  );
}

export function isSymbolConfirmSweepRow(row: Record<string, unknown>): boolean {
  return row.Confirm !== undefined && row.Primary !== undefined;
}

export function symbolConfirmSweepRowValues(row: Record<string, unknown>): string | null {
  if (!isSymbolConfirmSweepRow(row)) return null;

  const confirm = String(row.Confirm ?? "").trim();
  if (!confirm || confirm === "(none)") return "";
  return confirm
    .split("+")
    .map((item) => item.trim().toUpperCase())
    .filter(Boolean)
    .join(", ");
}

export function holdDaysSweepRowValues(
  row: Record<string, unknown>,
): { holdDays: number; profit: number } | null {
  if (!isHoldDaysSweepRow(row)) return null;

  const holdDays = Number(row.Days);
  const profit = Number(row.Prf);
  if (!Number.isFinite(holdDays) || holdDays < 1) return null;
  if (!Number.isFinite(profit) || profit < 0) return null;

  return { holdDays: Math.trunc(holdDays), profit: Math.trunc(profit) };
}

export function sweepRowSide(row: Record<string, unknown>): "Buy" | "Sell" | null {
  const raw = row.Buysell ?? row.BuySell ?? row.buysell;
  if (raw === "Buy" || raw === "Sell") return raw;
  return null;
}

export function resolveIndicatorId(name: string, indicators: IndicatorInfo[]): string | null {
  const trimmed = name.trim();
  if (!trimmed) return null;

  const exact = indicators.find((item) => item.id === trimmed);
  if (exact) return exact.id;

  const lower = trimmed.toLowerCase();
  for (const item of indicators) {
    if (item.id.toLowerCase() === lower) return item.id;
    if (item.aliases?.some((alias) => alias.toLowerCase() === lower)) return item.id;
  }
  return null;
}

function formatSweepValue(value: unknown): string {
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    const fixed = value.toFixed(2).replace(/\.?0+$/, "");
    return fixed;
  }
  return String(value).trim();
}

/** The condition an Entry condition check row ran without, and where it sat in the draft. */
export type RemovedCondition = { index: number; left: string; operator: string; right: string };

export function entryCheckRowRemoval(row: Record<string, unknown>): RemovedCondition | null {
  const value = row.RemovedCondition;
  if (!value || typeof value !== "object") return null;
  const removed = value as Record<string, unknown>;
  if (typeof removed.index !== "number" || typeof removed.left !== "string") return null;
  return {
    index: removed.index,
    left: removed.left,
    operator: String(removed.operator ?? ""),
    right: String(removed.right ?? ""),
  };
}

export function isEntryCheckRow(row: Record<string, unknown>): boolean {
  return entryCheckRowRemoval(row) !== null;
}

/**
 * Where the removed condition is in the builder now. The draft sends market regimes first, then
 * the entry conditions, so the row's index is checked there first. If the draft has changed since
 * the check ran (another row removed, say), the first condition that matches is used instead.
 */
export function locateRemovedCondition(
  removed: RemovedCondition,
  regimes: ConditionRow[],
  entries: ConditionRow[],
): { list: "regime" | "entry"; index: number } | null {
  const same = (item: ConditionRow) =>
    item.left === removed.left && item.operator === removed.operator && item.right === removed.right;
  const lists = [
    { list: "regime" as const, rows: regimes, offset: 0 },
    { list: "entry" as const, rows: entries, offset: regimes.length },
  ];
  for (const { list, rows, offset } of lists) {
    const index = removed.index - offset;
    if (index >= 0 && index < rows.length && same(rows[index])) return { list, index };
  }
  for (const { list, rows } of lists) {
    const index = rows.findIndex(same);
    if (index >= 0) return { list, index };
  }
  return null;
}

export function canAddSweepRowToBuilder(
  row: Record<string, unknown>,
  indicators: IndicatorInfo[],
): boolean {
  if (holdDaysSweepRowValues(row)) return true;
  if (isEntryCheckRow(row)) return true;
  if (isRegimeSweepRow(row)) return true;
  if (isStrategyComboSweepRow(row)) return true;
  if (isSymbolConfirmSweepRow(row)) return symbolConfirmSweepRowValues(row) !== null;
  if (isComboSweepRow(row)) return comboSweepRowToConditionRows(row, indicators) !== null;
  if (!isIndicatorSweepRow(row)) return false;
  if (!sweepRowSide(row)) return false;
  const indicator = String(row.Indicator);
  return resolveIndicatorId(indicator, indicators) !== null;
}

export function sweepRowToConditionRow(
  row: Record<string, unknown>,
  indicators: IndicatorInfo[],
): ConditionRow | null {
  if (isStrategyComboSweepRow(row)) {
    const mode = String(row.Mode).toUpperCase();
    return newConditionRow({
      left: `strategy:${String(row.SecondaryId).trim()}`,
      operator: "is true",
      right: "",
      logic: mode === "OR" ? "OR" : "AND",
    });
  }

  if (!isIndicatorSweepRow(row)) return null;

  const left = resolveIndicatorId(String(row.Indicator), indicators);
  if (!left) return null;

  const condition = String(row.Condition).toLowerCase();
  if (condition !== "more" && condition !== "less") return null;

  if (isFlagIndicator(indicators, left)) {
    return newConditionRow({
      left,
      operator: condition === "more" ? "is true" : "is false",
      right: "",
      logic: "AND",
    });
  }

  return newConditionRow({
    left,
    operator: condition === "more" ? ">=" : "<=",
    right: formatSweepValue(row.Value),
    logic: "AND",
  });
}

export function sweepRowAddLabel(row: Record<string, unknown>): string {
  if (isHoldDaysSweepRow(row)) return "Apply hold and profit";
  if (isEntryCheckRow(row)) return "Remove this condition from the draft";
  if (isRegimeSweepRow(row)) return "Add to market regimes";
  if (isComboSweepRow(row)) return "Add all filters to entry";
  if (isStrategyComboSweepRow(row)) return "Add strategy condition";
  if (isSymbolConfirmSweepRow(row)) {
    const value = symbolConfirmSweepRowValues(row);
    return value ? "Apply confirmation symbols" : "Clear confirmation";
  }
  const side = sweepRowSide(row);
  return side === "Sell" ? "Add to exit" : "Add to entry";
}
