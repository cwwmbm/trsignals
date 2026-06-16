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

export function canAddSweepRowToBuilder(
  row: Record<string, unknown>,
  indicators: IndicatorInfo[],
): boolean {
  if (holdDaysSweepRowValues(row)) return true;
  if (isStrategyComboSweepRow(row)) return true;
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
  if (isStrategyComboSweepRow(row)) return "Add strategy condition";
  const side = sweepRowSide(row);
  return side === "Sell" ? "Add to exit" : "Add to entry";
}
