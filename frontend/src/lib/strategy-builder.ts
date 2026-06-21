import type { BuilderConditionPayload, CompareMode, IndicatorInfo } from "@/api";
import { isFlagOperator } from "@/api";

const PRICE_LIKE_IDS = new Set([
  "CloseLag1",
  "CloseLag2",
  "CloseLag3",
  "LowLag1",
  "LowMin2Lag1",
]);

function isPriceLikeCompareTarget(leftId: string, rightId: string, left: IndicatorInfo | undefined) {
  if (!PRICE_LIKE_IDS.has(rightId)) return false;
  return PRICE_LIKE_IDS.has(leftId) || left?.kind === "price" || left?.kind === "trend";
}

export function isFlagIndicator(indicators: IndicatorInfo[], id: string) {
  return indicators.find((item) => item.id === id)?.valueType === "flag";
}

export function getIndicator(indicators: IndicatorInfo[], id: string) {
  return indicators.find((item) => item.id === id);
}

export function getCompareMode(indicators: IndicatorInfo[], leftId: string): CompareMode {
  const indicator = getIndicator(indicators, leftId);
  if (!indicator) return "number";
  if (indicator.compareMode) return indicator.compareMode;
  if (indicator.valueType === "flag") return "none";
  if (indicator.valueType === "percent" || indicator.valueType === "ratio") return "number";
  if (indicator.kind === "price" || indicator.kind === "trend") return "both";
  return "number";
}

export function filterCompareIndicators(
  indicators: IndicatorInfo[],
  leftId: string,
): IndicatorInfo[] {
  const left = getIndicator(indicators, leftId);
  if (!left) return [];

  const compareMode = getCompareMode(indicators, leftId);
  if (compareMode === "none" || compareMode === "number") return [];

  const allowedKinds = new Set(left.compareIndicatorKinds ?? ["price", "trend"]);
  return indicators.filter(
    (item) =>
      item.id !== leftId &&
      item.valueType !== "flag" &&
      item.builderEligible !== false &&
      item.available !== false &&
      (allowedKinds.has(item.kind) || isPriceLikeCompareTarget(leftId, item.id, left)),
  );
}

export function isCompareIndicatorAllowed(
  indicators: IndicatorInfo[],
  leftId: string,
  rightId: string,
) {
  const left = getIndicator(indicators, leftId);
  if (isPriceLikeCompareTarget(leftId, rightId, left)) return true;
  return filterCompareIndicators(indicators, leftId).some((item) => item.id === rightId);
}

export function formatConditionOperator(operator: string) {
  if (operator === "is true") return "Is true";
  if (operator === "is false") return "Is false";
  return operator;
}

export type CompareRightKind = "indicator" | "number";

export function resolveCompareRightKind(
  indicators: IndicatorInfo[],
  leftId: string,
  right: string,
  allIndicatorIds: Set<string>,
): CompareRightKind {
  const compareMode = getCompareMode(indicators, leftId);
  if (compareMode === "indicator") return "indicator";
  if (compareMode === "number") return "number";
  if (right && allIndicatorIds.has(right) && isCompareIndicatorAllowed(indicators, leftId, right)) {
    return "indicator";
  }
  return "number";
}

function defaultCompareRight(
  indicators: IndicatorInfo[],
  leftId: string,
  kind: CompareRightKind,
): string {
  const indicator = getIndicator(indicators, leftId);
  if (!indicator) return "";

  if (kind === "indicator") {
    return indicator.defaultCompareIndicator ?? "Close";
  }
  return indicator.defaultCompareNumber ?? "";
}

export function normalizeConditionForLeft(
  indicators: IndicatorInfo[],
  left: string,
  current: Pick<BuilderConditionPayload, "operator" | "right">,
  allIndicatorIds: Set<string>,
): Pick<BuilderConditionPayload, "operator" | "right"> | Record<string, never> {
  if (isFlagIndicator(indicators, left)) {
    return {
      operator: isFlagOperator(current.operator) ? current.operator : "is true",
      right: "",
    };
  }

  if (isFlagOperator(current.operator)) {
    const compareMode = getCompareMode(indicators, left);
    const rightKind: CompareRightKind =
      compareMode === "indicator" || compareMode === "both" ? "indicator" : "number";
    return {
      operator: "<=",
      right: defaultCompareRight(indicators, left, rightKind),
    };
  }

  const compareMode = getCompareMode(indicators, left);
  if (compareMode === "none") return {};

  const rightAllowed =
    compareMode === "number"
      ? current.right.trim() !== "" && !allIndicatorIds.has(current.right)
      : compareMode === "indicator"
        ? isCompareIndicatorAllowed(indicators, left, current.right)
        : allIndicatorIds.has(current.right)
          ? isCompareIndicatorAllowed(indicators, left, current.right)
          : current.right.trim() !== "";

  if (rightAllowed) return {};

  const defaultKind: CompareRightKind =
    compareMode === "number"
      ? "number"
      : compareMode === "indicator"
        ? "indicator"
        : indicatorHasDefaultCompareIndicator(indicators, left)
          ? "indicator"
          : "number";

  return {
    operator: current.operator,
    right: defaultCompareRight(indicators, left, defaultKind),
  };
}

function indicatorHasDefaultCompareIndicator(indicators: IndicatorInfo[], leftId: string) {
  const indicator = getIndicator(indicators, leftId);
  if (!indicator) return false;
  if (indicator.defaultCompareIndicator) return true;
  return indicator.kind === "price" || indicator.kind === "trend";
}

export function isInvalidCompareRight(
  indicators: IndicatorInfo[],
  leftId: string,
  right: string,
  allIndicatorIds: Set<string>,
) {
  if (!right.trim()) return false;
  const compareMode = getCompareMode(indicators, leftId);
  if (compareMode === "none") return false;

  const isIndicator = allIndicatorIds.has(right);
  if (compareMode === "number") return isIndicator;
  if (compareMode === "indicator") {
    return !isIndicator || !isCompareIndicatorAllowed(indicators, leftId, right);
  }
  if (isIndicator) {
    return !isCompareIndicatorAllowed(indicators, leftId, right);
  }
  return Number.isNaN(Number(right));
}

export function formatTypicalRangeHint(range: { min: number; max: number }) {
  const fmt = (value: number) => {
    const abs = Math.abs(value);
    if (abs >= 1 || abs === 0) return String(value);
    return value.toFixed(2).replace(/\.?0+$/, "");
  };
  return `${fmt(range.min)}–${fmt(range.max)}`;
}
