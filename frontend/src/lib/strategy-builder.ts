import type { BuilderConditionPayload, IndicatorInfo } from "@/api";
import { isFlagOperator } from "@/api";

export function isFlagIndicator(indicators: IndicatorInfo[], id: string) {
  return indicators.find((item) => item.id === id)?.valueType === "flag";
}

export function formatConditionOperator(operator: string) {
  if (operator === "is true") return "Is true";
  if (operator === "is false") return "Is false";
  return operator;
}

export function normalizeConditionForLeft(
  indicators: IndicatorInfo[],
  left: string,
  current: Pick<BuilderConditionPayload, "operator" | "right">,
): Pick<BuilderConditionPayload, "operator" | "right"> | Record<string, never> {
  if (isFlagIndicator(indicators, left)) {
    return {
      operator: isFlagOperator(current.operator) ? current.operator : "is true",
      right: "",
    };
  }
  if (isFlagOperator(current.operator)) {
    return { operator: "<=", right: "" };
  }
  return {};
}
