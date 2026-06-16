import type { IndicatorInfo } from "@/api";

export function groupIndicators(indicators: IndicatorInfo[]) {
  const groups = new Map<string, IndicatorInfo[]>();
  for (const indicator of indicators) {
    const list = groups.get(indicator.category) ?? [];
    list.push(indicator);
    groups.set(indicator.category, list);
  }
  return groups;
}
