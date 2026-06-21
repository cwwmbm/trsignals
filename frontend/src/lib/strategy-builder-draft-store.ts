import { useSyncExternalStore } from "react";

export type StrategyBuilderDraftPreview = {
  draftName: string;
  draftSymbol: string;
  draftHoldDays: number;
  draftProfit: number;
  entryPreview: string;
  exitPreview: string;
  draftValid: boolean;
  isCustomData: boolean;
  customDataBacktestLabel: string | null;
};

const defaultPreview: StrategyBuilderDraftPreview = {
  draftName: "",
  draftSymbol: "SPY",
  draftHoldDays: 2,
  draftProfit: 1,
  entryPreview: "",
  exitPreview: "",
  draftValid: false,
  isCustomData: false,
  customDataBacktestLabel: null,
};

let preview: StrategyBuilderDraftPreview = { ...defaultPreview };
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((listener) => listener());
}

export function updateStrategyBuilderDraftPreview(partial: Partial<StrategyBuilderDraftPreview>) {
  preview = { ...preview, ...partial };
  emit();
}

export function resetStrategyBuilderDraftPreview() {
  preview = { ...defaultPreview };
  emit();
}

export function getStrategyBuilderDraftPreview() {
  return preview;
}

export function subscribeStrategyBuilderDraftPreview(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useStrategyBuilderDraftPreview() {
  return useSyncExternalStore(subscribeStrategyBuilderDraftPreview, getStrategyBuilderDraftPreview);
}
