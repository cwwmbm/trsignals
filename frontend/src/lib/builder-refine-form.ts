import type { BuilderBacktestPayload, BuilderRefinePayload } from "@/api";
import type { BuilderRefineMode } from "@/lib/builder-refine-config";

export type BuilderRefineFormState = {
  mode: BuilderRefineMode;
  primarySymbol: string;
  symbolPool: string;
  maxDays: number;
  checkBreadth: boolean;
  checkBoth: boolean;
  isSell: boolean;
};

function csv(value: string) {
  return value
    .split(",")
    .map((item) => item.trim().toUpperCase())
    .filter(Boolean);
}

export function buildBuilderRefinePayload(
  draft: BuilderBacktestPayload,
  state: BuilderRefineFormState,
): BuilderRefinePayload {
  const base = {
    mode: state.mode,
    strategy: draft,
  };

  if (state.mode === "signal-combo-sweep") {
    return base;
  }

  if (state.mode === "symbol-confirm-sweep") {
    return {
      ...base,
      primary_symbol: state.primarySymbol.trim().toUpperCase(),
      symbol_pool: csv(state.symbolPool),
    };
  }

  if (state.mode === "hold-days-sweep") {
    return {
      ...base,
      max_days: state.maxDays,
    };
  }

  return {
    ...base,
    is_sell: state.isSell,
    check_breadth: state.checkBreadth,
    check_both: state.checkBoth,
  };
}
