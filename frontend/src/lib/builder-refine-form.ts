import type { BuilderBacktestPayload, BuilderRefinePayload } from "@/api";
import type { BuilderRefineMode } from "@/lib/builder-refine-config";
import { parseBasket } from "@/lib/baskets";

export type BuilderRefineFormState = {
  mode: BuilderRefineMode;
  primarySymbol: string;
  symbolPool: string;
  maxDays: number;
  checkBreadth: boolean;
  checkBoth: boolean;
  isSell: boolean;
  comboSize: 2 | 3;
  basketSymbols: string;
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

  if (state.mode === "signal-combo-sweep" || state.mode === "market-regime-sweep") {
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

  if (state.mode === "basket-sweep") {
    return { ...base, basket_symbols: parseBasket(state.basketSymbols) };
  }

  if (state.mode === "indicator-combo-sweep") {
    return {
      ...base,
      check_breadth: state.checkBreadth,
      check_both: state.checkBoth,
      combo_size: state.comboSize,
    };
  }

  return {
    ...base,
    is_sell: state.isSell,
    check_breadth: state.checkBreadth,
    check_both: state.checkBoth,
  };
}
