import type { RunMode, SignalExpression } from "@/api";

export function csv(value: string) {
  return value
    .split(",")
    .map((item) => item.trim().toUpperCase())
    .filter(Boolean);
}

export function isOgSignalName(name: string) {
  return name === "og_buy_signal" || name === "og_new_buy_signal";
}

export type BacktestFormState = {
  mode: RunMode;
  symbol: string;
  years: number;
  signalKind: "single" | "combined";
  signal: string;
  signalA: string;
  signalB: string;
  comboMode: "and" | "or";
  primarySymbol: string;
  symbolPool: string;
  confirmSymbols: string;
  maxDays: number;
  checkBreadth: boolean;
  checkBoth: boolean;
  isSell: boolean;
  mondayBuy: boolean;
  lowVolumeBuy: boolean;
  holdOnBuySignal: boolean;
};

export function signalExpression(state: Pick<BacktestFormState, "signalKind" | "signal" | "signalA" | "signalB" | "comboMode">): SignalExpression {
  if (state.signalKind === "combined") {
    return { kind: "combined", primary: state.signalA, secondary: state.signalB, mode: state.comboMode };
  }
  return { kind: "single", name: state.signal };
}

export function usesOgRuntimeOptions(state: BacktestFormState) {
  if (state.mode === "signal-combo-sweep") {
    return isOgSignalName(state.signalA) || isOgSignalName(state.signalB);
  }
  if (state.signalKind === "single") {
    return isOgSignalName(state.signal);
  }
  return isOgSignalName(state.signalA);
}

export function runtimeOptionsPayload(state: BacktestFormState) {
  const runtime_options: Record<string, boolean> = {
    hold_on_buy_signal: state.holdOnBuySignal,
  };
  if (usesOgRuntimeOptions(state)) {
    runtime_options.monday_buy = state.mondayBuy;
    runtime_options.low_volume_buy = state.lowVolumeBuy;
  }
  return { runtime_options };
}

export function buildPayload(state: BacktestFormState): Record<string, unknown> {
  const runtime = runtimeOptionsPayload(state);
  if (state.mode === "signal-combo-sweep") {
    return { symbol: state.symbol, years: state.years, signal_a: state.signalA, signal_b: state.signalB, ...runtime };
  }
  if (state.mode === "symbol-confirm-sweep") {
    return {
      primary_symbol: state.primarySymbol,
      symbol_pool: csv(state.symbolPool),
      years: state.years,
      signal: signalExpression(state),
      ...runtime,
    };
  }
  if (state.mode === "symbol-confirm-detail") {
    return {
      primary_symbol: state.primarySymbol,
      confirm_symbols: csv(state.confirmSymbols),
      years: state.years,
      signal: signalExpression(state),
      ...runtime,
    };
  }
  if (state.mode === "hold-days-sweep") {
    return { symbol: state.symbol, years: state.years, max_days: state.maxDays, signal: signalExpression(state), ...runtime };
  }
  if (state.mode === "indicator-sweep") {
    return {
      symbol: state.symbol,
      years: state.years,
      signal: signalExpression(state),
      is_sell: state.isSell,
      check_breadth: state.checkBreadth,
      check_both: state.checkBoth,
      ...runtime,
    };
  }
  return { symbol: state.symbol, years: state.years, signal: signalExpression(state), ...runtime };
}
