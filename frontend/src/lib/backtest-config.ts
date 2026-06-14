import type { RunMode } from "@/api";

export type RunModeOption = {
  id: RunMode;
  label: string;
  description: string;
};

export const RUN_MODES: RunModeOption[] = [
  {
    id: "single",
    label: "Single backtest",
    description: "Run one signal, or one combined signal, on one symbol and show detailed stats.",
  },
  {
    id: "signal-combo-sweep",
    label: "Signal combo sweep",
    description: "Compare all four same-symbol combinations: A AND B, A OR B, B AND A, B OR A.",
  },
  {
    id: "symbol-confirm-sweep",
    label: "Symbol confirmation sweep",
    description: "Trade the primary symbol and test every confirmation subset from the symbol pool.",
  },
  {
    id: "symbol-confirm-detail",
    label: "Symbol confirmation detail",
    description: "Drill into one primary symbol plus selected confirmation symbols with yearly detail.",
  },
  {
    id: "hold-days-sweep",
    label: "Hold-days sweep",
    description: "Search hold-days and profitable-close exits for the selected signal.",
  },
  {
    id: "indicator-sweep",
    label: "Indicator sweep",
    description: "Layer indicator threshold filters onto the selected signal and rank results.",
  },
];
