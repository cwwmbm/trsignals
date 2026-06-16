export type BuilderRefineMode =
  | "signal-combo-sweep"
  | "symbol-confirm-sweep"
  | "hold-days-sweep"
  | "indicator-sweep";

export type BuilderRefineModeOption = {
  id: BuilderRefineMode;
  label: string;
  description: string;
};

export const BUILDER_REFINE_MODES: BuilderRefineModeOption[] = [
  {
    id: "indicator-sweep",
    label: "Indicator sweep",
    description: "Layer indicator threshold filters onto the draft strategy and rank results.",
  },
  {
    id: "signal-combo-sweep",
    label: "Signal combo sweep",
    description:
      "Compare all four combinations of the draft strategy with a saved secondary strategy.",
  },
  {
    id: "hold-days-sweep",
    label: "Hold-days sweep",
    description: "Search hold-days and profitable-close exits for the draft strategy.",
  },
  {
    id: "symbol-confirm-sweep",
    label: "Symbol confirmation sweep",
    description:
      "Trade the primary symbol and test every confirmation subset from the symbol pool.",
  },
];

export function isSymbolConfirmRefineMode(mode: BuilderRefineMode) {
  return mode === "symbol-confirm-sweep";
}
