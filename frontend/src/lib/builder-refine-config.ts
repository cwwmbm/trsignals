export type BuilderRefineMode =
  | "signal-combo-sweep"
  | "symbol-confirm-sweep"
  | "hold-days-sweep"
  | "indicator-sweep"
  | "indicator-combo-sweep"
  | "basket-sweep"
  | "market-regime-sweep"
  | "entry-loo";

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
    id: "indicator-combo-sweep",
    label: "Indicator combo sweep",
    description:
      "Search pairs (or trios) of indicator filters layered onto the draft strategy. Pairs are searched exhaustively, trios by extending the best pairs. Ranked in-sample; check the full-period figures before trusting a row.",
  },
  {
    id: "basket-sweep",
    label: "Basket test",
    description:
      "Run the draft strategy on a basket of other symbols over their full history, to see whether it holds up beyond the one it was built on. Each row is ranked against the same entries placed at the wrong times.",
  },
  {
    id: "market-regime-sweep",
    label: "Market Regime Sweep",
    description:
      "Restrict the draft strategy to each market regime state, and to everything but it, and rank the results. Indicator sweeps never include regimes.",
  },
  {
    id: "entry-loo",
    label: "Entry condition check",
    description:
      "Rerun the draft once without each entry condition (leave one out), to see what each one adds in-sample and out-of-sample.",
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

export function refineModesForCustomData(): BuilderRefineModeOption[] {
  return BUILDER_REFINE_MODES.filter(
    (item) =>
      item.id !== "symbol-confirm-sweep" &&
      item.id !== "market-regime-sweep" &&
      item.id !== "basket-sweep",
  );
}
