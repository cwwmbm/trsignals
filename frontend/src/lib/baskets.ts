export type BasketPreset = { id: string; label: string; symbols: string[] };

/** Starting points for a basket test. The draft's own symbol is skipped by the backend. */
export const BASKET_PRESETS: BasketPreset[] = [
  {
    id: "megacap-tech",
    label: "Mega-cap tech",
    symbols: ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ORCL", "ADBE", "CRM", "CSCO", "NFLX"],
  },
  {
    id: "semis",
    label: "Semiconductors",
    symbols: ["NVDA", "AMD", "AVGO", "QCOM", "TXN", "MU", "INTC", "AMAT", "LRCX", "KLAC", "ADI", "MCHP"],
  },
  {
    id: "software",
    label: "Software",
    symbols: ["MSFT", "ORCL", "ADBE", "CRM", "INTU", "NOW", "SNPS", "CDNS", "PANW", "FTNT", "WDAY"],
  },
  {
    id: "banks",
    label: "Large banks",
    symbols: ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC"],
  },
  {
    id: "energy",
    label: "Energy",
    symbols: ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "PSX", "MPC"],
  },
  {
    id: "healthcare",
    label: "Healthcare",
    symbols: ["JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH", "AMGN", "BMY"],
  },
  {
    id: "consumer",
    label: "Consumer",
    symbols: ["WMT", "COST", "HD", "MCD", "NKE", "SBUX", "PG", "KO"],
  },
];

export const DEFAULT_BASKET_TEXT = BASKET_PRESETS[0].symbols.join(", ");

/** Symbols from free text: commas, spaces or new lines, upper-cased, de-duplicated. */
export function parseBasket(text: string): string[] {
  const seen: string[] = [];
  for (const part of text.split(/[\s,;]+/)) {
    const value = part.trim().toUpperCase();
    if (value && !seen.includes(value)) seen.push(value);
  }
  return seen;
}
