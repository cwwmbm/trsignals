export const SCAN_SYMBOL_ORDER = [
  'SPY',
  'SMH',
  'QQQ',
  'SOXX',
  'IWM',
  'FXI',
  'AAPL',
  'GDX',
  'MSFT',
  'GLD',
  'XBI',
  'TLT',
] as const

export function symbolSortRank(symbol: string): number {
  const index = SCAN_SYMBOL_ORDER.indexOf(symbol as (typeof SCAN_SYMBOL_ORDER)[number])
  return index === -1 ? SCAN_SYMBOL_ORDER.length : index
}

export function compareSymbols(a: string, b: string): number {
  return symbolSortRank(a) - symbolSortRank(b)
}

export function sortSymbols(symbols: string[]): string[] {
  return [...symbols].sort(compareSymbols)
}
