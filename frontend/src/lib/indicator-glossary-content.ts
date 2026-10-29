/** Category order mirrors api/indicator_catalog.py `_CATEGORY_ORDER`. */
export const INDICATOR_CATEGORY_ORDER = [
  'Price',
  'Volume',
  'Moving averages',
  'Momentum / oscillators',
  'Volatility / risk',
  'Efficiency / flow',
  'Spreads & composites',
  'VWAP / fair value',
  'Pattern / signal flags',
  'Crosses',
  'Breadth RSI',
  'Earnings',
  'Market regimes',
] as const

/** Short category blurbs for the Help → Indicator glossary. */
export const INDICATOR_CATEGORY_INTROS: Partial<
  Record<(typeof INDICATOR_CATEGORY_ORDER)[number], string>
> = {
  'Moving averages':
    'Trend levels: SMA, EMA, Bollinger bands, Donchian channels (20/55), Keltner channels (TTM standard), Kaufman adaptive MA (KAMA), and Parabolic SAR.',
  'Momentum / oscillators':
    'Oscillators and momentum: RSI, Stochastic, CCI, MACD, Hurst, ADX(14) with +DI/−DI, DI balance, DI cross age and signed ADX, Williams %R, ROC(20), TRIX, 20-bar linear regression slope, KAMA slope, and PSAR flip age.',
  'Volatility / risk':
    'VIX and its daily % change, ATR, realized volatility, volatility percentile vs trailing 1 year, Bollinger width and %B, BB/Keltner squeeze ratio and squeeze age, and change velocity.',
  Volume:
    'Share volume, OBV and OBV slope, Chaikin Money Flow (20), volume EMA diff, and VFI variants.',
  'Efficiency / flow': 'IBR, Kaufman efficiency ratio, and volume flow (VFI) indicators.',
  'Pattern / signal flags':
    'Boolean flags for Donchian/Keltner/PSAR breakouts, BB/Keltner squeeze, EMA(8) crosses, and close-pattern signals. Use is true / is false in the builder.',
  Crosses:
    'Cross flags for pairs of lines, such as close vs EMA(20) or SMA(50) vs SMA(200): crossed above or below today, and crossed above or below yesterday. Use is true / is false in the builder.',
  'Breadth RSI':
    'RSI applied to sector/style participation ratios — market, semis, financials, energy, and more.',
  Earnings:
    'Trading days to the next and since the last earnings reaction day, for stocks. A report before the open reacts the same day, one after the close the next session. ETFs and indexes have no earnings, so both read 999 and an entry rule such as Days to earnings > 3 never blocks them.',
  'Market regimes':
    'True/false flags for each market-regime bucket. Is true keeps only that bucket. Is false keeps every other defined day. Days before the series exists match neither.',
}

export function sortIndicatorCategories(categories: Iterable<string>): string[] {
  const order = new Map(INDICATOR_CATEGORY_ORDER.map((category, index) => [category, index]))
  return [...categories].sort(
    (a, b) => (order.get(a as (typeof INDICATOR_CATEGORY_ORDER)[number]) ?? 999) - (order.get(b as (typeof INDICATOR_CATEGORY_ORDER)[number]) ?? 999),
  )
}
