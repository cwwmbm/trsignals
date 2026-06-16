/** Category order mirrors api/indicator_catalog.py `_CATEGORY_ORDER`. */
export const INDICATOR_CATEGORY_ORDER = [
  'Price',
  'Volume',
  'Reference markets',
  'Breadth (raw ratios)',
  'Moving averages',
  'Momentum / oscillators',
  'Volatility / risk',
  'Efficiency / flow',
  'Spreads & composites',
  'Pattern / signal flags',
  'Breadth RSI',
] as const

/** Short category blurbs for the Help → Indicator glossary. */
export const INDICATOR_CATEGORY_INTROS: Partial<
  Record<(typeof INDICATOR_CATEGORY_ORDER)[number], string>
> = {
  'Moving averages':
    'Trend levels: SMA, EMA, Bollinger bands, Donchian channels (20/55), Keltner channels (TTM standard), and Parabolic SAR.',
  'Momentum / oscillators':
    'Oscillators and momentum: RSI, Stochastic, CCI, MACD, Hurst, ADX(14), Williams %R, ROC(20), TRIX, and 20-bar linear regression slope.',
  'Volatility / risk':
    'ATR, realized volatility, volatility percentile vs trailing 1 year, Bollinger width and %B, and change velocity.',
  Volume:
    'Share volume, OBV and OBV slope, Chaikin Money Flow (20), volume EMA diff, and VFI variants.',
  'Efficiency / flow': 'IBR, Kaufman efficiency ratio, and volume flow (VFI) indicators.',
  'Pattern / signal flags':
    'Boolean flags for Donchian/Keltner/PSAR breakouts, BB/Keltner squeeze, EMA crosses, and close-pattern signals. Use is true / is false in the builder.',
  'Breadth (raw ratios)': 'Sector and style ratios vs SPY (RSP, QQQ, SMH, XLF, etc.).',
  'Breadth RSI': 'RSI applied to breadth ratios — market and sector participation.',
  'Reference markets': 'VIX, SPY, QQQ, SOXX, and other benchmark closes.',
}

export function sortIndicatorCategories(categories: Iterable<string>): string[] {
  const order = new Map(INDICATOR_CATEGORY_ORDER.map((category, index) => [category, index]))
  return [...categories].sort(
    (a, b) => (order.get(a as (typeof INDICATOR_CATEGORY_ORDER)[number]) ?? 999) - (order.get(b as (typeof INDICATOR_CATEGORY_ORDER)[number]) ?? 999),
  )
}
