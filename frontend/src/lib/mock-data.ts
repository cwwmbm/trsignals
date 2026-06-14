export type RunMode = {
  id: string
  label: string
  description: string
}

export const RUN_MODES: RunMode[] = [
  {
    id: 'single',
    label: 'Single backtest',
    description: 'Run one signal, or one combined signal, on one symbol and show detailed stats.',
  },
  {
    id: 'signal-combo',
    label: 'Signal combo sweep',
    description: 'Sweep every combination of the selected signals to find the strongest blends.',
  },
  {
    id: 'symbol-conf-sweep',
    label: 'Symbol confirmation sweep',
    description: 'Test a signal against a basket of confirming symbols and rank the results.',
  },
  {
    id: 'symbol-conf-detail',
    label: 'Symbol confirmation detail',
    description: 'Show the detailed trade-by-trade breakdown for a single confirming symbol.',
  },
  {
    id: 'hold-days',
    label: 'Hold-days sweep',
    description: 'Sweep a range of holding periods to find the optimal time in trade.',
  },
  {
    id: 'indicator',
    label: 'Indicator sweep',
    description: 'Sweep indicator thresholds across a grid and surface the best parameter sets.',
  },
]

export const SIGNALS = [
  'buy_signal1',
  'buy_signal4',
  'buy_signal7',
  'buy_signal8',
  'buy_signal9',
  'buy_signal10',
  'buy_signal16',
  'buy_signal17',
  'buy_signal18',
  'buy_signal20',
  'buy_signal21',
  'buy_signal24',
  'og_buy_signal',
  'og_new_buy_signal',
]

export const SYMBOLS = ['SPY', 'QQQ', 'SOXX', 'SMH', 'IWM', 'FXI', 'GDX', 'XBI']

export type BacktestStats = {
  rollingPnl: number
  cagrPercent: number
  sharpe: number
  sortino: number
  maxDrawdown: number
  trades: number
  pctPositive: number
  excludedYear: number
  description: string
}

export const BACKTEST_STATS: BacktestStats = {
  rollingPnl: 11613833.65,
  cagrPercent: 23.74,
  sharpe: 0.68,
  sortino: 1.49,
  maxDrawdown: 0.62,
  trades: 712,
  pctPositive: 68.54,
  excludedYear: 2020,
  description: 'Long NQ: Close 1 day ago <= Close 3 days ago, IBR <= 0.4',
}

export type YearlyRow = {
  year: number
  pnlPercent: number
  drawdownPercent: number
  numTrades: number
  positiveTrades: number
}

export const YEARLY_BREAKDOWN: YearlyRow[] = [
  { year: 2026, pnlPercent: 21.95, drawdownPercent: 14.08, numTrades: 9, positiveTrades: 8 },
  { year: 2025, pnlPercent: 117.54, drawdownPercent: 12.34, numTrades: 27, positiveTrades: 20 },
  { year: 2024, pnlPercent: 98.95, drawdownPercent: 20.2, numTrades: 33, positiveTrades: 25 },
  { year: 2023, pnlPercent: 43.24, drawdownPercent: 15.78, numTrades: 23, positiveTrades: 15 },
  { year: 2022, pnlPercent: 22.32, drawdownPercent: 29.89, numTrades: 38, positiveTrades: 26 },
  { year: 2021, pnlPercent: 94.74, drawdownPercent: 17.45, numTrades: 22, positiveTrades: 17 },
  { year: 2020, pnlPercent: 201.53, drawdownPercent: 32.89, numTrades: 26, positiveTrades: 20 },
  { year: 2019, pnlPercent: 38.58, drawdownPercent: 20.5, numTrades: 23, positiveTrades: 15 },
  { year: 2018, pnlPercent: -10.42, drawdownPercent: 37.29, numTrades: 32, positiveTrades: 23 },
  { year: 2017, pnlPercent: 51.47, drawdownPercent: 14.82, numTrades: 20, positiveTrades: 17 },
  { year: 2016, pnlPercent: 0.31, drawdownPercent: 17.9, numTrades: 25, positiveTrades: 16 },
  { year: 2015, pnlPercent: 42.55, drawdownPercent: 23.82, numTrades: 35, positiveTrades: 27 },
  { year: 2014, pnlPercent: 52.84, drawdownPercent: 36.89, numTrades: 26, positiveTrades: 21 },
  { year: 2013, pnlPercent: 67.74, drawdownPercent: 62.38, numTrades: 27, positiveTrades: 22 },
  { year: 2012, pnlPercent: 33.18, drawdownPercent: 19.44, numTrades: 24, positiveTrades: 18 },
  { year: 2011, pnlPercent: -4.87, drawdownPercent: 41.02, numTrades: 29, positiveTrades: 19 },
]

export type TradeRow = {
  entryDate: string
  exitDate: string
  entryPrice: number
  exitPrice: number
  tradePnl: number
  daysInTrade: number
  status: 'Closed' | 'Open'
}

export const TRADES: TradeRow[] = [
  { entryDate: '2026-06-05', exitDate: '2026-06-08', entryPrice: 539.77, exitPrice: 571.45, tradePnl: 17.61, daysInTrade: 1, status: 'Closed' },
  { entryDate: '2026-05-29', exitDate: '2026-06-01', entryPrice: 569.08, exitPrice: 571.93, tradePnl: 1.5, daysInTrade: 1, status: 'Closed' },
  { entryDate: '2026-05-18', exitDate: '2026-05-19', entryPrice: 495.87, exitPrice: 496.74, tradePnl: 0.53, daysInTrade: 1, status: 'Closed' },
  { entryDate: '2026-03-27', exitDate: '2026-03-31', entryPrice: 323.48, exitPrice: 328.66, tradePnl: 4.8, daysInTrade: 2, status: 'Closed' },
  { entryDate: '2026-03-20', exitDate: '2026-03-23', entryPrice: 332.51, exitPrice: 336.59, tradePnl: 3.68, daysInTrade: 1, status: 'Closed' },
  { entryDate: '2026-03-13', exitDate: '2026-03-16', entryPrice: 331.12, exitPrice: 337.62, tradePnl: 5.89, daysInTrade: 1, status: 'Closed' },
  { entryDate: '2026-03-10', exitDate: '2026-03-11', entryPrice: 338.6, exitPrice: 341.88, tradePnl: 2.9, daysInTrade: 1, status: 'Closed' },
  { entryDate: '2026-02-03', exitDate: '2026-02-05', entryPrice: 345.43, exitPrice: 330.63, tradePnl: -12.85, daysInTrade: 2, status: 'Closed' },
  { entryDate: '2025-12-30', exitDate: '2026-01-02', entryPrice: 304.73, exitPrice: 313.5, tradePnl: 8.63, daysInTrade: 2, status: 'Closed' },
  { entryDate: '2025-12-18', exitDate: '2025-12-19', entryPrice: 291.86, exitPrice: 299.63, tradePnl: 7.98, daysInTrade: 1, status: 'Closed' },
  { entryDate: '2025-12-15', exitDate: '2025-12-17', entryPrice: 297.39, exitPrice: 285.05, tradePnl: -12.44, daysInTrade: 2, status: 'Closed' },
  { entryDate: '2025-11-20', exitDate: '2025-11-21', entryPrice: 267.54, exitPrice: 270.27, tradePnl: 3.05, daysInTrade: 1, status: 'Closed' },
  { entryDate: '2025-11-18', exitDate: '2025-11-19', entryPrice: 276.4, exitPrice: 281.02, tradePnl: 5.01, daysInTrade: 1, status: 'Closed' },
  { entryDate: '2025-11-13', exitDate: '2025-11-17', entryPrice: 288.78, exitPrice: 282.97, tradePnl: -6.03, daysInTrade: 2, status: 'Closed' },
  { entryDate: '2025-10-28', exitDate: '2025-10-30', entryPrice: 254.11, exitPrice: 263.4, tradePnl: 3.66, daysInTrade: 2, status: 'Closed' },
  { entryDate: '2025-10-09', exitDate: '2025-10-13', entryPrice: 248.92, exitPrice: 251.77, tradePnl: 1.14, daysInTrade: 2, status: 'Closed' },
  { entryDate: '2025-09-22', exitDate: '2025-09-24', entryPrice: 239.5, exitPrice: 233.18, tradePnl: -2.64, daysInTrade: 2, status: 'Closed' },
  { entryDate: '2025-09-04', exitDate: '2025-09-05', entryPrice: 231.7, exitPrice: 238.06, tradePnl: 2.74, daysInTrade: 1, status: 'Closed' },
]

// Equity curve: ~25 years of monthly compounded growth with drawdowns.
export type EquityPoint = { date: string; equity: number }

function buildEquityCurve(): EquityPoint[] {
  const points: EquityPoint[] = []
  let equity = 100000
  const start = new Date(2001, 0, 1)
  // deterministic pseudo-random for stable SSR/CSR output
  let seed = 42
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return seed / 0x7fffffff
  }
  const totalMonths = 25 * 12
  for (let i = 0; i <= totalMonths; i++) {
    const d = new Date(start.getFullYear(), start.getMonth() + i, 1)
    // drift up with occasional drawdowns
    const drift = 0.018
    const vol = (rand() - 0.42) * 0.11
    const shock = rand() > 0.94 ? -(0.1 + rand() * 0.2) : 0
    equity = Math.max(equity * (1 + drift + vol + shock), 20000)
    points.push({
      date: `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`,
      equity: Math.round(equity),
    })
  }
  return points
}

export const EQUITY_CURVE = buildEquityCurve()

export type ScanRow = {
  id: string
  symbol: string
  signal: string
  buySignal: boolean
  holdLong: boolean
  sellSignal: boolean
  days: number
  profit: number
  tradePnl: number
  kelly: number | null
  description: string
}

export const SCAN_ROWS: ScanRow[] = [
  { id: 's1', symbol: 'SPY', signal: 'buy_signal7', buySignal: false, holdLong: false, sellSignal: false, days: 2, profit: 1, tradePnl: 0, kelly: 24.56, description: 'Long NQ: Close 1 day ago <= Close 3 days ago, IBR <= 0.4' },
  { id: 's2', symbol: 'SPY', signal: 'buy_signal8', buySignal: false, holdLong: false, sellSignal: false, days: 3, profit: 1, tradePnl: 0, kelly: 48.71, description: 'Long NQ: ER(10) > 0.50, ValueCharts(5) > -12, IBR <= 0.5' },
  { id: 's3', symbol: 'SPY', signal: 'buy_signal9', buySignal: false, holdLong: false, sellSignal: false, days: 100, profit: 100, tradePnl: 0, kelly: 30.17, description: 'Stoch < 30, MACD < 0, IBR <= 0.2' },
  { id: 's4', symbol: 'SPY', signal: 'buy_signal10', buySignal: false, holdLong: false, sellSignal: false, days: 3, profit: 1, tradePnl: 0, kelly: 39.15, description: 'Long NQ: Low 1 day ago <= Lowest Low in 2 days' },
  { id: 's5', symbol: 'SPY', signal: 'buy_signal20', buySignal: false, holdLong: false, sellSignal: false, days: 50, profit: 50, tradePnl: 0, kelly: 24.92, description: 'Experimental Long signal' },
  { id: 's6', symbol: 'SPY', signal: 'buy_signal21', buySignal: false, holdLong: false, sellSignal: false, days: 1, profit: 1, tradePnl: 0, kelly: null, description: 'Close<200SMA, RSI2<40, RSI2SemisBreadth>30. Beta filter applied.' },
  { id: 's7', symbol: 'SPY', signal: 'og_buy_signal', buySignal: false, holdLong: true, sellSignal: false, days: 0, profit: 0, tradePnl: 1.8, kelly: 20.58, description: 'OG Long Spy strategy' },
  { id: 's8', symbol: 'SPY', signal: 'og_new_buy_signal', buySignal: false, holdLong: true, sellSignal: false, days: 0, profit: 0, tradePnl: 1.8, kelly: 22.15, description: 'OG Long Spy strategy with few extra conditions' },
  { id: 's9', symbol: 'SMH', signal: 'buy_signal7', buySignal: false, holdLong: false, sellSignal: false, days: 2, profit: 1, tradePnl: 0, kelly: 55.12, description: 'Long NQ: Close 1 day ago <= Close 3 days ago, IBR <= 0.4' },
  { id: 's10', symbol: 'SMH', signal: 'buy_signal10', buySignal: false, holdLong: false, sellSignal: false, days: 3, profit: 1, tradePnl: 0, kelly: 46.11, description: 'Long NQ: Low 1 day ago <= Lowest Low in 2 days' },
  { id: 's11', symbol: 'SMH', signal: 'buy_signal16', buySignal: false, holdLong: false, sellSignal: false, days: 4, profit: 1, tradePnl: 0, kelly: 60.88, description: 'high[0] > close[1], IBR[0] <= 50, SMA50>SMA200' },
  { id: 's12', symbol: 'SMH', signal: 'buy_signal17', buySignal: false, holdLong: false, sellSignal: false, days: 2, profit: 1, tradePnl: 0, kelly: 58.94, description: 'Long SMH: rsi(close,2)[0] <= 20, IBR[0] <= 30' },
  { id: 's13', symbol: 'QQQ', signal: 'buy_signal7', buySignal: false, holdLong: false, sellSignal: false, days: 2, profit: 1, tradePnl: 0, kelly: 24.96, description: 'Long NQ: Close 1 day ago <= Close 3 days ago, IBR <= 0.4' },
  { id: 's14', symbol: 'QQQ', signal: 'buy_signal8', buySignal: false, holdLong: false, sellSignal: false, days: 3, profit: 1, tradePnl: 0, kelly: 43.94, description: 'Long NQ: ER(10) > 0.50, ValueCharts(5) > -12, IBR <= 0.5' },
  { id: 's15', symbol: 'QQQ', signal: 'buy_signal9', buySignal: false, holdLong: false, sellSignal: false, days: 100, profit: 100, tradePnl: 0, kelly: 44.89, description: 'Stoch < 30, MACD < 0, IBR <= 0.2' },
  { id: 's16', symbol: 'QQQ', signal: 'buy_signal10', buySignal: false, holdLong: false, sellSignal: false, days: 3, profit: 1, tradePnl: 0, kelly: 40.51, description: 'Long NQ: Low 1 day ago <= Lowest Low in 2 days' },
  { id: 's17', symbol: 'QQQ', signal: 'buy_signal16', buySignal: false, holdLong: false, sellSignal: false, days: 4, profit: 1, tradePnl: 0, kelly: 41.17, description: 'high[0] > close[1], IBR[0] <= 50, SMA50>SMA200' },
  { id: 's18', symbol: 'QQQ', signal: 'buy_signal17', buySignal: false, holdLong: false, sellSignal: false, days: 2, profit: 1, tradePnl: 0, kelly: 56.21, description: 'Long SMH: rsi(close,2)[0] <= 20, IBR[0] <= 30' },
  { id: 's19', symbol: 'QQQ', signal: 'buy_signal20', buySignal: false, holdLong: false, sellSignal: false, days: 50, profit: 50, tradePnl: 0, kelly: 33.16, description: 'Experimental Long signal' },
  { id: 's20', symbol: 'QQQ', signal: 'og_new_buy_signal', buySignal: false, holdLong: true, sellSignal: false, days: 0, profit: 0, tradePnl: 12.33, kelly: 34.3, description: 'OG Long Spy strategy with few extra conditions' },
  { id: 's21', symbol: 'SOXX', signal: 'buy_signal7', buySignal: false, holdLong: false, sellSignal: false, days: 2, profit: 1, tradePnl: 0, kelly: 54.14, description: 'Long NQ: Close 1 day ago <= Close 3 days ago, IBR <= 0.4' },
  { id: 's22', symbol: 'SOXX', signal: 'buy_signal10', buySignal: false, holdLong: false, sellSignal: false, days: 3, profit: 1, tradePnl: 0, kelly: 57.01, description: 'Long NQ: Low 1 day ago <= Lowest Low in 2 days' },
  { id: 's23', symbol: 'SOXX', signal: 'buy_signal16', buySignal: false, holdLong: false, sellSignal: false, days: 4, profit: 1, tradePnl: 0, kelly: 42.87, description: 'high[0] > close[1], IBR[0] <= 50, SMA50>SMA200' },
  { id: 's24', symbol: 'SOXX', signal: 'buy_signal17', buySignal: false, holdLong: false, sellSignal: false, days: 2, profit: 1, tradePnl: 0, kelly: 51.14, description: 'Long SMH: rsi(close,2)[0] <= 20, IBR[0] <= 30' },
  { id: 's25', symbol: 'IWM', signal: 'buy_signal20', buySignal: false, holdLong: false, sellSignal: false, days: 50, profit: 50, tradePnl: 0, kelly: 29.11, description: 'Experimental Long signal' },
  { id: 's26', symbol: 'IWM', signal: 'og_new_buy_signal', buySignal: false, holdLong: false, sellSignal: false, days: 0, profit: 0, tradePnl: 0, kelly: 35.76, description: 'OG Long Spy strategy with few extra conditions' },
  { id: 's27', symbol: 'FXI', signal: 'buy_signal4', buySignal: false, holdLong: false, sellSignal: false, days: 3, profit: 1, tradePnl: 0, kelly: 13.85, description: 'For SPY, add ValueCharts<0 condition. New strategy.' },
  { id: 's28', symbol: 'FXI', signal: 'buy_signal7', buySignal: false, holdLong: false, sellSignal: false, days: 2, profit: 1, tradePnl: 0, kelly: 27.66, description: 'Long NQ: Close 1 day ago <= Close 3 days ago, IBR <= 0.4' },
  { id: 's29', symbol: 'GDX', signal: 'buy_signal18', buySignal: false, holdLong: false, sellSignal: false, days: 2, profit: 1, tradePnl: 0, kelly: 48.65, description: 'Long CL: open[1] <= lowest(open,2)[0], IBR[0] <= 40' },
  { id: 's30', symbol: 'GDX', signal: 'buy_signal24', buySignal: false, holdLong: false, sellSignal: false, days: 2, profit: 1, tradePnl: 0, kelly: 1.45, description: 'Testing' },
  { id: 's31', symbol: 'XBI', signal: 'buy_signal1', buySignal: false, holdLong: false, sellSignal: false, days: 2, profit: 1, tradePnl: 0, kelly: 50.89, description: 'Long IBB: RSI5EnergyBreadth < 70, Close_EMA8 < close' },
]

export type Strategy = {
  id: string
  name: string
  signal: string
  description: string
  cagr: number
  sharpe: number
  maxDrawdown: number
  trades: number
  pctPositive: number
  tags: string[]
  status: 'live' | 'draft' | 'archived'
}

export const STRATEGIES: Strategy[] = [
  { id: 'st1', name: 'NQ Mean Reversion', signal: 'buy_signal7', description: 'Long NQ: Close 1 day ago <= Close 3 days ago, IBR <= 0.4', cagr: 23.74, sharpe: 0.68, maxDrawdown: 32.89, trades: 712, pctPositive: 68.54, tags: ['mean-reversion', 'index'], status: 'live' },
  { id: 'st2', name: 'Efficiency Ratio Long', signal: 'buy_signal8', description: 'Long NQ: ER(10) > 0.50, ValueCharts(5) > -12, IBR <= 0.5', cagr: 18.12, sharpe: 0.74, maxDrawdown: 24.5, trades: 488, pctPositive: 64.2, tags: ['momentum'], status: 'live' },
  { id: 'st3', name: 'Stoch Oversold Dip', signal: 'buy_signal9', description: 'Stoch < 30, MACD < 0, IBR <= 0.2', cagr: 15.4, sharpe: 0.59, maxDrawdown: 41.2, trades: 350, pctPositive: 61.1, tags: ['oscillator', 'dip'], status: 'draft' },
  { id: 'st4', name: 'Lowest-Low Reversal', signal: 'buy_signal10', description: 'Long NQ: Low 1 day ago <= Lowest Low in 2 days', cagr: 21.05, sharpe: 0.71, maxDrawdown: 28.7, trades: 540, pctPositive: 66.9, tags: ['reversal'], status: 'live' },
  { id: 'st5', name: 'SMA Trend Breakout', signal: 'buy_signal16', description: 'high[0] > close[1], IBR[0] <= 50, SMA50 > SMA200', cagr: 26.3, sharpe: 0.81, maxDrawdown: 22.1, trades: 295, pctPositive: 70.2, tags: ['trend', 'breakout'], status: 'live' },
  { id: 'st6', name: 'SMH RSI(2) Long', signal: 'buy_signal17', description: 'Long SMH: rsi(close,2)[0] <= 20, IBR[0] <= 30', cagr: 31.8, sharpe: 0.88, maxDrawdown: 35.6, trades: 410, pctPositive: 72.4, tags: ['mean-reversion', 'semis'], status: 'live' },
  { id: 'st7', name: 'Experimental Long', signal: 'buy_signal20', description: 'Experimental Long signal', cagr: 9.2, sharpe: 0.41, maxDrawdown: 48.9, trades: 120, pctPositive: 55.0, tags: ['experimental'], status: 'draft' },
  { id: 'st8', name: 'OG Spy Long', signal: 'og_buy_signal', description: 'OG Long Spy strategy', cagr: 12.6, sharpe: 0.55, maxDrawdown: 19.8, trades: 210, pctPositive: 63.3, tags: ['legacy'], status: 'archived' },
]

export const INDICATORS = [
  'RSI(2)', 'RSI(5)', 'RSI(14)', 'IBR', 'Stoch', 'MACD', 'SMA(50)', 'SMA(200)',
  'EMA(8)', 'EMA(21)', 'ER(10)', 'ValueCharts(5)', 'ATR(14)', 'Close', 'Open', 'High', 'Low',
]

export const OPERATORS = ['<', '<=', '>', '>=', '=', 'crosses above', 'crosses below']
