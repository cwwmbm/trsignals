import type { SavedPortfolio, SavedStrategy, ScanLane, ScanRow } from '@/api'
import { compareSymbols } from '@/lib/symbol-order'

export const SCAN_LANES: ScanLane[] = ['active', 'testing', 'archived']

export const LANE_LABELS: Record<ScanLane, string> = {
  active: 'Active',
  testing: 'Testing',
  archived: 'Archived',
}

export type LaneRows = Record<ScanLane, ScanRow[]>

const SIGNAL_ORDER = [
  'buy_signal1',
  'buy_signal2',
  'buy_signal3',
  'buy_signal4',
  'buy_signal5',
  'buy_signal6',
  'buy_signal7',
  'buy_signal8',
  'buy_signal9',
  'buy_signal10',
  'buy_signal11',
  'buy_signal12',
  'buy_signal13',
  'buy_signal14',
  'buy_signal15',
  'buy_signal16',
  'buy_signal17',
  'buy_signal18',
  'buy_signal19',
  'buy_signal20',
  'buy_signal21',
  'buy_signal24',
  'og_buy_signal',
  'og_new_buy_signal',
] as const

export function laneContainerId(lane: ScanLane) {
  return `lane-${lane}`
}

export function parseLaneContainerId(id: string): ScanLane | null {
  if (!id.startsWith('lane-')) return null
  const lane = id.slice(5) as ScanLane
  return SCAN_LANES.includes(lane) ? lane : null
}

export function sortLegacyRows(rows: ScanRow[]): ScanRow[] {
  const signalRank = new Map(SIGNAL_ORDER.map((signal, index) => [signal, index]))
  return [...rows].sort((a, b) => {
    const symbolDiff = compareSymbols(a.symbol, b.symbol)
    if (symbolDiff !== 0) return symbolDiff
    return (
      (signalRank.get(a.signal as (typeof SIGNAL_ORDER)[number]) ?? SIGNAL_ORDER.length) -
      (signalRank.get(b.signal as (typeof SIGNAL_ORDER)[number]) ?? SIGNAL_ORDER.length)
    )
  })
}

export function resolveScanLane(strategy: SavedStrategy | undefined): ScanLane {
  return strategy?.scan_lane ?? 'testing'
}

export function resolvePortfolioScanLane(portfolio: SavedPortfolio | undefined): ScanLane {
  return portfolio?.scan_lane ?? 'testing'
}

function rowSortOrder(
  row: ScanRow,
  strategyById: Map<string, SavedStrategy>,
  portfolioById: Map<string, SavedPortfolio>,
): number {
  if (row.source === 'portfolio' && row.portfolio_id) {
    return portfolioById.get(row.portfolio_id)?.scan_sort_order ?? 0
  }
  if (row.strategy_id) {
    return strategyById.get(row.strategy_id)?.scan_sort_order ?? 0
  }
  return 0
}

function sortLaneRows(
  rows: ScanRow[],
  strategyById: Map<string, SavedStrategy>,
  portfolioById: Map<string, SavedPortfolio>,
): ScanRow[] {
  return [...rows].sort((a, b) => {
    const orderDiff = rowSortOrder(a, strategyById, portfolioById) - rowSortOrder(b, strategyById, portfolioById)
    if (orderDiff !== 0) return orderDiff
    const symbolDiff = compareSymbols(a.symbol, b.symbol)
    if (symbolDiff !== 0) return symbolDiff
    return a.signal.localeCompare(b.signal)
  })
}

export function groupScanRows(
  rows: ScanRow[],
  strategyById: Map<string, SavedStrategy>,
  portfolioById: Map<string, SavedPortfolio>,
): { lanes: LaneRows; legacy: ScanRow[] } {
  const lanes: LaneRows = {
    active: [],
    testing: [],
    archived: [],
  }
  const legacy: ScanRow[] = []

  for (const row of rows) {
    if (row.source === 'legacy') {
      legacy.push(row)
      continue
    }
    if (row.source === 'portfolio') {
      const portfolio = row.portfolio_id ? portfolioById.get(row.portfolio_id) : undefined
      lanes[resolvePortfolioScanLane(portfolio)].push(row)
      continue
    }
    const strategy = row.strategy_id ? strategyById.get(row.strategy_id) : undefined
    lanes[resolveScanLane(strategy)].push(row)
  }

  for (const lane of SCAN_LANES) {
    lanes[lane] = sortLaneRows(lanes[lane], strategyById, portfolioById)
  }

  return { lanes, legacy: sortLegacyRows(legacy) }
}

export function findLaneForRow(laneRows: LaneRows, rowId: string): ScanLane | null {
  for (const lane of SCAN_LANES) {
    if (laneRows[lane].some((row) => row.id === rowId)) return lane
  }
  return null
}

export function countActiveSignals(rows: ScanRow[]) {
  return rows.filter((row) => row.buy_signal || row.hold_long).length
}
