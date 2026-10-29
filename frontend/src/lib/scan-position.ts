import type { ScanAction, ScanHolding, ScanOrder, ScanPosition, ScanRow } from '@/api'

/** 55, or 54.6 when it is not a whole number, so a small change is never rounded away. */
export function formatShare(weight: number): string {
  const pct = weight * 100
  const whole = Math.round(pct)
  return Math.abs(pct - whole) < 0.05 ? `${whole}%` : `${pct.toFixed(1)}%`
}

/** The short line in the table: how much of the book is held, and where it is going. */
export function positionSummary(position: ScanPosition): string {
  switch (position.action) {
    case 'Flat':
      return '—'
    case 'Rebalance':
      // the total is unchanged; what moved is the split between instruments
      return `${formatShare(position.held_now)} (new split)`
    case 'Hold': {
      const drift = Math.abs(position.held_now - position.target) >= 0.005
      return drift
        ? `${formatShare(position.held_now)} (target ${formatShare(position.target)})`
        : formatShare(position.held_now)
    }
    default:
      return `${formatShare(position.held_before)} → ${formatShare(position.held_now)}`
  }
}

/** The buy / hold / sell flags the rest of the page still reads, from a position (as the server computes them). */
export function flagsFor(position: ScanPosition) {
  const action: ScanAction = position.action
  return {
    buy_signal: action === 'Enter' || action === 'Add' || action === 'Rebalance',
    hold_long: position.held_before > 1e-9,
    sell_signal: action === 'Exit' || action === 'Reduce' || action === 'Rebalance',
  }
}

/** A scan row with a new position (and the flags that follow from it). */
export function withPosition(row: ScanRow, position: ScanPosition): ScanRow {
  return { ...row, ...flagsFor(position), position }
}

export function actionChipClass(action: ScanAction): string {
  switch (action) {
    case 'Enter':
      return 'bg-[var(--gain)]/15 text-[var(--gain)]'
    case 'Add':
      return 'bg-[var(--gain)]/10 text-[var(--gain)]'
    case 'Reduce':
      return 'bg-[var(--scan-exit)]/10 text-[var(--scan-exit)]'
    case 'Exit':
      return 'bg-[var(--scan-exit)]/15 text-[var(--scan-exit)]'
    case 'Rebalance':
      return 'bg-muted text-foreground'
    case 'Hold':
      return 'bg-[var(--scan-hold)]/15 text-[var(--scan-hold)]'
    default:
      return 'bg-muted text-muted-foreground'
  }
}

/** The text of an instrument as you would say it: the product, or the symbol with its stand-in leverage. */
export function instrumentLabel(item: Pick<ScanOrder | ScanHolding, 'instrument' | 'underlying' | 'kind' | 'leverage'>): string {
  if (item.kind === 'vehicle') return item.instrument
  const leverage = item.leverage ?? 1
  return leverage === 1 ? item.underlying : `${item.underlying} ×${leverage} (stand-in)`
}

/** Whole shares can be quoted when the instrument is a real product or an unleveraged symbol. */
function canQuoteShares(item: Pick<ScanOrder, 'kind' | 'leverage'>): boolean {
  return item.kind === 'vehicle' || (item.leverage ?? 1) === 1
}

export type Sizing = { dollars: number; shares: number | null }

/** Dollars and (when it makes sense) shares for a move of `weight` of the book, given an account size. */
export function sizeMove(
  item: Pick<ScanOrder, 'kind' | 'leverage' | 'price'>,
  weight: number,
  capital: number | null,
): Sizing | null {
  if (capital === null || !(capital > 0)) return null
  const dollars = capital * Math.abs(weight)
  const shares = canQuoteShares(item) && item.price && item.price > 0 ? Math.round(dollars / item.price) : null
  return { dollars, shares }
}

export function formatDollars(value: number): string {
  return `$${Math.round(value).toLocaleString('en-US')}`
}

/** An account size typed as "50000", "$50,000" or "50k"; null when it is not a positive number. */
export function parseCapital(text: string): number | null {
  const cleaned = text.trim().toLowerCase().replace(/[$,\s]/g, '')
  if (!cleaned) return null
  const multiplier = cleaned.endsWith('k') ? 1_000 : cleaned.endsWith('m') ? 1_000_000 : 1
  const number = Number(multiplier === 1 ? cleaned : cleaned.slice(0, -1)) * multiplier
  return Number.isFinite(number) && number > 0 ? number : null
}

/** Whether a position has anything to show beyond its one-line summary. */
export function hasDetail(position: ScanPosition): boolean {
  return (
    position.orders.length > 0 ||
    position.holdings.length > 0 ||
    position.active_strategies.length > 0 ||
    position.approximate !== null
  )
}

/** The row's position, or one read from its buy / hold / sell flags when the server sent none. */
export function positionOf(row: ScanRow): ScanPosition {
  if (row.position) return row.position
  const held_before = row.hold_long ? 1 : 0
  const held_now = row.buy_signal || (row.hold_long && !row.sell_signal) ? 1 : 0
  const action: ScanAction =
    row.sell_signal && row.hold_long ? 'Exit' : row.buy_signal ? 'Enter' : row.hold_long ? 'Hold' : 'Flat'
  return {
    action,
    held_before,
    held_now,
    target: held_now,
    delta: held_now - held_before,
    active: 0,
    total: 0,
    active_strategies: [],
    entered: [],
    exited: [],
    reason: '',
    orders: [],
    holdings: [],
    model: 'Single strategy',
    as_of: null,
    approximate: null,
  }
}
