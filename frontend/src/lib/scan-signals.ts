import type { ScanRow } from '@/api'

export type ScanSignalKind = 'buy' | 'hold' | 'sell'
export type ScanRowMood = 'flat' | 'entry' | 'holding' | 'exit'

export function isScanSignalActive(row: ScanRow, kind: ScanSignalKind): boolean {
  if (kind === 'buy') return row.buy_signal
  if (kind === 'hold') return row.hold_long
  return row.sell_signal
}

/** Primary row state for scan accent colors. Buy and hold never fire on the same bar. */
export function resolveScanRowMood(row: ScanRow): ScanRowMood {
  if (row.sell_signal && row.hold_long) return 'exit'
  if (row.buy_signal) return 'entry'
  if (row.hold_long) return 'holding'
  return 'flat'
}

export function scanRowAccentClass(row: ScanRow): string {
  switch (resolveScanRowMood(row)) {
    case 'entry':
      return 'bg-[var(--gain)]/8'
    case 'holding':
      return 'bg-[var(--scan-hold)]/8'
    case 'exit':
      return 'bg-[var(--scan-exit)]/10'
    default:
      return ''
  }
}

export function scanCardAccentClass(row: ScanRow): string {
  switch (resolveScanRowMood(row)) {
    case 'entry':
      return 'border-[var(--gain)]/30 bg-[var(--gain)]/5'
    case 'holding':
      return 'border-[var(--scan-hold)]/30 bg-[var(--scan-hold)]/5'
    case 'exit':
      return 'border-[var(--scan-exit)]/35 bg-[var(--scan-exit)]/8'
    default:
      return ''
  }
}

export function scanSignalIconClass(row: ScanRow, kind: ScanSignalKind): string {
  if (!isScanSignalActive(row, kind)) {
    return 'text-muted-foreground/50'
  }

  if (kind === 'buy') {
    return 'text-[var(--gain)]'
  }

  if (kind === 'sell') {
    return 'text-[var(--scan-exit)]'
  }

  if (row.sell_signal) {
    return 'text-muted-foreground/40'
  }

  return 'text-[var(--scan-hold)]'
}

export function scanSignalBadgeClass(row: ScanRow, kind: ScanSignalKind): string {
  if (!isScanSignalActive(row, kind)) {
    return 'bg-muted text-muted-foreground'
  }

  if (kind === 'buy') {
    return 'bg-[var(--gain)]/15 text-[var(--gain)]'
  }

  if (kind === 'sell') {
    return 'bg-[var(--scan-exit)]/15 text-[var(--scan-exit)]'
  }

  if (row.sell_signal) {
    return 'bg-muted/80 text-muted-foreground/55'
  }

  return 'bg-[var(--scan-hold)]/15 text-[var(--scan-hold)]'
}