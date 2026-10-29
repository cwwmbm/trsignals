import type { ScanRow } from '@/api'

export type ScanRowMood = 'flat' | 'entry' | 'holding' | 'exit'

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
