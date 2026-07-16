import { Check, X } from 'lucide-react'
import type { ScanRow } from '@/api'
import {
  type ScanSignalKind,
  scanSignalBadgeClass,
  scanSignalIconClass,
} from '@/lib/scan-signals'
import { cn } from '@/lib/utils'

const BADGE_LABELS: Record<ScanSignalKind, string> = {
  buy: 'B',
  hold: 'H',
  sell: 'S',
}

export function ScanSignalCheck({
  row,
  kind,
}: {
  row: ScanRow
  kind: ScanSignalKind
}) {
  const active =
    kind === 'buy' ? row.buy_signal : kind === 'hold' ? row.hold_long : row.sell_signal

  return active ? (
    <Check className={cn('mx-auto size-3', scanSignalIconClass(row, kind))} aria-label="True" />
  ) : (
    <X className="mx-auto size-3 text-muted-foreground/50" aria-label="False" />
  )
}

export function ScanSignalBadge({
  row,
  kind,
}: {
  row: ScanRow
  kind: ScanSignalKind
}) {
  const active =
    kind === 'buy' ? row.buy_signal : kind === 'hold' ? row.hold_long : row.sell_signal

  return (
    <span
      className={cn(
        'rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide',
        scanSignalBadgeClass(row, kind),
      )}
    >
      {BADGE_LABELS[kind]}
    </span>
  )
}
