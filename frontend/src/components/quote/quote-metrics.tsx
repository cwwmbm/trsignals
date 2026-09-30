import type { QuoteSymbol } from '@/api'
import { formatMetric, formatSignedPercent } from '@/lib/format-metric'
import { cn } from '@/lib/utils'

export function changeTone(value: number | null): string {
  if (value == null || Number.isNaN(value) || value === 0) return 'text-foreground'
  return value > 0 ? 'text-[var(--gain)]' : 'text-[var(--loss)]'
}

export function formatMissingDay(iso: string): string {
  const [year, month, day] = iso.split('-').map(Number)
  if (!year || !month || !day) return iso
  const date = new Date(year, month - 1, day)
  const weekday = date.toLocaleDateString('en-US', { weekday: 'short' })
  return `${weekday} ${month}/${day}`
}

export function ChangeCell({
  row,
  className,
}: {
  row: QuoteSymbol
  className?: string
}) {
  const tone = changeTone(row.pct_change)
  if (row.symbol === 'VIX') {
    const level = formatMetric(row.close)
    const change = row.pct_change == null ? null : formatSignedPercent(row.pct_change)
    return (
      <span className={cn('font-mono tabular-nums', className)}>
        {level}
        {change ? <span className={cn('ml-1', tone)}>({change})</span> : null}
      </span>
    )
  }
  return (
    <span className={cn('font-mono tabular-nums', tone, className)}>
      {formatSignedPercent(row.pct_change)}
    </span>
  )
}
