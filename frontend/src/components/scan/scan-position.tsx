'use client'

import { useEffect, useRef, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Loader2 } from 'lucide-react'
import { getExactScanPosition, type ScanOrder, type ScanPosition, type ScanRow } from '@/api'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import {
  actionChipClass,
  formatDollars,
  formatShare,
  hasDetail,
  instrumentLabel,
  positionOf,
  positionSummary,
  sizeMove,
  withPosition,
} from '@/lib/scan-position'
import { cn } from '@/lib/utils'

export function ActionChip({ row, className }: { row: ScanRow; className?: string }) {
  const position = positionOf(row)
  return (
    <span
      className={cn(
        'inline-block rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide',
        actionChipClass(position.action),
        className,
      )}
    >
      {position.action}
      {position.approximate ? <span title={position.approximate}> ≈</span> : null}
    </span>
  )
}

export function ScanCapitalInput({
  text,
  onChange,
  className,
  id = 'scan-capital',
}: {
  text: string
  onChange: (value: string) => void
  className?: string
  id?: string
}) {
  return (
    <div className={cn('flex shrink-0 items-center gap-1.5', className)}>
      <label htmlFor={id} className="whitespace-nowrap text-xs text-muted-foreground">
        Account $
      </label>
      <Input
        id={id}
        inputMode="decimal"
        value={text}
        onChange={(event) => onChange(event.target.value)}
        placeholder="optional, e.g. 50k"
        className="h-7 w-32 text-xs placeholder:italic placeholder:text-muted-foreground/45"
        title="Shown as dollars and shares in a row's details. Kept in this browser only."
      />
    </div>
  )
}

function OrderLine({ order, capital }: { order: ScanOrder; capital: number | null }) {
  const sizing = sizeMove(order, order.weight_after - order.weight_before, capital)
  const quotable = order.kind === 'vehicle' || (order.leverage ?? 1) === 1
  return (
    <li className="flex items-start justify-between gap-3 text-[11px] leading-4">
      <span className={order.side === 'buy' ? 'text-[var(--gain)]' : 'text-[var(--scan-exit)]'}>
        <span className="font-medium">{order.side === 'buy' ? 'Buy' : 'Sell'}</span> {instrumentLabel(order)}
      </span>
      <span className="shrink-0 text-right font-mono tabular-nums">
        {formatShare(order.weight_before)} → {formatShare(order.weight_after)}
        {sizing ? (
          <span className="block text-muted-foreground">
            {formatDollars(sizing.dollars)}
            {sizing.shares !== null && order.price
              ? ` · ≈ ${sizing.shares.toLocaleString('en-US')} sh @ $${order.price.toFixed(2)}`
              : !quotable
                ? ' · no product to trade'
                : ''}
          </span>
        ) : null}
      </span>
    </li>
  )
}

function PositionDetail({
  row,
  position,
  capital,
}: {
  row: ScanRow
  position: ScanPosition
  capital: number | null
}) {
  const queryClient = useQueryClient()
  const exact = useMutation({
    mutationFn: () => getExactScanPosition(row.portfolio_id as string),
    onSuccess: (next) => {
      // Every scan result in the cache, whichever data source it came from, gets the exact row.
      queryClient.setQueriesData<ScanRow[]>({ queryKey: ['scan'] }, (rows) =>
        rows?.map((item) => (item.id === row.id ? withPosition(item, next) : item)),
      )
    },
  })
  const drift = Math.abs(position.held_now - position.target) >= 0.005
  return (
    <div className="max-h-80 overflow-y-auto">
      <div className="border-b border-border px-3 py-2 text-[11px] text-muted-foreground">
        <span className="font-medium text-foreground">{position.action}</span> · {position.model}
        {position.as_of ? ` · ${position.as_of}` : ''}
        {position.reason ? <span className="mt-0.5 block">{position.reason}</span> : null}
      </div>
      <div className="grid grid-cols-3 gap-2 border-b border-border px-3 py-2 text-[11px]">
        <div>
          <span className="block text-muted-foreground">Held before</span>
          <span className="font-mono tabular-nums">{formatShare(position.held_before)}</span>
        </div>
        <div>
          <span className="block text-muted-foreground">Held now</span>
          <span className="font-mono tabular-nums">{formatShare(position.held_now)}</span>
        </div>
        <div>
          <span className="block text-muted-foreground">Target</span>
          <span className="font-mono tabular-nums">{formatShare(position.target)}</span>
        </div>
        {drift && position.action === 'Hold' ? (
          <p className="col-span-3 text-muted-foreground">
            The position drifted with price. The model trades only when a signal changes, so there is nothing to do.
          </p>
        ) : null}
      </div>
      {position.orders.length > 0 ? (
        <div className="border-b border-border px-3 py-2">
          <p className="mb-1 text-[11px] text-muted-foreground">Orders today</p>
          <ul className="space-y-1.5">
            {position.orders.map((order) => (
              <OrderLine key={`${order.instrument}-${order.side}`} order={order} capital={capital} />
            ))}
          </ul>
        </div>
      ) : null}
      {position.holdings.length > 0 ? (
        <div className="border-b border-border px-3 py-2">
          <p className="mb-1 text-[11px] text-muted-foreground">Held after today</p>
          <ul className="space-y-1">
            {position.holdings.map((holding) => {
              const sizing = sizeMove(holding, holding.weight, capital)
              return (
                <li key={holding.instrument} className="flex justify-between gap-3 text-[11px]">
                  <span>{instrumentLabel(holding)}</span>
                  <span className="font-mono tabular-nums">
                    {formatShare(holding.weight)}
                    {sizing ? <span className="text-muted-foreground"> · {formatDollars(sizing.dollars)}</span> : null}
                  </span>
                </li>
              )
            })}
          </ul>
        </div>
      ) : null}
      {position.active_strategies.length > 0 ? (
        <div className="border-b border-border px-3 py-2 text-[11px]">
          <p className="mb-1 text-muted-foreground">
            Active ({position.active}
            {position.total ? ` of ${position.total}` : ''})
          </p>
          <p>{position.active_strategies.join(', ')}</p>
        </div>
      ) : null}
      {position.approximate ? (
        <div className="space-y-1.5 px-3 py-2 text-[11px]">
          <p className="text-muted-foreground">{position.approximate}</p>
          {row.portfolio_id ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-6 gap-1 px-2 text-[11px]"
              disabled={exact.isPending}
              onClick={() => exact.mutate()}
            >
              {exact.isPending ? <Loader2 className="size-3 animate-spin" /> : null}
              Compute exact
            </Button>
          ) : null}
          {exact.error ? <p className="text-destructive">{String(exact.error)}</p> : null}
        </div>
      ) : null}
    </div>
  )
}

/** The Position cell: how much of the book is held, and where it is going. Click for orders and details. */
export function PositionSummary({ row, capital }: { row: ScanRow; capital: number | null }) {
  const position = positionOf(row)
  const [open, setOpen] = useState(false)
  const triggerRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!open) return
    const onPointerDown = (event: MouseEvent) => {
      const target = event.target as Node
      if (triggerRef.current?.contains(target)) return
      const popup = document.querySelector('[data-slot="tooltip-content"]')
      if (popup?.contains(target)) return
      setOpen(false)
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('mousedown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  const text = positionSummary(position)
  if (!hasDetail(position)) {
    return <span className="font-mono text-[11px] tabular-nums text-muted-foreground">{text}</span>
  }
  return (
    <TooltipProvider delay={0}>
      <Tooltip open={open} onOpenChange={setOpen}>
        <TooltipTrigger className="inline-flex" onClick={(event) => event.stopPropagation()}>
          <button
            ref={triggerRef}
            type="button"
            className="font-mono text-[11px] tabular-nums underline decoration-dotted underline-offset-2 hover:text-foreground"
            aria-label="Show orders and position details"
            onClick={(event) => {
              event.stopPropagation()
              setOpen((current) => !current)
            }}
          >
            {text}
          </button>
        </TooltipTrigger>
        <TooltipContent
          side="bottom"
          align="start"
          className="max-w-sm min-w-[18rem] border border-border bg-card p-0 text-foreground shadow-md [&>svg]:hidden"
        >
          <PositionDetail row={row} position={position} capital={capital} />
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
