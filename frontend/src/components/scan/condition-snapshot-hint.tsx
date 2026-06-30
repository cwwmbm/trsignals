'use client'

import { useEffect, useRef, useState } from 'react'
import { CircleHelp } from 'lucide-react'
import type { ConditionSnapshotItem, ScanRow } from '@/api'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

function formatValues(item: ConditionSnapshotItem): string | null {
  if (item.left_value && item.right_value) {
    return `${item.left_value} / ${item.right_value}`
  }
  if (item.left_value) {
    return item.left_value
  }
  return null
}

type ConditionSnapshotHintProps = {
  row: ScanRow
}

export function ConditionSnapshotHint({ row }: ConditionSnapshotHintProps) {
  const snapshot = row.condition_snapshot
  const [open, setOpen] = useState(false)
  const triggerRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!open) return

    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node
      if (triggerRef.current?.contains(target)) return
      const popup = document.querySelector('[data-slot="tooltip-content"]')
      if (popup?.contains(target)) return
      setOpen(false)
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }

    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [open])

  if (row.source !== 'builder' || !snapshot?.length) {
    return null
  }

  const passed = row.condition_passed_count ?? snapshot.filter((item) => item.passed).length
  const total = row.condition_total_count ?? snapshot.length
  const asOf = row.condition_as_of

  return (
    <TooltipProvider delay={0}>
      <Tooltip open={open} onOpenChange={setOpen}>
        <TooltipTrigger
          className="inline-flex"
          onClick={(event) => event.stopPropagation()}
        >
          <button
            ref={triggerRef}
            type="button"
            className="inline-flex size-4 shrink-0 items-center justify-center rounded-full text-muted-foreground hover:bg-muted hover:text-foreground"
            aria-label="Show entry condition snapshot"
            onClick={(event) => {
              event.stopPropagation()
              setOpen((current) => !current)
            }}
          >
            <CircleHelp className="size-3.5" />
          </button>
        </TooltipTrigger>
        <TooltipContent
          side="bottom"
          align="start"
          className="max-w-sm border border-border bg-card p-0 text-foreground shadow-md [&>svg]:hidden"
        >
          <div className="border-b border-border px-3 py-2 text-[11px] text-muted-foreground">
            Entry conditions
            {asOf ? ` · ${asOf}` : ''}
            {` · ${passed}/${total} pass`}
          </div>
          <ul className="max-h-64 space-y-1 overflow-y-auto px-3 py-2">
            {snapshot.map((item, index) => {
              const values = formatValues(item)
              return (
                <li
                  key={`${item.left}-${item.operator}-${item.right}-${index}`}
                  className={cn(
                    'flex items-start justify-between gap-3 text-[11px] leading-4',
                    item.passed ? 'text-[var(--gain)]' : 'text-[var(--loss)]',
                  )}
                >
                  <span className="min-w-0">
                    {item.logic ? <span className="mr-1 opacity-70">{item.logic}</span> : null}
                    {item.label}
                  </span>
                  {values ? (
                    <span className="shrink-0 font-mono tabular-nums opacity-90">{values}</span>
                  ) : null}
                </li>
              )
            })}
          </ul>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
