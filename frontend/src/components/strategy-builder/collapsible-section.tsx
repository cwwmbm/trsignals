'use client'

import * as React from 'react'
import { ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'

export function CollapsibleSection({
  title,
  summary,
  defaultOpen = true,
  open,
  onOpenChange,
  children,
  actions,
}: {
  title: string
  summary?: string
  defaultOpen?: boolean
  open?: boolean
  onOpenChange?: (open: boolean) => void
  children: React.ReactNode
  actions?: React.ReactNode
}) {
  const isControlled = open !== undefined
  const [internalOpen, setInternalOpen] = React.useState(defaultOpen)
  const expanded = isControlled ? open : internalOpen

  function toggle() {
    const next = !expanded
    if (isControlled) {
      onOpenChange?.(next)
    } else {
      setInternalOpen(next)
    }
  }

  return (
    <div className="border-t border-border/60 pt-2 first:border-t-0 first:pt-0">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={toggle}
          className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
          aria-expanded={expanded}
        >
          <ChevronDown
            className={cn(
              'size-4 shrink-0 text-muted-foreground transition-transform',
              !expanded && '-rotate-90',
            )}
          />
          <span className="text-sm font-medium">{title}</span>
          {!expanded && summary && (
            <span className="truncate text-xs text-muted-foreground">{summary}</span>
          )}
        </button>
        {expanded && actions ? <div className="shrink-0">{actions}</div> : null}
      </div>
      {expanded ? <div className="mt-2">{children}</div> : null}
    </div>
  )
}
