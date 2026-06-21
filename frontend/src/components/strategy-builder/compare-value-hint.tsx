'use client'

import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { CircleHelp } from 'lucide-react'
import type { IndicatorInfo } from '@/api'
import { cn } from '@/lib/utils'
import { formatTypicalRangeHint } from '@/lib/strategy-builder'

const PANEL_WIDTH = 272
const PANEL_GAP = 6
const PANEL_ESTIMATED_HEIGHT = 160

function computePanelPosition(trigger: HTMLElement, panelHeight: number) {
  const rect = trigger.getBoundingClientRect()
  const viewportPadding = 8

  let left = rect.right - PANEL_WIDTH
  left = Math.max(viewportPadding, Math.min(left, window.innerWidth - PANEL_WIDTH - viewportPadding))

  const spaceBelow = window.innerHeight - rect.bottom - viewportPadding
  const spaceAbove = rect.top - viewportPadding
  const openAbove = spaceBelow < panelHeight + PANEL_GAP && spaceAbove > spaceBelow

  const top = openAbove
    ? Math.max(viewportPadding, rect.top - panelHeight - PANEL_GAP)
    : rect.bottom + PANEL_GAP

  return { top, left }
}

export function CompareValueHint({
  indicator,
  className,
}: {
  indicator: IndicatorInfo | undefined
  className?: string
}) {
  const [open, setOpen] = useState(false)
  const [mounted, setMounted] = useState(false)
  const [position, setPosition] = useState<{ top: number; left: number } | null>(null)
  const panelId = useId()
  const buttonRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)

  const description = indicator?.description?.trim()
  const typicalRange = indicator?.typicalRange

  useEffect(() => {
    setMounted(true)
  }, [])

  useLayoutEffect(() => {
    if (!open || !buttonRef.current) {
      setPosition(null)
      return
    }

    function updatePosition() {
      if (!buttonRef.current) return
      const panelHeight = panelRef.current?.offsetHeight ?? PANEL_ESTIMATED_HEIGHT
      setPosition(computePanelPosition(buttonRef.current, panelHeight))
    }

    updatePosition()
    const raf = window.requestAnimationFrame(updatePosition)

    window.addEventListener('resize', updatePosition)
    window.addEventListener('scroll', updatePosition, true)

    return () => {
      window.cancelAnimationFrame(raf)
      window.removeEventListener('resize', updatePosition)
      window.removeEventListener('scroll', updatePosition, true)
    }
  }, [open, description, typicalRange])

  useEffect(() => {
    if (!open) return
    function handlePointerDown(event: MouseEvent) {
      const target = event.target as Node
      if (buttonRef.current?.contains(target) || panelRef.current?.contains(target)) return
      setOpen(false)
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [open])

  if (!indicator || (!description && !typicalRange)) return null

  const panel =
    open && mounted ? (
      <div
        id={panelId}
        ref={panelRef}
        role="dialog"
        style={{
          top: position?.top ?? -9999,
          left: position?.left ?? -9999,
          width: PANEL_WIDTH,
          visibility: position ? 'visible' : 'hidden',
        }}
        className="fixed z-[100] rounded-md border border-border bg-popover p-3 text-popover-foreground shadow-md"
      >
        <p className="text-xs font-semibold leading-snug">{indicator.label}</p>
        {description ? (
          <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{description}</p>
        ) : null}
        {typicalRange ? (
          <div className="mt-2 space-y-1 border-t border-border/60 pt-2">
            <p className="text-[11px] font-medium text-foreground">
              Typical threshold: {formatTypicalRangeHint(typicalRange)}
            </p>
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              Indicator sweeps test values in this range to find useful entry and exit levels.
              Start near the default placeholder, then adjust for your symbol and timeframe.
            </p>
          </div>
        ) : null}
      </div>
    ) : null

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        aria-label={`About ${indicator.label}`}
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((value) => !value)}
        className={cn(
          'inline-flex size-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors',
          'hover:bg-muted hover:text-foreground',
          open && 'bg-muted text-foreground',
          className,
        )}
      >
        <CircleHelp className="size-3.5" />
      </button>
      {mounted && panel ? createPortal(panel, document.body) : null}
    </>
  )
}
