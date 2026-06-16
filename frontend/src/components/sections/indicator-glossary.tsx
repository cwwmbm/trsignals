'use client'

import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getIndicators } from '@/api'
import { groupIndicators } from '@/lib/indicator-groups'
import { CollapsibleSection } from '@/components/strategy-builder/collapsible-section'

function formatTypicalRange(min: number, max: number) {
  const fmt = (value: number) => {
    const abs = Math.abs(value)
    if (abs >= 1 || abs === 0) return String(value)
    return value.toFixed(2).replace(/\.?0+$/, '')
  }
  return `${fmt(min)}–${fmt(max)}`
}

export function IndicatorGlossary() {
  const { data: indicators = [], isLoading, isError } = useQuery({
    queryKey: ['indicators'],
    queryFn: () => getIndicators(true),
  })

  const groups = useMemo(() => groupIndicators(indicators), [indicators])

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading indicators…</p>
  }

  if (isError) {
    return <p className="text-sm text-muted-foreground">Could not load indicators.</p>
  }

  return (
    <div className="flex flex-col gap-1">
      {[...groups.entries()].map(([category, items]) => (
        <CollapsibleSection
          key={category}
          title={category}
          summary={`${items.length} indicator${items.length === 1 ? '' : 's'}`}
          defaultOpen={false}
        >
          <dl className="grid grid-cols-1 gap-x-8 gap-y-4 sm:grid-cols-2">
            {items.map((item) => (
              <div key={item.id} className="flex flex-col gap-0.5">
                <dt className="text-sm font-semibold">{item.label}</dt>
                <dd className="text-sm leading-relaxed text-muted-foreground">
                  {item.description ?? 'No description available.'}
                </dd>
                {item.typicalRange ? (
                  <dd className="text-xs text-muted-foreground/80">
                    Typical sweep range: {formatTypicalRange(item.typicalRange.min, item.typicalRange.max)}
                  </dd>
                ) : null}
              </div>
            ))}
          </dl>
        </CollapsibleSection>
      ))}
    </div>
  )
}
