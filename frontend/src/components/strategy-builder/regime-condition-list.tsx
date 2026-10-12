'use client'

import { Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  REGIME_DIMENSIONS,
  parseRegimeIndicatorId,
  regimeIndicatorId,
} from '@/lib/market-regimes'
import { newConditionRow, type ConditionRow } from '@/components/strategy-builder/condition-list'

/** A market regime row. Regimes always AND with the entry conditions, so there is no logic to pick. */
export function newRegimeRow(overrides: Partial<Omit<ConditionRow, 'id'>> = {}): ConditionRow {
  const first = REGIME_DIMENSIONS[0]
  return newConditionRow({
    left: regimeIndicatorId(first.id, first.buckets[0].key),
    operator: 'is true',
    right: '',
    logic: 'AND',
    ...overrides,
  })
}

const OPERATOR_LABEL: Record<string, string> = { 'is true': 'Is', 'is false': 'Is not' }

export function RegimeConditionList({
  conditions,
  onChange,
}: {
  conditions: ConditionRow[]
  onChange: (next: ConditionRow[] | ((prev: ConditionRow[]) => ConditionRow[])) => void
}) {
  const update = (id: string, patch: Partial<ConditionRow>) =>
    onChange((prev) => prev.map((row) => (row.id === id ? { ...row, ...patch } : row)))
  const remove = (id: string) => onChange((prev) => prev.filter((row) => row.id !== id))

  if (conditions.length === 0) {
    return (
      <p className="rounded-md border border-dashed border-border/60 px-3 py-2 text-xs text-muted-foreground">
        No market regime. Entries are not limited to any market condition.
      </p>
    )
  }

  return (
    <div className="flex flex-col gap-1.5">
      {conditions.map((row, index) => {
        const parsed = parseRegimeIndicatorId(row.left)
        const dimension = parsed?.dimension ?? REGIME_DIMENSIONS[0]
        return (
          <div
            key={row.id}
            className="flex flex-wrap items-center gap-1.5 rounded-md border border-border/50 bg-muted/15 px-2 py-1.5"
          >
            <span className="w-[52px] shrink-0 text-center text-[10px] font-medium text-muted-foreground">
              {index === 0 ? 'IF' : 'AND'}
            </span>
            <Select
              value={dimension.id}
              onValueChange={(value) => {
                const next = REGIME_DIMENSIONS.find((item) => item.id === value)
                if (next) update(row.id, { left: regimeIndicatorId(next.id, next.buckets[0].key) })
              }}
            >
              <SelectTrigger size="sm" className="h-8 min-w-[12rem] flex-[2] text-xs" aria-label="Market regime">
                <SelectValue>
                  {(value: string) => REGIME_DIMENSIONS.find((item) => item.id === value)?.title ?? value}
                </SelectValue>
              </SelectTrigger>
              <SelectContent className="max-h-[min(28rem,70vh)] w-max min-w-[var(--anchor-width)]">
                {REGIME_DIMENSIONS.map((item) => (
                  <SelectItem key={item.id} value={item.id} className="text-xs">
                    {item.title}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select
              value={row.operator === 'is false' ? 'is false' : 'is true'}
              onValueChange={(value) =>
                value && update(row.id, { operator: value as ConditionRow['operator'] })
              }
            >
              <SelectTrigger size="sm" className="h-8 w-[5.5rem] shrink-0 text-xs" aria-label="Is or is not">
                <SelectValue>{(value: string) => OPERATOR_LABEL[value] ?? value}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="is true">Is</SelectItem>
                <SelectItem value="is false">Is not</SelectItem>
              </SelectContent>
            </Select>
            <Select
              value={parsed?.bucket.key ?? dimension.buckets[0].key}
              onValueChange={(value) =>
                value && update(row.id, { left: regimeIndicatorId(dimension.id, value) })
              }
            >
              <SelectTrigger size="sm" className="h-8 min-w-[9rem] flex-1 text-xs" aria-label="Regime state">
                <SelectValue>
                  {(value: string) => dimension.buckets.find((item) => item.key === value)?.label ?? value}
                </SelectValue>
              </SelectTrigger>
              <SelectContent className="w-max min-w-[var(--anchor-width)]">
                {dimension.buckets.map((bucket) => (
                  <SelectItem key={bucket.key} value={bucket.key} className="text-xs">
                    {bucket.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              variant="ghost"
              size="icon"
              className="size-7 shrink-0 text-muted-foreground hover:text-[var(--loss)]"
              onClick={() => remove(row.id)}
              aria-label="Remove market regime"
            >
              <Trash2 className="size-3.5" />
            </Button>
          </div>
        )
      })}
      <p className="text-[11px] text-muted-foreground">
        Market regimes are always ANDed with the entry conditions: regimes AND (condition 1 OR condition 2).
        "Is not" means any other labeled state of that regime.
      </p>
    </div>
  )
}
