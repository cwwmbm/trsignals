'use client'

import { memo } from 'react'
import { Plus, Trash2 } from 'lucide-react'
import type { BuilderConditionPayload, IndicatorInfo } from '@/api'
import { isFlagOperator } from '@/api'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { cn } from '@/lib/utils'
import { OPERATORS } from '@/lib/mock-data'
import {
  formatConditionOperator,
  isFlagIndicator,
  normalizeConditionForLeft,
} from '@/lib/strategy-builder'
import { IndicatorSelect, indicatorLabel } from '@/components/strategy-builder/indicator-select'

export type ConditionRow = {
  id: string
  left: string
  operator: BuilderConditionPayload['operator']
  right: string
  logic: 'AND' | 'OR'
}

let counter = 0

export function newConditionRow(
  overrides: Partial<Omit<ConditionRow, 'id'>> = {},
): ConditionRow {
  return {
    id: `c${++counter}`,
    left: 'RSI2',
    operator: '<=',
    right: '20',
    logic: 'AND',
    ...overrides,
  }
}

export function formatConditionPreview(
  conditions: ConditionRow[],
  indicators: IndicatorInfo[],
  indicatorIds: Set<string>,
) {
  return conditions
    .map((c, i) => {
      const left = indicatorLabel(indicators, c.left)
      const flagCondition = isFlagIndicator(indicators, c.left)
      if (flagCondition) {
        return `${i > 0 ? ` ${c.logic} ` : ''}${left} ${formatConditionOperator(c.operator)}`
      }
      const right = indicatorIds.has(c.right) ? indicatorLabel(indicators, c.right) : c.right
      return `${i > 0 ? ` ${c.logic} ` : ''}${left} ${c.operator} ${right}`
    })
    .join('')
}

export const ConditionList = memo(function ConditionList({
  conditions,
  onChange,
  indicators,
  indicatorIds,
  rightIndicators = indicators,
  rightIndicatorIds = indicatorIds,
  minConditions = 1,
  emptyHint,
}: {
  conditions: ConditionRow[]
  onChange: (next: ConditionRow[] | ((prev: ConditionRow[]) => ConditionRow[])) => void
  indicators: IndicatorInfo[]
  indicatorIds: Set<string>
  rightIndicators?: IndicatorInfo[]
  rightIndicatorIds?: Set<string>
  minConditions?: number
  emptyHint?: string
}) {
  const update = (id: string, patch: Partial<ConditionRow>) =>
    onChange((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch } : c)))

  if (conditions.length === 0) {
    return (
      <p className="rounded-md border border-dashed border-border/60 px-3 py-2 text-xs text-muted-foreground">
        {emptyHint ?? 'No conditions yet.'}
      </p>
    )
  }

  return (
    <div className="flex flex-col gap-1.5">
      {conditions.map((c, i) => {
        const flagCondition = isFlagIndicator(indicators, c.left)
        return (
          <div
            key={c.id}
            className="flex flex-wrap items-center gap-1.5 rounded-md border border-border/50 bg-muted/15 px-2 py-1.5"
          >
            {i > 0 ? (
              <div className="flex shrink-0 overflow-hidden rounded border border-border/60">
                {(['AND', 'OR'] as const).map((l) => (
                  <button
                    key={l}
                    type="button"
                    onClick={() => update(c.id, { logic: l })}
                    className={cn(
                      'px-2 py-0.5 text-[10px] font-medium transition-colors',
                      c.logic === l
                        ? 'bg-primary/20 text-foreground'
                        : 'text-muted-foreground hover:bg-muted',
                    )}
                  >
                    {l}
                  </button>
                ))}
              </div>
            ) : (
              <span className="w-[52px] shrink-0 text-center text-[10px] font-medium text-muted-foreground">
                IF
              </span>
            )}

            <IndicatorSelect
              value={c.left}
              onChange={(left) =>
                update(c.id, { left, ...normalizeConditionForLeft(indicators, left, c) })
              }
              indicators={indicators}
              compact
              className="min-w-[12rem] flex-[2]"
            />
            {flagCondition ? (
              <Select
                value={c.operator}
                onValueChange={(v) =>
                  v && update(c.id, { operator: v as ConditionRow['operator'], right: '' })
                }
              >
                <SelectTrigger size="sm" className="h-8 min-w-[6.5rem] shrink-0 text-xs">
                  <SelectValue>{(value: string) => formatConditionOperator(value)}</SelectValue>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="is true">Is true</SelectItem>
                  <SelectItem value="is false">Is false</SelectItem>
                </SelectContent>
              </Select>
            ) : (
              <>
                <Select
                  value={c.operator}
                  onValueChange={(v) => v && update(c.id, { operator: v as ConditionRow['operator'] })}
                >
                  <SelectTrigger size="sm" className="h-8 w-[5.5rem] shrink-0 font-mono text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {OPERATORS.map((op) => (
                      <SelectItem key={op} value={op} className="font-mono text-xs">
                        {op}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <IndicatorSelect
                  value={rightIndicatorIds.has(c.right) ? c.right : ''}
                  onChange={(right) => update(c.id, { right })}
                  indicators={rightIndicators}
                  compact
                  placeholder="Indicator"
                  className="min-w-[12rem] flex-[2]"
                />
                <Input
                  value={rightIndicatorIds.has(c.right) ? '' : c.right}
                  onChange={(e) => update(c.id, { right: e.target.value })}
                  className="h-8 w-[5.5rem] shrink-0 font-mono text-xs"
                  placeholder="Value"
                />
              </>
            )}
            <Button
              variant="ghost"
              size="icon"
              className="size-7 shrink-0 text-muted-foreground hover:text-[var(--loss)]"
              onClick={() => onChange((prev) => prev.filter((x) => x.id !== c.id))}
              aria-label="Remove condition"
              disabled={conditions.length <= minConditions}
            >
              <Trash2 className="size-3.5" />
            </Button>
          </div>
        )
      })}
    </div>
  )
})

export function AddConditionButton({
  onClick,
  label = 'Add',
}: {
  onClick: () => void
  label?: string
}) {
  return (
    <Button variant="outline" size="sm" className="h-7 gap-1 px-2 text-xs" onClick={onClick}>
      <Plus className="size-3.5" />
      {label}
    </Button>
  )
}

export function validateConditions(
  conditions: ConditionRow[],
  indicators: IndicatorInfo[],
  { label, required }: { label: string; required: boolean },
) {
  if (required && conditions.length === 0) {
    throw new Error(`${label} needs at least one condition`)
  }
  for (const condition of conditions) {
    if (!condition.left) throw new Error(`Each ${label.toLowerCase()} needs a left indicator`)
    const flagCondition =
      isFlagIndicator(indicators, condition.left) || isFlagOperator(condition.operator)
    if (!flagCondition && !condition.right.trim()) {
      throw new Error(`Each ${label.toLowerCase()} needs a right value or indicator`)
    }
  }
}

export function toConditionPayload(conditions: ConditionRow[]): BuilderConditionPayload[] {
  return conditions.map(({ left, operator, right, logic }) => ({
    left,
    operator,
    right,
    logic,
  }))
}
