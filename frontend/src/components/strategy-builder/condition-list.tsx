'use client'

import { memo } from 'react'
import { Plus, Trash2 } from 'lucide-react'
import type { BuilderConditionPayload, CompareMode, IndicatorInfo } from '@/api'
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
  filterCompareIndicators,
  formatConditionOperator,
  getCompareMode,
  getIndicator,
  isFlagIndicator,
  isInvalidCompareRight,
  normalizeConditionForLeft,
  resolveCompareRightKind,
  type CompareRightKind,
} from '@/lib/strategy-builder'
import { IndicatorSelect, indicatorLabel } from '@/components/strategy-builder/indicator-select'
import { CompareValueHint } from '@/components/strategy-builder/compare-value-hint'

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

function CompareRightToggle({
  value,
  onChange,
}: {
  value: CompareRightKind
  onChange: (next: CompareRightKind) => void
}) {
  return (
    <div className="flex shrink-0 overflow-hidden rounded border border-border/60">
      {(['indicator', 'number'] as const).map((kind) => (
        <button
          key={kind}
          type="button"
          onClick={() => onChange(kind)}
          className={cn(
            'px-2 py-0.5 text-[10px] font-medium transition-colors',
            value === kind
              ? 'bg-primary/20 text-foreground'
              : 'text-muted-foreground hover:bg-muted',
          )}
        >
          {kind === 'indicator' ? 'Indicator' : 'Value'}
        </button>
      ))}
    </div>
  )
}

function CompareRightOperand({
  condition,
  indicators,
  rightIndicators,
  rightIndicatorIds,
  onChange,
}: {
  condition: ConditionRow
  indicators: IndicatorInfo[]
  rightIndicators: IndicatorInfo[]
  rightIndicatorIds: Set<string>
  onChange: (patch: Partial<ConditionRow>) => void
}) {
  const leftIndicator = getIndicator(indicators, condition.left)
  const compareMode = getCompareMode(indicators, condition.left)
  const filteredRightIndicators = filterCompareIndicators(rightIndicators, condition.left)
  const filteredRightIds = new Set(filteredRightIndicators.map((item) => item.id))
  const rightKind = resolveCompareRightKind(
    indicators,
    condition.left,
    condition.right,
    rightIndicatorIds,
  )
  const invalidRight = isInvalidCompareRight(
    indicators,
    condition.left,
    condition.right,
    rightIndicatorIds,
  )
  const numberPlaceholder = leftIndicator?.defaultCompareNumber ?? 'Value'

  const setRightKind = (kind: CompareRightKind) => {
    const defaultRight =
      kind === 'indicator'
        ? leftIndicator?.defaultCompareIndicator ?? 'Close'
        : leftIndicator?.defaultCompareNumber ?? ''
    onChange({ right: defaultRight })
  }

  if (compareMode === 'number') {
    return (
      <div className="flex items-center gap-1">
        <Input
          value={condition.right}
          onChange={(e) => onChange({ right: e.target.value })}
          className={cn(
            'h-8 w-[5.5rem] shrink-0 font-mono text-xs',
            invalidRight && 'border-[var(--loss)]',
          )}
          placeholder={numberPlaceholder}
        />
        <CompareValueHint indicator={leftIndicator} />
      </div>
    )
  }

  if (compareMode === 'indicator') {
    return (
      <IndicatorSelect
        value={filteredRightIds.has(condition.right) ? condition.right : ''}
        onChange={(right) => onChange({ right })}
        indicators={filteredRightIndicators}
        compact
        placeholder="Indicator"
        className={cn('min-w-[12rem] flex-[2]', invalidRight && 'ring-1 ring-[var(--loss)]')}
      />
    )
  }

  return (
    <>
      <CompareRightToggle value={rightKind} onChange={setRightKind} />
      {rightKind === 'indicator' ? (
        <IndicatorSelect
          value={filteredRightIds.has(condition.right) ? condition.right : ''}
          onChange={(right) => onChange({ right })}
          indicators={filteredRightIndicators}
          compact
          placeholder="Indicator"
          className={cn('min-w-[12rem] flex-[2]', invalidRight && 'ring-1 ring-[var(--loss)]')}
        />
      ) : (
        <div className="flex items-center gap-1">
          <Input
            value={condition.right}
            onChange={(e) => onChange({ right: e.target.value })}
            className={cn(
              'h-8 w-[5.5rem] shrink-0 font-mono text-xs',
              invalidRight && 'border-[var(--loss)]',
            )}
            placeholder={numberPlaceholder}
          />
          <CompareValueHint indicator={leftIndicator} />
        </div>
      )}
    </>
  )
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
        const compareMode: CompareMode = getCompareMode(indicators, c.left)
        const invalidRight =
          !flagCondition &&
          compareMode !== 'none' &&
          isInvalidCompareRight(indicators, c.left, c.right, rightIndicatorIds)

        return (
          <div
            key={c.id}
            className="flex flex-col gap-1 rounded-md border border-border/50 bg-muted/15 px-2 py-1.5"
          >
            <div className="flex flex-wrap items-center gap-1.5">
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
                  update(c.id, {
                    left,
                    ...normalizeConditionForLeft(indicators, left, c, rightIndicatorIds),
                  })
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
                    onValueChange={(v) =>
                      v && update(c.id, { operator: v as ConditionRow['operator'] })
                    }
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
                  <CompareRightOperand
                    condition={c}
                    indicators={indicators}
                    rightIndicators={rightIndicators}
                    rightIndicatorIds={rightIndicatorIds}
                    onChange={(patch) => update(c.id, patch)}
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
            {invalidRight ? (
              <p className="pl-[60px] text-[10px] text-[var(--loss)]">
                Choose a valid compare target for this indicator.
              </p>
            ) : null}
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
  {
    label,
    required,
    rightIndicators = indicators,
  }: { label: string; required: boolean; rightIndicators?: IndicatorInfo[] },
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
    if (!flagCondition) {
      const allRightIds = new Set(rightIndicators.map((item) => item.id))
      if (
        isInvalidCompareRight(
          indicators,
          condition.left,
          condition.right,
          allRightIds,
        )
      ) {
        throw new Error(`Each ${label.toLowerCase()} needs a valid compare target`)
      }
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
