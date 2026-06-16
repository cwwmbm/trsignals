'use client'

import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useState,
} from 'react'
import { Loader2, Play, RotateCcw, Save } from 'lucide-react'
import {
  type BuilderConditionPayload,
  type BuilderBacktestPayload,
  type IndicatorInfo,
  type SavedStrategy,
  type SaveStrategyPayload,
} from '@/api'
import { isFlagOperator } from '@/api'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { cn } from '@/lib/utils'
import { isFlagIndicator } from '@/lib/strategy-builder'
import { csv } from '@/lib/backtest-form'
import { updateStrategyBuilderDraftPreview } from '@/lib/strategy-builder-draft-store'
import { CollapsibleSection } from '@/components/strategy-builder/collapsible-section'
import {
  AddConditionButton,
  ConditionList,
  formatConditionPreview,
  newConditionRow,
  toConditionPayload,
  validateConditions,
  type ConditionRow,
} from '@/components/strategy-builder/condition-list'
import { indicatorLabel } from '@/components/strategy-builder/indicator-select'
import {
  holdDaysSweepRowValues,
  isHoldDaysSweepRow,
  isSymbolConfirmSweepRow,
  sweepRowSide,
  sweepRowToConditionRow,
  symbolConfirmSweepRowValues,
} from '@/lib/sweep-to-condition'

const STRATEGY_INDICATOR_PREFIX = 'strategy:'

function savedStrategyToIndicator(strategy: SavedStrategy): IndicatorInfo {
  return {
    id: `${STRATEGY_INDICATOR_PREFIX}${strategy.id}`,
    label: strategy.name,
    kind: 'signal_flag',
    valueType: 'flag',
    category: 'Saved strategies',
    description: strategy.description || `Saved strategy ${strategy.name}`,
    builderEligible: true,
  }
}

function uniqueSavedStrategiesByName(strategies: SavedStrategy[], preferredSymbol: string) {
  const seen = new Set<string>()
  const sorted = [...strategies].sort((a, b) => {
    const aPreferred = a.symbol === preferredSymbol
    const bPreferred = b.symbol === preferredSymbol
    if (aPreferred === bPreferred) return 0
    return aPreferred ? -1 : 1
  })
  return sorted.filter((strategy) => {
    const key = strategy.name.trim().toLowerCase()
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

function defaultEntryConditions(): ConditionRow[] {
  return [
    { id: 'c0', left: 'Close', operator: '<=', right: 'SMA200', logic: 'AND' },
  ]
}

function conditionRowsFromPayload(
  conditions: BuilderConditionPayload[] | undefined,
  prefix: string,
): ConditionRow[] {
  return (conditions ?? []).map((condition, index) => ({
    id: `${prefix}-${index}`,
    left: condition.left,
    operator: condition.operator,
    right: condition.right,
    logic: condition.logic,
  }))
}

function buildPayload(
  symbol: string,
  direction: string,
  holdDays: string,
  profitableCloses: string,
  name: string,
  description: string,
  confirmSymbols: string,
  proxySymbol: string,
  entryConditions: ConditionRow[],
  exitConditions: ConditionRow[],
  entryIndicators: IndicatorInfo[],
  exitIndicators: IndicatorInfo[],
) {
  const parsedHoldDays = Number(holdDays)
  const parsedProfit = Number(profitableCloses)
  if (!symbol.trim()) throw new Error('Primary symbol is required')
  if (!Number.isFinite(parsedHoldDays) || parsedHoldDays < 1) {
    throw new Error('Hold days must be at least 1')
  }
  if (!Number.isFinite(parsedProfit) || parsedProfit < 0) {
    throw new Error('Profitable closes must be 0 or more')
  }
  validateConditions(entryConditions, entryIndicators, {
    label: 'Entry condition',
    required: true,
  })
  validateConditions(exitConditions, exitIndicators, {
    label: 'Exit condition',
    required: false,
  })

  const primary = symbol.trim().toUpperCase()
  const proxy = proxySymbol.trim().toUpperCase()
  const normalizedProxy = proxy && proxy !== primary ? proxy : undefined

  return {
    symbol: primary,
    years: 25,
    direction: direction === 'short' ? 'short' : 'long',
    hold_days: parsedHoldDays,
    profit: parsedProfit,
    name: name.trim(),
    description: description.trim(),
    conditions: toConditionPayload(entryConditions),
    sell_conditions: toConditionPayload(exitConditions),
    confirm_symbols: csv(confirmSymbols),
    ...(normalizedProxy ? { proxy_symbol: normalizedProxy } : {}),
  } satisfies BuilderBacktestPayload
}

function buildSavePayload(
  symbol: string,
  direction: string,
  holdDays: string,
  profitableCloses: string,
  name: string,
  description: string,
  confirmSymbols: string,
  proxySymbol: string,
  entryConditions: ConditionRow[],
  exitConditions: ConditionRow[],
  entryIndicators: IndicatorInfo[],
  exitIndicators: IndicatorInfo[],
): SaveStrategyPayload {
  const payload = buildPayload(
    symbol,
    direction,
    holdDays,
    profitableCloses,
    name,
    description,
    confirmSymbols,
    proxySymbol,
    entryConditions,
    exitConditions,
    entryIndicators,
    exitIndicators,
  )
  if (!payload.name) {
    throw new Error('Strategy name is required to save')
  }
  return {
    name: payload.name,
    symbol: payload.symbol,
    direction: payload.direction,
    hold_days: payload.hold_days,
    profit: payload.profit ?? 1,
    description: payload.description,
    conditions: payload.conditions,
    sell_conditions: payload.sell_conditions,
    confirm_symbols: payload.confirm_symbols,
    ...(payload.proxy_symbol ? { proxy_symbol: payload.proxy_symbol } : {}),
  }
}

function Field({
  label,
  htmlFor,
  children,
  className,
}: {
  label: string
  htmlFor?: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex min-w-0 flex-col gap-1', className)}>
      <Label htmlFor={htmlFor} className="text-[11px] text-muted-foreground">
        {label}
      </Label>
      {children}
    </div>
  )
}

function exitSummary(
  holdDays: string,
  profitableCloses: string,
  exitConditions: ConditionRow[],
) {
  const rules =
    exitConditions.length === 0
      ? 'no indicator exits'
      : `${exitConditions.length} rule${exitConditions.length === 1 ? '' : 's'}`
  return `${holdDays}d hold · ${profitableCloses} profit close${profitableCloses === '1' ? '' : 's'} · ${rules}`
}

function notesSummary(description: string, entryPreview: string, exitPreview: string) {
  if (description.trim()) {
    const trimmed = description.trim()
    return trimmed.length > 48 ? `${trimmed.slice(0, 48)}…` : trimmed
  }
  const rules = [entryPreview, exitPreview].filter(Boolean).join(' · ')
  return rules || 'Optional notes'
}

function isIncompleteCondition(row: ConditionRow, indicators: IndicatorInfo[]) {
  if (isFlagIndicator(indicators, row.left) || isFlagOperator(row.operator)) {
    return false
  }
  return !row.right.trim()
}

function appendConditionRow(
  prev: ConditionRow[],
  next: ConditionRow,
  indicators: IndicatorInfo[],
  minKeep = 0,
) {
  const rows = [...prev]
  while (
    rows.length > minKeep &&
    isIncompleteCondition(rows[rows.length - 1], indicators)
  ) {
    rows.pop()
  }
  return [...rows, next]
}

export type StrategyBuilderSetupHandle = {
  buildBacktestPayload: () => BuilderBacktestPayload
  buildSavePayload: () => SaveStrategyPayload
  buildRefineDraft: () => BuilderBacktestPayload
  addFromSweepRow: (row: Record<string, unknown>) => string | null
}

type StrategyBuilderSetupProps = {
  indicators: IndicatorInfo[]
  savedStrategies: SavedStrategy[]
  initialStrategy?: SavedStrategy
  onSymbolChange?: (symbol: string) => void
  onRunBacktest: (payload: BuilderBacktestPayload) => void
  onSave: (payload: SaveStrategyPayload) => void
  onReset?: () => void
  isBacktestRunning: boolean
  isSaving: boolean
}

export const StrategyBuilderSetup = forwardRef<
  StrategyBuilderSetupHandle,
  StrategyBuilderSetupProps
>(function StrategyBuilderSetup(
  {
    indicators,
    savedStrategies,
    initialStrategy,
    onSymbolChange,
    onRunBacktest,
    onSave,
    onReset,
    isBacktestRunning,
    isSaving,
  },
  ref,
) {
  const [validationError, setValidationError] = useState<string | null>(null)
  const [entryOpen, setEntryOpen] = useState(true)
  const [exitOpen, setExitOpen] = useState(false)
  const [name, setName] = useState('')
  const [symbol, setSymbol] = useState('SPY')
  const [direction, setDirection] = useState('long')
  const [holdDays, setHoldDays] = useState('2')
  const [profitableCloses, setProfitableCloses] = useState('1')
  const [confirmSymbols, setConfirmSymbols] = useState('')
  const [proxySymbol, setProxySymbol] = useState('')
  const [description, setDescription] = useState('')
  const [entryConditions, setEntryConditions] = useState<ConditionRow[]>(defaultEntryConditions)
  const [exitConditions, setExitConditions] = useState<ConditionRow[]>([])

  const draftSymbol = symbol.trim().toUpperCase() || 'SPY'
  const builderIndicators = useMemo(
    () => [
      ...indicators,
      ...uniqueSavedStrategiesByName(savedStrategies, draftSymbol).map(savedStrategyToIndicator),
    ],
    [draftSymbol, indicators, savedStrategies],
  )
  const staticIndicatorIds = useMemo(
    () => new Set(indicators.map((item) => item.id)),
    [indicators],
  )
  const indicatorIds = useMemo(
    () => new Set(builderIndicators.map((item) => item.id)),
    [builderIndicators],
  )

  const entryPreview = formatConditionPreview(entryConditions, builderIndicators, indicatorIds)
  const exitPreview = formatConditionPreview(exitConditions, indicators, staticIndicatorIds)

  const draftValid = useMemo(() => {
    try {
      validateConditions(entryConditions, builderIndicators, {
        label: 'Entry condition',
        required: true,
      })
      return true
    } catch {
      return false
    }
  }, [entryConditions, builderIndicators])

  useEffect(() => {
    updateStrategyBuilderDraftPreview({
      draftName: name,
      draftSymbol,
      draftHoldDays: Number(holdDays) || 2,
      draftProfit: Number(profitableCloses) || 1,
      entryPreview,
      exitPreview,
      draftValid,
    })
  }, [
    name,
    draftSymbol,
    holdDays,
    profitableCloses,
    entryPreview,
    exitPreview,
    draftValid,
  ])

  useEffect(() => {
    onSymbolChange?.(draftSymbol)
  }, [draftSymbol, onSymbolChange])

  function resetFormToDefaults() {
    setValidationError(null)
    setName('')
    setSymbol('SPY')
    setDirection('long')
    setHoldDays('2')
    setProfitableCloses('1')
    setConfirmSymbols('')
    setProxySymbol('')
    setDescription('')
    setEntryConditions(defaultEntryConditions())
    setExitConditions([])
    setEntryOpen(true)
    setExitOpen(false)
    onReset?.()
  }

  useEffect(() => {
    if (!initialStrategy) {
      resetFormToDefaults()
      return
    }

    setValidationError(null)
    const entryRows = conditionRowsFromPayload(
      initialStrategy.conditions,
      `entry-${initialStrategy.id}`,
    )
    const exitRows = conditionRowsFromPayload(
      initialStrategy.sell_conditions,
      `exit-${initialStrategy.id}`,
    )
    setName(initialStrategy.name)
    setSymbol(initialStrategy.symbol)
    setDirection(initialStrategy.direction)
    setHoldDays(String(initialStrategy.hold_days))
    setProfitableCloses(String(initialStrategy.profit))
    setConfirmSymbols((initialStrategy.confirm_symbols ?? []).join(', '))
    setProxySymbol(initialStrategy.proxy_symbol ?? '')
    setDescription(initialStrategy.description ?? '')
    setEntryConditions(entryRows.length > 0 ? entryRows : defaultEntryConditions())
    setExitConditions(exitRows)
    setEntryOpen(true)
    setExitOpen(exitRows.length > 0)
  }, [initialStrategy])

  useImperativeHandle(
    ref,
    () => ({
      buildBacktestPayload: () =>
        buildPayload(
          symbol,
          direction,
          holdDays,
          profitableCloses,
          name,
          description,
          confirmSymbols,
          proxySymbol,
          entryConditions,
          exitConditions,
          builderIndicators,
          indicators,
        ),
      buildSavePayload: () =>
        buildSavePayload(
          symbol,
          direction,
          holdDays,
          profitableCloses,
          name,
          description,
          confirmSymbols,
          proxySymbol,
          entryConditions,
          exitConditions,
          builderIndicators,
          indicators,
        ),
      buildRefineDraft: () =>
        buildPayload(
          symbol,
          direction,
          holdDays,
          profitableCloses,
          name,
          description,
          confirmSymbols,
          proxySymbol,
          entryConditions,
          exitConditions,
          builderIndicators,
          indicators,
        ),
      addFromSweepRow: (row: Record<string, unknown>) => {
        if (isHoldDaysSweepRow(row)) {
          const values = holdDaysSweepRowValues(row)
          if (!values) return 'Could not apply hold days from this sweep row.'
          setHoldDays(String(values.holdDays))
          setProfitableCloses(String(values.profit))
          setExitOpen(true)
          return `Applied ${values.holdDays}d hold · ${values.profit} profit close${values.profit === 1 ? '' : 's'}.`
        }

        if (isSymbolConfirmSweepRow(row)) {
          const value = symbolConfirmSweepRowValues(row)
          if (value === null) return 'Could not apply confirmation symbols from this sweep row.'
          setConfirmSymbols(value)
          return value
            ? `Applied confirmation: ${value}.`
            : 'Cleared confirmation symbols.'
        }

        const mapped = sweepRowToConditionRow(row, builderIndicators)
        if (!mapped) return 'Could not map this sweep row to a builder condition.'

        const side = sweepRowSide(row)
        const target = side === 'Sell' ? 'exit' : 'entry'
        const previewLeft = indicatorLabel(builderIndicators, mapped.left)
        const preview =
          mapped.operator === 'is true' || mapped.operator === 'is false'
            ? `${previewLeft} ${mapped.operator}`
            : `${previewLeft} ${mapped.operator} ${mapped.right}`

        if (target === 'exit') {
          setExitConditions((prev) => appendConditionRow(prev, mapped, builderIndicators))
          setExitOpen(true)
        } else {
          setEntryConditions((prev) => appendConditionRow(prev, mapped, builderIndicators, 1))
          setEntryOpen(true)
        }

        return `Added ${preview} to ${target}.`
      },
    }),
    [
      symbol,
      direction,
      holdDays,
      profitableCloses,
      name,
      description,
      confirmSymbols,
      entryConditions,
      exitConditions,
      builderIndicators,
      indicators,
    ],
  )

  function handleRunBuilderBacktest() {
    setValidationError(null)
    try {
      onRunBacktest(
        buildPayload(
          symbol,
          direction,
          holdDays,
          profitableCloses,
          name,
          description,
          confirmSymbols,
          proxySymbol,
          entryConditions,
          exitConditions,
          builderIndicators,
          indicators,
        ),
      )
    } catch (error) {
      setValidationError(String(error))
    }
  }

  function handleSave() {
    setValidationError(null)
    try {
      onSave(
        buildSavePayload(
          symbol,
          direction,
          holdDays,
          profitableCloses,
          name,
          description,
          confirmSymbols,
          proxySymbol,
          entryConditions,
          exitConditions,
          builderIndicators,
          indicators,
        ),
      )
    } catch (error) {
      setValidationError(String(error))
    }
  }

  return (
    <>
      <Card className="border-border/60 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-semibold">Strategy builder</h2>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              className="h-8 gap-1.5"
              onClick={resetFormToDefaults}
            >
              <RotateCcw className="size-3.5" />
              Reset
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-8 gap-1.5"
              onClick={handleSave}
              disabled={isSaving}
            >
              {isSaving ? <Loader2 className="size-3.5 animate-spin" /> : <Save className="size-3.5" />}
              {isSaving ? 'Saving…' : 'Save'}
            </Button>
            <Button
              size="sm"
              className="h-8 gap-1.5"
              onClick={handleRunBuilderBacktest}
              disabled={isBacktestRunning}
            >
              {isBacktestRunning ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <Play className="size-3.5" />
              )}
              {isBacktestRunning ? 'Running…' : 'Backtest'}
            </Button>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-2 gap-x-3 gap-y-2 sm:grid-cols-3">
          <Field label="Name" htmlFor="strat-name" className="col-span-2 sm:col-span-1">
            <Input
              id="strat-name"
              placeholder="Required to save"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="h-8 text-sm"
            />
          </Field>
          <Field label="Symbol" htmlFor="strat-symbol">
            <Input
              id="strat-symbol"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              className="h-8 font-mono text-sm"
            />
          </Field>
          <Field label="Direction" htmlFor="strat-dir">
            <Select value={direction} onValueChange={(v) => v && setDirection(v)}>
              <SelectTrigger id="strat-dir" size="sm" className="h-8 w-full text-sm">
                <SelectValue>{(value: string) => (value === 'long' ? 'Long' : 'Short')}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="long">Long</SelectItem>
                <SelectItem value="short">Short</SelectItem>
              </SelectContent>
            </Select>
          </Field>
        </div>

        <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
          <Field label="Confirm symbols" htmlFor="strat-confirm-symbols">
            <Input
              id="strat-confirm-symbols"
              placeholder="Optional, e.g. SMH, QQQ"
              value={confirmSymbols}
              onChange={(e) => setConfirmSymbols(e.target.value.toUpperCase())}
              className="h-8 font-mono text-sm"
            />
          </Field>
          <Field label="Proxy symbol" htmlFor="strat-proxy-symbol">
            <Input
              id="strat-proxy-symbol"
              placeholder="Optional, e.g. SOXX"
              value={proxySymbol}
              onChange={(e) => setProxySymbol(e.target.value.toUpperCase())}
              className="h-8 font-mono text-sm"
            />
          </Field>
        </div>

        <div className="mt-3 space-y-2">
          <CollapsibleSection
            title="Entry conditions"
            summary={entryPreview || '—'}
            open={entryOpen}
            onOpenChange={setEntryOpen}
            actions={
              <AddConditionButton
                onClick={() => setEntryConditions((prev) => [...prev, newConditionRow()])}
              />
            }
          >
            <ConditionList
              conditions={entryConditions}
              onChange={setEntryConditions}
              indicators={builderIndicators}
              indicatorIds={indicatorIds}
              rightIndicators={indicators}
              rightIndicatorIds={staticIndicatorIds}
              minConditions={1}
            />
          </CollapsibleSection>

          <CollapsibleSection
            title="Exit conditions"
            summary={exitSummary(holdDays, profitableCloses, exitConditions)}
            open={exitOpen}
            onOpenChange={setExitOpen}
            actions={
              <AddConditionButton
                label="Add rule"
                onClick={() =>
                  setExitConditions((prev) => [...prev, newConditionRow({ logic: 'OR' })])
                }
              />
            }
          >
            <div className="mb-2 grid grid-cols-2 gap-x-3 gap-y-2 sm:max-w-md">
              <Field label="Hold days" htmlFor="strat-hold">
                <Input
                  id="strat-hold"
                  inputMode="numeric"
                  value={holdDays}
                  onChange={(e) => setHoldDays(e.target.value)}
                  className="h-8 font-mono text-sm"
                />
              </Field>
              <Field label="Profitable closes" htmlFor="strat-profit">
                <Input
                  id="strat-profit"
                  inputMode="numeric"
                  value={profitableCloses}
                  onChange={(e) => setProfitableCloses(e.target.value)}
                  className="h-8 font-mono text-sm"
                />
              </Field>
            </div>
            <p className="mb-2 text-[11px] text-muted-foreground">
              Optional indicator exits (OR-combined). Trades also exit after max hold days or enough
              profitable closes.
            </p>
            <ConditionList
              conditions={exitConditions}
              onChange={setExitConditions}
              indicators={indicators}
              indicatorIds={staticIndicatorIds}
              minConditions={0}
              emptyHint="No indicator exit rules — exits use hold days and profitable closes only."
            />
          </CollapsibleSection>

          <CollapsibleSection
            title="Description & preview"
            summary={notesSummary(description, entryPreview, exitPreview)}
            defaultOpen={false}
          >
            <Field label="Description" htmlFor="strat-desc">
              <Textarea
                id="strat-desc"
                placeholder="Optional notes…"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                className="min-h-0 resize-none text-sm"
              />
            </Field>
            <div className="mt-2 grid gap-2 sm:grid-cols-2">
              <Field label="Entry rule">
                <pre className="overflow-x-auto rounded-md border border-border/60 bg-background/60 px-2 py-1.5 font-mono text-[11px] leading-snug text-foreground">
                  {entryPreview || '—'}
                </pre>
              </Field>
              <Field label="Exit rule">
                <pre className="overflow-x-auto rounded-md border border-border/60 bg-background/60 px-2 py-1.5 font-mono text-[11px] leading-snug text-foreground">
                  {exitPreview || '—'}
                </pre>
              </Field>
            </div>
          </CollapsibleSection>
        </div>
      </Card>

      {validationError && (
        <Card className="border-destructive/50 bg-destructive/5 p-3">
          <pre className="overflow-x-auto whitespace-pre-wrap text-xs text-destructive">
            {validationError}
          </pre>
        </Card>
      )}
    </>
  )
})
