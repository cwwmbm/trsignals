'use client'

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import {
  ArrowUpRight,
  Check,
  ChevronDown,
  Loader2,
  Pencil,
  Search,
  Trash2,
  X,
} from 'lucide-react'
import {
  deleteStrategy,
  getIndicators,
  getSavedStrategies,
  updateStrategy,
  type IndicatorInfo,
  type SavedStrategy,
} from '@/api'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { formatConditionPreview } from '@/components/strategy-builder/condition-list'
import { compareSymbols } from '@/lib/symbol-order'
import { cn } from '@/lib/utils'

const STRATEGY_INDICATOR_PREFIX = 'strategy:'

const compactHead = 'h-7 px-1.5 py-0 text-[11px] font-medium'
const compactCell = 'px-1.5 py-0.5 align-top'

function savedStrategyIndicator(strategy: SavedStrategy): IndicatorInfo {
  return {
    id: `${STRATEGY_INDICATOR_PREFIX}${strategy.id}`,
    label: strategy.name,
    kind: 'signal_flag',
    valueType: 'flag',
    category: 'Saved strategies',
    builderEligible: true,
  }
}

function conditionPreview(
  conditions: SavedStrategy['conditions'],
  indicators: IndicatorInfo[],
  savedStrategies: SavedStrategy[],
) {
  if (conditions.length === 0) return '—'
  const allIndicators = [...indicators, ...savedStrategies.map(savedStrategyIndicator)]
  const ids = new Set(allIndicators.map((item) => item.id))
  return formatConditionPreview(
    conditions.map((condition, index) => ({
      id: `preview-${index}`,
      ...condition,
    })),
    allIndicators,
    ids,
  )
}

function exitPreview(
  strategy: SavedStrategy,
  indicators: IndicatorInfo[],
  savedStrategies: SavedStrategy[],
) {
  if (strategy.sell_conditions && strategy.sell_conditions.length > 0) {
    return conditionPreview(strategy.sell_conditions, indicators, savedStrategies)
  }
  return `${strategy.hold_days}d hold · ${strategy.profit} profit close${strategy.profit === 1 ? '' : 's'}`
}

function sortStrategies(strategies: SavedStrategy[]): SavedStrategy[] {
  return [...strategies].sort((a, b) => {
    const symbolDiff = compareSymbols(a.symbol, b.symbol)
    if (symbolDiff !== 0) return symbolDiff
    return a.name.localeCompare(b.name)
  })
}

function groupStrategiesBySymbol(strategies: SavedStrategy[]) {
  const groups = new Map<string, SavedStrategy[]>()
  for (const strategy of strategies) {
    const existing = groups.get(strategy.symbol) ?? []
    existing.push(strategy)
    groups.set(strategy.symbol, existing)
  }

  return [...groups.entries()]
    .sort(([symbolA], [symbolB]) => compareSymbols(symbolA, symbolB))
    .map(([symbol, items]) => ({
      symbol,
      strategies: [...items].sort((a, b) => a.name.localeCompare(b.name)),
    }))
}

type StrategyRowProps = {
  strategy: SavedStrategy
  strategies: SavedStrategy[]
  indicators: IndicatorInfo[]
  editingId: string | null
  descriptionDraft: string
  deleteConfirmId: string | null
  updatePending: boolean
  deletePending: boolean
  onDescriptionDraftChange: (value: string) => void
  onStartEditing: (strategy: SavedStrategy) => void
  onCancelEditing: () => void
  onSaveDescription: (strategy: SavedStrategy) => void
  onBacktest?: (strategy: SavedStrategy) => void
  onDeleteClick: (strategy: SavedStrategy, isDeleting: boolean) => void
}

function StrategyRow({
  strategy,
  strategies,
  indicators,
  editingId,
  descriptionDraft,
  deleteConfirmId,
  updatePending,
  deletePending,
  onDescriptionDraftChange,
  onStartEditing,
  onCancelEditing,
  onSaveDescription,
  onBacktest,
  onDeleteClick,
}: StrategyRowProps) {
  const isEditing = editingId === strategy.id
  const isDeleting = deleteConfirmId === strategy.id
  const entry = conditionPreview(strategy.conditions, indicators, strategies)
  const exit = exitPreview(strategy, indicators, strategies)
  const confirmLabel =
    strategy.confirm_symbols && strategy.confirm_symbols.length > 0
      ? strategy.confirm_symbols.join('+')
      : null

  return (
    <TableRow className="hover:bg-muted/30">
      <TableCell className={compactCell}>
        <div className="font-medium leading-snug">{strategy.name}</div>
        <div className="text-[11px] leading-snug text-muted-foreground">
          {strategy.direction === 'short' ? 'Short' : 'Long'}
          {confirmLabel ? ` · confirm ${confirmLabel}` : ''}
        </div>
      </TableCell>
      <TableCell className={cn(compactCell, 'whitespace-normal')}>
        <div className="space-y-0.5 leading-snug">
          <div>
            <span className="text-muted-foreground">Entry </span>
            <span className="font-mono text-[11px]">{entry}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Exit </span>
            <span className="font-mono text-[11px]">{exit}</span>
          </div>
        </div>
      </TableCell>
      <TableCell className={cn(compactCell, 'whitespace-normal')}>
        {isEditing ? (
          <div className="flex flex-col gap-1.5">
            <Textarea
              value={descriptionDraft}
              onChange={(event) => onDescriptionDraftChange(event.target.value)}
              rows={2}
              className="min-h-0 resize-none text-[11px]"
            />
            <div className="flex items-center gap-1">
              <Button
                size="sm"
                className="h-6 gap-1 px-1.5 text-[11px]"
                onClick={() => onSaveDescription(strategy)}
                disabled={updatePending}
              >
                {updatePending ? (
                  <Loader2 className="size-3 animate-spin" />
                ) : (
                  <Check className="size-3" />
                )}
                Save
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className="h-6 gap-1 px-1.5 text-[11px]"
                onClick={onCancelEditing}
              >
                <X className="size-3" />
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <div className="group/description flex items-start gap-1">
            <p
              className="max-w-[320px] truncate text-[11px] leading-snug text-muted-foreground"
              title={strategy.description || 'No description.'}
            >
              {strategy.description || 'No description.'}
            </p>
            <Button
              variant="ghost"
              size="icon"
              className="size-5 shrink-0 text-muted-foreground opacity-60 hover:opacity-100"
              onClick={() => onStartEditing(strategy)}
              aria-label={`Edit ${strategy.name} description`}
            >
              <Pencil className="size-3" />
            </Button>
          </div>
        )}
      </TableCell>
      <TableCell className={compactCell}>
        <div className="flex justify-end gap-0.5">
          <Button
            variant="ghost"
            size="sm"
            className="h-6 gap-1 px-1.5 text-[11px]"
            onClick={() => onBacktest?.(strategy)}
          >
            Backtest
            <ArrowUpRight className="size-3" />
          </Button>
          <Button
            variant={isDeleting ? 'destructive' : 'ghost'}
            size="sm"
            className="h-6 gap-1 px-1.5 text-[11px]"
            disabled={deletePending}
            onClick={() => onDeleteClick(strategy, isDeleting)}
          >
            {deletePending && isDeleting ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Trash2 className="size-3" />
            )}
            {isDeleting ? 'Confirm' : 'Delete'}
          </Button>
        </div>
      </TableCell>
    </TableRow>
  )
}

export function StrategiesSection({
  onNewStrategy,
  onBacktestStrategy,
}: {
  onNewStrategy?: () => void
  onBacktestStrategy?: (strategy: SavedStrategy) => void
}) {
  const queryClient = useQueryClient()
  const [query, setQuery] = useState('')
  const [expandedSymbols, setExpandedSymbols] = useState<Set<string>>(() => new Set())
  const [editingId, setEditingId] = useState<string | null>(null)
  const [descriptionDraft, setDescriptionDraft] = useState('')
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)

  const { data: strategies = [], isLoading, error } = useQuery({
    queryKey: ['strategies'],
    queryFn: getSavedStrategies,
  })
  const { data: indicators = [] } = useQuery({
    queryKey: ['indicators'],
    queryFn: () => getIndicators(true),
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, description }: { id: string; description: string }) =>
      updateStrategy(id, { description }),
    onSuccess: () => {
      setEditingId(null)
      setDescriptionDraft('')
      queryClient.invalidateQueries({ queryKey: ['strategies'] })
      queryClient.invalidateQueries({ queryKey: ['scan'] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteStrategy,
    onSuccess: () => {
      setDeleteConfirmId(null)
      queryClient.invalidateQueries({ queryKey: ['strategies'] })
      queryClient.invalidateQueries({ queryKey: ['scan'] })
    },
  })

  const filtered = useMemo(
    () =>
      sortStrategies(
        strategies.filter((strategy) => {
          if (!query) return true
          const q = query.toLowerCase()
          return (
            strategy.name.toLowerCase().includes(q) ||
            strategy.symbol.toLowerCase().includes(q) ||
            (strategy.description ?? '').toLowerCase().includes(q) ||
            conditionPreview(strategy.conditions, indicators, strategies).toLowerCase().includes(q)
          )
        }),
      ),
    [indicators, query, strategies],
  )

  const symbolGroups = useMemo(() => groupStrategiesBySymbol(filtered), [filtered])

  function toggleSymbol(symbol: string) {
    setExpandedSymbols((current) => {
      const next = new Set(current)
      if (next.has(symbol)) {
        next.delete(symbol)
      } else {
        next.add(symbol)
      }
      return next
    })
  }

  function startEditing(strategy: SavedStrategy) {
    setEditingId(strategy.id)
    setDescriptionDraft(strategy.description ?? '')
    setDeleteConfirmId(null)
  }

  function saveDescription(strategy: SavedStrategy) {
    updateMutation.mutate({ id: strategy.id, description: descriptionDraft })
  }

  function handleDeleteClick(strategy: SavedStrategy, isDeleting: boolean) {
    if (!isDeleting) {
      setDeleteConfirmId(strategy.id)
      setEditingId(null)
      return
    }
    deleteMutation.mutate(strategy.id)
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-xs text-muted-foreground">
          {strategies.length} saved strateg{strategies.length === 1 ? 'y' : 'ies'}
          {symbolGroups.length > 0 ? ` · ${symbolGroups.length} symbols` : ''}
        </p>
        <div className="flex min-w-0 flex-1 flex-wrap items-center justify-end gap-2">
          <div className="relative min-w-[180px] flex-1 sm:max-w-xs">
            <Search className="absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search strategies…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="h-7 pl-7 text-xs"
            />
          </div>
          <Button onClick={onNewStrategy} size="sm" className="h-7 gap-1 px-2 text-xs">
            New strategy
          </Button>
        </div>
      </div>

      {(error || updateMutation.error || deleteMutation.error) && (
        <Card className="border-destructive/50 bg-destructive/5 p-2">
          <pre className="overflow-x-auto whitespace-pre-wrap text-[11px] text-destructive">
            {String(error ?? updateMutation.error ?? deleteMutation.error)}
          </pre>
        </Card>
      )}

      {isLoading ? (
        <div className="flex items-center justify-center gap-2 py-6 text-xs text-muted-foreground">
          <Loader2 className="size-3.5 animate-spin" />
          Loading strategies…
        </div>
      ) : symbolGroups.length === 0 ? (
        <p className="py-3 text-center text-xs text-muted-foreground">No saved strategies found.</p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {symbolGroups.map(({ symbol, strategies: symbolStrategies }) => {
            const expanded = expandedSymbols.has(symbol)
            return (
              <div
                key={symbol}
                className="overflow-hidden rounded-md border border-border/60"
              >
                <button
                  type="button"
                  className="flex w-full items-center gap-2 bg-muted/30 px-2 py-1.5 text-left text-xs hover:bg-muted/50"
                  onClick={() => toggleSymbol(symbol)}
                  aria-expanded={expanded}
                >
                  <ChevronDown
                    className={cn(
                      'size-3.5 shrink-0 text-muted-foreground transition-transform',
                      !expanded && '-rotate-90',
                    )}
                  />
                  <span className="font-mono font-medium">{symbol}</span>
                  <span className="text-muted-foreground">
                    {symbolStrategies.length} strateg
                    {symbolStrategies.length === 1 ? 'y' : 'ies'}
                  </span>
                </button>
                {expanded ? (
                  <div className="overflow-x-auto border-t border-border/60">
                    <Table className="text-xs">
                      <TableHeader className="bg-muted/20">
                        <TableRow className="hover:bg-transparent">
                          <TableHead className={cn(compactHead, 'min-w-[120px]')}>
                            Strategy
                          </TableHead>
                          <TableHead className={cn(compactHead, 'min-w-[200px]')}>
                            Conditions
                          </TableHead>
                          <TableHead className={cn(compactHead, 'min-w-[180px]')}>
                            Description
                          </TableHead>
                          <TableHead className={cn(compactHead, 'w-28 text-right')}>
                            Actions
                          </TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {symbolStrategies.map((strategy) => (
                          <StrategyRow
                            key={strategy.id}
                            strategy={strategy}
                            strategies={strategies}
                            indicators={indicators}
                            editingId={editingId}
                            descriptionDraft={descriptionDraft}
                            deleteConfirmId={deleteConfirmId}
                            updatePending={updateMutation.isPending}
                            deletePending={deleteMutation.isPending}
                            onDescriptionDraftChange={setDescriptionDraft}
                            onStartEditing={startEditing}
                            onCancelEditing={() => {
                              setEditingId(null)
                              setDescriptionDraft('')
                            }}
                            onSaveDescription={saveDescription}
                            onBacktest={onBacktestStrategy}
                            onDeleteClick={handleDeleteClick}
                          />
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                ) : null}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
