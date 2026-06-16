'use client'

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { ArrowUpRight, Check, Loader2, Pencil, Search, Trash2, X } from 'lucide-react'
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

const STRATEGY_INDICATOR_PREFIX = 'strategy:'

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

export function StrategiesSection({
  onNewStrategy,
  onBacktestStrategy,
}: {
  onNewStrategy?: () => void
  onBacktestStrategy?: (strategy: SavedStrategy) => void
}) {
  const queryClient = useQueryClient()
  const [query, setQuery] = useState('')
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
    [indicators, query, strategies],
  )

  function startEditing(strategy: SavedStrategy) {
    setEditingId(strategy.id)
    setDescriptionDraft(strategy.description ?? '')
    setDeleteConfirmId(null)
  }

  function saveDescription(strategy: SavedStrategy) {
    updateMutation.mutate({ id: strategy.id, description: descriptionDraft })
  }

  return (
    <div className="flex flex-col gap-6">
      <Card className="border-border/60 p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold">All strategies</h2>
            <p className="text-sm text-muted-foreground">
              {strategies.length} saved strateg{strategies.length === 1 ? 'y' : 'ies'}
            </p>
          </div>
          <Button onClick={onNewStrategy} className="gap-2">
            New strategy
          </Button>
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <div className="relative flex-1 min-w-[220px]">
            <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search strategies…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="pl-9"
            />
          </div>
        </div>
      </Card>

      {(error || updateMutation.error || deleteMutation.error) && (
        <Card className="border-destructive/50 bg-destructive/5 p-3">
          <pre className="overflow-x-auto whitespace-pre-wrap text-xs text-destructive">
            {String(error ?? updateMutation.error ?? deleteMutation.error)}
          </pre>
        </Card>
      )}

      <Card className="border-border/60 p-3">
        {isLoading ? (
          <div className="flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Loading strategies…
          </div>
        ) : filtered.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No saved strategies found.
          </p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[14rem]">Strategy</TableHead>
                <TableHead>Conditions</TableHead>
                <TableHead className="w-[24rem]">Description</TableHead>
                <TableHead className="w-[12rem] text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((strategy) => {
                const isEditing = editingId === strategy.id
                const isDeleting = deleteConfirmId === strategy.id
                return (
                  <TableRow key={strategy.id}>
                    <TableCell className="align-top whitespace-normal">
                      <div className="font-medium">{strategy.name}</div>
                      <div className="mt-1 font-mono text-xs text-muted-foreground">
                        {strategy.symbol}
                      </div>
                    </TableCell>
                    <TableCell className="align-top whitespace-normal">
                      <div className="space-y-1 text-xs">
                        <div>
                          <span className="font-medium text-muted-foreground">Entry: </span>
                          <span className="font-mono">
                            {conditionPreview(strategy.conditions, indicators, strategies)}
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-muted-foreground">Exit: </span>
                          <span className="font-mono">
                            {exitPreview(strategy, indicators, strategies)}
                          </span>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell className="align-top whitespace-normal">
                      {isEditing ? (
                        <div className="flex flex-col gap-2">
                          <Textarea
                            value={descriptionDraft}
                            onChange={(event) => setDescriptionDraft(event.target.value)}
                            rows={3}
                            className="min-h-0 resize-none text-xs"
                          />
                          <div className="flex items-center gap-1">
                            <Button
                              size="sm"
                              className="h-7 gap-1 px-2 text-xs"
                              onClick={() => saveDescription(strategy)}
                              disabled={updateMutation.isPending}
                            >
                              {updateMutation.isPending ? (
                                <Loader2 className="size-3 animate-spin" />
                              ) : (
                                <Check className="size-3" />
                              )}
                              Save
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 gap-1 px-2 text-xs"
                              onClick={() => {
                                setEditingId(null)
                                setDescriptionDraft('')
                              }}
                            >
                              <X className="size-3" />
                              Cancel
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div className="group/description flex items-start gap-1.5">
                          <p className="text-xs leading-relaxed text-muted-foreground">
                            {strategy.description || 'No description.'}
                          </p>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="size-6 shrink-0 text-muted-foreground"
                            onClick={() => startEditing(strategy)}
                            aria-label={`Edit ${strategy.name} description`}
                          >
                            <Pencil className="size-3.5" />
                          </Button>
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="align-top">
                      <div className="flex justify-end gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 gap-1 px-2 text-xs"
                          onClick={() => onBacktestStrategy?.(strategy)}
                        >
                          Backtest
                          <ArrowUpRight className="size-3.5" />
                        </Button>
                        <Button
                          variant={isDeleting ? 'destructive' : 'ghost'}
                          size="sm"
                          className="h-7 gap-1 px-2 text-xs"
                          disabled={deleteMutation.isPending}
                          onClick={() => {
                            if (!isDeleting) {
                              setDeleteConfirmId(strategy.id)
                              setEditingId(null)
                              return
                            }
                            deleteMutation.mutate(strategy.id)
                          }}
                        >
                          {deleteMutation.isPending && isDeleting ? (
                            <Loader2 className="size-3 animate-spin" />
                          ) : (
                            <Trash2 className="size-3.5" />
                          )}
                          {isDeleting ? 'Confirm' : 'Delete'}
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        )}
      </Card>
    </div>
  )
}
