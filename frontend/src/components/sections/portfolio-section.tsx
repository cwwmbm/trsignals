'use client'

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ChevronDown, Loader2, Play, Save, Search, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type {
  DetailedResult,
  PortfolioAllocationModel,
  PortfolioOverlapMode,
  PortfolioProportionalMode,
  PortfolioSimulatePayload,
  RegimePenaltyFields,
  SavedPortfolio,
  SavedStrategy,
  ScanLane,
} from '@/api'
import {
  deletePortfolio,
  getSavedPortfolios,
  getSavedStrategies,
  runPortfolioSimulation,
  savePortfolio,
  updatePortfolio,
} from '@/api'
import { DetailResults } from '@/components/backtest/detail-results'
import { RegimePenaltyInputs } from '@/components/backtest/regime-penalty-inputs'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  DEFAULT_REGIME_FORM,
  regimeFields,
  regimeFormFromSaved,
  regimeSummary,
  type RegimeAllocationForm,
} from '@/lib/regime-allocation'
import { blockedChildren } from '@/lib/portfolio-graph'
import { LANE_LABELS, SCAN_LANES, resolveScanLane } from '@/lib/scan-board'
import { compareSymbols } from '@/lib/symbol-order'
import { cn } from '@/lib/utils'

// Portfolios always hold until every strategy exits. The API still takes the field.
const PORTFOLIO_OVERLAP_MODE: PortfolioOverlapMode = 'hold_until_all_exit'

const compactHead = 'h-7 px-1.5 py-0 text-[11px] font-medium'
const compactCell = 'px-1.5 py-0.5 align-top'

const PROPORTIONAL_MODES: Array<{
  value: PortfolioProportionalMode
  label: string
  formula: string
  hint: string
}> = [
  {
    value: 'linear',
    label: 'Linear',
    formula: 'active / total',
    hint: 'Exposure rises in even steps as strategies agree.',
  },
  {
    value: 'early_conviction',
    label: 'Early conviction',
    formula: '(active / total)^0.5',
    hint: 'A single signal already carries a large position.',
  },
  {
    value: 'confirmation_required',
    label: 'Confirmation required',
    formula: '(active / total)^2',
    hint: 'Stays small until most strategies agree.',
  },
]

const PROPORTIONAL_LABELS: Record<PortfolioProportionalMode, string> = {
  linear: 'Linear',
  early_conviction: 'Early conviction',
  confirmation_required: 'Confirmation required',
}

/** Everything a portfolio stores about how its book is sized. Strategies carry none of it. */
export type AllocationSettingsFields = RegimePenaltyFields &
  Pick<SavedPortfolio, 'allocation_model' | 'proportional_mode' | 'transaction_cost_bps'>

function allocationLabel(portfolio: AllocationSettingsFields) {
  if (portfolio.allocation_model === 'regime_adjusted') return 'Regime-adjusted'
  if (portfolio.allocation_model !== 'proportional') return 'All-in'
  return `Proportional · ${PROPORTIONAL_LABELS[portfolio.proportional_mode ?? 'linear']}`
}

export type PortfolioInitialState = {
  strategyIds: string[]
  portfolioIds?: string[]
  allocation?: AllocationSettingsFields
  proxySymbol?: string
  years?: string
  name?: string
  description?: string
}

function sortStrategies(strategies: SavedStrategy[]) {
  return [...strategies].sort((a, b) => {
    const orderDiff = (a.scan_sort_order ?? 0) - (b.scan_sort_order ?? 0)
    if (orderDiff !== 0) return orderDiff
    const symbolOrder = compareSymbols(a.symbol, b.symbol)
    if (symbolOrder !== 0) return symbolOrder
    return a.name.localeCompare(b.name)
  })
}

function groupStrategiesByLane(strategies: SavedStrategy[]) {
  const lanes: Record<ScanLane, SavedStrategy[]> = {
    active: [],
    testing: [],
    archived: [],
  }
  for (const strategy of strategies) {
    lanes[resolveScanLane(strategy)].push(strategy)
  }
  return SCAN_LANES.map((lane) => ({
    lane,
    strategies: sortStrategies(lanes[lane]),
  }))
}

type AllocationSetters = {
  setAllocationModel: (value: PortfolioAllocationModel) => void
  setProportionalMode: (value: PortfolioProportionalMode) => void
  setCostBps: (value: string) => void
  setRegimeForm: (value: RegimeAllocationForm) => void
}

function applyAllocationSettings(settings: AllocationSettingsFields, setters: AllocationSetters) {
  // Portfolios saved before allocation models existed have none of these fields.
  setters.setAllocationModel(settings.allocation_model ?? 'all_in')
  setters.setProportionalMode(settings.proportional_mode ?? 'linear')
  setters.setCostBps(String(settings.transaction_cost_bps ?? 0))
  setters.setRegimeForm(regimeFormFromSaved(settings))
}

function applyPortfolioSelection(
  portfolio: Pick<SavedPortfolio, 'strategy_ids' | 'portfolio_ids' | 'proxy_symbol'> &
    AllocationSettingsFields,
  setters: AllocationSetters & {
    setSelectedIds: (value: Set<string>) => void
    setSelectedPortfolioIds: (value: Set<string>) => void
    setProxySymbol: (value: string) => void
  },
) {
  setters.setSelectedIds(new Set(portfolio.strategy_ids))
  setters.setSelectedPortfolioIds(new Set(portfolio.portfolio_ids ?? []))
  setters.setProxySymbol(portfolio.proxy_symbol?.trim() ?? '')
  applyAllocationSettings(portfolio, setters)
}

export function PortfolioSection({
  initialState,
}: {
  initialState?: PortfolioInitialState
}) {
  const queryClient = useQueryClient()
  const { data: strategies = [], isLoading, error } = useQuery({
    queryKey: ['strategies'],
    queryFn: getSavedStrategies,
  })
  const { data: savedPortfolios = [], isLoading: portfoliosLoading } = useQuery({
    queryKey: ['portfolios'],
    queryFn: getSavedPortfolios,
  })

  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set())
  const [selectedPortfolioIds, setSelectedPortfolioIds] = useState<Set<string>>(() => new Set())
  const [search, setSearch] = useState('')
  const [expandedLanes, setExpandedLanes] = useState<Set<string>>(
    () => new Set(['active', 'testing']),
  )
  const [proxySymbol, setProxySymbol] = useState('')
  const [allocationModel, setAllocationModel] = useState<PortfolioAllocationModel>('all_in')
  const [proportionalMode, setProportionalMode] = useState<PortfolioProportionalMode>('linear')
  const [costBps, setCostBps] = useState('0')
  const [regimeForm, setRegimeForm] = useState<RegimeAllocationForm>(DEFAULT_REGIME_FORM)
  const [years, setYears] = useState('25')
  const [portfolioName, setPortfolioName] = useState('')
  const [portfolioDescription, setPortfolioDescription] = useState('')
  const [lastSimulateRequest, setLastSimulateRequest] =
    useState<PortfolioSimulatePayload | null>(null)

  const mutation = useMutation({
    mutationFn: runPortfolioSimulation,
  })
  const regimeSource = useMemo(
    () => (lastSimulateRequest ? { path: '/portfolios/simulate', payload: lastSimulateRequest } : null),
    [lastSimulateRequest],
  )

  const saveMutation = useMutation({
    mutationFn: async ({
      payload,
      existingId,
    }: {
      payload: {
        name: string
        description: string
        strategy_ids: string[]
        portfolio_ids?: string[]
        overlap_mode: PortfolioOverlapMode
        proxy_symbol?: string
        allocation_model: PortfolioAllocationModel
        proportional_mode: PortfolioProportionalMode
        transaction_cost_bps: number
      } & RegimePenaltyFields
      existingId?: string
    }) => {
      if (existingId) {
        return updatePortfolio(existingId, payload)
      }
      return savePortfolio(payload)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolios'] })
      queryClient.invalidateQueries({ queryKey: ['scan'] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deletePortfolio,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolios'] })
      queryClient.invalidateQueries({ queryKey: ['scan'] })
    },
  })

  useEffect(() => {
    if (!initialState) return
    setSelectedIds(new Set(initialState.strategyIds))
    setSelectedPortfolioIds(new Set(initialState.portfolioIds ?? []))
    if (initialState.allocation) {
      applyAllocationSettings(initialState.allocation, {
        setAllocationModel,
        setProportionalMode,
        setCostBps,
        setRegimeForm,
      })
    }
    setProxySymbol(initialState.proxySymbol?.trim() ?? '')
    if (initialState.years) setYears(initialState.years)
    if (initialState.name != null) setPortfolioName(initialState.name)
    if (initialState.description != null) setPortfolioDescription(initialState.description)
  }, [initialState])

  const parsedCostBps = Number(costBps)
  const costBpsValid =
    costBps.trim() !== '' &&
    Number.isFinite(parsedCostBps) &&
    parsedCostBps >= 0 &&
    parsedCostBps <= 1000
  const regimeActive = allocationModel === 'regime_adjusted'
  const allocationFields = {
    allocation_model: allocationModel,
    proportional_mode: proportionalMode,
    transaction_cost_bps: costBpsValid ? parsedCostBps : 0,
    ...regimeFields(regimeForm),
  }

  const sortedStrategies = useMemo(() => sortStrategies(strategies), [strategies])

  const filteredStrategies = useMemo(() => {
    const query = search.trim().toLowerCase()
    if (!query) return sortedStrategies
    return sortedStrategies.filter(
      (strategy) =>
        strategy.symbol.toLowerCase().includes(query) ||
        strategy.name.toLowerCase().includes(query) ||
        strategy.description?.toLowerCase().includes(query),
    )
  }, [search, sortedStrategies])

  const laneGroups = useMemo(
    () => groupStrategiesByLane(filteredStrategies),
    [filteredStrategies],
  )

  const searchActive = search.trim().length > 0

  const selectedInOrder = useMemo(
    () => sortedStrategies.filter((strategy) => selectedIds.has(strategy.id)).map((s) => s.id),
    [selectedIds, sortedStrategies],
  )

  function toggleLane(lane: ScanLane) {
    setExpandedLanes((current) => {
      const next = new Set(current)
      if (next.has(lane)) next.delete(lane)
      else next.add(lane)
      return next
    })
  }

  const selectedShort = useMemo(
    () => sortedStrategies.some((strategy) => selectedIds.has(strategy.id) && strategy.direction === 'short'),
    [selectedIds, sortedStrategies],
  )

  // Saving under an existing name overwrites that portfolio, so it is the one being edited.
  // It, and anything that contains it, cannot be added as its child without a cycle.
  const editingId = useMemo(() => {
    const key = portfolioName.trim().toLowerCase()
    if (!key) return undefined
    return [...savedPortfolios]
      .filter((portfolio) => portfolio.name.trim().toLowerCase() === key)
      .sort((a, b) => b.updated_at.localeCompare(a.updated_at))[0]?.id
  }, [portfolioName, savedPortfolios])
  const blockedChildIds = useMemo(
    () => blockedChildren(editingId, savedPortfolios),
    [editingId, savedPortfolios],
  )
  const selectedBlocked = useMemo(
    () => savedPortfolios.filter((p) => selectedPortfolioIds.has(p.id) && blockedChildIds.has(p.id)),
    [savedPortfolios, selectedPortfolioIds, blockedChildIds],
  )

  const canSimulate =
    (selectedInOrder.length + selectedPortfolioIds.size) > 0 &&
    !selectedShort &&
    costBpsValid &&
    !mutation.isPending
  const canSave =
    (selectedInOrder.length + selectedPortfolioIds.size) > 0 &&
    !selectedShort &&
    costBpsValid &&
    portfolioName.trim().length > 0 &&
    selectedBlocked.length === 0 &&
    !saveMutation.isPending

  const toggleStrategy = (strategy: SavedStrategy, checked: boolean) => {
    if (strategy.direction === 'short') return
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (checked) next.add(strategy.id)
      else next.delete(strategy.id)
      return next
    })
  }

  const handleSimulate = () => {
    const parsedYears = Number(years)
    const payload: PortfolioSimulatePayload = {
      strategy_ids: selectedInOrder,
      portfolio_ids: [...selectedPortfolioIds],
      overlap_mode: PORTFOLIO_OVERLAP_MODE,
      ...(proxySymbol.trim() ? { proxy_symbol: proxySymbol.trim().toUpperCase() } : {}),
      years: Number.isFinite(parsedYears) ? parsedYears : 25,
      ...allocationFields,
    }
    setLastSimulateRequest(payload)
    mutation.mutate(payload)
  }

  const handleSave = () => {
    const payload = {
      name: portfolioName.trim(),
      description: portfolioDescription.trim(),
      strategy_ids: selectedInOrder,
      portfolio_ids: [...selectedPortfolioIds],
      overlap_mode: PORTFOLIO_OVERLAP_MODE,
      ...(proxySymbol.trim() ? { proxy_symbol: proxySymbol.trim().toUpperCase() } : {}),
      ...allocationFields,
    }
    const nameKey = payload.name.toLowerCase()
    const existing = savedPortfolios
      .filter((portfolio) => portfolio.name.trim().toLowerCase() === nameKey)
      .sort((a, b) => b.updated_at.localeCompare(a.updated_at))[0]

    if (existing) {
      const override = window.confirm(
        `A portfolio named "${existing.name}" already exists. Override it?`,
      )
      if (!override) return
      saveMutation.mutate({ payload, existingId: existing.id })
      return
    }

    saveMutation.mutate(
      { payload },
      {
        onSuccess: () => {
          setPortfolioName('')
          setPortfolioDescription('')
        },
      },
    )
  }

  const handleLoadPortfolio = (portfolio: SavedPortfolio) => {
    applyPortfolioSelection(portfolio, {
      setSelectedIds,
      setSelectedPortfolioIds,
      setProxySymbol,
      setAllocationModel,
      setProportionalMode,
      setCostBps,
      setRegimeForm,
    })
    setPortfolioName(portfolio.name)
    setPortfolioDescription(portfolio.description ?? '')
  }

  const result = mutation.data

  return (
    <div className="space-y-4">
      <Card className="p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold">Strategies</h2>
            <p className="text-xs text-muted-foreground">
              Select strategies and saved portfolios. Each child keeps its own allocation model.
            </p>
          </div>
          <div className="relative w-full max-w-xs">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search symbol, name, description…"
              className="h-8 pl-8 text-sm"
            />
          </div>
        </div>

        {isLoading ? (
          <div className="mt-4 flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Loading strategies…
          </div>
        ) : error ? (
          <p className="mt-4 text-sm text-destructive">{String(error)}</p>
        ) : filteredStrategies.length === 0 ? (
          <p className="mt-4 text-sm text-muted-foreground">No strategies match your search.</p>
        ) : (
          <div className="mt-3 flex flex-col gap-2">
            {laneGroups
              .filter(({ strategies: laneStrategies }) => !searchActive || laneStrategies.length > 0)
              .map(({ lane, strategies: laneStrategies }) => {
              const expanded = searchActive || expandedLanes.has(lane)
              const selectedCount = laneStrategies.filter((strategy) =>
                selectedIds.has(strategy.id),
              ).length
              return (
                <div
                  key={lane}
                  className="overflow-hidden rounded-md border border-border/60"
                >
                  <button
                    type="button"
                    className="flex w-full items-center gap-2 bg-muted/20 px-3 py-2 text-left text-xs hover:bg-muted/30"
                    onClick={() => {
                      if (searchActive) return
                      toggleLane(lane)
                    }}
                    aria-expanded={expanded}
                  >
                    <ChevronDown
                      className={cn(
                        'size-3.5 shrink-0 transition-transform',
                        !expanded && '-rotate-90',
                      )}
                    />
                    <span className="font-medium">{LANE_LABELS[lane]}</span>
                    <span className="text-muted-foreground">
                      {laneStrategies.length} strateg
                      {laneStrategies.length === 1 ? 'y' : 'ies'}
                      {selectedCount > 0 ? ` · ${selectedCount} selected` : ''}
                    </span>
                  </button>
                  {expanded ? (
                    <div className="overflow-x-auto border-t border-border/60">
                      <Table>
                        <TableHeader>
                          <TableRow className="hover:bg-transparent">
                            <TableHead className={cn(compactHead, 'w-10')} />
                            <TableHead className={cn(compactHead, 'w-20')}>Symbol</TableHead>
                            <TableHead className={cn(compactHead, 'min-w-[10rem]')}>Name</TableHead>
                            <TableHead className={compactHead}>Description</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {laneStrategies.length === 0 ? (
                            <TableRow>
                              <TableCell
                                colSpan={4}
                                className="py-4 text-center text-[11px] text-muted-foreground"
                              >
                                No strategies in this category
                              </TableCell>
                            </TableRow>
                          ) : (
                            laneStrategies.map((strategy) => {
                              const isShort = strategy.direction === 'short'
                              const checked = selectedIds.has(strategy.id)
                              return (
                                <TableRow
                                  key={strategy.id}
                                  className={isShort ? 'opacity-60' : undefined}
                                >
                                  <TableCell className={compactCell}>
                                    <Checkbox
                                      checked={checked}
                                      disabled={isShort}
                                      onCheckedChange={(value) =>
                                        toggleStrategy(strategy, value === true)
                                      }
                                      aria-label={`Select ${strategy.name}`}
                                    />
                                  </TableCell>
                                  <TableCell className={cn(compactCell, 'font-mono text-xs')}>
                                    {strategy.symbol}
                                  </TableCell>
                                  <TableCell className={cn(compactCell, 'text-xs font-medium')}>
                                    {strategy.name}
                                    {isShort ? (
                                      <span className="ml-1.5 text-[10px] font-normal text-muted-foreground">
                                        (short — excluded)
                                      </span>
                                    ) : null}
                                  </TableCell>
                                  <TableCell
                                    className={cn(
                                      compactCell,
                                      'max-w-md text-xs text-muted-foreground',
                                    )}
                                  >
                                    {(strategy.description ?? '').trim() || '—'}
                                  </TableCell>
                                </TableRow>
                              )
                            })
                          )}
                        </TableBody>
                      </Table>
                    </div>
                  ) : null}
                </div>
              )
            })}
          </div>
        )}
      </Card>

      <Card className="p-4">
        <h2 className="text-sm font-semibold">Child portfolios</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Each child counts as one member. Its uninvested budget stays in cash, including when it rejects a BUY.
          Child settings are read from the saved portfolio on each run.
        </p>
        <div className="mt-3 space-y-2">
          {savedPortfolios.map((portfolio) => {
            const blocked = blockedChildIds.get(portfolio.id)
            return (
              <label
                key={portfolio.id}
                className={cn(
                  'flex items-center gap-3 text-sm',
                  blocked ? 'cursor-not-allowed opacity-60' : 'cursor-pointer',
                )}
              >
                <Checkbox
                  disabled={blocked !== undefined}
                  checked={selectedPortfolioIds.has(portfolio.id)}
                  onCheckedChange={(checked) => setSelectedPortfolioIds((current) => {
                    const next = new Set(current)
                    if (checked) next.add(portfolio.id)
                    else next.delete(portfolio.id)
                    return next
                  })}
                />
                <span>{portfolio.name}</span>
                <span className="text-xs text-muted-foreground">{allocationLabel(portfolio)}</span>
                {blocked ? (
                  <span className="text-xs text-muted-foreground">
                    {blocked === 'itself'
                      ? 'this portfolio'
                      : 'contains this portfolio, so it cannot be a child of it'}
                  </span>
                ) : null}
              </label>
            )
          })}
          {selectedBlocked.length > 0 ? (
            <p className="text-xs text-destructive">
              {selectedBlocked.map((p) => p.name).join(', ')} cannot be a child of the portfolio named
              above (it would contain itself). Untick it or change the name to save.
            </p>
          ) : null}
          {savedPortfolios.length === 0 ? <p className="text-xs text-muted-foreground">No saved portfolios yet.</p> : null}
        </div>
      </Card>

      <Card className="p-4">
        <h2 className="text-sm font-semibold">Settings</h2>
        {selectedPortfolioIds.size > 0 ? <p className="mt-1 text-xs text-muted-foreground">These settings apply to the parent. Children keep their own allocation, proxies, and trading costs.</p> : null}
        <div className="mt-3 space-y-4">
          <div>
            <Label className="text-xs text-muted-foreground">Allocation model</Label>
            <RadioGroup
              value={allocationModel}
              onValueChange={(value) => value && setAllocationModel(value as PortfolioAllocationModel)}
              className="mt-2 gap-3"
            >
              <div className="flex items-start gap-2">
                <RadioGroupItem value="all_in" id="alloc-all-in" className="mt-0.5" />
                <div>
                  <Label htmlFor="alloc-all-in" className="text-sm font-normal">
                    All-in
                  </Label>
                  <p className="text-[11px] text-muted-foreground">
                    Allocate the full budget equally across active members. Child portfolios
                    can keep part or all of their budget in cash.
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <RadioGroupItem value="proportional" id="alloc-proportional" className="mt-0.5" />
                <div>
                  <Label htmlFor="alloc-proportional" className="text-sm font-normal">
                    Proportional
                  </Label>
                  <p className="text-[11px] text-muted-foreground">
                    Target budget follows how many members are active out of all eligible
                    members. The position is rebalanced whenever that changes.
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <RadioGroupItem value="regime_adjusted" id="alloc-regime" className="mt-0.5" />
                <div>
                  <Label htmlFor="alloc-regime" className="text-sm font-normal">
                    Regime-adjusted
                  </Label>
                  <p className="text-[11px] text-muted-foreground">
                    Starts at 100% and takes confidence off for every current market regime where
                    the active strategies, or the portfolio, have scored poorly. Re-evaluated
                    whenever the set of active strategies changes.
                  </p>
                </div>
              </div>
            </RadioGroup>

            {regimeActive ? (
              <RegimePenaltyInputs form={regimeForm} onChange={setRegimeForm} />
            ) : null}

            {allocationModel === 'proportional' ? (
              <RadioGroup
                value={proportionalMode}
                onValueChange={(value) =>
                  value && setProportionalMode(value as PortfolioProportionalMode)
                }
                className="mt-3 ml-6 gap-2"
              >
                {PROPORTIONAL_MODES.map((mode) => (
                  <div key={mode.value} className="flex items-start gap-2">
                    <RadioGroupItem
                      value={mode.value}
                      id={`prop-${mode.value}`}
                      className="mt-0.5"
                    />
                    <div>
                      <Label htmlFor={`prop-${mode.value}`} className="text-sm font-normal">
                        {mode.label}{' '}
                        <span className="font-mono text-[11px] text-muted-foreground">
                          {mode.formula}
                        </span>
                      </Label>
                      <p className="text-[11px] text-muted-foreground">{mode.hint}</p>
                    </div>
                  </div>
                ))}
              </RadioGroup>
            ) : null}
          </div>

          <div className="grid gap-3 sm:grid-cols-3 sm:max-w-2xl">
            <div className="flex flex-col gap-1">
              <Label htmlFor="portfolio-proxy" className="text-xs text-muted-foreground">
                Proxy symbol
              </Label>
              <Input
                id="portfolio-proxy"
                value={proxySymbol}
                onChange={(e) => setProxySymbol(e.target.value.toUpperCase())}
                placeholder="Optional — direct strategies only"
                className="h-8 font-mono text-sm"
              />
            </div>
            <div className="flex flex-col gap-1">
              <Label htmlFor="portfolio-years" className="text-xs text-muted-foreground">
                Years
              </Label>
              <Input
                id="portfolio-years"
                inputMode="numeric"
                value={years}
                onChange={(e) => setYears(e.target.value)}
                className="h-8 font-mono text-sm"
              />
            </div>
            <div className="flex flex-col gap-1">
              <Label htmlFor="portfolio-cost" className="text-xs text-muted-foreground">
                Trading cost (bps, direct strategies)
              </Label>
              <Input
                id="portfolio-cost"
                inputMode="decimal"
                value={costBps}
                onChange={(e) => setCostBps(e.target.value)}
                aria-invalid={!costBpsValid}
                className={cn('h-8 font-mono text-sm', !costBpsValid && 'border-destructive')}
              />
            </div>
          </div>
        </div>

        <div className="mt-4 space-y-3 border-t border-border/60 pt-4">
          <h3 className="text-xs font-semibold">Save portfolio</h3>
          <div className="grid gap-3 sm:grid-cols-2 sm:max-w-2xl">
            <div className="flex flex-col gap-1">
              <Label htmlFor="portfolio-name" className="text-xs text-muted-foreground">
                Name
              </Label>
              <Input
                id="portfolio-name"
                value={portfolioName}
                onChange={(e) => setPortfolioName(e.target.value)}
                placeholder="Required"
                className="h-8 text-sm"
              />
            </div>
            <div className="flex flex-col gap-1 sm:col-span-2">
              <Label htmlFor="portfolio-description" className="text-xs text-muted-foreground">
                Description
              </Label>
              <Input
                id="portfolio-description"
                value={portfolioDescription}
                onChange={(e) => setPortfolioDescription(e.target.value)}
                placeholder="Optional"
                className="h-8 text-sm"
              />
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Button disabled={!canSave} onClick={handleSave} variant="outline" className="gap-2">
              {saveMutation.isPending ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Save className="size-4" />
              )}
              Save portfolio
            </Button>
            {saveMutation.error ? (
              <span className="text-xs text-destructive">{String(saveMutation.error)}</span>
            ) : null}
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <Button disabled={!canSimulate} onClick={handleSimulate} className="gap-2">
            {mutation.isPending ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Play className="size-4" />
            )}
            Simulate Performance
          </Button>
          <span className="text-xs text-muted-foreground">
            {selectedInOrder.length + selectedPortfolioIds.size === 0
              ? 'Select at least one strategy or portfolio'
              : selectedShort
                ? 'Short strategies cannot be included'
                : `${selectedInOrder.length} strategies · ${selectedPortfolioIds.size} portfolios selected`}
          </span>
        </div>

        {mutation.error ? (
          <Card className="mt-4 border-destructive/50 bg-destructive/5 p-3">
            <pre className="overflow-x-auto whitespace-pre-wrap text-xs text-destructive">
              {String(mutation.error)}
            </pre>
          </Card>
        ) : null}
      </Card>

      <Card className="p-4">
        <h2 className="text-sm font-semibold">Saved portfolios</h2>
        {deleteMutation.error ? (
          <p className="mt-1 text-xs text-destructive" role="alert">
            Could not delete: {deleteMutation.error.message}
          </p>
        ) : null}
        <p className="mt-1 text-xs text-muted-foreground">
          Load a saved combination into the form above. Saved portfolios also appear on Scan.
        </p>
        {portfoliosLoading ? (
          <div className="mt-4 flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Loading portfolios…
          </div>
        ) : savedPortfolios.length === 0 ? (
          <p className="mt-4 text-sm text-muted-foreground">No saved portfolios yet.</p>
        ) : (
          <div className="mt-3 overflow-x-auto rounded-md border border-border/60">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className={compactHead}>Name</TableHead>
                  <TableHead className={compactHead}>Members</TableHead>
                  <TableHead className={compactHead}>Allocation</TableHead>
                  <TableHead className={compactHead}>Proxy</TableHead>
                  <TableHead className={cn(compactHead, 'w-28 text-right')}>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {savedPortfolios.map((portfolio) => (
                  <TableRow key={portfolio.id}>
                    <TableCell className={cn(compactCell, 'text-xs font-medium')}>
                      {portfolio.name}
                      {portfolio.description?.trim() ? (
                        <p className="mt-0.5 text-[10px] font-normal text-muted-foreground">
                          {portfolio.description}
                        </p>
                      ) : null}
                    </TableCell>
                    <TableCell className={cn(compactCell, 'text-xs text-muted-foreground')}>
                      {portfolio.strategy_ids.length} strategies · {portfolio.portfolio_ids?.length ?? 0} portfolios
                    </TableCell>
                    <TableCell className={cn(compactCell, 'text-xs text-muted-foreground')}>
                      {allocationLabel(portfolio)}
                      {portfolio.allocation_model === 'regime_adjusted' ? (
                        <p className="mt-0.5 text-[10px]">{regimeSummary(portfolio)}</p>
                      ) : null}
                    </TableCell>
                    <TableCell className={cn(compactCell, 'font-mono text-xs')}>
                      {portfolio.proxy_symbol ?? '—'}
                    </TableCell>
                    <TableCell className={compactCell}>
                      <div className="flex items-center justify-end gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 px-2 text-[11px]"
                          onClick={() => handleLoadPortfolio(portfolio)}
                        >
                          Load
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 px-2 text-[11px] text-destructive hover:text-destructive"
                          disabled={deleteMutation.isPending}
                          onClick={() => deleteMutation.mutate(portfolio.id)}
                        >
                          <Trash2 className="size-3.5" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </Card>

      {result ? (
        <DetailResults
          result={result}
          shapleyRequest={lastSimulateRequest}
          regimeSource={regimeSource}
        />
      ) : null}
    </div>
  )
}
