'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useDroppable,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core'
import {
  SortableContext,
  arrayMove,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import {
  ArrowUpRight,
  Check,
  ChevronDown,
  GripVertical,
  Loader2,
  RefreshCw,
  Search,
  X,
} from 'lucide-react'
import {
  getSavedPortfolios,
  getSavedStrategies,
  getScan,
  updatePortfolio,
  updateStrategy,
  type SavedPortfolio,
  type SavedStrategy,
  type ScanLane,
  type ScanRow,
  type UpdatePortfolioPayload,
  type UpdateStrategyPayload,
} from '@/api'
import type { PortfolioInitialState } from '@/components/sections/portfolio-section'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { cn } from '@/lib/utils'
import { compareSymbols, sortSymbols } from '@/lib/symbol-order'

const SIGNAL_ORDER = [
  'buy_signal1',
  'buy_signal2',
  'buy_signal3',
  'buy_signal4',
  'buy_signal5',
  'buy_signal6',
  'buy_signal7',
  'buy_signal8',
  'buy_signal9',
  'buy_signal10',
  'buy_signal11',
  'buy_signal12',
  'buy_signal13',
  'buy_signal14',
  'buy_signal15',
  'buy_signal16',
  'buy_signal17',
  'buy_signal18',
  'buy_signal19',
  'buy_signal20',
  'buy_signal21',
  'buy_signal24',
  'og_buy_signal',
  'og_new_buy_signal',
] as const

const SCAN_LANES: ScanLane[] = ['active', 'testing', 'archived']

const LANE_LABELS: Record<ScanLane, string> = {
  active: 'Active',
  testing: 'Testing',
  archived: 'Archived',
}

const compactHead = 'h-7 px-1.5 py-0 text-[11px] font-medium'
const compactCell = 'px-1.5 py-0.5'

type LaneRows = Record<ScanLane, ScanRow[]>

function laneContainerId(lane: ScanLane) {
  return `lane-${lane}`
}

function parseLaneContainerId(id: string): ScanLane | null {
  if (!id.startsWith('lane-')) return null
  const lane = id.slice(5) as ScanLane
  return SCAN_LANES.includes(lane) ? lane : null
}

function sortLegacyRows(rows: ScanRow[]): ScanRow[] {
  const signalRank = new Map(SIGNAL_ORDER.map((signal, index) => [signal, index]))
  return [...rows].sort((a, b) => {
    const symbolDiff = compareSymbols(a.symbol, b.symbol)
    if (symbolDiff !== 0) return symbolDiff
    return (
      (signalRank.get(a.signal as (typeof SIGNAL_ORDER)[number]) ?? SIGNAL_ORDER.length) -
      (signalRank.get(b.signal as (typeof SIGNAL_ORDER)[number]) ?? SIGNAL_ORDER.length)
    )
  })
}

function resolveScanLane(strategy: SavedStrategy | undefined): ScanLane {
  return strategy?.scan_lane ?? 'testing'
}

function resolvePortfolioScanLane(portfolio: SavedPortfolio | undefined): ScanLane {
  return portfolio?.scan_lane ?? 'testing'
}

function rowSortOrder(
  row: ScanRow,
  strategyById: Map<string, SavedStrategy>,
  portfolioById: Map<string, SavedPortfolio>,
): number {
  if (row.source === 'portfolio' && row.portfolio_id) {
    return portfolioById.get(row.portfolio_id)?.scan_sort_order ?? 0
  }
  if (row.strategy_id) {
    return strategyById.get(row.strategy_id)?.scan_sort_order ?? 0
  }
  return 0
}

function sortLaneRows(
  rows: ScanRow[],
  strategyById: Map<string, SavedStrategy>,
  portfolioById: Map<string, SavedPortfolio>,
): ScanRow[] {
  return [...rows].sort((a, b) => {
    const orderDiff = rowSortOrder(a, strategyById, portfolioById) - rowSortOrder(b, strategyById, portfolioById)
    if (orderDiff !== 0) return orderDiff
    const symbolDiff = compareSymbols(a.symbol, b.symbol)
    if (symbolDiff !== 0) return symbolDiff
    return a.signal.localeCompare(b.signal)
  })
}

function groupScanRows(
  rows: ScanRow[],
  strategyById: Map<string, SavedStrategy>,
  portfolioById: Map<string, SavedPortfolio>,
): { lanes: LaneRows; legacy: ScanRow[] } {
  const lanes: LaneRows = {
    active: [],
    testing: [],
    archived: [],
  }
  const legacy: ScanRow[] = []

  for (const row of rows) {
    if (row.source === 'legacy') {
      legacy.push(row)
      continue
    }
    if (row.source === 'portfolio') {
      const portfolio = row.portfolio_id ? portfolioById.get(row.portfolio_id) : undefined
      lanes[resolvePortfolioScanLane(portfolio)].push(row)
      continue
    }
    const strategy = row.strategy_id ? strategyById.get(row.strategy_id) : undefined
    lanes[resolveScanLane(strategy)].push(row)
  }

  for (const lane of SCAN_LANES) {
    lanes[lane] = sortLaneRows(lanes[lane], strategyById, portfolioById)
  }

  return { lanes, legacy: sortLegacyRows(legacy) }
}

function findLaneForRow(laneRows: LaneRows, rowId: string): ScanLane | null {
  for (const lane of SCAN_LANES) {
    if (laneRows[lane].some((row) => row.id === rowId)) return lane
  }
  return null
}

function countActiveSignals(rows: ScanRow[]) {
  return rows.filter((row) => row.buy_signal || row.hold_long).length
}

function BoolCell({ value }: { value: boolean }) {
  return value ? (
    <Check className="mx-auto size-3 text-[var(--gain)]" aria-label="True" />
  ) : (
    <X className="mx-auto size-3 text-muted-foreground/50" aria-label="False" />
  )
}

function ScanTableHeader({ draggable }: { draggable?: boolean }) {
  return (
    <TableHeader className="bg-muted/30">
      <TableRow className="hover:bg-transparent">
        {draggable ? <TableHead className={cn(compactHead, 'w-8')} /> : null}
        <TableHead className={compactHead}>Symbol</TableHead>
        <TableHead className={cn(compactHead, 'min-w-[120px]')}>Signal</TableHead>
        <TableHead className={cn(compactHead, 'w-10 text-center')}>Buy</TableHead>
        <TableHead className={cn(compactHead, 'w-10 text-center')}>Hold</TableHead>
        <TableHead className={cn(compactHead, 'w-10 text-center')}>Sell</TableHead>
        <TableHead className={cn(compactHead, 'w-10 text-right')}>Days</TableHead>
        <TableHead className={cn(compactHead, 'w-10 text-right')}>Prf</TableHead>
        <TableHead className={cn(compactHead, 'w-14 text-right')}>PnL</TableHead>
        <TableHead className={cn(compactHead, 'w-14 text-right')}>Kelly</TableHead>
        <TableHead className={cn(compactHead, 'min-w-[200px]')}>Description</TableHead>
        <TableHead className={cn(compactHead, 'w-40 text-right')}>Actions</TableHead>
      </TableRow>
    </TableHeader>
  )
}

function ScanTableRowContent({
  row: r,
  strategy,
  portfolio,
  onBacktestStrategy,
  onOpenPortfolio,
  onLaneChange,
  onPortfolioLaneChange,
  lanePending,
  draggable = false,
  dragHandleProps,
}: {
  row: ScanRow
  strategy?: SavedStrategy
  portfolio?: SavedPortfolio
  onBacktestStrategy?: (strategy: SavedStrategy) => void
  onOpenPortfolio?: (state: PortfolioInitialState) => void
  onLaneChange?: (strategy: SavedStrategy, lane: ScanLane) => void
  onPortfolioLaneChange?: (portfolio: SavedPortfolio, lane: ScanLane) => void
  lanePending?: boolean
  draggable?: boolean
  dragHandleProps?: {
    setActivatorNodeRef: (element: HTMLElement | null) => void
    listeners: ReturnType<typeof useSortable>['listeners']
    attributes: ReturnType<typeof useSortable>['attributes']
  }
}) {
  const currentLane = portfolio ? resolvePortfolioScanLane(portfolio) : resolveScanLane(strategy)

  return (
    <>
      {draggable ? (
        <TableCell className={cn(compactCell, 'w-8')}>
          <button
            type="button"
            ref={dragHandleProps?.setActivatorNodeRef}
            className="flex size-6 items-center justify-center rounded text-muted-foreground hover:bg-muted"
            aria-label="Drag to reorder"
            {...dragHandleProps?.attributes}
            {...dragHandleProps?.listeners}
          >
            <GripVertical className="size-3.5" />
          </button>
        </TableCell>
      ) : null}
      <TableCell className={cn(compactCell, 'font-mono font-medium')}>{r.symbol}</TableCell>
      <TableCell className={cn(compactCell, 'font-mono text-[11px]')}>{r.signal}</TableCell>
      <TableCell className={compactCell}>
        <BoolCell value={r.buy_signal} />
      </TableCell>
      <TableCell className={compactCell}>
        <BoolCell value={r.hold_long} />
      </TableCell>
      <TableCell className={compactCell}>
        <BoolCell value={r.sell_signal} />
      </TableCell>
      <TableCell className={cn(compactCell, 'text-right font-mono tabular-nums')}>{r.days}</TableCell>
      <TableCell className={cn(compactCell, 'text-right font-mono tabular-nums')}>{r.profit}</TableCell>
      <TableCell
        className={cn(
          compactCell,
          'text-right font-mono tabular-nums',
          r.trade_pnl > 0 ? 'text-[var(--gain)]' : 'text-muted-foreground',
        )}
      >
        {r.trade_pnl.toFixed(1)}%
      </TableCell>
      <TableCell className={cn(compactCell, 'text-right font-mono tabular-nums text-muted-foreground')}>
        {r.kelly === null ? '—' : `${r.kelly.toFixed(1)}%`}
      </TableCell>
      <TableCell
        className={cn(compactCell, 'max-w-[360px] truncate text-[11px] text-muted-foreground')}
        title={r.description}
      >
        {r.description}
      </TableCell>
      <TableCell className={compactCell}>
        {strategy || portfolio ? (
          <div className="flex items-center justify-end gap-1">
            <Select
              value={currentLane}
              onValueChange={(value) => {
                if (!value || value === currentLane) return
                if (portfolio) onPortfolioLaneChange?.(portfolio, value as ScanLane)
                else if (strategy) onLaneChange?.(strategy, value as ScanLane)
              }}
              disabled={lanePending}
            >
              <SelectTrigger className="h-6 w-[92px] text-[10px]">
                <SelectValue>
                  {(value: string) => LANE_LABELS[value as ScanLane] ?? value}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {SCAN_LANES.map((lane) => (
                  <SelectItem key={lane} value={lane} className="text-xs">
                    {LANE_LABELS[lane]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {strategy ? (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 gap-1 px-1.5 text-[11px]"
                onClick={() => onBacktestStrategy?.(strategy)}
              >
                Backtest
                <ArrowUpRight className="size-3" />
              </Button>
            ) : null}
            {portfolio ? (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 gap-1 px-1.5 text-[11px]"
                onClick={() =>
                  onOpenPortfolio?.({
                    strategyIds: portfolio.strategy_ids,
                    overlapMode: portfolio.overlap_mode,
                    ...(portfolio.proxy_symbol ? { proxySymbol: portfolio.proxy_symbol } : {}),
                  })
                }
              >
                Portfolio
                <ArrowUpRight className="size-3" />
              </Button>
            ) : null}
          </div>
        ) : null}
      </TableCell>
    </>
  )
}

function SortableScanTableRow({
  row,
  strategy,
  portfolio,
  onBacktestStrategy,
  onOpenPortfolio,
  onLaneChange,
  onPortfolioLaneChange,
  lanePending,
}: {
  row: ScanRow
  strategy?: SavedStrategy
  portfolio?: SavedPortfolio
  onBacktestStrategy?: (strategy: SavedStrategy) => void
  onOpenPortfolio?: (state: PortfolioInitialState) => void
  onLaneChange?: (strategy: SavedStrategy, lane: ScanLane) => void
  onPortfolioLaneChange?: (portfolio: SavedPortfolio, lane: ScanLane) => void
  lanePending?: boolean
}) {
  const { attributes, listeners, setNodeRef, setActivatorNodeRef, transform, transition, isDragging } =
    useSortable({ id: row.id })

  return (
    <TableRow
      ref={setNodeRef}
      style={{
        transform: CSS.Transform.toString(transform),
        transition,
      }}
      className={cn(
        'hover:bg-muted/30',
        (row.buy_signal || row.hold_long) && 'bg-[var(--gain)]/8',
        isDragging && 'opacity-60',
      )}
    >
      <ScanTableRowContent
        row={row}
        strategy={strategy}
        portfolio={portfolio}
        onBacktestStrategy={onBacktestStrategy}
        onOpenPortfolio={onOpenPortfolio}
        onLaneChange={onLaneChange}
        onPortfolioLaneChange={onPortfolioLaneChange}
        lanePending={lanePending}
        draggable
        dragHandleProps={{ setActivatorNodeRef, listeners, attributes }}
      />
    </TableRow>
  )
}

function StaticScanTableRow({
  row,
  strategy,
  onBacktestStrategy,
}: {
  row: ScanRow
  strategy?: SavedStrategy
  onBacktestStrategy?: (strategy: SavedStrategy) => void
}) {
  return (
    <TableRow
      className={cn('hover:bg-muted/30', (row.buy_signal || row.hold_long) && 'bg-[var(--gain)]/8')}
    >
      <ScanTableRowContent row={row} strategy={strategy} onBacktestStrategy={onBacktestStrategy} />
    </TableRow>
  )
}

function ScanLaneSection({
  lane,
  rows,
  expanded,
  onToggle,
  strategyById,
  portfolioById,
  onBacktestStrategy,
  onOpenPortfolio,
  onLaneChange,
  onPortfolioLaneChange,
  lanePending,
}: {
  lane: ScanLane
  rows: ScanRow[]
  expanded: boolean
  onToggle: () => void
  strategyById: Map<string, SavedStrategy>
  portfolioById: Map<string, SavedPortfolio>
  onBacktestStrategy?: (strategy: SavedStrategy) => void
  onOpenPortfolio?: (state: PortfolioInitialState) => void
  onLaneChange?: (strategy: SavedStrategy, lane: ScanLane) => void
  onPortfolioLaneChange?: (portfolio: SavedPortfolio, lane: ScanLane) => void
  lanePending?: boolean
}) {
  const { setNodeRef, isOver } = useDroppable({ id: laneContainerId(lane) })
  const activeCount = countActiveSignals(rows)

  return (
    <div className="overflow-hidden rounded-md border border-border/60">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center gap-2 bg-muted/20 px-3 py-2 text-left text-xs hover:bg-muted/30"
      >
        <ChevronDown
          className={cn('size-3.5 shrink-0 transition-transform', !expanded && '-rotate-90')}
        />
        <span className="font-medium">{LANE_LABELS[lane]}</span>
        <span className="text-muted-foreground">
          {rows.length} rows · {activeCount} active
        </span>
      </button>
      {expanded ? (
        <div
          ref={setNodeRef}
          className={cn('overflow-x-auto', isOver && 'ring-1 ring-inset ring-primary/40')}
        >
          <Table className="text-xs">
            <ScanTableHeader draggable />
            <TableBody>
              <SortableContext items={rows.map((row) => row.id)} strategy={verticalListSortingStrategy}>
                {rows.map((row) => (
                  <SortableScanTableRow
                    key={row.id}
                    row={row}
                    strategy={row.strategy_id ? strategyById.get(row.strategy_id) : undefined}
                    portfolio={row.portfolio_id ? portfolioById.get(row.portfolio_id) : undefined}
                    onBacktestStrategy={onBacktestStrategy}
                    onOpenPortfolio={onOpenPortfolio}
                    onLaneChange={onLaneChange}
                    onPortfolioLaneChange={onPortfolioLaneChange}
                    lanePending={lanePending}
                  />
                ))}
              </SortableContext>
              {rows.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={12}
                    className="py-4 text-center text-[11px] text-muted-foreground"
                  >
                    Drop strategies here
                  </TableCell>
                </TableRow>
              ) : null}
            </TableBody>
          </Table>
        </div>
      ) : null}
    </div>
  )
}

function LegacyScanSection({
  rows,
  expanded,
  onToggle,
}: {
  rows: ScanRow[]
  expanded: boolean
  onToggle: () => void
}) {
  const activeCount = countActiveSignals(rows)

  return (
    <div className="overflow-hidden rounded-md border border-border/60">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center gap-2 bg-muted/20 px-3 py-2 text-left text-xs hover:bg-muted/30"
      >
        <ChevronDown
          className={cn('size-3.5 shrink-0 transition-transform', !expanded && '-rotate-90')}
        />
        <span className="font-medium">Legacy signals</span>
        <span className="text-muted-foreground">
          {rows.length} rows · {activeCount} active
        </span>
      </button>
      {expanded ? (
        <div className="overflow-x-auto">
          <Table className="text-xs">
            <ScanTableHeader />
            <TableBody>
              {rows.map((row) => (
                <StaticScanTableRow key={row.id} row={row} />
              ))}
            </TableBody>
          </Table>
        </div>
      ) : null}
    </div>
  )
}

function buildStrategyLaneUpdates(
  lane: ScanLane,
  rows: ScanRow[],
  strategyById: Map<string, SavedStrategy>,
): { id: string; payload: UpdateStrategyPayload }[] {
  const updates: { id: string; payload: UpdateStrategyPayload }[] = []
  rows.forEach((row, index) => {
    if (row.source !== 'builder' || !row.strategy_id) return
    const strategy = strategyById.get(row.strategy_id)
    if (!strategy) return
    const currentLane = resolveScanLane(strategy)
    const currentOrder = strategy.scan_sort_order ?? 0
    if (currentLane === lane && currentOrder === index) return
    updates.push({
      id: strategy.id,
      payload: { scan_lane: lane, scan_sort_order: index },
    })
  })
  return updates
}

function buildPortfolioLaneUpdates(
  lane: ScanLane,
  rows: ScanRow[],
  portfolioById: Map<string, SavedPortfolio>,
): { id: string; payload: UpdatePortfolioPayload }[] {
  const updates: { id: string; payload: UpdatePortfolioPayload }[] = []
  rows.forEach((row, index) => {
    if (row.source !== 'portfolio' || !row.portfolio_id) return
    const portfolio = portfolioById.get(row.portfolio_id)
    if (!portfolio) return
    const currentLane = resolvePortfolioScanLane(portfolio)
    const currentOrder = portfolio.scan_sort_order ?? 0
    if (currentLane === lane && currentOrder === index) return
    updates.push({
      id: portfolio.id,
      payload: { scan_lane: lane, scan_sort_order: index },
    })
  })
  return updates
}

export function ScanSection({
  onBacktestStrategy,
  onOpenPortfolio,
}: {
  onBacktestStrategy?: (strategy: SavedStrategy) => void
  onOpenPortfolio?: (state: PortfolioInitialState) => void
}) {
  const queryClient = useQueryClient()
  const [query, setQuery] = useState('')
  const [symbolFilter, setSymbolFilter] = useState<string>('all')
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    () => new Set(['active', 'legacy']),
  )
  const [laneRows, setLaneRows] = useState<LaneRows>({
    active: [],
    testing: [],
    archived: [],
  })
  const [legacyRows, setLegacyRows] = useState<ScanRow[]>([])

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )

  const {
    data: rows = [],
    isLoading,
    isFetching,
    error,
    refetch,
  } = useQuery({
    queryKey: ['scan'],
    queryFn: getScan,
  })
  const { data: savedStrategies = [] } = useQuery({
    queryKey: ['strategies'],
    queryFn: getSavedStrategies,
  })
  const { data: savedPortfolios = [] } = useQuery({
    queryKey: ['portfolios'],
    queryFn: getSavedPortfolios,
  })

  const strategyById = useMemo(
    () => new Map(savedStrategies.map((strategy) => [strategy.id, strategy])),
    [savedStrategies],
  )
  const portfolioById = useMemo(
    () => new Map(savedPortfolios.map((portfolio) => [portfolio.id, portfolio])),
    [savedPortfolios],
  )

  const filtered = useMemo(() => {
    return rows.filter((row) => {
      if (symbolFilter !== 'all' && row.symbol !== symbolFilter) return false
      if (!query) return true
      const q = query.toLowerCase()
      return (
        row.symbol.toLowerCase().includes(q) ||
        row.signal.toLowerCase().includes(q) ||
        row.description.toLowerCase().includes(q)
      )
    })
  }, [rows, query, symbolFilter])

  const grouped = useMemo(
    () => groupScanRows(filtered, strategyById, portfolioById),
    [filtered, strategyById, portfolioById],
  )

  useEffect(() => {
    setLaneRows(grouped.lanes)
    setLegacyRows(grouped.legacy)
  }, [grouped])

  const symbols = useMemo(
    () => sortSymbols([...new Set(rows.map((row) => row.symbol))]),
    [rows],
  )

  const activeSignals = filtered.filter((row) => row.buy_signal || row.hold_long).length

  const laneMutation = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: UpdateStrategyPayload }) =>
      updateStrategy(id, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] })
    },
  })

  const portfolioLaneMutation = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: UpdatePortfolioPayload }) =>
      updatePortfolio(id, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolios'] })
    },
  })

  const persistLaneLayout = useCallback(
    async (nextLanes: LaneRows, touched: Set<ScanLane>) => {
      const strategyUpdates = [...touched].flatMap((lane) =>
        buildStrategyLaneUpdates(lane, nextLanes[lane], strategyById),
      )
      const portfolioUpdates = [...touched].flatMap((lane) =>
        buildPortfolioLaneUpdates(lane, nextLanes[lane], portfolioById),
      )
      const uniqueStrategyUpdates = new Map<string, UpdateStrategyPayload>()
      for (const update of strategyUpdates) {
        uniqueStrategyUpdates.set(update.id, {
          ...uniqueStrategyUpdates.get(update.id),
          ...update.payload,
        })
      }
      const uniquePortfolioUpdates = new Map<string, UpdatePortfolioPayload>()
      for (const update of portfolioUpdates) {
        uniquePortfolioUpdates.set(update.id, {
          ...uniquePortfolioUpdates.get(update.id),
          ...update.payload,
        })
      }
      for (const [id, payload] of uniqueStrategyUpdates.entries()) {
        await laneMutation.mutateAsync({ id, payload })
      }
      for (const [id, payload] of uniquePortfolioUpdates.entries()) {
        await portfolioLaneMutation.mutateAsync({ id, payload })
      }
    },
    [laneMutation, portfolioLaneMutation, strategyById, portfolioById],
  )

  const handleLaneChange = useCallback(
    async (strategy: SavedStrategy, lane: ScanLane) => {
      const currentLane = resolveScanLane(strategy)
      if (currentLane === lane) return

      const nextLanes: LaneRows = {
        active: [...laneRows.active],
        testing: [...laneRows.testing],
        archived: [...laneRows.archived],
      }
      for (const sourceLane of SCAN_LANES) {
        nextLanes[sourceLane] = nextLanes[sourceLane].filter((row) => row.strategy_id !== strategy.id)
      }
      const movedRow =
        laneRows[currentLane].find((row) => row.strategy_id === strategy.id) ??
        laneRows.testing.find((row) => row.strategy_id === strategy.id) ??
        laneRows.active.find((row) => row.strategy_id === strategy.id) ??
        laneRows.archived.find((row) => row.strategy_id === strategy.id)
      if (movedRow) {
        nextLanes[lane] = [...nextLanes[lane], movedRow]
      }
      setLaneRows(nextLanes)
      setExpandedSections((prev) => new Set(prev).add(lane))
      try {
        await laneMutation.mutateAsync({
          id: strategy.id,
          payload: {
            scan_lane: lane,
            scan_sort_order: nextLanes[lane].length - 1,
          },
        })
      } catch {
        setLaneRows(grouped.lanes)
      }
    },
    [grouped.lanes, laneMutation, laneRows],
  )

  const handlePortfolioLaneChange = useCallback(
    async (portfolio: SavedPortfolio, lane: ScanLane) => {
      const currentLane = resolvePortfolioScanLane(portfolio)
      if (currentLane === lane) return

      const nextLanes: LaneRows = {
        active: [...laneRows.active],
        testing: [...laneRows.testing],
        archived: [...laneRows.archived],
      }
      for (const sourceLane of SCAN_LANES) {
        nextLanes[sourceLane] = nextLanes[sourceLane].filter(
          (row) => row.portfolio_id !== portfolio.id,
        )
      }
      const movedRow =
        laneRows[currentLane].find((row) => row.portfolio_id === portfolio.id) ??
        laneRows.testing.find((row) => row.portfolio_id === portfolio.id) ??
        laneRows.active.find((row) => row.portfolio_id === portfolio.id) ??
        laneRows.archived.find((row) => row.portfolio_id === portfolio.id)
      if (movedRow) {
        nextLanes[lane] = [...nextLanes[lane], movedRow]
      }
      setLaneRows(nextLanes)
      setExpandedSections((prev) => new Set(prev).add(lane))
      try {
        await portfolioLaneMutation.mutateAsync({
          id: portfolio.id,
          payload: {
            scan_lane: lane,
            scan_sort_order: nextLanes[lane].length - 1,
          },
        })
      } catch {
        setLaneRows(grouped.lanes)
      }
    },
    [grouped.lanes, laneRows, portfolioLaneMutation],
  )

  const handleDragEnd = useCallback(
    async (event: DragEndEvent) => {
      const { active, over } = event
      if (!over) return

      const activeId = String(active.id)
      const overId = String(over.id)
      const sourceLane = findLaneForRow(laneRows, activeId)
      if (!sourceLane) return

      let targetLane = parseLaneContainerId(overId) ?? findLaneForRow(laneRows, overId) ?? sourceLane

      const nextLanes: LaneRows = {
        active: [...laneRows.active],
        testing: [...laneRows.testing],
        archived: [...laneRows.archived],
      }

      const sourceRows = [...nextLanes[sourceLane]]
      const activeIndex = sourceRows.findIndex((row) => row.id === activeId)
      if (activeIndex < 0) return

      if (sourceLane === targetLane && !parseLaneContainerId(overId)) {
        const overIndex = sourceRows.findIndex((row) => row.id === overId)
        if (overIndex < 0 || activeIndex === overIndex) return
        nextLanes[sourceLane] = arrayMove(sourceRows, activeIndex, overIndex)
      } else {
        const [movedRow] = sourceRows.splice(activeIndex, 1)
        nextLanes[sourceLane] = sourceRows
        const targetRows = sourceLane === targetLane ? sourceRows : [...nextLanes[targetLane]]
        if (parseLaneContainerId(overId)) {
          targetRows.push(movedRow)
        } else {
          const overIndex = targetRows.findIndex((row) => row.id === overId)
          targetRows.splice(overIndex >= 0 ? overIndex : targetRows.length, 0, movedRow)
        }
        nextLanes[targetLane] = targetRows
      }

      setLaneRows(nextLanes)
      const touched = new Set<ScanLane>([sourceLane])
      if (targetLane !== sourceLane) touched.add(targetLane)
      try {
        await persistLaneLayout(nextLanes, touched)
      } catch {
        setLaneRows(grouped.lanes)
      }
    },
    [grouped.lanes, laneRows, persistLaneLayout],
  )

  const toggleSection = (sectionId: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev)
      if (next.has(sectionId)) next.delete(sectionId)
      else next.add(sectionId)
      return next
    })
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-xs text-muted-foreground">
          {symbols.length} symbols · {rows.length} rows ·{' '}
          <span className="text-[var(--gain)]">{activeSignals} active</span>
        </p>
        <div className="flex min-w-0 flex-1 flex-wrap items-center justify-end gap-2">
          <div className="relative min-w-[180px] flex-1 sm:max-w-xs">
            <Search className="absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="h-7 pl-7 text-xs"
            />
          </div>
          <Select value={symbolFilter} onValueChange={(v) => v && setSymbolFilter(v)}>
            <SelectTrigger className="h-7 w-[120px] text-xs">
              <SelectValue placeholder="Symbol">
                {(value: string) => (value === 'all' ? 'All symbols' : value)}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All symbols</SelectItem>
              {symbols.map((symbol) => (
                <SelectItem key={symbol} value={symbol} className="font-mono text-xs">
                  {symbol}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            size="sm"
            className="h-7 gap-1 px-2 text-xs"
            onClick={() => refetch()}
            disabled={isFetching}
          >
            {isFetching ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <RefreshCw className="size-3" />
            )}
            Refresh
          </Button>
        </div>
      </div>

      {error && (
        <Card className="border-destructive/50 bg-destructive/5 p-2">
          <pre className="overflow-x-auto whitespace-pre-wrap text-[11px] text-destructive">
            {String(error)}
          </pre>
        </Card>
      )}

      {isLoading ? (
        <div className="flex items-center justify-center gap-2 py-6 text-xs text-muted-foreground">
          <Loader2 className="size-3.5 animate-spin" />
          Running scan…
        </div>
      ) : (
        <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
          <div className="flex flex-col gap-2">
            {SCAN_LANES.map((lane) => (
              <ScanLaneSection
                key={lane}
                lane={lane}
                rows={laneRows[lane]}
                expanded={expandedSections.has(lane)}
                onToggle={() => toggleSection(lane)}
                strategyById={strategyById}
                portfolioById={portfolioById}
                onBacktestStrategy={onBacktestStrategy}
                onOpenPortfolio={onOpenPortfolio}
                onLaneChange={handleLaneChange}
                onPortfolioLaneChange={handlePortfolioLaneChange}
                lanePending={laneMutation.isPending || portfolioLaneMutation.isPending}
              />
            ))}
            <LegacyScanSection
              rows={legacyRows}
              expanded={expandedSections.has('legacy')}
              onToggle={() => toggleSection('legacy')}
            />
          </div>
        </DndContext>
      )}

      {!isLoading && filtered.length === 0 && (
        <p className="py-3 text-center text-xs text-muted-foreground">No rows match your filters.</p>
      )}
    </div>
  )
}
