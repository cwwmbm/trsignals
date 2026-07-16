'use client'

import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useDroppable,
  useSensor,
  useSensors,
} from '@dnd-kit/core'
import {
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import {
  ArrowUpRight,
  ChevronDown,
  GripVertical,
  Loader2,
  RefreshCw,
  Search,
} from 'lucide-react'
import type { SavedPortfolio, SavedStrategy, ScanLane, ScanRow } from '@/api'
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
import { ConditionSnapshotHint } from '@/components/scan/condition-snapshot-hint'
import { ScanSignalCheck } from '@/components/scan/scan-signal-indicator'
import { useScanBoard } from '@/hooks/use-scan-board'
import { scanRowAccentClass } from '@/lib/scan-signals'
import {
  LANE_LABELS,
  SCAN_LANES,
  countActiveSignals,
  laneContainerId,
  resolvePortfolioScanLane,
  resolveScanLane,
} from '@/lib/scan-board'

const compactHead = 'h-7 px-1.5 py-0 text-[11px] font-medium'
const compactCell = 'px-1.5 py-0.5'

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
      <TableCell className={cn(compactCell, 'font-mono text-[11px]')}>
        <div className="flex min-w-0 items-center gap-1">
          <span className="truncate">{r.signal}</span>
          <ConditionSnapshotHint row={r} />
        </div>
      </TableCell>
      <TableCell className={compactCell}>
        <ScanSignalCheck row={r} kind="buy" />
      </TableCell>
      <TableCell className={compactCell}>
        <ScanSignalCheck row={r} kind="hold" />
      </TableCell>
      <TableCell className={compactCell}>
        <ScanSignalCheck row={r} kind="sell" />
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
        scanRowAccentClass(row),
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
    <TableRow className={cn('hover:bg-muted/30', scanRowAccentClass(row))}>
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

export function ScanSection({
  onBacktestStrategy,
  onOpenPortfolio,
}: {
  onBacktestStrategy?: (strategy: SavedStrategy) => void
  onOpenPortfolio?: (state: PortfolioInitialState) => void
}) {
  const board = useScanBoard()

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-xs text-muted-foreground">
          {board.symbols.length} symbols · {board.rows.length} rows ·{' '}
          <span className="text-[var(--gain)]">{board.activeSignals} active</span>
        </p>
        <div className="flex min-w-0 flex-1 flex-wrap items-center justify-end gap-2">
          <div className="relative min-w-[180px] flex-1 sm:max-w-xs">
            <Search className="absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search…"
              value={board.query}
              onChange={(e) => board.setQuery(e.target.value)}
              className="h-7 pl-7 text-xs"
            />
          </div>
          <Select value={board.symbolFilter} onValueChange={(v) => v && board.setSymbolFilter(v)}>
            <SelectTrigger className="h-7 w-[120px] text-xs">
              <SelectValue placeholder="Symbol">
                {(value: string) => (value === 'all' ? 'All symbols' : value)}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All symbols</SelectItem>
              {board.symbols.map((symbol) => (
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
            onClick={() => board.refetch()}
            disabled={board.isFetching}
          >
            {board.isFetching ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <RefreshCw className="size-3" />
            )}
            Refresh
          </Button>
        </div>
      </div>

      {board.error && (
        <Card className="border-destructive/50 bg-destructive/5 p-2">
          <pre className="overflow-x-auto whitespace-pre-wrap text-[11px] text-destructive">
            {String(board.error)}
          </pre>
        </Card>
      )}

      {board.isLoading ? (
        <div className="flex items-center justify-center gap-2 py-6 text-xs text-muted-foreground">
          <Loader2 className="size-3.5 animate-spin" />
          Running scan…
        </div>
      ) : (
        <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={board.handleDragEnd}>
          <div className="flex flex-col gap-2">
            {SCAN_LANES.map((lane) => (
              <ScanLaneSection
                key={lane}
                lane={lane}
                rows={board.laneRows[lane]}
                expanded={board.expandedSections.has(lane)}
                onToggle={() => board.toggleSection(lane)}
                strategyById={board.strategyById}
                portfolioById={board.portfolioById}
                onBacktestStrategy={onBacktestStrategy}
                onOpenPortfolio={onOpenPortfolio}
                onLaneChange={board.handleLaneChange}
                onPortfolioLaneChange={board.handlePortfolioLaneChange}
                lanePending={board.lanePending}
              />
            ))}
            <LegacyScanSection
              rows={board.legacyRows}
              expanded={board.expandedSections.has('legacy')}
              onToggle={() => board.toggleSection('legacy')}
            />
          </div>
        </DndContext>
      )}

      {!board.isLoading && board.filtered.length === 0 && (
        <p className="py-3 text-center text-xs text-muted-foreground">No rows match your filters.</p>
      )}
    </div>
  )
}
