'use client'

import { useState } from 'react'
import { Loader2, RefreshCw, Search } from 'lucide-react'
import type { SavedPortfolio, SavedStrategy, ScanLane, ScanRow } from '@/api'
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
import { cn } from '@/lib/utils'
import { ConditionSnapshotHint } from '@/components/scan/condition-snapshot-hint'
import { ScanSignalBadge } from '@/components/scan/scan-signal-indicator'
import { useScanBoard } from '@/hooks/use-scan-board'
import { scanCardAccentClass } from '@/lib/scan-signals'
import {
  LANE_LABELS,
  SCAN_LANES,
  countActiveSignals,
  resolvePortfolioScanLane,
  resolveScanLane,
} from '@/lib/scan-board'

function ScanMobileCard({
  row,
  strategy,
  portfolio,
  onLaneChange,
  onPortfolioLaneChange,
  lanePending,
}: {
  row: ScanRow
  strategy?: SavedStrategy
  portfolio?: SavedPortfolio
  onLaneChange: (strategy: SavedStrategy, lane: ScanLane) => void
  onPortfolioLaneChange: (portfolio: SavedPortfolio, lane: ScanLane) => void
  lanePending: boolean
}) {
  const currentLane = portfolio ? resolvePortfolioScanLane(portfolio) : resolveScanLane(strategy)

  return (
    <Card
      className={cn('gap-2 p-3 shadow-none', scanCardAccentClass(row))}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="font-mono text-base font-semibold">{row.symbol}</p>
          <div className="mt-0.5 flex min-w-0 items-center gap-1">
            <p className="truncate font-mono text-xs text-muted-foreground">{row.signal}</p>
            <ConditionSnapshotHint row={row} />
          </div>
        </div>
        <div className="flex shrink-0 gap-1">
          <ScanSignalBadge row={row} kind="buy" />
          <ScanSignalBadge row={row} kind="hold" />
          <ScanSignalBadge row={row} kind="sell" />
        </div>
      </div>

      <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs tabular-nums">
        <span>
          PnL{' '}
          <span className={cn('font-mono', row.trade_pnl > 0 ? 'text-[var(--gain)]' : 'text-muted-foreground')}>
            {row.trade_pnl.toFixed(1)}%
          </span>
        </span>
        <span>
          Prf <span className="font-mono">{row.profit}</span>
        </span>
        <span>
          Days <span className="font-mono">{row.days}</span>
        </span>
        <span>
          Kelly{' '}
          <span className="font-mono text-muted-foreground">
            {row.kelly === null ? '—' : `${row.kelly.toFixed(1)}%`}
          </span>
        </span>
      </div>

      {row.description ? (
        <p className="line-clamp-2 text-[11px] text-muted-foreground">{row.description}</p>
      ) : null}

      {strategy || portfolio ? (
        <Select
          value={currentLane}
          onValueChange={(value) => {
            if (!value || value === currentLane) return
            if (portfolio) onPortfolioLaneChange(portfolio, value as ScanLane)
            else if (strategy) onLaneChange(strategy, value as ScanLane)
          }}
          disabled={lanePending}
        >
          <SelectTrigger className="h-8 w-full text-xs">
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
      ) : null}
    </Card>
  )
}

function LegacyMobileCard({ row }: { row: ScanRow }) {
  return (
    <Card className={cn('gap-2 p-3 shadow-none', scanCardAccentClass(row))}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="font-mono text-base font-semibold">{row.symbol}</p>
          <div className="mt-0.5 flex min-w-0 items-center gap-1">
            <p className="truncate font-mono text-xs text-muted-foreground">{row.signal}</p>
            <ConditionSnapshotHint row={row} />
          </div>
        </div>
        <div className="flex shrink-0 gap-1">
          <ScanSignalBadge row={row} kind="buy" />
          <ScanSignalBadge row={row} kind="hold" />
          <ScanSignalBadge row={row} kind="sell" />
        </div>
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs tabular-nums">
        <span>
          PnL{' '}
          <span className={cn('font-mono', row.trade_pnl > 0 ? 'text-[var(--gain)]' : 'text-muted-foreground')}>
            {row.trade_pnl.toFixed(1)}%
          </span>
        </span>
        <span>
          Days <span className="font-mono">{row.days}</span>
        </span>
      </div>
    </Card>
  )
}

export function ScanMobileSection() {
  const [activeLane, setActiveLane] = useState<ScanLane | 'legacy'>('active')
  const board = useScanBoard(['active'])

  const laneRows =
    activeLane === 'legacy' ? board.legacyRows : board.laneRows[activeLane as ScanLane]

  return (
    <div className="flex flex-col gap-3 pb-4">
      <div className="flex flex-col gap-2">
        <p className="text-xs text-muted-foreground">
          {board.symbols.length} symbols · {board.rows.length} rows ·{' '}
          <span className="text-[var(--gain)]">{board.activeSignals} active</span>
        </p>
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search symbol, signal…"
            value={board.query}
            onChange={(e) => board.setQuery(e.target.value)}
            className="h-9 pl-9 text-sm"
          />
        </div>
        <div className="flex gap-2">
          <Select value={board.symbolFilter} onValueChange={(v) => v && board.setSymbolFilter(v)}>
            <SelectTrigger className="h-9 flex-1 text-sm">
              <SelectValue placeholder="Symbol">
                {(value: string) => (value === 'all' ? 'All symbols' : value)}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All symbols</SelectItem>
              {board.symbols.map((symbol) => (
                <SelectItem key={symbol} value={symbol} className="font-mono text-sm">
                  {symbol}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            size="icon"
            className="size-9 shrink-0"
            onClick={() => board.refetch()}
            disabled={board.isFetching}
            aria-label="Refresh scan"
          >
            {board.isFetching ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <RefreshCw className="size-4" />
            )}
          </Button>
        </div>
      </div>

      <div className="flex gap-1 overflow-x-auto pb-1">
        {SCAN_LANES.map((lane) => {
          const count = board.laneRows[lane].length
          const active = countActiveSignals(board.laneRows[lane])
          return (
            <button
              key={lane}
              type="button"
              onClick={() => setActiveLane(lane)}
              className={cn(
                'shrink-0 rounded-full px-3 py-1.5 text-xs font-medium transition-colors',
                activeLane === lane
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground',
              )}
            >
              {LANE_LABELS[lane]} ({active}/{count})
            </button>
          )
        })}
        <button
          type="button"
          onClick={() => setActiveLane('legacy')}
          className={cn(
            'shrink-0 rounded-full px-3 py-1.5 text-xs font-medium transition-colors',
            activeLane === 'legacy'
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted text-muted-foreground',
          )}
        >
          Legacy ({board.legacyRows.length})
        </button>
      </div>

      {board.error && (
        <Card className="border-destructive/50 bg-destructive/5 p-3">
          <pre className="whitespace-pre-wrap text-xs text-destructive">{String(board.error)}</pre>
        </Card>
      )}

      {board.isLoading ? (
        <div className="flex items-center justify-center gap-2 py-12 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          Running scan…
        </div>
      ) : activeLane === 'legacy' ? (
        <div className="flex flex-col gap-2">
          {board.legacyRows.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted-foreground">No legacy rows.</p>
          ) : (
            board.legacyRows.map((row) => <LegacyMobileCard key={row.id} row={row} />)
          )}
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {laneRows.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted-foreground">No rows in this lane.</p>
          ) : (
            laneRows.map((row) => (
              <ScanMobileCard
                key={row.id}
                row={row}
                strategy={row.strategy_id ? board.strategyById.get(row.strategy_id) : undefined}
                portfolio={row.portfolio_id ? board.portfolioById.get(row.portfolio_id) : undefined}
                onLaneChange={board.handleLaneChange}
                onPortfolioLaneChange={board.handlePortfolioLaneChange}
                lanePending={board.lanePending}
              />
            ))
          )}
        </div>
      )}

      {!board.isLoading && board.filtered.length === 0 && (
        <p className="py-3 text-center text-sm text-muted-foreground">No rows match your filters.</p>
      )}
    </div>
  )
}
