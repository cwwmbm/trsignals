'use client'

import { useMutation, useQuery } from '@tanstack/react-query'
import { Loader2, Play, Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { DetailedResult, PortfolioOverlapMode, SavedStrategy } from '@/api'
import { getSavedStrategies, runPortfolioSimulation } from '@/api'
import { DetailResults } from '@/components/backtest/detail-results'
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
import { compareSymbols } from '@/lib/symbol-order'
import { cn } from '@/lib/utils'

const compactHead = 'h-7 px-1.5 py-0 text-[11px] font-medium'
const compactCell = 'px-1.5 py-0.5 align-top'

function sortStrategies(strategies: SavedStrategy[]) {
  return [...strategies].sort((a, b) => {
    const symbolOrder = compareSymbols(a.symbol, b.symbol)
    if (symbolOrder !== 0) return symbolOrder
    return a.name.localeCompare(b.name)
  })
}

export function PortfolioSection() {
  const { data: strategies = [], isLoading, error } = useQuery({
    queryKey: ['strategies'],
    queryFn: getSavedStrategies,
  })

  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set())
  const [search, setSearch] = useState('')
  const [overlapMode, setOverlapMode] = useState<PortfolioOverlapMode>('first_signal_only')
  const [proxySymbol, setProxySymbol] = useState('')
  const [years, setYears] = useState('25')

  const mutation = useMutation({
    mutationFn: runPortfolioSimulation,
  })

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

  const selectedInOrder = useMemo(
    () => sortedStrategies.filter((strategy) => selectedIds.has(strategy.id)).map((s) => s.id),
    [selectedIds, sortedStrategies],
  )

  const selectedShort = useMemo(
    () => sortedStrategies.some((strategy) => selectedIds.has(strategy.id) && strategy.direction === 'short'),
    [selectedIds, sortedStrategies],
  )

  const canSimulate = selectedInOrder.length > 0 && !selectedShort && !mutation.isPending

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
    mutation.mutate({
      strategy_ids: selectedInOrder,
      overlap_mode: overlapMode,
      ...(proxySymbol.trim() ? { proxy_symbol: proxySymbol.trim().toUpperCase() } : {}),
      years: Number.isFinite(parsedYears) ? parsedYears : 25,
    })
  }

  const result = mutation.data

  return (
    <div className="space-y-4">
      <Card className="p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold">Strategies</h2>
            <p className="text-xs text-muted-foreground">
              Select long strategies to combine into an all-in portfolio simulation.
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
          <div className="mt-3 overflow-x-auto rounded-md border border-border/60">
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
                {filteredStrategies.map((strategy) => {
                  const isShort = strategy.direction === 'short'
                  const checked = selectedIds.has(strategy.id)
                  return (
                    <TableRow key={strategy.id} className={isShort ? 'opacity-60' : undefined}>
                      <TableCell className={compactCell}>
                        <Checkbox
                          checked={checked}
                          disabled={isShort}
                          onCheckedChange={(value) => toggleStrategy(strategy, value === true)}
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
                      <TableCell className={cn(compactCell, 'max-w-md text-xs text-muted-foreground')}>
                        {(strategy.description ?? '').trim() || '—'}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </Card>

      <Card className="p-4">
        <h2 className="text-sm font-semibold">Settings</h2>
        <div className="mt-3 space-y-4">
          <div>
            <Label className="text-xs text-muted-foreground">Overlapping signals</Label>
            <RadioGroup
              value={overlapMode}
              onValueChange={(value) => value && setOverlapMode(value as PortfolioOverlapMode)}
              className="mt-2 gap-3"
            >
              <div className="flex items-start gap-2">
                <RadioGroupItem value="first_signal_only" id="overlap-first" className="mt-0.5" />
                <div>
                  <Label htmlFor="overlap-first" className="text-sm font-normal">
                    First signal only
                  </Label>
                  <p className="text-[11px] text-muted-foreground">
                    While the portfolio is in a trade, later buy signals from other strategies are
                    ignored. Exit when the strategy that triggered entry exits.
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <RadioGroupItem value="hold_until_all_exit" id="overlap-all" className="mt-0.5" />
                <div>
                  <Label htmlFor="overlap-all" className="text-sm font-normal">
                    Hold until all exit
                  </Label>
                  <p className="text-[11px] text-muted-foreground">
                    Stay in while any active strategy is still holding. New buys add to the active
                    set. Same-day exit and entry keeps the portfolio in the trade.
                  </p>
                </div>
              </div>
            </RadioGroup>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 sm:max-w-xl">
            <div className="flex flex-col gap-1">
              <Label htmlFor="portfolio-proxy" className="text-xs text-muted-foreground">
                Proxy symbol
              </Label>
              <Input
                id="portfolio-proxy"
                value={proxySymbol}
                onChange={(e) => setProxySymbol(e.target.value.toUpperCase())}
                placeholder="Optional — overrides strategy proxies"
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
            {selectedInOrder.length === 0
              ? 'Select at least one strategy'
              : selectedShort
                ? 'Short strategies cannot be included'
                : `${selectedInOrder.length} selected`}
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

      {result ? <DetailResults result={result} /> : null}
    </div>
  )
}
