'use client'

import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Check, Loader2, RefreshCw, Search, X } from 'lucide-react'
import { getScan, type ScanRow } from '@/api'
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

const SCAN_SYMBOL_ORDER = [
  'SPY',
  'SMH',
  'QQQ',
  'SOXX',
  'IWM',
  'FXI',
  'AAPL',
  'GDX',
  'MSFT',
  'GLD',
  'XBI',
  'TLT',
] as const

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

const compactHead = 'h-7 px-1.5 py-0 text-[11px] font-medium'
const compactCell = 'px-1.5 py-0.5'

function sortScanRows(rows: ScanRow[]): ScanRow[] {
  const symbolRank = new Map(SCAN_SYMBOL_ORDER.map((symbol, index) => [symbol, index]))
  const signalRank = new Map(SIGNAL_ORDER.map((signal, index) => [signal, index]))
  return [...rows].sort((a, b) => {
    const symbolDiff =
      (symbolRank.get(a.symbol as (typeof SCAN_SYMBOL_ORDER)[number]) ?? SCAN_SYMBOL_ORDER.length) -
      (symbolRank.get(b.symbol as (typeof SCAN_SYMBOL_ORDER)[number]) ?? SCAN_SYMBOL_ORDER.length)
    if (symbolDiff !== 0) return symbolDiff
    return (
      (signalRank.get(a.signal as (typeof SIGNAL_ORDER)[number]) ?? SIGNAL_ORDER.length) -
      (signalRank.get(b.signal as (typeof SIGNAL_ORDER)[number]) ?? SIGNAL_ORDER.length)
    )
  })
}

function sortSymbols(symbols: string[]): string[] {
  const symbolRank = new Map<string, number>(
    SCAN_SYMBOL_ORDER.map((symbol, index) => [symbol, index]),
  )
  return [...symbols].sort(
    (a, b) =>
      (symbolRank.get(a) ?? SCAN_SYMBOL_ORDER.length) -
      (symbolRank.get(b) ?? SCAN_SYMBOL_ORDER.length),
  )
}

function BoolCell({ value }: { value: boolean }) {
  return value ? (
    <Check className="mx-auto size-3 text-[var(--gain)]" aria-label="True" />
  ) : (
    <X className="mx-auto size-3 text-muted-foreground/50" aria-label="False" />
  )
}

export function ScanSection() {
  const [query, setQuery] = useState('')
  const [symbolFilter, setSymbolFilter] = useState<string>('all')

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

  const symbols = useMemo(
    () => sortSymbols([...new Set(rows.map((row) => row.symbol))]),
    [rows],
  )

  const filtered = useMemo(() => {
    return sortScanRows(
      rows.filter((r) => {
        if (symbolFilter !== 'all' && r.symbol !== symbolFilter) return false
        if (!query) return true
        const q = query.toLowerCase()
        return (
          r.symbol.toLowerCase().includes(q) ||
          r.signal.toLowerCase().includes(q) ||
          r.description.toLowerCase().includes(q)
        )
      }),
    )
  }, [rows, query, symbolFilter])

  const activeSignals = filtered.filter((r) => r.buy_signal || r.hold_long).length

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
              {symbols.map((s) => (
                <SelectItem key={s} value={s} className="font-mono text-xs">
                  {s}
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
        <div className="overflow-x-auto rounded-md border border-border/60">
          <Table className="text-xs">
            <TableHeader className="bg-muted/30">
              <TableRow className="hover:bg-transparent">
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
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((r) => (
                <ScanTableRow key={r.id} row={r} />
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {!isLoading && filtered.length === 0 && (
        <p className="py-3 text-center text-xs text-muted-foreground">No rows match your filters.</p>
      )}
    </div>
  )
}

function ScanTableRow({ row: r }: { row: ScanRow }) {
  return (
    <TableRow className={cn('hover:bg-muted/30', (r.buy_signal || r.hold_long) && 'bg-[var(--gain)]/8')}>
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
      <TableCell className={cn(compactCell, 'max-w-[360px] truncate text-[11px] text-muted-foreground')} title={r.description}>
        {r.description}
      </TableCell>
    </TableRow>
  )
}
