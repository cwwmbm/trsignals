'use client'

import { useMemo, useState } from 'react'
import { Check, Pencil, Search, X } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
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
import { SCAN_ROWS, SIGNALS, SYMBOLS, type ScanRow } from '@/lib/mock-data'

function BoolCell({ value }: { value: boolean }) {
  return value ? (
    <span className="inline-flex items-center gap-1 font-mono text-xs text-[var(--gain)]">
      <Check className="size-3.5" />
      True
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 font-mono text-xs text-muted-foreground">
      <X className="size-3.5" />
      False
    </span>
  )
}

export function ScanSection() {
  const [rows, setRows] = useState<ScanRow[]>(SCAN_ROWS)
  const [query, setQuery] = useState('')
  const [symbolFilter, setSymbolFilter] = useState<string>('all')
  const [editingDesc, setEditingDesc] = useState<string | null>(null)

  const filtered = useMemo(() => {
    return rows.filter((r) => {
      if (symbolFilter !== 'all' && r.symbol !== symbolFilter) return false
      if (!query) return true
      const q = query.toLowerCase()
      return (
        r.symbol.toLowerCase().includes(q) ||
        r.signal.toLowerCase().includes(q) ||
        r.description.toLowerCase().includes(q)
      )
    })
  }, [rows, query, symbolFilter])

  const updateRow = (id: string, patch: Partial<ScanRow>) =>
    setRows((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)))

  const activeSignals = filtered.filter((r) => r.buySignal || r.holdLong).length

  return (
    <div className="flex flex-col gap-6">
      <Card className="border-border/60 p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold">Scan</h2>
            <p className="text-sm text-muted-foreground">
              Latest scan across {SYMBOLS.length} symbols · {rows.length} signal rows ·{' '}
              <span className="text-[var(--gain)]">{activeSignals} active</span>
            </p>
          </div>
          <Badge variant="outline" className="font-mono text-xs">
            15 of 15 completed
          </Badge>
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <div className="relative flex-1 min-w-[220px]">
            <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search symbol, signal, or description…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="pl-9"
            />
          </div>
          <Select value={symbolFilter} onValueChange={(v) => v && setSymbolFilter(v)}>
            <SelectTrigger className="w-[160px]">
              <SelectValue placeholder="Symbol">
                {(value: string) => (value === 'all' ? 'All symbols' : value)}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All symbols</SelectItem>
              {SYMBOLS.map((s) => (
                <SelectItem key={s} value={s} className="font-mono">
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="mt-5 overflow-x-auto rounded-lg border border-border/60">
          <Table>
            <TableHeader className="bg-card">
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead className="min-w-[160px]">Signal</TableHead>
                <TableHead>Buy?</TableHead>
                <TableHead>HoldLong?</TableHead>
                <TableHead>Sell?</TableHead>
                <TableHead className="text-right">Days</TableHead>
                <TableHead className="text-right">Profit</TableHead>
                <TableHead className="text-right">TradePnL</TableHead>
                <TableHead className="text-right">Kelly</TableHead>
                <TableHead className="min-w-[320px]">Description</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((r) => (
                <TableRow
                  key={r.id}
                  className={cn((r.buySignal || r.holdLong) && 'bg-[var(--gain)]/8')}
                >
                  <TableCell className="font-mono font-medium">{r.symbol}</TableCell>

                  {/* Editable Signal */}
                  <TableCell>
                    <Select
                      value={r.signal}
                      onValueChange={(v) => v && updateRow(r.id, { signal: v })}
                    >
                      <SelectTrigger
                        size="sm"
                        className="h-8 w-full border-transparent bg-transparent font-mono text-xs hover:border-border hover:bg-muted/50 data-[state=open]:border-border"
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {SIGNALS.map((sig) => (
                          <SelectItem key={sig} value={sig} className="font-mono">
                            {sig}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </TableCell>

                  <TableCell><BoolCell value={r.buySignal} /></TableCell>
                  <TableCell><BoolCell value={r.holdLong} /></TableCell>
                  <TableCell><BoolCell value={r.sellSignal} /></TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{r.days}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{r.profit}</TableCell>
                  <TableCell
                    className={cn(
                      'text-right font-mono tabular-nums',
                      r.tradePnl > 0 ? 'text-[var(--gain)]' : 'text-muted-foreground',
                    )}
                  >
                    {r.tradePnl.toFixed(1)}%
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums text-muted-foreground">
                    {r.kelly === null ? 'nan%' : `${r.kelly.toFixed(2)}%`}
                  </TableCell>

                  {/* Editable Description */}
                  <TableCell>
                    {editingDesc === r.id ? (
                      <div className="flex items-center gap-1.5">
                        <Input
                          autoFocus
                          value={r.description}
                          onChange={(e) => updateRow(r.id, { description: e.target.value })}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === 'Escape') setEditingDesc(null)
                          }}
                          className="h-8 text-xs"
                        />
                        <Button
                          size="icon"
                          variant="ghost"
                          className="size-7 shrink-0"
                          onClick={() => setEditingDesc(null)}
                          aria-label="Save description"
                        >
                          <Check className="size-3.5" />
                        </Button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        onClick={() => setEditingDesc(r.id)}
                        className="group flex w-full items-center gap-2 rounded px-1 py-0.5 text-left text-xs text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                      >
                        <span className="line-clamp-1">{r.description}</span>
                        <Pencil className="size-3 shrink-0 opacity-0 transition-opacity group-hover:opacity-60" />
                      </button>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        {filtered.length === 0 && (
          <p className="mt-6 text-center text-sm text-muted-foreground">
            No rows match your filters.
          </p>
        )}
      </Card>
    </div>
  )
}
