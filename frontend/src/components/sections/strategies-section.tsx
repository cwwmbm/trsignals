'use client'

import { useMemo, useState } from 'react'
import { ArrowUpRight, Search } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { STRATEGIES, type Strategy } from '@/lib/mock-data'

const STATUS_STYLES: Record<Strategy['status'], string> = {
  live: 'border-[var(--gain)]/40 text-[var(--gain)] bg-[var(--gain)]/10',
  draft: 'border-[var(--chart-4)]/40 text-[var(--chart-4)] bg-[var(--chart-4)]/10',
  archived: 'border-border text-muted-foreground bg-muted/40',
}

function Metric({ label, value, tone }: { label: string; value: string; tone?: 'gain' | 'loss' }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <span
        className={cn(
          'font-mono text-sm font-semibold tabular-nums',
          tone === 'gain' && 'text-[var(--gain)]',
          tone === 'loss' && 'text-[var(--loss)]',
        )}
      >
        {value}
      </span>
    </div>
  )
}

export function StrategiesSection({ onNewStrategy }: { onNewStrategy?: () => void }) {
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState<'all' | Strategy['status']>('all')

  const filtered = useMemo(
    () =>
      STRATEGIES.filter((s) => {
        if (status !== 'all' && s.status !== status) return false
        if (!query) return true
        const q = query.toLowerCase()
        return (
          s.name.toLowerCase().includes(q) ||
          s.signal.toLowerCase().includes(q) ||
          s.tags.some((t) => t.includes(q))
        )
      }),
    [query, status],
  )

  return (
    <div className="flex flex-col gap-6">
      <Card className="border-border/60 p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold">All strategies</h2>
            <p className="text-sm text-muted-foreground">
              {STRATEGIES.length} strategies · {STRATEGIES.filter((s) => s.status === 'live').length} live
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
          <div className="flex items-center gap-1.5">
            {(['all', 'live', 'draft', 'archived'] as const).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setStatus(s)}
                className={cn(
                  'rounded-md border px-3 py-1.5 text-xs font-medium capitalize transition-colors',
                  status === s
                    ? 'border-primary/50 bg-primary/15 text-foreground'
                    : 'border-border/60 text-muted-foreground hover:bg-muted',
                )}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {filtered.map((s) => (
          <Card key={s.id} className="group gap-4 border-border/60 p-5 transition-colors hover:border-primary/40">
            <div className="flex items-start justify-between gap-3">
              <div className="flex flex-col gap-1">
                <h3 className="font-semibold leading-tight">{s.name}</h3>
                <span className="font-mono text-xs text-muted-foreground">{s.signal}</span>
              </div>
              <Badge variant="outline" className={cn('shrink-0 capitalize', STATUS_STYLES[s.status])}>
                {s.status}
              </Badge>
            </div>

            <p className="line-clamp-2 text-xs leading-relaxed text-muted-foreground">
              {s.description}
            </p>

            <div className="grid grid-cols-3 gap-3 rounded-lg border border-border/60 bg-muted/20 p-3">
              <Metric label="CAGR" value={`${s.cagr}%`} tone="gain" />
              <Metric label="Sharpe" value={s.sharpe.toFixed(2)} />
              <Metric label="Max DD" value={`${s.maxDrawdown}%`} tone="loss" />
              <Metric label="Trades" value={`${s.trades}`} />
              <Metric label="Win %" value={`${s.pctPositive}`} />
              <div className="flex items-end justify-end">
                <Button variant="ghost" size="sm" className="h-7 gap-1 px-2 text-xs">
                  Backtest
                  <ArrowUpRight className="size-3.5" />
                </Button>
              </div>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {s.tags.map((t) => (
                <span key={t} className="rounded bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
                  {t}
                </span>
              ))}
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}
