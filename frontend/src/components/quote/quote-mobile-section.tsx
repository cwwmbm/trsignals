'use client'

import { useQuery } from '@tanstack/react-query'
import { Loader2, RefreshCw } from 'lucide-react'
import { getQuote } from '@/api'
import { ChangeCell, formatMissingDay } from '@/components/quote/quote-metrics'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { formatMetric } from '@/lib/format-metric'
import { cn } from '@/lib/utils'

function QuoteMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <span className="font-mono text-sm tabular-nums">{value}</span>
    </div>
  )
}

export function QuoteMobileSection() {
  const { data, isLoading, isFetching, error, refetch } = useQuery({
    queryKey: ['quote'],
    queryFn: getQuote,
  })
  const quotes = data?.quotes ?? []
  const missingCount = quotes.reduce((count, row) => count + row.missing_days.length, 0)

  return (
    <div className="flex flex-col gap-3 pb-4">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-muted-foreground">
          {data?.as_of ? `As of ${data.as_of}` : 'Yahoo snapshot'}
          {missingCount > 0 ? (
            <span className="text-amber-700 dark:text-amber-400">
              {' '}
              · {missingCount} missing weekday{missingCount === 1 ? '' : 's'}
            </span>
          ) : quotes.length > 0 ? (
            <span> · last 5 days complete</span>
          ) : null}
        </p>
        <Button
          variant="outline"
          size="icon"
          className="size-9 shrink-0"
          onClick={() => refetch()}
          disabled={isFetching}
          aria-label="Refresh quotes"
        >
          {isFetching ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <RefreshCw className="size-4" />
          )}
        </Button>
      </div>

      {error ? (
        <Card className="border-destructive/50 bg-destructive/5 p-3">
          <pre className="whitespace-pre-wrap text-xs text-destructive">{String(error)}</pre>
        </Card>
      ) : null}

      {isLoading ? (
        <div className="flex items-center justify-center gap-2 py-12 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          Loading quotes…
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {quotes.map((row) => (
            <Card
              key={row.symbol}
              className={cn(
                'gap-3 p-3 shadow-none',
                row.missing_days.length > 0 && 'border-amber-500/40 bg-amber-500/5',
              )}
            >
              <div className="flex items-start justify-between gap-2">
                <p className="font-mono text-base font-semibold">{row.symbol}</p>
                <ChangeCell row={row} className="text-base font-semibold" />
              </div>
              <div className="grid grid-cols-4 gap-2">
                <QuoteMetric label="IBR" value={formatMetric(row.ibr)} />
                <QuoteMetric label="RSI2" value={formatMetric(row.rsi2)} />
                <QuoteMetric label="RSI5" value={formatMetric(row.rsi5)} />
                <QuoteMetric label="Stoch" value={formatMetric(row.stoch)} />
              </div>
              {row.missing_days.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                  {row.missing_days.map((day) => (
                    <span
                      key={day}
                      className="rounded-sm bg-amber-500/15 px-1.5 py-0.5 font-mono text-[10px] text-amber-800 dark:text-amber-300"
                    >
                      {formatMissingDay(day)}
                    </span>
                  ))}
                </div>
              ) : null}
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
