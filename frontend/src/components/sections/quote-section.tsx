'use client'

import { useQuery } from '@tanstack/react-query'
import { Loader2, RefreshCw } from 'lucide-react'
import { getQuote } from '@/api'
import { ChangeCell, formatMissingDay } from '@/components/quote/quote-metrics'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { formatMetric } from '@/lib/format-metric'
import { cn } from '@/lib/utils'

const compactHead = 'h-7 px-1.5 py-0 text-[11px] font-medium'
const compactCell = 'px-1.5 py-0.5'

export function QuoteSection() {
  const { data, isLoading, isFetching, error, refetch } = useQuery({
    queryKey: ['quote'],
    queryFn: getQuote,
  })
  const quotes = data?.quotes ?? []
  const missingCount = quotes.reduce((count, row) => count + row.missing_days.length, 0)

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
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
        <div className="ml-auto">
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

      {error ? (
        <Card className="border-destructive/50 bg-destructive/5 p-2">
          <pre className="overflow-x-auto whitespace-pre-wrap text-[11px] text-destructive">
            {String(error)}
          </pre>
        </Card>
      ) : null}

      {isLoading ? (
        <div className="flex items-center justify-center gap-2 py-6 text-xs text-muted-foreground">
          <Loader2 className="size-3.5 animate-spin" />
          Loading quotes…
        </div>
      ) : (
        <div className="overflow-x-auto rounded-md border border-border/60">
          <Table className="text-xs">
            <TableHeader className="bg-muted/30">
              <TableRow className="hover:bg-transparent">
                <TableHead className={compactHead}>Symbol</TableHead>
                <TableHead className={cn(compactHead, 'text-right')}>Change</TableHead>
                <TableHead className={cn(compactHead, 'text-right')}>IBR</TableHead>
                <TableHead className={cn(compactHead, 'text-right')}>RSI2</TableHead>
                <TableHead className={cn(compactHead, 'text-right')}>RSI5</TableHead>
                <TableHead className={cn(compactHead, 'text-right')}>Stoch</TableHead>
                <TableHead className={compactHead}>Missing (last 5d)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {quotes.map((row) => (
                <TableRow
                  key={row.symbol}
                  className={cn(row.missing_days.length > 0 && 'bg-amber-500/10')}
                >
                  <TableCell className={cn(compactCell, 'font-mono font-medium')}>
                    {row.symbol}
                  </TableCell>
                  <TableCell className={cn(compactCell, 'text-right')}>
                    <ChangeCell row={row} />
                  </TableCell>
                  <TableCell className={cn(compactCell, 'text-right font-mono tabular-nums')}>
                    {formatMetric(row.ibr)}
                  </TableCell>
                  <TableCell className={cn(compactCell, 'text-right font-mono tabular-nums')}>
                    {formatMetric(row.rsi2)}
                  </TableCell>
                  <TableCell className={cn(compactCell, 'text-right font-mono tabular-nums')}>
                    {formatMetric(row.rsi5)}
                  </TableCell>
                  <TableCell className={cn(compactCell, 'text-right font-mono tabular-nums')}>
                    {formatMetric(row.stoch)}
                  </TableCell>
                  <TableCell className={compactCell}>
                    {row.missing_days.length === 0 ? (
                      <span className="text-muted-foreground">—</span>
                    ) : (
                      <span className="flex flex-wrap gap-1">
                        {row.missing_days.map((day) => (
                          <span
                            key={day}
                            className="rounded-sm bg-amber-500/15 px-1.5 py-0.5 font-mono text-[10px] text-amber-800 dark:text-amber-300"
                          >
                            {formatMissingDay(day)}
                          </span>
                        ))}
                      </span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  )
}
