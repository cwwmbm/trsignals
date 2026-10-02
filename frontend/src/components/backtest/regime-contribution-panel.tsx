'use client'

import { useState, type ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type {
  PortfolioRegimeResult,
  RegimeBook,
  RegimeBookMetrics,
  RegimeContributionCell,
  RegimeMetricDelta,
} from '@/api'
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
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { REGIME_DIMENSIONS, type RegimeDimension } from '@/lib/market-regimes'
import { cn } from '@/lib/utils'

type MetricId = keyof RegimeMetricDelta
type ViewId = 'quality' | 'contribution'

const METRICS: { id: MetricId; label: string }[] = [
  { id: 'regime_score', label: 'Regime score' },
  { id: 'avg_trade_return', label: 'Return' },
  { id: 'cagr', label: 'CAGR' },
  { id: 'sortino', label: 'Sortino' },
  { id: 'max_drawdown', label: 'Max drawdown' },
  { id: 'robustness', label: 'Robustness' },
  { id: 'exposure', label: 'Exposure' },
]

const compactHead = 'h-7 px-1.5 py-0 text-[11px] font-medium'
const compactCell = 'px-1.5 py-0.5 font-mono text-[11px] tabular-nums'

function isFiniteNumber(value: number | null | undefined): value is number {
  return value != null && Number.isFinite(value)
}

function formatScore(value: number, signed: boolean) {
  const rounded = Math.round(value)
  if (!signed || rounded <= 0) return String(rounded)
  return `+${rounded}`
}

function formatRatio(value: number, signed: boolean) {
  const text = value.toFixed(2)
  if (!signed || value <= 0) return text
  return `+${text}`
}

function formatPercent(value: number, digits: number, signed: boolean) {
  const text = `${(value * 100).toFixed(digits)}%`
  if (!signed || value <= 0) return text
  return `+${text}`
}

function formatMetric(metric: MetricId, value: number | null | undefined, signed: boolean) {
  if (!isFiniteNumber(value)) return '—'
  if (metric === 'regime_score') return formatScore(value, signed)
  if (metric === 'sortino' || metric === 'robustness') return formatRatio(value, signed)
  if (metric === 'max_drawdown') return formatPercent(value, 0, signed)
  return formatPercent(value, 1, signed)
}

function contributionTone(metric: MetricId, value: number | null | undefined) {
  if (!isFiniteNumber(value) || value === 0) return undefined
  const improved = metric === 'max_drawdown' ? value < 0 : value > 0
  return improved ? 'text-[var(--gain)]' : 'text-[var(--loss)]'
}

function bookMetrics(book: RegimeBook, dimensionId: string, column: string): RegimeBookMetrics | null {
  if (column === 'base') return book.base
  return book.regimes[dimensionId]?.[column] ?? null
}

function contributionCell(
  book: PortfolioRegimeResult['strategies'][number]['contribution'],
  dimensionId: string,
  column: string,
): RegimeContributionCell | null {
  if (column === 'base') return book.base
  return book.regimes[dimensionId]?.[column] ?? null
}

function HoverLine({
  label,
  metric,
  before,
  after,
}: {
  label: string
  metric: MetricId
  before: number | null
  after: number | null
}) {
  return (
    <div className="grid grid-cols-[8.75rem_1fr] gap-3">
      <span className="text-background/70">{label}</span>
      <span className="text-right whitespace-nowrap">
        {formatMetric(metric, before, false)} → {formatMetric(metric, after, false)}
      </span>
    </div>
  )
}

function ContributionHover({ cell }: { cell: RegimeContributionCell }) {
  return (
    <div className="w-64 space-y-1 font-mono text-[11px] leading-snug">
      <div>
        Contribution: {formatMetric('regime_score', cell.delta.regime_score, true)}
      </div>
      <div className="pt-1 text-background/70">Without this strategy:</div>
      <HoverLine
        label="Portfolio score"
        metric="regime_score"
        before={cell.with.regime_score}
        after={cell.without.regime_score}
      />
      <HoverLine label="Sortino" metric="sortino" before={cell.with.sortino} after={cell.without.sortino} />
      <HoverLine
        label="Max DD"
        metric="max_drawdown"
        before={cell.with.max_drawdown}
        after={cell.without.max_drawdown}
      />
      <HoverLine
        label="Avg / trade"
        metric="avg_trade_return"
        before={cell.with.avg_trade_return}
        after={cell.without.avg_trade_return}
      />
      <HoverLine label="CAGR" metric="cagr" before={cell.with.cagr} after={cell.without.cagr} />
      <HoverLine
        label="Robustness"
        metric="robustness"
        before={cell.with.robustness}
        after={cell.without.robustness}
      />
      <HoverLine
        label="Avg / exposure day"
        metric="exposure"
        before={cell.with.exposure}
        after={cell.without.exposure}
      />
    </div>
  )
}

function csvCell(value: string | number | null | undefined) {
  if (value == null || value === '') return ''
  const text = String(value)
  if (/[",\n]/.test(text)) return `"${text.replaceAll('"', '""')}"`
  return text
}

function rawMetric(value: number | null | undefined) {
  return isFiniteNumber(value) ? String(value) : ''
}

function regimeAnalysisCsv(result: PortfolioRegimeResult) {
  const headers = [
    'Strategy',
    'Regime',
    'Bucket',
    ...METRICS.flatMap((metric) => [`Quality ${metric.label}`, `Contribution ${metric.label}`]),
  ]
  const rows = [headers]
  const books = [
    { name: 'Portfolio', quality: result.portfolio, contribution: null },
    ...result.strategies.map((strategy) => ({
      name: strategy.strategy_name,
      quality: strategy.quality,
      contribution: strategy.contribution,
    })),
  ]
  for (const dimension of REGIME_DIMENSIONS) {
    const columns = [{ key: 'base', label: 'BASE' }, ...dimension.buckets]
    for (const column of columns) {
      for (const book of books) {
        const quality = bookMetrics(book.quality, dimension.id, column.key)
        const cell = book.contribution
          ? contributionCell(book.contribution, dimension.id, column.key)
          : null
        rows.push([
          book.name,
          dimension.title,
          column.label,
          ...METRICS.flatMap((metric) => [
            rawMetric(quality?.[metric.id]),
            rawMetric(cell?.delta[metric.id]),
          ]),
        ])
      }
    }
  }
  return `\uFEFF${rows.map((row) => row.map(csvCell).join(',')).join('\n')}`
}

function downloadRegimeCsv(result: PortfolioRegimeResult) {
  const blob = new Blob([regimeAnalysisCsv(result)], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'regime-analysis.csv'
  link.click()
  URL.revokeObjectURL(url)
}

function RegimeTable({
  result,
  dimension,
  metric,
  view,
}: {
  result: PortfolioRegimeResult
  dimension: RegimeDimension
  metric: MetricId
  view: ViewId
}) {
  const columns = [{ key: 'base', label: 'BASE' }, ...dimension.buckets]

  return (
    <div className="overflow-x-auto border-t border-border/60">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className={cn(compactHead, 'sticky left-0 bg-background text-left')} />
            {columns.map((column) => (
              <TableHead key={column.key} className={cn(compactHead, 'text-right')}>
                {column.label}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow>
            <TableCell className={cn(compactCell, 'sticky left-0 bg-background font-sans font-medium')}>
              Portfolio
            </TableCell>
            {columns.map((column) => {
              const metrics = bookMetrics(result.portfolio, dimension.id, column.key)
              return (
                <TableCell key={column.key} className={cn(compactCell, 'text-right')}>
                  {formatMetric(metric, metrics?.[metric], false)}
                </TableCell>
              )
            })}
          </TableRow>
          {result.strategies.map((strategy) => (
            <TableRow key={strategy.strategy_id}>
              <TableCell className={cn(compactCell, 'sticky left-0 bg-background font-sans font-medium')}>
                {strategy.strategy_name}
              </TableCell>
              {columns.map((column) => {
                if (view === 'quality') {
                  const metrics = bookMetrics(strategy.quality, dimension.id, column.key)
                  return (
                    <TableCell key={column.key} className={cn(compactCell, 'text-right')}>
                      {formatMetric(metric, metrics?.[metric], false)}
                    </TableCell>
                  )
                }
                const cell = contributionCell(strategy.contribution, dimension.id, column.key)
                const value = cell?.delta[metric]
                const text = formatMetric(metric, value, true)
                return (
                  <TableCell
                    key={column.key}
                    className={cn(compactCell, 'text-right', contributionTone(metric, value))}
                  >
                    {cell ? (
                      <Tooltip>
                        <TooltipTrigger
                          type="button"
                          className={cn('font-mono text-[11px] tabular-nums', contributionTone(metric, value))}
                        >
                          {text}
                        </TooltipTrigger>
                        <TooltipContent side="bottom" className="max-w-none">
                          <ContributionHover cell={cell} />
                        </TooltipContent>
                      </Tooltip>
                    ) : (
                      text
                    )}
                  </TableCell>
                )
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

const WEAK_SCORE = 40
const SYNERGY_MARGIN = 5
const RESCUE_CONTRIBUTION = 10

type RegimeBucketRef = {
  dimensionId: string
  dimensionTitle: string
  bucketKey: string
  bucketLabel: string
}

function regimeBucketRefs(): RegimeBucketRef[] {
  return REGIME_DIMENSIONS.flatMap((dimension) =>
    dimension.buckets.map((bucket) => ({
      dimensionId: dimension.id,
      dimensionTitle: dimension.title,
      bucketKey: bucket.key,
      bucketLabel: bucket.label,
    })),
  )
}

function bucketLabel(bucket: RegimeBucketRef) {
  return `${bucket.dimensionTitle} · ${bucket.bucketLabel}`
}

function RegimeInsights({
  result,
  onOpenRegime,
}: {
  result: PortfolioRegimeResult
  onOpenRegime: (dimensionId: string) => void
}) {
  const buckets = regimeBucketRefs()
  const contributions = result.strategies
    .map((strategy) => {
      let positive = 0
      for (const bucket of buckets) {
        const delta = contributionCell(strategy.contribution, bucket.dimensionId, bucket.bucketKey)?.delta
          .regime_score
        if (isFiniteNumber(delta) && delta > 0) positive += 1
      }
      return { id: strategy.strategy_id, name: strategy.strategy_name, positive, total: buckets.length }
    })
    .sort((a, b) => b.positive - a.positive || a.name.localeCompare(b.name))

  const blindSpots = buckets.filter((bucket) => {
    const portfolio = bookMetrics(result.portfolio, bucket.dimensionId, bucket.bucketKey)?.regime_score
    if (!isFiniteNumber(portfolio) || portfolio >= WEAK_SCORE || result.strategies.length === 0) return false
    return result.strategies.every((strategy) => {
      const quality = bookMetrics(strategy.quality, bucket.dimensionId, bucket.bucketKey)?.regime_score
      const delta = contributionCell(strategy.contribution, bucket.dimensionId, bucket.bucketKey)?.delta
        .regime_score
      const weak = isFiniteNumber(quality) && quality < WEAK_SCORE
      const negative = isFiniteNumber(delta) && delta < 0
      return weak || negative
    })
  })

  const synergies = buckets.filter((bucket) => {
    const portfolio = bookMetrics(result.portfolio, bucket.dimensionId, bucket.bucketKey)?.regime_score
    if (!isFiniteNumber(portfolio) || result.strategies.length === 0) return false
    return result.strategies.every((strategy) => {
      const quality = bookMetrics(strategy.quality, bucket.dimensionId, bucket.bucketKey)?.regime_score
      return isFiniteNumber(quality) && portfolio > quality + SYNERGY_MARGIN
    })
  })

  const strongest = result.strategies.map((strategy) => {
    const ranked = buckets
      .flatMap((bucket) => {
        const delta = contributionCell(strategy.contribution, bucket.dimensionId, bucket.bucketKey)?.delta
          .regime_score
        return isFiniteNumber(delta) ? [{ bucket, delta }] : []
      })
      .sort((a, b) => b.delta - a.delta)
      .slice(0, 3)
    return { id: strategy.strategy_id, name: strategy.strategy_name, ranked }
  })

  const rescues = buckets.flatMap((bucket) =>
    result.strategies.flatMap((strategy) => {
      const cell = contributionCell(strategy.contribution, bucket.dimensionId, bucket.bucketKey)
      const without = cell?.without.regime_score
      const delta = cell?.delta.regime_score
      if (!isFiniteNumber(without) || without >= WEAK_SCORE) return []
      if (!isFiniteNumber(delta) || delta <= RESCUE_CONTRIBUTION) return []
      return [{ bucket, strategyId: strategy.strategy_id, strategyName: strategy.strategy_name, delta }]
    }),
  )

  function RegimeChip({ bucket, detail }: { bucket: RegimeBucketRef; detail?: string }) {
    return (
      <button
        type="button"
        className="rounded border border-border/60 px-1.5 py-0.5 text-left hover:bg-muted"
        onClick={() => onOpenRegime(bucket.dimensionId)}
      >
        {bucketLabel(bucket)}
        {detail ? <span className="text-[var(--gain)]"> · {detail}</span> : null}
      </button>
    )
  }

  const blocks: { title: string; rule: string; body: ReactNode }[] = [
    {
      title: 'Strategy contributions',
      rule: 'Positive contribution in each regime bucket.',
      body: (
        <div className="flex flex-col gap-0.5">
          {contributions.map((item) => (
            <div key={item.id} className="flex items-baseline gap-3">
              <span className="font-medium">{item.name}</span>
              <span className="font-mono tabular-nums">
                {item.positive} / {item.total}
              </span>
            </div>
          ))}
        </div>
      ),
    },
    {
      title: 'Strongest 3',
      rule: 'Highest contribution buckets for each strategy.',
      body: (
        <div className="flex flex-col gap-2">
          {strongest.map((strategy) => (
            <div key={strategy.id}>
              <div className="font-medium">{strategy.name}</div>
              {strategy.ranked.length ? (
                <div className="mt-0.5 flex flex-col items-start gap-1">
                  {strategy.ranked.map((item) => (
                    <RegimeChip
                      key={`${item.bucket.dimensionId}-${item.bucket.bucketKey}`}
                      bucket={item.bucket}
                      detail={formatScore(item.delta, true)}
                    />
                  ))}
                </div>
              ) : (
                <span className="text-muted-foreground">None</span>
              )}
            </div>
          ))}
        </div>
      ),
    },
    {
      title: 'Shared blind spot',
      rule: 'Portfolio under 40, and every strategy is under 40 or contributes negatively.',
      body: blindSpots.length ? (
        <div className="flex flex-col items-start gap-1">
          {blindSpots.map((bucket) => (
            <RegimeChip key={`${bucket.dimensionId}-${bucket.bucketKey}`} bucket={bucket} />
          ))}
        </div>
      ) : (
        <span className="text-muted-foreground">None</span>
      ),
    },
    {
      title: 'Portfolio synergy',
      rule: 'Portfolio score exceeds every strategy by more than 5.',
      body: synergies.length ? (
        <div className="flex flex-col items-start gap-1">
          {synergies.map((bucket) => (
            <RegimeChip key={`${bucket.dimensionId}-${bucket.bucketKey}`} bucket={bucket} />
          ))}
        </div>
      ) : (
        <span className="text-muted-foreground">None</span>
      ),
    },
    {
      title: 'Specialist rescue',
      rule: 'Portfolio under 40 without that strategy, and its contribution is more than 10.',
      body: rescues.length ? (
        <div className="flex flex-col items-start gap-1">
          {rescues.map((rescue) => (
            <RegimeChip
              key={`${rescue.bucket.dimensionId}-${rescue.bucket.bucketKey}-${rescue.strategyId}`}
              bucket={rescue.bucket}
              detail={`${rescue.strategyName} ${formatScore(rescue.delta, true)}`}
            />
          ))}
        </div>
      ) : (
        <span className="text-muted-foreground">None</span>
      ),
    },
  ]

  return (
    <section className="overflow-hidden rounded-md border border-border/60 text-[11px]">
      <h4 className="border-b border-border/60 px-3 py-2 text-xs font-medium">Insights</h4>
      <div className="divide-y divide-border/60">
        {blocks.map((block) => (
          <div key={block.title} className="grid gap-1 px-3 py-2 sm:grid-cols-[14rem_1fr] sm:gap-4">
            <div>
              <div className="font-medium">{block.title}</div>
              <p className="text-muted-foreground">{block.rule}</p>
            </div>
            <div className="min-w-0">{block.body}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

export function RegimeContributionResults({ result }: { result: PortfolioRegimeResult }) {
  const [metric, setMetric] = useState<MetricId>('regime_score')
  const [view, setView] = useState<ViewId>('quality')
  const [openIds, setOpenIds] = useState<Set<string>>(
    () => new Set(REGIME_DIMENSIONS.map((item) => item.id)),
  )

  function toggle(id: string) {
    setOpenIds((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  function openRegime(id: string) {
    setOpenIds((current) => {
      if (current.has(id)) return current
      const next = new Set(current)
      next.add(id)
      return next
    })
    requestAnimationFrame(() => {
      document.getElementById(`regime-${id}`)?.scrollIntoView({ block: 'start' })
    })
  }

  return (
    <div className="space-y-2">
      <div className="sticky top-0 z-10 flex flex-wrap items-center gap-x-4 gap-y-2 bg-background py-1 text-[11px]">
        <label className="flex items-center gap-2">
          <span className="text-muted-foreground">Metric</span>
          <Select value={metric} onValueChange={(value) => value && setMetric(value as MetricId)}>
            <SelectTrigger size="sm" className="h-7 w-[9.5rem] text-xs" aria-label="Metric">
              <SelectValue>
                {(value: string) => METRICS.find((item) => item.id === value)?.label ?? value}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {METRICS.map((item) => (
                <SelectItem key={item.id} value={item.id}>
                  {item.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground">View</span>
          <div className="inline-flex rounded-md border border-border/60 p-0.5">
            {(
              [
                ['quality', 'Quality'],
                ['contribution', 'Contribution'],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={cn(
                  'h-6 rounded px-2 text-[11px]',
                  view === id ? 'bg-muted text-foreground' : 'text-muted-foreground hover:text-foreground',
                )}
                aria-pressed={view === id}
                onClick={() => setView(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="ml-auto h-7 px-2.5 text-[11px]"
          onClick={() => downloadRegimeCsv(result)}
        >
          Export CSV
        </Button>
      </div>

      <RegimeInsights result={result} onOpenRegime={openRegime} />

      <div className="flex flex-col gap-2">
        {REGIME_DIMENSIONS.map((dimension) => {
          const expanded = openIds.has(dimension.id)
          return (
            <div
              key={dimension.id}
              id={`regime-${dimension.id}`}
              className="scroll-mt-10 overflow-hidden rounded-md border border-border/60"
            >
              <button
                type="button"
                className="flex w-full items-center gap-2 bg-muted/20 px-3 py-2 text-left text-xs hover:bg-muted/30"
                aria-expanded={expanded}
                onClick={() => toggle(dimension.id)}
              >
                <ChevronDown
                  className={cn('size-3.5 shrink-0 transition-transform', !expanded && '-rotate-90')}
                />
                <span className="font-medium">{dimension.title}</span>
              </button>
              {expanded ? (
                <RegimeTable result={result} dimension={dimension} metric={metric} view={view} />
              ) : null}
            </div>
          )
        })}
      </div>
    </div>
  )
}
