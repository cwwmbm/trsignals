'use client'

import { useMutation } from '@tanstack/react-query'
import { ArrowDown, ArrowUp, ArrowUpDown, Info, Loader2 } from 'lucide-react'
import { Fragment, useEffect, useMemo, useState } from 'react'
import type {
  PortfolioSimulatePayload,
  PortfolioShapleyResult,
  StrategyContribution,
} from '@/api'
import { runPortfolioShapley } from '@/api'
import { Button } from '@/components/ui/button'
import {
  formatHoldingPercent,
  formatMetric,
  formatPpDelta,
  formatSignedDelta,
} from '@/lib/format-metric'
import { compareRowValues, type SortDirection } from '@/lib/sort-table-rows'
import { cn } from '@/lib/utils'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

const compactHead = 'h-7 px-1.5 py-0 text-[11px] font-medium'
const compactCell = 'px-1.5 py-0.5 align-top font-mono text-[11px] tabular-nums'

type SortKey =
  | 'strategy_name'
  | 'cagr_contribution_pp'
  | 'marginal_utility'
  | 'sharpe_delta'
  | 'sortino_delta'
  | 'calmar_delta'
  | 'max_drawdown_effect_pp'
  | 'ulcer_index_delta'
  | 'time_under_water_delta_pp'
  | 'added_exposure_pp'
  | 'exposure_adjusted_return_delta'
  | 'marginal_cagr_per_10pp_exposure'
  | 'added_portfolio_trades'
  | 'unique_holding_percent'
  | 'redundant_holding_percent'
  | 'abs_cagr'

const COLUMNS: Array<{
  key: SortKey
  label: string
  align?: 'left' | 'right'
  tooltip: string
  invertTone?: boolean
  format: 'pp' | 'signed' | 'holding' | 'trades' | 'name'
}> = [
  {
    key: 'strategy_name',
    label: 'Strategy',
    align: 'left',
    tooltip: 'Selected strategy. Click a row for full vs without-strategy metrics and holding overlap.',
    format: 'name',
  },
  {
    key: 'cagr_contribution_pp',
    label: 'CAGR contribution',
    align: 'right',
    tooltip:
      'Full portfolio CAGR minus CAGR without this strategy (percentage points). Positive means the strategy improved CAGR.',
    format: 'pp',
  },
  {
    key: 'marginal_utility',
    label: 'Marginal utility',
    align: 'right',
    tooltip:
      'utility(full) − utility(without). Utility = 1×CAGR% + 4×Sharpe − 0.5×MaxDD% − 0.1×TimeInMarket% − 0.02×TradesPerYear. Positive means the strategy improved scored utility.',
    format: 'signed',
  },
  {
    key: 'sharpe_delta',
    label: 'Δ Sharpe',
    align: 'right',
    tooltip:
      'Full portfolio Sharpe minus Sharpe without this strategy. Positive means the strategy improved risk-adjusted return.',
    format: 'signed',
  },
  {
    key: 'sortino_delta',
    label: 'Δ Sortino',
    align: 'right',
    tooltip:
      'Full portfolio Sortino minus Sortino without this strategy. Positive means improved downside risk-adjusted return.',
    format: 'signed',
  },
  {
    key: 'calmar_delta',
    label: 'Δ Calmar',
    align: 'right',
    tooltip:
      'Full portfolio Calmar (CAGR% / MaxDD%) minus Calmar without this strategy. Positive means improvement.',
    format: 'signed',
  },
  {
    key: 'max_drawdown_effect_pp',
    label: 'Max-DD effect',
    align: 'right',
    tooltip: 'Positive values mean worse max drawdown; negative values mean improvement.',
    invertTone: true,
    format: 'pp',
  },
  {
    key: 'ulcer_index_delta',
    label: 'Δ Ulcer',
    align: 'right',
    tooltip:
      'Change in ulcer index (sqrt of mean squared percent drawdown). Positive means worse pain from drawdowns.',
    invertTone: true,
    format: 'signed',
  },
  {
    key: 'time_under_water_delta_pp',
    label: 'Δ Time under water',
    align: 'right',
    tooltip:
      'Change in percent of days below peak equity (percentage points). Positive means more time underwater.',
    invertTone: true,
    format: 'pp',
  },
  {
    key: 'added_exposure_pp',
    label: 'Added exposure',
    align: 'right',
    tooltip:
      'Change in time in market (percentage points) when this strategy is included. Positive means more days invested.',
    format: 'pp',
  },
  {
    key: 'exposure_adjusted_return_delta',
    label: 'Δ Exp-adj return',
    align: 'right',
    tooltip:
      'Change in exposure-adjusted return (CAGR% ÷ time-in-market%). Positive means more return per unit of market exposure.',
    format: 'signed',
  },
  {
    key: 'marginal_cagr_per_10pp_exposure',
    label: 'CAGR / 10% exp',
    align: 'right',
    tooltip:
      'Incremental CAGR ÷ incremental exposure × 10. CAGR percentage points gained per +10pp of time in market.',
    format: 'pp',
  },
  {
    key: 'added_portfolio_trades',
    label: 'Added portfolio trades',
    align: 'right',
    tooltip:
      'Change in portfolio-level trade count when this strategy is included. Not the same as that strategy’s own trade count.',
    format: 'trades',
  },
  {
    key: 'unique_holding_percent',
    label: 'Unique exposure',
    align: 'right',
    tooltip:
      'Share of this strategy’s holding days when the rest of the portfolio is flat. High unique means it adds standalone exposure.',
    format: 'holding',
  },
  {
    key: 'redundant_holding_percent',
    label: 'Redundant exposure',
    align: 'right',
    tooltip:
      'Share of this strategy’s holding days that overlap with the portfolio without it. High redundant means it mostly overlaps others.',
    format: 'holding',
  },
]

function sortValue(row: StrategyContribution, key: SortKey): unknown {
  if (key === 'abs_cagr') {
    const value = row.cagr_contribution_pp
    return value === null || value === undefined ? null : Math.abs(value)
  }
  return row[key]
}

function deltaTone(
  value: number | null,
  options?: { invert?: boolean },
): 'neutral' | 'gain' | 'loss' {
  if (value === null || value === 0) return 'neutral'
  const positiveIsGood = !options?.invert
  if (value > 0) return positiveIsGood ? 'gain' : 'loss'
  return positiveIsGood ? 'loss' : 'gain'
}

function toneClass(tone: 'neutral' | 'gain' | 'loss') {
  if (tone === 'gain') return 'text-[var(--gain)]'
  if (tone === 'loss') return 'text-[var(--loss)]'
  return ''
}

function formatCell(
  row: StrategyContribution,
  column: (typeof COLUMNS)[number],
): string {
  const value = row[column.key as keyof StrategyContribution]
  if (column.format === 'name') return String(value ?? '')
  if (column.format === 'pp') return formatPpDelta(value)
  if (column.format === 'signed') return formatSignedDelta(value)
  if (column.format === 'holding') return formatHoldingPercent(value)
  if (column.format === 'trades') {
    if (typeof value !== 'number') return '-'
    return `${value > 0 ? '+' : ''}${value}`
  }
  return '-'
}

function MetricCompareRow({
  label,
  full,
  without,
  contribution,
  contributionLabel = 'Contribution',
  worsePositive = false,
}: {
  label: string
  full: string
  without: string
  contribution: string
  contributionLabel?: string
  worsePositive?: boolean
}) {
  return (
    <div className="grid gap-0.5 border-b border-border/40 py-1.5 last:border-b-0 sm:grid-cols-[9rem_1fr_1fr]">
      <div className="text-[11px] font-medium text-muted-foreground">{label}</div>
      <div className="font-mono text-[11px] tabular-nums">
        {full} <span className="text-muted-foreground">vs</span> {without}
      </div>
      <div
        className={cn(
          'font-mono text-[11px] tabular-nums',
          worsePositive ? 'text-[var(--loss)]' : '',
        )}
      >
        <span className="text-muted-foreground">{contributionLabel}: </span>
        {contribution}
      </div>
    </div>
  )
}

function ContributionDetails({ row }: { row: StrategyContribution }) {
  const maxDdWorse =
    row.max_drawdown_effect_pp !== null && row.max_drawdown_effect_pp > 0
  const ulcerWorse = row.ulcer_index_delta !== null && row.ulcer_index_delta > 0
  const tuwWorse =
    row.time_under_water_delta_pp !== null && row.time_under_water_delta_pp > 0
  return (
    <div className="space-y-2 border-t border-border/50 bg-muted/20 px-3 py-2">
      <p className="text-xs font-medium">
        Full portfolio vs without &ldquo;{row.strategy_name}&rdquo;
      </p>
      <div>
        <MetricCompareRow
          label="CAGR"
          full={formatMetric(row.full.cagr_percent, 'cagr_percent')}
          without={formatMetric(row.without_strategy.cagr_percent, 'cagr_percent')}
          contribution={formatPpDelta(row.cagr_contribution_pp)}
        />
        <MetricCompareRow
          label="Utility"
          full={formatMetric(row.full.utility)}
          without={formatMetric(row.without_strategy.utility)}
          contribution={formatSignedDelta(row.marginal_utility)}
          contributionLabel="Marginal"
        />
        <MetricCompareRow
          label="Sharpe"
          full={formatMetric(row.full.sharpe)}
          without={formatMetric(row.without_strategy.sharpe)}
          contribution={formatSignedDelta(row.sharpe_delta)}
          contributionLabel="Delta"
        />
        <MetricCompareRow
          label="Sortino"
          full={formatMetric(row.full.sortino)}
          without={formatMetric(row.without_strategy.sortino)}
          contribution={formatSignedDelta(row.sortino_delta)}
          contributionLabel="Delta"
        />
        <MetricCompareRow
          label="Calmar"
          full={formatMetric(row.full.calmar)}
          without={formatMetric(row.without_strategy.calmar)}
          contribution={formatSignedDelta(row.calmar_delta)}
          contributionLabel="Delta"
        />
        <MetricCompareRow
          label="Max drawdown"
          full={formatMetric(row.full.max_drawdown, 'max_drawdown')}
          without={formatMetric(row.without_strategy.max_drawdown, 'max_drawdown')}
          contribution={`${formatPpDelta(row.max_drawdown_effect_pp)}${maxDdWorse ? ' worse' : row.max_drawdown_effect_pp !== null && row.max_drawdown_effect_pp < 0 ? ' better' : ''}`}
          contributionLabel="Effect"
          worsePositive={maxDdWorse}
        />
        <MetricCompareRow
          label="Ulcer index"
          full={formatMetric(row.full.ulcer_index)}
          without={formatMetric(row.without_strategy.ulcer_index)}
          contribution={`${formatSignedDelta(row.ulcer_index_delta)}${ulcerWorse ? ' worse' : ''}`}
          contributionLabel="Delta"
          worsePositive={ulcerWorse}
        />
        <MetricCompareRow
          label="Time under water"
          full={
            row.full.time_under_water_percent === null
              ? '-'
              : `${row.full.time_under_water_percent.toFixed(1)}%`
          }
          without={
            row.without_strategy.time_under_water_percent === null
              ? '-'
              : `${row.without_strategy.time_under_water_percent.toFixed(1)}%`
          }
          contribution={`${formatPpDelta(row.time_under_water_delta_pp)}${tuwWorse ? ' worse' : ''}`}
          contributionLabel="Delta"
          worsePositive={tuwWorse}
        />
        <MetricCompareRow
          label="Time in market"
          full={
            row.full.time_in_market_percent === null
              ? '-'
              : `${row.full.time_in_market_percent.toFixed(1)}%`
          }
          without={
            row.without_strategy.time_in_market_percent === null
              ? '-'
              : `${row.without_strategy.time_in_market_percent.toFixed(1)}%`
          }
          contribution={formatPpDelta(row.added_exposure_pp)}
          contributionLabel="Added exposure"
        />
        <MetricCompareRow
          label="Exp-adj return"
          full={formatMetric(row.full.exposure_adjusted_return)}
          without={formatMetric(row.without_strategy.exposure_adjusted_return)}
          contribution={formatSignedDelta(row.exposure_adjusted_return_delta)}
          contributionLabel="Delta"
        />
        <MetricCompareRow
          label="CAGR / 10% exp"
          full="—"
          without="—"
          contribution={formatPpDelta(row.marginal_cagr_per_10pp_exposure)}
          contributionLabel="Marginal"
        />
        <MetricCompareRow
          label="Portfolio trades"
          full={formatMetric(row.full.trades, 'Trades')}
          without={formatMetric(row.without_strategy.trades, 'Trades')}
          contribution={
            row.added_portfolio_trades === null
              ? '-'
              : `${row.added_portfolio_trades > 0 ? '+' : ''}${row.added_portfolio_trades}`
          }
          contributionLabel="Added trades"
        />
        <MetricCompareRow
          label="Final equity"
          full={formatMetric(row.full.rolling_pnl)}
          without={formatMetric(row.without_strategy.rolling_pnl)}
          contribution={
            row.final_equity_delta === null
              ? '-'
              : `${row.final_equity_delta > 0 ? '+' : ''}${formatMetric(row.final_equity_delta)}`
          }
          contributionLabel="Delta"
        />
      </div>
      <div className="pt-1 text-[11px]">
        <span className="font-medium text-muted-foreground">Holding overlap </span>
        <span className="font-mono tabular-nums">
          {formatHoldingPercent(row.unique_holding_percent)} unique
          {' · '}
          {formatHoldingPercent(row.redundant_holding_percent)} redundant
          {' · '}
          {row.unique_holding_days}/{row.candidate_holding_days} unique days
          {' · '}
          {row.overlapping_holding_days}/{row.candidate_holding_days} overlapping days
        </span>
      </div>
    </div>
  )
}

const SHAPLEY_COLUMNS = COLUMNS.filter(
  (column) =>
    column.key !== 'unique_holding_percent' && column.key !== 'redundant_holding_percent',
)

function ContributionTable({
  rows,
  columns,
  expandable = false,
}: {
  rows: StrategyContribution[]
  columns: typeof COLUMNS
  expandable?: boolean
}) {
  const [sort, setSort] = useState<{ key: SortKey; direction: SortDirection }>({
    key: 'abs_cagr',
    direction: 'desc',
  })
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const sorted = useMemo(() => {
    return [...rows].sort((left, right) =>
      compareRowValues(sortValue(left, sort.key), sortValue(right, sort.key), sort.direction),
    )
  }, [rows, sort])

  function toggleSort(key: SortKey) {
    setSort((current) => {
      const normalizedCurrent =
        current.key === 'abs_cagr' && key === 'cagr_contribution_pp'
          ? 'cagr_contribution_pp'
          : current.key
      if (normalizedCurrent === key || (current.key === 'abs_cagr' && key === 'cagr_contribution_pp')) {
        if (current.key === 'abs_cagr' && key === 'cagr_contribution_pp') {
          return { key: 'cagr_contribution_pp', direction: 'desc' }
        }
        return { key, direction: current.direction === 'desc' ? 'asc' : 'desc' }
      }
      return { key, direction: key === 'strategy_name' ? 'asc' : 'desc' }
    })
  }

  function SortIcon({ column }: { column: SortKey }) {
    const active =
      sort.key === column || (sort.key === 'abs_cagr' && column === 'cagr_contribution_pp')
    if (!active) return <ArrowUpDown className="size-3 opacity-40" />
    return sort.direction === 'desc' ? (
      <ArrowDown className="size-3" />
    ) : (
      <ArrowUp className="size-3" />
    )
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            {columns.map((column) => (
              <TableHead
                key={column.key}
                className={cn(
                  compactHead,
                  column.align === 'right' && 'text-right',
                  'sticky top-0 z-10 cursor-pointer select-none bg-card',
                )}
                onClick={() => toggleSort(column.key)}
              >
                <span
                  className={cn(
                    'inline-flex items-center gap-1',
                    column.align === 'right' && 'justify-end',
                  )}
                >
                  {column.label}
                  <Tooltip>
                    <TooltipTrigger
                      className="inline-flex"
                      onClick={(event) => event.stopPropagation()}
                    >
                      <Info className="size-3 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs text-xs">{column.tooltip}</TooltipContent>
                  </Tooltip>
                  <SortIcon column={column.key} />
                </span>
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((row) => {
            const expanded = expandable && expandedId === row.strategy_id
            return (
              <Fragment key={row.strategy_id}>
                <TableRow
                  className={cn(expandable && 'cursor-pointer', expanded && 'bg-muted/30')}
                  onClick={
                    expandable
                      ? () =>
                          setExpandedId((current) =>
                            current === row.strategy_id ? null : row.strategy_id,
                          )
                      : undefined
                  }
                >
                  {columns.map((column) => {
                    const raw = row[column.key as keyof StrategyContribution]
                    const numeric = typeof raw === 'number' ? raw : null
                    const tone =
                      column.format === 'name' ||
                      column.format === 'holding' ||
                      column.format === 'trades'
                        ? 'neutral'
                        : deltaTone(numeric, { invert: column.invertTone })
                    return (
                      <TableCell
                        key={column.key}
                        className={cn(
                          compactCell,
                          column.key === 'strategy_name' && 'font-sans font-medium',
                          column.align === 'right' && 'text-right',
                          toneClass(tone),
                        )}
                      >
                        {formatCell(row, column)}
                      </TableCell>
                    )
                  })}
                </TableRow>
                {expanded ? (
                  <TableRow className="hover:bg-transparent">
                    <TableCell colSpan={columns.length} className="p-0">
                      <ContributionDetails row={row} />
                    </TableCell>
                  </TableRow>
                ) : null}
              </Fragment>
            )
          })}
        </TableBody>
      </Table>
    </div>
  )
}

export function ContributionPanel({
  rows,
  shapleyRequest,
  className,
}: {
  rows: StrategyContribution[]
  shapleyRequest?: PortfolioSimulatePayload | null
  className?: string
}) {
  const [shapleyResult, setShapleyResult] = useState<PortfolioShapleyResult | null>(null)

  useEffect(() => {
    setShapleyResult(null)
  }, [rows, shapleyRequest])

  const shapleyMutation = useMutation({
    mutationFn: runPortfolioShapley,
    onSuccess: (data) => setShapleyResult(data),
  })

  if (!rows.length) {
    return <p className="text-xs text-muted-foreground">No contribution data.</p>
  }

  const canRunShapley = Boolean(shapleyRequest?.strategy_ids?.length)

  return (
    <TooltipProvider delay={0}>
      <div
        className={cn(
          'min-h-0 overflow-auto rounded-md border border-border/60',
          className,
        )}
      >
        <div className="space-y-3 p-3">
          <div className="space-y-1.5">
            <h3 className="text-xs font-medium">Leave One Out Analysis</h3>
            <ContributionTable rows={rows} columns={COLUMNS} expandable />
          </div>

          <div className="space-y-2 border-t border-border/60 pt-3">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div className="min-w-0 space-y-0.5">
                <h3 className="text-xs font-medium">Shapley analysis</h3>
                <p className="text-[11px] text-muted-foreground">
                  Values average each strategy&apos;s marginal contribution across random
                  portfolio orderings (exact when the strategy count is small).
                </p>
              </div>
              <Button
                type="button"
                size="sm"
                variant="outline"
                className="h-7 shrink-0 px-2.5 text-[11px]"
                disabled={!canRunShapley || shapleyMutation.isPending}
                onClick={() => {
                  if (!shapleyRequest) return
                  shapleyMutation.mutate(shapleyRequest)
                }}
              >
                {shapleyMutation.isPending ? (
                  <>
                    <Loader2 className="size-3.5 animate-spin" />
                    Calculating…
                  </>
                ) : (
                  'Calculate Shapley'
                )}
              </Button>
            </div>
            {shapleyMutation.isError ? (
              <p className="text-[11px] text-[var(--loss)]">
                {shapleyMutation.error instanceof Error
                  ? shapleyMutation.error.message
                  : 'Shapley calculation failed.'}
              </p>
            ) : null}
            {shapleyResult ? (
              <div className="space-y-1.5">
                <p className="font-mono text-[10px] text-muted-foreground">
                  {shapleyResult.exact ? 'Exact' : 'Approximate'} · {shapleyResult.samples_used}{' '}
                  {shapleyResult.exact ? 'permutations' : 'samples'}
                </p>
                <ContributionTable
                  rows={shapleyResult.shapley}
                  columns={SHAPLEY_COLUMNS}
                  expandable={false}
                />
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </TooltipProvider>
  )
}
