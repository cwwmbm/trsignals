'use client'

import { ArrowDown, ArrowUp, ArrowUpDown, ChevronDown, ChevronRight, Info } from 'lucide-react'
import { Fragment, useMemo, useState } from 'react'
import type {
  BaselineDrawdownEpisode,
  BelowSmaEpisode,
  PortfolioRegimeResult,
  RegimeDimension,
  RegimeEvidence,
  RegimeStateRow,
  StrategyRegimeContribution,
} from '@/api'
import {
  formatHoldingPercent,
  formatSignedPercent,
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
  TooltipTrigger,
} from '@/components/ui/tooltip'

const compactHead = 'h-7 px-1.5 py-0 text-[11px] font-medium'
const compactCell = 'px-1.5 py-0.5 align-top font-mono text-[11px] tabular-nums'

const DIMENSION_ORDER: RegimeDimension[] = [
  'spy_trend',
  'spy_volatility',
  'baseline_drawdown',
  'baseline_stress',
]

const DIMENSION_LABELS: Record<RegimeDimension, string> = {
  spy_trend: 'SPY market trend',
  spy_volatility: 'SPY realized volatility',
  baseline_drawdown: 'Baseline portfolio drawdown',
  baseline_stress: 'Baseline stress days',
}

const STATE_LABELS: Record<string, string> = {
  above: 'Above SMA200',
  below: 'Below SMA200',
  low: 'Low volatility',
  normal: 'Normal volatility',
  high: 'High volatility',
  drawdown: 'Baseline drawdown',
  severe_drawdown: 'Baseline severe drawdown',
  stress: 'Baseline stress days',
  non_stress: 'Baseline non-stress days',
  flat: 'Baseline idle (both flat)',
}

// baseline_drawdown "normal" overlaps with vol "normal" — disambiguate by dimension
function stateLabel(dimension: RegimeDimension, state: string): string {
  if (dimension === 'baseline_drawdown' && state === 'normal') {
    return 'Baseline normal'
  }
  if (dimension === 'spy_volatility' && state === 'normal') {
    return 'Normal volatility'
  }
  return STATE_LABELS[state] ?? state
}

function toneClass(value: number | null | undefined, invert = false): string | undefined {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) return undefined
  const positive = invert ? value < 0 : value > 0
  return positive ? 'text-[var(--gain)]' : 'text-[var(--loss)]'
}

/** Evidence emphasis without overriding gain/loss color classes. */
function evidenceStyle(evidence: RegimeEvidence): string {
  if (evidence === 'strong') return ''
  if (evidence === 'moderate') return 'opacity-90'
  return 'italic opacity-75'
}

function evidenceLabelClass(evidence: RegimeEvidence): string {
  if (evidence === 'strong') return 'text-foreground'
  if (evidence === 'moderate') return 'text-muted-foreground'
  return 'text-muted-foreground/80 italic'
}

function HeaderTip({ label, tip }: { label: string; tip: string }) {
  return (
    <span className="inline-flex items-center gap-0.5">
      {label}
      <Tooltip>
        <TooltipTrigger
          className="inline-flex text-muted-foreground hover:text-foreground"
          onClick={(event) => event.stopPropagation()}
        >
          <Info className="size-3" />
        </TooltipTrigger>
        <TooltipContent className="max-w-xs text-[11px] leading-snug">{tip}</TooltipContent>
      </Tooltip>
    </span>
  )
}

function RegimeSummaryTable({ regimes }: { regimes: RegimeStateRow[] }) {
  const grouped = useMemo(() => {
    return DIMENSION_ORDER.map((dimension) => ({
      dimension,
      rows: regimes.filter((row) => row.dimension === dimension),
    })).filter((group) => group.rows.length > 0)
  }, [regimes])

  return (
    <div className="space-y-2">
      <p className="text-[10px] text-muted-foreground">
        Regime dimensions overlap — a day can be below SMA200, high-vol, in drawdown, and a stress
        day at once. Do not sum rows across dimensions. Compounded regime returns combine
        multiplicatively, not by adding percentages. Baseline drawdown is vs a running equity peak
        (can remain severe for years).
      </p>
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className={cn(compactHead, 'text-left')}>
              <HeaderTip
                label="Regime"
                tip="Market regimes always use SPY closes, regardless of traded symbols. Baseline regimes use the leave-one-out portfolio without this strategy."
              />
            </TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>
              <HeaderTip
                label="Marginal return"
                tip="exp(Σ marginal log returns) − 1 over regime days. Not CAGR. Over long samples this can be thousands of percent — compare Ann. cond. or Contrib share across regimes. Regimes combine multiplicatively across dimensions."
              />
            </TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>
              <HeaderTip
                label="Ann. cond."
                tip="Annualized conditional contribution rate = 252 × mean daily marginal log return. Best for comparing stress vs non-stress (horizon-normalized)."
              />
            </TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>
              <HeaderTip
                label="Contrib share"
                tip="Regime ΣM ÷ total ΣM across all eligible days. Shares can exceed 0–100% when regimes offset. Null when total ΣM ≈ 0."
              />
            </TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>
              <HeaderTip
                label="Added exposure"
                tip="Share of regime days where the full portfolio is invested and the baseline is flat."
              />
            </TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>
              <HeaderTip
                label="M / unique-exp day"
                tip="Diagnostic: Σ all regime marginal log return ÷ unique-exposure days only. Includes extensions/bridges in the numerator — not an efficiency metric."
              />
            </TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>
              <HeaderTip
                label="ES effect"
                tip="Expected-shortfall improvement: ES(full) − ES(baseline) on worst 10% daily simple returns. Positive means better (less negative) tails."
              />
            </TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>
              <HeaderTip
                label="Worst port. Δ"
                tip="Unpaired: min(full returns) − min(baseline returns) in the regime (may be different dates)."
              />
            </TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>
              <HeaderTip
                label="Worst marg. day"
                tip="Paired: min(R_full − R_base) on the same regime days. Strategy’s worst single-day marginal effect."
              />
            </TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>
              <HeaderTip
                label="+M rate"
                tip="Share of regime days with M_t > 0 (raw sign, not economic materiality)."
              />
            </TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>
              <HeaderTip
                label="+M eff."
                tip="Share of regime days with M_t above the material marginal-return threshold (economic materiality)."
              />
            </TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>Days</TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>
              <HeaderTip
                label="Eff. days"
                tip="Days where |M_t| exceeds the material threshold. Evidence uses this count."
              />
            </TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>Episodes</TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>
              <HeaderTip
                label="Evidence"
                tip="Requires regime days, candidate-active days, and effective contribution days. Limited/insufficient still shows metrics."
              />
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {grouped.map((group) => (
            <Fragment key={group.dimension}>
              <TableRow className="hover:bg-transparent">
                <TableCell
                  colSpan={15}
                  className="bg-muted/40 px-1.5 py-1 text-[10px] font-medium text-muted-foreground"
                >
                  {DIMENSION_LABELS[group.dimension]}
                  {group.dimension === 'baseline_stress' ? (
                    <span className="ml-1 font-normal">
                      (Q10 from baseline invested days; unique-exposure days classified by
                      full-portfolio return vs that Q10 — needed for OR overlays where
                      contribution is ~0 while baseline is already invested; idle = both flat)
                    </span>
                  ) : null}
                  {group.dimension === 'baseline_drawdown' ? (
                    <span className="ml-1 font-normal">
                      (vs running peak; severe episode count = troughs ≤ −15%)
                    </span>
                  ) : null}
                </TableCell>
              </TableRow>
              {group.rows.map((row) => (
                <TableRow key={`${row.dimension}-${row.state}`} className="hover:bg-muted/20">
                  <TableCell className={cn(compactCell, 'text-left font-sans')}>
                    {stateLabel(row.dimension, row.state)}
                    {row.unavailable_reason ? (
                      <span className="ml-1 text-muted-foreground">({row.unavailable_reason})</span>
                    ) : null}
                  </TableCell>
                  <TableCell
                    className={cn(
                      compactCell,
                      'text-right',
                      evidenceStyle(row.evidence),
                      toneClass(row.compounded_marginal_return),
                    )}
                  >
                    {formatSignedPercent(row.compounded_marginal_return)}
                  </TableCell>
                  <TableCell
                    className={cn(
                      compactCell,
                      'text-right',
                      evidenceStyle(row.evidence),
                      toneClass(row.annualized_conditional_contribution_rate),
                    )}
                  >
                    {formatSignedPercent(row.annualized_conditional_contribution_rate, 2)}
                  </TableCell>
                  <TableCell
                    className={cn(
                      compactCell,
                      'text-right',
                      evidenceStyle(row.evidence),
                      toneClass(row.contribution_share),
                    )}
                  >
                    {formatHoldingPercent(row.contribution_share, 0)}
                  </TableCell>
                  <TableCell className={cn(compactCell, 'text-right', evidenceLabelClass(row.evidence))}>
                    {formatHoldingPercent(row.added_exposure_percent, 0)}
                  </TableCell>
                  <TableCell
                    className={cn(
                      compactCell,
                      'text-right',
                      evidenceStyle(row.evidence),
                      toneClass(row.marginal_return_per_added_exposure_day),
                    )}
                  >
                    {row.marginal_return_per_added_exposure_day === null ||
                    row.marginal_return_per_added_exposure_day === undefined
                      ? '—'
                      : formatSignedPercent(row.marginal_return_per_added_exposure_day, 3)}
                  </TableCell>
                  <TableCell
                    className={cn(
                      compactCell,
                      'text-right',
                      evidenceStyle(row.evidence),
                      toneClass(row.expected_shortfall_effect),
                    )}
                  >
                    {formatSignedPercent(row.expected_shortfall_effect, 2)}
                  </TableCell>
                  <TableCell
                    className={cn(
                      compactCell,
                      'text-right',
                      evidenceStyle(row.evidence),
                      toneClass(row.worst_day_effect),
                    )}
                  >
                    {formatSignedPercent(row.worst_day_effect, 2)}
                  </TableCell>
                  <TableCell
                    className={cn(
                      compactCell,
                      'text-right',
                      evidenceStyle(row.evidence),
                      toneClass(row.worst_marginal_day),
                    )}
                  >
                    {formatSignedPercent(row.worst_marginal_day, 2)}
                  </TableCell>
                  <TableCell className={cn(compactCell, 'text-right', evidenceLabelClass(row.evidence))}>
                    {formatHoldingPercent(row.positive_marginal_day_rate, 0)}
                  </TableCell>
                  <TableCell className={cn(compactCell, 'text-right', evidenceLabelClass(row.evidence))}>
                    {formatHoldingPercent(row.positive_effective_day_rate, 0)}
                  </TableCell>
                  <TableCell className={cn(compactCell, 'text-right', evidenceLabelClass(row.evidence))}>
                    {row.regime_days}
                  </TableCell>
                  <TableCell className={cn(compactCell, 'text-right', evidenceLabelClass(row.evidence))}>
                    {row.effective_contribution_days}
                  </TableCell>
                  <TableCell className={cn(compactCell, 'text-right', evidenceLabelClass(row.evidence))}>
                    {row.episode_count ?? '—'}
                  </TableCell>
                  <TableCell className={cn(compactCell, 'text-right capitalize', evidenceLabelClass(row.evidence))}>
                    {row.evidence}
                  </TableCell>
                </TableRow>
              ))}
            </Fragment>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

type EpisodeSortKey =
  | 'start_date'
  | 'compounded_marginal_return'
  | 'max_drawdown_improvement'
  | 'trough_improvement'
  | 'recovery_acceleration'
  | 'trading_days'

function SortableHead({
  label,
  tip,
  active,
  direction,
  onClick,
}: {
  label: string
  tip?: string
  active: boolean
  direction: SortDirection
  onClick: () => void
}) {
  const Icon = !active ? ArrowUpDown : direction === 'asc' ? ArrowUp : ArrowDown
  return (
    <TableHead className={cn(compactHead, 'text-right')}>
      <button
        type="button"
        className="inline-flex items-center gap-0.5 hover:text-foreground"
        onClick={onClick}
      >
        {tip ? <HeaderTip label={label} tip={tip} /> : label}
        <Icon className="size-3 opacity-60" />
      </button>
    </TableHead>
  )
}

function BelowSmaEpisodeTable({ episodes }: { episodes: BelowSmaEpisode[] }) {
  const [sort, setSort] = useState<{ key: EpisodeSortKey; direction: SortDirection }>({
    key: 'start_date',
    direction: 'asc',
  })

  const sorted = useMemo(() => {
    const copy = [...episodes]
    copy.sort((a, b) =>
      compareRowValues(
        a[sort.key as keyof BelowSmaEpisode] as never,
        b[sort.key as keyof BelowSmaEpisode] as never,
        sort.direction,
      ),
    )
    return copy
  }, [episodes, sort])

  const toggle = (key: EpisodeSortKey) => {
    setSort((current) =>
      current.key === key
        ? { key, direction: current.direction === 'asc' ? 'desc' : 'asc' }
        : { key, direction: key === 'start_date' ? 'asc' : 'desc' },
    )
  }

  if (!episodes.length) {
    return <p className="text-[11px] text-muted-foreground">No below-SMA200 episodes.</p>
  }

  return (
    <div className="space-y-1">
      <p className="text-[10px] text-muted-foreground">
        Open episodes are incomplete (still active at period end).
      </p>
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <SortableHead
              label="Dates"
              active={sort.key === 'start_date'}
              direction={sort.direction}
              onClick={() => toggle('start_date')}
            />
            <TableHead className={cn(compactHead, 'text-left')}>Status</TableHead>
            <SortableHead
              label="Days"
              active={sort.key === 'trading_days'}
              direction={sort.direction}
              onClick={() => toggle('trading_days')}
            />
            <TableHead className={cn(compactHead, 'text-right')}>Baseline</TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>Full</TableHead>
            <SortableHead
              label="Marginal"
              active={sort.key === 'compounded_marginal_return'}
              direction={sort.direction}
              onClick={() => toggle('compounded_marginal_return')}
            />
            <SortableHead
              label="MaxDD Δ"
              tip="Max drawdown improvement within the episode (signed). Positive means full was less severe."
              active={sort.key === 'max_drawdown_improvement'}
              direction={sort.direction}
              onClick={() => toggle('max_drawdown_improvement')}
            />
            <TableHead className={cn(compactHead, 'text-right')}>Added exp.</TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>Helped</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((ep) => (
            <TableRow key={`${ep.start_date}-${ep.end_date}`} className="hover:bg-muted/20">
              <TableCell className={cn(compactCell, 'text-left')}>
                {ep.start_date} → {ep.end_date}
              </TableCell>
              <TableCell className={cn(compactCell, 'text-left capitalize font-sans')}>
                {ep.status}
              </TableCell>
              <TableCell className={cn(compactCell, 'text-right')}>{ep.trading_days}</TableCell>
              <TableCell
                className={cn(compactCell, 'text-right', toneClass(ep.baseline_compounded_return))}
              >
                {formatSignedPercent(ep.baseline_compounded_return)}
              </TableCell>
              <TableCell
                className={cn(compactCell, 'text-right', toneClass(ep.full_compounded_return))}
              >
                {formatSignedPercent(ep.full_compounded_return)}
              </TableCell>
              <TableCell
                className={cn(compactCell, 'text-right', toneClass(ep.compounded_marginal_return))}
              >
                {formatSignedPercent(ep.compounded_marginal_return)}
              </TableCell>
              <TableCell
                className={cn(compactCell, 'text-right', toneClass(ep.max_drawdown_improvement))}
              >
                {formatSignedPercent(ep.max_drawdown_improvement)}
              </TableCell>
              <TableCell className={cn(compactCell, 'text-right')}>{ep.added_exposure_days}</TableCell>
              <TableCell className={cn(compactCell, 'text-right')}>{ep.helped ? 'Yes' : 'No'}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function DrawdownEpisodeTable({ episodes }: { episodes: BaselineDrawdownEpisode[] }) {
  const [sort, setSort] = useState<{ key: EpisodeSortKey; direction: SortDirection }>({
    key: 'start_date',
    direction: 'asc',
  })

  const sorted = useMemo(() => {
    const copy = [...episodes]
    copy.sort((a, b) => {
      const left = a[sort.key as keyof BaselineDrawdownEpisode]
      const right = b[sort.key as keyof BaselineDrawdownEpisode]
      return compareRowValues(left as never, right as never, sort.direction)
    })
    return copy
  }, [episodes, sort])

  const toggle = (key: EpisodeSortKey) => {
    setSort((current) =>
      current.key === key
        ? { key, direction: current.direction === 'asc' ? 'desc' : 'asc' }
        : { key, direction: key === 'start_date' ? 'asc' : 'desc' },
    )
  }

  if (!episodes.length) {
    return <p className="text-[11px] text-muted-foreground">No baseline drawdown episodes.</p>
  }

  return (
    <div className="space-y-1">
      <p className="text-[10px] text-muted-foreground">
        Open episodes have not recovered to the prior baseline peak. Recovery acceleration is null when
        either recovery is incomplete.
      </p>
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <SortableHead
              label="Dates"
              active={sort.key === 'start_date'}
              direction={sort.direction}
              onClick={() => toggle('start_date')}
            />
            <TableHead className={cn(compactHead, 'text-left')}>Status</TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>Base trough</TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>Full trough</TableHead>
            <SortableHead
              label="Trough Δ"
              tip="Full trough drawdown minus baseline trough. Positive means the strategy improved the trough."
              active={sort.key === 'trough_improvement'}
              direction={sort.direction}
              onClick={() => toggle('trough_improvement')}
            />
            <SortableHead
              label="Recovery Δ"
              tip="Baseline recovery days − full recovery days. Positive means the full portfolio recovered faster."
              active={sort.key === 'recovery_acceleration'}
              direction={sort.direction}
              onClick={() => toggle('recovery_acceleration')}
            />
            <SortableHead
              label="Marginal"
              active={sort.key === 'compounded_marginal_return'}
              direction={sort.direction}
              onClick={() => toggle('compounded_marginal_return')}
            />
            <TableHead className={cn(compactHead, 'text-right')}>Added exp.</TableHead>
            <TableHead className={cn(compactHead, 'text-right')}>Helped</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((ep) => (
            <TableRow key={`${ep.start_date}-${ep.end_date}`} className="hover:bg-muted/20">
              <TableCell className={cn(compactCell, 'text-left')}>
                {ep.start_date} → {ep.end_date}
                <span className="block text-[10px] text-muted-foreground">
                  trough {ep.trough_date}
                </span>
              </TableCell>
              <TableCell className={cn(compactCell, 'text-left capitalize font-sans')}>
                {ep.status}
              </TableCell>
              <TableCell className={cn(compactCell, 'text-right')}>
                {formatSignedPercent(ep.baseline_trough_drawdown)}
              </TableCell>
              <TableCell className={cn(compactCell, 'text-right')}>
                {formatSignedPercent(ep.full_trough_drawdown)}
              </TableCell>
              <TableCell className={cn(compactCell, 'text-right', toneClass(ep.trough_improvement))}>
                {formatSignedPercent(ep.trough_improvement)}
              </TableCell>
              <TableCell
                className={cn(compactCell, 'text-right', toneClass(ep.recovery_acceleration))}
              >
                {ep.recovery_acceleration === null || ep.recovery_acceleration === undefined
                  ? '—'
                  : ep.recovery_acceleration}
              </TableCell>
              <TableCell
                className={cn(compactCell, 'text-right', toneClass(ep.compounded_marginal_return))}
              >
                {formatSignedPercent(ep.compounded_marginal_return)}
              </TableCell>
              <TableCell className={cn(compactCell, 'text-right')}>{ep.added_exposure_days}</TableCell>
              <TableCell className={cn(compactCell, 'text-right')}>{ep.helped ? 'Yes' : 'No'}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function StrategyRegimeBlock({ strategy }: { strategy: StrategyRegimeContribution }) {
  const [open, setOpen] = useState(true)
  const [showBelowEps, setShowBelowEps] = useState(false)
  const [showDdEps, setShowDdEps] = useState(false)

  const belowRow = strategy.regimes.find((row) => row.dimension === 'spy_trend' && row.state === 'below')
  const ddRow = strategy.regimes.find(
    (row) => row.dimension === 'baseline_drawdown' && row.state === 'drawdown',
  )
  const belowEps = (belowRow?.episodes ?? []) as BelowSmaEpisode[]
  const ddEps = (ddRow?.episodes ?? []) as BaselineDrawdownEpisode[]

  return (
    <div className="space-y-2 rounded-md border border-border/40 p-2">
      <button
        type="button"
        className="flex w-full items-center gap-1.5 text-left text-xs font-medium"
        onClick={() => setOpen((value) => !value)}
      >
        {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
        {strategy.strategy_name}
        <span className="ml-auto font-mono text-[10px] font-normal text-muted-foreground">
          {strategy.total_eligible_days} eligible days
          {strategy.total_marginal_log_return !== null &&
          strategy.total_marginal_log_return !== undefined ? (
            <>
              {' · ΣM '}
              <span className={toneClass(Math.exp(strategy.total_marginal_log_return) - 1)}>
                {formatSignedPercent(Math.exp(strategy.total_marginal_log_return) - 1)}
              </span>
            </>
          ) : null}
        </span>
      </button>

      {open ? (
        <div className="space-y-3">
          {strategy.errors?.length ? (
            <ul className="space-y-0.5 text-[11px] text-[var(--loss)]">
              {strategy.errors.map((error) => (
                <li key={`${error.dimension}-${error.message}`}>
                  {error.dimension}: {error.message}
                </li>
              ))}
            </ul>
          ) : null}

          <RegimeSummaryTable regimes={strategy.regimes} />

          {belowRow?.concentration_warning ? (
            <p className="text-[10px] text-muted-foreground">{belowRow.concentration_message}</p>
          ) : null}
          {ddRow?.concentration_warning ? (
            <p className="text-[10px] text-muted-foreground">{ddRow.concentration_message}</p>
          ) : null}

          <div className="space-y-1">
            <button
              type="button"
              className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
              onClick={() => setShowBelowEps((value) => !value)}
            >
              {showBelowEps ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
              Below SMA200 episodes ({belowEps.length})
            </button>
            {showBelowEps ? <BelowSmaEpisodeTable episodes={belowEps} /> : null}
          </div>

          <div className="space-y-1">
            <button
              type="button"
              className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
              onClick={() => setShowDdEps((value) => !value)}
            >
              {showDdEps ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
              Baseline drawdown episodes ({ddEps.length})
            </button>
            {showDdEps ? <DrawdownEpisodeTable episodes={ddEps} /> : null}
          </div>
        </div>
      ) : null}
    </div>
  )
}

export function RegimeContributionResults({ result }: { result: PortfolioRegimeResult }) {
  const lag = String(result.parameters.execution_lag_convention ?? '')
  return (
    <div className="space-y-2">
      <p className="text-[10px] text-muted-foreground">
        SPY is always the market benchmark. Marginal return is full vs leave-one-out baseline (log-additive;
        compounded display is not CAGR). Annualized conditional contribution rate = 252 × mean daily
        marginal log return (hypothetical). Daily path does not drop the best calendar year used in summary
        CAGR. Execution lag: {lag || 'same_bar_close_synchronized'}.
      </p>
      <div className="space-y-2">
        {result.strategies.map((strategy) => (
          <StrategyRegimeBlock key={strategy.strategy_id} strategy={strategy} />
        ))}
      </div>
    </div>
  )
}
