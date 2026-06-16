'use client'

import { BookOpen, FlaskConical, ListChecks, Radar, Wrench } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { RUN_MODES } from '@/lib/mock-data'
import { IndicatorGlossary } from '@/components/sections/indicator-glossary'

const SECTION_HELP = [
  {
    icon: FlaskConical,
    title: 'Backtest',
    body: 'Configure a run mode, symbol, and signal, then run a historical simulation. Results include headline stats, a year-by-year breakdown, the full trade list, and an equity curve with a log-scale toggle.',
  },
  {
    icon: ListChecks,
    title: 'All strategies',
    body: 'Browse every saved strategy with its key performance metrics. Filter by status (live, draft, archived) or search by name, signal, or tag, and jump straight into a backtest.',
  },
  {
    icon: Wrench,
    title: 'New strategy builder',
    body: 'Compose entry conditions from indicators and operators, set direction and holding period, and preview the generated rule before saving or backtesting it.',
  },
  {
    icon: Radar,
    title: 'Scan',
    body: 'See the latest scan across all symbols and signals. The Signal and Description fields are editable inline so you can refine definitions without leaving the table.',
  },
]

const GLOSSARY = [
  { term: 'CAGR', def: 'Compound annual growth rate of the strategy equity.' },
  { term: 'Sharpe', def: 'Risk-adjusted return using total volatility as the denominator.' },
  { term: 'Sortino', def: 'Risk-adjusted return penalizing only downside volatility.' },
  { term: 'Max Drawdown', def: 'Largest peak-to-trough decline over the test window.' },
  { term: 'Kelly', def: 'Suggested fraction of capital to allocate per the Kelly criterion.' },
  { term: 'IBR', def: 'Internal Bar Range — where the close sits within the day’s range.' },
  { term: 'Pct Positive', def: 'Share of trades (or years) that closed profitably.' },
  { term: 'TradePnL', def: 'Profit or loss for the current open position, as a percent.' },
]

export function HelpSection() {
  return (
    <div className="flex flex-col gap-6">
      <Card className="border-border/60 p-6">
        <div className="flex items-center gap-3">
          <span className="flex size-9 items-center justify-center rounded-lg bg-primary/15 text-primary">
            <BookOpen className="size-5" />
          </span>
          <div>
            <h2 className="text-lg font-semibold">Help &amp; documentation</h2>
            <p className="text-sm text-muted-foreground">
              How each section works, the available run modes, a metrics glossary, and an indicator glossary.
            </p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {SECTION_HELP.map((s) => (
          <Card key={s.title} className="gap-3 border-border/60 p-5">
            <div className="flex items-center gap-2.5">
              <s.icon className="size-4 text-primary" />
              <h3 className="font-semibold">{s.title}</h3>
            </div>
            <p className="text-sm leading-relaxed text-muted-foreground">{s.body}</p>
          </Card>
        ))}
      </div>

      <Card className="border-border/60 p-6">
        <h3 className="text-sm font-semibold">Run modes</h3>
        <div className="mt-4 flex flex-col divide-y divide-border/60">
          {RUN_MODES.map((m) => (
            <div key={m.id} className="flex flex-col gap-1 py-3 first:pt-0 last:pb-0 sm:flex-row sm:gap-6">
              <span className="w-56 shrink-0 font-medium">{m.label}</span>
              <span className="text-sm text-muted-foreground">{m.description}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card className="border-border/60 p-6">
        <h3 className="text-sm font-semibold">Metrics glossary</h3>
        <dl className="mt-4 grid grid-cols-1 gap-x-8 gap-y-3 sm:grid-cols-2">
          {GLOSSARY.map((g) => (
            <div key={g.term} className="flex flex-col gap-0.5">
              <dt className="font-mono text-sm font-semibold text-primary">{g.term}</dt>
              <dd className="text-sm text-muted-foreground">{g.def}</dd>
            </div>
          ))}
        </dl>
      </Card>

      <Card className="border-border/60 p-6">
        <h3 className="text-sm font-semibold">Indicator glossary</h3>
        <p className="mt-1 text-sm text-muted-foreground">
          Brief definitions for every indicator available in the strategy builder, grouped by category.
        </p>
        <div className="mt-4">
          <IndicatorGlossary />
        </div>
      </Card>
    </div>
  )
}
