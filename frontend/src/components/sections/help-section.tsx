'use client'

import { BookOpen, LineChart, ListChecks, Radar, Wrench } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { IndicatorGlossary } from '@/components/sections/indicator-glossary'

const SECTION_HELP = [
  {
    icon: ListChecks,
    title: 'All strategies',
    body: 'Browse every saved strategy with its key performance metrics. Filter by status (live, draft, archived) or search by name, signal, or tag, and open one in the Strategy Builder to backtest it.',
  },
  {
    icon: Wrench,
    title: 'Strategy Builder',
    body: 'Compose entry conditions from indicators and operators, add market regimes, set direction and holding period, and preview the generated rule before saving or backtesting it. Use Load saved strategy to start from any saved strategy.',
  },
  {
    icon: Radar,
    title: 'Scan',
    body: 'See the latest scan across all symbols and signals. The Signal and Description fields are editable inline so you can refine definitions without leaving the table.',
  },
  {
    icon: LineChart,
    title: 'Quote',
    body: 'Yahoo snapshot of SPY, QQQ, SOXX, and VIX: percent change (VIX as level with change in brackets), IBR, RSI2, RSI5, and Stochastic. Weekdays in the last five calendar days with no bar are highlighted; weekends are ignored. Exchange holidays may light up so you can judge them. On a phone, open /quote (the mobile header also links Scan and Quote).',
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
              How each section works, a metrics glossary, and an indicator glossary.
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
          Definitions for every indicator in the strategy builder, grouped by category. Descriptions
          and sweep ranges are loaded from the API catalog.
        </p>
        <div className="mt-4">
          <IndicatorGlossary />
        </div>
      </Card>
    </div>
  )
}
