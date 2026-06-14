'use client'

import { useState } from 'react'
import { Plus, Trash2, Play, Save } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { cn } from '@/lib/utils'
import { INDICATORS, OPERATORS } from '@/lib/mock-data'

type Condition = {
  id: string
  left: string
  operator: string
  right: string
  logic: 'AND' | 'OR'
}

let counter = 0
const newCondition = (): Condition => ({
  id: `c${++counter}`,
  left: 'RSI(2)',
  operator: '<=',
  right: '20',
  logic: 'AND',
})

export function StrategyBuilderSection() {
  const [name, setName] = useState('')
  const [symbol, setSymbol] = useState('SPY')
  const [direction, setDirection] = useState('long')
  const [holdDays, setHoldDays] = useState('2')
  const [description, setDescription] = useState('')
  const [conditions, setConditions] = useState<Condition[]>([
    { id: 'c0', left: 'Close', operator: '<', right: 'SMA(200)', logic: 'AND' },
    newCondition(),
  ])

  const update = (id: string, patch: Partial<Condition>) =>
    setConditions((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch } : c)))

  const preview = conditions
    .map((c, i) => `${i > 0 ? ` ${c.logic} ` : ''}${c.left} ${c.operator} ${c.right}`)
    .join('')

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
      {/* Builder */}
      <div className="flex flex-col gap-6 xl:col-span-2">
        <Card className="border-border/60 p-6">
          <h2 className="text-lg font-semibold">New strategy</h2>
          <p className="text-sm text-muted-foreground">
            Define entry conditions, holding rules, and metadata for a new signal.
          </p>

          <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="flex flex-col gap-2">
              <Label htmlFor="strat-name">Strategy name</Label>
              <Input
                id="strat-name"
                placeholder="e.g. RSI(2) Mean Reversion"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="strat-symbol">Primary symbol</Label>
              <Input
                id="strat-symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="font-mono"
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="strat-dir">Direction</Label>
              <Select value={direction} onValueChange={(v) => v && setDirection(v)}>
                <SelectTrigger id="strat-dir" className="w-full">
                  <SelectValue>
                    {(value: string) => (value === 'long' ? 'Long' : 'Short')}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="long">Long</SelectItem>
                  <SelectItem value="short">Short</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="strat-hold">Hold days</Label>
              <Input
                id="strat-hold"
                inputMode="numeric"
                value={holdDays}
                onChange={(e) => setHoldDays(e.target.value)}
                className="font-mono"
              />
            </div>
          </div>
        </Card>

        <Card className="border-border/60 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold">Entry conditions</h3>
              <p className="text-xs text-muted-foreground">
                All conditions are combined in order using the chosen logic.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="gap-1.5"
              onClick={() => setConditions((prev) => [...prev, newCondition()])}
            >
              <Plus className="size-4" />
              Add
            </Button>
          </div>

          <div className="mt-4 flex flex-col gap-3">
            {conditions.map((c, i) => (
              <div key={c.id} className="flex flex-col gap-2">
                {i > 0 && (
                  <div className="flex w-fit overflow-hidden rounded-md border border-border/60">
                    {(['AND', 'OR'] as const).map((l) => (
                      <button
                        key={l}
                        type="button"
                        onClick={() => update(c.id, { logic: l })}
                        className={cn(
                          'px-3 py-1 text-xs font-medium transition-colors',
                          c.logic === l
                            ? 'bg-primary/20 text-foreground'
                            : 'text-muted-foreground hover:bg-muted',
                        )}
                      >
                        {l}
                      </button>
                    ))}
                  </div>
                )}
                <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border/60 bg-muted/20 p-3">
                  <Select value={c.left} onValueChange={(v) => v && update(c.id, { left: v })}>
                    <SelectTrigger className="w-[150px] font-mono text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {INDICATORS.map((ind) => (
                        <SelectItem key={ind} value={ind} className="font-mono">
                          {ind}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select value={c.operator} onValueChange={(v) => v && update(c.id, { operator: v })}>
                    <SelectTrigger className="w-[150px] font-mono text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {OPERATORS.map((op) => (
                        <SelectItem key={op} value={op} className="font-mono">
                          {op}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Input
                    value={c.right}
                    onChange={(e) => update(c.id, { right: e.target.value })}
                    className="h-9 w-[150px] flex-1 font-mono text-xs"
                    placeholder="value or indicator"
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-9 shrink-0 text-muted-foreground hover:text-[var(--loss)]"
                    onClick={() => setConditions((prev) => prev.filter((x) => x.id !== c.id))}
                    aria-label="Remove condition"
                    disabled={conditions.length === 1}
                  >
                    <Trash2 className="size-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-5 flex flex-col gap-2">
            <Label htmlFor="strat-desc">Description</Label>
            <Textarea
              id="strat-desc"
              placeholder="Describe the rationale and any filters for this strategy…"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
            />
          </div>
        </Card>
      </div>

      {/* Preview / summary */}
      <div className="flex flex-col gap-6">
        <Card className="sticky top-6 border-border/60 p-6">
          <h3 className="text-sm font-semibold">Summary</h3>
          <dl className="mt-4 flex flex-col gap-3 text-sm">
            <div className="flex items-center justify-between">
              <dt className="text-muted-foreground">Name</dt>
              <dd className="font-medium">{name || '—'}</dd>
            </div>
            <div className="flex items-center justify-between">
              <dt className="text-muted-foreground">Symbol</dt>
              <dd className="font-mono">{symbol || '—'}</dd>
            </div>
            <div className="flex items-center justify-between">
              <dt className="text-muted-foreground">Direction</dt>
              <dd>
                <Badge variant="outline" className="capitalize">
                  {direction}
                </Badge>
              </dd>
            </div>
            <div className="flex items-center justify-between">
              <dt className="text-muted-foreground">Hold days</dt>
              <dd className="font-mono">{holdDays || '—'}</dd>
            </div>
            <div className="flex items-center justify-between">
              <dt className="text-muted-foreground">Conditions</dt>
              <dd className="font-mono">{conditions.length}</dd>
            </div>
          </dl>

          <div className="mt-5">
            <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Generated rule
            </span>
            <pre className="mt-2 overflow-x-auto rounded-lg border border-border/60 bg-background/60 p-3 font-mono text-xs leading-relaxed text-foreground">
              {preview || 'No conditions yet'}
            </pre>
          </div>

          <div className="mt-5 flex flex-col gap-2">
            <Button className="w-full gap-2">
              <Save className="size-4" />
              Save strategy
            </Button>
            <Button variant="outline" className="w-full gap-2">
              <Play className="size-4" />
              Save & backtest
            </Button>
          </div>
        </Card>
      </div>
    </div>
  )
}
