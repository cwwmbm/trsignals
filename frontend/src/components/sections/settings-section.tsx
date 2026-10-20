'use client'

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Loader2, Plus, Save, Sparkles, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { LeverageEstimate } from '@/api'
import { estimateLeverage, getLeverageSettings, saveLeverageSettings } from '@/api'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  applyEstimate,
  describeEstimate,
  emptyRow,
  formFromSettings,
  isDirty,
  settingsFromForm,
  type SettingsForm,
  type SymbolRow,
} from '@/lib/leverage-settings'
import { cn } from '@/lib/utils'

const cellInput = 'h-8 text-xs'

type Estimated = { estimate?: LeverageEstimate; error?: string; pending?: boolean }

export function SettingsSection() {
  const queryClient = useQueryClient()
  const { data: saved, isLoading, error } = useQuery({
    queryKey: ['leverage-settings'],
    queryFn: getLeverageSettings,
  })
  const [form, setForm] = useState<SettingsForm | null>(null)
  const [estimates, setEstimates] = useState<Record<number, Estimated>>({})
  const [clientError, setClientError] = useState<string | null>(null)

  useEffect(() => {
    if (saved && form === null) setForm(formFromSettings(saved))
  }, [saved, form])

  const save = useMutation({
    mutationFn: saveLeverageSettings,
    onSuccess: (stored) => {
      queryClient.setQueryData(['leverage-settings'], stored)
      setForm(formFromSettings(stored))
      setEstimates({})
    },
  })

  if (isLoading || form === null) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Loading settings…
      </div>
    )
  }
  if (error) return <p className="text-sm text-destructive">{String(error)}</p>

  const dirty = isDirty(form, saved)

  const update = (key: number, patch: Partial<SymbolRow>) =>
    setForm({ ...form, rows: form.rows.map((row) => (row.key === key ? { ...row, ...patch } : row)) })

  const estimate = async (row: SymbolRow) => {
    const symbol = row.symbol.trim().toUpperCase()
    const vehicle = row.leverageSymbol.trim().toUpperCase()
    if (!symbol || !vehicle) {
      setEstimates((current) => ({ ...current, [row.key]: { error: 'Enter the symbol and its leverage symbol first.' } }))
      return
    }
    setEstimates((current) => ({ ...current, [row.key]: { pending: true } }))
    try {
      const fit = await estimateLeverage(symbol, vehicle)
      setForm((current) =>
        current ? { ...current, rows: current.rows.map((r) => (r.key === row.key ? applyEstimate(r, fit) : r)) } : current,
      )
      setEstimates((current) => ({ ...current, [row.key]: { estimate: fit } }))
    } catch (failure) {
      setEstimates((current) => ({
        ...current,
        [row.key]: { error: failure instanceof Error ? failure.message : String(failure) },
      }))
    }
  }

  const submit = () => {
    const check = settingsFromForm(form)
    if (!check.ok) {
      setClientError(check.error)
      return
    }
    setClientError(null)
    save.mutate(check.settings)
  }

  return (
    <div className="space-y-4">
      <Card className="p-4">
        <h2 className="text-sm font-semibold">General leverage</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Every symbol without an entry below is traded as its own daily return times this factor
          (a daily-reset leveraged product). Entry signals are never affected.
        </p>
        <div className="mt-3 flex items-center gap-2">
          <Label htmlFor="general-leverage" className="text-xs text-muted-foreground">
            Leverage
          </Label>
          <Input
            id="general-leverage"
            inputMode="decimal"
            value={form.general}
            onChange={(event) => setForm({ ...form, general: event.target.value })}
            className="h-8 w-20 text-xs"
          />
          <span className="text-xs text-muted-foreground">×</span>
        </div>
      </Card>

      <Card className="p-4">
        <h2 className="text-sm font-semibold">Symbols</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          For a symbol listed here, strategies still read their signals from the symbol itself, but the
          position is traded in its leverage symbol for as long as that has price history. A trade is
          opened in the leverage symbol if it exists that day, and stays in whatever it opened in until it
          closes. Earlier history, and any trade already open when the leverage symbol appears, uses the
          symbol's own return times the leverage, with the cost model below if you fill it in. Shorts
          never use a leverage symbol.
        </p>

        <div className="mt-3 space-y-3">
          {form.rows.length === 0 ? (
            <p className="text-xs text-muted-foreground">No symbols yet. Everything uses the general leverage.</p>
          ) : null}
          {form.rows.map((row) => {
            const note = estimates[row.key]
            return (
              <div key={row.key} className="rounded-md border border-border/60 p-3">
                <div className="flex flex-wrap items-end gap-3">
                  <div className="flex flex-col gap-1">
                    <Label htmlFor={`symbol-${row.key}`} className="text-[11px] text-muted-foreground">
                      Symbol
                    </Label>
                    <Input
                      id={`symbol-${row.key}`}
                      value={row.symbol}
                      onChange={(event) => update(row.key, { symbol: event.target.value })}
                      placeholder="SOXX"
                      className={cn(cellInput, 'w-24 uppercase')}
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <Label htmlFor={`vehicle-${row.key}`} className="text-[11px] text-muted-foreground">
                      Leverage symbol
                    </Label>
                    <Input
                      id={`vehicle-${row.key}`}
                      value={row.leverageSymbol}
                      onChange={(event) => update(row.key, { leverageSymbol: event.target.value })}
                      placeholder="SOXL"
                      className={cn(cellInput, 'w-24 uppercase')}
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <Label htmlFor={`leverage-${row.key}`} className="text-[11px] text-muted-foreground">
                      Leverage
                    </Label>
                    <Input
                      id={`leverage-${row.key}`}
                      inputMode="decimal"
                      value={row.leverage}
                      onChange={(event) => update(row.key, { leverage: event.target.value })}
                      className={cn(cellInput, 'w-16')}
                    />
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="ml-auto h-8 gap-1 px-2 text-xs text-destructive hover:text-destructive"
                    onClick={() => setForm({ ...form, rows: form.rows.filter((r) => r.key !== row.key) })}
                  >
                    <Trash2 className="size-3.5" />
                    Remove
                  </Button>
                </div>

                <div className="mt-3 flex flex-wrap items-end gap-3 border-t border-border/40 pt-3">
                  <span className="w-full text-[11px] font-medium text-muted-foreground">
                    Cost model for the stand-in before the leverage symbol exists (optional)
                  </span>
                  <div className="flex flex-col gap-1">
                    <Label htmlFor={`beta-${row.key}`} className="text-[11px] text-muted-foreground">
                      Beta
                    </Label>
                    <Input
                      id={`beta-${row.key}`}
                      inputMode="decimal"
                      value={row.beta}
                      onChange={(event) => update(row.key, { beta: event.target.value })}
                      placeholder={row.leverage}
                      className={cn(cellInput, 'w-16')}
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <Label htmlFor={`fee-${row.key}`} className="text-[11px] text-muted-foreground">
                      Annual fee %
                    </Label>
                    <Input
                      id={`fee-${row.key}`}
                      inputMode="decimal"
                      value={row.feePct}
                      onChange={(event) => update(row.key, { feePct: event.target.value })}
                      placeholder="0"
                      className={cn(cellInput, 'w-20')}
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <Label htmlFor={`financing-${row.key}`} className="text-[11px] text-muted-foreground">
                      Financing × T-bill
                    </Label>
                    <Input
                      id={`financing-${row.key}`}
                      inputMode="decimal"
                      value={row.financing}
                      onChange={(event) => update(row.key, { financing: event.target.value })}
                      placeholder="0"
                      className={cn(cellInput, 'w-20')}
                    />
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1 px-2 text-xs"
                    disabled={note?.pending}
                    onClick={() => estimate(row)}
                  >
                    {note?.pending ? <Loader2 className="size-3.5 animate-spin" /> : <Sparkles className="size-3.5" />}
                    Estimate from history
                  </Button>
                </div>
                {note?.estimate ? (
                  <p className="mt-2 text-[11px] text-muted-foreground">
                    {describeEstimate(note.estimate)} {note.estimate.leverage_symbol_first_date} is when{' '}
                    {row.leverageSymbol.trim().toUpperCase()} starts; the figures are filled in above, save to keep
                    them.
                  </p>
                ) : null}
                {note?.error ? <p className="mt-2 text-[11px] text-destructive">{note.error}</p> : null}
              </div>
            )
          })}
        </div>

        <Button
          type="button"
          variant="outline"
          size="sm"
          className="mt-3 gap-1.5 text-xs"
          onClick={() => setForm({ ...form, rows: [...form.rows, emptyRow()] })}
        >
          <Plus className="size-3.5" />
          Add symbol
        </Button>
      </Card>

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={submit} disabled={!dirty || save.isPending} className="gap-2">
          {save.isPending ? <Loader2 className="size-4 animate-spin" /> : <Save className="size-4" />}
          Save settings
        </Button>
        {dirty ? (
          <Button variant="ghost" onClick={() => setForm(saved ? formFromSettings(saved) : form)} className="text-xs">
            Discard changes
          </Button>
        ) : save.isSuccess ? (
          <span className="text-xs text-muted-foreground">Saved. New runs use these settings.</span>
        ) : null}
        {clientError ? <span className="text-xs text-destructive">{clientError}</span> : null}
        {save.error ? <span className="text-xs text-destructive">{save.error.message}</span> : null}
      </div>
    </div>
  )
}
