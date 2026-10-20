import type { LeverageEstimate, LeverageSettings, SymbolLeverage } from '@/api'

/** A symbol row as typed: text boxes, so a half-typed number is not lost. */
export type SymbolRow = {
  key: number
  symbol: string
  leverageSymbol: string
  leverage: string
  /** Blank means "the leverage". */
  beta: string
  /** Percent per year, as people say it (1.9), not the stored fraction. */
  feePct: string
  financing: string
}

export type SettingsForm = { general: string; rows: SymbolRow[] }

let nextKey = 1
export const newKey = () => nextKey++

const trim = (value: number) => String(Number(value.toFixed(4)))

export function emptyRow(): SymbolRow {
  return { key: newKey(), symbol: '', leverageSymbol: '', leverage: '3', beta: '', feePct: '', financing: '' }
}

export function formFromSettings(settings: LeverageSettings): SettingsForm {
  return {
    general: trim(settings.general_leverage),
    rows: settings.symbols.map((entry) => ({
      key: newKey(),
      symbol: entry.symbol,
      leverageSymbol: entry.leverage_symbol,
      leverage: trim(entry.leverage),
      beta: entry.beta == null ? '' : trim(entry.beta),
      feePct: entry.annual_fee ? trim(entry.annual_fee * 100) : '',
      financing: entry.financing_multiple ? trim(entry.financing_multiple) : '',
    })),
  }
}

function number(text: string): number {
  return text.trim() === '' ? Number.NaN : Number(text)
}

export type FormCheck = { ok: true; settings: LeverageSettings } | { ok: false; error: string }

/** Turn the form into what the server stores. The server repeats every rule; this catches typos early. */
export function settingsFromForm(form: SettingsForm): FormCheck {
  const general = number(form.general)
  if (!Number.isFinite(general)) return { ok: false, error: 'General leverage must be a number.' }
  const symbols: SymbolLeverage[] = []
  for (const row of form.rows) {
    const label = row.symbol.trim().toUpperCase() || 'A symbol'
    const leverage = number(row.leverage)
    if (!row.symbol.trim() || !row.leverageSymbol.trim()) {
      return { ok: false, error: `${label}: enter both the symbol and its leverage symbol.` }
    }
    if (!Number.isFinite(leverage)) return { ok: false, error: `${label}: leverage must be a number.` }
    const beta = row.beta.trim() === '' ? null : number(row.beta)
    if (beta !== null && !Number.isFinite(beta)) return { ok: false, error: `${label}: beta must be a number or blank.` }
    const fee = row.feePct.trim() === '' ? 0 : number(row.feePct)
    const financing = row.financing.trim() === '' ? 0 : number(row.financing)
    if (!Number.isFinite(fee) || !Number.isFinite(financing)) {
      return { ok: false, error: `${label}: fee and financing must be numbers or blank.` }
    }
    symbols.push({
      symbol: row.symbol.trim().toUpperCase(),
      leverage_symbol: row.leverageSymbol.trim().toUpperCase(),
      leverage,
      beta,
      // percent to fraction, without 1.88 / 100 leaving 0.018799999999999997 in the file
      annual_fee: Number((fee / 100).toFixed(8)),
      financing_multiple: financing,
    })
  }
  return { ok: true, settings: { general_leverage: general, symbols } }
}

/** Put a fitted cost model into a row, leaving what the user typed for leverage alone. */
export function applyEstimate(row: SymbolRow, estimate: LeverageEstimate): SymbolRow {
  return {
    ...row,
    beta: trim(estimate.beta),
    feePct: trim(Math.max(0, estimate.fee) * 100),
    financing: trim(Math.max(0, estimate.financing_multiple)),
  }
}

export function describeEstimate(estimate: LeverageEstimate): string {
  return (
    `Fitted on ${estimate.days.toLocaleString()} days (${estimate.first_date} to ${estimate.last_date}): ` +
    `correlation ${estimate.correlation.toFixed(3)}, tracking error ${(estimate.tracking_error * 100).toFixed(1)}% a year.`
  )
}

/** True when the form differs from what is saved, ignoring row keys. */
export function isDirty(form: SettingsForm, saved: LeverageSettings | undefined): boolean {
  if (!saved) return false
  const current = formFromSettings(saved)
  const strip = (value: SettingsForm) => ({ general: value.general, rows: value.rows.map(({ key: _key, ...rest }) => rest) })
  return JSON.stringify(strip(form)) !== JSON.stringify(strip(current))
}
