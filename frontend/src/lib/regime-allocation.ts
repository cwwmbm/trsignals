import type { RegimePenaltyFields, RegimePenaltyPreset, RegimeScoreKind } from '@/api'

export type RegimeAllocationForm = {
  score: RegimeScoreKind
  penalty: RegimePenaltyPreset
}

export const DEFAULT_REGIME_FORM: RegimeAllocationForm = {
  score: 'robustness',
  penalty: 'medium',
}

/** Must match `PENALTY_PRESETS` in api/regime_allocation.py. */
export const REGIME_PENALTY_OPTIONS: Array<{
  value: RegimePenaltyPreset
  label: string
  rule: string
}> = [
  { value: 'harsh', label: 'Harsh', rule: 'score ≤10: −100%' },
  { value: 'strict', label: 'Strict', rule: '≤10: −50% · 10–25: −25%' },
  { value: 'medium', label: 'Medium', rule: '≤10: −33% · 10–25: −10%' },
  { value: 'low', label: 'Low', rule: '≤10: −25% · 10–25: −5%' },
]

export const REGIME_SCORE_OPTIONS: Array<{
  value: RegimeScoreKind
  label: string
  hint: string
}> = [
  {
    value: 'robustness',
    label: 'Robustness score',
    hint: 'Average trade return and year-to-year robustness.',
  },
  {
    value: 'portfolio',
    label: 'Regime score',
    hint: 'Adds Sortino and drawdown to the robustness score.',
  },
]

/** Fill the form from a saved portfolio. Portfolios saved earlier have no regime fields. */
export function regimeFormFromSaved(saved: RegimePenaltyFields | undefined): RegimeAllocationForm {
  return {
    score: saved?.regime_score ?? DEFAULT_REGIME_FORM.score,
    penalty: saved?.regime_penalty ?? DEFAULT_REGIME_FORM.penalty,
  }
}

export function regimeFields(form: RegimeAllocationForm): Required<RegimePenaltyFields> {
  return { regime_score: form.score, regime_penalty: form.penalty }
}

/** One line for the saved-portfolios table, e.g. "Medium · Robustness". */
export function regimeSummary(saved: RegimePenaltyFields): string {
  const form = regimeFormFromSaved(saved)
  const penalty = REGIME_PENALTY_OPTIONS.find((option) => option.value === form.penalty)
  const score = REGIME_SCORE_OPTIONS.find((option) => option.value === form.score)
  return `${penalty?.label ?? form.penalty} · ${score?.label ?? form.score}`
}
