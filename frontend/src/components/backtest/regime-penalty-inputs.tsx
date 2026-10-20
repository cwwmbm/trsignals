'use client'

import type { RegimePenaltyPreset, RegimeScoreKind } from '@/api'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  REGIME_PENALTY_OPTIONS,
  REGIME_SCORE_OPTIONS,
  type RegimeAllocationForm,
} from '@/lib/regime-allocation'

type Props = {
  form: RegimeAllocationForm
  onChange: (form: RegimeAllocationForm) => void
}

/** The regime-adjusted allocation settings: how hard to punish, and which score to read. */
export function RegimePenaltyInputs({ form, onChange }: Props) {
  const current = REGIME_PENALTY_OPTIONS.find((option) => option.value === form.penalty)

  return (
    <div className="mt-3 ml-6 space-y-3">
      <div className="flex flex-col gap-1">
        <Label htmlFor="regime-penalty" className="text-[11px] text-muted-foreground">
          Penalties
        </Label>
        <Select
          value={form.penalty}
          onValueChange={(value) =>
            value && onChange({ ...form, penalty: value as RegimePenaltyPreset })
          }
        >
          <SelectTrigger id="regime-penalty" className="w-44">
            <SelectValue>{current?.label}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {REGIME_PENALTY_OPTIONS.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}{' '}
                <span className="text-[11px] text-muted-foreground">{option.rule}</span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-[11px] text-muted-foreground">
          Each current regime state where a strategy, or the portfolio, has scored at or below a
          cutoff takes that much off its confidence. States with fewer than 20 closed trades
          cost nothing. Exposure = strategy confidence × (70% + 30% × portfolio confidence).
        </p>
      </div>

      <div>
        <Label className="text-[11px] text-muted-foreground">Score to read</Label>
        <RadioGroup
          value={form.score}
          onValueChange={(value) => value && onChange({ ...form, score: value as RegimeScoreKind })}
          className="mt-1 gap-1.5"
        >
          {REGIME_SCORE_OPTIONS.map((option) => (
            <div key={option.value} className="flex items-start gap-2">
              <RadioGroupItem
                value={option.value}
                id={`regime-score-${option.value}`}
                className="mt-0.5"
              />
              <div>
                <Label htmlFor={`regime-score-${option.value}`} className="text-sm font-normal">
                  {option.label}
                </Label>
                <p className="text-[11px] text-muted-foreground">{option.hint}</p>
              </div>
            </div>
          ))}
        </RadioGroup>
      </div>
    </div>
  )
}
