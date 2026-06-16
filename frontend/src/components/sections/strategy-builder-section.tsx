'use client'

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useRef, useState } from 'react'
import {
  getIndicators,
  getSavedStrategies,
  runBuilderBacktest,
  runBuilderRefine,
  saveStrategy,
  type DetailedResult,
  type SaveStrategyPayload,
  type SavedStrategy,
  type SweepResult,
} from '@/api'
import { MemoizedStrategyBuilderResultsPane } from '@/components/backtest/strategy-builder-results-pane'
import {
  StrategyBuilderSetup,
  type StrategyBuilderSetupHandle,
} from '@/components/sections/strategy-builder-setup'
import { Card } from '@/components/ui/card'
import type { BuilderRefineMode } from '@/lib/builder-refine-config'
import { buildBuilderRefinePayload } from '@/lib/builder-refine-form'
import { canAddSweepRowToBuilder, sweepRowAddLabel } from '@/lib/sweep-to-condition'

export function StrategyBuilderSection({
  initialStrategy,
}: {
  initialStrategy?: SavedStrategy
}) {
  const queryClient = useQueryClient()
  const setupRef = useRef<StrategyBuilderSetupHandle>(null)

  const { data: indicators = [] } = useQuery({
    queryKey: ['indicators'],
    queryFn: () => getIndicators(true),
  })
  const { data: savedStrategies = [] } = useQuery({
    queryKey: ['strategies'],
    queryFn: getSavedStrategies,
  })

  const [builderResult, setBuilderResult] = useState<DetailedResult | SweepResult | undefined>()
  const [builderResultsVersion, setBuilderResultsVersion] = useState(0)
  const [builderSelectedSweepRow, setBuilderSelectedSweepRow] = useState<
    Record<string, unknown> | undefined
  >()

  const [refineResult, setRefineResult] = useState<DetailedResult | SweepResult | undefined>()
  const [refineResultsVersion, setRefineResultsVersion] = useState(0)
  const [refineSelectedSweepRow, setRefineSelectedSweepRow] = useState<
    Record<string, unknown> | undefined
  >()

  const mutation = useMutation({
    mutationFn: runBuilderBacktest,
    onSuccess: (data) => {
      setBuilderResult(data)
      setBuilderResultsVersion((v) => v + 1)
    },
  })

  const refineMutation = useMutation({
    mutationFn: runBuilderRefine,
    onSuccess: (data) => {
      setRefineSelectedSweepRow(undefined)
      setRefineResult(data)
      setRefineResultsVersion((v) => v + 1)
    },
  })

  const saveMutation = useMutation({
    mutationFn: saveStrategy,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scan'] })
      queryClient.invalidateQueries({ queryKey: ['strategies'] })
    },
  })

  const [validationError, setValidationError] = useState<string | null>(null)
  const [saveMessage, setSaveMessage] = useState<string | null>(null)
  const [addMessage, setAddMessage] = useState<string | null>(null)

  const [refineMode, setRefineMode] = useState<BuilderRefineMode>('indicator-sweep')
  const [refinePrimarySymbol, setRefinePrimarySymbol] = useState('SPY')
  const [refineSymbolPool, setRefineSymbolPool] = useState('SPY, SMH, QQQ, SOXX')
  const [refineMaxDays, setRefineMaxDays] = useState(7)
  const [refineCheckBreadth, setRefineCheckBreadth] = useState(false)
  const [refineCheckBoth, setRefineCheckBoth] = useState(false)
  const [refineIsSell, setRefineIsSell] = useState(false)
  const [paneResetVersion, setPaneResetVersion] = useState(0)

  const handleSymbolChange = useCallback((symbol: string) => {
    setRefinePrimarySymbol(symbol)
  }, [])

  const handleReset = useCallback(() => {
    setBuilderResult(undefined)
    setBuilderSelectedSweepRow(undefined)
    setRefineResult(undefined)
    setRefineSelectedSweepRow(undefined)
    setValidationError(null)
    setSaveMessage(null)
    setAddMessage(null)
    setPaneResetVersion((version) => version + 1)
    mutation.reset()
    refineMutation.reset()
  }, [mutation, refineMutation])

  const handleRunBacktest = useCallback(
    (payload: Parameters<typeof runBuilderBacktest>[0]) => {
      setValidationError(null)
      setSaveMessage(null)
      mutation.mutate(payload)
    },
    [mutation],
  )

  const handleSave = useCallback(
    (payload: SaveStrategyPayload) => {
      setValidationError(null)
      setSaveMessage(null)
      saveMutation.mutate(payload, {
        onSuccess: (saved) => {
          setSaveMessage(`Saved "${saved.name}" — it will appear on the Scan page.`)
        },
        onError: (error) => {
          setValidationError(String(error))
        },
      })
    },
    [saveMutation],
  )

  const handleAddFromSweepRow = useCallback((row: Record<string, unknown>) => {
    setValidationError(null)
    setSaveMessage(null)
    const message = setupRef.current?.addFromSweepRow(row)
    if (!message) return
    if (message.startsWith('Could not')) {
      setValidationError(message)
    } else {
      setAddMessage(message)
    }
  }, [])

  const canAddSweepRow = useCallback(
    (row: Record<string, unknown>) => canAddSweepRowToBuilder(row, indicators),
    [indicators],
  )

  const handleRefineRun = useCallback(() => {
    setValidationError(null)
    setAddMessage(null)
    try {
      const draft = setupRef.current?.buildRefineDraft()
      if (!draft) return
      refineMutation.mutate(
        buildBuilderRefinePayload(draft, {
          mode: refineMode,
          primarySymbol: refinePrimarySymbol,
          symbolPool: refineSymbolPool,
          maxDays: refineMaxDays,
          checkBreadth: refineCheckBreadth,
          checkBoth: refineCheckBoth,
          isSell: refineIsSell,
        }),
      )
    } catch (error) {
      setValidationError(String(error))
    }
  }, [
    refineMode,
    refinePrimarySymbol,
    refineSymbolPool,
    refineMaxDays,
    refineCheckBreadth,
    refineCheckBoth,
    refineIsSell,
    refineMutation,
  ])

  return (
    <div className="flex flex-col gap-2">
      <StrategyBuilderSetup
        ref={setupRef}
        indicators={indicators}
        savedStrategies={savedStrategies}
        initialStrategy={initialStrategy}
        onSymbolChange={handleSymbolChange}
        onRunBacktest={handleRunBacktest}
        onSave={handleSave}
        onReset={handleReset}
        isBacktestRunning={mutation.isPending}
        isSaving={saveMutation.isPending}
      />

      {(validationError || mutation.error || saveMutation.error || refineMutation.error) && (
        <Card className="border-destructive/50 bg-destructive/5 p-3">
          <pre className="overflow-x-auto whitespace-pre-wrap text-xs text-destructive">
            {validationError ??
              String(mutation.error ?? saveMutation.error ?? refineMutation.error)}
          </pre>
        </Card>
      )}

      {saveMessage && (
        <Card className="border-[var(--gain)]/40 bg-[var(--gain)]/5 p-3">
          <p className="text-sm text-foreground">{saveMessage}</p>
        </Card>
      )}

      {addMessage && (
        <Card className="border-[var(--gain)]/40 bg-[var(--gain)]/5 p-3">
          <p className="text-sm text-foreground">{addMessage}</p>
        </Card>
      )}

      <MemoizedStrategyBuilderResultsPane
        builderResult={builderResult}
        builderResultsVersion={builderResultsVersion}
        builderSelectedSweepRow={builderSelectedSweepRow}
        onBuilderSelectSweepRow={setBuilderSelectedSweepRow}
        refineResult={refineResult}
        refineResultsVersion={refineResultsVersion}
        refineSelectedSweepRow={refineSelectedSweepRow}
        onRefineSelectSweepRow={setRefineSelectedSweepRow}
        resetVersion={paneResetVersion}
        onAddConditionFromSweepRow={handleAddFromSweepRow}
        canAddSweepRow={canAddSweepRow}
        sweepRowAddLabel={sweepRowAddLabel}
        refineProps={{
          mode: refineMode,
          onModeChange: setRefineMode,
          savedStrategies,
          primarySymbol: refinePrimarySymbol,
          onPrimarySymbolChange: setRefinePrimarySymbol,
          symbolPool: refineSymbolPool,
          onSymbolPoolChange: setRefineSymbolPool,
          maxDays: refineMaxDays,
          onMaxDaysChange: setRefineMaxDays,
          checkBreadth: refineCheckBreadth,
          onCheckBreadthChange: setRefineCheckBreadth,
          checkBoth: refineCheckBoth,
          onCheckBothChange: setRefineCheckBoth,
          isSell: refineIsSell,
          onIsSellChange: setRefineIsSell,
          onRun: handleRefineRun,
          isRunning: refineMutation.isPending,
        }}
      />
    </div>
  )
}
