import { useCallback, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import type { DragEndEvent } from '@dnd-kit/core'
import { arrayMove } from '@dnd-kit/sortable'
import {
  getSavedPortfolios,
  getSavedStrategies,
  getScan,
  updatePortfolio,
  updateStrategy,
  type SavedPortfolio,
  type SavedStrategy,
  type ScanLane,
  type ScanRow,
  type UpdatePortfolioPayload,
  type UpdateStrategyPayload,
} from '@/api'
import { sortSymbols } from '@/lib/symbol-order'
import {
  SCAN_LANES,
  findLaneForRow,
  groupScanRows,
  parseLaneContainerId,
  resolvePortfolioScanLane,
  resolveScanLane,
  type LaneRows,
} from '@/lib/scan-board'

function buildStrategyLaneUpdates(
  lane: ScanLane,
  rows: ScanRow[],
  strategyById: Map<string, SavedStrategy>,
): { id: string; payload: UpdateStrategyPayload }[] {
  const updates: { id: string; payload: UpdateStrategyPayload }[] = []
  rows.forEach((row, index) => {
    if (row.source !== 'builder' || !row.strategy_id) return
    const strategy = strategyById.get(row.strategy_id)
    if (!strategy) return
    const currentLane = resolveScanLane(strategy)
    const currentOrder = strategy.scan_sort_order ?? 0
    if (currentLane === lane && currentOrder === index) return
    updates.push({
      id: strategy.id,
      payload: { scan_lane: lane, scan_sort_order: index },
    })
  })
  return updates
}

function buildPortfolioLaneUpdates(
  lane: ScanLane,
  rows: ScanRow[],
  portfolioById: Map<string, SavedPortfolio>,
): { id: string; payload: UpdatePortfolioPayload }[] {
  const updates: { id: string; payload: UpdatePortfolioPayload }[] = []
  rows.forEach((row, index) => {
    if (row.source !== 'portfolio' || !row.portfolio_id) return
    const portfolio = portfolioById.get(row.portfolio_id)
    if (!portfolio) return
    const currentLane = resolvePortfolioScanLane(portfolio)
    const currentOrder = portfolio.scan_sort_order ?? 0
    if (currentLane === lane && currentOrder === index) return
    updates.push({
      id: portfolio.id,
      payload: { scan_lane: lane, scan_sort_order: index },
    })
  })
  return updates
}

export function useScanBoard(initialExpanded: string[] = ['active', 'legacy']) {
  const queryClient = useQueryClient()
  const [query, setQuery] = useState('')
  const [symbolFilter, setSymbolFilter] = useState<string>('all')
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    () => new Set(initialExpanded),
  )
  const [laneRows, setLaneRows] = useState<LaneRows>({
    active: [],
    testing: [],
    archived: [],
  })
  const [legacyRows, setLegacyRows] = useState<ScanRow[]>([])

  const {
    data: rows = [],
    isLoading,
    isFetching,
    error,
    refetch,
  } = useQuery({
    queryKey: ['scan'],
    queryFn: getScan,
  })
  const { data: savedStrategies = [] } = useQuery({
    queryKey: ['strategies'],
    queryFn: getSavedStrategies,
  })
  const { data: savedPortfolios = [] } = useQuery({
    queryKey: ['portfolios'],
    queryFn: getSavedPortfolios,
  })

  const strategyById = useMemo(
    () => new Map(savedStrategies.map((strategy) => [strategy.id, strategy])),
    [savedStrategies],
  )
  const portfolioById = useMemo(
    () => new Map(savedPortfolios.map((portfolio) => [portfolio.id, portfolio])),
    [savedPortfolios],
  )

  const filtered = useMemo(() => {
    return rows.filter((row) => {
      if (symbolFilter !== 'all' && row.symbol !== symbolFilter) return false
      if (!query) return true
      const q = query.toLowerCase()
      return (
        row.symbol.toLowerCase().includes(q) ||
        row.signal.toLowerCase().includes(q) ||
        row.description.toLowerCase().includes(q)
      )
    })
  }, [rows, query, symbolFilter])

  const grouped = useMemo(
    () => groupScanRows(filtered, strategyById, portfolioById),
    [filtered, strategyById, portfolioById],
  )

  useEffect(() => {
    setLaneRows(grouped.lanes)
    setLegacyRows(grouped.legacy)
  }, [grouped])

  const symbols = useMemo(
    () => sortSymbols([...new Set(rows.map((row) => row.symbol))]),
    [rows],
  )

  const activeSignals = filtered.filter((row) => row.buy_signal || row.hold_long).length

  const laneMutation = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: UpdateStrategyPayload }) =>
      updateStrategy(id, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] })
    },
  })

  const portfolioLaneMutation = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: UpdatePortfolioPayload }) =>
      updatePortfolio(id, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolios'] })
    },
  })

  const persistLaneLayout = useCallback(
    async (nextLanes: LaneRows, touched: Set<ScanLane>) => {
      const strategyUpdates = [...touched].flatMap((lane) =>
        buildStrategyLaneUpdates(lane, nextLanes[lane], strategyById),
      )
      const portfolioUpdates = [...touched].flatMap((lane) =>
        buildPortfolioLaneUpdates(lane, nextLanes[lane], portfolioById),
      )
      const uniqueStrategyUpdates = new Map<string, UpdateStrategyPayload>()
      for (const update of strategyUpdates) {
        uniqueStrategyUpdates.set(update.id, {
          ...uniqueStrategyUpdates.get(update.id),
          ...update.payload,
        })
      }
      const uniquePortfolioUpdates = new Map<string, UpdatePortfolioPayload>()
      for (const update of portfolioUpdates) {
        uniquePortfolioUpdates.set(update.id, {
          ...uniquePortfolioUpdates.get(update.id),
          ...update.payload,
        })
      }
      for (const [id, payload] of uniqueStrategyUpdates.entries()) {
        await laneMutation.mutateAsync({ id, payload })
      }
      for (const [id, payload] of uniquePortfolioUpdates.entries()) {
        await portfolioLaneMutation.mutateAsync({ id, payload })
      }
    },
    [laneMutation, portfolioLaneMutation, strategyById, portfolioById],
  )

  const handleLaneChange = useCallback(
    async (strategy: SavedStrategy, lane: ScanLane) => {
      const currentLane = resolveScanLane(strategy)
      if (currentLane === lane) return

      const nextLanes: LaneRows = {
        active: [...laneRows.active],
        testing: [...laneRows.testing],
        archived: [...laneRows.archived],
      }
      for (const sourceLane of SCAN_LANES) {
        nextLanes[sourceLane] = nextLanes[sourceLane].filter((row) => row.strategy_id !== strategy.id)
      }
      const movedRow =
        laneRows[currentLane].find((row) => row.strategy_id === strategy.id) ??
        laneRows.testing.find((row) => row.strategy_id === strategy.id) ??
        laneRows.active.find((row) => row.strategy_id === strategy.id) ??
        laneRows.archived.find((row) => row.strategy_id === strategy.id)
      if (movedRow) {
        nextLanes[lane] = [...nextLanes[lane], movedRow]
      }
      setLaneRows(nextLanes)
      setExpandedSections((prev) => new Set(prev).add(lane))
      try {
        await laneMutation.mutateAsync({
          id: strategy.id,
          payload: {
            scan_lane: lane,
            scan_sort_order: nextLanes[lane].length - 1,
          },
        })
      } catch {
        setLaneRows(grouped.lanes)
      }
    },
    [grouped.lanes, laneMutation, laneRows],
  )

  const handlePortfolioLaneChange = useCallback(
    async (portfolio: SavedPortfolio, lane: ScanLane) => {
      const currentLane = resolvePortfolioScanLane(portfolio)
      if (currentLane === lane) return

      const nextLanes: LaneRows = {
        active: [...laneRows.active],
        testing: [...laneRows.testing],
        archived: [...laneRows.archived],
      }
      for (const sourceLane of SCAN_LANES) {
        nextLanes[sourceLane] = nextLanes[sourceLane].filter(
          (row) => row.portfolio_id !== portfolio.id,
        )
      }
      const movedRow =
        laneRows[currentLane].find((row) => row.portfolio_id === portfolio.id) ??
        laneRows.testing.find((row) => row.portfolio_id === portfolio.id) ??
        laneRows.active.find((row) => row.portfolio_id === portfolio.id) ??
        laneRows.archived.find((row) => row.portfolio_id === portfolio.id)
      if (movedRow) {
        nextLanes[lane] = [...nextLanes[lane], movedRow]
      }
      setLaneRows(nextLanes)
      setExpandedSections((prev) => new Set(prev).add(lane))
      try {
        await portfolioLaneMutation.mutateAsync({
          id: portfolio.id,
          payload: {
            scan_lane: lane,
            scan_sort_order: nextLanes[lane].length - 1,
          },
        })
      } catch {
        setLaneRows(grouped.lanes)
      }
    },
    [grouped.lanes, laneRows, portfolioLaneMutation],
  )

  const handleDragEnd = useCallback(
    async (event: DragEndEvent) => {
      const { active, over } = event
      if (!over) return

      const activeId = String(active.id)
      const overId = String(over.id)
      const sourceLane = findLaneForRow(laneRows, activeId)
      if (!sourceLane) return

      const targetLane = parseLaneContainerId(overId) ?? findLaneForRow(laneRows, overId) ?? sourceLane

      const nextLanes: LaneRows = {
        active: [...laneRows.active],
        testing: [...laneRows.testing],
        archived: [...laneRows.archived],
      }

      const sourceRows = [...nextLanes[sourceLane]]
      const activeIndex = sourceRows.findIndex((row) => row.id === activeId)
      if (activeIndex < 0) return

      if (sourceLane === targetLane && !parseLaneContainerId(overId)) {
        const overIndex = sourceRows.findIndex((row) => row.id === overId)
        if (overIndex < 0 || activeIndex === overIndex) return
        nextLanes[sourceLane] = arrayMove(sourceRows, activeIndex, overIndex)
      } else {
        const [movedRow] = sourceRows.splice(activeIndex, 1)
        nextLanes[sourceLane] = sourceRows
        const targetRows = sourceLane === targetLane ? sourceRows : [...nextLanes[targetLane]]
        if (parseLaneContainerId(overId)) {
          targetRows.push(movedRow)
        } else {
          const overIndex = targetRows.findIndex((row) => row.id === overId)
          targetRows.splice(overIndex >= 0 ? overIndex : targetRows.length, 0, movedRow)
        }
        nextLanes[targetLane] = targetRows
      }

      setLaneRows(nextLanes)
      const touched = new Set<ScanLane>([sourceLane])
      if (targetLane !== sourceLane) touched.add(targetLane)
      try {
        await persistLaneLayout(nextLanes, touched)
      } catch {
        setLaneRows(grouped.lanes)
      }
    },
    [grouped.lanes, laneRows, persistLaneLayout],
  )

  const toggleSection = useCallback((sectionId: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev)
      if (next.has(sectionId)) next.delete(sectionId)
      else next.add(sectionId)
      return next
    })
  }, [])

  const lanePending = laneMutation.isPending || portfolioLaneMutation.isPending

  return {
    query,
    setQuery,
    symbolFilter,
    setSymbolFilter,
    expandedSections,
    toggleSection,
    laneRows,
    legacyRows,
    strategyById,
    portfolioById,
    symbols,
    rows,
    filtered,
    activeSignals,
    isLoading,
    isFetching,
    error,
    refetch,
    handleLaneChange,
    handlePortfolioLaneChange,
    handleDragEnd,
    lanePending,
  }
}
