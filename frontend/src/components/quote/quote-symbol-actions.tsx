'use client'

import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Loader2, Plus, Trash2 } from 'lucide-react'
import { addQuoteSymbol, deleteQuoteSymbol, getQuote, type QuoteSnapshot } from '@/api'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

function actionMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim()) return error.message
  return 'Could not update the quote list'
}

export function useQuoteBoard() {
  const queryClient = useQueryClient()
  const query = useQuery({
    queryKey: ['quote'],
    queryFn: getQuote,
  })
  const [draft, setDraft] = useState('')
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  function storeSnapshot(snapshot: QuoteSnapshot) {
    queryClient.setQueryData(['quote'], snapshot)
    setActionError(null)
  }

  const addMutation = useMutation({
    mutationFn: addQuoteSymbol,
    onMutate: async () => {
      await queryClient.cancelQueries({ queryKey: ['quote'] })
    },
    onSuccess: (snapshot) => {
      storeSnapshot(snapshot)
      setDraft('')
    },
    onError: (error) => {
      setActionError(actionMessage(error))
      queryClient.invalidateQueries({ queryKey: ['quote'] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteQuoteSymbol,
    onMutate: async () => {
      await queryClient.cancelQueries({ queryKey: ['quote'] })
    },
    onSuccess: (snapshot) => {
      storeSnapshot(snapshot)
      setPendingDelete(null)
    },
    onError: (error) => {
      setActionError(actionMessage(error))
      setPendingDelete(null)
      queryClient.invalidateQueries({ queryKey: ['quote'] })
    },
  })

  const quotes = query.data?.quotes ?? []

  return {
    quotes,
    asOf: query.data?.as_of ?? null,
    missingCount: quotes.reduce((count, row) => count + row.missing_days.length, 0),
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    error: query.error,
    refetch: query.refetch,
    draft,
    setDraft: (value: string) => {
      setDraft(value.toUpperCase())
      if (actionError) setActionError(null)
    },
    addSymbol: () => {
      const symbol = draft.trim()
      if (!symbol || addMutation.isPending) return
      addMutation.mutate(symbol)
    },
    addPending: addMutation.isPending,
    actionError,
    pendingDelete,
    deletePendingSymbol: deleteMutation.isPending ? deleteMutation.variables : null,
    requestDelete: (symbol: string) => {
      if (deleteMutation.isPending) return
      if (pendingDelete === symbol) {
        deleteMutation.mutate(symbol)
        return
      }
      setPendingDelete(symbol)
    },
  }
}

export type QuoteBoard = ReturnType<typeof useQuoteBoard>

export function QuoteAddForm({
  board,
  className,
  inputClassName,
  buttonClassName,
}: {
  board: QuoteBoard
  className?: string
  inputClassName?: string
  buttonClassName?: string
}) {
  return (
    <form
      className={cn('flex items-center gap-1.5', className)}
      onSubmit={(event) => {
        event.preventDefault()
        board.addSymbol()
      }}
    >
      <Input
        value={board.draft}
        onChange={(event) => board.setDraft(event.target.value)}
        placeholder="Add symbol"
        aria-label="Add symbol"
        autoCapitalize="characters"
        autoComplete="off"
        spellCheck={false}
        disabled={board.addPending}
        className={cn('w-28 font-mono uppercase', inputClassName)}
      />
      <Button
        type="submit"
        size="sm"
        className={cn('h-7 gap-1 px-2 text-xs', buttonClassName)}
        disabled={board.addPending || board.draft.trim().length === 0}
      >
        {board.addPending ? <Loader2 className="size-3 animate-spin" /> : <Plus className="size-3" />}
        Add
      </Button>
    </form>
  )
}

export function QuoteRemoveButton({
  symbol,
  board,
  className,
}: {
  symbol: string
  board: QuoteBoard
  className?: string
}) {
  const confirming = board.pendingDelete === symbol
  const pending = board.deletePendingSymbol === symbol
  return (
    <Button
      type="button"
      variant={confirming ? 'destructive' : 'ghost'}
      size="sm"
      className={cn('h-6 gap-1 px-1.5 text-[11px]', className)}
      disabled={pending}
      aria-label={confirming ? `Confirm remove ${symbol}` : `Remove ${symbol}`}
      onClick={() => board.requestDelete(symbol)}
    >
      {pending ? <Loader2 className="size-3 animate-spin" /> : <Trash2 className="size-3" />}
      {confirming ? 'Remove' : null}
    </Button>
  )
}
