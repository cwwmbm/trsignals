import type { SavedPortfolio } from '@/api'

export type ChildBlock = 'itself' | 'contains it'

/**
 * Saved portfolios that cannot become a child of `editingId`: the portfolio itself, and every
 * portfolio that already contains it, directly or through other portfolios. Adding either one
 * would create a cycle, which the server rejects on save.
 */
export function blockedChildren(
  editingId: string | undefined,
  portfolios: Array<Pick<SavedPortfolio, 'id' | 'portfolio_ids'>>,
): Map<string, ChildBlock> {
  const blocked = new Map<string, ChildBlock>()
  if (!editingId) return blocked
  blocked.set(editingId, 'itself')
  let frontier = [editingId]
  while (frontier.length > 0) {
    const next: string[] = []
    for (const portfolio of portfolios) {
      if (blocked.has(portfolio.id)) continue
      if ((portfolio.portfolio_ids ?? []).some((id) => frontier.includes(id))) {
        blocked.set(portfolio.id, 'contains it')
        next.push(portfolio.id)
      }
    }
    frontier = next
  }
  return blocked
}
