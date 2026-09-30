'use client'

import { Link, useLocation } from 'react-router-dom'
import { ThemeToggle } from '@/components/theme-toggle'
import { cn } from '@/lib/utils'

const TABS = [
  { to: '/scan', label: 'Scan' },
  { to: '/quote', label: 'Quote' },
] as const

export function MobileShellHeader() {
  const { pathname } = useLocation()

  return (
    <header className="sticky top-0 z-10 flex items-center justify-between border-b border-border/60 bg-background/95 px-4 py-3 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <nav className="flex gap-1">
        {TABS.map((tab) => {
          const isActive = pathname === tab.to
          return (
            <Link
              key={tab.to}
              to={tab.to}
              className={cn(
                'rounded-full px-3 py-1.5 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground',
              )}
            >
              {tab.label}
            </Link>
          )
        })}
      </nav>
      <ThemeToggle />
    </header>
  )
}
