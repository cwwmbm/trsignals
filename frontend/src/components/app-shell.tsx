'use client'

import { useState } from 'react'
import {
  CandlestickChart,
  FlaskConical,
  HelpCircle,
  ListChecks,
  Menu,
  Radar,
  Wrench,
  X,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { ThemeToggle } from '@/components/theme-toggle'
import type { SavedStrategy } from '@/api'
import { BacktestSection } from '@/components/sections/backtest-section'
import { ScanSection } from '@/components/sections/scan-section'
import { StrategiesSection } from '@/components/sections/strategies-section'
import { StrategyBuilderSection } from '@/components/sections/strategy-builder-section'
import { HelpSection } from '@/components/sections/help-section'

type SectionId = 'backtest' | 'strategies' | 'builder' | 'scan' | 'help'

const NAV: { id: SectionId; label: string; icon: typeof FlaskConical }[] = [
  { id: 'backtest', label: 'Backtest', icon: FlaskConical },
  { id: 'strategies', label: 'All strategies', icon: ListChecks },
  { id: 'builder', label: 'New strategy builder', icon: Wrench },
  { id: 'scan', label: 'Scan', icon: Radar },
  { id: 'help', label: 'Help', icon: HelpCircle },
]

const TITLES: Record<SectionId, { title: string; subtitle: string }> = {
  backtest: { title: 'Backtest', subtitle: 'Run historical simulations and inspect detailed performance.' },
  strategies: { title: 'All strategies', subtitle: 'Every saved strategy with its key metrics.' },
  builder: { title: 'New strategy builder', subtitle: 'Compose entry rules and save a new signal.' },
  scan: { title: 'Scan', subtitle: 'Latest signal scan across the symbol universe.' },
  help: { title: 'Help', subtitle: 'Documentation, run modes, and glossary.' },
}

export function AppShell() {
  const [active, setActive] = useState<SectionId>('backtest')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [builderInitialStrategy, setBuilderInitialStrategy] = useState<SavedStrategy | undefined>()

  const goTo = (id: SectionId) => {
    if (id === 'builder') {
      setBuilderInitialStrategy(undefined)
    }
    setActive(id)
    setMobileOpen(false)
  }

  return (
    <div className="flex min-h-screen bg-background">
      {/* Sidebar */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-border/60 bg-sidebar transition-transform lg:static lg:translate-x-0',
          mobileOpen ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        <div className="flex h-16 items-center gap-2.5 border-b border-border/60 px-5">
          <span className="flex size-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <CandlestickChart className="size-5" />
          </span>
          <div className="flex flex-col leading-none">
            <span className="text-sm font-semibold tracking-tight">TradingStrategy</span>
            <span className="text-[11px] text-muted-foreground">Backtest Studio</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="ml-auto size-8 lg:hidden"
            onClick={() => setMobileOpen(false)}
            aria-label="Close menu"
          >
            <X className="size-4" />
          </Button>
        </div>

        <nav className="flex flex-1 flex-col gap-1 p-3">
          <span className="px-2 py-2 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Workspace
          </span>
          {NAV.map((item) => {
            const isActive = active === item.id
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => goTo(item.id)}
                className={cn(
                  'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-sidebar-accent text-foreground'
                    : 'text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground',
                )}
                aria-current={isActive ? 'page' : undefined}
              >
                <item.icon
                  className={cn('size-4', isActive ? 'text-primary' : 'text-muted-foreground')}
                />
                {item.label}
              </button>
            )
          })}
        </nav>

        <div className="border-t border-border/60 p-4">
          <div className="rounded-lg border border-border/60 bg-card/40 p-3">
            <div className="flex items-center gap-2">
              <span className="size-2 rounded-full bg-[var(--gain)]" />
              <span className="text-xs font-medium">Data feed connected</span>
            </div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              Last sync · today, 09:47 ET
            </p>
          </div>
        </div>
      </aside>

      {/* Backdrop for mobile */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-30 bg-background/70 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
          aria-hidden
        />
      )}

      {/* Main */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex h-16 items-center gap-3 border-b border-border/60 bg-background/80 px-5 backdrop-blur lg:px-8">
          <Button
            variant="ghost"
            size="icon"
            className="size-8 lg:hidden"
            onClick={() => setMobileOpen(true)}
            aria-label="Open menu"
          >
            <Menu className="size-4" />
          </Button>
          <div className="flex flex-col leading-tight">
            <h1 className="text-base font-semibold">{TITLES[active].title}</h1>
            <p className="hidden text-xs text-muted-foreground sm:block">
              {TITLES[active].subtitle}
            </p>
          </div>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </header>

        <main className="flex-1 p-5 lg:p-8">
          {active === 'backtest' && <BacktestSection />}
          {active === 'strategies' && (
            <StrategiesSection
              onNewStrategy={() => {
                setBuilderInitialStrategy(undefined)
                setActive('builder')
              }}
              onBacktestStrategy={(strategy) => {
                setBuilderInitialStrategy(strategy)
                setActive('builder')
              }}
            />
          )}
          {active === 'builder' && <StrategyBuilderSection initialStrategy={builderInitialStrategy} />}
          {active === 'scan' && <ScanSection />}
          {active === 'help' && <HelpSection />}
        </main>
      </div>
    </div>
  )
}
