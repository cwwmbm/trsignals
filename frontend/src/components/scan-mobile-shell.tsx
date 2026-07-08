'use client'

import { Radar } from 'lucide-react'
import { ScanMobileSection } from '@/components/scan/scan-mobile-section'
import { ThemeToggle } from '@/components/theme-toggle'

export function ScanMobileShell() {
  return (
    <div className="flex min-h-svh flex-col bg-background text-foreground">
      <header className="sticky top-0 z-10 flex items-center justify-between border-b border-border/60 bg-background/95 px-4 py-3 backdrop-blur supports-[backdrop-filter]:bg-background/80">
        <div className="flex items-center gap-2">
          <Radar className="size-5 text-primary" />
          <div>
            <h1 className="text-base font-semibold leading-tight">Scan</h1>
            <p className="text-[11px] text-muted-foreground">Latest signals</p>
          </div>
        </div>
        <ThemeToggle />
      </header>
      <main className="flex-1 px-4 pt-3">
        <ScanMobileSection />
      </main>
    </div>
  )
}
