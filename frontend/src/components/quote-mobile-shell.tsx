'use client'

import { QuoteMobileSection } from '@/components/quote/quote-mobile-section'
import { MobileShellHeader } from '@/components/mobile-shell-header'

export function QuoteMobileShell() {
  return (
    <div className="flex min-h-svh flex-col bg-background text-foreground">
      <MobileShellHeader />
      <main className="flex-1 px-4 pt-3">
        <QuoteMobileSection />
      </main>
    </div>
  )
}
