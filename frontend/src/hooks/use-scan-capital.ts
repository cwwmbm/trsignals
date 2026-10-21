import { useCallback, useState } from 'react'
import { parseCapital } from '@/lib/scan-position'

const KEY = 'scan-capital'

function read(): string {
  try {
    return window.localStorage.getItem(KEY) ?? ''
  } catch {
    return ''
  }
}

/** The account size typed on the Scan page. Kept in this browser only; the server never sees it. */
export function useScanCapital() {
  const [text, setTextState] = useState<string>(read)
  const setText = useCallback((value: string) => {
    setTextState(value)
    try {
      window.localStorage.setItem(KEY, value)
    } catch {
      /* private window or blocked storage: it still works until the page is closed */
    }
  }, [])
  return { text, setText, capital: parseCapital(text) }
}
