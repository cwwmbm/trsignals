const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type SignalExpression =
  | { kind: "single"; name: string }
  | { kind: "combined"; primary: string; secondary: string; mode: "and" | "or" };

export type RunMode =
  | "single"
  | "signal-combo-sweep"
  | "symbol-confirm-sweep"
  | "symbol-confirm-detail"
  | "hold-days-sweep"
  | "indicator-sweep";

export interface SignalInfo {
  name: string;
  label: string;
}

export interface DetailedResult {
  summary: Record<string, number | string | null>;
  yearly: Array<Record<string, number | string | null>>;
  equity_curve: Array<{ date: string; rolling_pnl: number; drawdown: number }>;
  trades: Array<Record<string, number | string | null>>;
}

export type SweepResult = Array<Record<string, number | string | null>>;

export async function getSignals(): Promise<SignalInfo[]> {
  const response = await fetch(`${API_URL}/signals`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export function runBacktest(mode: RunMode, payload: Record<string, unknown>) {
  const pathByMode: Record<RunMode, string> = {
    single: "/backtests/single",
    "signal-combo-sweep": "/backtests/signal-combo-sweep",
    "symbol-confirm-sweep": "/backtests/symbol-confirm-sweep",
    "symbol-confirm-detail": "/backtests/symbol-confirm-detail",
    "hold-days-sweep": "/backtests/hold-days-sweep",
    "indicator-sweep": "/backtests/indicator-sweep",
  };

  return postJson<DetailedResult | SweepResult>(pathByMode[mode], payload);
}
