import { useMutation, useQuery } from "@tanstack/react-query";
import { Loader2, Play } from "lucide-react";
import { FormEvent, useState } from "react";
import type { DetailedResult, RunMode, SweepResult } from "@/api";
import { getSignals, runBacktest } from "@/api";
import { DetailResults } from "@/components/backtest/detail-results";
import { SignalPanel } from "@/components/backtest/signal-panel";
import { SweepResults } from "@/components/backtest/sweep-results";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RUN_MODES } from "@/lib/backtest-config";
import { buildPayload, usesOgRuntimeOptions, type BacktestFormState } from "@/lib/backtest-form";

function isDetailedResult(result: DetailedResult | SweepResult | undefined): result is DetailedResult {
  return Boolean(result && !Array.isArray(result) && "summary" in result);
}

export function BacktestSection() {
  const { data: signals = [] } = useQuery({ queryKey: ["signals"], queryFn: getSignals });
  const mutation = useMutation({
    mutationFn: ({ mode, payload }: { mode: RunMode; payload: Record<string, unknown> }) =>
      runBacktest(mode, payload),
  });

  const [mode, setMode] = useState<RunMode>("single");
  const [symbol, setSymbol] = useState("SOXX");
  const [years, setYears] = useState(25);
  const [signalKind, setSignalKind] = useState<"single" | "combined">("single");
  const [signal, setSignal] = useState("buy_signal7");
  const [signalA, setSignalA] = useState("buy_signal16");
  const [signalB, setSignalB] = useState("buy_signal7");
  const [comboMode, setComboMode] = useState<"and" | "or">("or");
  const [primarySymbol, setPrimarySymbol] = useState("SOXX");
  const [symbolPool, setSymbolPool] = useState("SOXX, SMH, QQQ, SPY");
  const [confirmSymbols, setConfirmSymbols] = useState("SMH, QQQ");
  const [maxDays, setMaxDays] = useState(7);
  const [checkBreadth, setCheckBreadth] = useState(false);
  const [checkBoth, setCheckBoth] = useState(false);
  const [isSell, setIsSell] = useState(false);
  const [mondayBuy, setMondayBuy] = useState(false);
  const [lowVolumeBuy, setLowVolumeBuy] = useState(false);
  const [holdOnBuySignal, setHoldOnBuySignal] = useState(false);
  const [selectedSweepRow, setSelectedSweepRow] = useState<Record<string, unknown> | undefined>();

  const defaultSignal = signals.find((s) => s.name === "buy_signal7")?.name ?? signals[0]?.name ?? "";
  const signalOptions = signals.length ? signals : [{ name: defaultSignal, label: defaultSignal }];
  const selectedRunMode = RUN_MODES.find((item) => item.id === mode)!;

  const formState: BacktestFormState = {
    mode,
    symbol,
    years,
    signalKind,
    signal,
    signalA,
    signalB,
    comboMode,
    primarySymbol,
    symbolPool,
    confirmSymbols,
    maxDays,
    checkBreadth,
    checkBoth,
    isSell,
    mondayBuy,
    lowVolumeBuy,
    holdOnBuySignal,
  };

  const showOgOptions = usesOgRuntimeOptions(formState);
  const showSymbolField = mode !== "symbol-confirm-sweep" && mode !== "symbol-confirm-detail";
  const showSignalPanel = mode !== "signal-combo-sweep";

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSelectedSweepRow(undefined);
    mutation.mutate({ mode, payload: buildPayload(formState) });
  }

  return (
    <div className="flex flex-col gap-6">
      <Card className="border-border/60 p-6">
        <form onSubmit={onSubmit} className="flex flex-col gap-0">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="flex flex-col gap-2">
              <Label htmlFor="run-mode">Run mode</Label>
              <Select value={mode} onValueChange={(v) => v && setMode(v as RunMode)}>
                <SelectTrigger id="run-mode" className="w-full">
                  <SelectValue>
                    {(value: string) => RUN_MODES.find((m) => m.id === value)?.label}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {RUN_MODES.map((m) => (
                    <SelectItem key={m.id} value={m.id}>
                      {m.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs leading-relaxed text-muted-foreground">{selectedRunMode.description}</p>
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="years">Years</Label>
              <Input
                id="years"
                type="number"
                min={1}
                value={years}
                onChange={(e) => setYears(Number(e.target.value))}
                className="font-mono"
              />
            </div>

            {showSymbolField && (
              <div className="flex flex-col gap-2">
                <Label htmlFor="symbol">Symbol</Label>
                <Input
                  id="symbol"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  className="font-mono"
                />
              </div>
            )}
          </div>

          {showSignalPanel && (
            <SignalPanel
              signalKind={signalKind}
              onSignalKindChange={setSignalKind}
              signal={signal}
              onSignalChange={setSignal}
              signalA={signalA}
              onSignalAChange={setSignalA}
              signalB={signalB}
              onSignalBChange={setSignalB}
              comboMode={comboMode}
              onComboModeChange={setComboMode}
              signalOptions={signalOptions}
            />
          )}

          {mode === "signal-combo-sweep" && (
            <div className="mt-5 rounded-lg border border-border/60 bg-muted/30 p-5">
              <h3 className="text-sm font-semibold">Signal Combo Sweep</h3>
              <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
                <div className="flex flex-col gap-2">
                  <Label htmlFor="signal-a-sweep">Signal A</Label>
                  <Select value={signalA} onValueChange={(v) => v && setSignalA(v)}>
                    <SelectTrigger id="signal-a-sweep" className="font-mono">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {signalOptions.map((item) => (
                        <SelectItem key={item.name} value={item.name} className="font-mono">
                          {item.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex flex-col gap-2">
                  <Label htmlFor="signal-b-sweep">Signal B</Label>
                  <Select value={signalB} onValueChange={(v) => v && setSignalB(v)}>
                    <SelectTrigger id="signal-b-sweep" className="font-mono">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {signalOptions.map((item) => (
                        <SelectItem key={item.name} value={item.name} className="font-mono">
                          {item.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          )}

          {showOgOptions && (
            <div className="mt-5 rounded-lg border border-border/60 bg-muted/30 p-5">
              <h3 className="text-sm font-semibold">OG Runtime Options</h3>
              <p className="mt-1 text-xs text-muted-foreground">
                Used only when an OG strategy is the primary strategy path. These override MondayBuy and
                LowVolumeBuy for this run only.
              </p>
              <div className="mt-4 flex flex-wrap gap-6">
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="monday-buy"
                    checked={mondayBuy}
                    onCheckedChange={(checked) => setMondayBuy(checked === true)}
                  />
                  <Label htmlFor="monday-buy" className="font-normal">
                    Monday buy
                  </Label>
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="low-volume-buy"
                    checked={lowVolumeBuy}
                    onCheckedChange={(checked) => setLowVolumeBuy(checked === true)}
                  />
                  <Label htmlFor="low-volume-buy" className="font-normal">
                    Low-volume buy
                  </Label>
                </div>
              </div>
            </div>
          )}

          <div className="mt-5 rounded-lg border border-border/60 bg-muted/30 p-5">
            <h3 className="text-sm font-semibold">Backtest Options</h3>
            <p className="mt-1 text-xs text-muted-foreground">
              Skip sell, hold-days, and profitable-closes exits on days where the entry signal is still
              active.
            </p>
            <div className="mt-4 flex flex-wrap gap-6">
              <div className="flex items-center gap-2">
                <Checkbox
                  id="hold-on-buy-signal"
                  checked={holdOnBuySignal}
                  onCheckedChange={(checked) => setHoldOnBuySignal(checked === true)}
                />
                <Label htmlFor="hold-on-buy-signal" className="font-normal">
                  Hold on buy signal
                </Label>
              </div>
            </div>
          </div>

          {(mode === "symbol-confirm-sweep" || mode === "symbol-confirm-detail") && (
            <div className="mt-5 rounded-lg border border-border/60 bg-muted/30 p-5">
              <h3 className="text-sm font-semibold">Symbol Confirmation</h3>
              <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
                <div className="flex flex-col gap-2">
                  <Label htmlFor="primary-symbol">Primary symbol</Label>
                  <Input
                    id="primary-symbol"
                    value={primarySymbol}
                    onChange={(e) => setPrimarySymbol(e.target.value.toUpperCase())}
                    className="font-mono"
                  />
                </div>
                {mode === "symbol-confirm-sweep" ? (
                  <div className="flex flex-col gap-2">
                    <Label htmlFor="symbol-pool">Symbol pool</Label>
                    <Input
                      id="symbol-pool"
                      value={symbolPool}
                      onChange={(e) => setSymbolPool(e.target.value)}
                      className="font-mono"
                    />
                  </div>
                ) : (
                  <div className="flex flex-col gap-2">
                    <Label htmlFor="confirm-symbols">Confirm symbols</Label>
                    <Input
                      id="confirm-symbols"
                      value={confirmSymbols}
                      onChange={(e) => setConfirmSymbols(e.target.value)}
                      className="font-mono"
                    />
                  </div>
                )}
              </div>
            </div>
          )}

          {mode === "hold-days-sweep" && (
            <div className="mt-5 flex max-w-xs flex-col gap-2">
              <Label htmlFor="max-days">Max hold days</Label>
              <Input
                id="max-days"
                type="number"
                min={1}
                value={maxDays}
                onChange={(e) => setMaxDays(Number(e.target.value))}
                className="font-mono"
              />
            </div>
          )}

          {mode === "indicator-sweep" && (
            <div className="mt-5 rounded-lg border border-border/60 bg-muted/30 p-5">
              <h3 className="text-sm font-semibold">Indicator Sweep</h3>
              <div className="mt-4 flex flex-wrap gap-6">
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="is-sell"
                    checked={isSell}
                    onCheckedChange={(checked) => setIsSell(checked === true)}
                  />
                  <Label htmlFor="is-sell" className="font-normal">
                    Sweep sell filters
                  </Label>
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="check-breadth"
                    checked={checkBreadth}
                    onCheckedChange={(checked) => setCheckBreadth(checked === true)}
                  />
                  <Label htmlFor="check-breadth" className="font-normal">
                    Include breadth
                  </Label>
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="check-both"
                    checked={checkBoth}
                    onCheckedChange={(checked) => setCheckBoth(checked === true)}
                  />
                  <Label htmlFor="check-both" className="font-normal">
                    Include price/momentum
                  </Label>
                </div>
              </div>
            </div>
          )}

          <div className="mt-5">
            <Button type="submit" disabled={mutation.isPending} className="gap-2">
              {mutation.isPending ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Play className="size-4" />
              )}
              {mutation.isPending ? "Running..." : "Run backtest"}
            </Button>
          </div>
        </form>
      </Card>

      {mutation.error && (
        <Card className="border-destructive/50 bg-destructive/5 p-4">
          <pre className="overflow-x-auto whitespace-pre-wrap text-sm text-destructive">
            {String(mutation.error)}
          </pre>
        </Card>
      )}

      {isDetailedResult(mutation.data) && <DetailResults result={mutation.data} />}
      {Array.isArray(mutation.data) && (
        <SweepResults
          rows={mutation.data}
          selectedRow={selectedSweepRow}
          onSelectRow={setSelectedSweepRow}
        />
      )}
    </div>
  );
}
