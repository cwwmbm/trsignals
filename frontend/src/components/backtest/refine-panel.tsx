import { Loader2, Play } from "lucide-react";
import type { ReactNode } from "react";
import type { RunMode, SignalInfo } from "@/api";
import { SignalPanel } from "@/components/backtest/signal-panel";
import { Button } from "@/components/ui/button";
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
import { RUN_MODES, showSignalPanel } from "@/lib/backtest-config";
import { usesOgRuntimeOptions, type BacktestFormState } from "@/lib/backtest-form";
import { cn } from "@/lib/utils";

function Field({
  label,
  htmlFor,
  children,
  className,
}: {
  label: string;
  htmlFor?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex min-w-0 flex-col gap-1", className)}>
      <Label htmlFor={htmlFor} className="text-[11px] text-muted-foreground">
        {label}
      </Label>
      {children}
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-md border border-border/50 bg-muted/15 p-3">
      <h3 className="text-xs font-medium">{title}</h3>
      <div className="mt-2">{children}</div>
    </div>
  );
}

export type BacktestRefinePanelProps = {
  formState: BacktestFormState;
  signalOptions: SignalInfo[];
  onModeChange: (mode: RunMode) => void;
  onSignalKindChange: (kind: "single" | "combined") => void;
  onSignalChange: (signal: string) => void;
  onSignalAChange: (signal: string) => void;
  onSignalBChange: (signal: string) => void;
  onComboModeChange: (mode: "and" | "or") => void;
  onPrimarySymbolChange: (value: string) => void;
  onSymbolPoolChange: (value: string) => void;
  onConfirmSymbolsChange: (value: string) => void;
  onMaxDaysChange: (value: number) => void;
  onCheckBreadthChange: (value: boolean) => void;
  onCheckBothChange: (value: boolean) => void;
  onIsSellChange: (value: boolean) => void;
  onMondayBuyChange: (value: boolean) => void;
  onLowVolumeBuyChange: (value: boolean) => void;
  onHoldOnBuySignalChange: (value: boolean) => void;
  onRun: () => void;
  isRunning: boolean;
};

export function BacktestRefinePanel({
  formState,
  signalOptions,
  onModeChange,
  onSignalKindChange,
  onSignalChange,
  onSignalAChange,
  onSignalBChange,
  onComboModeChange,
  onPrimarySymbolChange,
  onSymbolPoolChange,
  onConfirmSymbolsChange,
  onMaxDaysChange,
  onCheckBreadthChange,
  onCheckBothChange,
  onIsSellChange,
  onMondayBuyChange,
  onLowVolumeBuyChange,
  onHoldOnBuySignalChange,
  onRun,
  isRunning,
}: BacktestRefinePanelProps) {
  const { mode } = formState;
  const selectedRunMode = RUN_MODES.find((item) => item.id === mode)!;
  const showOgOptions = usesOgRuntimeOptions(formState);

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <Field label="Run mode" htmlFor="run-mode">
          <Select value={mode} onValueChange={(v) => v && onModeChange(v as RunMode)}>
            <SelectTrigger id="run-mode" size="sm" className="h-8 w-full text-sm">
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
        </Field>
        <p className="self-end text-[11px] leading-snug text-muted-foreground md:pb-1">
          {selectedRunMode.description}
        </p>
      </div>

      {showSignalPanel(mode) && (
        <SignalPanel
          signalKind={formState.signalKind}
          onSignalKindChange={onSignalKindChange}
          signal={formState.signal}
          onSignalChange={onSignalChange}
          signalA={formState.signalA}
          onSignalAChange={onSignalAChange}
          signalB={formState.signalB}
          onSignalBChange={onSignalBChange}
          comboMode={formState.comboMode}
          onComboModeChange={onComboModeChange}
          signalOptions={signalOptions}
        />
      )}

      {mode === "signal-combo-sweep" && (
        <Section title="Signal combo sweep">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Field label="Signal A" htmlFor="signal-a-sweep">
              <Select value={formState.signalA} onValueChange={(v) => v && onSignalAChange(v)}>
                <SelectTrigger id="signal-a-sweep" size="sm" className="h-8 font-mono text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {signalOptions.map((item) => (
                    <SelectItem key={item.name} value={item.name} className="font-mono text-xs">
                      {item.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <Field label="Signal B" htmlFor="signal-b-sweep">
              <Select value={formState.signalB} onValueChange={(v) => v && onSignalBChange(v)}>
                <SelectTrigger id="signal-b-sweep" size="sm" className="h-8 font-mono text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {signalOptions.map((item) => (
                    <SelectItem key={item.name} value={item.name} className="font-mono text-xs">
                      {item.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          </div>
        </Section>
      )}

      {(mode === "symbol-confirm-sweep" || mode === "symbol-confirm-detail") && (
        <Section title="Symbol confirmation">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Field label="Primary symbol" htmlFor="primary-symbol">
              <Input
                id="primary-symbol"
                value={formState.primarySymbol}
                onChange={(e) => onPrimarySymbolChange(e.target.value.toUpperCase())}
                className="h-8 font-mono text-sm"
              />
            </Field>
            {mode === "symbol-confirm-sweep" ? (
              <Field label="Symbol pool" htmlFor="symbol-pool">
                <Input
                  id="symbol-pool"
                  value={formState.symbolPool}
                  onChange={(e) => onSymbolPoolChange(e.target.value)}
                  className="h-8 font-mono text-sm"
                />
              </Field>
            ) : (
              <Field label="Confirm symbols" htmlFor="confirm-symbols">
                <Input
                  id="confirm-symbols"
                  value={formState.confirmSymbols}
                  onChange={(e) => onConfirmSymbolsChange(e.target.value)}
                  className="h-8 font-mono text-sm"
                />
              </Field>
            )}
          </div>
        </Section>
      )}

      {mode === "hold-days-sweep" && (
        <Field label="Max hold days" htmlFor="max-days" className="max-w-xs">
          <Input
            id="max-days"
            type="number"
            min={1}
            value={formState.maxDays}
            onChange={(e) => onMaxDaysChange(Number(e.target.value))}
            className="h-8 font-mono text-sm"
          />
        </Field>
      )}

      {mode === "indicator-sweep" && (
        <Section title="Indicator sweep">
          <div className="flex flex-wrap gap-4">
            <div className="flex items-center gap-2">
              <Checkbox
                id="is-sell"
                checked={formState.isSell}
                onCheckedChange={(checked) => onIsSellChange(checked === true)}
              />
              <Label htmlFor="is-sell" className="text-xs font-normal">
                Sweep sell filters
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id="check-breadth"
                checked={formState.checkBreadth}
                onCheckedChange={(checked) => onCheckBreadthChange(checked === true)}
              />
              <Label htmlFor="check-breadth" className="text-xs font-normal">
                Include breadth
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id="check-both"
                checked={formState.checkBoth}
                onCheckedChange={(checked) => onCheckBothChange(checked === true)}
              />
              <Label htmlFor="check-both" className="text-xs font-normal">
                Include price/momentum
              </Label>
            </div>
          </div>
        </Section>
      )}

      {showOgOptions && (
        <Section title="OG runtime options">
          <p className="mb-2 text-[11px] text-muted-foreground">
            Override MondayBuy and LowVolumeBuy for this run only.
          </p>
          <div className="flex flex-wrap gap-4">
            <div className="flex items-center gap-2">
              <Checkbox
                id="refine-monday-buy"
                checked={formState.mondayBuy}
                onCheckedChange={(checked) => onMondayBuyChange(checked === true)}
              />
              <Label htmlFor="refine-monday-buy" className="text-xs font-normal">
                Monday buy
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id="refine-low-volume-buy"
                checked={formState.lowVolumeBuy}
                onCheckedChange={(checked) => onLowVolumeBuyChange(checked === true)}
              />
              <Label htmlFor="refine-low-volume-buy" className="text-xs font-normal">
                Low-volume buy
              </Label>
            </div>
          </div>
        </Section>
      )}

      <Section title="Backtest options">
        <p className="mb-2 text-[11px] text-muted-foreground">
          Skip sell, hold-days, and profitable-closes exits while entry conditions remain true.
        </p>
        <div className="flex flex-wrap gap-4">
          <div className="flex items-center gap-2">
            <Checkbox
              id="refine-hold-on-buy-signal"
              checked={formState.holdOnBuySignal}
              onCheckedChange={(checked) => onHoldOnBuySignalChange(checked === true)}
            />
            <Label htmlFor="refine-hold-on-buy-signal" className="text-xs font-normal">
              Hold on buy signal
            </Label>
          </div>
        </div>
      </Section>

      <div>
        <Button
          type="button"
          size="sm"
          className="h-8 gap-1.5"
          disabled={isRunning}
          onClick={onRun}
        >
          {isRunning ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />}
          {isRunning ? "Running…" : "Run backtest"}
        </Button>
      </div>
    </div>
  );
}
