import { Loader2, Play } from "lucide-react";
import { useEffect, useMemo, type ReactNode } from "react";
import type { SavedStrategy } from "@/api";
import { useStrategyBuilderDraftPreview } from "@/lib/strategy-builder-draft-store";
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
import {
  BUILDER_REFINE_MODES,
  isSymbolConfirmRefineMode,
  type BuilderRefineMode,
} from "@/lib/builder-refine-config";
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

function DraftStrategySection() {
  const {
    draftName,
    draftSymbol,
    draftHoldDays,
    draftProfit,
    entryPreview,
    exitPreview,
    draftValid,
  } = useStrategyBuilderDraftPreview();

  return (
    <Section title="Draft strategy (primary)">
      <div className="grid gap-2 sm:grid-cols-2">
        <div>
          <p className="text-xs font-medium text-foreground">
            {draftName.trim() || "Untitled draft"}
          </p>
          <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">
            {draftSymbol} · {draftHoldDays}d hold · {draftProfit} profit close
            {draftProfit === 1 ? "" : "s"}
          </p>
        </div>
        <div className="space-y-1">
          <p className="text-[11px] text-muted-foreground">Entry</p>
          <pre className="overflow-x-auto rounded border border-border/50 bg-background/60 px-2 py-1 font-mono text-[10px] leading-snug">
            {entryPreview || "—"}
          </pre>
          {exitPreview ? (
            <>
              <p className="text-[11px] text-muted-foreground">Exit</p>
              <pre className="overflow-x-auto rounded border border-border/50 bg-background/60 px-2 py-1 font-mono text-[10px] leading-snug">
                {exitPreview}
              </pre>
            </>
          ) : null}
        </div>
      </div>
      {!draftValid ? (
        <p className="mt-2 text-[11px] text-destructive">
          Add at least one valid entry condition before running refinement.
        </p>
      ) : null}
    </Section>
  );
}

export type BuilderRefinePanelProps = {
  mode: BuilderRefineMode;
  onModeChange: (mode: BuilderRefineMode) => void;
  savedStrategies: SavedStrategy[];
  secondaryStrategyId: string;
  onSecondaryStrategyChange: (id: string) => void;
  primarySymbol: string;
  onPrimarySymbolChange: (value: string) => void;
  symbolPool: string;
  onSymbolPoolChange: (value: string) => void;
  maxDays: number;
  onMaxDaysChange: (value: number) => void;
  checkBreadth: boolean;
  onCheckBreadthChange: (value: boolean) => void;
  checkBoth: boolean;
  onCheckBothChange: (value: boolean) => void;
  isSell: boolean;
  onIsSellChange: (value: boolean) => void;
  onRun: () => void;
  isRunning: boolean;
};

export function BuilderRefinePanel({
  mode,
  onModeChange,
  savedStrategies,
  secondaryStrategyId,
  onSecondaryStrategyChange,
  primarySymbol,
  onPrimarySymbolChange,
  symbolPool,
  onSymbolPoolChange,
  maxDays,
  onMaxDaysChange,
  checkBreadth,
  onCheckBreadthChange,
  checkBoth,
  onCheckBothChange,
  isSell,
  onIsSellChange,
  onRun,
  isRunning,
}: BuilderRefinePanelProps) {
  const { draftSymbol, draftValid } = useStrategyBuilderDraftPreview();
  const selectedMode = BUILDER_REFINE_MODES.find((item) => item.id === mode)!;

  const comboSecondaryStrategies = useMemo(
    () => savedStrategies.filter((item) => item.symbol === draftSymbol),
    [savedStrategies, draftSymbol],
  );

  useEffect(() => {
    if (mode !== "signal-combo-sweep") return;
    const currentValid = comboSecondaryStrategies.some(
      (item) => item.id === secondaryStrategyId,
    );
    if (currentValid) return;
    onSecondaryStrategyChange(comboSecondaryStrategies[0]?.id ?? "");
  }, [mode, comboSecondaryStrategies, secondaryStrategyId, onSecondaryStrategyChange]);

  const comboReady = mode !== "signal-combo-sweep" || Boolean(secondaryStrategyId);

  return (
    <div className="flex flex-col gap-3">
      <DraftStrategySection />

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <Field label="Run mode" htmlFor="builder-run-mode">
          <Select value={mode} onValueChange={(v) => v && onModeChange(v as BuilderRefineMode)}>
            <SelectTrigger id="builder-run-mode" size="sm" className="h-8 w-full text-sm">
              <SelectValue>
                {(value: string) => BUILDER_REFINE_MODES.find((m) => m.id === value)?.label}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {BUILDER_REFINE_MODES.map((item) => (
                <SelectItem key={item.id} value={item.id}>
                  {item.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <p className="self-end text-[11px] leading-snug text-muted-foreground md:pb-1">
          {selectedMode.description}
        </p>
      </div>

      {mode === "signal-combo-sweep" && (
        <Section title="Secondary strategy">
          <Field label="Compare draft against" htmlFor="secondary-strategy">
            <Select
              value={secondaryStrategyId}
              onValueChange={(v) => v && onSecondaryStrategyChange(v)}
            >
              <SelectTrigger id="secondary-strategy" size="sm" className="h-8 w-full text-sm">
                <SelectValue placeholder="Select a saved strategy">
                  {(value: string) => {
                    const match = savedStrategies.find((item) => item.id === value);
                    return match ? `${match.name} · ${match.symbol}` : "Select a saved strategy";
                  }}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {comboSecondaryStrategies.length === 0 ? (
                  <SelectItem value="__none" disabled>
                    No saved strategies for {draftSymbol}
                  </SelectItem>
                ) : (
                  comboSecondaryStrategies.map((item) => (
                    <SelectItem key={item.id} value={item.id}>
                      {item.name} · {item.symbol}
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
          </Field>
        </Section>
      )}

      {isSymbolConfirmRefineMode(mode) && (
        <Section title="Symbol confirmation">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Field label="Primary symbol" htmlFor="builder-primary-symbol">
              <Input
                id="builder-primary-symbol"
                value={primarySymbol}
                onChange={(e) => onPrimarySymbolChange(e.target.value.toUpperCase())}
                className="h-8 font-mono text-sm"
              />
            </Field>
            <Field label="Symbol pool" htmlFor="builder-symbol-pool">
              <Input
                id="builder-symbol-pool"
                value={symbolPool}
                onChange={(e) => onSymbolPoolChange(e.target.value)}
                className="h-8 font-mono text-sm"
              />
            </Field>
          </div>
        </Section>
      )}

      {mode === "hold-days-sweep" && (
        <Field label="Max hold days" htmlFor="builder-max-days" className="max-w-xs">
          <Input
            id="builder-max-days"
            type="number"
            min={1}
            value={maxDays}
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
                id="builder-is-sell"
                checked={isSell}
                onCheckedChange={(checked) => onIsSellChange(checked === true)}
              />
              <Label htmlFor="builder-is-sell" className="text-xs font-normal">
                Sweep sell filters
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id="builder-check-breadth"
                checked={checkBreadth}
                onCheckedChange={(checked) => onCheckBreadthChange(checked === true)}
              />
              <Label htmlFor="builder-check-breadth" className="text-xs font-normal">
                Include breadth
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id="builder-check-both"
                checked={checkBoth}
                onCheckedChange={(checked) => onCheckBothChange(checked === true)}
              />
              <Label htmlFor="builder-check-both" className="text-xs font-normal">
                Include price/momentum
              </Label>
            </div>
          </div>
        </Section>
      )}

      <div>
        <Button
          type="button"
          size="sm"
          className="h-8 gap-1.5"
          disabled={isRunning || !draftValid || !comboReady}
          onClick={onRun}
        >
          {isRunning ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />}
          {isRunning ? "Running…" : "Run refinement"}
        </Button>
      </div>
    </div>
  );
}
