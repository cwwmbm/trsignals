import { Loader2, Play } from "lucide-react";
import { useMemo, type ReactNode } from "react";
import type { SavedStrategy } from "@/api";
import { useStrategyBuilderDraftPreview } from "@/lib/strategy-builder-draft-store";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
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
  refineModesForCustomData,
  type BuilderRefineMode,
} from "@/lib/builder-refine-config";
import { BASKET_PRESETS, parseBasket } from "@/lib/baskets";
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
  comboSize: 2 | 3;
  onComboSizeChange: (value: 2 | 3) => void;
  basketSymbols: string;
  onBasketSymbolsChange: (value: string) => void;
  onRun: () => void;
  isRunning: boolean;
  isCustomData?: boolean;
};

export function BuilderRefinePanel({
  mode,
  onModeChange,
  savedStrategies,
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
  comboSize,
  onComboSizeChange,
  basketSymbols,
  onBasketSymbolsChange,
  onRun,
  isRunning,
  isCustomData = false,
}: BuilderRefinePanelProps) {
  const { draftSymbol, draftValid, customDataBacktestLabel } = useStrategyBuilderDraftPreview();
  const modeOptions = isCustomData ? refineModesForCustomData() : BUILDER_REFINE_MODES;
  const selectedMode = modeOptions.find((item) => item.id === mode) ?? modeOptions[0];

  const comboSecondaryStrategies = useMemo(
    () => savedStrategies.filter((item) => item.symbol === draftSymbol),
    [savedStrategies, draftSymbol],
  );

  const comboReady = mode !== "signal-combo-sweep" || comboSecondaryStrategies.length > 0;
  const basketCount = useMemo(
    () => parseBasket(basketSymbols).filter((symbol) => symbol !== draftSymbol).length,
    [basketSymbols, draftSymbol],
  );
  const basketReady = mode !== "basket-sweep" || basketCount > 0;

  return (
    <div className="flex flex-col gap-3">
      {isCustomData ? (
        <p className="text-xs text-muted-foreground">
          Custom intraday data: symbol confirmation is unavailable. Indicator sweeps skip VIX,
          breadth, and reference-market filters.
          {customDataBacktestLabel ? (
            <>
              {' '}
              Backtest and refine use <span className="font-medium text-foreground">{customDataBacktestLabel}</span>
              (set via Intraday options above).
            </>
          ) : null}
        </p>
      ) : null}
      <DraftStrategySection />

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <Field label="Run mode" htmlFor="builder-run-mode">
          <Select value={mode} onValueChange={(v) => v && onModeChange(v as BuilderRefineMode)}>
            <SelectTrigger id="builder-run-mode" size="sm" className="h-8 w-full text-sm">
              <SelectValue>
                {(value: string) => modeOptions.find((m) => m.id === value)?.label}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {modeOptions.map((item) => (
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
        <Section title="Signal combo sweep">
          <p className="text-xs leading-relaxed text-muted-foreground">
            {comboSecondaryStrategies.length > 0
              ? `Will sweep draft AND/OR against ${comboSecondaryStrategies.length} saved strateg${comboSecondaryStrategies.length === 1 ? "y" : "ies"} for ${draftSymbol}.`
              : `No saved strategies found for ${draftSymbol}. Save a strategy for this symbol before running a combo sweep.`}
          </p>
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

      {mode === "basket-sweep" && (
        <Section title="Basket test">
          <div className="flex flex-col gap-2">
            <Field label="Symbols (comma or space separated)" htmlFor="builder-basket-symbols">
              <Textarea
                id="builder-basket-symbols"
                value={basketSymbols}
                onChange={(e) => onBasketSymbolsChange(e.target.value.toUpperCase())}
                rows={2}
                className="min-h-14 font-mono text-sm"
              />
            </Field>
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[11px] text-muted-foreground">Presets</span>
              {BASKET_PRESETS.map((preset) => (
                <Button
                  key={preset.id}
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-6 px-2 text-[11px]"
                  onClick={() => onBasketSymbolsChange(preset.symbols.join(", "))}
                >
                  {preset.label}
                </Button>
              ))}
            </div>
            <p className="text-[11px] leading-snug text-muted-foreground">
              {basketCount > 0
                ? `${basketCount} symbol${basketCount === 1 ? "" : "s"} besides ${draftSymbol}, which is run as the reference row.`
                : "Add at least one symbol other than the draft's own."}{" "}
              Each symbol is loaded and cached on first use, so a new basket can take a few seconds per
              symbol. A confirm symbol equal to a basket symbol is dropped for that symbol, and the proxy
              symbol is ignored. Stocks in a basket tend to move together, so treat the rows as related
              checks, not independent proof.
            </p>
          </div>
        </Section>
      )}

      {mode === "hold-days-sweep" && (
        <Field
          label={isCustomData ? "Max hold bars" : "Max hold days"}
          htmlFor="builder-max-days"
          className="max-w-xs"
        >
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

      {mode === "market-regime-sweep" && (
        <Section title="Market regime sweep">
          <p className="text-xs leading-relaxed text-muted-foreground">
            Runs the draft strategy once for every market regime state, both ways: “Is” (only that
            state) and “Is not” (every other labeled state). Regimes are ANDed with the entry
            conditions, so each row is the strategy restricted that way. Regimes the draft already
            uses are skipped. Use “Add to market regimes” on a row to keep it.
          </p>
        </Section>
      )}

      {(mode === "indicator-sweep" || mode === "indicator-combo-sweep") && (
        <Section title={mode === "indicator-combo-sweep" ? "Indicator combo sweep" : "Indicator sweep"}>
          <div className="flex flex-wrap gap-4">
            {mode === "indicator-sweep" ? (
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
            ) : (
              <Field label="Filters per combination" htmlFor="builder-combo-size">
                <Select
                  value={String(comboSize)}
                  onValueChange={(v) => v && onComboSizeChange(v === "3" ? 3 : 2)}
                >
                  <SelectTrigger id="builder-combo-size" size="sm" className="h-8 w-32 text-sm">
                    <SelectValue>{(value: string) => (value === "3" ? "Trios" : "Pairs")}</SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="2">Pairs</SelectItem>
                    <SelectItem value="3">Trios</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
            )}
            <div className="flex items-center gap-2">
              <Checkbox
                id="builder-check-breadth"
                checked={checkBreadth}
                disabled={isCustomData}
                onCheckedChange={(checked) => onCheckBreadthChange(checked === true)}
              />
              <Label htmlFor="builder-check-breadth" className="text-xs font-normal">
                Include breadth{isCustomData ? " (unavailable)" : ""}
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
          {mode === "indicator-combo-sweep" ? (
            <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
              Entry filters only. With confirmation symbols, only market-wide indicators are
              combined. Searching many combinations finds lucky ones: compare the in-sample row
              with its full-period figures.
            </p>
          ) : null}
        </Section>
      )}

      <div>
        <Button
          type="button"
          size="sm"
          className="h-8 gap-1.5"
          disabled={isRunning || !draftValid || !comboReady || !basketReady}
          onClick={onRun}
        >
          {isRunning ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />}
          {isRunning ? "Running…" : "Run refinement"}
        </Button>
      </div>
    </div>
  );
}
