import { useMutation } from "@tanstack/react-query";
import { Loader2, Play } from "lucide-react";
import { useState, type FormEvent } from "react";
import { analyzeMarketRegime } from "@/api";
import { RegimeYearTable } from "@/components/backtest/regime-year-table";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { MARKET_WIDE_REGIMES } from "@/lib/market-regimes";

export function RegimesSection() {
  const [regimeId, setRegimeId] = useState(MARKET_WIDE_REGIMES[0].id);
  const [years, setYears] = useState("25");
  const mutation = useMutation({ mutationFn: analyzeMarketRegime });
  const selected = MARKET_WIDE_REGIMES.find((regime) => regime.id === regimeId) ?? MARKET_WIDE_REGIMES[0];
  const yearsValue = Number(years);
  const yearsValid = Number.isInteger(yearsValue) && yearsValue >= 1 && yearsValue <= 100;

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!yearsValid) return;
    mutation.mutate({ regime_id: regimeId, years: yearsValue });
  }

  const result = mutation.data;

  return (
    <div className="flex flex-col gap-4">
      <Card className="border-border/60 p-6">
        <form onSubmit={onSubmit} className="flex flex-col gap-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="flex flex-col gap-2 md:col-span-2">
              <Label htmlFor="regime-select">Regime</Label>
              <Select value={regimeId} onValueChange={(value) => value && setRegimeId(value)}>
                <SelectTrigger id="regime-select" className="w-full">
                  <SelectValue>
                    {(value: string) => MARKET_WIDE_REGIMES.find((regime) => regime.id === value)?.title ?? value}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {MARKET_WIDE_REGIMES.map((regime) => (
                    <SelectItem key={regime.id} value={regime.id}>
                      {regime.title}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs leading-relaxed text-muted-foreground">{selected.hint}</p>
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="regime-years">Years of history</Label>
              <Input
                id="regime-years"
                type="number"
                min={1}
                max={100}
                value={years}
                onChange={(event) => setYears(event.target.value)}
                className="font-mono"
              />
            </div>
          </div>
          <div>
            <Button type="submit" disabled={mutation.isPending || !yearsValid} className="gap-2">
              {mutation.isPending ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
              {mutation.isPending ? "Analyzing..." : "Analyze"}
            </Button>
          </div>
        </form>
      </Card>

      {mutation.error ? (
        <Card className="border-destructive/50 bg-destructive/5 p-4">
          <pre className="overflow-x-auto whitespace-pre-wrap text-sm text-destructive">
            {String(mutation.error)}
          </pre>
        </Card>
      ) : null}

      {result ? (
        <Card className="border-border/60 p-4">
          <div className="mb-3">
            <h2 className="text-sm font-semibold">{result.regime.title}</h2>
            <p className="text-xs text-muted-foreground">
              Trading days each state was active, per year.
              {result.coverage
                ? ` Labeled from ${result.coverage.start} to ${result.coverage.end}.`
                : " No sessions could be labeled."}
            </p>
          </div>
          <RegimeYearTable states={result.states} rows={result.rows} />
        </Card>
      ) : null}
    </div>
  );
}
