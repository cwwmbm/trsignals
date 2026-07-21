"use client";

import { memo, useEffect, useState } from "react";
import type {
  DetailedResult,
  SweepMeta,
  SweepResult,
  SweepResultResponse,
} from "@/api";
import {
  BuilderRefinePanel,
  type BuilderRefinePanelProps,
} from "@/components/backtest/builder-refine-panel";
import { DetailResults } from "@/components/backtest/detail-results";
import { SweepResults } from "@/components/backtest/sweep-results";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

function isDetailedResult(
  result: DetailedResult | SweepResult | SweepResultResponse | undefined,
): result is DetailedResult {
  return Boolean(result && !Array.isArray(result) && "summary" in result);
}

function isSweepResultResponse(
  result: DetailedResult | SweepResult | SweepResultResponse | undefined,
): result is SweepResultResponse {
  return Boolean(result && !Array.isArray(result) && "rows" in result && "meta" in result);
}

function sweepRows(
  result: DetailedResult | SweepResult | SweepResultResponse | undefined,
): SweepResult | undefined {
  if (Array.isArray(result)) return result;
  if (isSweepResultResponse(result)) return result.rows;
  return undefined;
}

function sweepMeta(
  result: DetailedResult | SweepResult | SweepResultResponse | undefined,
): SweepMeta | undefined {
  if (isSweepResultResponse(result)) return result.meta;
  return undefined;
}

const ResultsBody = memo(function ResultsBody({
  result,
  selectedSweepRow,
  onSelectSweepRow,
  emptyMessage,
  onAddRow,
  canAddRow,
  addRowLabel,
}: {
  result: DetailedResult | SweepResult | SweepResultResponse | undefined;
  selectedSweepRow: Record<string, unknown> | undefined;
  onSelectSweepRow: (row: Record<string, unknown>) => void;
  emptyMessage: string;
  onAddRow?: (row: Record<string, unknown>) => void;
  canAddRow?: (row: Record<string, unknown>) => boolean;
  addRowLabel?: (row: Record<string, unknown>) => string;
}) {
  if (result === undefined) {
    return <p className="py-6 text-center text-xs text-muted-foreground">{emptyMessage}</p>;
  }
  if (isDetailedResult(result)) {
    return <DetailResults result={result} embedded />;
  }
  const rows = sweepRows(result);
  if (!rows) {
    return <p className="py-6 text-center text-xs text-muted-foreground">{emptyMessage}</p>;
  }
  return (
    <SweepResults
      rows={rows}
      selectedRow={selectedSweepRow}
      onSelectRow={onSelectSweepRow}
      embedded
      onAddRow={onAddRow}
      canAddRow={canAddRow}
      addRowLabel={addRowLabel}
      meta={sweepMeta(result)}
    />
  );
});

type StrategyBuilderResultsPaneProps = {
  builderResult: DetailedResult | SweepResult | undefined;
  builderResultsVersion: number;
  builderSelectedSweepRow: Record<string, unknown> | undefined;
  onBuilderSelectSweepRow: (row: Record<string, unknown>) => void;
  refineResult: DetailedResult | SweepResult | SweepResultResponse | undefined;
  refineResultsVersion: number;
  refineSelectedSweepRow: Record<string, unknown> | undefined;
  onRefineSelectSweepRow: (row: Record<string, unknown>) => void;
  onAddConditionFromSweepRow?: (row: Record<string, unknown>) => void;
  canAddSweepRow?: (row: Record<string, unknown>) => boolean;
  sweepRowAddLabel?: (row: Record<string, unknown>) => string;
  refineProps: BuilderRefinePanelProps;
  resetVersion?: number;
};

function StrategyBuilderResultsPane({
  builderResult,
  builderResultsVersion,
  builderSelectedSweepRow,
  onBuilderSelectSweepRow,
  refineResult,
  refineResultsVersion,
  refineSelectedSweepRow,
  onRefineSelectSweepRow,
  onAddConditionFromSweepRow,
  canAddSweepRow,
  sweepRowAddLabel,
  refineProps,
  resetVersion = 0,
}: StrategyBuilderResultsPaneProps) {
  const [activeTab, setActiveTab] = useState<"results" | "refine" | "refinement-results">("results");

  useEffect(() => {
    setActiveTab("results");
  }, [resetVersion]);

  useEffect(() => {
    if (builderResultsVersion > 0) {
      setActiveTab("results");
    }
  }, [builderResultsVersion]);

  useEffect(() => {
    if (refineResultsVersion > 0) {
      setActiveTab("refinement-results");
    }
  }, [refineResultsVersion]);

  return (
    <Card className="border-border/60 p-3">
      <Tabs
        value={activeTab}
        onValueChange={(value) =>
          value && setActiveTab(value as "results" | "refine" | "refinement-results")
        }
      >
        <TabsList
          variant="line"
          className="h-8 w-full justify-start rounded-none border-b border-border/60 bg-transparent p-0"
        >
          <TabsTrigger value="results" className="h-8 rounded-none px-3 text-xs">
            Results
          </TabsTrigger>
          <TabsTrigger value="refine" className="h-8 rounded-none px-3 text-xs">
            Refine
          </TabsTrigger>
          <TabsTrigger value="refinement-results" className="h-8 rounded-none px-3 text-xs">
            Refinement Results
          </TabsTrigger>
        </TabsList>

        <TabsContent value="results" className="mt-3">
          {activeTab === "results" ? (
            <ResultsBody
              result={builderResult}
              selectedSweepRow={builderSelectedSweepRow}
              onSelectSweepRow={onBuilderSelectSweepRow}
              emptyMessage="Run Backtest above to see strategy results."
            />
          ) : null}
        </TabsContent>

        <TabsContent value="refine" className="mt-3">
          {activeTab === "refine" ? <BuilderRefinePanel {...refineProps} /> : null}
        </TabsContent>

        <TabsContent value="refinement-results" className="mt-3">
          {activeTab === "refinement-results" ? (
            <ResultsBody
              result={refineResult}
              selectedSweepRow={refineSelectedSweepRow}
              onSelectSweepRow={onRefineSelectSweepRow}
              emptyMessage="Run a refinement from the Refine tab to see results here."
              onAddRow={onAddConditionFromSweepRow}
              canAddRow={canAddSweepRow}
              addRowLabel={sweepRowAddLabel}
            />
          ) : null}
        </TabsContent>
      </Tabs>
    </Card>
  );
}

export const MemoizedStrategyBuilderResultsPane = memo(StrategyBuilderResultsPane);

export { StrategyBuilderResultsPane };
