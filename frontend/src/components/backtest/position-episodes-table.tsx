import { ChevronRight } from "lucide-react";
import { Fragment, useMemo, useState } from "react";
import type { PositionEpisode, PositionEvent, PositionEventKind, PositionsPayload } from "@/api";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  formatExposure,
  formatMoney,
  formatNumber,
  formatSignedPercent,
} from "@/lib/position-format";
import { cn } from "@/lib/utils";

const head = "h-7 px-1.5 py-0 text-[11px] font-medium";
const cell = "px-1.5 py-0.5 align-top font-mono text-[11px] tabular-nums";
const PAGE = 50;

const EVENT_STYLE: Record<PositionEventKind, { color: string; label: string }> = {
  Enter: { color: "var(--chart-1)", label: "Enter" },
  Add: { color: "var(--chart-3)", label: "Add" },
  Reduce: { color: "var(--chart-4)", label: "Reduce" },
  Exit: { color: "var(--chart-2)", label: "Exit" },
};

function pnlTone(value: number | null | undefined) {
  if (value == null || value === 0) return undefined;
  return value > 0 ? "text-[var(--gain)]" : "text-[var(--loss)]";
}

/** Realized P&L as a fraction of portfolio equity just before the trade. Only sells realize. */
function tradePnl(event: PositionEvent): number | null {
  if (event.event !== "Reduce" && event.event !== "Exit") return null;
  return event.equity_before > 0 ? event.realized_pnl / event.equity_before : null;
}

function EventBadge({ kind }: { kind: PositionEventKind }) {
  const { color, label } = EVENT_STYLE[kind];
  return (
    <span
      className="inline-flex h-5 items-center rounded-md px-1.5 text-[10px] font-semibold uppercase tracking-wide"
      style={{
        color,
        background: `color-mix(in oklch, ${color} 14%, transparent)`,
        border: `1px solid color-mix(in oklch, ${color} 40%, transparent)`,
      }}
    >
      {label}
    </span>
  );
}

/** "+Strategy1 -Strategy2": who entered and who exited on the bar. */
function Trigger({ changes }: { changes: PositionEvent["signal_changes"] }) {
  if (changes.length === 0) return <span className="text-muted-foreground">-</span>;
  return (
    <span className="flex flex-wrap gap-x-2 gap-y-0.5">
      {changes.map((change) => (
        <span
          key={`${change.strategy_id}-${change.change}`}
          className={change.change === "entered" ? "text-[var(--gain)]" : "text-[var(--loss)]"}
        >
          {change.change === "entered" ? "+" : "\u2212"}
          {change.strategy_name}
        </span>
      ))}
    </span>
  );
}

function StatusBadge({ status }: { status: PositionEpisode["status"] }) {
  return status === "Open" ? (
    <span className="inline-flex h-5 items-center rounded-md border border-border bg-muted px-1.5 text-[10px] font-semibold uppercase tracking-wide text-foreground">
      Open
    </span>
  ) : (
    <span className="text-[10px] uppercase tracking-wide text-muted-foreground">Closed</span>
  );
}

function EventsTable({ events, detail }: { events: PositionEvent[]; detail: boolean }) {
  return (
    <div className="overflow-x-auto rounded-md border border-border/60 bg-background/60">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className={head}>Date</TableHead>
            <TableHead className={head}>Event</TableHead>
            <TableHead className={head}>Trigger</TableHead>
            <TableHead className={cn(head, "text-right")}>Active signals</TableHead>
            <TableHead className={cn(head, "text-right")}>Exposure</TableHead>
            <TableHead
              className={cn(head, "text-right")}
              title="The underlying's closing price on the trade date"
            >
              Price
            </TableHead>
            <TableHead
              className={cn(head, "text-right")}
              title="Realized P&L of this trade, before fees, as a share of total portfolio equity just before it"
            >
              Trade P&amp;L
            </TableHead>
            <TableHead
              className={cn(head, "text-right")}
              title="Episode P&L so far (realized minus fees, plus the open position at this event's price) as a share of portfolio equity before the episode's first entry. The last row of a closed episode equals Episode %."
            >
              Rolling P&amp;L
            </TableHead>
            <TableHead className={cn(head, "text-right")}>Fees</TableHead>
            {detail ? (
              <>
                <TableHead className={cn(head, "text-right")}>Cash before → after</TableHead>
                <TableHead className={cn(head, "text-right")}>Equity before → after</TableHead>
                <TableHead className={cn(head, "text-right")}>Position value</TableHead>
                <TableHead className={cn(head, "text-right")}>Unrealized P&amp;L</TableHead>
                <TableHead className={head}>Active strategies</TableHead>
              </>
            ) : null}
          </TableRow>
        </TableHeader>
        <TableBody>
          {events.map((event, index) => (
            <TableRow key={`${event.date}-${event.symbol}-${index}`}>
              <TableCell className={cell}>{event.date}</TableCell>
              <TableCell className={cell}>
                <EventBadge kind={event.event} />
              </TableCell>
              <TableCell className="px-1.5 py-0.5 align-top font-mono text-[11px] whitespace-normal">
                <Trigger changes={event.signal_changes} />
              </TableCell>
              <TableCell className={cn(cell, "text-right")}>
                {event.active_signals}/{event.eligible_signals}
              </TableCell>
              <TableCell className={cn(cell, "text-right")}>
                {formatExposure(event.exposure_after)}
              </TableCell>
              <TableCell className={cn(cell, "text-right")}>
                {event.underlying_close == null ? "-" : formatNumber(event.underlying_close)}
              </TableCell>
              <TableCell className={cn(cell, "text-right", pnlTone(tradePnl(event)))}>
                {tradePnl(event) == null ? "-" : formatSignedPercent(tradePnl(event))}
              </TableCell>
              <TableCell
                className={cn(cell, "text-right", pnlTone(event.episode_return_to_date))}
              >
                {event.event === "Enter" ? "-" : formatSignedPercent(event.episode_return_to_date)}
              </TableCell>
              <TableCell className={cn(cell, "text-right")}>{formatMoney(event.fees)}</TableCell>
              {detail ? (
                <>
                  <TableCell className={cn(cell, "text-right")}>
                    {formatMoney(event.cash_before)} → {formatMoney(event.cash_after)}
                  </TableCell>
                  <TableCell className={cn(cell, "text-right")}>
                    {formatMoney(event.equity_before)} → {formatMoney(event.equity_after)}
                  </TableCell>
                  <TableCell className={cn(cell, "text-right")}>
                    {formatMoney(event.position_value_after)}
                  </TableCell>
                  <TableCell className={cn(cell, "text-right", pnlTone(event.unrealized_pnl_after))}>
                    {event.position_value_after > 0
                      ? formatMoney(event.unrealized_pnl_after, true)
                      : "-"}
                  </TableCell>
                  <TableCell className="px-1.5 py-0.5 align-top text-[11px] text-muted-foreground">
                    {event.active_strategies.length ? event.active_strategies.join(", ") : "none"}
                  </TableCell>
                </>
              ) : null}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export function PositionEpisodesTable({
  positions,
  className,
}: {
  positions: PositionsPayload;
  className?: string;
}) {
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const [detail, setDetail] = useState(false);
  const [shown, setShown] = useState(PAGE);

  // Newest first. Events inside an episode stay oldest first.
  const episodes = useMemo(() => [...positions.episodes].reverse(), [positions.episodes]);
  const multiAsset = useMemo(
    () => new Set(positions.episodes.map((episode) => episode.symbol)).size > 1,
    [positions.episodes],
  );
  const visible = episodes.slice(0, shown);

  const toggle = (id: string) =>
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const { reconciliation, price_basis: priceBasis } = positions;
  const columns = 11 + (multiAsset ? 1 : 0);

  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h3 className="text-xs font-medium text-muted-foreground">Position Episodes</h3>
          <p className="text-[11px] text-muted-foreground">
            {episodes.length.toLocaleString()} episodes, {reconciliation.transaction_count.toLocaleString()}{" "}
            transactions. Click a row for its rebalance events.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <Label htmlFor="episode-detail" className="text-[11px] text-muted-foreground">
              Accounting detail
            </Label>
            <Switch id="episode-detail" checked={detail} onCheckedChange={setDetail} />
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-[11px]"
            onClick={() =>
              setExpanded((current) =>
                current.size ? new Set() : new Set(visible.map((episode) => episode.id)),
              )
            }
          >
            {expanded.size ? "Collapse all" : "Expand shown"}
          </Button>
        </div>
      </div>

      <div className="overflow-x-auto rounded-md border border-border/60">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className={cn(head, "w-6")} />
              {multiAsset ? <TableHead className={head}>Asset</TableHead> : null}
              <TableHead className={head}>Start</TableHead>
              <TableHead className={head}>End</TableHead>
              <TableHead className={cn(head, "text-right")} title="Sessions the position was carried into">
                Days
              </TableHead>
              <TableHead
                className={cn(head, "text-right")}
                title="The underlying's closing price on the day of the first entry"
              >
                Initial entry price
              </TableHead>
              <TableHead
                className={cn(head, "text-right")}
                title="The underlying's closing price on the day of the final exit"
              >
                Final exit price
              </TableHead>
              <TableHead className={cn(head, "text-right")}>Max exposure</TableHead>
              <TableHead className={cn(head, "text-right")} title="Session-weighted actual exposure">
                Avg exposure
              </TableHead>
              <TableHead
                className={cn(head, "text-right")}
                title="Episode P&L (realized + unrealized, net of fees) / portfolio equity just before the first entry"
              >
                Episode %
              </TableHead>
              <TableHead className={cn(head, "text-right")}>Rebalances</TableHead>
              <TableHead className={head}>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {visible.map((episode) => {
              const isOpen = expanded.has(episode.id);
              return (
                <Fragment key={episode.id}>
                  <TableRow
                    className={cn(
                      "cursor-pointer",
                      episode.status === "Open"
                        ? "bg-muted/40"
                        : episode.pnl < 0
                          ? "bg-[var(--loss)]/8"
                          : undefined,
                    )}
                    onClick={() => toggle(episode.id)}
                    aria-expanded={isOpen}
                  >
                    <TableCell className="px-1 py-0.5 align-middle">
                      <ChevronRight
                        className={cn(
                          "size-3.5 text-muted-foreground transition-transform",
                          isOpen && "rotate-90",
                        )}
                      />
                    </TableCell>
                    {multiAsset ? <TableCell className={cell}>{episode.symbol}</TableCell> : null}
                    <TableCell className={cell}>{episode.start}</TableCell>
                    <TableCell className={cell}>{episode.end ?? "Open"}</TableCell>
                    <TableCell className={cn(cell, "text-right")}>{episode.days}</TableCell>
                    <TableCell className={cn(cell, "text-right")}>
                      {episode.initial_underlying_close == null
                        ? "-"
                        : formatNumber(episode.initial_underlying_close)}
                    </TableCell>
                    <TableCell className={cn(cell, "text-right")}>
                      {episode.final_underlying_close == null
                        ? ""
                        : formatNumber(episode.final_underlying_close)}
                    </TableCell>
                    <TableCell className={cn(cell, "text-right")}>
                      {formatExposure(episode.max_exposure)}
                    </TableCell>
                    <TableCell className={cn(cell, "text-right")}>
                      {formatExposure(episode.avg_exposure)}
                    </TableCell>
                    <TableCell className={cn(cell, "text-right", pnlTone(episode.return))}>
                      {formatSignedPercent(episode.return)}
                    </TableCell>
                    <TableCell className={cn(cell, "text-right")}>{episode.rebalance_count}</TableCell>
                    <TableCell className="px-1.5 py-0.5 align-middle">
                      <StatusBadge status={episode.status} />
                    </TableCell>
                  </TableRow>
                  {isOpen ? (
                    <TableRow className="hover:bg-transparent">
                      <TableCell colSpan={columns} className="bg-muted/20 p-2">
                        <div className="flex flex-col gap-1.5">
                          <p className="text-[11px] text-muted-foreground">
                            P&amp;L {formatMoney(episode.pnl, true)} on entry equity{" "}
                            {formatMoney(episode.equity_before_entry)} · realized{" "}
                            {formatMoney(episode.realized_pnl, true)} · unrealized{" "}
                            {formatMoney(episode.unrealized_pnl, true)} · fees{" "}
                            {formatMoney(episode.fees)}
                          </p>
                          <EventsTable events={episode.events} detail={detail} />
                        </div>
                      </TableCell>
                    </TableRow>
                  ) : null}
                </Fragment>
              );
            })}
          </TableBody>
        </Table>
      </div>

      {shown < episodes.length ? (
        <Button
          variant="ghost"
          size="sm"
          className="h-7 self-start px-2 text-[11px]"
          onClick={() => setShown((current) => current + PAGE)}
        >
          Show {Math.min(PAGE, episodes.length - shown)} older episodes ({episodes.length - shown}{" "}
          remaining)
        </Button>
      ) : null}

      <p className="text-[10px] leading-snug text-muted-foreground">
        {priceBasis.description} Episode P&amp;L totals {formatMoney(reconciliation.episode_pnl_total, true)}{" "}
        against a portfolio equity change of {formatMoney(reconciliation.equity_change, true)}
        {Math.abs(reconciliation.episode_vs_equity_difference) <
        1e-6 * Math.max(1, Math.abs(reconciliation.equity_change))
          ? " (reconciled)."
          : ` (differs by ${formatMoney(reconciliation.episode_vs_equity_difference, true)}).`}
        {positions.signal_changes_without_transaction > 0
          ? ` ${positions.signal_changes_without_transaction.toLocaleString()} signal changes did not change the position and are not listed as transactions.`
          : ""}
      </p>
    </div>
  );
}
