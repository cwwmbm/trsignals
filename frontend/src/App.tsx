import { useMutation, useQuery } from "@tanstack/react-query";
import { FormEvent, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  DetailedResult,
  RunMode,
  SignalExpression,
  SweepResult,
  getSignals,
  runBacktest,
} from "./api";

const runModes: Array<{ value: RunMode; label: string; description: string }> = [
  {
    value: "single",
    label: "Single backtest",
    description: "Run one signal, or one combined signal, on one symbol and show detailed stats.",
  },
  {
    value: "signal-combo-sweep",
    label: "Signal combo sweep",
    description: "Compare all four same-symbol combinations: A AND B, A OR B, B AND A, B OR A.",
  },
  {
    value: "symbol-confirm-sweep",
    label: "Symbol confirmation sweep",
    description: "Trade the primary symbol and test every confirmation subset from the symbol pool.",
  },
  {
    value: "symbol-confirm-detail",
    label: "Symbol confirmation detail",
    description: "Drill into one primary symbol plus selected confirmation symbols with yearly detail.",
  },
  {
    value: "hold-days-sweep",
    label: "Hold-days sweep",
    description: "Search hold-days and profitable-close exits for the selected signal.",
  },
  {
    value: "indicator-sweep",
    label: "Indicator sweep",
    description: "Layer indicator threshold filters onto the selected signal and rank results.",
  },
];

function csv(value: string) {
  return value
    .split(",")
    .map((item) => item.trim().toUpperCase())
    .filter(Boolean);
}

function isOgSignalName(name: string) {
  return name === "og_buy_signal" || name === "og_new_buy_signal";
}

function isDetailedResult(result: DetailedResult | SweepResult | undefined): result is DetailedResult {
  return Boolean(result && !Array.isArray(result) && "summary" in result);
}

function formatMetric(value: unknown, column?: string) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const integerColumns = new Set([
    "year",
    "num_trades",
    "positive_trades",
    "days_in_trade",
    "Days",
    "Profit",
    "Trades",
    "Prf",
  ]);
  const percentColumns = new Set([
    "pnl_percent",
    "drawdown_percent",
    "cagr_percent",
    "pct_positive",
  ]);
  const fractionalPercentColumns = new Set(["trade_pnl", "max_drawdown", "drawdown"]);
  if (typeof value === "number") {
    if (column && integerColumns.has(column)) return String(Math.trunc(value));
    if (column && fractionalPercentColumns.has(column)) return `${(value * 100).toFixed(2)}%`;
    if (column && percentColumns.has(column)) return `${value.toFixed(2)}%`;
    if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
    return value.toFixed(2);
  }
  return String(value);
}

function Table({
  rows,
  rowClassName,
  hiddenColumns = [],
  onRowClick,
  selectedRow,
}: {
  rows: Array<Record<string, unknown>>;
  rowClassName?: (row: Record<string, unknown>) => string | undefined;
  hiddenColumns?: string[];
  onRowClick?: (row: Record<string, unknown>) => void;
  selectedRow?: Record<string, unknown>;
}) {
  if (!rows.length) return <p>No rows returned.</p>;
  const columns = Object.keys(rows[0]).filter((column) => !hiddenColumns.includes(column));
  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => {
            const classNames = [
              rowClassName?.(row),
              onRowClick ? "clickableRow" : undefined,
              selectedRow === row ? "selectedRow" : undefined,
            ]
              .filter(Boolean)
              .join(" ");
            return (
            <tr key={idx} className={classNames} onClick={() => onRowClick?.(row)}>
              {columns.map((column) => (
                <td key={column}>{formatMetric(row[column], column)}</td>
              ))}
            </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function SummaryCards({ summary }: { summary: Record<string, unknown> }) {
  const keys = [
    "rolling_pnl",
    "cagr_percent",
    "sharpe",
    "sortino",
    "max_drawdown",
    "trades",
    "pct_positive",
    "excluded_year",
  ];
  return (
    <div className="cards">
      {keys.map((key) => (
        <div className="card" key={key}>
          <span>{key.replace(/_/g, " ")}</span>
          <strong>{formatMetric(summary[key])}</strong>
        </div>
      ))}
    </div>
  );
}

function DetailView({ result }: { result: DetailedResult }) {
  const [logScale, setLogScale] = useState(false);
  const latestTrades = [...result.trades].reverse().slice(0, 100);
  const latestYears = [...result.yearly].reverse();
  const tradeRowClass = (row: Record<string, unknown>) => {
    if (row.status === "Open") return "openRow";
    return typeof row.trade_pnl === "number" && row.trade_pnl < 0 ? "negativeRow" : undefined;
  };
  const negativeYearRow = (row: Record<string, unknown>) =>
    typeof row.pnl_percent === "number" && row.pnl_percent < 0 ? "negativeRow" : undefined;
  return (
    <section className="results">
      <h2>Results</h2>
      <SummaryCards summary={result.summary} />
      <p className="description">{String(result.summary.description ?? "")}</p>

      <h3>Yearly Breakdown</h3>
      <Table rows={latestYears} rowClassName={negativeYearRow} />

      <h3>Trades</h3>
      <Table rows={latestTrades} rowClassName={tradeRowClass} />

      <div className="sectionHeader">
        <h3>Equity Curve</h3>
        <label className="inline chartToggle">
          <input type="checkbox" checked={logScale} onChange={(event) => setLogScale(event.target.checked)} />
          Log scale
        </label>
      </div>
      <div className="chart">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={result.equity_curve}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" minTickGap={48} />
            <YAxis scale={logScale ? "log" : "auto"} domain={["auto", "auto"]} />
            <Tooltip />
            <Line type="monotone" dataKey="rolling_pnl" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function App() {
  const { data: signals = [] } = useQuery({ queryKey: ["signals"], queryFn: getSignals });
  const mutation = useMutation({
    mutationFn: ({ mode, payload }: { mode: RunMode; payload: Record<string, unknown> }) =>
      runBacktest(mode, payload),
  });

  const defaultSignal = signals.find((signal) => signal.name === "buy_signal7")?.name ?? signals[0]?.name ?? "";
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
  const [selectedSweepRow, setSelectedSweepRow] = useState<Record<string, unknown> | undefined>();
  const [mondayBuy, setMondayBuy] = useState(false);
  const [lowVolumeBuy, setLowVolumeBuy] = useState(false);

  const signalOptions = signals.length ? signals : [{ name: defaultSignal, label: defaultSignal }];
  const selectedRunMode = runModes.find((item) => item.value === mode);
  const usesOgRuntimeOptions =
    mode === "signal-combo-sweep"
      ? isOgSignalName(signalA) || isOgSignalName(signalB)
      : signalKind === "single"
        ? isOgSignalName(signal)
        : isOgSignalName(signalA);

  function runtimeOptionsPayload() {
    return usesOgRuntimeOptions
      ? { runtime_options: { monday_buy: mondayBuy, low_volume_buy: lowVolumeBuy } }
      : {};
  }

  function signalExpression(): SignalExpression {
    if (signalKind === "combined") {
      return { kind: "combined", primary: signalA, secondary: signalB, mode: comboMode };
    }
    return { kind: "single", name: signal };
  }

  function buildPayload(): Record<string, unknown> {
    if (mode === "signal-combo-sweep") {
      return { symbol, years, signal_a: signalA, signal_b: signalB, ...runtimeOptionsPayload() };
    }
    if (mode === "symbol-confirm-sweep") {
      return {
        primary_symbol: primarySymbol,
        symbol_pool: csv(symbolPool),
        years,
        signal: signalExpression(),
        ...runtimeOptionsPayload(),
      };
    }
    if (mode === "symbol-confirm-detail") {
      return {
        primary_symbol: primarySymbol,
        confirm_symbols: csv(confirmSymbols),
        years,
        signal: signalExpression(),
        ...runtimeOptionsPayload(),
      };
    }
    if (mode === "hold-days-sweep") {
      return { symbol, years, max_days: maxDays, signal: signalExpression(), ...runtimeOptionsPayload() };
    }
    if (mode === "indicator-sweep") {
      return {
        symbol,
        years,
        signal: signalExpression(),
        is_sell: isSell,
        check_breadth: checkBreadth,
        check_both: checkBoth,
        ...runtimeOptionsPayload(),
      };
    }
    return { symbol, years, signal: signalExpression(), ...runtimeOptionsPayload() };
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSelectedSweepRow(undefined);
    mutation.mutate({ mode, payload: buildPayload() });
  }

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">Local research dashboard</p>
          <h1>TradingStrategy</h1>
        </div>
        <span className="pill">FastAPI + React</span>
      </header>

      <form className="panel" onSubmit={onSubmit}>
        <div className="grid">
          <label>
            Run mode
            <select value={mode} onChange={(event) => setMode(event.target.value as RunMode)}>
              {runModes.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
            {selectedRunMode && <span className="helpText">{selectedRunMode.description}</span>}
          </label>

          <label>
            Years
            <input type="number" min={1} value={years} onChange={(event) => setYears(Number(event.target.value))} />
          </label>

          {mode !== "symbol-confirm-sweep" && mode !== "symbol-confirm-detail" && (
            <label>
              Symbol
              <input value={symbol} onChange={(event) => setSymbol(event.target.value.toUpperCase())} />
            </label>
          )}
        </div>

        {mode !== "signal-combo-sweep" && (
          <section className="subpanel">
            <h3>Signal</h3>
            <label className="inline">
              <input
                type="radio"
                checked={signalKind === "single"}
                onChange={() => setSignalKind("single")}
              />
              Single
            </label>
            <label className="inline">
              <input
                type="radio"
                checked={signalKind === "combined"}
                onChange={() => setSignalKind("combined")}
              />
              Combined
            </label>

            {signalKind === "single" ? (
              <label>
                Signal
                <select value={signal} onChange={(event) => setSignal(event.target.value)}>
                  {signalOptions.map((item) => (
                    <option key={item.name} value={item.name}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <div className="grid three">
                <label>
                  Primary
                  <select value={signalA} onChange={(event) => setSignalA(event.target.value)}>
                    {signalOptions.map((item) => (
                      <option key={item.name} value={item.name}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Mode
                  <select value={comboMode} onChange={(event) => setComboMode(event.target.value as "and" | "or")}>
                    <option value="and">AND</option>
                    <option value="or">OR</option>
                  </select>
                </label>
                <label>
                  Secondary
                  <select value={signalB} onChange={(event) => setSignalB(event.target.value)}>
                    {signalOptions.map((item) => (
                      <option key={item.name} value={item.name}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            )}
          </section>
        )}

        {mode === "signal-combo-sweep" && (
          <section className="subpanel">
            <h3>Signal Combo Sweep</h3>
            <div className="grid">
              <label>
                Signal A
                <select value={signalA} onChange={(event) => setSignalA(event.target.value)}>
                  {signalOptions.map((item) => (
                    <option key={item.name} value={item.name}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Signal B
                <select value={signalB} onChange={(event) => setSignalB(event.target.value)}>
                  {signalOptions.map((item) => (
                    <option key={item.name} value={item.name}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </section>
        )}

        {usesOgRuntimeOptions && (
          <section className="subpanel">
            <h3>OG Runtime Options</h3>
            <p className="helpText">
              Used only when an OG strategy is the primary strategy path. These override `MondayBuy` and
              `LowVolumeBuy` for this run only.
            </p>
            <label className="inline">
              <input type="checkbox" checked={mondayBuy} onChange={(event) => setMondayBuy(event.target.checked)} />
              Monday buy
            </label>
            <label className="inline">
              <input
                type="checkbox"
                checked={lowVolumeBuy}
                onChange={(event) => setLowVolumeBuy(event.target.checked)}
              />
              Low-volume buy
            </label>
          </section>
        )}

        {(mode === "symbol-confirm-sweep" || mode === "symbol-confirm-detail") && (
          <section className="subpanel">
            <h3>Symbol Confirmation</h3>
            <div className="grid">
              <label>
                Primary symbol
                <input value={primarySymbol} onChange={(event) => setPrimarySymbol(event.target.value.toUpperCase())} />
              </label>
              {mode === "symbol-confirm-sweep" ? (
                <label>
                  Symbol pool
                  <input value={symbolPool} onChange={(event) => setSymbolPool(event.target.value)} />
                </label>
              ) : (
                <label>
                  Confirm symbols
                  <input value={confirmSymbols} onChange={(event) => setConfirmSymbols(event.target.value)} />
                </label>
              )}
            </div>
          </section>
        )}

        {mode === "hold-days-sweep" && (
          <label>
            Max hold days
            <input type="number" min={1} value={maxDays} onChange={(event) => setMaxDays(Number(event.target.value))} />
          </label>
        )}

        {mode === "indicator-sweep" && (
          <section className="subpanel">
            <h3>Indicator Sweep</h3>
            <label className="inline">
              <input type="checkbox" checked={isSell} onChange={(event) => setIsSell(event.target.checked)} />
              Sweep sell filters
            </label>
            <label className="inline">
              <input type="checkbox" checked={checkBreadth} onChange={(event) => setCheckBreadth(event.target.checked)} />
              Include breadth
            </label>
            <label className="inline">
              <input type="checkbox" checked={checkBoth} onChange={(event) => setCheckBoth(event.target.checked)} />
              Include price/momentum
            </label>
          </section>
        )}

        <button type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? "Running..." : "Run backtest"}
        </button>
      </form>

      {mutation.error && <pre className="error">{String(mutation.error)}</pre>}

      {isDetailedResult(mutation.data) && <DetailView result={mutation.data} />}
      {Array.isArray(mutation.data) && (
        <section className="results">
          <h2>Sweep Results</h2>
          <Table
            rows={mutation.data}
            hiddenColumns={["Yearly"]}
            selectedRow={selectedSweepRow}
            onRowClick={setSelectedSweepRow}
          />
          {Array.isArray(selectedSweepRow?.Yearly) && (
            <>
              <h3>Selected Outcome Annual Breakdown</h3>
              <Table
                rows={[...(selectedSweepRow.Yearly as Array<Record<string, unknown>>)].reverse()}
                rowClassName={(row) =>
                  typeof row.pnl_percent === "number" && row.pnl_percent < 0 ? "negativeRow" : undefined
                }
              />
            </>
          )}
        </section>
      )}
    </main>
  );
}

export default App;
