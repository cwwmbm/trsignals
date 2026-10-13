# TradingStrategy

A Python-based trading **signal and backtesting system**. This repo does **not** place trades — it fetches end-of-day market data, computes technical indicators, evaluates buy/hold/sell signals, and backtests strategies against historical data.

## What it does

| Capability | Description |
|---|---|
| **Daily signal scan** | Pulls EOD prices from Yahoo Finance and reports which strategies are firing buy, hold, or sell across a watchlist of symbols |
| **Backtesting** | Simulates strategies over years of history with PnL, drawdown, Sharpe/Sortino, Kelly, and CAGR |
| **Latest quotes** | Prints current price and key indicators for the configured ticker |
| **Signal exploration** | Combine signals (AND/OR), sweep indicator filters, and require cross-symbol confirmation before entry |

## Quick start

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install streamlit               # for signal_check.py only

# Edit the config block in unused/run_backtest.py, then:
python unused/run_backtest.py
```

### Web interface

Run the local API and React frontend in two terminals:

```bash
# Terminal 1: backend
source venv/bin/activate
python -m uvicorn api.main:app --reload
```

```bash
# Terminal 2: frontend
cd frontend
npm install
npm run dev
```

Open the Vite URL (usually `http://localhost:5173`). The frontend talks to the backend at `http://localhost:8000` via `frontend/.env.development`. Override with `VITE_API_URL` if needed.

### Mobile Scan and Quote (remote access)

Access Scan and Quote from your phone while the Mac stays running at home.

**1. Start the unified server** (builds frontend + serves API + UI on one port):

```bash
source venv/bin/activate
./scripts/serve-mobile.sh
```

Or manually:

```bash
cd frontend && npm run build && cd ..
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/scan` or `http://localhost:8000/quote` on desktop, or use remote access below. The mobile header switches between Scan and Quote.

**2. Remote access with port forwarding**

No third-party accounts. Your router forwards traffic from the internet to the Mac.

1. Start the unified server (above). It binds to `0.0.0.0:8000`.
2. Find your Mac’s LAN IP: **System Settings → Network** (e.g. `192.168.1.42`), or:
   ```bash
   ipconfig getifaddr en0   # Wi‑Fi; use en1 etc. if needed
   ```
3. In your router admin UI, add a **port forwarding** rule:
   - External port: `8000` (or another port if your ISP blocks 8000)
   - Internal IP: your Mac’s LAN IP
   - Internal port: `8000`
   - Protocol: TCP
4. Optional but recommended: give the Mac a **DHCP reservation** so its LAN IP doesn’t change.
5. Find your **public IP** (whatismyip.com, or `curl -s ifconfig.me`). If it changes often, set up **dynamic DNS** on your router (No-IP, DuckDNS, etc.).
6. On your phone (any network): `http://<public-ip>:8000/scan` or `/quote`  
   If you used a non‑8000 external port: `http://<public-ip>:<external-port>/scan` (or `/quote`)
7. **macOS firewall:** if enabled, allow incoming connections for Python when prompted, or add a rule in **System Settings → Network → Firewall**.
8. **Add to Home Screen** (Safari Share → Add to Home Screen) for an app-like experience.

**Security:** The app has no login. Anyone who can reach that URL can see your scan data. Use a non-obvious external port, restrict by IP on the router if supported, or add auth before exposing broadly.

**Same Wi‑Fi only (no port forwarding):** skip steps 3–5 and use `http://<mac-lan-ip>:8000/scan` or `/quote` on your phone while at home.

**Keep the Mac awake** while away: System Settings → Battery/Energy → disable sleep on power adapter, or run `caffeinate -dims` in a terminal.

**Alternative: Tailscale (private, no port forwarding)**

Install [Tailscale](https://tailscale.com/download) on Mac and phone, sign in, then open `http://<mac-tailscale-ip>:8000/scan` or `/quote`. No public exposure; requires a free account.

**Alternative: Cloudflare quick tunnel (HTTPS, no router config)**

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8000
```

Bookmark the generated `https://….trycloudflare.com/scan` or `/quote` URL. The link is public while the tunnel runs — treat it like an open port.

## Project layout

```
TradingStrategy/
├── api/                 ← FastAPI backend for the web UI
├── frontend/            ← React/Vite frontend
├── tests/               ← unittest suite
├── unused/              ← standalone / legacy scripts (not used by web stack)
├── config.py             ← strategy parameters (ticker, RSI, leverage, etc.)
├── getdata.py            ← Yahoo Finance fetch, breadth, holiday filtering
├── indicators.py         ← indicators + 24+ buy signal definitions
├── backtest.py           ← strategy engine (og_strat, long_strat) + sweeps
├── stats.py              ← aggregate metrics, yearly breakdown, outlier exclusion
├── backtest_runners.py   ← load data, single-symbol & cross-symbol runners
├── indicator_sweep.py    ← grid-search indicator filters on a signal
├── signal_check.py       ← Streamlit daily signal dashboard
├── quote.py              ← quick indicator snapshot
├── warn_config.py        ← suppresses third-party FutureWarnings
├── requirements.txt
└── CSV/                  ← backtest output (gitignored)
```

### Data flow

```
Yahoo Finance (yfinance)
        ↓
getdata.py  — OHLCV + VIX + sector breadth ratios (RSP/SPY, QQQ/SPY, …)
        ↓
indicators.add_indicators()  — RSI, EMA, IBR, ValueCharts, VFI, …
        ↓
buy_signalN() / combined_signal()  — Buy/Sell booleans + hold rules
        ↓
backtest.execute_strategy()  — entries, exits, RollingPnL
        ↓
stats.compute_aggregate_metrics()  — Sharpe, CAGR, etc. (optional outlier exclusion)
        ↓
Output  — summary table, print_stats, or CSV
```

## Configuration

Strategy parameters live in [`config.py`](config.py):

| Parameter | Default | Purpose |
|---|---|---|
| `ticker` | `SOXX` | Default symbol for quote / backtest scripts |
| `RSI2Buy` / `RSI5Buy` | 15 / 35 | OG strategy oversold thresholds |
| `RSI2Sell` / `RSI5Sell` | 95 / 70 | OG strategy overbought thresholds |
| `stop_loss` | 0.15 | Exit when trade PnL drops below −15% |
| `Leverage` | 3 | Multiplier applied to returns |
| `VolumeEMAThreashold` | 0.6 | Volume vs 8-day EMA filter |
| `VolatilityThreashold` | 0.1 | Annualized volatility exit filter |
| `MondayBuy` / `LowVolumeBuy` | True | One-day buy rules in OG strategy |
| `UseProxyUnderlying` | False | Track PnL via leveraged proxy (e.g. SOXX) |
| `ExcludeBestReturnYear` | True | Drop best positive year from aggregate metrics |
| `api_key` | `""` | Pushbullet key (commented out in signal_check.py) |

Backtest run settings live in the **config block at the top of [`unused/run_backtest.py`](unused/run_backtest.py)**.

## Backtesting (`unused/run_backtest.py`)

Edit the config block, set `RUN_MODE`, and run:

```bash
python unused/run_backtest.py
```

`python unused/IBConnect.py` also works (legacy alias).

### Run modes

| `RUN_MODE` | What it does |
|---|---|
| `single` | One signal, one symbol — full stats + yearly breakdown + CSV |
| `indicator_sweep` | Grid-search indicator filters layered on a signal (ranked by Sharpe) |
| `signal_combo_sweep` | Compare 4 AND/OR combos of two signals on the same symbol |
| `symbol_confirm_sweep` | Sweep all cross-symbol confirmation subsets from a pool |
| `symbol_confirm_detail` | One primary + chosen confirm symbols — full yearly breakdown |
| `hold_days_sweep` | Search days-in-trade × profitable closes |

### Example configs

**Single signal, one symbol:**

```python
RUN_MODE = 'single'
SYMBOL = 'SOXX'
SIGNAL = ind.buy_signal7
YEARS = 25
```

**Combined signal (AND/OR on same symbol):**

```python
RUN_MODE = 'single'
SIGNAL = ind.combined_signal(ind.buy_signal16, ind.buy_signal7, 'or')
```

**Compare all 4 AND/OR combinations:**

```python
RUN_MODE = 'signal_combo_sweep'
SIGNAL_A = ind.buy_signal16
SIGNAL_B = ind.buy_signal7
SYMBOL = 'SOXX'
```

**Cross-symbol confirmation sweep** (primary traded at leverage; all confirm symbols must also show buy):

```python
RUN_MODE = 'symbol_confirm_sweep'
SIGNAL = ind.combined_signal(ind.buy_signal16, ind.buy_signal7, 'or')
PRIMARY_SYMBOL = 'SOXX'
SYMBOL_POOL = ['SOXX', 'SMH', 'QQQ', 'SPY']
```

Auto-generates rows for `(none)`, `SMH`, `QQQ`, `SPY`, `SMH+QQQ`, … sorted by Sharpe.

**Drill into one confirm set** (after picking from sweep):

```python
RUN_MODE = 'symbol_confirm_detail'
PRIMARY_SYMBOL = 'SOXX'
CONFIRM_SYMBOLS = ['SMH', 'QQQ']
```

**Indicator filter sweep:**

```python
RUN_MODE = 'indicator_sweep'
SIGNAL = ind.buy_signal7
INDICATOR_SWEEP = dict(is_sell=False, check_breadth=False, check_both=False)
```

### Aggregate metrics & outlier exclusion

When `ExcludeBestReturnYear = True` in `config.py`, aggregate stats (Sharpe, Sortino, CAGR, MaxDD, trade count, win rate, Kelly) **exclude the single calendar year with the highest positive return**. This keeps one extreme upside year from dominating comparisons.

- **Latest Rolling PnL** always reflects the full backtest (real total)
- **Yearly breakdown** always shows every year
- A note is printed when a year is excluded, e.g. `(Aggregate metrics exclude 2020 — best year at 906.88%)`

Set `ExcludeBestReturnYear = False` to restore the original behavior.

### Output

**Summary table** (sweeps): Sharpe-sorted rows with PnL, MaxDD, Trades, %Pstv, CAGR.

**Detailed stats** (`single`, `symbol_confirm_detail`):

```
Number of trades: 318
Latest Rolling PnL: $1,539,583.00
Maximum drawdown: 30.81%
CAGR: 32.54%
Sharpe ratio: 0.45
(Aggregate metrics exclude 2020 — best year at 906.88%)
          PnL% Drawdown%  Num_Trades  Positive_Trades
Date
2019  216.07%    16.81%          61               47
2020  906.88%    32.89%          64               50   ← still shown here
...
```

CSV files are saved to `CSV/` (create the folder if needed).

## Web interface

The web app is local-first and stateless: each run is configured in the browser, sent to FastAPI, and returned directly as JSON. No database or auth is used in v1.

### Backend API

Start it with:

```bash
python -m uvicorn api.main:app --reload
```

Available endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | API health check |
| `GET /signals` | Signal names for dropdowns |
| `GET /config` | Current defaults from `config.py` |
| `POST /backtests/single` | One signal or combined signal on one symbol |
| `POST /backtests/signal-combo-sweep` | Four AND/OR combos between two signals |
| `POST /backtests/symbol-confirm-sweep` | Confirmation subset sweep from a symbol pool |
| `POST /backtests/symbol-confirm-detail` | One primary + chosen confirmations with yearly breakdown |
| `POST /backtests/hold-days-sweep` | Hold-days/profitable-close grid |
| `POST /backtests/indicator-sweep` | Indicator threshold sweep |

Signal expressions sent to the API look like:

```json
{ "kind": "single", "name": "buy_signal7" }
```

or:

```json
{ "kind": "combined", "primary": "buy_signal16", "secondary": "buy_signal7", "mode": "or" }
```

Detailed backtests return `summary`, `yearly`, `equity_curve`, and `trades`. Sweep endpoints return Sharpe-sorted rows with numeric values so the frontend can format them.

### Frontend

The React app lives in `frontend/`.

```bash
cd frontend
npm install
npm run dev
```

The UI includes:

- Backtest Builder with mode-specific fields
- Summary cards for PnL, CAGR, Sharpe, Sortino, MaxDD, trades, and excluded year
- Equity curve chart
- Yearly breakdown table
- Sweep results table
- Trade list for detailed runs

## Signal combination (same symbol)

Two signals on the **same** symbol can be merged with AND or OR. Days/profit/sell come from the **primary** signal.

```python
# In indicators.py:
SIGNAL = ind.combined_signal(ind.buy_signal16, ind.buy_signal7, 'and')  # both must fire
SIGNAL = ind.combined_signal(ind.buy_signal16, ind.buy_signal7, 'or')   # either fires
```

Low-level API:

```python
buy, sell, days, profit, desc, _, is_long, _ = ind.combine_buy_signals(
    ind.buy_signal16, ind.buy_signal7, data, mode='and'
)
```

`signal_combo_sweep` runs all four variants (A primary AND B, A primary OR B, B primary AND A, B primary OR A).

## Cross-symbol confirmation

Trade the **primary** symbol at leverage. Primary buy only fires when every confirm symbol is **active** that day: either its entry signal is true, or it is still in a simulated hold from a prior entry and has no sell signal. **Sell** is evaluated on the primary only.

```python
# Sweep — compare confirm combinations
symbol_confirmation_tryout(SIGNAL, 'SOXX', ['SOXX', 'SMH', 'QQQ'], years=25)

# Detail — one chosen combo with yearly breakdown
symbol_confirmation_detail(SIGNAL, 'SOXX', confirm_symbols=['SMH', 'QQQ'], years=25)
```

Typical workflow: **`symbol_confirm_sweep` → pick winner → `symbol_confirm_detail`**.

## Portfolio allocation models

A portfolio combines saved long strategies into one book. How much of that book is invested is the **allocation model**, chosen per request and saved with the portfolio. Strategy signals, entry/exit timing, and price conventions are never changed by it.

| Model | Target exposure |
|---|---|
| **All-in** (default) | 100% while any strategy is active, 0% otherwise. This is the original behavior and runs through the original overlay code, so existing results are unchanged. Overlap mode (`first_signal_only` / `hold_until_all_exit`) applies. |
| **Proportional, linear** | `active / total` |
| **Proportional, early conviction** | `(active / total) ^ 0.5` |
| **Proportional, confirmation required** | `(active / total) ^ 2` |

All proportional modes give 0% with nothing active and 100% with everything active. Proportional ignores overlap mode: every strategy keeps its own entries and exits, and each counts toward exposure.

**Simulation rules (proportional)**

- *Active* means a strategy holds into the next bar: held and not exiting, or entering. *Total* counts every strategy that is **eligible**, whether or not it has ever entered.
- A strategy becomes eligible on the first bar its entry could be evaluated (`EntryReady`: indicator warmup, instrument listing, regime labels) and stays in the denominator after that. Gaps in readiness later on are counted and reported, not dropped. A strategy whose data ends is removed from the denominator and can no longer be active. Strategies without an `EntryReady` column are eligible from their first bar. Metrics start on the first bar any strategy is eligible.
- Exposure is a **target share of current portfolio equity**, not a separate investment per strategy. It is recomputed whenever the effective state changes (who is active, or who is eligible), and the existing position is rebalanced to it: partial entries, partial exits, and full liquidation all come from the same rebalance. Between changes the position drifts with price.
- With several instruments (different strategy symbols, no portfolio proxy), the exposure is split evenly across the active strategies and each share is held in that strategy's instrument.
- Timing matches the single-strategy backtester: a signal on a bar's close is traded at that close and earns the next bar's move. Each instrument trades as its leveraged return series (`config.Leverage * pct_change`), so a fully invested book compounds exactly like the all-in overlay.
- Cash earns nothing. Trading cost is `transaction_cost_bps` of traded notional, on buys and sells, default `0`. It is charged to every model, including all-in, so comparisons are fair. With cost `0` the all-in path is untouched.
- The book tracks cash, invested capital, quantity, average-cost basis, realized and unrealized P&L (realized is gross of costs; costs are tracked separately), and total transaction costs.

**Analytics** (Allocation tab on a portfolio result, and `allocation` in the API payload)

- Average exposure, time invested, exposure when invested, rebalance count (entries, exits, partial adjustments), turnover (total and annualized), and trading costs.
- Performance attributable to allocation: against an all-in book on the same signals, the gap splits into an allocation effect, a cost effect, and a remainder (basket drift), with a breakdown by number of active strategies.
- Per-strategy warmup and eligibility.
- **Compare allocation models** (on demand, `POST /portfolios/allocation-comparison`): All-in versus each proportional mode over the same window, rebased to the same starting capital, with metrics and an equity-curve overlay.

**Code layout** (each piece is replaceable on its own)

```
api/allocation.py            pure models: SignalState -> target exposure. No prices, cash, or dates.
api/rebalancer.py            PortfolioLedger: trades to target weights, keeps the books.
api/allocation_engine.py     effective signal state, then model -> ledger, bar by bar.
api/allocation_analytics.py  exposure, turnover, attribution, comparison.
api/portfolio_service.py     wiring: eligibility from strategies, dispatch, payloads.
```

To add a model (volatility targeting, regime-adjusted sizing, correlation-aware weighting), implement `target_exposure(state)` as a class in `api/allocation.py` and register it in `resolve_allocation_model`. Signals the model needs beyond the active count can travel in `SignalState.context`. The ledger, analytics, and API do not change.

Request fields on `/portfolios/simulate` and saved portfolios: `allocation_model` (`all_in` | `proportional`), `proportional_mode` (`linear` | `early_conviction` | `confirmation_required`), `transaction_cost_bps`. Portfolios saved before these existed load as all-in.

## Position episodes and rebalance events

On a portfolio result the old Trades table is replaced by **Position Episodes**, which is the same under every allocation model. Partial entries, partial exits, and rebalances make "one entry, one exit, one P&L" the wrong unit, so the portfolio's behavior is shown as two things:

- **Position episode**: a continuous period during which the portfolio holds a nonzero position in an instrument, from its first entry to full liquidation. Partial entries and exits never start a new episode. Newest first.
- **Rebalance event**: one executed transaction inside an episode, classified from the position's quantity change alone: **Enter** (zero to positive), **Add**, **Reduce**, **Exit** (to zero). Only trades that actually executed appear. A signal change that leaves the book unchanged is not a buy or a sell; those are only counted (`signal_changes_without_transaction`).

**Where the numbers come from.** Every figure is the ledger's own record of the fill (units, execution price, fee, realized P&L, position and average cost after), never rebuilt from target percentages. Episodes are derived from those transactions, not from signals. Signals only explain *why*.

| Episode column | Definition |
|---|---|
| Days | Sessions the position was carried into, on the simulation calendar. Weekends and holidays are not sessions. A buy on Monday's close sold on Wednesday's close is 2. |
| Initial entry / final exit price | The underlying's closing price on the day of the first entry and of the last exit (blank while open). The same prices the old Trades table showed. |
| Max / Avg exposure | Highest actual exposure, and session-weighted mean actual exposure, during the episode. Actual, not target: it drifts with price and fees. |
| Episode % | Episode P&L / portfolio equity immediately before the first entry. Episode P&L is realized (average cost, gross) minus fees, plus unrealized P&L on units still held at the final mark (zero once closed); the dollar figure is in the expanded row. A contribution to portfolio equity, not the return on deployed capital and not the instrument's price move. |
| Rebalances | Executed transactions in the episode. |

Each event row shows its date, type, **Trigger**, active/eligible strategies, **Exposure**, **Price** (the underlying's close that day), **Trade P&L**, and fees. Trade P&L is the trade's realized P&L (before fees) as a percent of total portfolio equity just before it; entries and adds show `-` because only sells realize. **Rolling P&L** is the episode's P&L so far (realized minus fees, plus the units still held at that event's price) as a percent of the same pre-entry equity as Episode %, so it builds line by line and the last line of a closed episode equals Episode %. The first line is `-`. Trigger is just who moved on that bar: `+Strategy1` entered, `-Strategy2` exited, simultaneous changes all listed. Exposure is the actual portfolio exposure after the trade. **Accounting detail** adds cash and equity before and after, position value, unrealized P&L, and the names of the active strategies.

A trade can move the book without a signal deciding the size, for example when a swap leaves the count unchanged but price has pushed the position off its share. The trigger still lists the strategies that moved; the exposure shows where the book ended up.

**All-In uses the same models.** The all-in overlay still produces the all-in equity curve, so its results are unchanged. Its position flags are replayed through the same engine and ledger, which yields the same transactions and episodes as every other model. There is one accounting implementation. The replay's equity is compared with the overlay's, and the largest gap is reported in the payload (about 4e-15 on the saved portfolios). An All-In episode is normally one Enter and one Exit.

**Prices are real prices.** Every price shown is the underlying's closing price on the trade date, as in the old Trades table. Internally the ledger trades each instrument as its leveraged return series (`config.Leverage * pct_change`, starting at 100), and that simulated price never appears as a price in the UI. The payload keeps it under `simulated_price`, `initial_simulated_price` and `final_simulated_price` for audit, and the real close under `underlying_close`, `initial_underlying_close` and `final_underlying_close`. Because the position is leveraged, the move between an entry and exit price is not the episode's return; the episode and event percentages come from the ledger.

**Episode % under All-In is the old Trades table's number, exactly.** It is taken from the unchanged overlay code, so every All-In trade matches the old build field for field (percent, dates, prices, days, status), and the closing line of the events table agrees with it. That figure is the leveraged price return of the underlying, `Leverage * (exit close / entry close - 1)`. It is not what the equity curve did over the trade, because the curve compounds the leverage daily. For one trade 2026-10-06 to 2026-10-12 (closes 589.45, 582.82, 563.28, 559.40, 546.26) the old formula gives -21.98% and the compounded equity change is -20.89%. Over the 799 closed trades of "OG + Semis combined", chaining the old percentages gives 87.4M times the starting capital against 48.4M for the equity curve itself. The ledger's own figure is kept as `return_on_equity` in the payload. Under the proportional models, where there is no old figure and fractional positions make a price return meaningless, Episode % is that ledger figure (episode P&L / equity just before the first entry).

**Reconciliation** is in the payload (`positions.reconciliation`): episode P&L added up against the change in portfolio equity, against the sum of event P&L, and the money moved by transactions against the change in cash. All three tie out to float noise. Slippage is supported by the ledger (execution price moves against the trade, equity is marked at the mark) but is not exposed as a setting, because the all-in overlay does not model it.

**Code layout**

```
api/rebalancer.py          Fill: the authoritative transaction record.
api/position_episodes.py   TransactionRecord, Enter/Add/Reduce/Exit, episode aggregation.
api/allocation_engine.py   records a TransactionRecord for each fill, with its signal context.
api/portfolio_service.py   all-in replay and the `positions` payload.
frontend/.../position-episodes-table.tsx   the expandable table.
```

## Other scripts

### Daily signal check (Streamlit)

```bash
streamlit run signal_check.py
```

Scans ~18 symbols against 24 buy signals. Shows buy/hold/sell state plus SPY market summary.

### Latest quotes

```bash
python quote.py
```

Prints today's close, RSI, EMA, Stochastic, breadth, and volume for `config.ticker`.

### Compare all signals on one symbol

```bash
python unused/test_data.py
```

Edit `yfticker` in the file to change the symbol.

### One signal across many symbols

```bash
python unused/test_indicator.py
```

Change `buy_signal = ind.buy_signal11` at the top.

## Running tests

```bash
python -m unittest discover -s tests -v
```

## Signals

Each signal in [`indicators.py`](indicators.py) returns an 8-tuple:

```python
buy, sell, days, profit, description, verdict, is_long, ignore = buy_signalN(data, symbol)
```

| Field | Meaning |
|---|---|
| `buy` | Entry condition per bar |
| `sell` | Exit condition (`False` if none) |
| `days` | Max hold days (`0` = OG hold-until-sell) |
| `profit` | Exit after N profitable closes |
| `is_long` | Long vs short |
| `ignore` | Skip in signal scan if symbol not in `allowed_symbols` |

| Signal | Allowed symbols | Style | Summary |
|---|---|---|---|
| `buy_signal7` | SMH, QQQ, FXI, SOXX, SPY | 2d/1p | Close pullback + IBR ≤ 0.4 |
| `buy_signal10` | SMH, SPY, SOXX, QQQ | 3d/1p | New low + IBR |
| `buy_signal16` | SMH, QQQ, SOXX | 4d/1p | High > prior close + IBR |
| `og_buy_signal` | SPY | OG | RSI2/RSI5 oversold + volume filter |
| `og_new_buy_signal` | SPY, IWM, QQQ | OG | OG buy + Stoch + SMA trend |
| `buy_signal1`–`24` | Various | Mixed | See `indicators.py` for full list |

> Many signals have empty `allowed_symbols`, so they are skipped in `signal_check.py` unless you add your ticker to that list.

## Strategy engine

[`backtest.execute_strategy()`](backtest.py) routes to:

| Engine | When | Behavior |
|---|---|---|
| `long_strat` | `days > 0` | Enter on Buy, exit on Sell / max days / N profitable closes |
| `og_strat` | `days == 0` | Hold until sell signal, stop loss, or one-day buy |
| `long_og_strat_proxy` | `UseProxyUnderlying=True` | OG logic, PnL via leveraged proxy |

Key output columns: `LongTradeIn`, `LongTradeOut`, `HoldLong`, `TradePnL`, `RollingPnL`, `Drawdown`.

### Sweep functions

| Function | Purpose |
|---|---|
| `backtest_days(data, max_days)` | Grid search hold days × profitable closes |
| `backtest_ind(…)` | Filter buys by indicator threshold |
| `backtest_sell_ind(…)` | Filter sells by indicator threshold |
| `backtest_signal_combinations(a, b, data)` | 4 AND/OR combos of two signals |
| `backtest_symbol_confirmation_sweep(…)` | All confirm subsets from a symbol pool |

## Indicators

[`indicators.add_indicators()`](indicators.py) adds:

- **Trend:** SMA, EMA, Bollinger Bands, Donchian (20/55), Keltner (TTM standard), Parabolic SAR, ATR
- **Momentum:** RSI (2/5/14), Stochastic, CCI, MACD, ADX(14), Williams %R, ROC(20), TRIX, linear regression slope
- **Volatility:** Bollinger width/%B, realized volatility, volatility percentile, BB/Keltner squeeze flag
- **Volume / flow:** OBV, OBV slope, Chaikin Money Flow, VFI, volume vs EMA
- **Custom:** IBR, Kaufman ER, ValueCharts, Hurst, Change Velocity
- **Breadth:** sector/index ratios vs SPY with RSI overlays
- **Context:** VIX, SPY 50/200 bull flag, Donchian/Keltner/PSAR breakout flags

Session VWAP is not available (daily EOD data only).

## Data sources

**Yahoo Finance** via `yfinance`. [`getdata.py`](getdata.py) provides:

- Single-ticker fetch (`get_data_yf`) and bulk download (`get_bulk_data`)
- Shared market context (VIX, breadth ratios) via `load_symbol_dataset`
- Futures/FX suffix mapping (`NQ=F`, `GBPUSD=X`)
- NYSE holiday filtering

Legacy Interactive Brokers code is commented out and unused.

## Tips

- **Start with `unused/run_backtest.py`** — all backtest modes are configured in one place.
- **Sweep then detail** — use `symbol_confirm_sweep` or `signal_combo_sweep` first, then drill in with `single` or `symbol_confirm_detail`.
- **`SIGNAL` vs `SIGNAL_A`/`SIGNAL_B`** — `single` and confirm modes use `SIGNAL`; combo sweep uses `SIGNAL_A` and `SIGNAL_B`.
- **Leverage** is applied in `long_strat` and `%Change` — adjust `Leverage` in `config.py`.
- **CSV output** goes to `CSV/` (gitignored).

## Disclaimer

This software is for **research and education only**. It generates signals and simulates historical performance — it does not execute live trades. Past backtest performance does not guarantee future results. Use at your own risk.
