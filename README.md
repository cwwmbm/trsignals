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
├── api/fast_sweep.py     ← compiled screen for indicator pairs/trios (numba)
├── api/earnings_data.py  ← earnings dates (Yahoo, cached) behind DaysToEarnings / DaysSinceEarnings
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
| `Leverage` | 3 | Default for the General leverage setting (the live value is on the Settings page), and the fixed factor behind the `%Change` indicator |
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

**Single-indicator sweep speed** (`indicator_sweep.py`, `api/confirm_sweep.py`, `api/fast_metrics.py`):

The single-indicator sweep used to run a full backtest for every grid point (about 1,000 per sweep, and for symbol-specific filters on a strategy with confirm symbols about 0.7 s each). It now screens the whole grid with the compiled loop from `api/fast_sweep.py` and runs only each indicator's best three through the engine.

- Market-wide filters and the legacy sweep score precomputed masks (`Scorer`). Symbol-specific filters on confirm strategies use `ConfirmSweepContext`, which composes each candidate's entry from per-symbol buy/hold arrays the way `apply_cross_symbol_signal` does. It checks itself against the real frame first and falls back to the slow path if it ever disagrees.
- The rows that come back (and each row's full-period figures) are produced by `fast_metrics.ranking_metrics`, a numpy port of `_ranking_metrics` that is tested against the real engine and returns None, so the real path runs, for anything unusual (zero-day strategies, blown-up equity, no numba).
- The refine sweep skips thresholds with fewer than `MIN_IN_SAMPLE_TRADES` trades when picking the top three, so a thin threshold no longer takes a place that is dropped afterwards.
- Zero-day ("og") strategies keep the old path.

**Indicator combo sweep** (Strategy Builder → Refine → *Indicator combo sweep*, `POST /backtests/builder/refine` with `mode: "indicator-combo-sweep"`):

Searches *pairs* (exhaustive) or *trios* (the best 150 pairs extended by every third filter) of the same threshold filters the indicator sweep tries, ANDed onto the draft's entry.

- Each filter's mask is computed once. `api/fast_sweep.py` runs a numba copy of `long_strat` plus a Sharpe that mirrors `stats.py` on `base AND mask_a AND mask_b`, about 0.2 ms per combination against about 22 ms through `execute_strategy`. A 25-year SPY pair sweep (about 100k to 160k combinations) takes a few seconds.
- It is a screen: the top rows are re-run through the real engine, so every figure shown comes from `execute_strategy`. `tests/test_fast_sweep.py` checks the screen against the engine (long, short, hold-on-buy, proxy, leverage vehicle). Change `long_strat` and that test shows where the screen drifts.
- Ranked in-sample like every refine mode; each row carries its full-period figures. The header shows how many combinations were tested. The more tested, the luckier the winner, so check the full-period figures.
- Entry filters only, hold-days strategies only. With confirmation symbols only market-wide indicators are combined.
- Needs `numba` (in `requirements.txt`). Without it the mode returns an error.

**Basket test** (Strategy Builder → Refine → *Basket test*, `POST /backtests/builder/refine` with `mode: "basket-sweep"` and `basket_symbols`):

Runs the draft strategy on other symbols over their full history to see whether it holds up beyond the one it was built on (`api/basket_sweep.py`).

- Each symbol goes through the same loader and engine as the Backtest button, so the draft's own symbol, run as the reference row, matches Backtest.
- Per symbol the strategy is adapted: a confirm symbol equal to the basket symbol is dropped, and the proxy symbol is ignored.
- `NullSharpe` and `NullP` rank each symbol against the same entry signals shifted in time (300 circular shifts, via `fast_sweep.shift_null`). A stock that rose a lot looks good under any timing; beating its null is the evidence the entries add something there. By chance about 5% of symbols clear p <= 0.05. Without numba these columns are empty.
- Symbols that fail to load or have under 250 bars become rows with an `Error`; the sweep carries on.
- Data is downloaded one symbol at a time (yfinance breaks under concurrent downloads) and cached, so a new basket takes a few seconds per symbol the first time.
- Stocks in a basket move together, so rows are related checks, not independent proof.

### Draft baseline in refinement results

Every refinement result except the basket test opens with **Draft as it stands**: the unmodified draft scored with the same frame preparation and metric code as the sweep rows, on the window the rows are ranked on (in-sample) and on the full period (`meta.draft` in the `/backtests/builder/refine` response, from `services._draft_baseline`). The columns match the table below it, so a row can be read straight against it, and a line counts how many rows score above the draft's in-sample Sharpe. Like the rows, these figures leave out each series' best year. A draft that cannot be scored gives no baseline (`null`) and the sweep still runs. A basket test already shows the draft's own symbol as its reference row.

### Aggregate metrics & outlier exclusion

A result's cards mix two bases on purpose:

- **CAGR, Sharpe and Sortino leave out the best year** (described next). This is also what sweeps, refine tables, regime scores and the contribution table use.
- **Max drawdown covers every year**, so the worst fall shows whichever year it happened in. It comes from the full-period block (`full_max_drawdown` in the summary payload, from `stats.full_period_metrics`). Sweep tables, which rank on the ex-best-year figures, still show the ex-best-year drawdown.
- A **best year share** card shows how much of the summed yearly log return the best year supplies (it can pass 100% when the other years lose money). A line under the cards gives CAGR, Sharpe and Sortino with the best year left in, and says what the drawdown would be without it when that differs.

The full-period block is computed only on request (`compute_aggregate_metrics(..., include_full=True)`), so sweeps pay nothing for it.

When `ExcludeBestReturnYear = True` in `config.py`, aggregate stats (Sharpe, Sortino, CAGR, MaxDD, trade count, win rate, Kelly) **exclude the single calendar year with the highest positive return**. This keeps one extreme upside year from dominating comparisons.

- Sharpe and Sortino take daily returns from the **unbroken** equity series and then leave out the excluded year's own returns. Dropping the year's rows first would book its whole gain as one phantom daily return across the gap (+1,570% for 2020 on one portfolio) and collapse Sharpe (0.44 against 1.29 done properly). The first session after the removed year is still measured against the prior close.
- **Latest Rolling PnL** always reflects the full backtest (real total)
- **Yearly breakdown** always shows every year
- A note is printed when a year is excluded, e.g. `(Aggregate metrics exclude 2020 — best year at 906.88%)`

Set `ExcludeBestReturnYear = False` to restore the original behavior.

### Buy-and-hold benchmark

Every daily backtest and portfolio result is compared with holding the instrument all the time (`api/benchmark.py`, `benchmark` in the payload, `benchmark_underlying` and `benchmark_vehicle` on each equity-curve point). Two benchmarks are drawn on the equity chart (each has a switch) and listed in the **Versus buy & hold** table under the cards:

- **Underlying**: the traded symbol, unleveraged (`SOXX buy & hold`).
- **Vehicle**: holding the traded product the whole time, which is what the engine would have earned with no signals: the leverage symbol's real returns where it exists, the leveraged stand-in before that (`SOXL buy & hold (as SOXX 3x before 2010-03-11)`). A symbol with no leverage-settings entry earns the general leverage times its return. With leverage 1 and no product the two would be the same line, so only one is shown.

Both start from the result's first scored equity value, over the bars the metrics cover, so the lines overlay directly. The table uses the same basis as the result cards (CAGR, Sharpe and Sortino without that series' own best year, drawdown over every year) and adds growth as a multiple and the share of bars the result was invested.

A strategy with a proxy symbol is compared with the proxy. A portfolio is compared with the symbol most of its strategies trade (ties go alphabetically; every traded symbol is listed under `benchmark.trading_symbols`). Not shown for short strategies, custom intraday data, or the Regimes views.

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

## Leverage settings

The **Settings** page decides what each trading symbol *earns*. Signals never change: strategies read their entry and exit signals from the original symbol, and exits (the "profitable closes" count) read the unlevered closes. So a strategy makes exactly the same trades whatever it is traded as; only the returns differ.

- **General leverage** (default 3, from `config.Leverage`): any symbol without an entry earns its own daily return times this factor, compounded, a daily-reset leveraged product.
- **Symbols**: an entry names a symbol, a *leverage symbol* and a leverage, for example SOXX, SOXL, 3. The lookup uses the symbol a strategy actually trades (its proxy, or the portfolio's proxy), not the symbol its signals come from.
- **A symbol with no leverage symbol**: leave the leverage symbol blank (or type the symbol itself) and the symbol is traded itself at the leverage you give. `RIVN`, blank, `1` trades RIVN unleveraged while everything else keeps the general leverage, which is how to test strategies on single stocks without leverage. There is no product to look up, so the stand-in carries every trade for the whole history, and the cost model fields (shown only when the leverage is not 1) apply to every day. The leverage symbol is stored as `null`; entries that have one are unchanged.

How an entry is used, in a backtest or a portfolio:

1. **After the leverage symbol exists**, a new position is opened in it and earns its real daily returns (leverage 1: SOXL's own prices, so splits, fees and financing are already in them).
2. **Before it exists**, the position is held in a *stand-in*: the symbol's own return times the entry's leverage, with an optional cost model (below).
3. **The instrument is chosen when a position opens and kept until it closes.** A trade already open on the day the leverage symbol first has a close finishes on the stand-in; the next entry uses the leverage symbol. In a portfolio the rule applies per symbol to the whole book: while the book holds nothing in a symbol, its next position opens in the leverage symbol if that has a close that day; adds, partial exits and strategies joining stay in the instrument that is open. A day the leverage symbol has no usable return uses the stand-in's return for that day.
4. **Shorts never use a leverage symbol** (SOXL is a long product); they keep the general leverage.
5. If the leverage symbol has no price history, the stand-in carries every trade and the result says so. With no leverage symbol at all the result says the symbol was traded itself at its leverage.

**The stand-in's cost model** is optional. An ideal `leverage * return` flatters history from before the leverage symbol existed, because a real leveraged fund pays fees and swap financing. SOXL against 3 x SOXX over 2010 to 2026 (4,176 days) shows a slope of 2.96 and a gap of about -6.4% a year that tracks the T-bill rate, roughly 1.9% plus 2.6 times the 3-month T-bill rate (the yearly gap was about -1% in 2010 to 2016 and -14% in 2023 to 2026). On the "New OG Based" signals that ideal stand-in overstates All-In CAGR by about 5 points before March 2010 and understates the max drawdown by about 2 points. Three optional fields shape the stand-in: *beta* (blank means the leverage), *annual fee* and *financing x T-bill*. **Estimate from history** fits all three on the days both symbols traded and shows how well the fit tracks (correlation, tracking error); the T-bill rate is Yahoo's `^IRX`. A flat rate series is left out of the fit, since it cannot be told apart from the fee.

What you will see:

- Position Episodes label each episode with the instrument it held (`SOXL leverage symbol`, or `SOXX x3 stand-in`) and show that instrument's real closing price. The allocation panel says which leverage symbol was used and since when; single-strategy backtests report how many trades used each.
- Trade percentages everywhere are the compounded return the position earned. Equity curves for symbols without an entry are unchanged.
- Entering a leverage symbol changes results: on the "New OG Based" signals, All-In from March 2010 on drops from 144% to 137% CAGR with SOXX traded as SOXL.

Details worth knowing:

- Settings live in `data/settings.json` and apply to every run immediately, including saved portfolios (like child portfolios, they are read live). A missing file means general leverage 3 and no entries, the behavior before this feature.
- One entry per symbol; a leverage symbol, when there is one, cannot be shared and cannot itself be a listed symbol (no chains); several symbols may have none. A portfolio that also trades a leverage symbol as a symbol of its own is refused with a message.
- Leverage-symbol closes and the T-bill rate are downloaded once per trading day into `data/market_cache/vehicle/` and are independent of how many years a backtest asks for.
- All-In on a symbol with an entry runs through the allocation engine (the original overlay follows one plain return stream); on a symbol without one it is unchanged.
- The `%Change` indicator stays at the fixed legacy factor of 3 so saved rules that use it never change meaning when the general leverage changes. The legacy "original" strategy paths (`hold_days` of 0) are not covered by these settings.
- Not yet: extending history beyond a symbol's own listing with an index (for example ^SOX for SOXX, back to 1994). Signals of strategies that read SOXX itself would also need it.

## Market regime scores

The Regimes screen scores how a strategy (or portfolio) has done in each market regime state, counting only the trades **entered** in that state. Two scores run from 5 to 100, and each is a weighted geometric mean of parts that are floored at 0.05, so one terrible part cannot hide behind good ones.

| Part | Weight | Full marks at |
|---|---|---|
| Sortino, over the bars the book was invested | 0.30 | Sortino 3 |
| Average trade return | 0.25 | +3% per trade |
| Max drawdown | 0.15 | 30% or less (zero at 70%) |
| Robustness: how much of the average yearly log return survives dropping the best year | 0.30 | |

- **Robustness score** (the default on the Regimes screen) uses only the average-trade and robustness parts, with their weights renormalized. It is the score regime-adjusted allocation reads by default.
- **Regime score** uses all four.
- Scores are absolute, not relative to the strategy's own baseline: a poor strategy is poor in every state and is scored that way.
- Sortino is taken over the bars the book was invested. Flat bars used to count as zero returns, which shrank Sortino by about the square root of the share of time invested and made rarely-invested regimes look worse than they were. Note the Sortino part now reaches full marks at a Sortino of 3 often (about four in five regime states when tested), so it separates good states from great ones less than it separates bad from good.
- **VIX daily change** is the one regime with **overlapping** buckets: the VIX's close-to-close % change, in `≤ -15%`, `≤ -10%`, `≤ -5%`, `≤ 0%`, `≥ 0%`, `≥ 5%`, `≥ 10%`, `≥ 15%` and `≥ 20%`. A day (or a trade's entry bar) belongs to every bucket it satisfies, so a -12% day is in `≤ -10%`, `≤ -5%` and `≤ 0%`, and exactly 0% is in both `≤ 0%` and `≥ 0%`. A trade counts in each of its buckets, so bucket trade counts and days add up to more than the totals. "State changes" on the analysis table means any change in the set of buckets. Its builder flags are `Regime_vix_change_<bucket>`. A missing VIX close blanks that day and the next. It is not one of the 15 regimes regime-adjusted allocation reads, because one day would then take several penalties from the same regime.
- **VIX vs realized vol** is the VIX divided by SPY's 21-day realized volatility (std of daily log returns × √252, in %), ranked as an **expanding percentile** against every earlier day of that ratio. Nothing after the day is ever used, and a day is not labeled until 252 earlier readings exist. Buckets: `≤ 10th` extreme realized dominance, `10–30th` realized dominance, `30–70th` typical, `70–90th` elevated implied premium, `> 90th` extreme implied premium (shared edges go to the lower bucket). The ranking starts at the beginning of the downloaded regime calendar (the backtest span plus two years), so the same date can land in a different bucket on a 5-year and a 25-year backtest. Not one of the 15 allocation regimes. Builder flags are `Regime_vix_rv_<bucket>`.

## Portfolio allocation models

A portfolio combines saved long strategies into one book. How much of that book is invested is the **allocation model**, chosen per request and saved with the portfolio. Strategy signals, entry/exit timing, and price conventions are never changed by it.

| Model | Target exposure |
|---|---|
| **All-in** (default) | 100% while any strategy is active, 0% otherwise. On one trading symbol this is the original behavior and runs through the original overlay code, so existing results are unchanged. On several symbols it stays fully invested and splits by active strategies (see below). |
| **Proportional, linear** | `active / total` |
| **Proportional, early conviction** | `(active / total) ^ 0.5` |
| **Proportional, confirmation required** | `(active / total) ^ 2` |
| **Regime-adjusted** | `strategy confidence x (70% + 30% x portfolio confidence)`, both from how the current market regimes have scored. See below. |
| **Tiered** | One strategy active: `tier_single_pct` of the book (default **55%**, editable, 0-100). Two or more active: **100%**, fixed. 0% with nothing active. |

All proportional modes give 0% with nothing active and 100% with everything active. Proportional, regime-adjusted and tiered let every strategy keep its own entries and exits, and each counts toward exposure.

**Tiered** exists because how many strategies agree is the one thing that held up in research on the SOXX book: the next-day Sharpe of the held 3x instrument rose from 1.4 with one of four strategies active to 3.3, 4.1 and 8.2 with two, three and four, in both halves of the history. Reducing only the lone-strategy days gave about early conviction's return with a shallower drawdown. It is also all that fits under a 100% cap: no margin. It counts the portfolio's **direct members**, so use a flat portfolio of strategies (a portfolio of two child portfolios counts the two children). A single weight of 100% is All-In. Lower it for a shallower drawdown and a lower return; the Drawdown risk panel shows the trade. Only who is active matters, so a change in who is eligible does not trigger a rebalance.

**Several trading symbols.** A strategy trades its *proxy* if it has one (or the portfolio's proxy, which overrides every strategy), otherwise its own symbol. That trading symbol, not the signal symbol, decides whether a book is one symbol: SPY-signal strategies with a SOXX proxy and SOXX strategies are all SOXX.

- **All-in on several symbols** is always 100% invested while any strategy is active, split across the active strategies' symbols in proportion to how many strategies are active in each (linear proportional, scaled to 100%). With 3 SOXX strategies and 1 XBI strategy: only SOXX active is 100% SOXX, only XBI is 100% XBI, one SOXX plus XBI is 50/50, all four is 75/25. When another symbol's strategy fires, the book sells down to the new split; when it exits, the book goes back up. It rebalances only when a strategy fires or exits, so weights drift with price in between.
- **One symbol** keeps the original overlay, so single-symbol results do not change. The two agree on one symbol (the tests check it on random signal sets).
- Proportional and regime-adjusted already split each strategy's share into its own instrument.
- The engine's All-in is the benchmark everywhere else: attribution, the comparison table, the regime model's portfolio-consensus baseline, and the leave-one-out and Shapley coalitions use it on mixed books. The old overlay follows only the first holder's instrument and ignores the others, so it is no longer used for mixed books.
- The ATR regime label needs one symbol and is blank on a mixed book.

**Overlapping signals:** every portfolio holds until all strategies exit: it stays in while any active strategy is still holding, new buys join the active set, and a same-day exit and entry keeps the portfolio in. There is no setting for it. Saved portfolios and API requests may still carry an `overlap_mode`, which is read and ignored, so a portfolio saved as `first_signal_only` now runs as hold-until-all-exit.

**Simulation rules (proportional)**

- *Active* means a strategy holds into the next bar: held and not exiting, or entering. *Total* counts every strategy that is **eligible**, whether or not it has ever entered.
- A strategy becomes eligible on the first bar its entry could be evaluated (`EntryReady`: indicator warmup, instrument listing, regime labels) and stays in the denominator after that. Gaps in readiness later on are counted and reported, not dropped. A strategy whose data ends is removed from the denominator and can no longer be active. Strategies without an `EntryReady` column are eligible from their first bar. Metrics start on the first bar any strategy is eligible.
- Exposure is a **target share of current portfolio equity**, not a separate investment per strategy. It is recomputed whenever the effective state changes (who is active, or who is eligible), and the existing position is rebalanced to it: partial entries, partial exits, and full liquidation all come from the same rebalance. Between changes the position drifts with price.
- With several instruments (different strategy symbols, no portfolio proxy), the exposure is split evenly across the active strategies and each share is held in that strategy's instrument.
- Timing matches the single-strategy backtester: a signal on a bar's close is traded at that close and earns the next bar's move. Each instrument trades as its leveraged return series (the general leverage times its daily return, or its leverage symbol's real returns; see Leverage settings), so a fully invested book compounds exactly like the all-in overlay.
- Cash earns nothing. Trading cost is `transaction_cost_bps` of traded notional, on buys and sells, default `0`. It is charged to every model, including all-in, so comparisons are fair. With cost `0` the all-in path is untouched.
- The book tracks cash, invested capital, quantity, average-cost basis, realized and unrealized P&L (realized is gross of costs; costs are tracked separately), and total transaction costs.

**Analytics** (Allocation tab on a portfolio result, and `allocation` in the API payload)

- Average exposure, time invested, exposure when invested, rebalance count (entries, exits, partial adjustments), turnover (total and annualized), and trading costs.
- Performance attributable to allocation: against an all-in book on the same signals, the gap splits into an allocation effect, a cost effect, and a remainder (basket drift), with a breakdown by number of active strategies.
- Per-strategy warmup and eligibility.
- **Compare allocation models** (on demand, `POST /portfolios/allocation-comparison`): All-in, regime-adjusted at each of the four penalty presets, each proportional mode, and Tiered at the request's `tier_single_pct`, over the same window, rebased to the same starting capital, with metrics and an equity-curve overlay. The All-in row is always 100% while any strategy is active (split across symbols on a mixed book). The regime rows share one scoring pass and read the request's `regime_score`.

  **Include child portfolio nesting** (a checkbox before Calculate, off by default; `include_nesting` in the request) also regroups the portfolio's strategies into **child portfolios**, each with its own allocation model, and adds the results as rows. For four strategies that is the three ways to pair them (`A+B | C+D`, `A+C | B+D`, `A+D | B+C`), and for three a pair and a single. Strategies are lettered A, B, C... in the portfolio's own order, and the legend above the table says which is which. Each variant pairs a parent model with a child model (both children share one) from All-in, Linear, Early conviction, Confirmation required and Tiered, so 25 pairs per grouping and 75 rows for four strategies, which takes about 40 seconds. Regime-adjusted is left out, because it needs its own regime history for every child. Rows read `grouping · parent model / child model`, for example `A+B | C+D · Linear / All-in` is two child portfolios held in proportion to how many are active, each child all-in on its members.

  Each variant is a real run of the nested engine, the same one a saved portfolio of child portfolios goes through, with the portfolio's trading cost on every portfolio in it. A grouping of one strategy stays a direct member of the parent. Nothing else is nested: it is one level of children, never children of children. Beyond four strategies the number of ways to split them grows quickly, so the run is capped at about 80 variants: the model list shrinks first (to four, then three), then groupings are dropped, spread evenly across the list. The comparison says what was cut. A portfolio that already has child portfolios, or fewer than three strategies, has nothing to regroup and the checkbox is disabled.

  With dozens of rows the chart plots only the ticked ones. It starts with the **top 5 by Sharpe** (also with nesting off, where there are nine rows), and the table has a checkbox on every row, plus buttons for the top 5 again, none, and the flat models only. Nested rows come after the flat ones, best Sharpe first. **Every column heading sorts the table** (all rows together): the first click puts the best value first (the lowest for Max DD, the drawdown probabilities, rebalances and turnover), the second reverses, the third returns to the default order. Missing values always go last, and sorting never changes which rows are on the chart or their colours. Treat the top rows with care: picking the best of 75 variants on the same history that scored them is selection, and some of the best Sharpe is luck.

**Code layout** (each piece is replaceable on its own)

```
api/allocation.py            pure models: SignalState -> target exposure. No prices, cash, or dates.
api/rebalancer.py            PortfolioLedger: trades to target weights, keeps the books.
api/allocation_engine.py     effective signal state, then model -> ledger, bar by bar.
api/allocation_analytics.py  exposure, turnover, attribution, comparison.
api/regime_allocation.py     regime-adjusted inputs: penalties, confidence, decision bars.
api/regime_scoring.py        regime scores as they stood on any past bar, in one pass.
api/portfolio_service.py     wiring: eligibility from strategies, dispatch, payloads.
```

To add a model (volatility targeting, correlation-aware weighting), implement `target_exposure(state)` as a class in `api/allocation.py` and register it in `resolve_allocation_model`. Signals the model needs beyond the active count can travel in `SignalState.context`. The ledger, analytics, and API do not change.

Request fields on `/portfolios/simulate` and saved portfolios: `allocation_model` (`all_in` | `proportional` | `regime_adjusted` | `tiered`), `proportional_mode` (`linear` | `early_conviction` | `confirmation_required`), `tier_single_pct` (tiered only, 0-100, default 55), `transaction_cost_bps`, and the regime fields below. **A portfolio owns its allocation**: saving it stores all of these, loading it (also from the Scan screen) restores them, and strategies carry none of it. Portfolios saved before these existed load as all-in.

### Regime-adjusted allocation

Starts at 100% and takes confidence off for every market regime state that has treated the active strategies, or the portfolio, badly. Pick how hard to punish (Portfolio screen, Settings, Allocation model, Penalties):

| `regime_penalty` | Score at or below 10 | Score above 10, up to 25 |
|---|---|---|
| `harsh` | -100% | none |
| `strict` | -50% | -25% |
| `medium` (default) | -33% | -10% |
| `low` | -25% | -5% |

`regime_score` picks the score that is read: `robustness` (default) or `portfolio` (the four-part Regime score). A regime state needs 20 closed trades before its score counts; until then it costs nothing.

How it decides, using only what was known on the day:

1. When a strategy joins, or one leaves while another stays (and for the first entry), every **active** strategy is scored against the 15 regime states in force that day, using its own backtest history up to that day. Each state at or below a cutoff takes its penalty off that strategy's confidence, which starts at 100%.
2. **Strategy confidence** is the average of the active strategies' confidences.
3. **Portfolio consensus** is the same calculation on the portfolio's own history: the unadjusted all-in portfolio of the same strategies (so the decision does not feed on itself).
4. Exposure = strategy confidence x (70% + 30% x portfolio consensus). The position moves to that share of current equity at the bar's close. It can rise when a strategy exits and another stays; it is only revisited when the set of active strategies changes, and it is flat when none is active.

The 15 regimes are SPY trend, VIX, equity-risk breadth, sector deviation from its 50-day average, copper/gold, materials strength, credit risk-on, bond duration, housing strength, housing vs defensive, rate shock, 10Y-3M curve, rate shock x curve, 10Y-3M curve change, and dollar + rates. With every penalty at 0% the model is All-in.

Things to know:

- Early history is fully invested because no state has 20 trades yet. A drawdown in the first years is therefore the same as all-in's.
- Scores come from `api/regime_scoring.py`, which keeps running statistics instead of rescoring from scratch at every decision (seconds instead of about twelve minutes), and is tested against the direct scorer bar for bar.
- Contribution and Shapley use the same rules for every smaller portfolio, reusing each strategy's scores. A Shapley run scores one portfolio history per coalition, so it takes noticeably longer than for other models.
- The Scan screen sizes a regime-adjusted portfolio like a linear one and marks it `≈`, because its scores need 25 years of history. **Compute exact** in the row's details runs the full history for that portfolio.

## Scan: what to do with each row today

The Scan page used to read three flags per row, buy, hold and sell, from whether a portfolio crossed 0%, so a move from 55% to 100% or back looked like "hold". Each row now shows an **Action** and a **Position**, read from the last bar of the same engine run that backtests use.

| Action | Meaning |
|---|---|
| **Enter** | from nothing to a position |
| **Add** | the position grows (for example 55% to 100% when a second strategy joins) |
| **Reduce** | the position shrinks but stays open |
| **Exit** | back to nothing |
| **Rebalance** | the total is unchanged but the split between instruments moved (a mixed-symbol book) |
| **Hold** | no change today, whatever price did to the weight |
| **Flat** | not in the market |

The Position column is `held before → held now` as a share of that portfolio's own book (100% is the whole portfolio, never of an account, which the app does not know). On a Hold day it shows the drifted share and the model's target, for example `61% (target 55%)`: the model only trades when a signal changes, so shares drift with price in between and there is nothing to do. Any change counts, however small; there is no minimum trade size.

Click a portfolio's position for the details: the reason ("Sig 16 exited, 1 of 4 active"), held before and now, the **orders** per instrument (the product the engine would route to, SOXL once it has a close and the stand-in before, with the ordinary price), what is held afterwards, and the active strategies. An **Account $** box next to the search (kept in your browser only, `50k` works) turns each order into dollars and whole shares at the last close. A stand-in with leverage other than 1 has no product to buy, so it shows dollars only.

Single strategies are all-in, so they show Enter, Hold, Exit or Flat at 0% or 100%. For an all-in portfolio the actions map one-to-one onto the old Buy, Hold, Sell. The old flags are still in the response (`buy_signal` is true for Enter, Add and Rebalance; `sell_signal` for Exit, Reduce and Rebalance) next to the new `position` object ([`api/scan_position.py`](api/scan_position.py)). A portfolio's exact position at its real sizing is also available at `GET /api/scan/portfolios/{id}/position`.

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

**Prices are real prices.** Every price shown is the closing price on the trade date of the instrument traded: the underlying's own, or its leverage symbol's. Internally the ledger trades each stand-in instrument as its leveraged return series (starting at 100), and that simulated price never appears as a price in the UI. The payload keeps it under `simulated_price`, `initial_simulated_price` and `final_simulated_price` for audit, and the real close under `underlying_close`, `initial_underlying_close` and `final_underlying_close`. Because the position is leveraged, the move between an entry and exit price is not the episode's return; the episode and event percentages come from the ledger.

**Episode % is what the trade earned.** A trade's percent is its daily leveraged returns compounded, which is the equity the position made over the trade. Under All-In on one plain symbol the old overlay reports the same number, and the ledger's own figure (episode P&L / equity just before the first entry) agrees with it, so the table never contradicts itself. This replaced an earlier convention, `Leverage * (exit close / entry close - 1)`, which ignored that the leverage compounds daily and so did not match the equity curve (for one trade 2026-10-06 to 2026-10-12 it gave -21.98% against -20.89% earned). The ledger figure is also kept as `return_on_equity`. Under the proportional models, where positions are fractional, Episode % is the same ledger figure.

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

## Drawdown risk

A backtest is one path, so its maximum drawdown is one draw from a wide distribution. Every portfolio
result carries a **Drawdown risk** panel with `P(DD > 40%)`, `P(DD > 50%)` and `P(DD > 75%)`: the share
of paths, as long as the backtest, whose worst fall from a peak went beyond that depth. The Compare
allocation models table has the same three columns for every model.

The paths come from reshuffling the finished equity curve's daily returns in 20-day blocks
(a circular block bootstrap, 4,000 paths, 1,500 per row in the comparison; fixed seed, so a result
repeats). Blocks keep the clustering of good and bad days. Because it works on the curve, it applies to
any portfolio however it was sized. It costs about 0.6 s per result.

Read it as a way to compare rules, not a forecast: it resamples this history, so a regime the history
never contained is not in it. The share has a sampling error (shown in the panel), larger for
probabilities near 50% and relatively larger for rare tails. The code is in
[`api/drawdown_risk.py`](api/drawdown_risk.py); the thresholds are `THRESHOLDS` there.

## Running tests

Run the tests with the project's virtual environment (the one created in the setup above), not a
system or conda Python. The tests need packages that only the venv is guaranteed to have: pydantic 2,
`ib_insync` and `python-multipart`. With another interpreter some tests fail on imports or
pydantic behavior even though the code is fine.

```bash
venv/bin/python -m unittest discover -s tests -t . -v
```

With the venv already activated (`source venv/bin/activate`) this is the same as
`python -m unittest discover -s tests -t . -v`. A single file runs with
`venv/bin/python -m unittest tests.test_allocation`.

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
- **Earnings:** `DaysToEarnings` and `DaysSinceEarnings`, for stocks (see below)
- **Context:** VIX, VIX daily change (`VixChange`, in percent points: -10 means the VIX fell 10% that day), SPY 50/200 bull flag, Donchian/Keltner/PSAR breakout flags

Session VWAP is not available (daily EOD data only).

### Earnings indicators

`DaysToEarnings` and `DaysSinceEarnings` let a rule avoid, or look for, the days around a company's earnings report, for example *don't hold into earnings*:

- Entry: `Days to earnings > 3`. With a 3-day hold, nothing opened under this rule is exposed to a report. Use a number at least as large as the hold.
- Exit: `Days to earnings <= 1` closes a trade at the close before the report.
- `Days since earnings` is the other side: `= 0` is the reaction day itself, `<= 2` the first sessions after.

They count **trading bars** to the next / since the last **reaction day**, the first session whose price reflects a report. A report before the open (a time before noon, Eastern) reacts the same day; one after the close reacts the next session. So `DaysToEarnings` is 1 on the bar before the reaction day, where a position opened at that close is exposed to the report, and 0 on the reaction day. As a check on AAPL over ten years, the average absolute move on the 43 reaction days was 3.7%, against 1.2% on every other day.

Where no date is known (an ETF or index, a symbol Yahoo has nothing for, bars before the first or after the last date it lists) both read **999**. That keeps `Days to earnings > 3` true on an ETF instead of blocking every trade, and keeps an exit such as `<= 1` from ever firing there. They are not available for uploaded intraday data. Like the VIX filters, they apply to the traded symbol after any cross-symbol confirmation, so a confirm symbol's earnings never gate your trade. They are also in the indicator sweeps.

The dates come from Yahoo through yfinance ([`api/earnings_data.py`](api/earnings_data.py)), which scrapes them, at most 100 rows a call, so the module pages. History reaches back to the late 1990s or early 2000s for large companies and to the IPO for newer ones. Each symbol is cached in `data/market_cache/earnings/` and refreshed weekly (a symbol with no dates is rechecked monthly). A failed download keeps the old cache and is never remembered as "no earnings". Known limits:

- **A quarter is occasionally missing** (AAPL has none listed from January to October 2001), and during such a stretch the rule does not protect. `earnings_data.coverage("AAPL")` lists the gaps, the number of reports and the upcoming date.
- **The next report is a projection until the company confirms it**, usually two to five weeks ahead. A rule of up to about 10 trading days is realistic in live trading; a much longer one would use a date you would not have known.
- **ETFs have no earnings**, so a rule never sees, say, NVDA's report moving SOXX.
- A weekend report is moved to Monday, and a reaction day beyond the last bar is counted in weekdays, so a market holiday in between can be off by a day.

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
- **Leverage** is set on the Settings page (general leverage, and a leverage symbol per symbol such as SOXX to SOXL). `config.Leverage` only supplies the default and the fixed factor of the `%Change` indicator.
- **CSV output** goes to `CSV/` (gitignored).

## Disclaimer

This software is for **research and education only**. It generates signals and simulates historical performance — it does not execute live trades. Past backtest performance does not guarantee future results. Use at your own risk.


### Nested portfolios

A portfolio can select direct strategies, saved child portfolios, or both. `portfolio_ids`
contains live references; each run resolves a consistent snapshot of saved settings.
Old portfolios default to no children. Cycles, missing references, duplicate child references,
and empty portfolios are rejected. A referenced child cannot be deleted until its parents
remove the reference. The same strategy can intentionally appear in separate sleeves.

Each child counts as one immediate member, regardless of its strategy count. Its signal is
active while any eligible descendant strategy is active, even when its allocation model
rejects all exposure. Parent allocation assigns capital budgets; child models independently
invest their budgets. For example, a 50% child budget at 40% child exposure invests 20% of
parent equity and reserves 30% as cash. V1 respects child rejection: cash is not redistributed.
A future redistribution policy is intentionally not enabled yet.

Every level uses its own All-In, proportional, or regime-adjusted model. Changes to child
signals can rebalance within its budget without rebalancing siblings. Parent member-state
changes resize budgets using current equity. Holdings and budgets drift between events;
resizing never resets strategy signals. Regime scores for a child use its standalone allocated
history available at the decision date; parent consensus uses an All-In parent with unchanged
child models. Warmup remains 20 trades. Model comparisons change only the parent model.

Parent proxies and cost settings apply only to direct strategies. Children retain their own
proxies and costs. All trades use the existing underlying-close and leveraged-return timing
conventions. Child sleeves plan budgets and holdings, then targets are aggregated by symbol
and executed through the same single weighted-average-cost ledger as flat portfolios.
Internal transfers create no trades, fees or realized P&L. Only the net account order is
executed. When children have different fee settings, its rate is weighted by the planned
same-direction order quantities; offsetting quantities are matched proportionally. Execution
costs reduce all sleeve budgets proportionally, preserving reserved-cash weights. Episode
reports use the account's executed fills and cost basis. No invented child-fund price appears
in the trades table.

Allocation analytics use actual underlying exposure including child cash. The member table
shows end-of-run budget, invested value and cash as percentages of total equity. Contributions
and Shapley operate on immediate members, removing an entire child as one unit. Saved children
are loaded by reference, so editing a child affects subsequent parent replays. Existing flat
portfolio execution paths remain unchanged.
