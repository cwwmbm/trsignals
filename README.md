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

# Edit the config block in run_backtest.py, then:
python run_backtest.py
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

Open the Vite URL (usually `http://localhost:5173`). The frontend talks to the backend at `http://localhost:8000` by default. Override it with `VITE_API_URL` if needed.

## Project layout

```
TradingStrategy/
├── api/                 ← FastAPI backend for the web UI
├── frontend/            ← React/Vite frontend
├── run_backtest.py       ← main entry point — configure RUN_MODE here
├── config.py             ← strategy parameters (ticker, RSI, leverage, etc.)
├── getdata.py            ← Yahoo Finance fetch, breadth, holiday filtering
├── indicators.py         ← indicators + 24+ buy signal definitions
├── backtest.py           ← strategy engine (og_strat, long_strat) + sweeps
├── stats.py              ← aggregate metrics, yearly breakdown, outlier exclusion
├── backtest_runners.py   ← load data, single-symbol & cross-symbol runners
├── indicator_sweep.py    ← grid-search indicator filters on a signal
├── signal_check.py       ← Streamlit daily signal dashboard
├── quote.py              ← quick indicator snapshot
├── IBConnect.py          ← legacy shim (delegates to run_backtest.py)
├── test_data.py          ← all signals × one symbol
├── test_indicator.py     ← one signal × many symbols
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

Backtest run settings live in the **config block at the top of [`run_backtest.py`](run_backtest.py)**.

## Backtesting (`run_backtest.py`)

Edit the config block, set `RUN_MODE`, and run:

```bash
python run_backtest.py
```

`python IBConnect.py` also works (legacy alias).

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

Trade the **primary** symbol at leverage. Buy only fires when the signal is true on **all** symbols in the confirm list. **Sell** is evaluated on the primary only.

```python
# Sweep — compare confirm combinations
symbol_confirmation_tryout(SIGNAL, 'SOXX', ['SOXX', 'SMH', 'QQQ'], years=25)

# Detail — one chosen combo with yearly breakdown
symbol_confirmation_detail(SIGNAL, 'SOXX', confirm_symbols=['SMH', 'QQQ'], years=25)
```

Typical workflow: **`symbol_confirm_sweep` → pick winner → `symbol_confirm_detail`**.

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
python test_data.py
```

Edit `yfticker` in the file to change the symbol.

### One signal across many symbols

```bash
python test_indicator.py
```

Change `buy_signal = ind.buy_signal11` at the top.

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

- **Trend:** SMA, EMA, Bollinger Bands, ATR
- **Momentum:** RSI (2/5/14), Stochastic, CCI, MACD
- **Custom:** IBR, Kaufman ER, ValueCharts, VFI, Hurst, Change Velocity
- **Breadth:** sector/index ratios vs SPY with RSI overlays
- **Context:** VIX, SPY 50/200 bull flag, volume vs EMA

## Data sources

**Yahoo Finance** via `yfinance`. [`getdata.py`](getdata.py) provides:

- Single-ticker fetch (`get_data_yf`) and bulk download (`get_bulk_data`)
- Shared market context (VIX, breadth ratios) via `load_symbol_dataset`
- Futures/FX suffix mapping (`NQ=F`, `GBPUSD=X`)
- NYSE holiday filtering

Legacy Interactive Brokers code is commented out and unused.

## Tips

- **Start with `run_backtest.py`** — all backtest modes are configured in one place.
- **Sweep then detail** — use `symbol_confirm_sweep` or `signal_combo_sweep` first, then drill in with `single` or `symbol_confirm_detail`.
- **`SIGNAL` vs `SIGNAL_A`/`SIGNAL_B`** — `single` and confirm modes use `SIGNAL`; combo sweep uses `SIGNAL_A` and `SIGNAL_B`.
- **Leverage** is applied in `long_strat` and `%Change` — adjust `Leverage` in `config.py`.
- **CSV output** goes to `CSV/` (gitignored).

## Disclaimer

This software is for **research and education only**. It generates signals and simulates historical performance — it does not execute live trades. Past backtest performance does not guarantee future results. Use at your own risk.
