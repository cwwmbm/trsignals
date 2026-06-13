# TradingStrategy

A Python-based trading **signal and backtesting system**. This repo does **not** place trades — it fetches end-of-day market data, computes technical indicators, evaluates buy/hold/sell signals, and backtests strategies against historical data.

## What it does

| Capability | Description |
|---|---|
| **Daily signal scan** | Pulls EOD prices from Yahoo Finance and reports which strategies are firing buy, hold, or sell across a watchlist of symbols |
| **Backtesting** | Simulates strategies over years of history with PnL, drawdown, Sharpe/Sortino, Kelly criterion, and CAGR |
| **Latest quotes** | Prints current price and key indicators (RSI, EMA, Stochastic, breadth, volume) for the configured ticker |

## Architecture

```
config.py          ← Strategy parameters (ticker, RSI thresholds, stop loss, leverage, etc.)
getdata.py         ← Yahoo Finance data fetch, market breadth, NYSE holiday filtering
indicators.py      ← Technical indicators + 24+ buy signal definitions
backtest.py        ← Strategy execution engine (og_strat, long_strat) + backtest sweeps
signal_check.py    ← Streamlit daily signal dashboard (multi-symbol)
quote.py           ← Quick indicator snapshot for config ticker
IBConnect.py       ← Single-strategy backtest runner with full stats output
test_data.py       ← Compare all signals on one symbol
test_indicator.py  ← Run one signal across many symbols
```

### Data flow

```
Yahoo Finance (yfinance)
        ↓
getdata.py  — OHLCV + VIX + sector breadth ratios (RSP/SPY, QQQ/SPY, etc.)
        ↓
indicators.py  — RSI, EMA, IBR, ValueCharts, VFI, Hurst, MACD, …
        ↓
buy_signalN()  — Boolean Buy/Sell arrays + hold days + profit target
        ↓
backtest.execute_strategy()  — Simulate entries, exits, rolling PnL
        ↓
Output  — Signal table, stats, or CSV
```

## Requirements

- Python 3.9+
- Internet access (Yahoo Finance)

### Install

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install streamlit        # required for signal_check.py
pip install ib_insync        # optional; only needed for IBConnect.py live IB features
```

Core dependencies: `yfinance`, `pandas`, `numpy`, `ta`, `hurst`, `pandas_market_calendars`.

## Configuration

All strategy tuning lives in [`config.py`](config.py):

| Parameter | Default | Purpose |
|---|---|---|
| `ticker` | `SPY` | Primary symbol for quote/backtest scripts |
| `RSI2Buy` / `RSI5Buy` | 15 / 35 | OG strategy oversold thresholds |
| `RSI2Sell` / `RSI5Sell` | 95 / 70 | OG strategy overbought thresholds |
| `stop_loss` | 0.15 | Exit when trade PnL drops below −15% |
| `Leverage` | 3 | Multiplier applied to returns |
| `VolumeEMAThreashold` | 0.6 | Volume vs 8-day EMA filter |
| `VolatilityThreashold` | 0.1 | Annualized volatility exit filter |
| `MondayBuy` / `LowVolumeBuy` | True | One-day buy rules in OG strategy |
| `UseProxyUnderlying` | False | Track PnL via leveraged proxy (e.g. SOXX) |
| `api_key` | `""` | Pushbullet key for mobile notifications (commented out) |

Change `ticker` and thresholds here before running `quote.py` or `IBConnect.py`.

## Usage

### 1. Daily signal check (Streamlit)

Scans ~18 symbols against 24 buy signals and shows today's buy/hold/sell state plus SPY market summary.

```bash
streamlit run signal_check.py
```

**Symbols scanned:** SPY, SMH, QQQ, SOXX, XLI, XLU, XLE, XLF, IWM, FXI, AAPL, GDX, MSFT, GLD, XBI, TLT (breadth-only symbols like ^VIX and RSP are used as inputs but not scanned).

**Output columns:**

| Column | Meaning |
|---|---|
| `Buy signal?` | Enter long today |
| `HoldLong?` | Currently in a position |
| `Sell signal?` | Exit long today |
| `Days` / `Profit` | Hold period and profitable-close target for timed strategies |
| `TradePnL` | Cumulative backtested return through today |
| `Kelly` | Kelly criterion sizing estimate |
| `Description` / `Verdict` | Human-readable signal notes |

Optional Pushbullet notifications are wired but commented out — set `api_key` in `config.py` and uncomment the send calls in `signal_check.py`.

### 2. Latest quotes and indicators

Prints today's close and key indicators for the ticker set in `config.py`:

```bash
python quote.py
```

Example output:

```
SPY: 0.45%
QQQ: 0.62%
Today's Close: 542.31
EMA8: 538.12
RSI2: 42.5
RSI5: 48.3
Stoch: 55.2
BreadthRSI5: 52.1
Volume EMA: -12.4%
```

### 3. Backtest a single strategy

[`IBConnect.py`](IBConnect.py) is the main backtest runner despite the name — it uses Yahoo Finance, not Interactive Brokers, for historical data.

```bash
python IBConnect.py
```

By default it runs `buy_signal4` on the configured ticker with 25 years of history. Edit the `buy_signal` assignment near the bottom of `main()` to test a different signal:

```python
buy_signal = ind.buy_signal7   # change this
```

**Printed stats:** trade count, rolling PnL, max drawdown, CAGR, win rate, avg win/loss, Sharpe, Sortino, Kelly, and yearly breakdown.

Results are saved to `CSV/{ticker}_{signal_name}.csv`.

#### Exploring a signal further

`IBConnect.py` supports three levels of backtesting on top of the base signal:

**1. Base run (default)** — Apply the signal's built-in `days`/`profit` hold rules and print full stats.

**2. Indicator tandem sweep** — Test how the signal performs when combined with *additional* indicator filters on top of its existing buy/sell logic. Uncomment in `main()`:

```python
results = indicator_tryout(data, days, profit, is_long, is_sell=False, check_breadth=False, check_both=False)
```

`indicator_tryout()` grid-searches dozens of indicators (RSI, IBR, breadth, VFI, Stochastic, ValueCharts, VIX, etc.) and ranks combinations by Sharpe ratio. For each indicator it tries threshold values across a range:

- **Buy side** (`backtest_ind`): extra filter is AND-ed onto the signal's `Buy` column — e.g. "only take `buy_signal7` entries when `RSI2 <= 30`"
- **Sell side** (`backtest_sell_ind`, set `is_sell=True`): extra filter is OR-ed onto `Sell` — e.g. "also exit when `Stoch > 80`"

Use `check_breadth=True` to include sector breadth RSI sweeps, or `check_both=True` to also sweep price/momentum indicators. You can also run a targeted sweep on one indicator:

```python
results = bt.backtest_ind(data, days, profit, is_long, 'RSI2GoldBreadth', 'both', 0, 100, 10)
```

**3. Hold-period sweep** — Find optimal days-to-hold and profitable-close targets for the signal's entry logic, ignoring the signal's default `days`/`profit`. Uncomment:

```python
results = bt.backtest_days(data, max_days=7, is_long=is_long)
```

This tries every combination of hold days (1–7) × profitable closes (1–days) and ranks by Sharpe.

**4. Manual one-off filters** — Ad-hoc tandem tests without a full sweep:

```python
data['Buy'] = data['Buy'] & (data['ValueCharts'] < 0)
data['Sell'] = data['Sell'] | (data['RSI14'] < 50)
```

These lines are already in `main()` as commented examples.

### 4. Compare all signals on one symbol

[`test_data.py`](test_data.py) runs every buy signal against a single ticker and prints a summary table (PnL, drawdown, trade count, win rate):

```bash
python test_data.py
```

Edit the `yfticker` variable (default `UUP`) to change the symbol. Set `years` in the `get_data_yf` call for lookback length.

### 5. Run one signal across many symbols

[`test_indicator.py`](test_indicator.py) backtests a single signal (default `buy_signal11`) across a broad symbol list with 20 years of data:

```bash
python test_indicator.py
```

Change `buy_signal = ind.buy_signal11` at the top to test a different strategy.

## Signals

Each signal is a function in [`indicators.py`](indicators.py) returning an 8-tuple:

```python
buy, sell, days, profit, description, verdict, is_long, ignore = buy_signalN(data, symbol)
```

| Field | Type | Meaning |
|---|---|---|
| `buy` | `Series[bool]` | Entry condition per bar |
| `sell` | `Series[bool]` or `False` | Exit condition (OG strategies define explicit sells) |
| `days` | `int` | Max days to hold (`0` = OG hold-until-sell strategy) |
| `profit` | `int` | Exit after N profitable closes |
| `is_long` | `bool` | Long vs short |
| `ignore` | `bool` | Skip this signal for symbols not in `allowed_symbols` |

### Signal inventory

| Signal | Allowed symbols | Style | Summary |
|---|---|---|---|
| `buy_signal1` | XBI | 2d/1p | RSI breadth + Close vs EMA8 + IBR |
| `buy_signal2` | — | 2d/1p | Pullback: close below 2d ago, 10d return negative, IBR ≤ 0.5 |
| `buy_signal3` | — | 2d/1p | Energy breadth + IBR3 + VIX + VFI10 |
| `buy_signal4` | FXI | 3d/1p | Gold/semis breadth + Stochastic |
| `buy_signal5` | — | 10d/100p | Kaufman ER + IBR (currently `buy=True`) |
| `buy_signal6` | — | 2d/1p | Risk breadth + ValueCharts |
| `buy_signal7` | SMH, QQQ, FXI, SOXX, SPY | 2d/1p | Close pullback + IBR ≤ 0.4 |
| `buy_signal8` | QQQ, SPY | 3d/1p | ER + ValueCharts + RSI2 + IBR |
| `buy_signal9` | SPY, QQQ | 100d | Stoch + MACD + IBR with sell rules |
| `buy_signal10` | SMH, SPY, SOXX, QQQ | 3d/1p | New low + IBR |
| `buy_signal11`–`24` | Various | Mixed | Experimental / symbol-specific |
| `og_buy_signal` | SPY | OG | Classic RSI2/RSI5 oversold + volume/volatility filter |
| `og_new_buy_signal` | SPY, IWM, QQQ | OG | OG buy + Stoch + SMA trend + ER filter |

> **Note:** Many signals have `allowed_symbols = []`, which means `ignore=True` for all symbols in the signal scan. Only signals with matching `allowed_symbols` appear in `signal_check.py` output for a given ticker.

## Strategy engine

[`backtest.py`](backtest.py) simulates position management:

### `execute_strategy(data, days, profit, is_long)`

Routes to one of three engines:

| Engine | When | Behavior |
|---|---|---|
| **`long_strat`** | `days > 0` | Enter on `Buy`, exit on `Sell`, max days, or N profitable closes |
| **`og_strat`** | `days == 0` | Hold until sell signal, stop loss, or one-day buy rules |
| **`long_og_strat_proxy`** | `UseProxyUnderlying=True` | OG logic but PnL tracked via leveraged proxy symbol |

**Position columns added to the dataframe:**

- `LongTradeIn` / `LongTradeOut` — entry/exit flags
- `HoldLong` — currently in a trade
- `DaysInTrade` / `ProfitableCloses` — exit timers
- `TradePnL` — per-trade return (leverage-adjusted)
- `RollingPnL` — compounded equity curve (starts at $15,000)
- `Drawdown` — peak-to-trough decline

**One-day buy rules** (OG strategies): Monday decline buy and low-volume capitulation buy, controlled by `MondayBuy`, `LowVolumeBuy`, `DownDays`, and `VolumeEMAThreasholdBuy` in config.

### Backtest sweeps

| Function | Purpose |
|---|---|
| `backtest_days(data, max_days)` | Grid search over hold days × profitable closes |
| `backtest_ind(data, …, column, condition, min, max, step)` | Filter buys by indicator threshold |
| `backtest_sell_ind(…)` | Same for sell-side indicator filters |

## Indicators

[`indicators.add_indicators()`](indicators.py) enriches OHLCV data with:

**Price / trend:** SMA (10/20/50/100/200), EMA (8/20/100), Bollinger Bands, ATR

**Momentum:** RSI (2/5/14), Stochastic, CCI, MACD histogram

**Custom:** IBR (Internal Bar Ratio), Kaufman Efficiency Ratio, ValueCharts, VFI (Volume Flow Indicator), Hurst exponent, Change Velocity

**Breadth (sector/index ratios vs SPY):** RSP, QQQ, SMH, XLF, XLE, XLU, XLI, IWM, GLD, TLT — each with RSI overlays

**Market context:** VIX, SPY 50/200 SMA bull flag, volume vs 8-day EMA

## Data sources

Primary source is **Yahoo Finance** via `yfinance`. [`getdata.py`](getdata.py) handles:

- Multi-ticker bulk downloads with shared breadth columns
- Futures suffix mapping (`NQ` → `NQ=F`, `GBPUSD` → `GBPUSD=X`)
- NYSE trading-day filtering via `pandas_market_calendars`
- Optional local CSV cache (`Local=True`, files in `CSV/`)

Legacy Interactive Brokers code exists but is commented out in `getdata.py`. `IBConnect.py` imports `ib_insync` but the active `main()` path uses Yahoo Finance only.

## Project layout

```
TradingStrategy/
├── config.py           # Strategy parameters
├── getdata.py          # Data ingestion
├── indicators.py       # Indicators + signal definitions
├── backtest.py         # Strategy simulation + backtest sweeps
├── signal_check.py     # Streamlit daily signal dashboard
├── quote.py            # Quick indicator snapshot
├── IBConnect.py        # Single-strategy backtest + stats
├── test_data.py        # All signals × one symbol
├── test_indicator.py   # One signal × many symbols
├── requirements.txt    # Python dependencies
├── CSV/                # Backtest output (gitignored)
└── yfinance_update_summary.md  # Notes on yfinance column changes
```

## Tips

- **Change the active signal** in `IBConnect.py` (`buy_signal = ind.buy_signalN`) or `test_indicator.py` before backtesting.
- **Change the symbol list** in `signal_check.py` or `test_indicator.py` to match your watchlist.
- **Symbol filtering:** Each signal's `allowed_symbols` list controls which tickers it runs on during the daily scan. Set `ignore=False` logic by adding your symbol to that list.
- **Leverage:** Returns are multiplied by `Leverage` (default 3×) in `long_strat` and `%Change` calculation — adjust in `config.py` for realistic sizing.
- **CSV output:** Backtest detail files land in `CSV/` (gitignored). Create the folder if it doesn't exist.

## Disclaimer

This software is for **research and education only**. It generates signals and simulates historical performance — it does not connect to a broker for live order execution (except optional legacy IB code that is currently disabled). Past backtest performance does not guarantee future results. Use at your own risk.
