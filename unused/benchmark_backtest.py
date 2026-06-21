import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

import warn_config  # noqa: F401
import indicators as ind
from backtest_runners import load_ticker_data
from indicator_sweep import indicator_tryout


BENCHMARK_DIR = Path("data/benchmarks")
REGRESSION_DIR = Path("data/regression_baselines")


def _load_data(symbol, years, use_regression_snapshot):
    if use_regression_snapshot:
        snapshot = REGRESSION_DIR / f"{symbol}_{years}y_input.csv"
        if not snapshot.exists():
            raise FileNotFoundError(f"No regression snapshot found at {snapshot}")
        return pd.read_csv(snapshot, parse_dates=["Date"])
    return load_ticker_data(symbol, years=years)


def run_benchmark(symbol, signal_name, years, check_breadth, check_both, use_regression_snapshot, timing):
    signal = getattr(ind, signal_name)
    data = _load_data(symbol, years, use_regression_snapshot)
    data["Buy"], data["Sell"], days, profit, description, _, is_long, _ = signal(data, symbol)

    started = perf_counter()
    results = indicator_tryout(
        data,
        days,
        profit,
        is_long,
        check_breadth=check_breadth,
        check_both=check_both,
        verbose=False,
        timing=timing,
    )
    elapsed = perf_counter() - started

    return {
        "symbol": symbol,
        "signal": signal_name,
        "description": description,
        "years": years,
        "check_breadth": check_breadth,
        "check_both": check_both,
        "use_regression_snapshot": use_regression_snapshot,
        "rows": int(results.shape[0]),
        "elapsed_seconds": round(elapsed, 3),
        "top_results": results.head(5).drop(columns=["Yearly"], errors="ignore").to_dict(orient="records"),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark indicator sweep runtime")
    parser.add_argument("--symbol", default="SOXX")
    parser.add_argument("--signal", default="buy_signal7")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--price-only", action="store_true", help="Skip breadth sweeps")
    parser.add_argument("--breadth-only", action="store_true", help="Skip price/VFI sweeps")
    parser.add_argument("--use-regression-snapshot", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--output", default=str(BENCHMARK_DIR / "latest_indicator_sweep.json"))
    args = parser.parse_args()

    if args.price_only and args.breadth_only:
        parser.error("--price-only and --breadth-only are mutually exclusive")

    payload = run_benchmark(
        symbol=args.symbol,
        signal_name=args.signal,
        years=args.years,
        check_breadth=not args.price_only,
        check_both=not args.breadth_only,
        use_regression_snapshot=args.use_regression_snapshot,
        timing=args.timing,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
