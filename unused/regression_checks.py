import _bootstrap  # noqa: F401
import argparse
import json
import math
from pathlib import Path
from typing import Any

import warn_config  # noqa: F401
import backtest as bt
import indicators as ind
from api.serializers import dataframe_records, detailed_backtest_payload
from backtest_runners import load_ticker_data


BASELINE_DIR = Path("data/regression_baselines")
BASELINE_FILE = BASELINE_DIR / "backtest_regression.json"
YEARS = 5
FLOAT_TOLERANCE = 1e-4
USE_SNAPSHOTS = False

COMPARE_COLUMNS = [
    "Buy",
    "Sell",
    "LongTradeIn",
    "LongTradeOut",
    "HoldLong",
    "DaysInTrade",
    "ProfitableCloses",
    "TradeEntry",
    "TradePnL",
    "RollingPnL",
    "Drawdown",
]


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (bool, int, str)):
        return value
    return value


def _series_payload(data, column):
    if column not in data.columns:
        return []
    return [_clean_value(value) for value in data[column].tolist()]


def _snapshot_path(symbol):
    return BASELINE_DIR / f"{symbol}_{YEARS}y_input.csv"


def _load_data(symbol, years=YEARS):
    path = _snapshot_path(symbol)
    if path.exists():
        import pandas as pd

        return pd.read_csv(path, parse_dates=["Date"])

    data = load_ticker_data(symbol, years=years)
    if not USE_SNAPSHOTS:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
    return data.copy()


def _strategy_payload(name, data, days, profit, description):
    result = detailed_backtest_payload(data, days, profit, description)
    result["columns"] = {column: _series_payload(data, column) for column in COMPARE_COLUMNS}
    result["last_date"] = str(data["Date"].iloc[-1].date())
    return {"name": name, "result": result}


def _run_signal_case(name, signal, symbol, years=YEARS):
    data = _load_data(symbol, years=years)
    data["Buy"], data["Sell"], days, profit, description, _, is_long, _ = signal(data, symbol)
    data = bt.execute_strategy(data, days, profit, is_long)
    return _strategy_payload(name, data, days, profit, description)


def _run_with_runtime(name, signal, symbol, monday_buy, low_volume_buy, years=YEARS):
    original = (bt.MondayBuy, bt.LowVolumeBuy)
    bt.MondayBuy = monday_buy
    bt.LowVolumeBuy = low_volume_buy
    try:
        return _run_signal_case(name, signal, symbol, years)
    finally:
        bt.MondayBuy, bt.LowVolumeBuy = original


def _run_cross_symbol_case(years=YEARS):
    signal = ind.combined_signal(ind.buy_signal16, ind.buy_signal7, "or")
    symbol_data = {symbol: _load_data(symbol, years=years) for symbol in ["SOXX", "SMH", "QQQ"]}
    data, days, profit, description, _ = bt.backtest_cross_symbol(
        signal,
        primary_symbol="SOXX",
        confirm_symbols=["SMH", "QQQ"],
        years=years,
        symbol_data=symbol_data,
    )
    return _strategy_payload("cross_symbol_soxx_smh_qqq", data, days, profit, description)


def _run_indicator_sample(years=YEARS):
    symbol = "SOXX"
    signal = ind.combined_signal(ind.buy_signal16, ind.buy_signal7, "or")
    data = _load_data(symbol, years=years)
    data["Buy"], data["Sell"], days, profit, _, _, is_long, _ = signal(data, symbol)
    results = bt.backtest_ind(data, days, profit, is_long, "IBR", "both", 0.1, 0.2, 0.1)
    return {"name": "indicator_sample_ibr", "result": dataframe_records(results)}


def run_cases():
    return {
        "years": YEARS,
        "cases": [
            _run_signal_case("buy_signal7_soxx", ind.buy_signal7, "SOXX"),
            _run_signal_case("buy_signal7_smh", ind.buy_signal7, "SMH"),
            _run_signal_case("buy_signal7_spy", ind.buy_signal7, "SPY"),
            _run_signal_case(
                "combined_signal16_or_signal7_soxx",
                ind.combined_signal(ind.buy_signal16, ind.buy_signal7, "or"),
                "SOXX",
            ),
            _run_with_runtime("og_buy_signal_spy_monday_off_lowvol_off", ind.og_buy_signal, "SPY", False, False),
            _run_with_runtime("og_new_buy_signal_spy_monday_on_lowvol_on", ind.og_new_buy_signal, "SPY", True, True),
            _run_cross_symbol_case(),
            _run_indicator_sample(),
        ],
    }


def _compare_values(path, expected, actual, failures):
    if isinstance(expected, float) or isinstance(actual, float):
        if expected is None or actual is None:
            if expected != actual:
                failures.append(f"{path}: expected {expected!r}, got {actual!r}")
            return
        if not math.isclose(float(expected), float(actual), rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE):
            failures.append(f"{path}: expected {expected!r}, got {actual!r}")
        return

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            failures.append(f"{path}: expected dict, got {type(actual).__name__}")
            return
        if set(expected) != set(actual):
            failures.append(f"{path}: keys differ expected={sorted(expected)} actual={sorted(actual)}")
            return
        for key in expected:
            _compare_values(f"{path}.{key}", expected[key], actual[key], failures)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            failures.append(f"{path}: expected list, got {type(actual).__name__}")
            return
        if len(expected) != len(actual):
            failures.append(f"{path}: length differs expected={len(expected)} actual={len(actual)}")
            return
        for idx, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _compare_values(f"{path}[{idx}]", expected_item, actual_item, failures)
        return

    if expected != actual:
        failures.append(f"{path}: expected {expected!r}, got {actual!r}")


def write_baseline():
    global USE_SNAPSHOTS
    USE_SNAPSHOTS = False
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for path in BASELINE_DIR.glob(f"*_{YEARS}y_input.csv"):
        path.unlink()
    payload = run_cases()
    BASELINE_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote baseline to {BASELINE_FILE}")


def compare_baseline():
    global USE_SNAPSHOTS
    USE_SNAPSHOTS = True
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(f"No baseline found at {BASELINE_FILE}. Run --write-baseline first.")
    expected = json.loads(BASELINE_FILE.read_text())
    actual = run_cases()
    failures = []
    _compare_values("baseline", expected, actual, failures)
    if failures:
        print("Regression comparison failed:")
        for failure in failures[:50]:
            print(f" - {failure}")
        if len(failures) > 50:
            print(f" ... {len(failures) - 50} more failures")
        raise SystemExit(1)
    print("Regression comparison passed.")


def main():
    parser = argparse.ArgumentParser(description="Backtest regression baseline checks")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.write_baseline == args.compare:
        parser.error("Pass exactly one of --write-baseline or --compare")

    if args.write_baseline:
        write_baseline()
    else:
        compare_baseline()


if __name__ == "__main__":
    main()
