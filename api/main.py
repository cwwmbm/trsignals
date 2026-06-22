import warn_config  # noqa: F401
import logging
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import config
from api.schemas import (
    BuilderBacktestRequest,
    BuilderRefineRequest,
    CustomDatasetResponse,
    HoldDaysSweepRequest,
    IndicatorSweepRequest,
    SaveStrategyRequest,
    SignalComboSweepRequest,
    SingleBacktestRequest,
    SymbolConfirmDetailRequest,
    SymbolConfirmSweepRequest,
    PortfolioSimulateRequest,
    UpdateStrategyRequest,
)
from api.services import (
    delete_custom_dataset,
    delete_saved_strategy,
    get_custom_dataset_metadata,
    run_builder_backtest,
    run_builder_refine,
    run_hold_days_sweep,
    run_indicator_sweep,
    run_live_scan,
    run_portfolio_simulation,
    run_signal_combo_sweep,
    run_single_backtest,
    run_symbol_confirm_detail,
    run_symbol_confirm_sweep,
    list_saved_strategies,
    save_strategy,
    update_saved_strategy,
    upload_custom_dataset,
)
from api.indicator_catalog import list_indicators
from api.signal_registry import list_signals


app = FastAPI(title="TradingStrategy API")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/signals")
def signals() -> list[dict]:
    return list_signals()


@app.get("/indicators")
def indicators(builder_only: bool = True) -> list[dict]:
    return list_indicators(builder_only=builder_only)


@app.get("/config")
def read_config() -> dict:
    return {
        "ticker": config.ticker,
        "leverage": config.Leverage,
        "exclude_best_return_year": config.ExcludeBestReturnYear,
        "rsi2_buy": config.RSI2Buy,
        "rsi5_buy": config.RSI5Buy,
        "rsi2_sell": config.RSI2Sell,
        "rsi5_sell": config.RSI5Sell,
        "monday_buy": config.MondayBuy,
        "low_volume_buy": config.LowVolumeBuy,
    }


def _handle_errors(fn, request):
    try:
        return fn(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/backtests/single")
def single_backtest(request: SingleBacktestRequest) -> dict:
    return _handle_errors(run_single_backtest, request)


@app.post("/backtests/signal-combo-sweep")
def signal_combo_sweep(request: SignalComboSweepRequest) -> list[dict]:
    return _handle_errors(run_signal_combo_sweep, request)


@app.post("/backtests/symbol-confirm-sweep")
def symbol_confirm_sweep(request: SymbolConfirmSweepRequest) -> list[dict]:
    return _handle_errors(run_symbol_confirm_sweep, request)


@app.post("/backtests/symbol-confirm-detail")
def symbol_confirm_detail(request: SymbolConfirmDetailRequest) -> dict:
    return _handle_errors(run_symbol_confirm_detail, request)


@app.post("/backtests/hold-days-sweep")
def hold_days_sweep(request: HoldDaysSweepRequest) -> list[dict]:
    return _handle_errors(run_hold_days_sweep, request)


@app.post("/backtests/indicator-sweep")
def indicator_sweep(request: IndicatorSweepRequest) -> list[dict]:
    return _handle_errors(run_indicator_sweep, request)


@app.post("/backtests/builder")
def builder_backtest(request: BuilderBacktestRequest) -> dict:
    return _handle_errors(run_builder_backtest, request)


@app.post("/datasets/custom", response_model=CustomDatasetResponse)
async def upload_custom_data(file: UploadFile = File(...)) -> dict:
    try:
        logger.info("Custom dataset upload started: %s", file.filename)
        content = await file.read()
        result = upload_custom_dataset(content, filename=file.filename)
        logger.info(
            "Custom dataset upload finished: %s (%s rows, %s)",
            result.get("symbol"),
            result.get("row_count"),
            result.get("interval_label"),
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/datasets/custom/{dataset_id}", response_model=CustomDatasetResponse)
def get_custom_data(dataset_id: str) -> dict:
    try:
        return get_custom_dataset_metadata(dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/datasets/custom/{dataset_id}")
def remove_custom_data(dataset_id: str) -> dict:
    try:
        return delete_custom_dataset(dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/backtests/builder/refine")
def builder_refine(request: BuilderRefineRequest) -> dict | list[dict]:
    return _handle_errors(run_builder_refine, request)


@app.post("/strategies")
def create_saved_strategy(request: SaveStrategyRequest) -> dict:
    return _handle_errors(save_strategy, request)


@app.patch("/strategies/{strategy_id}")
def update_strategy_description(strategy_id: str, request: UpdateStrategyRequest) -> dict:
    try:
        return update_saved_strategy(strategy_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/strategies/{strategy_id}")
def remove_saved_strategy(strategy_id: str) -> dict:
    try:
        return delete_saved_strategy(strategy_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/strategies")
def get_saved_strategies() -> list[dict]:
    return list_saved_strategies()


@app.get("/scan")
def scan() -> list[dict]:
    try:
        return run_live_scan()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/portfolios/simulate")
def portfolio_simulate(request: PortfolioSimulateRequest) -> dict:
    return _handle_errors(run_portfolio_simulation, request)
