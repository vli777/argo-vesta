# file: api_main.py

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from config import Config
from main import iterative_pipeline_runner
from boxplot import generate_boxplot_data
from src.pipeline.data_processing import calculate_returns, load_data, load_symbols
from src.utils.date_utils import calculate_start_end_dates


app = FastAPI(
    title="Argo API",
    description="portfolio optimization and market data analytics",
    version="1.0.0",
)


class PipelineRequest(BaseModel):
    symbols: Optional[List[str]] = None
    max_epochs: Optional[int] = 1
    min_weight: Optional[float] = None
    portfolio_max_size: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "symbols": ["AAPL", "MSFT", "TSLA", "SPY", "TLT", "GLD"],
                "max_epochs": 15,
                "min_weight": 0.02,
                "portfolio_max_size": 20,
            }
        }


@app.post("/inference")
def inference(req: PipelineRequest):
    """
    Run the pipeline, optionally overriding config parameters.
    """
    # Load the default configuration
    default_config_path = "config.yaml"
    try:
        config_obj = Config.from_yaml(default_config_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Default configuration file '{default_config_path}' not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading default configuration: {str(e)}"
        )

    try:
        # Prepare arguments for the pipeline runner
        pipeline_args = req.dict()
        pipeline_args["config"] = config_obj

        # Run the pipeline
        result = iterative_pipeline_runner(**pipeline_args)

    except TypeError as te:
        raise HTTPException(
            status_code=400, detail=f"Invalid parameter type: {str(te)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Pipeline execution failed: {str(e)}"
        )

    return {"status": "success", "result": result}


class BoxplotRequest(BaseModel):
    symbols: Optional[List[str]] = None
    period: Optional[float] = 1.0  # Defaults to 1 year

    class Config:
        schema_extra = {
            "example": {
                "symbols": ["AAPL", "MSFT", "TSLA", "SPY", "TLT", "GLD"],
                "period": 1.0,
            }
        }


@app.post("/daily-statistics")
def daily_statistics(req: BoxplotRequest):
    """
    Generate boxplot data for the requested symbols and time period.
    """
    default_config_path = "config.yaml"
    try:
        config_obj = Config.from_yaml(default_config_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Default configuration file '{default_config_path}' not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading default configuration: {str(e)}"
        )

    try:
        all_symbols = load_symbols(config_obj, symbols_override=req.symbols)
        start, end = calculate_start_end_dates(
            float(req.period) if isinstance(req.period, (str, float)) else 1.0
        )
        df_all = load_data(all_symbols, start, end, config=config_obj)
        returns_df = calculate_returns(df_all)

        boxplot_stats = generate_boxplot_data(returns_df)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Boxplot generation failed: {str(e)}"
        )

    return {"status": "success", "boxplot_data": boxplot_stats}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
