# file: api_main.py

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional
import uvicorn

from config import Config
from main import pipeline_runner
from boxplot import generate_boxplot_data
from pipeline.data_processing import calculate_returns, load_data, load_symbols
from utils.date_utils import calculate_start_end_dates


app = FastAPI(
    title="Argo API",
    description="portfolio optimization and market data analytics",
    version="1.0.0",
)


class PipelineOptions(BaseModel):
    min_weight: Optional[float] = Field(
        default=0.01, description="Minimum weight for inclusion in final weights"
    )
    portfolio_max_size: Optional[int] = Field(
        default=20, description="Maximum number of assets in the portfolio"
    )
    use_anomaly_filter: Optional[bool] = Field(
        default=False, description="Whether to filter assets with anomalous returns"
    )
    use_decorrelation: Optional[bool] = Field(
        default=False, description="Whether to filter correlated assets"
    )
    # TO-DO: Add additional options descriptions


class PipelineRequest(BaseModel):
    symbols: Optional[list[str]] = Field(
        None, description="List of ticker symbols (e.g., AAPL, MSFT, TSLA, etc.)"
    )
    options: Optional[PipelineOptions] = Field(
        None,
        description="Dynamic API options that override default configuration parameters",
    )

    class Config:
        schema_extra = {
            "example": {
                "symbols": [
                    "AAPL",
                    "AMZN",
                    "MSFT",
                    "TSLA",
                    "NVDA",
                    "GOOG",
                    "META",
                    "NFLX",
                    "SPY",
                    "TLT",
                    "GLD",
                ],
                "options": {
                    "min_weight": 0.01,
                    "portfolio_max_size": 20,
                    "use_anomaly_filter": False,
                    "use_decorrelation": False,
                },
            }
        }


@app.post("/inference")
def inference(req: PipelineRequest):
    """
    Run the pipeline, optionally overriding configuration parameters.
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
            status_code=500, detail=f"Error loading configuration: {str(e)}"
        )

    # Update options with any API-provided overrides.
    if req.options:
        config_obj.update_options(req.options.model_dump())

    try:
        pipeline_args = req.model_dump()
        pipeline_args["config"] = config_obj
        result = pipeline_runner(**pipeline_args)
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
    period: Optional[float] = Field(
        default=1.0, ge=1.0, description="Period in years, default is 1.0"
    )

    @field_validator("period")
    def check_period(cls, v):
        if v < 1.0:
            raise ValueError("Period must be at least 1 year")
        return v

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

    return {"status": "success", "data": boxplot_stats}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
