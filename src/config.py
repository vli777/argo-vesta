# src/config.py


import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Config:
    # Core (non-API) parameters
    data_dir: str
    input_files_dir: str
    input_files: List[str]

    download: bool
    min_weight: float
    max_weight: float
    portfolio_max_size: Optional[int]
    portfolio_max_vol: Optional[float]
    portfolio_max_cvar: Optional[float]
    portfolio_risk_priority: str
    risk_free_rate: float
    allow_short: bool
    max_gross_exposure: float

    plot_daily_returns: bool
    plot_cumulative_returns: bool
    plot_contribution: bool
    plot_anomalies: bool
    plot_clustering: bool
    plot_reversion: bool
    plot_optimization: bool

    use_anomaly_filter: bool
    use_decorrelation: bool
    use_reversion: bool
    reversion_type: Optional[str]

    optimization_objective: Optional[str]
    use_global_optimization: bool
    global_optimization_type: Optional[str]

    test_mode: bool
    test_data_visible_pct: float

    models: Dict[str, List[str]] = field(
        default_factory=lambda: {"1.00": ["nested_clustering"]}
    )
    # API options override
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_file: str) -> "Config":
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        # Centralize all default values here.
        defaults = {
            "download": False,
            "min_weight": 0.01,
            "max_weight": 1.0,
            "portfolio_max_size": None,
            "portfolio_max_vol": None,
            "portfolio_max_cvar": None,
            "portfolio_risk_priority": "both",
            "risk_free_rate": 0.0,
            "allow_short": False,
            "max_gross_exposure": 1.3,
            "plot_daily_returns": False,
            "plot_cumulative_returns": False,
            "plot_contribution": False,
            "plot_anomalies": False,
            "plot_clustering": False,
            "plot_reversion": False,
            "plot_optimization": False,
            "use_anomaly_filter": False,
            "use_decorrelation": False,
            "use_reversion": False,
            "reversion_type": None,  # If use_reversion is enabled, default later to "z"
            "optimization_objective": "sharpe",
            "use_global_optimization": False,
            "global_optimization_type": None,
            "test_mode": False,
            "test_data_visible_pct": 0.1,
            "models": {"1.00": ["nested_clustering"]},
            "options": {},
        }
        # Merge defaults with the loaded config (the YAML values override the defaults)
        merged = {**defaults, **config_dict}

        # Required keys check
        if "data_dir" not in merged:
            raise ValueError("data_dir must be specified in the configuration file")
        if "input_files" not in merged:
            raise ValueError("input_files must be specified in the configuration file")

        # Ensure input_files_dir is set
        merged["input_files_dir"] = merged.get("input_files_dir", "watchlists")

        # Create directories if needed
        os.makedirs(merged["data_dir"], exist_ok=True)
        os.makedirs(merged["input_files_dir"], exist_ok=True)

        # Set reversion_type default if use_reversion is enabled
        if merged.get("use_reversion") and merged.get("reversion_type") is None:
            merged["reversion_type"] = "z"

        return cls(**merged)
