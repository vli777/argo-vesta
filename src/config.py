# src/config.py

import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os


@dataclass
class Config:
    data_dir: str
    input_files_dir: str
    input_files: List[str]

    download: bool
    min_weight: float
    max_weight: float
    portfolio_max_size: int
    portfolio_max_vol: float
    portfolio_max_cvar: float
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
    reversion_type: Optional[str]  # Can be "ou", "z", or None

    optimization_objective: Optional[str]
    use_global_optimization: bool
    global_optimization_type: Optional[str]  # Can be "annealing", "diffusion", or None

    test_mode: bool
    test_data_visible_pct: float
    models: Dict[str, List[str]] = field(
        default_factory=lambda: {"1.00": ["nested_clustering"]}
    )

    @classmethod
    def from_yaml(cls, config_file: str) -> "Config":
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        # Ensure default values for missing or empty keys
        config_dict["models"] = config_dict.get(
            "models", {"1.00": ["nested_clustering"]}
        )

        data_dir = config_dict["data_dir"]
        input_files_dir = config_dict.get("input_files_dir", "watchlists")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(input_files_dir, exist_ok=True)

        use_reversion = config_dict.get("use_reversion", False)
        reversion_type = (
            config_dict.get("reversion_type", "ou") if use_reversion else None
        )

        return cls(
            data_dir=data_dir,
            input_files_dir=input_files_dir,
            input_files=config_dict["input_files"],
            models=config_dict["models"],
            download=config_dict.get("download", False),
            min_weight=config_dict.get("min_weight", 0.01),
            max_weight=config_dict.get("max_weight", 1.0),
            portfolio_max_size=config_dict.get("portfolio_max_size"),
            portfolio_max_vol=config_dict.get("portfolio_max_vol"),
            portfolio_max_cvar=config_dict.get("portfolio_max_cvar"),
            portfolio_risk_priority=config_dict.get("portfolio_risk_priority", "both"),
            risk_free_rate=config_dict.get("risk_free_rate", 0.0),
            allow_short=config_dict.get("allow_short", False),
            max_gross_exposure=config_dict.get("max_gross_exposure", 1.3),
            plot_daily_returns=config_dict.get("plot_daily_returns", False),
            plot_cumulative_returns=config_dict.get("plot_cumulative_returns", False),
            plot_contribution=config_dict.get("plot_contribution", False),
            plot_clustering=config_dict.get("plot_clustering", False),
            plot_anomalies=config_dict.get("plot_anomalies", False),
            plot_reversion=config_dict.get("plot_reversion", False),
            plot_optimization=config_dict.get("plot_optimization", False),
            use_anomaly_filter=config_dict.get("use_anomaly_filter", False),
            use_decorrelation=config_dict.get("use_decorrelation", False),
            use_reversion=use_reversion,
            reversion_type=reversion_type,  # Defaults to "ou" if enabled, else None
            optimization_objective=config_dict.get("optimization_objective")
            or "sharpe",
            use_global_optimization=config_dict.get("use_global_optimization", False),
            global_optimization_type=config_dict.get("global_optimization_type"),
            test_mode=config_dict.get("test_mode", False),
            test_data_visible_pct=config_dict.get("test_data_visible_pct", 0.1),
        )
