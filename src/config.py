# src/config.py


import os
import yaml
from dataclasses import asdict, dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Config:
    # Core (non-overridable) parameters:
    data_dir: str
    input_files_dir: str
    input_files: List[str]
    models: Dict[str, List[str]]
    download: bool
    test_mode: bool
    test_data_visible_pct: float

    # Overridable (options) parameters:
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
    use_regime_detection: bool
    clustering_type: Optional[str]
    top_n_performers: Optional[int]
    use_reversion: bool
    reversion_type: Optional[str]
    optimization_objective: Optional[str]
    use_global_optimization: bool
    global_optimization_type: Optional[str]

    options: Dict[str, Any] = field(init=False)
    _overridable_keys: set = field(init=False, repr=False)
    _core_keys: set = field(init=False, repr=False)

    def __post_init__(self):
        # Initialize options so that asdict() can find it.
        self.options = {}

    @classmethod
    def from_yaml(cls, config_file: str) -> "Config":
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        # Defaults for all parameters.
        defaults = {
            "download": False,
            "input_files": [],
            "min_weight": 0.0,
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
            "use_regime_detection": False,
            "clustering_type": "spectral",
            "top_n_performers": None,
            "use_reversion": False,
            "reversion_type": None,
            "optimization_objective": "sharpe",
            "use_global_optimization": False,
            "global_optimization_type": None,
            "test_mode": False,
            "test_data_visible_pct": 0.1,
            "models": {"1.00": ["nested_clustering"]},
        }

        # Split the YAML into two parts:
        # - Anything at the top level (except "options") is core.
        # - The "options" group contains keys that can be overridden.
        core_yaml = {
            k: v for k, v in config_dict.items() if k != "options" and v is not None
        }
        options_yaml = {
            k: v for k, v in config_dict.get("options", {}).items() if v is not None
        }

        # Build the final configuration:
        # Start with defaults, then let core YAML override defaults for core keys,
        # and finally let the options YAML override defaults for overridable keys.
        final_config = {**defaults, **core_yaml, **options_yaml}

        # Instantiate the Config object.
        instance = cls(**final_config)

        # Ensure required core fields are present.
        if "data_dir" not in final_config:
            raise ValueError("data_dir must be specified in the configuration file")

        # Set a default for input_files_dir if not provided.
        instance.input_files_dir = final_config.get("input_files_dir", "watchlists")
        os.makedirs(instance.data_dir, exist_ok=True)
        os.makedirs(instance.input_files_dir, exist_ok=True)

        # Set default reversion_type if used.
        if instance.use_reversion and instance.reversion_type is None:
            instance.reversion_type = "z"

        # Set default global_optimization_type if used.
        if (
            instance.use_global_optimization
            and instance.global_optimization_type is None
        ):
            instance.global_optimization_type = "diffusion"

        # Record which keys are overridable (from the YAML's options block).
        instance._overridable_keys = set(options_yaml.keys())
        # All other keys in the final configuration are considered core.
        instance._core_keys = set(final_config.keys()) - instance._overridable_keys

        # Build the options dictionary from only the overridable parameters.
        instance.options = {
            k: v for k, v in asdict(instance).items() if k in instance._overridable_keys
        }
        return instance

    def update_options(self, overrides: Dict[str, Any]) -> None:
        """
        Update only the overridable parameters (those originally specified in YAML's "options" group).
        """
        for key, value in overrides.items():
            if key in self._overridable_keys:
                setattr(self, key, value)
        # Refresh the options dictionary.
        self.options = {
            k: v for k, v in asdict(self).items() if k in self._overridable_keys
        }

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)
