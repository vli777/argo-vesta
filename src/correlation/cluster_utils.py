from typing import Optional
import pandas as pd
from utils import logger


def get_clusters_top_performers(
    clusters: dict, perf_series: pd.Series, top_n: Optional[int] = None
) -> list[str]:
    selected_tickers: list[str] = []
    for label, tickers in clusters.items():
        # For noise (label == -1), include all tickers.
        if label == -1:
            selected_tickers.extend(tickers)
        else:
            group_perf = perf_series[tickers].sort_values(ascending=False)
            if top_n is not None and top_n >= 1:
                # Use the explicit number, but not more than available tickers.
                top_candidates = group_perf.index.tolist()[: min(top_n, len(tickers))]
            else:
                # Default behavior based on cluster size.
                if len(tickers) < 10:
                    n_top = len(tickers)
                elif len(tickers) < 20:
                    n_top = max(1, int(0.50 * len(tickers)))
                else:
                    n_top = max(1, int(0.33 * len(tickers)))
                top_candidates = group_perf.index.tolist()[:n_top]
            selected_tickers.extend(top_candidates)
            logger.info(
                f"Cluster {label}: {len(tickers)} assets; keeping {top_candidates}"
            )
    return selected_tickers
