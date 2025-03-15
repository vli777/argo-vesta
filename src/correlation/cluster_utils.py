import pandas as pd
from utils import logger


def get_clusters_top_performers(clusters: dict, perf_series: pd.Series) -> list[str]:
    selected_tickers: list[str] = []
    for label, tickers in clusters.items():
        # For noise (label == -1), include all tickers
        if label == -1:
            selected_tickers.extend(tickers)
        else:
            group_perf = perf_series[tickers].sort_values(ascending=False)
            if len(tickers) < 10:
                top_n = len(tickers)
            elif len(tickers) < 20:
                top_n = max(1, int(0.50 * len(tickers)))
            else:
                top_n = max(1, int(0.33 * len(tickers)))
            top_candidates = group_perf.index.tolist()[:top_n]
            selected_tickers.extend(top_candidates)
            logger.info(
                f"Cluster {label}: {len(tickers)} assets; keeping {top_candidates}"
            )
    return selected_tickers
