import os
import sys
import requests
import pandas as pd
from pathlib import Path
import time
from dotenv import load_dotenv
from utils.logger import logger

load_dotenv()


API_KEY = os.getenv("POLYGON_API_KEY")


API_CALL_COUNT = 0  # Global counter for API calls


def polygon_get(url, max_retries=3):
    """
    A wrapper around requests.get that enforces a delay between calls and
    handles 429 rate-limit responses by waiting and retrying.

    Args:
        url (str): The URL to request.
        max_retries (int): Maximum number of retries for a 429 response.

    Returns:
        requests.Response: The HTTP response.
    """
    retries = 0
    while True:
        response = requests.get(url)
        # If we get a rate limit error, wait and retry
        if response.status_code == 429:
            retries += 1
            logger.warning(
                f"Rate limit exceeded for URL {url}. "
                f"Sleeping for 60 seconds before retrying ({retries}/{max_retries})."
            )
            time.sleep(60)
            if retries >= max_retries:
                logger.error("Maximum retries reached. Returning the last response.")
                return response
            continue

        # Enforce a delay between all calls (one call every 12 seconds)
        time.sleep(12)
        return response


def download_polygon_data(ticker, start_date, end_date):
    """
    Downloads daily aggregated data for a single ticker from Polygon,
    handling pagination if needed, and renames the columns to match the
    existing schema.

    Args:
        ticker (str): Ticker symbol.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        pd.DataFrame or None: DataFrame containing the data with renamed columns,
                              or None if an error occurred or no data was returned.
    """
    base_url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={API_KEY}"
    )

    url = base_url

    response = polygon_get(url)
    if response.status_code != 200:
        logger.error(
            f"Failed to download data for {ticker}. Status code: {response.status_code}. "
            f"Response: {response.text}"
        )
        return None

    data = response.json()
    if data.get("status") != "OK":
        logger.error(f"API returned error for {ticker}: {data}")
        return None

    results = data.get("results", [])

    if not results:
        logger.warning(
            f"No data returned for {ticker} between {start_date} and {end_date}"
        )
        return None

    # Create DataFrame from results and rename columns to match the existing schema
    df = pd.DataFrame(results)
    rename_map = {
        "t": "Date",
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume",
        "vw": "VWAP",
    }
    df.rename(columns=rename_map, inplace=True)

    # Copy Close to Adj Close since the data is already adjusted
    df["Adj Close"] = df["Close"]

    # Drop unnecessary columns (like 'n')
    if "n" in df.columns:
        df.drop(columns=["n"], inplace=True)

    # Order columns; include 'vw' if available
    cols = ["Adj Close", "Close", "High", "Low", "Open", "Volume", "Date", "VWAP"]
    df = df[cols]

    return df


def download_and_save_ticker_data(
    tickers, start_date, end_date, data_path, force_download=False
):
    """
    Loops through a list of tickers, downloads their daily data from Polygon,
    and saves each as a Parquet file.

    Args:
        tickers (list[str]): List of ticker symbols.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        data_path (str): Directory path to save Parquet files.
        force_download (bool): If True, re-download data even if a file already exists.
    """
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        file_path = data_path / f"{ticker}.parquet"
        if file_path.exists() and not force_download:
            logger.info(
                f"Data for {ticker} already exists at {file_path}. Skipping download."
            )
            continue

        logger.info(f"Downloading data for {ticker}...")
        df = download_polygon_data(ticker, start_date, end_date)
        if df is None:
            logger.warning(f"No data downloaded for {ticker}.")
            continue

        try:
            df.to_parquet(file_path)
            logger.info(f"Saved data for {ticker} to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data for {ticker}: {e}")
