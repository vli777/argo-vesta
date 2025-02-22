import time
from typing import Optional
import unicodedata
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import re
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

from charles_schwab_dl import download_schwab_data
from utils.date_utils import get_last_date
from .logger import logger


def is_valid_ticker(symbol, retries=3, delay=20):
    """
    Checks if a ticker is valid using yfinance.
    Retries if transient errors occur (such as rate limits or unauthorized responses).
    """
    for attempt in range(1, retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            # If info is not empty, assume the ticker is valid.
            if info:
                return True
            else:
                logger.error(f"No info returned for {symbol}.")
                return False
        except Exception as e:
            error_msg = str(e)
            # If the error seems transient, retry.
            if "Too Many Requests" in error_msg or "Unauthorized" in error_msg:
                logger.warning(
                    f"Attempt {attempt}/{retries} for {symbol} failed with error: {error_msg}"
                )
                time.sleep(delay)
            else:
                logger.error(f"Error validating ticker {symbol}: {error_msg}")
                return False
    # If all attempts fail, consider the ticker invalid.
    return False


def clean_ticker(ticker):
    """
    Normalize the ticker string by removing special characters and converting to uppercase.
    """
    # Normalize to NFKD, then encode to ASCII (ignoring errors) and decode back to a string.
    normalized = unicodedata.normalize("NFKD", ticker)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    # Remove any non-alphabetical characters.
    cleaned = re.sub(r"[^A-Za-z]", "", ascii_text)
    return cleaned.upper()


def process_input_files(input_file_paths):
    """
    Reads CSV files, cleans ticker symbols by removing special characters and encoding issues,
    writes the cleaned tickers back to the CSV, and extracts valid ticker symbols.
    A valid ticker consists solely of uppercase alphabetical characters and passes the `is_valid_ticker` check.
    """
    symbols = set()

    for file_path in input_file_paths:
        # Ensure the file has a .csv suffix.
        if file_path.suffix.lower() != ".csv":
            file_path = file_path.with_suffix(".csv")

        cleaned_lines = []
        try:
            with file_path.open("r", encoding="utf-8") as file:
                for line in file:
                    original = line.strip()
                    if not original:
                        continue
                    # Leave comments unchanged.
                    if original.startswith("#"):
                        cleaned_line = original
                    else:
                        cleaned_line = clean_ticker(original)
                    cleaned_lines.append(cleaned_line)

                    # Process non-comment lines.
                    if not cleaned_line.startswith("#") and cleaned_line:
                        if re.fullmatch(r"[A-Z]+", cleaned_line):
                            logger.debug(
                                f"Checking validity for ticker: {cleaned_line}"
                            )
                            # if is_valid_ticker(cleaned_line):
                            symbols.add(cleaned_line)
                            # else:
                            #     logger.warning(
                            #         f"Ignoring invalid ticker symbol: {cleaned_line}"
                            #     )
                        else:
                            logger.warning(
                                f"Ticker '{original}' cleaned to '{cleaned_line}' does not match pattern."
                            )
            # Overwrite the file with cleaned tickers.
            with file_path.open("w", encoding="utf-8") as file:
                for cl in cleaned_lines:
                    file.write(cl + "\n")
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")

        logger.info(f"Processed file: {file_path}")

    return sorted(symbols)


def get_stock_data(
    symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    download stock price data from yahoo finance with optional write to csv
    """
    logger.info(f"Downloading {symbol} {start_date} - {end_date} ...")
    # charles schwab api uses unix timestamps instead of pd.Timestamp
    start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)

    stock_data = download_schwab_data(
        symbol=symbol, start_date=start_timestamp, end_date=end_timestamp
    )
    symbol_df = convert_to_yfinance_format(data=stock_data, symbol=symbol)
    # symbol_df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)

    return symbol_df


def update_store(data_path, symbol, start_date, end_date):
    """
    Reads existing data for `symbol` from a Parquet file (if present).
    Fetches missing data from Yahoo Finance if required (both past and future data).
    Merges old and new data, ensuring completeness.
    Writes the combined dataset back to a Parquet file.
    Returns:
        pd.DataFrame: The updated DataFrame (in memory).
    """
    pq_filename = Path(data_path) / f"{symbol}.parquet"

    # Load existing data if available
    if pq_filename.is_file():
        old_df = pd.read_parquet(pq_filename)
        if not old_df.empty:
            earliest_stored_date = old_df.index.min().date()
            latest_stored_date = old_df.index.max().date()
        else:
            earliest_stored_date, latest_stored_date = None, None
    else:
        old_df = pd.DataFrame()
        earliest_stored_date, latest_stored_date = None, None

    # Determine missing historical data (prepend missing records)
    if earliest_stored_date is None or start_date < earliest_stored_date:
        logger.info(
            f"Fetching missing history for {symbol} from {start_date} to {earliest_stored_date}..."
        )
        new_history_df = yf.download(symbol, start=start_date, end=earliest_stored_date)
    else:
        new_history_df = pd.DataFrame()

    # Determine missing future data (append missing records)
    if latest_stored_date is None or end_date > latest_stored_date:
        logger.info(
            f"Fetching recent data for {symbol} from {latest_stored_date} to {end_date}..."
        )
        new_recent_df = yf.download(
            symbol, start=latest_stored_date + timedelta(days=1), end=end_date
        )
    else:
        new_recent_df = pd.DataFrame()

    # Ensure datetime index for merging
    for df in [new_history_df, new_recent_df]:
        if not df.empty and not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    # Merge all data, ensuring chronological order
    combined_df = (
        pd.concat([new_history_df, old_df, new_recent_df])
        .sort_index()
        .drop_duplicates()
    )

    # Save the updated dataset back to Parquet
    combined_df.to_parquet(pq_filename, index=True)
    return combined_df


def download_all_tickers(watchlist_files, data_path, years=5):
    """
    1) Collect tickers from watchlist CSV files.
    2) For each ticker, determine last date or fallback to X years ago.
    3) Download/append to a Parquet file via update_store_parquet().
    """
    symbols = process_input_files(watchlist_files)  # Same as your CSV-based code
    today = datetime.now().date()

    for symbol in symbols:
        pq_filename = Path(data_path) / f"{symbol}.parquet"

        # Check if parquet file exists
        if pq_filename.is_file():
            # Attempt to read last date from the existing Parquet
            last_date = get_last_date(pq_filename)
            if last_date:
                start_date = last_date + timedelta(days=1)
            else:
                # If file has no valid data
                start_date = today - relativedelta(years=years)
        else:
            # No existing file => fetch from X years ago
            start_date = today - relativedelta(years=years)

        # Only download if we actually need data
        if start_date <= today:
            update_store(data_path, symbol, start_date, today)
        else:
            logger.info(f"No update needed for {symbol} — up to date.")


def get_last_date_parquet(parquet_file: Path) -> Optional[date]:
    """
    Reads the Parquet file, inspects the last row, and returns that date as a `date` object.
    """
    try:
        df = pd.read_parquet(parquet_file)
        if not df.empty:
            return df.index.max().date()
        return None
    except Exception as e:
        logger.warning(f"Could not read Parquet file {parquet_file}: {e}")
        return None


def format_to_df_format(df, symbol):
    """
    Ensure the DataFrame has the necessary OHLC columns without renaming them to the ticker symbol.
    """
    # Verify that necessary columns exist
    required_columns = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    existing_columns = set(df.columns)

    if not required_columns.issubset(existing_columns):
        logger.warning(
            f"{symbol}: Missing required columns. Available columns: {existing_columns}"
        )
        # Handle missing columns as needed, e.g., fill with NaN or drop the symbol
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan

    return df


def flatten_columns(df, symbol):
    """
    Flatten the DataFrame's columns by removing MultiIndex or trailing symbol suffixes.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    else:
        # Remove trailing symbol from column names if present.
        df.columns = [col.replace(f" {symbol}", "") for col in df.columns]
    return df


def convert_all_csv_to_parquet(data_folder: str):
    """
    Convert all CSV files in `data_folder` to Parquet.
    Assumes the first column of each CSV is the date column and should be the index.
    """
    data_path = Path(data_folder)

    # Loop through every CSV file in data_path
    for csv_file in data_path.glob("*.csv"):
        logger.info(f"Converting {csv_file} to Parquet...")

        # Read CSV, parse first column as dates, set as index
        df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)

        # Optionally ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Construct the Parquet file path (same base name, different extension)
        parquet_file = csv_file.with_suffix(".parquet")

        # Write DataFrame to Parquet
        df.to_parquet(parquet_file)

        logger.info(f"Converted {csv_file.name} -> {parquet_file.name}")


def ensure_unique_timestamps(
    df: pd.DataFrame, symbol: str, keep: str = "first"
) -> pd.DataFrame:
    """
    Ensures the DataFrame has a unique datetime index.
    Logs a warning and removes duplicates if any are found.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.
        symbol (str): Ticker symbol for logging context.
        keep (str): 'first' or 'last' row to keep if duplicates exist.

    Returns:
        pd.DataFrame: A DataFrame with a unique DatetimeIndex.
    """
    if not df.index.is_unique:
        # Identify duplicated timestamps
        duplicates = df.index[df.index.duplicated(keep=False)]
        logger.warning(
            f"{symbol}: Found duplicate timestamps in index: {duplicates.unique().tolist()}"
        )

        # Remove duplicates based on 'keep' strategy
        df = df[~df.index.duplicated(keep=keep)]
        logger.warning(
            f"{symbol}: Removed {len(duplicates.unique())} duplicated timestamps."
        )
    return df


def force_unique_index(df):
    """
    Reset the index, drop duplicates based on the index column, and reassign the index.
    This uses the current index name if available; otherwise, it uses "index".
    """
    idx_name = df.index.name if df.index.name is not None else "index"
    df_reset = df.reset_index()
    # Now use the actual index column name for dropping duplicates.
    return df_reset.drop_duplicates(subset=idx_name, keep="first").set_index(idx_name)


def download_multi_ticker_data(
    symbols, start_date, end_date, data_path, download=False
) -> pd.DataFrame:
    """
    Downloads data for multiple tickers at once using yfinance.download.
    It checks for cached data in `data_path` (as individual Parquet files). For any ticker
    missing cached data (or if forced via download=True), it downloads them in one call.
    Finally, it returns a combined DataFrame with multi-level columns (ticker, OHLCV).

    Args:
        symbols (list): List of ticker symbols.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        data_path (str): Directory for caching symbol data.
        download (bool): If True, force re-download even if a cache exists.

    Returns:
        pd.DataFrame: Combined DataFrame with multi-index columns.
    """
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    tickers_to_download = []
    cached_data = {}

    # Check each symbol for cached data
    for symbol in symbols:
        file_path = data_path / f"{symbol}.parquet"
        if not file_path.exists() or download:
            tickers_to_download.append(symbol)
        else:
            try:
                df = pd.read_parquet(file_path)
                cached_data[symbol] = df
            except Exception as e:
                logger.error(f"Error reading cached data for {symbol}: {e}")
                tickers_to_download.append(symbol)

    # Download data for tickers missing from cache
    if tickers_to_download:
        logger.info(f"Downloading data for: {', '.join(tickers_to_download)}")
        # Use the multi-ticker download API.
        # With the default parameters, yf.download returns a DataFrame with MultiIndex columns.
        df_downloaded = yf.download(
            tickers=tickers_to_download,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            threads=True,
        )
        # When using multiple tickers, the resulting DataFrame typically has a MultiIndex on columns.
        if isinstance(df_downloaded.columns, pd.MultiIndex):
            # Each top-level key should correspond to a ticker.
            for symbol in tickers_to_download:
                if symbol in df_downloaded.columns.levels[0]:
                    df_symbol = df_downloaded[symbol].copy()
                    # Remove duplicate indices if any
                    df_symbol = df_symbol.loc[~df_symbol.index.duplicated(keep="first")]
                    # (Optional) Reformat the DataFrame if needed
                    df_symbol = format_to_df_format(df_symbol, symbol)
                    cached_data[symbol] = df_symbol
                    file_path = data_path / f"{symbol}.parquet"
                    df_symbol.to_parquet(file_path)
                else:
                    logger.warning(f"No data found in download for {symbol}")
        else:
            # In some cases, yf.download may return a dict-like structure.
            for symbol, df_symbol in df_downloaded.items():
                df_symbol = df_symbol.loc[~df_symbol.index.duplicated(keep="first")]
                df_symbol = format_to_df_format(df_symbol, symbol)
                cached_data[symbol] = df_symbol
                file_path = data_path / f"{symbol}.parquet"
                df_symbol.to_parquet(file_path)

    # Combine cached data into a single DataFrame with multi-index columns
    combined_data = None
    for symbol, df in cached_data.items():
        # Make sure each DataFrame has columns in the (ticker, field) format.
        df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
        if combined_data is None:
            combined_data = df
        else:
            combined_data = combined_data.join(df, how="outer")
            combined_data = combined_data.loc[
                ~combined_data.index.duplicated(keep="first")
            ]

    if combined_data is not None:
        combined_data.sort_index(inplace=True)
        combined_data.ffill(inplace=True)
        combined_data.bfill(inplace=True)
        # Optionally, drop rows that are completely NaN
        combined_data.dropna(how="all", inplace=True)
    else:
        combined_data = pd.DataFrame()

    return combined_data


def convert_to_yfinance_format(data: dict, symbol: str) -> pd.DataFrame:
    candles = data.get("candles", [])
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df["datetime"] = df["datetime"].dt.floor("D")
    df.set_index("datetime", inplace=True)

    # Create a multi-index DataFrame similar to yfinance output
    multi_index_data = {
        (symbol, "Open"): df["open"],
        (symbol, "High"): df["high"],
        (symbol, "Low"): df["low"],
        (symbol, "Close"): df["close"],
        (symbol, "Adj Close"): df["close"],
        (symbol, "Volume"): df["volume"],
    }

    multi_index_df = pd.DataFrame(multi_index_data)
    multi_index_df.index.name = "Date"
    multi_index_df.columns = pd.MultiIndex.from_tuples(multi_index_df.columns)

    if len(multi_index_df.columns.levels[0]) == 1:
        # Drop the symbol level, leaving single-level columns
        multi_index_df.columns = multi_index_df.columns.droplevel(0)

    return multi_index_df
