from datetime import datetime
from pathlib import Path
import pandas as pd
import pytz

from utils import logger
from utils.data_utils import (
    ensure_unique_timestamps,
    flatten_columns,
    force_unique_index,
    format_to_df_format,
    get_stock_data,
)
from utils.date_utils import find_valid_trading_date, is_after_4pm_est


def process_symbols(symbols, start_date, end_date, data_path, download) -> pd.DataFrame:
    """
    Loads or downloads data for each symbol, ensuring proper historical alignment.
    Skips any symbols with no data.
    """
    combined_data = pd.DataFrame()

    for symbol in symbols:
        symbol_data = load_or_download_symbol_data(
            symbol, start_date, end_date, data_path, download
        )
        if symbol_data.empty:
            logger.warning(f"No data for {symbol}, skipping.")
            continue

        # Remove duplicate indices and format the DataFrame.
        symbol_data = symbol_data.loc[~symbol_data.index.duplicated(keep="first")]
        symbol_data = format_to_df_format(symbol_data, symbol)
        earliest_date = symbol_data.index.min()
        logger.info(
            f"{symbol} starts at {earliest_date}. Using full available history."
        )

        # Convert columns to a multi-level index with (ticker, OHLC) structure.
        symbol_data.columns = pd.MultiIndex.from_product(
            [[symbol], symbol_data.columns]
        )

        # Merge symbol data into the main DataFrame.
        if combined_data.empty:
            combined_data = symbol_data
        else:
            combined_data = combined_data.join(symbol_data, how="outer")
            # Ensure the index is unique after the join.
            combined_data = combined_data.loc[
                ~combined_data.index.duplicated(keep="first")
            ]

        duplicates = combined_data.index[combined_data.index.duplicated()]
        if len(duplicates) > 0:
            logger.debug(
                f"Duplicates in combined data after merging {symbol}: {duplicates}"
            )

    # Forward-fill then backward-fill missing values.
    combined_data.ffill(inplace=True)
    combined_data.bfill(inplace=True)

    # Drop any rows that remain completely null.
    if combined_data.isna().any().any():
        logger.warning(
            "Data still has missing values after fill. Dropping remaining nulls."
        )
        combined_data.dropna(how="any", inplace=True)

    return combined_data


def load_or_download_symbol_data(symbol, start_date, end_date, data_path, download):
    """
    Loads symbol data from a Parquet file or downloads it if missing or outdated.
    Returns an empty DataFrame if no data is found.
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    pq_file = Path(data_path) / f"{symbol}.parquet"

    est = pytz.timezone("US/Eastern")
    now_est = datetime.now(est)
    today_ts = pd.Timestamp(now_est.date())

    # 1. If the Parquet file does not exist, download full history.
    if not pq_file.is_file():
        logger.info(
            f"No file for {symbol}. Downloading full history {start_ts} -> {end_ts}"
        )
        try:
            data = get_stock_data(symbol, start_date=start_ts, end_date=end_ts)
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return pd.DataFrame()  # No existing data to fall back on.
        data = flatten_columns(data, symbol)
        if data.empty:
            logger.warning(f"No data for {symbol} in range {start_ts} - {end_ts}.")
            return (
                pd.DataFrame()
            )  # Return an empty DataFrame because there's nothing to load.
        data = data.loc[~data.index.duplicated(keep="first")]
        data.to_parquet(pq_file)
        return format_to_df_format(data, symbol)

    # 2. If it's after 4 PM EST and today's data is already present, skip downloading.
    if is_after_4pm_est():
        existing_data = pd.read_parquet(pq_file)
        existing_data = flatten_columns(existing_data, symbol)
        existing_data = ensure_unique_timestamps(existing_data, symbol)
        if (
            existing_data.index.max() is not None
            and existing_data.index.max() >= today_ts
        ):
            logger.info(
                f"{symbol}: Already have today's data after market close, skipping."
            )
            return format_to_df_format(existing_data, symbol)

    # 3. Determine the effective end timestamp (adjusting for market open times).
    effective_end_ts = end_ts
    market_open = datetime.strptime("09:30", "%H:%M").time()
    if now_est.time() < market_open:
        effective_end_ts = find_valid_trading_date(end_ts, tz=est, direction="backward")
    if effective_end_ts.normalize() >= today_ts.normalize():
        effective_end_ts = find_valid_trading_date(
            today_ts - pd.Timedelta(days=1), tz=est, direction="backward"
        )

    # 4. Load existing data and ensure unique timestamps.
    existing_data = pd.read_parquet(pq_file)
    existing_data = flatten_columns(existing_data, symbol)
    existing_data = ensure_unique_timestamps(existing_data, symbol=symbol, keep="first")
    last_date = existing_data.index.max() if not existing_data.empty else None
    first_date = existing_data.index.min() if not existing_data.empty else None

    # 5. Prepend missing historical data if needed.
    if first_date is None or start_ts < first_date:
        logger.info(f"{symbol}: Fetching missing history {start_ts} -> {first_date}")
        history_data = get_stock_data(symbol, start_date=start_ts, end_date=first_date)
        history_data = flatten_columns(history_data, symbol)
        history_data = ensure_unique_timestamps(history_data, symbol)
        if not history_data.empty:
            history_data = history_data.loc[
                ~history_data.index.duplicated(keep="first")
            ]
            existing_data = existing_data.loc[
                ~existing_data.index.duplicated(keep="first")
            ]
            # Force uniqueness by resetting the index.
            existing_data = force_unique_index(existing_data)
            history_data = force_unique_index(history_data)
            existing_data = pd.concat([history_data, existing_data]).sort_index()
            existing_data = force_unique_index(existing_data)
        existing_data.to_parquet(pq_file)

    # 5b. Forced download branch.
    if download:
        logger.info(f"{symbol}: Forced download {start_ts} -> {effective_end_ts}")
        new_data = get_stock_data(
            symbol, start_date=start_ts, end_date=effective_end_ts
        )
        new_data = flatten_columns(new_data, symbol)
        if new_data.empty:
            return format_to_df_format(existing_data, symbol)

        #  --- Overwrite Same Dates With New Rows ---
        # Remove duplicates in both sets.
        existing_data = existing_data.loc[~existing_data.index.duplicated(keep="first")]
        new_data = new_data.loc[~new_data.index.duplicated(keep="first")]

        # Identify overlapping dates.
        overlap_idx = existing_data.index.intersection(new_data.index)

        # Drop any overlapping indices from existing_data, so new_data overwrites it.
        if not overlap_idx.empty:
            existing_data.drop(overlap_idx, inplace=True)

        # Force uniqueness by resetting the index on both.
        existing_data = force_unique_index(existing_data)
        new_data = force_unique_index(new_data)

        df_combined = pd.concat([existing_data, new_data]).sort_index()
        df_combined = force_unique_index(df_combined)
        df_combined = ensure_unique_timestamps(df_combined, symbol)
        df_combined.to_parquet(pq_file)

        return format_to_df_format(df_combined, symbol)

    # 6. Append missing data if existing data is outdated.
    if last_date is None or last_date < effective_end_ts:
        update_start = last_date + pd.Timedelta(days=1) if last_date else start_ts
        update_start = find_valid_trading_date(
            update_start, tz=est, direction="forward"
        )
        if update_start >= effective_end_ts:
            logger.debug(f"{symbol}: Data already up-to-date.")
            return format_to_df_format(existing_data, symbol)

        logger.info(f"{symbol}: Updating {update_start} -> {effective_end_ts}")
        df_new = get_stock_data(
            symbol, start_date=update_start, end_date=effective_end_ts
        )
        df_new = flatten_columns(df_new, symbol)

        if not df_new.empty:
            logger.debug(f"Checking duplicates for {symbol} before concat...")

            existing_data = existing_data.loc[
                ~existing_data.index.duplicated(keep="first")
            ]
            df_new = df_new.loc[~df_new.index.duplicated(keep="first")]

            # --- Overwrite Same Dates With New Rows ---
            overlap_idx = existing_data.index.intersection(df_new.index)
            if not overlap_idx.empty:
                existing_data.drop(overlap_idx, inplace=True)

            existing_data = force_unique_index(existing_data)
            df_new = force_unique_index(df_new)

            df_combined = pd.concat([existing_data, df_new]).sort_index()
            df_combined = force_unique_index(df_combined)
            df_combined.to_parquet(pq_file)
            return format_to_df_format(df_combined, symbol)
        else:
            logger.info(f"{symbol}: No new data found for update period.")
            return format_to_df_format(existing_data, symbol)
    else:
        return format_to_df_format(existing_data, symbol)
