import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.schwabapi.com/marketdata/v1/pricehistory"


def get_bearer_token():
    load_dotenv()  # Reload environment variables each time
    return os.getenv("SCHWAB_API_BEARER_TOKEN")


def download_schwab_data(
    symbol: str,
    period_type: str = "month",
    period: int = 1,
    frequency_type: str = "daily",
    frequency: int = 1,
    start_date: int = None,
    end_date: int = None,
    need_extended_hours_data: bool = False,
    need_previous_close: bool = False,
) -> dict:
    url = f"{BASE_URL}?symbol={symbol}&periodType={period_type}&period={period}&frequencyType={frequency_type}&frequency={frequency}"
    if start_date:
        url += f"&startDate={start_date}"
    if end_date:
        url += f"&endDate={end_date}"
    if need_extended_hours_data:
        url += f"&needExtendedHoursData=true"
    if need_previous_close:
        url += f"&needPreviousClose=true"

    headers = {
        "Authorization": f"Bearer {get_bearer_token()}",
        "Accept": "application/json",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()
