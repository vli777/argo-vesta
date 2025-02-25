import requests
from bs4 import BeautifulSoup
from typing import List

# not working due to rate limiting

def get_etf_holdings(etf_symbol: str) -> List[str]:
    """
    Scrapes the top 10 holdings for a given ETF symbol from ETFdb.com
    based on the provided HTML structure.

    Args:
        etf_symbol (str): The ticker symbol of the ETF.

    Returns:
        List[str]: A list of top 10 holding ticker symbols.
    """
    url = f"https://etfdb.com/etf/{etf_symbol}/#holdings"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        + "AppleWebKit/537.36 (KHTML, like Gecko) "
        + "Chrome/58.0.3029.110 Safari/537.3"
    }
    try:
        response = requests.get(url, headers=headers)
        # Check for 404 Not Found early
        if response.status_code == 404:
            print(f"{etf_symbol} not found or does not have holdings available.")
            return []

        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Locate the table by its ID
        holdings_table = soup.find("table", id="etf-holdings")
        if not holdings_table:
            print(f"Holdings table not found for ETF: {etf_symbol}")
            return []

        tbody = holdings_table.find("tbody")
        if not tbody:
            print(f"No table body found for ETF: {etf_symbol}")
            return []

        top_holdings = []
        # Iterate over the first 10 rows in tbody
        rows = tbody.find_all("tr")[:10]
        for row in rows:
            # Find the cell with data-th="Symbol"
            symbol_cell = row.find("td", {"data-th": "Symbol"})
            if symbol_cell:
                # Extract the text from the <a> tag inside the cell
                link = symbol_cell.find("a")
                if link and link.text:
                    top_holdings.append(link.text.strip())

        return top_holdings

    except requests.HTTPError as http_err:
        print(f"HTTP error occurred while fetching {etf_symbol}: {http_err}")
    except Exception as err:
        print(f"An error occurred while fetching {etf_symbol}: {err}")
    return []
