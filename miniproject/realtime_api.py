# realtime_api.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # loads .env file if present

GOLDAPI_KEY = os.getenv("GOLDAPI_KEY")


def fetch_live_gold_price():
    """
    Fetches the current gold price (XAU/USD) from GoldAPI.
    Returns: (price_float, error_message or None)
    """
    if not GOLDAPI_KEY:
        return None, "GOLDAPI_KEY not set. Please add it to .env."

    url = "https://www.goldapi.io/api/XAU/USD"
    headers = {
        "x-access-token": GOLDAPI_KEY,
        "Content-Type": "application/json"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None, f"API error: {resp.status_code} {resp.text}"

        data = resp.json()
        # GoldAPI normally returns 'price' as the main field
        live_price = data.get("price")
        if live_price is None:
            return None, "Could not find 'price' field in response."

        return float(live_price), None

    except Exception as e:
        return None, str(e)
    


