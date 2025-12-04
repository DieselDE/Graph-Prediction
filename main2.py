from client import client
from coingecko_sdk import RateLimitError, APIError

# $env:COINGECKO_PRO_API_KEY = 'your_pro_api_key_here'

def get_bitcoin_price():
    try:
        price_data = client.simple.price.get(
            ids="bitcoin",
            vs_currencies="eur",
        )
        # Access data safely whether the SDK returns a dict or Pydantic-like objects
        btc = None
        if isinstance(price_data, dict):
            btc = price_data.get("bitcoin")
            if isinstance(btc, dict):
                return btc.get("eur")
            # If inner object is a model-like object with attributes
            if hasattr(btc, "eur"):
                return getattr(btc, "eur")

        # If top-level object is model-like (not a dict)
        if hasattr(price_data, "bitcoin"):
            btc = getattr(price_data, "bitcoin")
            if hasattr(btc, "eur"):
                return getattr(btc, "eur")

        # If we get here we didn't find the expected value
        return None
    except RateLimitError:
        print("Rate limit exceeded. Please try again later.")
        return None
    except APIError as e:
        print(f"An API error occurred: {e}")
        return None


if __name__ == "__main__":
    price = get_bitcoin_price()
    if price is not None:
        print(f"The current price of Bitcoin is: €{price}")
    else:
        print("Could not retrieve Bitcoin price.")