import time
import prediction as pr
from client import client
from coingecko_sdk import RateLimitError, APIError

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
    
    i = 0
    
    while(True):
        price = get_bitcoin_price()
        if price is not None:
            print(f"The current price of Bitcoin is: €{price}")
            with open("test_data.tsv", "a") as f:
                f.write(str(price) + "\n")
            i += 1
            print(f"We are at {i} items in file")
            
            # Read your Bitcoin price data
            data = pr.read_data("test_data.tsv")

            # Get prediction
            prediction = []
            if i % 5 == 0:
                prediction = pr.find_pattern(data, 20, 5, 1)

            print(f"Pattern prediction: €{prediction}")
            with open("pr_pattern.tsv", "a") as g:
                for j in range(len(prediction)):
                    g.write(str(round(prediction[j], 2)) + "\n")
        else:
            print("Could not retrieve Bitcoin price.")
        
        time.sleep(60)