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
            #print(f"The current price of Bitcoin is: €{price}")
            with open("test_data.tsv", "a") as f:
                f.write(str(price) + "\n")
            i += 1
            print(f"We are at {i} items in file")
        else:
            print("Could not retrieve Bitcoin price.")
        
        # Read your Bitcoin price data
        data = pr.read_data("test_data.tsv")

        # Get prediction
        prediction = pr.predict_next_price(
            data,
            pattern_length=10,    # Look at last 20 prices
            tema_period=10,       # TEMA with 20 period
            tema_weight=0.6      # Favor TEMA slightly (60/40)
        )

        print(f"Current price: €{data[-1]:.2f}")
        print(f"\nTEMA prediction: €{prediction['tema_prediction']}")
        with open("pr_tema.tsv", "a") as f:
            f.write(str(prediction['tema_prediction']) + "\n")
        print(f"Pattern prediction: €{prediction['pattern_prediction']}")
        with open("pr_pattern.tsv", "a") as g:
            g.write(str(prediction['pattern_prediction']) + "\n")
        print(f"Hybrid prediction: €{prediction['hybrid_prediction']}")
        with open("pr_comb.tsv", "a") as h:
            h.write(str(prediction['hybrid_prediction']) + "\n")
        print(f"Confidence: {prediction['confidence']}")
        
        time.sleep(60)