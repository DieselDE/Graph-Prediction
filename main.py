import prediction as pr


# ============= Simple Usage Example =============

if __name__ == "__main__":
    # Read your Bitcoin price data
    data = pr.read_data("test_data.tsv")
    
    # Get prediction
    prediction = pr.predict_next_price(
        data,
        pattern_length=20,    # Look at last 20 prices
        tema_period=20,       # TEMA with 20 period
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