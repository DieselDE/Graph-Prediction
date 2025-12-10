import matplotlib.pyplot as plt
import prediction as pr

if __name__ == "__main__":
    # Read all data files
    actual_data = pr.read_data("test_data.tsv")  # Your actual BTC prices
    #ema_data = pr.read_data("pr_ema.tsv")      # Hybrid predictions
    pattern_data = pr.read_data("pr_pattern.tsv") # Pattern-only predictions
    tema_data = pr.read_data("pr_tema.tsv")       # TEMA-only predictions
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot actual prices
    plt.plot(actual_data, label='Actual BTC Price', color='black', linewidth=2, alpha=0.8)
    
    # Plot predictions
    #plt.plot(ema_data, label='EMA Prediction', color='blue', linewidth=1.5, linestyle='--', alpha=0.7)
    plt.plot(tema_data, label='TEMA Prediction', color='green', linewidth=1.5, linestyle='--', alpha=0.7)
    plt.plot(pattern_data, label='Pattern Prediction', color='red', linewidth=1.5, linestyle='--', alpha=0.7)
    
    # Styling
    plt.title('Bitcoin Price Predictions Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Time (data points)', fontsize=12)
    plt.ylabel('Price (€)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Optional: Print some statistics
    # print("\n=== Prediction Statistics ===")
    # print(f"Data points: {len(actual_data)}")
    # print(f"\nActual price range: €{min(actual_data):.2f} - €{max(actual_data):.2f}")
    # print(f"Latest actual price: €{actual_data[-1]:.2f}")
    # print(f"\nLatest predictions:")
    # print(f"  EMA: €{ema_data[-1]:.2f}")
    # print(f"  TEMA: €{tema_data[-1]:.2f}")
    # print(f"  Pattern: €{pattern_data[-1]:.2f}")