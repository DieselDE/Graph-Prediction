# 📈 Graph-Prediction

A Bitcoin price prediction and trading signal tool built with Python. It collects live Bitcoin prices, applies technical analysis (TEMA), recognizes historical price patterns, and generates buy/sell signals — all running autonomously in a loop.

> ⚠️ **Work in Progress** — actively under development.

---

## 🧠 How It Works

The system runs a continuous loop (every 60 seconds) and does four things:

1. **Collects** the current Bitcoin price in EUR
2. **Calculates** a Triple Exponential Moving Average (TEMA) to track the trend
3. **Recognizes patterns** in historical price changes to predict the next 5 values
4. **Signals** whether to buy or sell based on both indicators

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point — runs the main loop, coordinates all modules |
| `data_collection.py` | Fetches live Bitcoin price data |
| `prediction.py` | EMA/TEMA calculation and pattern-matching algorithm |
| `strategy.py` | Buy/sell signal logic based on TEMA and pattern output |
| `graph_creation.py` | Generates graph data |
| `plotgraph.py` | Visualizes price data and predictions |
| `simple_neural_network.py` | Experimental neural network module |
| `client.py` | Client-side interface / additional tooling |

### Generated Data Files

| File | Content |
|------|---------|
| `test_data.tsv` | Collected Bitcoin prices (one per minute) |
| `pr_pattern.tsv` | Pattern-based price predictions |
| `pr_tema.tsv` | TEMA values over time |
| `stock_aquired.tsv` | Logged buy events |
| `stock_sold.tsv` | Logged sell events |

---

## 🔬 Prediction Methods

### Triple Exponential Moving Average (TEMA)
TEMA is a smoothed trend indicator that reduces lag compared to a standard EMA. It is calculated as:

```
TEMA = 3×EMA1 − 3×EMA2 + EMA3
```

Where EMA1, EMA2, EMA3 are successively smoothed exponential moving averages of the price data.

### Pattern Recognition
The algorithm searches historical price-change sequences for the closest match to the most recent `N` price movements (using Sum of Squared Errors). The top 1% most similar past patterns are weighted and averaged to predict the next 5 price points.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Required packages (install via pip):

```bash
pip install requests  # or whatever your data_collection.py uses
```

> **Note:** A full `requirements.txt` is not yet included. Check the imports in each module for the exact dependencies.

### Running the Bot

```bash
python main.py
```

The bot will start collecting Bitcoin prices every 60 seconds. After enough data is collected, it will begin outputting pattern predictions and buy/sell signals to the console and TSV files.

---

## 📊 Buy / Sell Strategy

**Buy signal** is triggered when:
- The price has been **below** TEMA for the last `N` periods (potential upswing)
- The pattern prediction suggests the price will **rise** above the current price

**Sell signal** is triggered when:
- The price has been **above** TEMA for the last `N` periods (potential downswing)
- The pattern prediction suggests the price will **fall** below the current price

---

## 🗺️ Roadmap

- [x] Live Bitcoin price collection
- [x] TEMA-based trend analysis
- [x] Pattern recognition & weighted prediction
- [x] Basic buy/sell signal strategy
- [ ] Neural network integration (`simple_neural_network.py`)
- [ ] Graph visualization (`plotgraph.py`)
- [ ] Backtesting framework
- [ ] `requirements.txt`
- [ ] Configuration file (periods, thresholds, polling interval)

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is not financial advice. Do not use this tool to make real trading decisions.

---

## 👤 Author

**DieselDE** — combining Python and machine learning, one commit at a time.
