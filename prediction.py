import math
from statistics import mean

def read_data(file_name: str) -> list[float]:
    """Read price data from file."""
    with open(file_name, "r") as file:
        return [float(line.strip()) for line in file]


# ============= EMA/TEMA Functions =============

def ema(data: list[float], period: int = 20) -> list[float]:
    """Calculate Exponential Moving Average."""
    if not data or period <= 0:
        return []
    
    ema_values = [data[0]]
    k = 2 / (period + 1)
    
    for i in range(1, len(data)):
        ema_values.append((data[i] - ema_values[-1]) * k + ema_values[-1])
    
    return ema_values


def tema(data: list[float], period: int) -> list[float]:
    """Calculate Triple Exponential Moving Average."""
    if len(data) < period:
        return []
    
    ema1 = ema(data, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    
    return [3 * e1 - 3 * e2 + e3 for e1, e2, e3 in zip(ema1, ema2, ema3)]


# ============= Pattern Recognition Functions =============

def find_pattern(data: list[float], pattern_len: int = 10, prediction_len: int = 5, tolerance: int = 1) -> list[float] | None:

    if len(data) < pattern_len + prediction_len:
        print("data has less points than pattern_len + prediction_len")
        return None

    # 1. Build change list
    data_change = [data[i+1] - data[i] for i in range(len(data)-1)]

    # 2. Pattern we want to match
    pattern = data_change[-pattern_len:]

    weighted_segments = []

    # 3. Loop all possible past patterns
    for i in range(len(data_change) - pattern_len - prediction_len):
        past = data_change[i : i + pattern_len]

        # similarity score (SSE)
        diff = sum((pattern[j] - past[j])**2 for j in range(pattern_len))
        weight = 1.0 / (diff + tolerance)

        future = data_change[i + pattern_len : i + pattern_len + prediction_len]
        
        if len(future) == prediction_len:
            weighted_segments.append((future, weight))

    if not weighted_segments:
        print("no matches found")
        return None

    # ---- NEW: sort by weight descending & select top 1% ----
    weighted_segments.sort(key=lambda x: x[1], reverse=True)
    top_n = max(1, int(len(weighted_segments) * 0.01))  # at least one
    weighted_segments = weighted_segments[:top_n]

    # 4. Weighted average
    prediction_diffs = [0.0] * prediction_len
    total_weight = sum(w for _, w in weighted_segments)

    for future, w in weighted_segments:
        for k in range(prediction_len):
            prediction_diffs[k] += future[k] * w

    prediction_diffs = [p / total_weight for p in prediction_diffs]

    # 5. Convert diffs → actual values
    predicted_values = []
    last_value = data[-1]

    for d in prediction_diffs:
        last_value += d
        predicted_values.append(last_value)

    return predicted_values
