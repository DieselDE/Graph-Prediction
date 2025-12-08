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
    """Finds possible patterns

    Args:
        data (list[float]): list with all the data
        pattern_len (int, optional): len of analyzed pattern. Defaults to 10.
        prediction_len (int, optional): len of prediction. Defaults to 5.
        tolerance (float, optional): for finetuning. Defaults to 1.

    Returns:
        list[float]: returns the possible values
    """
    
    # break conditions
    if len(data) < pattern_len + prediction_len:
        print("data has less points than pattern_len + prediction_len")
        return None

    # ---- 1. Build difference list ----
    data_change = []
    for i in range(len(data) - 1):
        data_change.append(data[i+1] - data[i])

    # ---- 2. Pattern to match ----
    pattern = data_change[-pattern_len:]

    weighted_segments = []

    # ---- 3. Slide across all possible past patterns ----
    for i in range(len(data_change) - pattern_len - prediction_len):
        checked_pattern = data_change[i : i + pattern_len]

        # similarity (sum of squared error)
        diff = 0.0
        for j in range(pattern_len):
            diff += (pattern[j] - checked_pattern[j]) ** 2

        weight = 1.0 / (diff + tolerance)

        # future diffs after this pattern
        future_segment = data_change[i + pattern_len : i + pattern_len + prediction_len]

        if len(future_segment) == prediction_len:
            weighted_segments.append((future_segment, weight))

    if not weighted_segments:
        print("no matches found")
        return None

    # ---- 4. Weighted average of predicted diffs ----
    prediction_diffs = [0.0] * prediction_len
    total_weight = sum(w for _, w in weighted_segments)

    for future, w in weighted_segments:
        for k in range(prediction_len):
            prediction_diffs[k] += future[k] * w

    prediction_diffs = [p / total_weight for p in prediction_diffs]

    # ---- 5. Convert diffs → actual predicted values ----
    predicted_values = []
    last_value = data[-1]

    for d in prediction_diffs:
        next_value = last_value + d
        predicted_values.append(next_value)
        last_value = next_value

    return predicted_values

