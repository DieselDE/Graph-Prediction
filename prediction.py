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

# irelevant
def predict_tema(data: list[float], period: int = 5) -> float | None:
    """Predict next value using TEMA trend."""
    t = tema(data, period)
    if len(t) < 2:
        return None
    
    trend = t[-1] - t[-2]
    return t[-1] + trend


# ============= Pattern Recognition Functions =============

#def find_similar_patterns(data: list[float], pattern_length: int = 10, tolerance: float = 1.0) -> float | None:
#    """
#    Find patterns in historical data similar to recent prices.
#    
#    Args:
#        data: Full price history
#        pattern_length: How many recent prices to use as pattern (default 5)
#        tolerance: How strict the matching is (lower = stricter)
#    
#    Returns:
#        Predicted next price based on weighted average of similar patterns
#    """
#    if len(data) < pattern_length + 2:
#        return None
#    
#    # The pattern is the last N prices
#    recent_pattern = data[-pattern_length:]
#    
#    # Convert to percentage changes (more generalizable than absolute values)
#    pattern_changes = [
#        ((recent_pattern[i+1] - recent_pattern[i]) / recent_pattern[i]) * 100
#        for i in range(len(recent_pattern) - 1)
#    ]
#    
#    # Search through historical data for similar patterns
#    weighted_predictions = []
#    
#    # Stop before the recent pattern (we need history, not current data)
#    for i in range(len(data) - pattern_length - 1):
#        window = data[i : i + pattern_length]
#        window_changes = [
#            ((window[j+1] - window[j]) / window[j]) * 100
#            for j in range(len(window) - 1)
#        ]
#        
#        # Calculate similarity using Gaussian weighting
#        similarity = 1.0
#        for j in range(len(pattern_changes)):
#            diff = window_changes[j] - pattern_changes[j]
#            weight = math.exp(-(diff ** 2) / (2 * tolerance ** 2))
#            similarity *= weight
#        
#        # If similar enough, use the next value that followed this pattern
#        if similarity > 0.01:  # Minimum threshold
#            next_value = data[i + pattern_length]
#            weighted_predictions.append((next_value, similarity))
#    
#    if not weighted_predictions:
#        return None
#    
#    # Calculate weighted average of all predictions
#    total_weight = sum(w for _, w in weighted_predictions)
#    prediction = sum(val * w for val, w in weighted_predictions) / total_weight
#    
#    return prediction

def find_similar_patterns(data: list[float], pattern_length: int = 20, prediction_horizon: int = 5,tolerance: float = 1.0) -> list[float] | None:
    """
    Find patterns in historical data similar to recent prices and predict multiple future values.
    
    Args:
        data: Full price history
        pattern_length: How many recent prices to use as pattern (default 20)
        prediction_horizon: How many future prices to predict (default 5)
        tolerance: How strict the matching is (lower = stricter)
    
    Returns:
        List of predicted next prices based on weighted average of similar patterns,
        or None if insufficient data
    """
    # Need enough data for: pattern + prediction_horizon + at least one historical match
    if len(data) < pattern_length + prediction_horizon + 1:
        return None
    
    # The pattern is the last N prices
    recent_pattern = data[-pattern_length:]
    
    # Convert to percentage changes (more generalizable than absolute values)
    pattern_changes = [
        ((recent_pattern[i+1] - recent_pattern[i]) / recent_pattern[i]) * 100
        for i in range(len(recent_pattern) - 1)
    ]
    
    # Search through historical data for similar patterns
    # We need pattern_length + prediction_horizon values for each match
    weighted_predictions = []
    
    # Stop early enough to have prediction_horizon values after the pattern
    for i in range(len(data) - pattern_length - prediction_horizon):
        window = data[i : i + pattern_length]
        window_changes = [
            ((window[j+1] - window[j]) / window[j]) * 100
            for j in range(len(window) - 1)
        ]
        
        # Calculate similarity using Gaussian weighting
        similarity = 1.0
        for j in range(len(pattern_changes)):
            diff = window_changes[j] - pattern_changes[j]
            weight = math.exp(-(diff ** 2) / (2 * tolerance ** 2))
            similarity *= weight
        
        # If similar enough, use the next values that followed this pattern
        if similarity > 0.01:  # Minimum threshold
            # Get the next prediction_horizon values after this pattern
            next_values = data[i + pattern_length : i + pattern_length + prediction_horizon]
            weighted_predictions.append((next_values, similarity))
    
    if not weighted_predictions:
        return None
    
    # Calculate weighted average for each future time step
    total_weight = sum(w for _, w in weighted_predictions)
    predictions = []
    
    for step in range(prediction_horizon):
        # Average the prediction for this specific future time step
        step_prediction = sum(
            vals[step] * w 
            for vals, w in weighted_predictions 
            if step < len(vals)  # Safety check
        ) / total_weight
        predictions.append(step_prediction)
    
    return predictions

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

# ============= Hybrid Prediction System =============

# irelevant
def predict_next_price(data: list[float], pattern_length: int = 5, tema_period: int = 5, tema_weight: float = 0.6) -> dict:
    """
    Hybrid prediction combining TEMA and pattern recognition.
    
    Args:
        data: Full price history
        pattern_length: How many recent prices to use for pattern matching
        tema_period: Period for TEMA calculation
        tema_weight: How much to weight TEMA vs pattern (0.0-1.0)
    
    Returns:
        Dictionary with predictions and confidence metrics
    """
    tema_pred = predict_tema(data, tema_period)
    pattern_pred = find_similar_patterns(data, pattern_length)
    
    result = {
        'tema_prediction': round(tema_pred, 2) if tema_pred else None,
        'pattern_prediction': round(pattern_pred, 2) if pattern_pred else None,
        'hybrid_prediction': None,
        'confidence': 'low'
    }
    
    # If both methods agree (within 2%), high confidence
    if tema_pred and pattern_pred:
        diff_pct = abs(tema_pred - pattern_pred) / tema_pred * 100
        
        if diff_pct < 2:
            result['confidence'] = 'high'
        elif diff_pct < 5:
            result['confidence'] = 'medium'
        
        # Combine predictions
        hybrid = tema_weight * tema_pred + (1 - tema_weight) * pattern_pred
        result['hybrid_prediction'] = round(hybrid, 2)
    
    # Fallback to whichever method worked
    elif tema_pred:
        result['hybrid_prediction'] = round(tema_pred, 2)
    elif pattern_pred:
        result['hybrid_prediction'] = round(pattern_pred, 2)
    
    return result