import math
from statistics import mean

# Global variables
data_file = "test_data.tsv"
pattern_file = "test_pattern.tsv"

# Main functions

def read_data(file_name: str) -> list[float]:
    """
    Read data from file.
    For now its just a row of values.
    
    >>> data = read_data(data_file)
    >>> data = [100.0, 101.0, 102.0, 106.0, 107.0, 107.0, 105.0, 104.0, 99.0, 96.0]
    """

    if file_name is None:
        print("File name is missing or some other issue occured.")
    
    with open(file_name, "r") as file:
        data = [float(line.strip()) for line in file]
    
    return data

def recognize_pattern(data: list[float], newPattern: list[float], tol: float = 1.0, precision: int = 3, abs_weight: float = 0.5) -> float | None:
    """
    Predict the next value after newPattern by finding similar difference patterns in data.
    Uses a Gaussian-weighted fuzzy match, combining absolute and percentage-based diffs.

    tol: standard deviation (sigma) controlling tolerance (smaller = stricter)
    precision: number of digits after the decimal point
    abs_weight: 0.0-1.0 — how much to favor absolute vs percentage prediction
    """

    if len(newPattern) < 2 or len(data) < 2:
        return None

    # Helper to compute weighted prediction for any diff function
    def weighted_prediction(diff_func):
        newPatternDiff = diff_func(newPattern)
        dataDiff = diff_func(data)
        weighted_predictions = []

        for i in range(len(dataDiff) - len(newPatternDiff)):
            total_weight = 1.0
            for j in range(len(newPatternDiff)):
                diff_err = dataDiff[i + j] - newPatternDiff[j]
                weight = math.exp(- (diff_err ** 2) / (2 * tol ** 2))
                total_weight *= weight
            if total_weight > 1e-6 and i + len(newPatternDiff) < len(dataDiff):
                next_diff = dataDiff[i + len(newPatternDiff)]
                weighted_predictions.append((next_diff, total_weight))

        if not weighted_predictions:
            return None

        numerator = sum(diff * w for diff, w in weighted_predictions)
        denominator = sum(w for _, w in weighted_predictions)
        return numerator / denominator

    # Absolute difference prediction
    abs_pred_diff = weighted_prediction(helper_value_to_diff)
    if abs_pred_diff is None:
        return None
    abs_pred_value = newPattern[-1] + abs_pred_diff

    # Percentage difference prediction
    pct_pred_diff = weighted_prediction(helper_value_to_pct_diff)
    if pct_pred_diff is None:
        return round(abs_pred_value, precision)
    pct_pred_value = newPattern[-1] * (1 + pct_pred_diff / 100.0)

    # Combine both predictions
    final_prediction = abs_weight * abs_pred_value + (1 - abs_weight) * pct_pred_value
    return round(final_prediction, precision)


# Helper functions

# Read list and return difference
def helper_value_to_diff(lst: list[float]) -> list[float]:
    return [lst[i + 1] - lst[i] for i in range(len(lst) - 1)]

def helper_value_to_pct_diff(lst: list[float]) -> list[float]:
    return [((lst[i + 1] - lst[i]) / lst[i]) * 100 for i in range(len(lst) - 1) if lst[i] != 0]