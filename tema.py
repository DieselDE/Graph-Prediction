# Global variables
tema_file = "test_tema.tsv"

# Main functions

def ema(data: list[float], period: int) -> list[float]:
    """
    Calculates Exponential Moving Average: EMA
    """

    if not data or period <= 0:
        return []
    
    ema_values = []
    k = 2 / (period + 1)
    ema_values.append(data[0])
    
    for i in range(1, len(data)):
        ema_values.append((data[i] - ema_values[-1]) * k + ema_values[-1])
    
    return ema_values

def tema(data: list[float], period: int) -> list[float]:
    """
    Calculates Triple Exponential Moving Average: TEMA
    """
    
    if len(data) < period:
        return []
    
    ema1 = ema(data, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    
    return [3 * e1 - 3 * e2 + e3 for e1, e2, e3 in zip(ema1, ema2, ema3)]

def predict_next_tema(data: list[float], period: int = 5, precision = 3) -> float | None:
    """
    Predicts next value based on TEMA trend
    Uses the last difference between recent TEMA values as extrapolation.
    """

    t = tema(data, period)
    if len(t) < 2:
        return None
    
    last_diff = t[-1] - t[-2]
    prediction = t[-1] + last_diff
    
    return round(prediction, precision)

def ema_to_file(data: list[float], period: int = 5):
    """
    Putting ema data to a file to clear the noise for the pattern recognitions algorithm.
    """

    ema_data = tema(data, period)

    with open(tema_file, "w") as file:
        for x in ema_data:
            file.write(f"{x:.2f}\n")