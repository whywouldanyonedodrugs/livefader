# live/live_indicators.py
"""
Lightweight, dependency-free technical indicator calculations for the live bot.
These functions operate on lists or deques of numbers and do not require
pandas, numpy, or TA-Lib.
"""
from collections import deque
from typing import Deque, List

# --- EMA (Exponential Moving Average) ---

def ema_from_list(data: List[float], period: int) -> float:
    """
    Calculates an initial EMA value from a list of historical data.
    Uses a Simple Moving Average of the first `period` elements as the seed.
    """
    if not data or len(data) < period:
        return 0.0

    # Start with a simple moving average for the first value
    initial_sma = sum(data[:period]) / period
    
    # Apply EMA formula for the rest of the data
    k = 2 / (period + 1)
    ema = initial_sma
    for price in data[period:]:
        ema = price * k + ema * (1 - k)
        
    return ema

def next_ema(price: float, prev_ema: float, period: int) -> float:
    """
    Calculates the next EMA value incrementally.
    """
    if prev_ema == 0.0: # Handle first-time calculation
        return price
        
    k = 2 / (period + 1)
    return price * k + prev_ema * (1 - k)

# --- RSI (Relative Strength Index) ---

def rsi_from_deque(prices: Deque[float], period: int = 14) -> float:
    """
    Calculates RSI from a deque of recent prices.
    """
    if len(prices) < period + 1:
        return 50.0  # Return neutral value if not enough data

    # Get the last `period + 1` prices to calculate `period` changes
    recent_prices = list(prices)[-(period + 1):]
    
    gains = 0.0
    losses = 0.0

    for i in range(1, len(recent_prices)):
        change = recent_prices[i] - recent_prices[i-1]
        if change > 0:
            gains += change
        else:
            losses -= change # losses are stored as positive values

    avg_gain = gains / period
    avg_loss = losses / period

    if avg_loss == 0:
        return 100.0 # All gains, RSI is 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# --- ATR (Average True Range) ---

def atr_from_deques(highs: Deque[float], lows: Deque[float], closes: Deque[float], period: int = 14) -> float:
    """
    Calculates ATR from deques of recent OHLC data.
    Uses a simple moving average of the True Range.
    """
    if len(closes) < period + 1:
        return 0.0 # Not enough data

    # Get the last `period + 1` points to calculate `period` true ranges
    recent_highs = list(highs)[-(period + 1):]
    recent_lows = list(lows)[-(period + 1):]
    recent_closes = list(closes)[-(period + 1):]

    true_ranges = []
    for i in range(1, len(recent_closes)):
        high = recent_highs[i]
        low = recent_lows[i]
        prev_close = recent_closes[i-1]
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    if not true_ranges:
        return 0.0

    if len(true_ranges) < period:
        return 0.0 # Not enough TRs to calculate

    atr = sum(true_ranges[:period]) / period
    
    # Apply smoothing for subsequent values to get the most recent ATR
    for i in range(period, len(true_ranges)):
        atr = (atr * (period - 1) + true_ranges[i]) / period
        
    return atr

    return sum(true_ranges) / len(true_ranges)