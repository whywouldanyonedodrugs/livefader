# live/live_indicators.py
"""
Lightweight, dependency-free technical indicator calculations for the live bot.
These functions operate on lists or deques of numbers and do not require
pandas, numpy, or TA-Lib.

v2.0: Switched to EMA-based smoothing for RSI and ATR to align with
      standard libraries like TA-Lib and pandas_ta.
"""
from collections import deque
from typing import Deque, List, Tuple

# --- EMA (Exponential Moving Average) ---

def ema_from_list(data: List[float], period: int) -> float:
    """
    Calculates an initial EMA value from a list of historical data.
    Uses a Simple Moving Average of the first `period` elements as the seed.
    """
    if not data or len(data) < period:
        return 0.0

    initial_sma = sum(data[:period]) / period
    
    k = 2 / (period + 1)
    ema = initial_sma
    for price in data[period:]:
        ema = price * k + ema * (1 - k)
        
    return ema

def next_ema(price: float, prev_ema: float, period: int) -> float:
    """
    Calculates the next EMA value incrementally.
    """
    if prev_ema == 0.0:
        return price
        
    k = 2 / (period + 1)
    return price * k + prev_ema * (1 - k)

# --- RSI (Relative Strength Index) ---

def initial_rsi(prices: List[float], period: int = 14) -> Tuple[float, float, float]:
    """
    Calculates the initial RSI, Average Gain, and Average Loss from a list of prices.
    The first average is a simple average, subsequent values are smoothed.
    Returns (rsi, avg_gain, avg_loss)
    """
    if len(prices) < period + 1:
        return 50.0, 0.0, 0.0

    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    initial_gains = sum(c for c in changes[:period] if c > 0)
    initial_losses = sum(-c for c in changes[:period] if c < 0)

    avg_gain = initial_gains / period
    avg_loss = initial_losses / period

    # Smooth subsequent values
    for i in range(period, len(changes)):
        change = changes[i]
        gain = change if change > 0 else 0
        loss = -change if change < 0 else 0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0, avg_gain, avg_loss

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi, avg_gain, avg_loss

def next_rsi(price: float, prev_price: float, prev_avg_gain: float, prev_avg_loss: float, period: int = 14) -> Tuple[float, float, float]:
    """
    Calculates the next RSI value incrementally using previous smoothed averages.
    Returns (rsi, new_avg_gain, new_avg_loss)
    """
    change = price - prev_price
    gain = change if change > 0 else 0.0
    loss = -change if change < 0 else 0.0

    avg_gain = (prev_avg_gain * (period - 1) + gain) / period
    avg_loss = (prev_avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0, avg_gain, avg_loss

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi, avg_gain, avg_loss

# --- ATR (Average True Range) ---

def initial_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """
    Calculates the initial ATR from lists of historical data.
    Uses Wilder's Smoothing Method (same as TA-Lib).
    """
    if len(closes) < period + 1:
        return 0.0

    true_ranges = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        true_ranges.append(tr)

    if not true_ranges or len(true_ranges) < period:
        return 0.0

    # The first ATR is a simple average of the first `period` TRs
    atr = sum(true_ranges[:period]) / period
    
    # Apply Wilder's smoothing for the rest of the historical data
    for i in range(period, len(true_ranges)):
        atr = (atr * (period - 1) + true_ranges[i]) / period
        
    return atr

def next_atr(prev_atr: float, high: float, low: float, close: float, prev_close: float, period: int = 14) -> float:
    """
    Calculates the next ATR value incrementally using Wilder's Smoothing.
    """
    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
    
    # If this is the first calculation, prev_atr might be a simple average.
    # The formula remains the same for subsequent calculations.
    return (prev_atr * (period - 1) + tr) / period