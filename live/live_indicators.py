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

# --- ADX (Average Directional Index) --------------------------------------
def _tr(high, low, prev_close):
    return max(high - low, abs(high - prev_close), abs(low - prev_close))

# --- ADX (Average Directional Index) --------------------------------------
def _calculate_dms_and_tr(high, low, close, prev_high, prev_low, prev_close):
    """Calculates Directional Movement and True Range for a single period."""
    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
    
    move_up = high - prev_high
    move_down = prev_low - low
    
    plus_dm = move_up if move_up > move_down and move_up > 0 else 0
    minus_dm = move_down if move_down > move_up and move_down > 0 else 0
    
    return plus_dm, minus_dm, tr

def initial_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Tuple[float, tuple]:
    """
    Calculates initial ADX using Wilder's smoothing.
    Returns (adx, state_tuple)
    """
    if len(closes) < period * 2: # Need enough data for initial smoothing
        return 0.0, (0.0, 0.0, 0.0, 0.0)

    # Calculate initial series of +DM, -DM, and TR
    plus_dms, minus_dms, trs = [], [], []
    for i in range(1, len(closes)):
        pdm, mdm, tr = _calculate_dms_and_tr(highs[i], lows[i], closes[i], highs[i-1], lows[i-1], closes[i-1])
        plus_dms.append(pdm)
        minus_dms.append(mdm)
        trs.append(tr)

    # First smoothed values are simple averages
    atr = sum(trs[:period]) / period
    smooth_plus_dm = sum(plus_dms[:period]) / period
    smooth_minus_dm = sum(minus_dms[:period]) / period

    # Wilder's smoothing for the rest of the initial data
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
        smooth_plus_dm = (smooth_plus_dm * (period - 1) + plus_dms[i]) / period
        smooth_minus_dm = (smooth_minus_dm * (period - 1) + minus_dms[i]) / period

    # Calculate DI and DX
    dxs = []
    for i in range(period, len(trs)):
        plus_di = 100 * (smooth_plus_dm / atr) if atr != 0 else 0
        minus_di = 100 * (smooth_minus_dm / atr) if atr != 0 else 0
        di_sum = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum != 0 else 0
        dxs.append(dx)

    # Initial ADX is a simple average of the first DX values
    adx = sum(dxs[:period]) / period if dxs else 0
    
    # Smooth the rest of the DXs to get the final ADX
    for i in range(period, len(dxs)):
        adx = (adx * (period - 1) + dxs[i]) / period

    # State: prev_adx, prev_plus_dm, prev_minus_dm, prev_tr
    state = (adx, smooth_plus_dm, smooth_minus_dm, atr)
    return adx, state

def next_adx(high: float, low: float, close: float, prev_high: float, prev_low: float, prev_close: float, prev_state: tuple, period: int = 14) -> Tuple[float, tuple]:
    """
    Calculates the next ADX value incrementally.
    prev_state = (prev_adx, prev_smooth_plus_dm, prev_smooth_minus_dm, prev_atr)
    """
    prev_adx, prev_smooth_plus_dm, prev_smooth_minus_dm, prev_atr = prev_state
    
    # 1. Calculate current +DM, -DM, TR
    plus_dm, minus_dm, tr = _calculate_dms_and_tr(high, low, close, prev_high, prev_low, prev_close)

    # 2. Calculate smoothed values
    smooth_atr = (prev_atr * (period - 1) + tr) / period
    smooth_plus_dm = (prev_smooth_plus_dm * (period - 1) + plus_dm) / period
    smooth_minus_dm = (prev_smooth_minus_dm * (period - 1) + minus_dm) / period

    # 3. Calculate current DI and DX
    plus_di = 100 * (smooth_plus_dm / smooth_atr) if smooth_atr != 0 else 0
    minus_di = 100 * (smooth_minus_dm / smooth_atr) if smooth_atr != 0 else 0
    di_sum = plus_di + minus_di
    dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum != 0 else 0

    # 4. Smooth DX to get the new ADX
    new_adx = (prev_adx * (period - 1) + dx) / period
    
    new_state = (new_adx, smooth_plus_dm, smooth_minus_dm, smooth_atr)
    return new_adx, new_state