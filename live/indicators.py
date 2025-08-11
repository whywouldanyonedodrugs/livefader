# =============================================
# indicators.py – TA‑Lib / pandas_ta wrapper
# =============================================
"""Compute technical indicators.

• Uses TA‑Lib if installed (fast C bindings).
• Falls back to pandas_ta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated.*")

# ----------------------------------------------------------------------
# NumPy 2 compatibility patch for older pandas_ta versions
# ----------------------------------------------------------------------
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore

# Prefer TA‑Lib; else pandas_ta
try:
    import talib
    _HAS_TA = True
except ImportError:
    try:
        import pandas_ta as pta  # type: ignore
    except ImportError as exc:
        raise ImportError("Neither TA‑Lib nor pandas_ta is installed.\n"
                          "Run `pip install ta-lib‑binary` (Windows) or `pip install pandas_ta`."
                          ) from exc
    _HAS_TA = False

# --- MODIFICATION: Add 'adx' to the export list ---
__all__ = ["ema", "atr", "rsi", "macd", "bollinger", "lbr_310", "adx"]


def ema(series: pd.Series, span: int) -> pd.Series:
    if _HAS_TA:
        return pd.Series(talib.EMA(series, timeperiod=span), index=series.index)
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    if _HAS_TA:
        return pd.Series(talib.ATR(df["high"], df["low"], df["close"], timeperiod=period), index=df.index)
    atr_series = pta.atr(high=df["high"], low=df["low"], close=df["close"], length=period)
    if atr_series is None:
        return pd.Series(dtype='float64', index=df.index)
    return atr_series


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if _HAS_TA:
        return pd.Series(talib.RSI(series, timeperiod=period), index=series.index)
    rsi_series = pta.rsi(series, length=period)
    if rsi_series is None:
        return pd.Series(dtype='float64', index=series.index)
    return rsi_series

def macd(series: pd.Series) -> pd.DataFrame:
    """
    Return DataFrame with columns 'macd', 'signal', 'hist', regardless of
    whether TA-Lib or pandas_ta is used behind the scenes.
    """
    if _HAS_TA:
        macd, sig, hist = talib.MACD(series)
        return pd.DataFrame(
            {"macd": macd, "signal": sig, "hist": hist}, index=series.index
        )

    df_raw = pta.macd(series)
    mapping = {}
    for col in df_raw.columns:
        if "MACDh" in col or "hist" in col: mapping[col] = "hist"
        elif "MACDs" in col: mapping[col] = "signal"
        elif "MACD" in col: mapping[col] = "macd"
    df = df_raw.rename(columns=mapping)

    if "hist" not in df.columns: df["hist"] = df["macd"] - df["signal"]
    if "macd" not in df.columns: df["macd"] = np.nan
    if "signal" not in df.columns: df["signal"] = np.nan

    return df[["macd", "signal", "hist"]]


def bollinger(series: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    if _HAS_TA:
        upper, mid, lower = talib.BBANDS(series, timeperiod=length, nbdevup=std, nbdevdn=std)
        return pd.DataFrame({"upper": upper, "mid": mid, "lower": lower}, index=series.index)
    
    bbands = pta.bbands(series, length=length, std=std)
    
    if bbands is None or bbands.empty:
        return pd.DataFrame(columns=["upper", "mid", "lower"], index=series.index)

    upper_col = [col for col in bbands.columns if 'BBU' in col.upper()]
    mid_col = [col for col in bbands.columns if 'BBM' in col.upper()]
    lower_col = [col for col in bbands.columns if 'BBL' in col.upper()]

    if not (upper_col and mid_col and lower_col):
         return pd.DataFrame(columns=["upper", "mid", "lower"], index=series.index)

    return pd.DataFrame({
        "upper": bbands[upper_col[0]],
        "mid": bbands[mid_col[0]],
        "lower": bbands[lower_col[0]],
    }, index=series.index)


def lbr_310(series: pd.Series) -> pd.Series:
    """Linda Bradford Raschke 3‑10 oscillator = SMA3(close) – SMA10(close)."""
    return series.rolling(3).mean() - series.rolling(10).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates the Average Directional Index (ADX).
    Returns a Series containing the ADX values, ensuring it's robust.
    """
    # Ensure there's enough data to calculate
    if len(df) < period:
        return pd.Series(dtype='float64', index=df.index)

    if _HAS_TA:
        adx_series = talib.ADX(df["high"], df["low"], df["close"], timeperiod=period)
        return pd.Series(adx_series, index=df.index)
    
    # pandas_ta fallback
    adx_df = pta.adx(high=df["high"], low=df["low"], close=df["close"], length=period)
    if adx_df is None or adx_df.empty:
        return pd.Series(dtype='float64', index=df.index)
        
    # Find the column that contains 'ADX'
    adx_col = [col for col in adx_df.columns if 'ADX' in col.upper()]
    if not adx_col:
        return pd.Series(dtype='float64', index=df.index)
        
    return adx_df[adx_col[0]]

def vwap_stack_features(df: pd.DataFrame, lookback_bars: int = 12, band_pct: float = 0.004):
    """
    df: columns ['open','high','low','close','volume'] ascending.
    Returns:
      vwap_frac_in_band: share of prior window closes inside ±band_pct around rolling VWAP (current bar excluded)
      vwap_expansion_pct: |last_close / current_vwap - 1|
      vwap_slope_pph: VWAP slope (percent per hour) over the last hour (approx)
    """
    px = df["close"].astype(float).values
    vol = df["volume"].astype(float).values
    n = len(df)
    if n < lookback_bars + 2 or np.nansum(vol) == 0:
        return {"vwap_frac_in_band": 0.0, "vwap_expansion_pct": 0.0, "vwap_slope_pph": 0.0}

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tpv = (tp * df["volume"]).rolling(lookback_bars).sum()
    vv = df["volume"].rolling(lookback_bars).sum()
    rvwap = (tpv / vv).shift(1)  # exclude current bar for consolidation check

    cur_close = df["close"].iloc[-1]
    cur_vwap = (tpv / vv).iloc[-1]

    prior = df.iloc[-(lookback_bars+1):-1].copy()
    vwap_prior = rvwap.iloc[-(lookback_bars): -0].values
    band_hi = vwap_prior * (1 + band_pct)
    band_lo = vwap_prior * (1 - band_pct)
    closes_prior = prior["close"].astype(float).values
    in_band = (closes_prior >= band_lo) & (closes_prior <= band_hi)
    frac = float(in_band.mean()) if len(in_band) else 0.0

    expansion = abs(cur_close / cur_vwap - 1.0) if cur_vwap and np.isfinite(cur_vwap) else 0.0

    k = min(lookback_bars, 12)
    vsub = (tpv.iloc[-k:] / vv.iloc[-k:]).values
    slope = (vsub[-1] - vsub[0]) / vsub[0] if (len(vsub) >= 2 and vsub[0]) else 0.0
    slope_pph = float(slope * (60/5) / k)  # convert to percent-per-hour on 5m bars

    return {"vwap_frac_in_band": float(frac),
            "vwap_expansion_pct": float(expansion),
            "vwap_slope_pph": float(slope_pph)}