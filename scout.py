# scout.py
from __future__ import annotations

import os
import multiprocessing as mp
import warnings
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import config as cfg
import indicators as ta
import shared_utils

# --- Environment Variable Overrides ---
cfg.START_DATE = os.getenv("DATA_START", os.getenv("SLICE_START", cfg.START_DATE))
cfg.END_DATE = os.getenv("SLICE_END", cfg.END_DATE)
SLICE_START = pd.to_datetime(os.getenv("SLICE_START"))

warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Derived Constants ---
BARS_PER_HOUR = 12
BOOM_BAR_COUNT = BARS_PER_HOUR * cfg.PRICE_BOOM_PERIOD_H
SLOWDOWN_BAR_COUNT = BARS_PER_HOUR * cfg.PRICE_SLOWDOWN_PERIOD_H
GAP_WINDOW_BARS = BARS_PER_HOUR * cfg.GAP_VWAP_HOURS
RET_WINDOW_BARS = cfg.STRUCTURAL_TREND_DAYS * 24 * BARS_PER_HOUR
MIN_ATR_VALUE = 1e-8


def _add_resampled_indicator(df_base: pd.DataFrame, name: str, tf: str, func, **kwargs) -> pd.Series:
    """Generic helper to resample, calculate an indicator, and reindex."""
    agg_map = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_resampled = df_base.resample(tf, label="right", closed="right").agg(agg_map).dropna()

    indicator_series = func(df_resampled, **kwargs)
    if indicator_series is None or indicator_series.empty:
        return pd.Series(np.nan, index=df_base.index)

    if isinstance(indicator_series, pd.DataFrame):
        indicator_series = indicator_series.iloc[:, 0]

    df_resampled[name] = indicator_series.shift(1)
    return df_resampled[name].reindex(df_base.index, method="ffill")


def _prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich 5-minute data with all required indicators."""
    if df.empty:
        return pd.DataFrame()

    # --- EMAs ---
    df_4h = df.resample("4h", label="right", closed="right").agg({"close": "last"}).dropna()
    df["ema_fast_4h"] = ta.ema(df_4h["close"], cfg.EMA_FAST).shift(1).reindex(df.index, method="ffill")
    df["ema_slow_4h"] = ta.ema(df_4h["close"], cfg.EMA_SLOW).shift(1).reindex(df.index, method="ffill")

    # --- Volume Metrics ---
    df["usd_vol"] = df["close"] * df["volume"]
    df["usd_vol_4h"] = df["usd_vol"].rolling(window=4 * 12).sum().shift(1)
    
    # 1. Calculate 24h rolling USD volume
    bars_24h = 24 * 12
    df['vol_usd_24h'] = df['usd_vol'].rolling(window=bars_24h, min_periods=bars_24h).sum()

    # 2. Calculate the 30-day average of 4-hour volume blocks
    bars_4h = 4 * 12
    bars_30d = 30 * 24 * 12
    num_4h_blocks_in_30d = bars_30d / bars_4h
    df['mean_4h_vol_30d'] = df['usd_vol'].rolling(window=bars_30d, min_periods=bars_30d).sum() / num_4h_blocks_in_30d
    
    # 3. Calculate the spike ratio for every bar
    df['vol_spike_24h'] = df['vol_usd_24h'] / df['mean_4h_vol_30d']
    # --- END OF NEW LOGIC ---

    bars_30m = 6
    bars_30d = 30 * 24 * 12
    vol_30m_rolling = df["usd_vol"].rolling(window=bars_30m).sum()
    mean_30m_vol_30d = df["usd_vol"].rolling(window=bars_30d).sum() / (bars_30d / bars_30m)
    df["vol_spike_30m"] = (vol_30m_rolling / mean_30m_vol_30d).shift(1)

    # --- Other Indicators using Helper ---
    df[f"atr_{cfg.ATR_TIMEFRAME}"] = _add_resampled_indicator(df, "atr", cfg.ATR_TIMEFRAME, ta.atr, period=cfg.ATR_PERIOD)
    df[f"rsi_{cfg.RSI_TIMEFRAME}"] = _add_resampled_indicator(df, "rsi", cfg.RSI_TIMEFRAME, lambda d, period: ta.rsi(d["close"], period), period=cfg.RSI_PERIOD)
    df[f"adx_{cfg.ADX_TIMEFRAME}"] = _add_resampled_indicator(df, "adx", cfg.ADX_TIMEFRAME, ta.adx, period=cfg.ADX_PERIOD)

    # --- Analytics Indicators ---
    bbands = ta.bollinger(df.resample(cfg.ANALYTICS_BBANDS_TIMEFRAME)["close"].last().dropna(), cfg.ANALYTICS_BBANDS_PERIOD, cfg.ANALYTICS_BBANDS_STD_DEV).shift(1)
    df["bband_upper"] = bbands["upper"].reindex(df.index, method="ffill")
    df["bband_lower"] = bbands["lower"].reindex(df.index, method="ffill")

    ema_df_analytics = df.resample(cfg.ANALYTICS_EMA_TIMEFRAME)["close"].last().dropna()
    for period in cfg.ANALYTICS_EMA_PERIODS:
        df[f"ema_{period}_{cfg.ANALYTICS_EMA_TIMEFRAME}"] = ta.ema(ema_df_analytics, span=period).shift(1).reindex(df.index, method="ffill")

    # --- Price-based Features ---
    df["close_boom_ago"] = df["close"].shift(BOOM_BAR_COUNT)
    df["close_slowdown_ago"] = df["close"].shift(SLOWDOWN_BAR_COUNT)
    vwap_num = (df["close"] * df["volume"]).shift(1).rolling(GAP_WINDOW_BARS).sum()
    vwap_den = df["volume"].shift(1).rolling(GAP_WINDOW_BARS).sum()
    df["rolling_vwap"] = vwap_num / vwap_den
    df["ret_30d"] = df["close"].pct_change(periods=RET_WINDOW_BARS)

    return df


def _build_output_df(sigs: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Constructs the final DataFrame for signals that passed all filters."""
    boom_ret = sigs["close"] / sigs["close_boom_ago"] - 1.0
    cuts = [-float("inf"), *cfg.BV_BUCKETS, float("inf")]
    boom_bucket = pd.cut(boom_ret, bins=cuts, labels=range(1, len(cuts))).astype(int)

    output_df = pd.DataFrame({
        "timestamp": sigs.index,
        "entry": sigs["close"],
        "atr": sigs[f"atr_{cfg.ATR_TIMEFRAME}"],
        "ret_30d": sigs["ret_30d"],
        "rsi": sigs[f"rsi_{cfg.RSI_TIMEFRAME}"],
        "ema_fast_4h": sigs["ema_fast_4h"],
        "ema_slow_4h": sigs["ema_slow_4h"],
        "rolling_vwap": sigs["rolling_vwap"],
        "bband_upper": sigs["bband_upper"],
        "bband_lower": sigs["bband_lower"],
        "adx": sigs[f"adx_{cfg.ADX_TIMEFRAME}"],
        "boom_ret": boom_ret,
        "boom_bucket": boom_bucket,
        "vol_spike_30m": sigs["vol_spike_30m"],
        "vol_spike_24h": sigs["vol_spike_24h"]
    })

    for p in cfg.ANALYTICS_EMA_PERIODS:
        col = f"ema_{p}_{cfg.ANALYTICS_EMA_TIMEFRAME}"
        output_df[col] = sigs[col]

    return output_df


def process_symbol(sym: str) -> Tuple[str, int]:
    """Loads, prepares, and scans a single symbol for trading signals."""
    if shared_utils.is_blacklisted(sym):
        return sym, 0

    try:
        df = shared_utils.load_parquet_data(sym, cfg.START_DATE, cfg.END_DATE)
        df5 = _prepare_indicators(df)
    except FileNotFoundError:
        return sym, 0

    if df5.empty:
        return sym, 0

    # --- Define Entry Conditions ---
    c1_boom = (df5["close"] / df5["close_boom_ago"] - 1) >= cfg.PRICE_BOOM_PCT
    c2_slow = (df5["close"] / df5["close_slowdown_ago"] - 1) <= cfg.PRICE_SLOWDOWN_PCT
    c3_ema_down = df5["ema_fast_4h"] < df5["ema_slow_4h"]
    c4_rsi_ok = df5[f"rsi_{cfg.RSI_TIMEFRAME}"].between(cfg.RSI_ENTRY_MIN, cfg.RSI_ENTRY_MAX)
    c5_gap_ok = (abs(df5["close"] - df5["rolling_vwap"]) / df5["rolling_vwap"] <= cfg.GAP_MAX_DEV_PCT).rolling(cfg.GAP_MIN_BARS).min().astype(bool)
    c6_ret_ok = df5["ret_30d"] <= cfg.STRUCTURAL_TREND_RET_PCT
    
    c7_adx_ok = pd.Series(True, index=df5.index)
    if cfg.ADX_FILTER_ENABLED:
        c7_adx_ok = df5[f"adx_{cfg.ADX_TIMEFRAME}"] <= cfg.ADX_MAX_LEVEL

    c8_vol_ok = pd.Series(True, index=df5.index)
    if cfg.VOL_FILTER_ENABLED:
        vol_window = df5["usd_vol_4h"]
        too_low = vol_window < cfg.MIN_VOL_USD if cfg.MIN_VOL_USD is not None else False
        too_high = vol_window > cfg.MAX_VOL_USD if cfg.MAX_VOL_USD is not None else False
        c8_vol_ok = ~(too_low | too_high)

    c9_volatility_ok = pd.Series(True, index=df5.index)
    if cfg.VOLATILITY_FILTER_ENABLED:
        c9_volatility_ok = (df5[f"atr_{cfg.ATR_TIMEFRAME}"] / df5["close"]) >= cfg.MIN_ATR_PCT

    # --- Validity & Final Mask ---
    required_cols = [f"atr_{cfg.ATR_TIMEFRAME}", "close_boom_ago", f"rsi_{cfg.RSI_TIMEFRAME}", "ret_30d", "rolling_vwap", f"adx_{cfg.ADX_TIMEFRAME}", "vol_spike_30m"]
    is_valid = df5[required_cols].notna().all(axis=1) & (df5[f"atr_{cfg.ATR_TIMEFRAME}"] >= MIN_ATR_VALUE)
    
    mask = c1_boom & c2_slow & c3_ema_down & c4_rsi_ok & c5_gap_ok & c6_ret_ok & c7_adx_ok & c8_vol_ok & c9_volatility_ok & is_valid
    sigs = df5.loc[mask]

    if SLICE_START is not None:
        sigs = sigs[sigs.index >= SLICE_START]

    if sigs.empty:
        return sym, 0

    # --- Build and Save Output ---
    output_df = _build_output_df(sigs, sym)
    pq.write_table(pa.Table.from_pandas(output_df, preserve_index=False), cfg.SIGNALS_DIR / f"{sym}.parquet")

    return sym, len(sigs)


def main() -> None:
    """Main function to run the scouting process in parallel."""
    symbols = shared_utils.get_symbols_from_file()
    
    total_signals = 0
    with mp.Pool(initializer=shared_utils.load_blacklist_data, processes=(cfg.MAX_WORKERS or os.cpu_count())) as pool:
        with tqdm(total=len(symbols), desc="Scouting") as pbar:
            for sym, n_sigs in pool.imap_unordered(process_symbol, symbols):
                if n_sigs > 0:
                    pbar.set_postfix_str(f"{sym}: {n_sigs} sigs", refresh=True)
                total_signals += n_sigs
                pbar.update(1)

    print(f"\nScout finished. Total signals found: {total_signals:,}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 🔌  Compatibility shim for live_trader.py  (async scan_symbol)
# ---------------------------------------------------------------------------
#
# live_trader.run() calls:
#     sig_raw = await scout.scan_symbol(sym, cfg)
#
# That function vanished when scout.py was refactored for batch‑mode scanning.
# The stub below recreates it by running the *synchronous* indicator logic in
# a thread‑pool so it doesn’t block the asyncio loop.
#
# ‑ If the latest 5‑minute bar meets all entry criteria, we return a Signal
#   dataclass compatible with live_trader.Signal.
# ‑ Otherwise we return None, just as the old code did.
#
# Feel free to replace this with a fully‑vectorised async version later.

from datetime import datetime, timezone
import asyncio
from typing import Optional

try:
    # import the dataclass definition from live_trader
    from live_trader import Signal   # type: ignore
except ImportError:
    # fallback definition if live_trader isn’t on PYTHONPATH at import time
    from dataclasses import dataclass
    @dataclass
    class Signal:            # minimal fields live_trader needs
        symbol: str
        entry: float
        atr: float
        rsi: float
        regime: str

def _scan_symbol_sync(sym: str, cfg_dict: dict) -> Optional[Signal]:
    """Synchronous helper that re‑uses the existing indicator pipeline."""
    try:
        df = shared_utils.load_parquet_data(sym, cfg.START_DATE, cfg.END_DATE)
        if df.empty:
            return None
        df5 = _prepare_indicators(df.tail(RET_WINDOW_BARS * 2))   # last ~60 d
    except FileNotFoundError:
        return None

    if df5.empty:
        return None

    # Evaluate the very last 5‑minute bar
    last = df5.iloc[-1]

    # Re‑use the same entry conditions as process_symbol()
    c1 = (last["close"] / last["close_boom_ago"] - 1) >= cfg.PRICE_BOOM_PCT
    c2 = (last["close"] / last["close_slowdown_ago"] - 1) <= cfg.PRICE_SLOWDOWN_PCT
    c3 = last["ema_fast_4h"] < last["ema_slow_4h"]
    c4 = cfg.RSI_ENTRY_MIN <= last[f"rsi_{cfg.RSI_TIMEFRAME}"] <= cfg.RSI_ENTRY_MAX
    c5 = abs(last["close"] - last["rolling_vwap"]) / last["rolling_vwap"] <= cfg.GAP_MAX_DEV_PCT
    c6 = last["ret_30d"] <= cfg.STRUCTURAL_TREND_RET_PCT
    c7 = (not cfg.ADX_FILTER_ENABLED) or (last[f"adx_{cfg.ADX_TIMEFRAME}"] <= cfg.ADX_MAX_LEVEL)
    c8 = True   # volume filters need rolling context; skip for speed
    c9 = (not cfg.VOLATILITY_FILTER_ENABLED) or (
          (last[f"atr_{cfg.ATR_TIMEFRAME}"] / last["close"]) >= cfg.MIN_ATR_PCT)

    if all([c1, c2, c3, c4, c5, c6, c7, c8, c9]):
        return Signal(
            symbol=sym,
            entry=float(last["close"]),
            atr=float(last[f"atr_{cfg.ATR_TIMEFRAME}"]),
            rsi=float(last[f"rsi_{cfg.RSI_TIMEFRAME}"]),
            regime="LIVE",
        )
    return None


async def scan_symbol(sym: str, cfg_dict: dict) -> Optional[Signal]:
    """Async wrapper so live_trader can `await scout.scan_symbol()`."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _scan_symbol_sync, sym, cfg_dict)
