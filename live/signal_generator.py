# live/signal_generator.py
from dataclasses import dataclass
from typing import Optional, Deque
from collections import deque
import config as cfg
import logging
import re
import traceback
import ccxt
import pandas as pd
import numpy as np
from . import indicators as ta
from .exchange_proxy import fetch_ohlcv_paginated

LOG = logging.getLogger(__name__)

@dataclass
class Signal:
    symbol: str
    entry: float
    atr: float
    rsi: float
    ret_30d: float
    adx:   float

def _get_hours_from_timeframe(tf: str) -> float:
    """Converts timeframe string like '1h', '4h', '1d' to hours."""
    match = re.match(r"(\d+)(\w)", tf)
    if not match:
        return 1.0 # Default to 1 hour if format is unexpected
    
    val, unit = int(match.group(1)), match.group(2).lower()
    
    if unit == 'm':
        return val / 60
    if unit == 'h':
        return float(val)
    if unit == 'd':
        return float(val * 24)
    if unit == 'w':
        return float(val * 24 * 7)
    return 1.0

class SignalGenerator:
    def __init__(self, symbol: str, exchange):
        """
        Initializes the generator. All async operations are moved to warm_up().
        """
        self.symbol = symbol
        self.exchange = exchange
        self.is_warmed_up = False
        self.last_processed_timestamp = 0

        # --- Configurable Parameters ---
        self.ema_tf = cfg.EMA_TIMEFRAME
        self.rsi_tf = cfg.RSI_TIMEFRAME
        self.adx_tf = cfg.ADX_TIMEFRAME
        self.rsi_period = cfg.RSI_PERIOD
        self.atr_period = cfg.ADX_PERIOD  # Using ADX_PERIOD for ATR as per your code
        self.adx_period = cfg.ADX_PERIOD

        # --- State for raw 5m data (for boom/bust check) ---
        hours_per_candle = _get_hours_from_timeframe(cfg.TIMEFRAME)
        boom_candles = int(cfg.PRICE_BOOM_PERIOD_H / hours_per_candle)
        self.price_history: Deque[float] = deque(maxlen=boom_candles + 5)

        # --- State for resampled data ---
        self.ema_closes_4h: Deque[float] = deque(maxlen=cfg.EMA_SLOW_PERIOD + 2)
        self.hr_data: Deque[dict] = deque(maxlen=self.rsi_period + 20) # Stores {'high', 'low', 'close'}
        self.day_closes: Deque[float] = deque(maxlen=cfg.STRUCTURAL_TREND_DAYS + 2)

        # --- Indicator State ---
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.atr = 0.0 # ATR will be calculated on the 1h timeframe with RSI/ADX
        self.rsi = 50.0
        self.adx = 0.0
        self.ret_30d = 0.0

        # --- Internal state for incremental calculations ---
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._adx_state = (0.0, 0.0, 0.0, 0.0) # prev_adx, prev_plus, prev_minus, prev_tr

    async def warm_up(self):
        """
        Fetch initial historical data and calculate baseline indicators.
        This is where all `await` calls belong.
        """
        LOG.info("Warming up indicators for %s...", self.symbol)

        try:
            
            ema_ohlc_list = await self.exchange.fetch_ohlcv(self.symbol, self.ema_tf, since=None, limit=500)
            hr_ohlc_list = await self.exchange.fetch_ohlcv(self.symbol, self.rsi_tf, since=None, limit=500)
            day_ohlc_list = await self.exchange.fetch_ohlcv(self.symbol, "1d", since=None, limit=100)
            ohlcv_5m_list = await self.exchange.fetch_ohlcv(self.symbol, cfg.TIMEFRAME, since=None, limit=500)

            # --- Pre-flight check: Do we have enough data? ---
            min_bars_required = self.adx_period * 2
            if not hr_ohlc_list or len(hr_ohlc_list) < min_bars_required:
                LOG.warning(
                    f"Skipping {self.symbol}: Not enough historical data on {self.rsi_tf} timeframe. "
                    f"Required >{min_bars_required} bars, but found only {len(hr_ohlc_list) if hr_ohlc_list else 0}."
                )
                return None

            # --- Convert to Pandas DataFrames ---
            df_ema = pd.DataFrame(ema_ohlc_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_hr = pd.DataFrame(hr_ohlc_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_day = pd.DataFrame(day_ohlc_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # ── Strip the still‑forming candle (volume == 0 or NaNs) ─────────────────────
            for _df in (df_ema, df_hr, df_day):
                if _df.iloc[-1]['volume'] == 0 or _df.iloc[-1].isna().any():
                    _df.drop(index=_df.index[-1], inplace=True)

            # ── Calculate indicators and pick the last **non‑NaN** value ────────────────
            rsi_series = ta.rsi(df_hr['close'], self.rsi_period).dropna()
            atr_series = ta.atr(df_hr,            self.atr_period).dropna()
            adx_series = ta.adx(df_hr,            self.adx_period).dropna()

            self.rsi = rsi_series.iloc[-1] if not rsi_series.empty else np.nan
            self.atr = atr_series.iloc[-1] if not atr_series.empty else np.nan
            self.adx = adx_series.iloc[-1] if not adx_series.empty else np.nan

            # --- Calculate Indicators ---
            self.ema_fast = ta.ema(df_ema['close'], cfg.EMA_FAST_PERIOD).iloc[-1]
            self.ema_slow = ta.ema(df_ema['close'], cfg.EMA_SLOW_PERIOD).iloc[-1]
            
            self.rsi = ta.rsi(df_hr['close'], self.rsi_period).iloc[-1]
            self.atr = ta.atr(df_hr, self.atr_period).iloc[-1]
            self.adx = ta.adx(df_hr, self.adx_period).iloc[-1]

            if pd.isna(self.rsi) or pd.isna(self.atr) or pd.isna(self.adx):
                LOG.warning(f"Indicator calculation resulted in NaN for {self.symbol}. Skipping.")
                return None

            # --- Calculate 30-day return ---
            if len(df_day) > cfg.STRUCTURAL_TREND_DAYS:
                price_30d_ago = df_day['close'].iloc[-cfg.STRUCTURAL_TREND_DAYS]
                current_price = df_day['close'].iloc[-1]
                self.ret_30d = (current_price / price_30d_ago - 1) if price_30d_ago else 0.0

            # --- Populate deques for incremental updates ---
            self.ema_closes_4h.extend(df_ema['close'].tail(cfg.EMA_SLOW_PERIOD + 2))
            self.hr_data.extend(df_hr.tail(self.rsi_period + 20).to_dict('records'))
            self.day_closes.extend(df_day['close'].tail(cfg.STRUCTURAL_TREND_DAYS + 2))
            self.price_history.extend([c[4] for c in ohlcv_5m_list])
            self.last_processed_timestamp = ohlcv_5m_list[-1][0]

        except ccxt.BadSymbol as e:
            LOG.warning("Could not warm up %s: Invalid symbol on exchange. Skipping. Error: %s", self.symbol, e)
            return None
        except Exception as e:
            LOG.error("An unexpected error occurred during warm up for %s: %s", self.symbol, e)
            traceback.print_exc()
            return None

        self.is_warmed_up = True
        LOG.info("Signal generator for %s is warmed up. ATR=%.8f, RSI=%.2f, ADX=%.2f", self.symbol, self.atr, self.rsi, self.adx)
        
        return ohlcv_5m_list

    def update_and_check(self, candle: list) -> Optional[Signal]:
        """
        Updates indicators with a new 5m candle, resampling where necessary, and checks for a signal.
        """
        if not self.is_warmed_up or not candle:
            return None

        timestamp, _, high, low, close, _ = candle
        if timestamp <= self.last_processed_timestamp:
            return None

        prev_ts = self.last_processed_timestamp
        self.last_processed_timestamp = timestamp

        # --- Update raw 5m price history for boom/bust check ---
        self.price_history.append(close)

        # --- Check for timeframe boundaries to update resampled indicators ---

        # 4-hour boundary
        if timestamp // (4 * 3600 * 1000) > prev_ts // (4 * 3600 * 1000):
            self.ema_closes_4h.append(close)
            self.ema_fast = ta.next_ema(close, self.ema_fast, cfg.EMA_FAST_PERIOD)
            self.ema_slow = ta.next_ema(close, self.ema_slow, cfg.EMA_SLOW_PERIOD)

        # 1-hour boundary
        if timestamp // (3600 * 1000) > prev_ts // (3600 * 1000):
            prev_hr_candle = self.hr_data[-1]
            self.hr_data.append({'high': high, 'low': low, 'close': close})
            
            # Update RSI
            self.rsi, self._avg_gain, self._avg_loss = ta.next_rsi(
                close, prev_hr_candle['close'], self._avg_gain, self._avg_loss, self.rsi_period
            )
            # Update ATR
            self.atr = ta.next_atr(
                self.atr, high, low, close, prev_hr_candle['close'], self.atr_period
            )
            # Update ADX
            self.adx, self._adx_state = ta.next_adx(
                high, low, close, 
                prev_hr_candle['high'], prev_hr_candle['low'], prev_hr_candle['close'],
                self._adx_state, self.adx_period
            )

        # Daily boundary (for 30-day return)
        if timestamp // (24 * 3600 * 1000) > prev_ts // (24 * 3600 * 1000):
            self.day_closes.append(close)
            if len(self.day_closes) > cfg.STRUCTURAL_TREND_DAYS:
                price_30d_ago = self.day_closes[0]
                self.ret_30d = (close / price_30d_ago - 1) if price_30d_ago else 0.0

        # --- Check Entry Conditions ---
        hours_per_candle = _get_hours_from_timeframe(cfg.TIMEFRAME)
        boom_periods = int(cfg.PRICE_BOOM_PERIOD_H / hours_per_candle)
        if len(self.price_history) < boom_periods + 1:
            return None
        price_then = self.price_history[-boom_periods - 1]
        price_boom = (close / price_then - 1) > cfg.PRICE_BOOM_PCT if price_then > 0 else False

        slowdown_periods = int(cfg.PRICE_SLOWDOWN_PERIOD_H / hours_per_candle)
        if len(self.price_history) < slowdown_periods + 1:
            return None
        price_recent = self.price_history[-slowdown_periods - 1]
        price_slowdown = abs(close / price_recent - 1) < cfg.PRICE_SLOWDOWN_PCT if price_recent > 0 else False

        c3_ema_down = self.ema_fast < self.ema_slow

        LOG.debug(
            "%s check: Boom=%s, Slowdown=%s, EMA_Down=%s, ATR=%.4f, RSI=%.2f, ADX=%.2f",
            self.symbol, price_boom, price_slowdown, c3_ema_down, self.atr, self.rsi, self.adx
        )

        if price_boom and price_slowdown and c3_ema_down:
            LOG.info("SIGNAL FOUND for %s at price %.4f", self.symbol, close)
            return Signal(
                symbol=self.symbol,
                entry=float(close),
                atr=float(self.atr),
                rsi=float(self.rsi),
                ret_30d=float(self.ret_30d),
                adx=float(self.adx),
            )
        return None