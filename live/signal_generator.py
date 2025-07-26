# live/signal_generator.py
from dataclasses import dataclass
from typing import Optional, Deque
from collections import deque
import config as cfg
import logging
import re
from live import live_indicators as ta
from exchange_proxy import fetch_ohlcv_paginated

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

        self.ema_tf      = cfg.EMA_TIMEFRAME
        self.rsi_tf      = cfg.RSI_TIMEFRAME
        self.adx_tf      = cfg.ADX_TIMEFRAME
        self.rsi_period  = cfg.RSI_PERIOD
        self.atr_period  = cfg.ADX_PERIOD     # ATR still 14 unless overridden
        self.adx_period  = cfg.ADX_PERIOD

        self.symbol = symbol
        self.exchange = exchange
        self.last_processed_timestamp = 0

        # State for indicators
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.atr = 0.0
        self.rsi = 50.0
        # --- NEW: State for smoothed RSI ---
        self.avg_gain = 0.0
        self.avg_loss = 0.0

        hours_per_candle = _get_hours_from_timeframe(cfg.TIMEFRAME)
        boom_candles = int(cfg.PRICE_BOOM_PERIOD_H / hours_per_candle)
        
        self.price_history: Deque[float] = deque(maxlen=boom_candles + 5)

        indicator_deque_len = max(self.RSI_PERIOD, self.ATR_PERIOD) + 2 # Need prev_close
        self.highs: Deque[float] = deque(maxlen=indicator_deque_len)
        self.lows: Deque[float] = deque(maxlen=indicator_deque_len)
        self.closes: Deque[float] = deque(maxlen=indicator_deque_len)

        self.ema_closes_4h: Deque[float] = deque(maxlen=cfg.EMA_SLOW_PERIOD + 2)
        self.hr_closes:     Deque[float] = deque(maxlen=self.rsi_period + 2)
        self.day_closes:    Deque[float] = deque(maxlen=cfg.STRUCTURAL_TREND_DAYS + 2)

        # 4‑hour closes for EMAs
        ema_ohlc = await fetch_ohlcv_paginated(self.exchange, self.symbol,
                                               self.ema_tf, cfg.EMA_SLOW_PERIOD + 5)
        self.ema_closes_4h.extend([c[4] for c in ema_ohlc])
        self.ema_fast = ta.ema_from_list(self.ema_closes_4h, cfg.EMA_FAST_PERIOD)
        self.ema_slow = ta.ema_from_list(self.ema_closes_4h, cfg.EMA_SLOW_PERIOD)

        # 1‑hour closes for RSI / ADX
        hr_ohlc = await fetch_ohlcv_paginated(self.exchange, self.symbol,
                                              self.rsi_tf, self.rsi_period + 20)
        self.hr_closes.extend([c[4] for c in hr_ohlc])
        highs  = [c[2] for c in hr_ohlc]
        lows   = [c[3] for c in hr_ohlc]
        self.adx, self._adx_state = ta.initial_adx(highs, lows, self.hr_closes, self.adx_period)
        self.rsi, self.avg_gain, self.avg_loss = ta.initial_rsi(self.hr_closes, self.rsi_period)

        # daily closes for 30‑day return
        day_ohlc = await fetch_ohlcv_paginated(self.exchange, self.symbol,
                                               "1d", cfg.STRUCTURAL_TREND_DAYS + 2)
        self.day_closes.extend([c[4] for c in day_ohlc])

        self.is_warmed_up = False

    async def warm_up(self):
        """Fetch initial historical data to calculate baseline indicators."""
        LOG.info("Warming up indicators for %s...", self.symbol)

        wanted = max(
            cfg.EMA_SLOW_PERIOD + 1,
            self.price_history.maxlen,
            self.closes.maxlen,
        )

        ohlcv = await fetch_ohlcv_paginated(
            self.exchange,
            self.symbol,
            cfg.TIMEFRAME,
            wanted,
        )

        if len(ohlcv) < wanted:
            LOG.warning("Not enough historical data to warm up %s. Disabling.", self.symbol)
            return

        try:
            limit = max(cfg.EMA_SLOW_PERIOD + 1, self.price_history.maxlen, self.closes.maxlen)
            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, cfg.TIMEFRAME, limit=limit)

            if len(ohlcv) < cfg.EMA_SLOW_PERIOD + 1:
                LOG.warning("Not enough historical data to warm up %s. Disabling.", self.symbol)
                return

            all_closes = [c[4] for c in ohlcv]
            all_highs = [c[2] for c in ohlcv]
            all_lows = [c[3] for c in ohlcv]

            self.price_history.extend(all_closes)
            self.highs.extend(all_highs)
            self.lows.extend(all_lows)
            self.closes.extend(all_closes)

            self.ema_fast = ta.ema_from_list(all_closes, cfg.EMA_FAST_PERIOD)
            self.ema_slow = ta.ema_from_list(all_closes, cfg.EMA_SLOW_PERIOD)
            
            # Use initial calculation methods and store state
            self.atr = ta.initial_atr(all_highs, all_lows, all_closes, self.ATR_PERIOD)
            self.rsi, self.avg_gain, self.avg_loss = ta.initial_rsi(all_closes, self.RSI_PERIOD)

            self.last_processed_timestamp = ohlcv[-1][0]
            self.is_warmed_up = True
            LOG.info("Signal generator for %s is warmed up. ATR=%.4f, RSI=%.2f", self.symbol, self.atr, self.rsi)
        except Exception as e:
            LOG.error("Error during warm-up for %s: %s", self.symbol, e)

    def update_and_check(self, candle: list) -> Optional[Signal]:
        """
        Updates indicators with a new closed candle and checks for a signal.
        """

        if timestamp // (4*60*60*1000) > self.last_processed_timestamp // (4*60*60*1000):
            self.ema_closes_4h.append(close)
            self.ema_fast = ta.next_ema(close, self.ema_fast, cfg.EMA_FAST_PERIOD)
            self.ema_slow = ta.next_ema(close, self.ema_slow, cfg.EMA_SLOW_PERIOD)

        # 1‑hour boundary
        if timestamp // (60*60*1000) > self.last_processed_timestamp // (60*60*1000):
            prev_hr_close = self.hr_closes[-1] if self.hr_closes else close
            self.hr_closes.append(close)
            self.rsi, self.avg_gain, self.avg_loss = ta.next_rsi(
                close, prev_hr_close, self.avg_gain, self.avg_loss, self.rsi_period)
            self.adx, self._adx_state = ta.next_adx(
                high, low, close, prev_hr_close, self._adx_state, self.adx_period)

        # daily boundary (for 30‑day return)
        if timestamp // (24*60*60*1000) > self.last_processed_timestamp // (24*60*60*1000):
            self.day_closes.append(close)

        ret_30d = 0.0
        if len(self.day_closes) > cfg.STRUCTURAL_TREND_DAYS:
            price_30d_ago = self.day_closes[-cfg.STRUCTURAL_TREND_DAYS-1]
            ret_30d = (close / price_30d_ago - 1) if price_30d_ago else 0.0

        if not self.is_warmed_up or not candle:
            return None

        timestamp, _, high, low, close, _ = candle

        if timestamp <= self.last_processed_timestamp:
            return None

        prev_close = self.closes[-1] if self.closes else close

        self.price_history.append(close)
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)

        # Incrementally update indicators using previous state
        self.ema_fast = ta.next_ema(close, self.ema_fast, cfg.EMA_FAST_PERIOD)
        self.ema_slow = ta.next_ema(close, self.ema_slow, cfg.EMA_SLOW_PERIOD)
        self.atr = ta.next_atr(self.atr, high, low, close, prev_close, self.ATR_PERIOD)
        self.rsi, self.avg_gain, self.avg_loss = ta.next_rsi(
            close, prev_close, self.avg_gain, self.avg_loss, self.RSI_PERIOD
        )

        self.last_processed_timestamp = timestamp

        # Check Entry Conditions
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
            "%s check: Boom=%s, Slowdown=%s, EMA_Down=%s, ATR=%.4f, RSI=%.2f",
            self.symbol, price_boom, price_slowdown, c3_ema_down, self.atr, self.rsi
        )

        if price_boom and price_slowdown and c3_ema_down:
            LOG.info("SIGNAL FOUND for %s at price %.4f", self.symbol, close)
        
            return Signal(
                symbol=self.symbol,
                entry=float(close),
                atr=float(self.atr),
                rsi=float(self.rsi),
                ret_30d=float(ret_30d),
                adx=float(self.adx),
            )

        return None