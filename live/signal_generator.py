# live/signal_generator.py
from dataclasses import dataclass
from typing import Optional, Deque
from collections import deque
import config as cfg
import logging
import re
from live import live_indicators as ta

LOG = logging.getLogger(__name__)

@dataclass
class Signal:
    symbol: str
    entry: float
    atr: float
    rsi: float

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

    RSI_PERIOD = 14
    ATR_PERIOD = 14

    def __init__(self, symbol: str, exchange):
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

        self.is_warmed_up = False

    async def warm_up(self):
        """Fetch initial historical data to calculate baseline indicators."""
        LOG.info("Warming up indicators for %s...", self.symbol)
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
            )
        return None