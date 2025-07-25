# live/signal_generator.py
from dataclasses import dataclass
from typing import Optional, Deque
from collections import deque
import config as cfg
import logging
from . import live_indicators as ta

LOG = logging.getLogger(__name__)

@dataclass
class Signal:
    symbol: str
    entry: float
    atr: float
    rsi: float

class SignalGenerator:

    RSI_PERIOD = 14
    ATR_PERIOD = 14

    def __init__(self, symbol: str, exchange):
        self.symbol = symbol
        self.exchange = exchange
        self.last_processed_timestamp = 0

        # State for indicators (can be simple values or deques)
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.atr = 0.0
        self.rsi = 50.0

        # --- NEW: Deque for historical price data ---
        # We need enough data for the longest lookback (PRICE_BOOM_PERIOD_H)
        # The 4h timeframe means we need (PRICE_BOOM_PERIOD_H / 4) candles.
        boom_candles = int(cfg.PRICE_BOOM_PERIOD_H / 4)
        self.price_history: Deque[float] = deque(maxlen=boom_candles + 5) # Add buffer

        # For ATR and RSI calculations
        indicator_deque_len = max(self.RSI_PERIOD, self.ATR_PERIOD) + 1
        self.highs: Deque[float] = deque(maxlen=indicator_deque_len)
        self.lows: Deque[float] = deque(maxlen=indicator_deque_len)
        self.closes: Deque[float] = deque(maxlen=indicator_deque_len)

        self.is_warmed_up = False

    async def warm_up(self):
        """Fetch initial historical data to calculate baseline indicators."""
        LOG.info("Warming up indicators for %s...", self.symbol)
        try:
            # Fetch enough data for all lookback periods
            limit = max(cfg.EMA_SLOW + 1, self.price_history.maxlen, self.closes.maxlen)
            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, '4h', limit=limit)

            if len(ohlcv) < cfg.EMA_SLOW + 1:
                LOG.warning("Not enough historical data to warm up %s. Disabling.", self.symbol)
                return

            # Populate all deques and calculate initial indicator values
            all_closes = [c[4] for c in ohlcv]
            self.price_history.extend(all_closes)
            self.highs.extend([c[2] for c in ohlcv])
            self.lows.extend([c[3] for c in ohlcv])
            self.closes.extend(all_closes)

            self.ema_fast = ta.ema_from_list(all_closes, cfg.EMA_FAST_PERIOD)
            self.ema_slow = ta.ema_from_list(all_closes, cfg.EMA_SLOW_PERIOD)
            self.atr = ta.atr_from_deques(self.highs, self.lows, self.closes, self.ATR_PERIOD)
            self.rsi = ta.rsi_from_deque(self.closes, self.RSI_PERIOD)

            self.last_processed_timestamp = ohlcv[-1][0]
            self.is_warmed_up = True
            LOG.info("Signal generator for %s is warmed up. ATR=%.4f, RSI=%.2f", self.symbol, self.atr, self.rsi)
        except Exception as e:
            LOG.error("Error during warm-up for %s: %s", self.symbol, e)

    def update_and_check(self, candle: list) -> Optional[Signal]:
        """
        Updates indicators with a new closed candle and checks for a signal.
        This is the core logic.
        """
        if not self.is_warmed_up or not candle:
            return None

        timestamp, _, high, low, close, _ = candle

        # Avoid processing the same candle twice
        if timestamp <= self.last_processed_timestamp:
            return None

        # --- Update deques with new data ---
        self.price_history.append(close)
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)

        # --- Incrementally or fully update indicators ---
        self.ema_fast = ta.next_ema(close, self.ema_fast, cfg.EMA_FAST_PERIOD)
        self.ema_slow = ta.next_ema(close, self.ema_slow, cfg.EMA_SLOW_PERIOD)
        self.atr = ta.atr_from_deques(self.highs, self.lows, self.closes, self.ATR_PERIOD)
        self.rsi = ta.rsi_from_deque(self.closes, self.RSI_PERIOD)

        self.last_processed_timestamp = timestamp

        # --- Check Entry Conditions (from your original scout.py) ---
        # C1: Price "Boom" Condition
        boom_periods = int(cfg.PRICE_BOOM_PERIOD_H / 4)
        if len(self.price_history) < boom_periods + 1:
            return None
        price_then = self.price_history[-boom_periods - 1]
        price_boom = (close / price_then - 1) > cfg.PRICE_BOOM_PCT if price_then > 0 else False

        # C2: Price "Slowdown" Condition
        slowdown_periods = int(cfg.PRICE_SLOWDOWN_PERIOD_H / 4)
        if len(self.price_history) < slowdown_periods + 1:
            return None
        price_recent = self.price_history[-slowdown_periods - 1]
        price_slowdown = abs(close / price_recent - 1) < cfg.PRICE_SLOWDOWN_PCT if price_recent > 0 else False

        # C3: EMA cross condition
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