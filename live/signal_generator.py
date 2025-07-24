# live/signal_generator.py
from dataclasses import dataclass
from typing import Optional
import config as cfg
import logging 
# Assume you create this file with simple, list/deque-based math
from . import live_indicators as ta 

LOG = logging.getLogger(__name__)

@dataclass
class Signal:
    symbol: str
    entry: float
    atr: float
    rsi: float

class SignalGenerator:
    def __init__(self, symbol: str, exchange):
        self.symbol = symbol
        self.exchange = exchange
        self.last_processed_timestamp = 0
        
        # State for indicators (can be simple values or deques)
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.atr = 0.0
        self.rsi = 0.0
        
        self.is_warmed_up = False

    async def warm_up(self):
        """Fetch initial historical data to calculate baseline indicators."""
        print(f"Warming up indicators for {self.symbol}...")
        try:
            # Fetch last 201 4h candles to have enough data for EMA_SLOW=200
            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, '4h', limit=cfg.EMA_SLOW + 1)
            if len(ohlcv) < cfg.EMA_SLOW + 1:
                print(f"WARNING: Not enough historical data to warm up {self.symbol}. Disabling.")
                return

            # Use a simple loop for initial calculation
            closes = [c[4] for c in ohlcv]
            self.ema_fast = sum(closes[-cfg.EMA_FAST:]) / cfg.EMA_FAST
            self.ema_slow = sum(closes[-cfg.EMA_SLOW:]) / cfg.EMA_SLOW
            
            # Store the timestamp of the most recent candle
            self.last_processed_timestamp = ohlcv[-1][0]
            self.is_warmed_up = True
            LOG.info("Signal generator for %s is warmed up.", self.symbol)
        except Exception as e:
            print(f"Error during warm-up for {self.symbol}: {e}")

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
        
        # --- Incrementally update indicators ---
        # A proper implementation would use live_indicators.py
        # For now, we'll use a simplified calculation
        k_fast = 2 / (cfg.EMA_FAST + 1)
        k_slow = 2 / (cfg.EMA_SLOW + 1)
        self.ema_fast = close * k_fast + self.ema_fast * (1 - k_fast)
        self.ema_slow = close * k_slow + self.ema_slow * (1 - k_slow)
        
        # NOTE: Real ATR/RSI would need more historical data (e.g., from a deque)
        # This is a placeholder for the logic.
        self.atr = high - low # Highly simplified placeholder
        self.rsi = 50 # Placeholder

        LOG.debug(
            "Updated %s. EMA Fast: %.2f, EMA Slow: %.2f", 
            self.symbol, self.ema_fast, self.ema_slow
        ) # Use DEBUG for frequent, verbose messages
        
        if c3_ema_down:
            LOG.info("SIGNAL FOUND for %s at price %.4f", self.symbol, close)

        self.last_processed_timestamp = timestamp

        # --- Check Entry Conditions (from your original scout.py) ---
        c3_ema_down = self.ema_fast < self.ema_slow
        # Add other conditions here (c1, c2, c4, etc.)
        # For this example, we only check the EMA cross
        
        if c3_ema_down:
            print(f"SIGNAL FOUND for {self.symbol} at price {close}")
            return Signal(
                symbol=self.symbol,
                entry=float(close),
                atr=float(self.atr),
                rsi=float(self.rsi),
            )
        return None