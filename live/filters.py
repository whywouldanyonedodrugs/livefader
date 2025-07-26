"""
filters.py – v2.2 (VWAP Backtester Alignment)
Aligns VWAP filter with backtester logic by using a bar-based rolling window
and checking for sustained consolidation.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, date
from typing import Deque, Dict, List, Tuple, Optional
import re

import config as cfg

# Helper function to parse timeframe string to minutes
def _timeframe_to_minutes(tf: str) -> int:
    """Converts timeframe string like '5m', '1h' to minutes."""
    match = re.match(r"(\d+)(\w)", tf)
    if not match:
        return 1  # Default to 1 minute if format is unexpected
    
    val, unit = int(match.group(1)), match.group(2).lower()
    
    if unit == 'm':
        return val
    if unit == 'h':
        return val * 60
    if unit == 'd':
        return val * 24 * 60
    return 1

# ════════════════════════════════════════════════════════════════════════════
# Runtime cache
# ════════════════════════════════════════════════════════════════════════════
class Runtime:
    """Cheap per‑symbol market cache."""

    def __init__(self) -> None:
        self._last_px: Dict[str, float] = {}
        self._ema_fast: Dict[str, float] = {}
        self._ema_slow: Dict[str, float] = {}
        self._oi: Dict[str, float] = {}

        tf_minutes = _timeframe_to_minutes(cfg.TIMEFRAME)
        vwap_bars = int((cfg.GAP_VWAP_HOURS * 60) / tf_minutes) if tf_minutes > 0 else 0

        self._vwap_len = vwap_bars

        # Buffer for VWAP calculation: deque[(px*vol, vol)]
        self._vwap_calc_buf: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=vwap_bars)
        )

        # Buffer to track the gap condition over the last N bars
        self._vwap_gap_ok_buf: Dict[str, Deque[bool]] = defaultdict(
            lambda: deque(maxlen=cfg.GAP_MIN_BARS)
        )

        # listing dates (UTC)
        self._listing_dates: Dict[str, date] = {}

    # ── update hooks ──────────────────────────────────────────────────────
    def update_ticker(self, symbol: str, price: float, volume: Optional[float] = None) -> None:
        self._last_px[symbol] = price

        k_fast = 2 / (cfg.EMA_FAST_PERIOD + 1)
        k_slow = 2 / (cfg.EMA_SLOW_PERIOD + 1)
        self._ema_fast[symbol] = (
            price if symbol not in self._ema_fast else self._ema_fast[symbol] + k_fast * (price - self._ema_fast[symbol])
        )
        self._ema_slow[symbol] = (
            price if symbol not in self._ema_slow else self._ema_slow[symbol] + k_slow * (price - self._ema_slow[symbol])
        )

        if volume and volume > 0:
            # ----- dynamic window length: recreate if config changed -----
            if self._vwap_calc_buf[symbol].maxlen != self._vwap_len:
                self._vwap_calc_buf[symbol] = deque(self._vwap_calc_buf[symbol],
                                                    maxlen=self._vwap_len)
            # -------------------------------------------------------------

            # 1. Update the buffer for VWAP calculation. `maxlen` handles the rolling window.
            self._vwap_calc_buf[symbol].append((price * volume, volume))

            # 2. Calculate the current VWAP (it will be None until the buffer is full).
            current_vwap = self.vwap(symbol)

            # 3. Check the gap condition and update the history buffer.
            if current_vwap is not None:
                deviation = abs(price - current_vwap) / current_vwap
                is_ok = deviation <= cfg.GAP_MAX_DEV_PCT
                self._vwap_gap_ok_buf[symbol].append(is_ok)

    def update_open_interest(self, symbol: str, oi: float) -> None:
        self._oi[symbol] = oi

    # ── listing‑date helpers ──────────────────────────────────────────────
    def set_listing_date(self, symbol: str, dt: Optional[date]) -> None:
        if dt:
            self._listing_dates[symbol] = dt

    def listing_age_days(self, symbol: str) -> Optional[int]:
        if symbol not in self._listing_dates:
            return None
        return (datetime.utcnow().date() - self._listing_dates[symbol]).days

    # ── query helpers ─────────────────────────────────────────────────────
    def vwap(self, symbol: str) -> Optional[float]:

        buf = self._vwap_calc_buf.get(symbol)
        if not buf or len(buf) < max(cfg.GAP_MIN_BARS + 1, 2):
            return None
        
        # Only calculate if the buffer is full, matching the backtester's rolling window behavior.
        if not buf or len(buf) < buf.maxlen:
            return None
            
        pv_slice = list(buf)[:-1]
        pv_sum = sum(pv for pv, _ in pv_slice)
        v_sum = sum(v  for _,  v in pv_slice)
        return pv_sum / v_sum if v_sum else None

    def is_vwap_gap_consolidated(self, symbol: str) -> bool:
        """Checks if the gap condition has been true for the required number of bars."""
        buf = self._vwap_gap_ok_buf.get(symbol)
        # Condition is met if the buffer is full and all values within it are True.
        if not buf or len(buf) < buf.maxlen:
            return False
        return all(buf)

    def last_price(self, symbol: str) -> Optional[float]:
        return self._last_px.get(symbol)

    def emas(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        return self._ema_fast.get(symbol), self._ema_slow.get(symbol)

    def open_int(self, symbol: str) -> Optional[float]:
        return self._oi.get(symbol)

# ════════════════════════════════════════════════════════════════════════════
# Veto gate
# ════════════════════════════════════════════════════════════════════════════
def evaluate(
    sig, rt: Runtime, *, open_positions: int, equity: float
) -> Tuple[bool, List[str]]:
    """Return (ok?, [tags])."""
    vetoes: List[str] = []
    ok = True

    # ── 1. VWAP gap check (MODIFIED) ────────────────────────────────────────
    # This now checks if the gap has been small for the last `GAP_MIN_BARS`, matching the backtester.
    if not rt.is_vwap_gap_consolidated(sig.symbol):
        vetoes.append("GAP")
        ok = False

    # ── 2. Coin‑age veto ────────────────────────────────────────────────
    age = rt.listing_age_days(sig.symbol)
    if age is not None:
        if age < cfg.MIN_COIN_AGE_DAYS:
            vetoes.append("AGE_TOO_NEW")
            ok = False
        elif age > cfg.MAX_COIN_AGE_DAYS:
            vetoes.append("AGE_TOO_OLD")
            ok = False

    # ── 3. Primary RSI window ───────────────────────────────────────────
    if not (cfg.RSI_ENTRY_MIN <= sig.rsi <= cfg.RSI_ENTRY_MAX):
        vetoes.append("RSI_RANGE")
        ok = False

    # ── 4. 30‑day structural‑trend veto ─────────────────────────────────
    if sig.ret_30d is not None and sig.ret_30d > cfg.STRUCTURAL_TREND_RET_PCT:
        vetoes.append("STRUCTURAL_TREND")
        ok = False

    # ── 5. Optional ADX trend‑strength veto ─────────────────────────────
    if cfg.ADX_FILTER_ENABLED:
        if not (cfg.ADX_MIN <= sig.adx <= cfg.ADX_MAX):
            vetoes.append("ADX")
            ok = False

    # ── 6. Portfolio / equity caps ──────────────────────────────────────
    if open_positions >= cfg.MAX_OPEN:
        vetoes.append("MAX_OPEN")
        ok = False
    if equity < getattr(cfg, "MIN_EQUITY_USDT", 0):
        vetoes.append("LOW_EQUITY")
        ok = False

    return ok, vetoes