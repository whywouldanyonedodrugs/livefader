"""
filters.py – v2.1
Adds 5‑min VWAP “GAP” veto, coin‑age veto, and keeps fast/slow EMA + OI checks.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, date
from typing import Deque, Dict, List, Tuple, Optional

import config as cfg

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

        # 5‑min rolling VWAP buffer: deque[(ts, px*vol, vol)]
        self._vwap_buf: Dict[str, Deque[Tuple[datetime, float, float]]] = defaultdict(
            lambda: deque(maxlen=300)
        )

        # listing dates (UTC)
        self._listing_dates: Dict[str, date] = {}

    # ── update hooks ──────────────────────────────────────────────────────
    def update_ticker(self, symbol: str, price: float, volume: Optional[float] = None) -> None:
        self._last_px[symbol] = price

        k_fast = 2 / (cfg.FAST_EMA_LEN + 1)
        k_slow = 2 / (cfg.SLOW_EMA_LEN + 1)
        self._ema_fast[symbol] = (
            price if symbol not in self._ema_fast else self._ema_fast[symbol] + k_fast * (price - self._ema_fast[symbol])
        )
        self._ema_slow[symbol] = (
            price if symbol not in self._ema_slow else self._ema_slow[symbol] + k_slow * (price - self._ema_slow[symbol])
        )

        if volume:
            now = datetime.utcnow()
            self._vwap_buf[symbol].append((now, price * volume, volume))
            cut = now - timedelta(minutes=5)
            while self._vwap_buf[symbol] and self._vwap_buf[symbol][0][0] < cut:
                self._vwap_buf[symbol].popleft()

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
        buf = self._vwap_buf[symbol]
        if not buf:
            return None
        pv_sum = sum(pv for _ts, pv, _v in buf)
        v_sum = sum(v for _ts, _pv, v in buf)
        return pv_sum / v_sum if v_sum else None

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

    # ── 1. VWAP gap check ───────────────────────────────────────────────
    last = rt.last_price(sig.symbol)
    vwap = rt.vwap(sig.symbol)
    if last is not None and vwap is not None:
        if abs(last - vwap) / vwap > cfg.GAP_MAX_DEV_PCT:
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

    # ── 3. Fast > slow EMA trend filter (BTC/ALT) ──────────────────────
    ema_fast, ema_slow = rt.emas(sig.symbol)
    if ema_fast and ema_slow and ema_fast > ema_slow:
        vetoes.append("EMA_UPTREND")
        ok = False

    # ── 4. Portfolio / equity caps ──────────────────────────────────────
    if open_positions >= cfg.MAX_OPEN:
        vetoes.append("MAX_OPEN")
        ok = False
    if equity < cfg.MIN_EQUITY_USDT:
        vetoes.append("LOW_EQUITY")
        ok = False

    return ok, vetoes
