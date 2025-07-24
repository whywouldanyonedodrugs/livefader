"""
filters.py – v2.0
Rolling‑VWAP “gap” veto + small tidy‑ups
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Tuple

import config as cfg   # needs GAP_MAX_DEV_PCT, FAST_EMA_LEN, SLOW_EMA_LEN, etc.

# --------------------------------------------------------------------------- #
#  Runtime caches                                                             
# --------------------------------------------------------------------------- #
class Runtime:
    """
    Holds live market snapshots for cheap veto checks.

    Added in v2.0
    ----------------
    * 5‑minute rolling VWAP buffer per symbol
    """

    # ─────────────────────────── init ──────────────────────────
    def __init__(self) -> None:
        self._last_px: Dict[str, float] = {}
        self._ema_fast: Dict[str, float] = {}
        self._ema_slow: Dict[str, float] = {}
        self._oi: Dict[str, float] = {}

        # Each entry: deque[Tuple[ts, px×vol, vol]]
        self._vwap_buf: Dict[str, Deque[Tuple[datetime, float, float]]] = defaultdict(
            lambda: deque(maxlen=300)  # ≈5 min at one update / sec
        )

    # ─────────────────────────── updates ──────────────────────
    def update_ticker(self, symbol: str, price: float, volume: float | None = None) -> None:
        """Feed from the price/volume poller.  
        `volume` is **per‑tick** volume (or any small‑granularity bucket);
        pass `None` if you don’t have it – VWAP will simply ignore that sample.
        """
        self._last_px[symbol] = price

        # --- Fast / slow EMA (simple α‑EMA) --------------------
        def _ema(prev: float | None, x: float, α: float) -> float:
            return x if prev is None else prev + α * (x - prev)

        k_fast = 2 / (cfg.FAST_EMA_LEN + 1)
        k_slow = 2 / (cfg.SLOW_EMA_LEN + 1)
        self._ema_fast[symbol] = _ema(self._ema_fast.get(symbol), price, k_fast)
        self._ema_slow[symbol] = _ema(self._ema_slow.get(symbol), price, k_slow)

        # --- Rolling 5‑min VWAP --------------------------------
        if volume:
            now = datetime.utcnow()
            self._vwap_buf[symbol].append((now, price * volume, volume))
            cut = now - timedelta(minutes=5)
            while self._vwap_buf[symbol] and self._vwap_buf[symbol][0][0] < cut:
                self._vwap_buf[symbol].popleft()

    def update_open_interest(self, symbol: str, oi: float) -> None:
        self._oi[symbol] = oi

    # ------------------------------------------------------------------ #
    #  Accessors                                                         #
    # ------------------------------------------------------------------ #
    def vwap(self, symbol: str) -> float | None:
        buf = self._vwap_buf[symbol]
        if not buf:
            return None
        tot_pv = sum(pv for _ts, pv, _v in buf)
        tot_v = sum(v for _ts, _pv, v in buf)
        return tot_pv / tot_v if tot_v else None

    def last_price(self, symbol: str) -> float | None:
        return self._last_px.get(symbol)

    def emas(self, symbol: str) -> tuple[float | None, float | None]:
        return self._ema_fast.get(symbol), self._ema_slow.get(symbol)

    def open_int(self, symbol: str) -> float | None:
        return self._oi.get(symbol)


# --------------------------------------------------------------------------- #
#  Veto logic                                                                
# --------------------------------------------------------------------------- #
def evaluate(
    sig, rt: Runtime, *, open_positions: int, equity: float
) -> tuple[bool, List[str]]:
    """
    Central clearing‑house for all run‑time vetoes.

    Returns
    -------
    ok : bool
        True ⇒ trade may proceed.
    vetoes : list[str]
        Tags for anything that blocked the trade (useful for audit).
    """
    vetoes: List[str] = []
    ok = True

    # --------‑‑ VWAP gap (NEW) -------------------------------------------
    last = rt.last_price(sig.symbol)
    vwap = rt.vwap(sig.symbol)
    if last is not None and vwap is not None:
        gap = abs(last - vwap) / vwap
        if gap > cfg.GAP_MAX_DEV_PCT:
            vetoes.append("GAP")
            ok = False

    # --------‑‑ Fast/slow EMA slope (existing) ---------------------------
    ema_fast, ema_slow = rt.emas(sig.symbol)
    if ema_fast and ema_slow and ema_fast > ema_slow:
        vetoes.append("EMA_UPTREND_BTC")  # example tag
        ok = False

    # --------‑‑ Open‑interest confirmation (existing) --------------------
    oi_now = rt.open_int(sig.symbol)
    if oi_now and oi_now < sig.oi_baseline * cfg.OI_MIN_FACTOR:
        vetoes.append("OI_WEAK")
        ok = False

    # --------‑‑ Portfolio caps etc. (unchanged) --------------------------
    if open_positions >= cfg.MAX_OPEN:
        vetoes.append("MAX_OPEN")
        ok = False
    if equity < cfg.MIN_EQUITY_USDT:
        vetoes.append("LOW_EQUITY")
        ok = False

    return ok, vetoes
