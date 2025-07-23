# filters.py
"""
Dynamic, back‑tester–parity veto filters for live trading.

How to use:
    runtime = Runtime()                 # 1️⃣  created once in LiveTrader.__init__
    runtime.update_ticker("BTCUSDT",  ...)   # 2️⃣  updated by ticker websockets
    runtime.update_open_interest("SOLUSDT", ...)
    can_trade, tags = evaluate(signal, runtime, open_positions, equity)
"""

from __future__ import annotations
from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Deque, Optional

import config as cfg   # ← unified strategy settings

# --------------------------------------------------------------------------- #
# 0 ▸ helpers                                                                  #
# --------------------------------------------------------------------------- #
def _ema(price: float, prev: Optional[float], period: int) -> float:
    alpha = 2 / (period + 1)
    return price if prev is None else price * alpha + prev * (1 - alpha)


# --------------------------------------------------------------------------- #
# 1 ▸ runtime cache fed by websockets                                          #
# --------------------------------------------------------------------------- #
class Runtime:
    """Keeps the rolling state (EMAs, OI deque) needed by the filters."""
    def __init__(self) -> None:
        self.btc_close = self.alt_close = None
        self.btc_fast_ema = self.btc_slow_ema = None
        self.alt_fast_ema = self.alt_slow_ema = None
        self.oi_cache: Dict[str, Deque[Tuple[datetime, float]]] = defaultdict(
            lambda: deque(maxlen=cfg.OI_LOOKBACK_PERIOD_BARS + 1)
        )

    # ─── market data updaters ─────────────────────────────────────────────── #
    def update_ticker(self, symbol: str, price: float) -> None:
        if symbol == "BTCUSDT":
            self.btc_close = price
            self.btc_fast_ema = _ema(price, self.btc_fast_ema, cfg.BTC_FAST_EMA_PERIOD)
            self.btc_slow_ema = _ema(price, self.btc_slow_ema, cfg.BTC_SLOW_EMA_PERIOD)
        elif symbol == cfg.ALT_SYMBOL:
            self.alt_close = price
            self.alt_fast_ema = _ema(price, self.alt_fast_ema, cfg.ALT_FAST_EMA_PERIOD)
            self.alt_slow_ema = _ema(price, self.alt_slow_ema, cfg.ALT_SLOW_EMA_PERIOD)

    def update_open_interest(self, symbol: str, open_interest: float) -> None:
        self.oi_cache[symbol].append((datetime.now(timezone.utc), open_interest))

    # ─── convenient snapshot for evaluate() ──────────────────────────────── #
    def snapshot(self) -> Dict[str, float | None]:
        return {
            "btc_close": self.btc_close,
            "btc_fast_ema": self.btc_fast_ema,
            "btc_slow_ema": self.btc_slow_ema,
            "alt_close": self.alt_close,
            "alt_fast_ema": self.alt_fast_ema,
            "alt_slow_ema": self.alt_slow_ema,
        }


# --------------------------------------------------------------------------- #
# 2 ▸ main filter function                                                    #
# --------------------------------------------------------------------------- #
FilterResult = Tuple[bool, List[str]]       # (pass?, failed_tags)

def evaluate(
    sig,                                       # live_trader.Signal
    rt: Runtime,                               # shared runtime object
    open_positions: int,
    equity: float,
) -> FilterResult:
    """
    Returns (True, [])  if the trade is allowed, otherwise (False, ["TAG_A", …])
    Implements the same logic as the back‑tester vetoes.
    """
    tags: List[str] = []

    # ── PORTFOLIO size ───────────────────────────────────────────────────── #
    if open_positions >= cfg.MAX_OPEN:
        tags.append("PORTFOLIO")

    # ── BTC & ALT EMA trend filters ─────────────────────────────────────── #
    s = rt.snapshot()  # shorthand
    if cfg.BTC_FAST_FILTER_ENABLED and (s["btc_close"] is None or s["btc_close"] > s["btc_fast_ema"]):
        tags.append("BTC_FAST")
    if cfg.BTC_SLOW_FILTER_ENABLED and (s["btc_close"] is None or s["btc_close"] > s["btc_slow_ema"]):
        tags.append("BTC_SLOW")
    if cfg.ALT_FAST_FILTER_ENABLED and (s["alt_close"] is None or s["alt_close"] > s["alt_fast_ema"]):
        tags.append("ALT_FAST")
    if cfg.ALT_SLOW_FILTER_ENABLED and (s["alt_close"] is None or s["alt_close"] > s["alt_slow_ema"]):
        tags.append("ALT_SLOW")

    # ── MIN_STOP and NOTIONAL guards ─────────────────────────────────────── #
    stop_dist = cfg.SL_ATR_MULT * sig.atr     # distance (USD) for a short trade
    if stop_dist < cfg.MIN_STOP_DIST_USD or stop_dist / sig.entry < cfg.MIN_STOP_DIST_PCT:
        tags.append("MIN_STOP")

    risk_cash = equity * cfg.RISK_PCT
    notional  = risk_cash / stop_dist * sig.entry
    if not (cfg.MIN_NOTIONAL <= notional <= equity * cfg.MAX_LEVERAGE):
        tags.append("NOTIONAL")

    # ── Open‑Interest confirmation ───────────────────────────────────────── #
    if cfg.OI_FILTER_ENABLED:
        dq = rt.oi_cache[sig.symbol]
        if len(dq) < cfg.OI_LOOKBACK_PERIOD_BARS:
            tags.append("OI_CONFIRM")
        else:
            past = dq[0][1]
            now  = dq[-1][1]
            if past > 0:
                change_pct = (now / past - 1) * 100
                if not (cfg.OI_MIN_CHANGE_PCT <= change_pct <= cfg.OI_MAX_CHANGE_PCT):
                    tags.append("OI_CONFIRM")
            else:
                tags.append("OI_CONFIRM")

    # ── BV‑rule (boom bucket × volume spike) ─────────────────────────────── #
    if cfg.BV_RULE_ENABLED:
        bucket = sig.boom_bucket - 1  # boom_bucket is 1‑based
        if 0 <= bucket < len(cfg.BV_MIN_VOL_SPIKE):
            min_v, max_v = cfg.BV_MIN_VOL_SPIKE[bucket], cfg.BV_MAX_VOL_SPIKE[bucket]
            if not (min_v <= sig.vol_spike_24h <= max_v):
                tags.append("BV_RULE")

    return (len(tags) == 0, tags)
