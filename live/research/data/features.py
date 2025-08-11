# research/data/features.py
import asyncio
import aiohttp
import logging
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

BYBIT_BASE = "https://api.bybit.com"

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _ms(ts: pd.Timestamp) -> int:
    """Convert pandas Timestamp to milliseconds since epoch (UTC)."""
    ts = pd.to_datetime(ts, utc=True)
    return int(ts.value // 10**6)

async def _fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    params: Dict,
    retries: int = 3,
    backoff: float = 0.5,
) -> Dict:
    """GET JSON with simple retries + exponential backoff."""
    last_err: Optional[str] = None
    for i in range(retries):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status == 200:
                    return await resp.json()
                last_err = f"HTTP {resp.status} :: {await resp.text()}"
        except Exception as e:
            last_err = str(e)
        await asyncio.sleep(backoff * (2 ** i))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")

# --------------------------------------------------------------------------------------
# Bybit V5 PUBLIC endpoints (no auth needed)
# Docs:
#  - Funding history:  GET /v5/market/history-fund-rate
#  - Open interest:    GET /v5/market/open-interest
# Key params:
#  - category: 'linear' or 'inverse'
#  - symbol: e.g., 'BTCUSDT'
#  - startTime/endTime (ms)
#  - intervalTime for OI: one of ['5min','15min','30min','1h','4h','1d']
# --------------------------------------------------------------------------------------
# References:
# Funding history docs + API explorer pages. :contentReference[oaicite:2]{index=2}
# Open interest docs + API explorer pages.   :contentReference[oaicite:3]{index=3}
# --------------------------------------------------------------------------------------

async def fetch_funding_for_window(
    session: aiohttp.ClientSession,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    category: str = "linear",
) -> Optional[float]:
    """
    Return the latest funding rate (float) whose timestamp <= end, querying within [start, end].
    If none available, returns None.
    Note: Bybit’s funding interval can vary by symbol; we simply take the last known rate before entry. :contentReference[oaicite:4]{index=4}
    """
    url = f"{BYBIT_BASE}/v5/market/history-fund-rate"
    params = {
        "category": category,
        "symbol": symbol,
        "startTime": _ms(start),
        "endTime": _ms(end),
        "limit": 200,  # max per docs
    }
    try:
        js = await _fetch_json(session, url, params)
        rows = js.get("result", {}).get("list", [])
        if not rows:
            return None
        # Sort by Bybit's funding timestamp and take the latest within window
        rows.sort(key=lambda r: int(r.get("fundingRateTimestamp", 0)))
        return float(rows[-1]["fundingRate"])
    except Exception as e:
        LOG.warning("Funding fetch/parse error for %s: %s", symbol, e)
        return None

async def fetch_oi_features_for_window(
    session: aiohttp.ClientSession,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str = "1h",
    category: str = "linear",
) -> Dict[str, Optional[float]]:
    """
    Pull open interest series for the window and compute:
      - oi_last: last OI level in the window
      - oi_delta_pct_win: (OI_last / OI_first - 1) over the window (if >=2 points)
    interval must be one of ['5min','15min','30min','1h','4h','1d'] per docs. :contentReference[oaicite:5]{index=5}
    """
    url = f"{BYBIT_BASE}/v5/market/open-interest"
    params = {
        "category": category,
        "symbol": symbol,
        "intervalTime": interval,
        "startTime": _ms(start),
        "endTime": _ms(end),
        "limit": 200,  # max per docs
    }
    try:
        js = await _fetch_json(session, url, params)
        rows = js.get("result", {}).get("list", [])
        if not rows:
            return {"oi_last": None, "oi_delta_pct_win": None}

        # Sort ascending by timestamp, ensure floats
        rows.sort(key=lambda r: int(r.get("timestamp", 0)))
        vals = np.array([float(r["openInterest"]) for r in rows], dtype=float)
        oi_last = float(vals[-1])
        if len(vals) >= 2 and vals[0] != 0:
            delta = float(vals[-1] / vals[0] - 1.0)
        else:
            delta = None
        return {"oi_last": oi_last, "oi_delta_pct_win": delta}
    except Exception as e:
        LOG.warning("Open-interest fetch/parse error for %s: %s", symbol, e)
        return {"oi_last": None, "oi_delta_pct_win": None}

async def enrich_funding_oi_features(
    df: pd.DataFrame,
    window_hours: float = 8.0,
    oi_interval: str = "1h",
    category: str = "linear",
    concurrency: int = 16,
) -> pd.DataFrame:
    """
    For each trade row (must have 'symbol' and 'opened_at'), fetch:
      - funding_last_at_entry
      - oi_last_at_entry
      - oi_delta_pct_win   (over the window)
    Returns a new DataFrame with added columns. Uses limited concurrency to be nice to the API.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out["opened_at"] = pd.to_datetime(out["opened_at"], utc=True)

    starts: List[pd.Timestamp] = []
    ends:   List[pd.Timestamp] = []
    for ts in out["opened_at"]:
        end = pd.to_datetime(ts, utc=True)
        start = end - pd.to_timedelta(window_hours, unit="h")
        starts.append(start)
        ends.append(end)

    sem = asyncio.Semaphore(concurrency)

    async def _one_row(sym: str, st: pd.Timestamp, en: pd.Timestamp):
        async with sem:
            async with aiohttp.ClientSession(headers={"User-Agent": "livefader-research/1.0"}) as session:
                f_task = fetch_funding_for_window(session, sym, st, en, category=category)
                oi_task = fetch_oi_features_for_window(session, sym, st, en, interval=oi_interval, category=category)
                f_val, oi_vals = await asyncio.gather(f_task, oi_task, return_exceptions=True)

                funding_val = None if isinstance(f_val, Exception) else f_val
                if isinstance(oi_vals, Exception):
                    oi_last = None
                    oi_delta = None
                else:
                    oi_last = oi_vals.get("oi_last")
                    oi_delta = oi_vals.get("oi_delta_pct_win")
                return funding_val, oi_last, oi_delta

    results = await asyncio.gather(
        *[_one_row(sym, st, en) for sym, st, en in zip(out["symbol"], starts, ends)],
        return_exceptions=False,
    )

    fundings, oi_last_list, oi_delta_list = [], [], []
    for funding_val, oi_last, oi_delta in results:
        fundings.append(funding_val)
        oi_last_list.append(oi_last)
        oi_delta_list.append(oi_delta)

    out["funding_last_at_entry"] = fundings
    out["oi_last_at_entry"] = oi_last_list
    out["oi_delta_pct_win"] = oi_delta_list
    return out

# --------------------------------------------------------------------------------------
# Static, entry-time feature engineering from existing DB columns (no look-ahead)
# --------------------------------------------------------------------------------------

def derive_static_entry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer a few features from the columns you already store at entry time.
    - ema_spread_pct_at_entry
    - is_ema_crossed_down_at_entry
    Also casts common numeric entry columns safely.
    """
    out = df.copy()

    # numeric casts (robust to missing)
    numeric_cols = [
        "rsi_at_entry", "adx_at_entry",
        "price_boom_pct_at_entry", "price_slowdown_pct_at_entry",
        "vwap_z_at_entry", "ema_fast_at_entry", "ema_slow_at_entry",
        "listing_age_days_at_entry", "day_of_week_at_entry", "hour_of_day_at_entry",
        "eth_macdhist_at_entry",
        "vwap_stack_frac_at_entry", "vwap_stack_expansion_pct_at_entry", "vwap_stack_slope_pph_at_entry",
    ]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # EMA spread (%)
    if {"ema_fast_at_entry", "ema_slow_at_entry"}.issubset(out.columns):
        fast = out["ema_fast_at_entry"].astype(float)
        slow = out["ema_slow_at_entry"].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["ema_spread_pct_at_entry"] = np.where(slow != 0.0, (fast - slow) / slow, 0.0)
    else:
        out["ema_spread_pct_at_entry"] = 0.0

    # EMA cross flag
    if {"ema_fast_at_entry", "ema_slow_at_entry"}.issubset(out.columns):
        out["is_ema_crossed_down_at_entry"] = (out["ema_fast_at_entry"] < out["ema_slow_at_entry"]).astype(int)
    else:
        out["is_ema_crossed_down_at_entry"] = 0

    # Categorical/session cols: leave as-is (the modeling pipeline will encode)
    # Ensure opened_at is datetime for downstream joins/ordering
    if "opened_at" in out.columns:
        out["opened_at"] = pd.to_datetime(out["opened_at"], utc=True, errors="coerce")

    return out
