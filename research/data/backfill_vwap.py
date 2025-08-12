# research/data/backfill_vwap.py
import os, asyncio, logging, aiohttp, asyncpg, pandas as pd, numpy as np
from datetime import timezone
from typing import Optional, Tuple
from live.indicators import vwap_stack_features  # reuse your implementation

LOG = logging.getLogger("backfill_vwap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BYBIT_BASE = "https://api.bybit.com"

def _interval_to_str(tf: str | int) -> str:
    # bybit v5 uses minutes: "1","3","5","15","60","240","D","W","M" etc.
    if isinstance(tf, int):
        return str(tf)
    return tf

async def fetch_klines(session: aiohttp.ClientSession, symbol: str, interval: str, limit: int, end_ms: int) -> pd.DataFrame:
    """
    Pull OHLCV from Bybit v5 public API up to end_ms (inclusive).
    Endpoint: /v5/market/kline
    """
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": str(limit),
        "end": str(end_ms),
    }
    url = f"{BYBIT_BASE}/v5/market/kline"
    for attempt in range(3):
        try:
            async with session.get(url, params=params, timeout=30) as r:
                if r.status != 200:
                    txt = await r.text()
                    raise RuntimeError(f"HTTP {r.status} :: {txt[:200]}")
                js = await r.json()
                if js.get("retCode") != 0:
                    raise RuntimeError(f"API retCode {js.get('retCode')} :: {js.get('retMsg')}")
                rows = js["result"]["list"]  # [start,open,high,low,close,volume,turnover]
                rows = sorted(rows, key=lambda x: int(x[0]))
                df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume","turnover"])
                df = df[["ts","open","high","low","close","volume"]].copy()
                df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
        except Exception as e:
            if attempt == 2:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))
    return pd.DataFrame()

async def _backfill_one(pool: asyncpg.Pool, session: aiohttp.ClientSession, pid: int, symbol: str, opened_at) -> Tuple[int, bool, str]:
    """
    Compute VWAP-stack at entry and UPDATE that row in DB.
    Uses its own pooled connection to avoid 'another operation is in progress'.
    """
    try:
        interval = _interval_to_str(5)
        ts = pd.Timestamp(opened_at)
        # normalize tz → UTC
        if ts.tzinfo is None or ts.tz is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)
        end_ms = int(ts.timestamp() * 1000)

        df5 = await fetch_klines(session, symbol, interval=interval, limit=300, end_ms=end_ms)
        if df5.empty or len(df5) < 50:
            return pid, False, "insufficient_klines"

        # Compute features with your live helper
        feat = vwap_stack_features(
            df5["close"].values.astype(float),
            df5["volume"].values.astype(float),
            lookback_bars=int(os.getenv("VWAP_STACK_LOOKBACK_BARS", "12")),
            band_pct=float(os.getenv("VWAP_STACK_BAND_PCT", "0.004")),
            dt_index=df5["ts"],
        )
        frac = float(feat.get("occupancy_frac", np.nan))
        exp  = float(feat.get("expansion_pct", np.nan))
        slope= float(feat.get("slope_pph", np.nan))

        # Write with a dedicated connection from the pool
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE positions
                   SET vwap_stack_frac_at_entry=$1,
                       vwap_stack_expansion_pct_at_entry=$2,
                       vwap_stack_slope_pph_at_entry=$3
                 WHERE id=$4
                """,
                frac, exp, slope, pid
            )
        return pid, True, "ok"
    except Exception as e:
        return pid, False, f"error: {e!s}"

async def main():
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        raise RuntimeError("Set PG_DSN to your Postgres DSN")
    workers = int(os.getenv("VWAP_BACKFILL_WORKERS", "8"))

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=max(4, workers))
    # Fetch all rows once, then fan out
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, symbol, opened_at
              FROM positions
             WHERE opened_at IS NOT NULL
               AND (vwap_stack_frac_at_entry IS NULL
                 OR vwap_stack_expansion_pct_at_entry IS NULL
                 OR vwap_stack_slope_pph_at_entry IS NULL)
            ORDER BY opened_at
            """
        )
    if not rows:
        LOG.info("Nothing to backfill. All rows already have VWAP-stack fields.")
        await pool.close()
        return

    ok_count = 0
    skip_count = 0
    sem = asyncio.Semaphore(workers)

    async def worker(row):
        nonlocal ok_count, skip_count
        async with sem:
            pid, success, msg = await _backfill_one(pool, session, row["id"], row["symbol"], row["opened_at"])
            LOG.info("pid=%s → %s (%s)", pid, "UPDATED" if success else "SKIPPED", msg)
            if success: ok_count += 1
            else: skip_count += 1

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(worker(r) for r in rows))

    LOG.info("Done. UPDATED=%d  SKIPPED=%d", ok_count, skip_count)
    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
