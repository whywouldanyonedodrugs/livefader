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
    tf = str(tf).lower()
    if tf.endswith("m"):
        return tf.replace("m","")
    if tf == "4h": return "240"
    if tf == "1h": return "60"
    if tf == "5":  return "5"
    return tf  # last resort

async def fetch_klines(session: aiohttp.ClientSession, symbol: str, interval="5", limit=300, end_ms: Optional[int]=None):
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": str(limit),
    }
    if end_ms is not None:
        params["end"] = str(end_ms)
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
                rows = js["result"]["list"]  # list of lists: [start,open,high,low,close,volume,turnover]
                rows = sorted(rows, key=lambda x: int(x[0]))
                # to DataFrame (matching your indicator expectations)
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

async def _backfill_one(conn: asyncpg.Connection, session: aiohttp.ClientSession, pid: int, symbol: str, opened_at) -> Tuple[int, bool, str]:
    try:
        interval = _interval_to_str(5)
        end_ms = int(pd.Timestamp(opened_at).tz_convert(timezone.utc).timestamp() * 1000)
        df5 = await fetch_klines(session, symbol, interval=interval, limit=300, end_ms=end_ms)
        if df5.empty or len(df5) < 40:
            return pid, False, "insufficient_klines"

        lookback = int(os.environ.get("VWAP_STACK_LOOKBACK_BARS", "12"))
        band_pct = float(os.environ.get("VWAP_STACK_BAND_PCT", "0.004"))

        feat = vwap_stack_features(df5.rename(columns={"ts":"timestamp"}), lookback_bars=lookback, band_pct=band_pct)
        frac = float(feat.get("vwap_frac_in_band", np.nan))
        exp  = float(feat.get("vwap_expansion_pct", np.nan))  # already percent (0.35 → 0.35%)
        slope = float(feat.get("vwap_slope_pph", np.nan))

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
        return pid, False, f"error: {e}"

async def main():
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        raise RuntimeError("Set PG_DSN to your Postgres DSN")
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
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
        return

    async with aiohttp.ClientSession() as session:
        async with pool.acquire() as conn:
            tasks = [ _backfill_one(conn, session, r["id"], r["symbol"], r["opened_at"]) for r in rows ]
            for fut in asyncio.as_completed(tasks):
                pid, ok, msg = await fut
                LOG.info("pid=%s → %s (%s)", pid, "UPDATED" if ok else "SKIPPED", msg)

    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
