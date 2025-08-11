# /opt/livefader/src/enrich_data.py

import asyncio
import asyncpg
import logging
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
from tqdm import tqdm
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger(__name__)

# --- Counterfactual Parameters (defaults) ---
LOOK_FORWARD_HOURS = 4
COUNTERFACTUAL_TP_MULT = 2.0   # e.g., 2.0x ATR
COUNTERFACTUAL_SL_MULT = 2.5   # e.g., 2.5x ATR

# --- VWAP-stack defaults (RVWAP over 5m) ---
VWAP_LOOKBACK_BARS = 12        # ~1h on 5m bars
VWAP_BAND_PCT = 0.004          # ±0.40%

# ----- Helpers -----

def _ensure_env() -> str:
    load_dotenv()
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL not found in environment.")
    return dsn

async def _fetch_ohlcv(exchange, symbol: str, tf: str, limit: int, end_ts_ms: int):
    """
    Fetch candles up to (and including) end_ts_ms, then trim to limit bars ending at that time.
    """
    data = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=max(limit, 200))
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df[df["ts"] <= end_ts_ms].tail(limit).reset_index(drop=True)
    return df

def _vwap_stack_features(df5: pd.DataFrame, lookback: int, band_pct: float) -> dict:
    """
    Rolling VWAP (tp*vol / vol) over 'lookback', shifted for consolidation; last bar for expansion.
    Returns dict with vwap_frac_in_band / vwap_expansion_pct / vwap_slope_pph.
    """
    if df5.empty or len(df5) < lookback + 2:
        return {"vwap_frac_in_band": 0.0, "vwap_expansion_pct": 0.0, "vwap_slope_pph": 0.0}

    df5 = df5.copy()
    tp = (df5["high"] + df5["low"] + df5["close"]) / 3.0
    tpv = (tp * df5["volume"]).rolling(lookback).sum()
    vv = df5["volume"].rolling(lookback).sum()

    # Current rolling VWAP (unshifted) for expansion
    cur_vwap = (tpv / vv).iloc[-1]
    cur_close = float(df5["close"].iloc[-1])
    expansion = abs(cur_close / cur_vwap - 1.0) if cur_vwap and np.isfinite(cur_vwap) else 0.0

    # Prior window RVWAP for consolidation (exclude current bar)
    rvwap = (tpv / vv).shift(1)
    vwap_prior = rvwap.iloc[-lookback:-0].values
    closes_prior = df5["close"].iloc[-(lookback+1):-1].astype(float).values

    band_hi = vwap_prior * (1 + band_pct)
    band_lo = vwap_prior * (1 - band_pct)
    in_band = (closes_prior >= band_lo) & (closes_prior <= band_hi)
    frac = float(in_band.mean()) if len(in_band) else 0.0

    # Simple slope proxy (percent-per-hour over ~lookback)
    k = min(lookback, 12)
    vsub = (tpv.iloc[-k:] / vv.iloc[-k:]).values
    slope = (vsub[-1] - vsub[0]) / vsub[0] if (len(vsub) >= 2 and vsub[0]) else 0.0
    slope_pph = float(slope * (60 / 5) / k)

    return {
        "vwap_frac_in_band": float(frac),
        "vwap_expansion_pct": float(expansion),
        "vwap_slope_pph": float(slope_pph),
    }

# ----- Tasks -----

async def enrich_counterfactuals(conn: asyncpg.Connection, exchange):
    """
    Enrich CLOSED trades that lack cf_* fields using a 4h look-forward on 1m bars.
    """
    query = """
        SELECT * FROM positions
        WHERE status='CLOSED' AND cf_would_hit_tp_1x_atr IS NULL
        ORDER BY id
    """
    trades = await conn.fetch(query)
    if not trades:
        LOG.info("No new trades found to enrich (counterfactuals).")
        return

    LOG.info("Enriching %d trades with counterfactuals...", len(trades))
    for trade in tqdm(trades, desc="CF"):
        try:
            trade_id = trade["id"]
            symbol = trade["symbol"]
            closed_at = trade["closed_at"]
            if closed_at and closed_at.tzinfo is None:
                closed_at = closed_at.replace(tzinfo=timezone.utc)
            entry_price = float(trade["entry_price"])
            atr_at_entry = float(trade["atr"])
            size = float(trade["size"])

            cf_tp_price_1x = entry_price - (atr_at_entry * 1.0)
            cf_tp_price_2x = entry_price - (atr_at_entry * COUNTERFACTUAL_TP_MULT)
            cf_sl_price_2_5x = entry_price + (atr_at_entry * COUNTERFACTUAL_SL_MULT)

            since_ts = int((closed_at or datetime.now(timezone.utc)).timestamp() * 1000)
            limit = LOOK_FORWARD_HOURS * 60

            ohlcv_future = await exchange.fetch_ohlcv(symbol, "1m", since=since_ts, limit=limit)
            if not ohlcv_future:
                LOG.warning("No future OHLCV for trade %s (%s). Skipping.", trade_id, symbol)
                continue
            df_future = pd.DataFrame(ohlcv_future, columns=["ts", "open", "high", "low", "close", "volume"])

            would_hit_tp_1x = (df_future["low"] <= cf_tp_price_1x).any()
            would_hit_tp_2x = (df_future["low"] <= cf_tp_price_2x).any()
            would_hit_sl_2_5x = (df_future["high"] >= cf_sl_price_2_5x).any()

            max_adverse_price = float(df_future["high"].max())
            max_favorable_price = float(df_future["low"].min())

            cf_mae_usd = (max_adverse_price - entry_price) * size
            cf_mfe_usd = (entry_price - max_favorable_price) * size

            cf_mae_over_atr = (cf_mae_usd / size) / atr_at_entry if atr_at_entry > 0 else 0.0
            cf_mfe_over_atr = (cf_mfe_usd / size) / atr_at_entry if atr_at_entry > 0 else 0.0

            await conn.execute(
                """
                UPDATE positions SET
                    cf_would_hit_tp_2x_atr = $1,
                    cf_would_hit_sl_2_5x_atr = $2,
                    cf_mae_over_atr_4h = $3,
                    cf_mfe_over_atr_4h = $4,
                    cf_would_hit_tp_1x_atr = $5
                WHERE id = $6
                """,
                bool(would_hit_tp_2x),
                bool(would_hit_sl_2_5x),
                float(cf_mae_over_atr),
                float(cf_mfe_over_atr),
                bool(would_hit_tp_1x),
                trade_id,
            )
        except Exception as e:
            LOG.error("CF enrich failed for id=%s %s: %s", trade["id"], trade["symbol"], e, exc_info=True)
            continue
        await asyncio.sleep(0.25)

    LOG.info("Counterfactual enrichment complete.")

async def backfill_vwap_stack(conn: asyncpg.Connection, exchange, lookback_bars=VWAP_LOOKBACK_BARS, band_pct=VWAP_BAND_PCT):
    """
    Backfill VWAP-stack features at entry for trades missing them.
    """
    rows = await conn.fetch(
        """
        SELECT id, symbol, opened_at
        FROM positions
        WHERE (vwap_stack_frac_at_entry IS NULL
            OR vwap_stack_expansion_pct_at_entry IS NULL
            OR vwap_stack_slope_pph_at_entry IS NULL)
        ORDER BY id
        """
    )
    if not rows:
        LOG.info("No positions need VWAP-stack backfill.")
        return

    LOG.info("Backfilling VWAP-stack fields for %d positions...", len(rows))
    for r in tqdm(rows, desc="VWAP"):
        pid = r["id"]
        sym = r["symbol"]
        opened_at = r["opened_at"]
        if opened_at and opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        end_ts_ms = int((opened_at or datetime.now(timezone.utc)).timestamp() * 1000)

        try:
            df5 = await _fetch_ohlcv(exchange, sym, "5m", max(200, lookback_bars + 20), end_ts_ms)
            if df5.empty:
                continue
            vw = _vwap_stack_features(df5, lookback_bars, band_pct)
            await conn.execute(
                """
                UPDATE positions
                SET vwap_stack_frac_at_entry=$2,
                    vwap_stack_expansion_pct_at_entry=$3,
                    vwap_stack_slope_pph_at_entry=$4
                WHERE id=$1
                """,
                pid,
                float(vw["vwap_frac_in_band"]),
                float(vw["vwap_expansion_pct"]),
                float(vw["vwap_slope_pph"]),
            )
        except Exception as e:
            LOG.error("VWAP backfill failed for id=%s %s: %s", pid, sym, e)
            continue
        await asyncio.sleep(0.1)

    LOG.info("VWAP-stack backfill complete.")

# ----- CLI entrypoint -----

async def _amain(args):
    dsn = _ensure_env()
    conn = await asyncpg.connect(dsn=dsn)
    exchange = ccxt.bybit({"options": {"defaultType": "future"}, "enableRateLimit": True})
    try:
        if args.counterfactuals:
            await enrich_counterfactuals(conn, exchange)
        if args.vwap_stack:
            await backfill_vwap_stack(conn, exchange, args.vwap_lookback, args.vwap_band)
    finally:
        await exchange.close()
        await conn.close()

def main():
    ap = argparse.ArgumentParser(description="Enrich positions with counterfactuals and VWAP-stack")
    ap.add_argument("--counterfactuals", action="store_true", help="Run counterfactual enrichment")
    ap.add_argument("--vwap-stack", action="store_true", help="Run VWAP-stack backfill at entry")
    ap.add_argument("--vwap-lookback", type=int, default=VWAP_LOOKBACK_BARS, help="VWAP lookback bars (5m)")
    ap.add_argument("--vwap-band", type=float, default=VWAP_BAND_PCT, help="VWAP band pct (e.g., 0.004 for ±0.4%)")
    args = ap.parse_args()

    if not args.counterfactuals and not args.vwap_stack:
        # default: run both
        args.counterfactuals = True
        args.vwap_stack = True

    asyncio.run(_amain(args))

if __name__ == "__main__":
    main()
