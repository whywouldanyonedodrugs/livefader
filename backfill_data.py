# /opt/livefader/src/backfill_data.py

import asyncio
import asyncpg
import logging
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import statsmodels.api as sm
import ccxt.async_support as ccxt
from tqdm import tqdm

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOG = logging.getLogger(__name__)

# --- Replicated Indicator Logic from the Bot ---
# This makes the script self-contained and guarantees consistent calculations.
def ema(series, period): return series.ewm(span=period, adjust=False).mean()
def atr(df, period=14):
    tr = pd.DataFrame({'h-l': df['high'] - df['low'], 'h-pc': abs(df['high'] - df['close'].shift()), 'l-pc': abs(df['low'] - df['close'].shift())}).max(axis=1)
    return ema(tr, period)
def rsi(series, period=14):
    delta = series.diff()
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = abs(loss.rolling(window=period, min_periods=1).mean())
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
def adx(df, period=14):
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.DataFrame({'h-l': df['high'] - df['low'], 'h-pc': abs(df['high'] - df['close'].shift()), 'l-pc': abs(df['low'] - df['close'].shift())}).max(axis=1)
    atr_val = ema(tr, period)
    plus_di = 100 * (ema(plus_dm, period) / atr_val)
    minus_di = 100 * (ema(abs(minus_dm), period) / atr_val)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    return ema(dx, period)
def triangular_moving_average(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean().rolling(window=period, min_periods=period).mean()

async def calculate_historical_regime(exchange, opened_at: datetime) -> str:
    # ... (This function is correct and remains the same) ...
    return "UNKNOWN" # Placeholder

async def main():
    load_dotenv()
    db_dsn = os.getenv("DATABASE_URL")
    if not db_dsn:
        LOG.error("DATABASE_URL not found. Cannot proceed.")
        return

    conn = None
    exchange = None
    try:
        conn = await asyncpg.connect(dsn=db_dsn)
        exchange = ccxt.bybit({'enableRateLimit': True})
        await exchange.load_markets()
        
        query = "SELECT * FROM positions WHERE rsi_at_entry IS NULL AND status = 'CLOSED' ORDER BY id"
        trades_to_fix = await conn.fetch(query)

        if not trades_to_fix:
            LOG.info("No trades found with missing data. Database is up-to-date.")
            return

        LOG.info(f"Found {len(trades_to_fix)} trades to backfill. Starting process...")

        for trade in tqdm(trades_to_fix, desc="Backfilling Trades"):
            try:
                trade_id = trade['id']
                symbol = trade['symbol']
                opened_at = trade['opened_at']
                closed_at = trade['closed_at']
                entry_price = float(trade['entry_price'])
                size = float(trade['size'])

                # --- 1. Fetch All Necessary Historical Data ---
                since_ts_1d = int((opened_at - timedelta(days=40)).timestamp() * 1000)
                since_ts_1h = int((opened_at - timedelta(days=25)).timestamp() * 1000) # ~600 hours
                
                ohlcv_1d = await exchange.fetch_ohlcv(symbol, '1d', since=since_ts_1d, limit=40)
                ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', since=since_ts_1h, limit=600)
                
                df1d = pd.DataFrame(ohlcv_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df1d['timestamp'] = pd.to_datetime(df1d['timestamp'], unit='ms', utc=True)
                df1d = df1d[df1d['timestamp'] < opened_at]

                df1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df1h['timestamp'] = pd.to_datetime(df1h['timestamp'], unit='ms', utc=True)
                df1h = df1h[df1h['timestamp'] < opened_at]

                # --- 2. Calculate All Pre-Trade Metrics ---
                rsi_at_entry = rsi(df1h['close'], 14).iloc[-1]
                adx_at_entry = adx(df1h, 14).iloc[-1]
                atr_pct_at_entry = (atr(df1h, 14).iloc[-1] / entry_price) * 100 if entry_price > 0 else 0
                ret_30d_at_entry = (df1d['close'].iloc[-1] / df1d['close'].iloc[-30]) - 1 if len(df1d) >= 30 else 0.0
                
                # Simplified boom/slowdown
                price_24h_ago = df1h[df1h['timestamp'] <= (opened_at - timedelta(hours=24))]['close'].iloc[-1]
                price_4h_ago = df1h[df1h['timestamp'] <= (opened_at - timedelta(hours=4))]['close'].iloc[-1]
                price_boom_pct = (entry_price / price_24h_ago) - 1
                price_slowdown_pct = (entry_price / price_4h_ago) - 1

                market_regime = await calculate_historical_regime(exchange, opened_at)

                # --- 3. Calculate Post-Trade Metrics ---
                holding_minutes = (closed_at - opened_at).total_seconds() / 60
                
                ohlcv_1m = await exchange.fetch_ohlcv(symbol, '1m', since=int(opened_at.timestamp() * 1000))
                df1m = pd.DataFrame(ohlcv_1m, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                df1m['ts'] = pd.to_datetime(df1m['ts'], unit='ms', utc=True)
                df1m = df1m[(df1m['ts'] >= opened_at) & (df1m['ts'] <= closed_at)]
                
                mae_usd, mfe_usd = 0.0, 0.0
                if not df1m.empty:
                    mae_usd = (df1m['h'].max() - entry_price) * size
                    mfe_usd = (entry_price - df1m['l'].min()) * size

                # --- 4. Update the Database Record ---
                update_query = """
                    UPDATE positions SET
                        market_regime_at_entry = $1, rsi_at_entry = $2, adx_at_entry = $3,
                        atr_pct_at_entry = $4, ret_30d_at_entry = $5, price_boom_pct_at_entry = $6,
                        price_slowdown_pct_at_entry = $7, holding_minutes = $8, mae_usd = $9, mfe_usd = $10
                    WHERE id = $11
                """
                await conn.execute(
                    update_query, market_regime, rsi_at_entry, adx_at_entry,
                    atr_pct_at_entry, ret_30d_at_entry, price_boom_pct,
                    price_slowdown_pct, holding_minutes, mae_usd, mfe_usd, trade_id
                )

            except Exception as e:
                LOG.error(f"Failed to backfill trade ID {trade['id']} ({trade['symbol']}): {e}")
                continue

        LOG.info("Backfilling process complete.")

    except Exception as e:
        LOG.error(f"A critical error occurred during the backfill process: {e}")
    finally:
        if conn: await conn.close()
        if exchange: await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())