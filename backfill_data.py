# /opt/livefader/src/backfill_data_full.py

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
def ema(series, period): return series.ewm(span=period, adjust=False).mean()
def atr(df, period=14):
    tr = pd.DataFrame({'h-l': df['h'] - df['l'], 'h-pc': abs(df['h'] - df['c'].shift()), 'l-pc': abs(df['l'] - df['c'].shift())}).max(axis=1)
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
    plus_dm = df['h'].diff()
    minus_dm = df['l'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.DataFrame({'h-l': df['h'] - df['l'], 'h-pc': abs(df['h'] - df['c'].shift()), 'l-pc': abs(df['l'] - df['c'].shift())}).max(axis=1)
    atr_val = ema(tr, period)
    plus_di = 100 * (ema(plus_dm, period) / atr_val)
    minus_di = 100 * (ema(abs(minus_dm), period) / atr_val)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    return ema(dx, period)

async def calculate_historical_regime(exchange, opened_at: datetime) -> str:
    # This is a complex calculation, for now we will keep it simple
    # In a future version, we can replicate the full logic from the live bot
    return "BACKFILLED"

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
        
        query = """
            SELECT * FROM positions 
            WHERE status = 'CLOSED' AND (
                adx_at_entry IS NULL OR 
                exit_reason IS NULL OR 
                exit_reason IN ('UNKNOWN', 'FALLBACK', 'UNKNOWN_CLOSE')
            ) ORDER BY id
        """
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
                pnl = float(trade['pnl'])
                atr_at_entry_db = float(trade['atr'])

                inferred_exit_reason = 'TP' if pnl > 0 else 'SL'
                hour_of_day = opened_at.hour
                day_of_week = opened_at.weekday()
                session_tag = "ASIA" if 0 <= hour_of_day < 8 else "EUROPE" if 8 <= hour_of_day < 16 else "US"

                since_ts_4h = int((opened_at - timedelta(days=100)).timestamp() * 1000)
                since_ts_1h = int((opened_at - timedelta(days=25)).timestamp() * 1000)
                since_ts_5m = int((opened_at - timedelta(days=5)).timestamp() * 1000)

                ohlcv_4h, ohlcv_1h, ohlcv_5m = await asyncio.gather(
                    exchange.fetch_ohlcv(symbol, '4h', since=since_ts_4h, limit=600),
                    exchange.fetch_ohlcv(symbol, '1h', since=since_ts_1h, limit=600),
                    exchange.fetch_ohlcv(symbol, '5m', since=since_ts_5m, limit=1440)
                )

                df4h = pd.DataFrame(ohlcv_4h, columns=['ts', 'o', 'h', 'l', 'c', 'v']).assign(ts=lambda x: pd.to_datetime(x['ts'], unit='ms', utc=True)).set_index('ts').loc[:opened_at]
                df1h = pd.DataFrame(ohlcv_1h, columns=['ts', 'o', 'h', 'l', 'c', 'v']).assign(ts=lambda x: pd.to_datetime(x['ts'], unit='ms', utc=True)).set_index('ts').loc[:opened_at]
                df5m = pd.DataFrame(ohlcv_5m, columns=['ts', 'o', 'h', 'l', 'c', 'v']).assign(ts=lambda x: pd.to_datetime(x['ts'], unit='ms', utc=True)).set_index('ts').loc[:opened_at]

                adx_at_entry = adx(df1h, 14).iloc[-1]
                ema_fast_at_entry = ema(df4h['c'], 12).iloc[-1]
                ema_slow_at_entry = ema(df4h['c'], 26).iloc[-1]
                
                vwap_num = (df5m['c'] * df5m['v']).rolling(48).sum()
                vwap_den = df5m['v'].rolling(48).sum()
                vwap = (vwap_num / vwap_den).iloc[-1]
                vwap_dev_pct_at_entry = abs(entry_price - vwap) / vwap if vwap > 0 else 0.0

                ohlcv_1m_asset, ohlcv_1m_btc = await asyncio.gather(
                    exchange.fetch_ohlcv(symbol, '1m', since=int(opened_at.timestamp() * 1000)),
                    exchange.fetch_ohlcv('BTCUSDT', '1m', since=int(opened_at.timestamp() * 1000))
                )
                
                mae_over_atr, mfe_over_atr, btc_beta_during_trade = 0.0, 0.0, 0.0
                if ohlcv_1m_asset:
                    df1m_asset = pd.DataFrame(ohlcv_1m_asset, columns=['ts', 'o', 'h', 'l', 'c', 'v']).assign(ts=lambda x: pd.to_datetime(x['ts'], unit='ms', utc=True)).set_index('ts').loc[opened_at:closed_at]
                    if not df1m_asset.empty:
                        mae = (df1m_asset['h'].max() - entry_price)
                        mfe = (entry_price - df1m_asset['l'].min())
                        mae_over_atr = mae / atr_at_entry_db if atr_at_entry_db > 0 else 0
                        mfe_over_atr = mfe / atr_at_entry_db if atr_at_entry_db > 0 else 0

                if ohlcv_1m_asset and ohlcv_1m_btc:
                    df1m_btc = pd.DataFrame(ohlcv_1m_btc, columns=['ts', 'o', 'h', 'l', 'c', 'v']).assign(ts=lambda x: pd.to_datetime(x['ts'], unit='ms', utc=True)).set_index('ts').loc[opened_at:closed_at]
                    if not df1m_asset.empty and not df1m_btc.empty:
                        asset_returns = df1m_asset['c'].pct_change().rename('asset')
                        btc_returns = df1m_btc['c'].pct_change().rename('btc')
                        combined_df = pd.concat([asset_returns, btc_returns], axis=1).dropna()
                        if len(combined_df) > 2:
                            covariance = combined_df['asset'].cov(combined_df['btc'])
                            variance = combined_df['btc'].var()
                            btc_beta_during_trade = covariance / variance if variance > 0 else 0.0

                update_query = """
                    UPDATE positions SET
                        exit_reason = $1, adx_at_entry = $2, vwap_dev_pct_at_entry = $3,
                        ema_fast_at_entry = $4, ema_slow_at_entry = $5, mae_over_atr = $6,
                        mfe_over_atr = $7, session_tag_at_entry = $8, day_of_week_at_entry = $9,
                        hour_of_day_at_entry = $10, btc_beta_during_trade = $11
                    WHERE id = $12
                """
                await conn.execute(
                    update_query, inferred_exit_reason, adx_at_entry, vwap_dev_pct_at_entry,
                    ema_fast_at_entry, ema_slow_at_entry, mae_over_atr, mfe_over_atr,
                    session_tag, day_of_week, hour_of_day, btc_beta_during_trade, trade_id
                )

            except Exception as e:
                LOG.error(f"Failed to backfill trade ID {trade['id']} ({trade['symbol']}): {e}")
                continue

        LOG.info("Comprehensive backfilling process complete.")

    except Exception as e:
        LOG.error(f"A critical error occurred during the backfill process: {e}")
    finally:
        if conn: await conn.close()
        if exchange: await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())