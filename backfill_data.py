# /opt/livefader/src/backfill_data_full.py

import asyncio
import asyncpg
import logging
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt

# --- Bot-Consistent Imports ---
import config as cfg
from live import indicators as ta

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOG = logging.getLogger(__name__)

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
        exchange = ccxt.bybit({
            'apiKey': os.getenv("BYBIT_API_KEY"),
            'secret': os.getenv("BYBIT_API_SECRET"),
            'enableRateLimit': True
        })
        await exchange.load_markets()
        
        query = "SELECT * FROM positions WHERE status = 'CLOSED' ORDER BY id"
        trades_to_fix = await conn.fetch(query)

        if not trades_to_fix:
            LOG.info("No trades found to backfill.")
            return

        LOG.info(f"Found {len(trades_to_fix)} trades to process...")
        
        trade_count = 0
        for trade in trades_to_fix:
            trade_count += 1
            try:
                trade_id = trade['id']
                symbol = trade['symbol']
                LOG.info(f"Processing trade {trade_count}/{len(trades_to_fix)}: ID {trade_id} ({symbol})")
                
                opened_at = trade['opened_at'].replace(tzinfo=timezone.utc)
                closed_at = trade['closed_at'].replace(tzinfo=timezone.utc)
                entry_price = float(trade['entry_price'])
                size = float(trade['size'])
                atr_at_entry_db = float(trade['atr']) if trade['atr'] is not None else 0.01

                # --- PnL and Exit Reason Logic ---
                my_trades = await exchange.fetch_my_trades(symbol, since=int(opened_at.timestamp() * 1000), limit=100)
                start_ts_ms = int(opened_at.timestamp() * 1000)
                end_ts_ms = int((closed_at + timedelta(minutes=5)).timestamp() * 1000)
                position_trades = [t for t in my_trades if start_ts_ms <= t['timestamp'] <= end_ts_ms]
                closing_fill = next((t for t in reversed(position_trades) if t['side'] == 'buy'), None)

                if not closing_fill:
                    exit_price = entry_price - (float(trade['pnl']) / size) if trade['pnl'] is not None and size > 0 else 0
                else:
                    exit_price = float(closing_fill['price'])

                pnl = (entry_price - exit_price) * size
                pnl_pct = ((entry_price / exit_price) - 1) * 100 if exit_price > 0 else 0
                holding_minutes = (closed_at - opened_at).total_seconds() / 60
                
                if abs(holding_minutes - (cfg.TIME_EXIT_HOURS * 60)) < 5:
                     inferred_exit_reason = "TIME_EXIT"
                elif pnl > 0:
                     inferred_exit_reason = "TP"
                else:
                     inferred_exit_reason = "SL"

                # --- Fetch Historical Data ---
                since_ts_1d = int((opened_at - timedelta(days=cfg.STRUCTURAL_TREND_DAYS + 5)).timestamp() * 1000)
                since_ts_4h = int((opened_at - timedelta(days=40)).timestamp() * 1000)
                since_ts_1h = int((opened_at - timedelta(days=10)).timestamp() * 1000)
                since_ts_5m = int((opened_at - timedelta(hours=cfg.GAP_VWAP_HOURS + 1)).timestamp() * 1000)

                ohlcv_1d, ohlcv_4h, ohlcv_1h, ohlcv_5m = await asyncio.gather(
                    exchange.fetch_ohlcv(symbol, '1d', since=since_ts_1d, limit=100),
                    exchange.fetch_ohlcv(symbol, cfg.EMA_TIMEFRAME, since=since_ts_4h, limit=500),
                    exchange.fetch_ohlcv(symbol, cfg.RSI_TIMEFRAME, since=since_ts_1h, limit=500),
                    exchange.fetch_ohlcv(symbol, '5m', since=since_ts_5m, limit=50)
                )

                df1d = pd.DataFrame(ohlcv_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).assign(timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='ms', utc=True)).set_index('timestamp').loc[:opened_at]
                df4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).assign(timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='ms', utc=True)).set_index('timestamp').loc[:opened_at]
                df1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).assign(timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='ms', utc=True)).set_index('timestamp').loc[:opened_at]
                df5m = pd.DataFrame(ohlcv_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).assign(timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='ms', utc=True)).set_index('timestamp').loc[:opened_at]

                # --- Calculate All Pre-Trade Metrics ---
                rsi_at_entry = ta.rsi(df1h['close'], period=cfg.RSI_PERIOD).iloc[-1]
                adx_at_entry = ta.adx(df1h, period=cfg.ADX_PERIOD).iloc[-1]
                ema_fast_at_entry = ta.ema(df4h['close'], span=cfg.EMA_FAST_PERIOD).iloc[-1]
                ema_slow_at_entry = ta.ema(df4h['close'], span=cfg.EMA_SLOW_PERIOD).iloc[-1]
                ret_30d_at_entry = (df1d['close'].iloc[-1] / df1d['close'].iloc[-cfg.STRUCTURAL_TREND_DAYS]) - 1 if len(df1d) >= cfg.STRUCTURAL_TREND_DAYS else 0.0
                
                _is_ema_crossed_down = ema_fast_at_entry < ema_slow_at_entry
                ema_spread_pct_at_entry = (ema_fast_at_entry - ema_slow_at_entry) / ema_slow_at_entry if ema_slow_at_entry > 0 else 0.0

                tf_minutes = 5
                vwap_bars = int((cfg.GAP_VWAP_HOURS * 60) / tf_minutes)
                vwap_num = (df5m['close'] * df5m['volume']).shift(1).rolling(vwap_bars).sum()
                vwap_den = df5m['volume'].shift(1).rolling(vwap_bars).sum()
                vwap = vwap_num / vwap_den
                vwap_dev_raw = df5m['close'] - vwap
                price_std = df5m['close'].rolling(vwap_bars).std()
                vwap_z_score = vwap_dev_raw / price_std
                vwap_at_entry = vwap.iloc[-1]
                vwap_dev_pct_at_entry = abs(entry_price - vwap_at_entry) / vwap_at_entry if vwap_at_entry > 0 else 0.0
                vwap_z_at_entry = vwap_z_score.iloc[-1]

                df5m['vwap_dev_pct'] = (df5m['close'] - vwap) / vwap
                df5m['vwap_ok'] = df5m['vwap_dev_pct'].abs() <= cfg.GAP_MAX_DEV_PCT
                df5m['vwap_consolidated'] = df5m['vwap_ok'].rolling(cfg.GAP_MIN_BARS).min().fillna(0).astype(bool)
                _vwap_consolidated = df5m['vwap_consolidated'].iloc[-1]

                # --- Definitive Boolean Conversion Fix ---
                is_ema_crossed_down_at_entry = True if _is_ema_crossed_down else False
                vwap_consolidated_at_entry = True if _vwap_consolidated else False

                # --- MAE/MFE Calculation ---
                mae_usd, mfe_usd = 0.0, 0.0
                ohlcv_1m_asset = await exchange.fetch_ohlcv(symbol, '1m', since=int(opened_at.timestamp() * 1000), limit=1000)
                if ohlcv_1m_asset:
                    df1m_asset = pd.DataFrame(ohlcv_1m_asset, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).assign(timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='ms', utc=True)).set_index('timestamp').loc[opened_at:closed_at]
                    if not df1m_asset.empty:
                        if inferred_exit_reason == "SL": mae_usd = (exit_price - entry_price) * size
                        else: mae_usd = (df1m_asset['high'].max() - entry_price) * size
                        if inferred_exit_reason == "TP": mfe_usd = (entry_price - exit_price) * size
                        else: mfe_usd = (entry_price - df1m_asset['low'].min()) * size
                mae_usd = max(0, mae_usd)
                mfe_usd = max(0, mfe_usd)
                mae_over_atr = (mae_usd / size) / atr_at_entry_db if atr_at_entry_db > 0 and size > 0 else 0
                mfe_over_atr = (mfe_usd / size) / atr_at_entry_db if atr_at_entry_db > 0 and size > 0 else 0

                # --- ETH MACD Barometer Calculation (4H and 1H) ---
                eth_macd_4h, eth_macdsignal_4h, eth_macdhist_4h = None, None, None
                eth_macd_1h, eth_macdsignal_1h, eth_macdhist_1h = None, None, None
                try:
                    # Fetch 4H data
                    since_ts_eth_4h = int((opened_at - timedelta(days=50)).timestamp() * 1000)
                    eth_ohlcv_4h = await exchange.fetch_ohlcv('ETHUSDT', '4h', since=since_ts_eth_4h, limit=300)
                    if eth_ohlcv_4h:
                        df_eth_4h = pd.DataFrame(eth_ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df_eth_4h['timestamp'] = pd.to_datetime(df_eth_4h['timestamp'], unit='ms', utc=True)
                        df_eth_4h = df_eth_4h.set_index('timestamp').loc[:opened_at]
                        if not df_eth_4h.empty:
                            macd_df_4h = ta.macd(df_eth_4h['close'])
                            latest_macd_4h = macd_df_4h.iloc[-1]
                            eth_macd_4h = latest_macd_4h['macd']
                            eth_macdsignal_4h = latest_macd_4h['signal']
                            eth_macdhist_4h = latest_macd_4h['hist']
                    
                    # Fetch 1H data
                    since_ts_eth_1h = int((opened_at - timedelta(days=15)).timestamp() * 1000)
                    eth_ohlcv_1h = await exchange.fetch_ohlcv('ETHUSDT', '1h', since=since_ts_eth_1h, limit=360)
                    if eth_ohlcv_1h:
                        df_eth_1h = pd.DataFrame(eth_ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df_eth_1h['timestamp'] = pd.to_datetime(df_eth_1h['timestamp'], unit='ms', utc=True)
                        df_eth_1h = df_eth_1h.set_index('timestamp').loc[:opened_at]
                        if not df_eth_1h.empty:
                            macd_df_1h = ta.macd(df_eth_1h['close'])
                            latest_macd_1h = macd_df_1h.iloc[-1]
                            eth_macd_1h = latest_macd_1h['macd']
                            eth_macdsignal_1h = latest_macd_1h['signal']
                            eth_macdhist_1h = latest_macd_1h['hist']
                except Exception as e:
                    LOG.warning(f"Could not backfill ETH MACD for trade {trade_id}: {e}")

                # --- Final Database Update ---
                update_query = """
                    UPDATE positions SET
                        pnl = $1, pnl_pct = $2, exit_reason = $3, holding_minutes = $4,
                        rsi_at_entry = $5, adx_at_entry = $6, ema_fast_at_entry = $7,
                        ema_slow_at_entry = $8, ret_30d_at_entry = $9, mae_usd = $10,
                        mfe_usd = $11, mae_over_atr = $12, mfe_over_atr = $13,
                        vwap_dev_pct_at_entry = $14, vwap_z_at_entry = $15,
                        is_ema_crossed_down_at_entry = $16,
                        ema_spread_pct_at_entry = $17,
                        vwap_consolidated_at_entry = $18,
                        eth_macd_at_entry = $19,
                        eth_macdsignal_at_entry = $20,
                        eth_macdhist_at_entry = $21,
                        eth_macd_1h_at_entry = $22,
                        eth_macdsignal_1h_at_entry = $23,
                        eth_macdhist_1h_at_entry = $24
                    WHERE id = $25
                """
                await conn.execute(
                    update_query,
                    pnl, pnl_pct, inferred_exit_reason, holding_minutes,
                    rsi_at_entry, adx_at_entry, ema_fast_at_entry, ema_slow_at_entry,
                    ret_30d_at_entry, mae_usd, mfe_usd, mae_over_atr, mfe_over_atr,
                    vwap_dev_pct_at_entry, vwap_z_at_entry,
                    is_ema_crossed_down_at_entry,
                    ema_spread_pct_at_entry,
                    vwap_consolidated_at_entry,
                    eth_macd_4h,
                    eth_macdsignal_4h,
                    eth_macdhist_4h,
                    eth_macd_1h,
                    eth_macdsignal_1h,
                    eth_macdhist_1h,
                    trade_id
                )

            except Exception as e:
                LOG.error(f"Failed to backfill trade ID {trade['id']} ({trade['symbol']}): {e}", exc_info=True)
                continue
            
            await asyncio.sleep(0.5)

        LOG.info("Comprehensive backfilling process complete.")

    except Exception as e:
        LOG.error(f"A critical error occurred during the backfill process: {e}", exc_info=True)
    finally:
        if conn: await conn.close()
        if exchange: await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())