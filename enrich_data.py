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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOG = logging.getLogger(__name__)

# --- Counterfactual Parameters ---
LOOK_FORWARD_HOURS = 4
COUNTERFACTUAL_TP_MULT = 2.0  # e.g., 2.0x ATR
COUNTERFACTUAL_SL_MULT = 2.5  # e.g., 2.5x ATR

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
        
        # Find all closed trades that have NOT been enriched yet
        query = "SELECT * FROM positions WHERE status = 'CLOSED' AND cf_mae_over_atr_4h IS NULL ORDER BY id"
        trades_to_enrich = await conn.fetch(query)

        if not trades_to_enrich:
            LOG.info("No new trades found to enrich. Database is up-to-date.")
            return

        LOG.info(f"Found {len(trades_to_enrich)} trades to enrich with counterfactual data. Starting process...")

        for trade in tqdm(trades_to_enrich, desc="Enriching Trades"):
            try:
                trade_id = trade['id']
                symbol = trade['symbol']
                closed_at = trade['closed_at'].replace(tzinfo=timezone.utc)
                entry_price = float(trade['entry_price'])
                atr_at_entry = float(trade['atr'])

                # Define the counterfactual SL and TP levels
                cf_tp_price = entry_price - (atr_at_entry * COUNTERFACTUAL_TP_MULT)
                cf_sl_price = entry_price + (atr_at_entry * COUNTERFACTUAL_SL_MULT)

                # --- Fetch the "Future" Data ---
                # We need 1-minute data for the 4 hours AFTER the trade closed
                since_ts = int(closed_at.timestamp() * 1000)
                limit = LOOK_FORWARD_HOURS * 60 # 4 hours * 60 minutes
                
                ohlcv_future = await exchange.fetch_ohlcv(symbol, '1m', since=since_ts, limit=limit)

                if not ohlcv_future:
                    LOG.warning(f"Could not fetch future OHLCV for trade {trade_id}. Skipping.")
                    continue

                df_future = pd.DataFrame(ohlcv_future, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # --- Perform Counterfactual Calculations ---
                # Check if the hypothetical TP or SL would have been hit
                would_hit_tp = (df_future['low'] <= cf_tp_price).any()
                would_hit_sl = (df_future['high'] >= cf_sl_price).any()

                # Calculate the "ultimate" MAE and MFE over the next 4 hours
                # For a short position:
                max_adverse_price = df_future['high'].max()
                max_favorable_price = df_future['low'].min()
                
                cf_mae_usd = (max_adverse_price - entry_price) * float(trade['size'])
                cf_mfe_usd = (entry_price - max_favorable_price) * float(trade['size'])
                
                cf_mae_over_atr = (cf_mae_usd / float(trade['size'])) / atr_at_entry if atr_at_entry > 0 else 0
                cf_mfe_over_atr = (cf_mfe_usd / float(trade['size'])) / atr_at_entry if atr_at_entry > 0 else 0

                # --- Update the Database Record ---
                update_query = """
                    UPDATE positions SET
                        cf_would_hit_tp_2x_atr = $1,
                        cf_would_hit_sl_2_5x_atr = $2,
                        cf_mae_over_atr_4h = $3,
                        cf_mfe_over_atr_4h = $4
                    WHERE id = $5
                """
                await conn.execute(
                    update_query,
                    bool(would_hit_tp),
                    bool(would_hit_sl),
                    cf_mae_over_atr,
                    cf_mfe_over_atr,
                    trade_id
                )

            except Exception as e:
                LOG.error(f"Failed to enrich trade ID {trade['id']} ({trade['symbol']}): {e}", exc_info=True)
                continue
            
            # Be polite to the API
            await asyncio.sleep(0.5)

        LOG.info("Counterfactual data enrichment complete.")

    except Exception as e:
        LOG.error(f"A critical error occurred during the enrichment process: {e}", exc_info=True)
    finally:
        if conn: await conn.close()
        if exchange: await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())