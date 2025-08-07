# /opt/livefader/src/fix_from_bybit_csv.py

import asyncio
import asyncpg
import logging
import os
import argparse
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOG = logging.getLogger(__name__)

async def main(csv_path: str):
    load_dotenv()
    db_dsn = os.getenv("DATABASE_URL")
    if not db_dsn:
        LOG.error("DATABASE_URL not found. Cannot proceed.")
        return

    try:
        # Load the trade history CSV from Bybit
        bybit_df = pd.read_csv(csv_path)
        
        # --- Data Cleaning and Preparation ---
        bybit_df.columns = [c.strip().replace(' ', '_').replace('(UTC+0)', '') for c in bybit_df.columns]
        
        # --- FIX: Specify the correct date format ---
        # This tells pandas to expect Day/Month/Year format
        date_format = '%d/%m/%Y %H:%M'
        bybit_df['Filled/Settlement_Time'] = pd.to_datetime(bybit_df['Filled/Settlement_Time'], format=date_format, utc=True)
        bybit_df['Create_Time'] = pd.to_datetime(bybit_df['Create_Time'], format=date_format, utc=True)
        # --- END OF FIX ---
        
        LOG.info(f"Loaded and processed {len(bybit_df)} trade records from {csv_path}")

    except FileNotFoundError:
        LOG.error(f"CSV file not found at: {csv_path}")
        return
    except Exception as e:
        LOG.error(f"Failed to read or parse CSV file: {e}")
        return

    conn = None
    try:
        conn = await asyncpg.connect(dsn=db_dsn)
        
        update_count = 0
        not_found_count = 0

        # Iterate through each row in the Bybit CSV
        for true_trade in tqdm(bybit_df.itertuples(), total=len(bybit_df), desc="Fixing Trades from CSV"):
            symbol = true_trade.Contracts
            true_open_time = true_trade.Create_Time
            
            query = """
                SELECT id, pnl FROM positions
                WHERE symbol = $1 AND status = 'CLOSED'
                AND opened_at BETWEEN $2 AND $3
            """
            match_window_start = true_open_time - timedelta(minutes=5)
            match_window_end = true_open_time + timedelta(minutes=5)
            
            db_match = await conn.fetchrow(query, symbol, match_window_start, match_window_end)

            if db_match:
                pos_id = db_match['id']
                db_pnl = float(db_match['pnl'])
                true_pnl = float(true_trade.Realized_P_L)

                if abs(db_pnl - true_pnl) > 0.01:
                    LOG.info(f"Fixing trade {pos_id} ({symbol}): DB PnL {db_pnl:.4f} -> Exchange PnL {true_pnl:.4f}")
                    
                    true_closed_at = true_trade._9 # Corresponds to 'Filled/Settlement_Time'
                    holding_minutes = (true_closed_at - true_open_time).total_seconds() / 60
                    
                    update_query = """
                        UPDATE positions SET
                            pnl = $1,
                            closed_at = $2,
                            holding_minutes = $3,
                            exit_reason = $4
                        WHERE id = $5
                    """
                    exit_reason = "TP" if true_pnl > 0 else "SL"
                    
                    await conn.execute(
                        update_query,
                        true_pnl,
                        true_closed_at,
                        holding_minutes,
                        exit_reason,
                        pos_id
                    )
                    update_count += 1
            else:
                LOG.warning(f"Could not find a matching trade in the database for {symbol} opened around {true_open_time}.")
                not_found_count += 1
        
        LOG.info(f"Correction process complete. Updated {update_count} trades. Could not find a match for {not_found_count} trades.")

    except Exception as e:
        LOG.error(f"An error occurred during the database update process: {e}", exc_info=True)
    finally:
        if conn: await conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix trade history from a Bybit trade history CSV.")
    parser.add_argument("--file", required=True, help="Path to the trade history CSV downloaded from Bybit.")
    args = parser.parse_args()
    asyncio.run(main(args.file))