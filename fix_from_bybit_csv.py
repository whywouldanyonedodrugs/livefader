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
        
        # --- FIX: Specify the correct date format ---
        date_format = '%d/%m/%Y %H:%M'
        # We apply the format to the correct columns by their original names
        bybit_df['Filled/Settlement Time(UTC+0)'] = pd.to_datetime(bybit_df['Filled/Settlement Time(UTC+0)'], dayfirst=True, utc=True)
        bybit_df['Create Time'] = pd.to_datetime(bybit_df['Create Time'], dayfirst=True, utc=True)
        
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

        # --- FIX: Use itertuples(index=False) and numerical indices for robust access ---
        # This avoids all issues with column names containing spaces or special characters.
        # Column indices from your CSV:
        # 0: Contracts, 4: Realized P&L, 7: Filled/Settlement Time, 8: Create Time
        for true_trade in tqdm(bybit_df.itertuples(index=False, name=None), total=len(bybit_df), desc="Fixing Trades from CSV"):
            symbol = true_trade[0]
            true_open_time = true_trade[8]
            
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
                true_pnl = float(true_trade[4]) # Access Realized P&L by index 4

                if abs(db_pnl - true_pnl) > 0.01:
                    LOG.info(f"Fixing trade {pos_id} ({symbol}): DB PnL {db_pnl:.4f} -> Exchange PnL {true_pnl:.4f}")
                    
                    true_closed_at = true_trade[7] # Access Filled/Settlement Time by index 7
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